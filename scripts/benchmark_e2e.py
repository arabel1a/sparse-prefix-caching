"""End-to-end prefix caching evaluation on real conversation traces.

Runs checkpoint placement strategies with FIFO prefix cache.
All checkpoint strategies store both GDN recurrent states and attention KV.
"""
import sparse_prefix_caching
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from sparse_prefix_caching.checkpoint_cache import (
    prefill_from_checkpoint,
)
from sparse_prefix_caching.datasets.base import Dataset
from sparse_prefix_caching.patches import capture_gdn_states
from sparse_prefix_caching.utils import (
    setup_output_dir,
    resolve_strategies,
    make_model,
    make_model_config,
    gdn_checkpoint_bytes,
    kv_per_token_bytes,
    prefill_baseline,
    _model_device,
    _sync_device,
    build_store_from_captures,
    PrefixCache,
    FixedSizeCache,
    DryRunStore,
    _save_jsonl,
)
from sparse_prefix_caching.strategies import checkpoint_positions, HistogramTracker

log = logging.getLogger(__name__)


def _make_dataset(cfg) -> Dataset:
    cls = hydra.utils.get_class(cfg.data._target_)
    return cls(cfg=cfg.data)


def _make_cache(cfg):
    """Instantiate cache manager from config."""
    cls = hydra.utils.get_class(cfg.cache_manager._target_)
    return cls(**{k: v for k, v in cfg.cache_manager.items() if k != "_target_"})


def simulate_dry(dataset, requests, vocab_size, strategy, cache,
                 kv_bytes_per_tok, gdn_bytes_per_ckpt,
                 progress=False, histogram_tracker=None, warmup=False):
    """Dry-run: simulate cache behavior without model, approximate cost as tokens to process."""
    uses_cache = strategy.type != "no_cache"
    is_hist = _is_histogram_strategy(strategy.type)
    online_hist = is_hist and strategy.type != 'histogram_frozen' and not warmup
    per_request = []

    for req in tqdm(requests, desc=f"{strategy.tag}{' warmup' if warmup else ''}",
                    disable=not progress):
        conv_id = dataset.conv_id(req)
        all_toks = dataset.get_tokens(req)
        input_ids = torch.tensor(all_toks, dtype=torch.long) % vocab_size
        seq_len = len(input_ids)

        hit = False
        tokens_saved = 0
        match_len = 0
        reusable_kv = 0
        reusable_gdn = 0

        if uses_cache:
            cached_store, match_len = cache.find_best_prefix(conv_id, input_ids)

            if is_hist and histogram_tracker is not None:
                histogram_tracker.observe(match_len)

            if cached_store is not None:
                hit = True
                kv_len = min(cached_store.kv_len, seq_len, match_len)
                reusable_kv = kv_len
                best_pos = cached_store.best_checkpoint(kv_len)
                if best_pos is not None:
                    tokens_saved = best_pos
                    reusable_gdn = best_pos
        elif is_hist and histogram_tracker is not None:
            histogram_tracker.observe(0)

        positions = []
        if uses_cache:
            positions = checkpoint_positions(seq_len, histogram_tracker=histogram_tracker, **strategy)
            store = DryRunStore(input_ids, positions, kv_bytes_per_tok, gdn_bytes_per_ckpt)
            cache.put(conv_id, store)

        dt = (seq_len - tokens_saved) * 0.05  # 1 token ≈ 50ms

        per_request.append({
            "conv_id": str(conv_id), "seq_len": seq_len,
            "added_positions": positions,
            "time_s": dt, "capture_s": 0.0, "hit": hit,
            "tokens_saved": tokens_saved,
            "reusable_kv": reusable_kv, "reusable_gdn": reusable_gdn,
            "prefix_match": match_len,
            "n_cache_entries": cache.n_entries,
            "cache_kv_bytes": cache.kv_used,
            "cache_gdn_bytes": cache.gdn_used,
        })

    if warmup and is_hist and histogram_tracker is not None:
        if strategy.type == 'histogram_frozen':
            histogram_tracker.freeze()
        else:
            histogram_tracker.solve()

    total_time = sum(e["time_s"] for e in per_request)
    return {
        "total_time": total_time,
        "total_capture_time": 0.0,
        "wall_time": total_time,
        "hits": sum(1 for e in per_request if e["hit"]),
        "tokens_saved": sum(e["tokens_saved"] for e in per_request),
        "tokens_total": sum(e["seq_len"] for e in per_request),
        "cache_stats": cache.stats(),
        "per_request": per_request,
    }


def _is_histogram_strategy(stype):
    return stype in ("histogram_frozen", "histogram_periodic", "histogram_exp_decay")


def _make_histogram_tracker(strategy, max_len):
    """Create a HistogramTracker from strategy config."""
    stype = strategy.type
    mode = {'histogram_frozen': 'frozen',
            'histogram_periodic': 'periodic',
            'histogram_exp_decay': 'exp_decay'}[stype]
    budget = strategy.n_blocks
    gamma = strategy['gamma']
    laplace_alpha = strategy.laplace_alpha
    replan_interval = strategy['replan_interval']
    bin_size = strategy['bin_size']
    log.info("histogram %s [%s]: budget=%d, bin_size=%d, max_len=%d",
             strategy.tag, stype, budget, bin_size, max_len)
    adaptive_backtrack = strategy['adaptive_backtrack']
    return HistogramTracker(max_len, budget, mode=mode, gamma=gamma, alpha=laplace_alpha,
                            replan_interval=replan_interval, bin_size=bin_size,
                            adaptive_backtrack=adaptive_backtrack)


def warmup_cache(model, dataset, requests, vocab_size, strategy, cache,
                 progress=False, histogram_tracker=None):
    """Run requests through the model to fill the cache, without timing."""
    dev = _model_device(model)
    config = model.config
    uses_cache = strategy.type != "no_cache"
    is_hist = _is_histogram_strategy(strategy.type)

    for req in tqdm(requests, desc=f"{strategy.tag} warmup", disable=not progress):
        conv_id = dataset.conv_id(req)
        all_toks = dataset.get_tokens(req)
        input_ids = torch.tensor([all_toks], dtype=torch.long) % vocab_size
        seq_len = input_ids.shape[1]

        cached_store, match_len = (cache.find_best_prefix(conv_id, input_ids[0])
                                   if uses_cache else (None, 0))
        input_ids = input_ids.to(dev)
        if is_hist and histogram_tracker is not None:
            histogram_tracker.observe(match_len)

        positions = (checkpoint_positions(seq_len, histogram_tracker=histogram_tracker, **strategy)
                     if uses_cache else [])

        with capture_gdn_states(positions) as captured:
            if cached_store is not None:
                _, model_cache = prefill_from_checkpoint(model, input_ids, cached_store, match_len=match_len)
            else:
                _, model_cache = prefill_baseline(model, input_ids)
        _sync_device(dev)
        if uses_cache:
            store = build_store_from_captures(
                captured, input_ids, positions, model_cache, config,
                existing_store=cached_store,
            )
            store.to("cpu")
            cache.put(conv_id, store)

    if is_hist and histogram_tracker is not None:
        if strategy.type == 'histogram_frozen':
            histogram_tracker.freeze()
        else:
            # First DP solve on accumulated warmup data (like frozen),
            # but keep the mode for online re-solving during simulate.
            histogram_tracker.solve()

    return cache


def simulate(model, dataset, requests, vocab_size, strategy,
             cache, progress=False,
             histogram_tracker=None):
    """Run all requests through the model with given caching strategy."""
    dev = _model_device(model)
    config = model.config
    uses_cache = strategy.type != "no_cache"
    is_hist = _is_histogram_strategy(strategy.type)
    online_hist = is_hist and strategy.type != 'histogram_frozen'
    per_request = []
    wall_t0 = time.perf_counter()

    for req in tqdm(requests, desc=strategy.tag, disable=not progress):
        conv_id = dataset.conv_id(req)
        all_toks = dataset.get_tokens(req)
        input_ids = torch.tensor([all_toks], dtype=torch.long) % vocab_size
        seq_len = input_ids.shape[1]

        hit = False
        tokens_saved = 0
        cached_store = None
        match_len = 0
        reusable_kv = 0
        reusable_gdn = 0

        if uses_cache:
            cached_store, match_len = cache.find_best_prefix(conv_id, input_ids[0])
        input_ids = input_ids.to(dev)

        if online_hist and histogram_tracker is not None:
            histogram_tracker.observe(match_len)

        positions = (checkpoint_positions(seq_len, histogram_tracker=histogram_tracker, **strategy)
                     if uses_cache else [])

        _sync_device(dev)
        t0 = time.perf_counter()
        with capture_gdn_states(positions) as captured:
            if cached_store is not None:
                _, model_cache = prefill_from_checkpoint(model, input_ids, cached_store, match_len=match_len)
            else:
                _, model_cache = prefill_baseline(model, input_ids)
        _sync_device(dev)
        dt = time.perf_counter() - t0

        if cached_store is not None:
            hit = True
            kv_len = min(cached_store.kv_len, seq_len, match_len)
            reusable_kv = kv_len
            ckpt = cached_store.best_checkpoint(kv_len)
            if ckpt:
                tokens_saved = ckpt.position
                reusable_gdn = ckpt.position

        capture_s = 0.0
        if uses_cache:
            store = build_store_from_captures(
                captured, input_ids, positions, model_cache, config,
                existing_store=cached_store,
            )
            store.to("cpu")
            cache.put(conv_id, store)
        _sync_device(dev)

        per_request.append({
            "conv_id": str(conv_id), "seq_len": seq_len,
            "added_positions": positions,
            "time_s": dt, "capture_s": capture_s, "hit": hit,
            "tokens_saved": tokens_saved,
            "reusable_kv": reusable_kv, "reusable_gdn": reusable_gdn,
            "prefix_match": match_len,
            "n_cache_entries": cache.n_entries,
            "cache_kv_bytes": cache.kv_used,
            "cache_gdn_bytes": cache.gdn_used,
        })

    wall_time = time.perf_counter() - wall_t0
    return {
        "total_time": sum(e["time_s"] for e in per_request),
        "total_capture_time": sum(e["capture_s"] for e in per_request),
        "wall_time": wall_time,
        "hits": sum(1 for e in per_request if e["hit"]),
        "tokens_saved": sum(e["tokens_saved"] for e in per_request),
        "tokens_total": sum(e["seq_len"] for e in per_request),
        "cache_stats": cache.stats(),
        "per_request": per_request,
    }


def run_strategy(model, dataset, strat, train_requests, test_requests,
                 vocab_size, cfg, max_seq_len, progress):
    """Run a single strategy end-to-end. All caches are local and freed on return."""
    hist_tracker = None
    if _is_histogram_strategy(strat.type):
        hist_tracker = _make_histogram_tracker(strat, max_seq_len)

    cache = _make_cache(cfg)
    log.info("Strategy: %s — warming cache on train split...", strat.tag)
    cache = warmup_cache(model, dataset, train_requests, vocab_size,
                         strat, cache, progress=progress,
                         histogram_tracker=hist_tracker)
    log.info("Strategy: %s — evaluating on test split...", strat.tag)
    res = simulate(model, dataset, test_requests, vocab_size,
                   strat, cache, progress=progress,
                   histogram_tracker=hist_tracker)

    log.info("  %s: prefill=%.1fs capture=%.1fs wall=%.1fs",
             strat.tag, res["total_time"], res["total_capture_time"], res["wall_time"])
    if res["hits"] > 0:
        log.info("    hits: %d/%d (%.1f%%)",
                 res["hits"], len(test_requests), res["hits"] / len(test_requests) * 100)
    if hist_tracker is not None:
        res["histogram_log"] = [
            {"n_obs": entry["n_obs"], "counts": entry["counts"].tolist()}
            for entry in hist_tracker.histogram_log
        ]
        res["laplace_alpha"] = float(strat.laplace_alpha)
        res["bin_size"] = hist_tracker.bin_size
    return res


def run_strategy_dry(dataset, strat, train_requests, test_requests,
                     vocab_size, cfg, max_seq_len, progress):
    """Dry-run a single strategy: no model, approximate cost as token counts."""
    hist_tracker = None
    if _is_histogram_strategy(strat.type):
        hist_tracker = _make_histogram_tracker(strat, max_seq_len)

    config = make_model_config(cfg)
    kv_bpt = kv_per_token_bytes(config)
    gdn_bpc = gdn_checkpoint_bytes(config)

    cache = _make_cache(cfg)
    log.info("Strategy: %s — dry warmup...", strat.tag)
    simulate_dry(dataset, train_requests, vocab_size, strat, cache,
                 kv_bpt, gdn_bpc, progress=progress,
                 histogram_tracker=hist_tracker, warmup=True)
    log.info("Strategy: %s — dry eval...", strat.tag)
    res = simulate_dry(dataset, test_requests, vocab_size, strat, cache,
                       kv_bpt, gdn_bpc, progress=progress,
                       histogram_tracker=hist_tracker)

    tok_total = res["tokens_total"]
    tok_saved = res["tokens_saved"]
    log.info("  %s: tokens_saved=%d/%d (%.1f%%)",
             strat.tag, tok_saved, tok_total,
             tok_saved / tok_total * 100 if tok_total > 0 else 0)
    if hist_tracker is not None:
        res["histogram_log"] = [
            {"n_obs": entry["n_obs"], "counts": entry["counts"].tolist()}
            for entry in hist_tracker.histogram_log
        ]
        res["laplace_alpha"] = float(strat.laplace_alpha)
        res["bin_size"] = hist_tracker.bin_size
    return res


@hydra.main(config_path=r"../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = setup_output_dir(cfg, "benchmark_e2e")
    resolve_strategies(cfg)
    e2e = cfg.benchmark_e2e
    dry_run = e2e.dry_run
    strategies = list(cfg.strategies)

    if dry_run:
        vocab_size = cfg.model.vocab_size
    else:
        model = make_model(cfg)
        vocab_size = model.config.vocab_size

    # Load dataset (interleaving happens inside)
    dataset = _make_dataset(cfg)
    dataset.load(seed=cfg.seed)
    train_requests, test_requests = dataset.train_test_split()

    total_tokens = sum(len(dataset.get_tokens(r)) for r in test_requests)
    log.info("Train: %d, Test: %d requests, test tokens: %d, cache: %s%s",
             len(train_requests), len(test_requests), total_tokens,
             cfg.cache_manager._target_, " [DRY RUN]" if dry_run else "")

    summary = {
        "model_name": cfg.model.name,
        "dry_run": dry_run,
        "n_train_requests": len(train_requests),
        "n_test_requests": len(test_requests),
        "total_tokens": total_tokens,
        "train_frac": cfg.data.train_frac,
        "cache_manager_config": OmegaConf.to_container(cfg.cache_manager, resolve=True),
        "strategies": {},
        "strategy_styles": OmegaConf.to_container(cfg.strategies, resolve=True),
    }
    summary_path = out_dir / "e2e_summary.json"

    if not dry_run:
        # initial warmup to alleviate gpu clock control, etc
        print("Warming up cores")
        cache = _make_cache(cfg)
        cache = warmup_cache(model, dataset, train_requests[:e2e.warmup_seqs], vocab_size,
                       strategies[0], cache, progress=e2e.progress)
        simulate(model, dataset, test_requests[:e2e.warmup_seqs], vocab_size,
                        strategies[0], cache, progress=e2e.progress)

    for strat in strategies:
        if dry_run:
            res = run_strategy_dry(dataset, strat, train_requests, test_requests,
                                   vocab_size, cfg, cfg.data.max_seq_len, e2e.progress)
        else:
            res = run_strategy(model, dataset, strat, train_requests, test_requests,
                               vocab_size, cfg, cfg.data.max_seq_len, e2e.progress)

        _save_jsonl(out_dir / f"e2e_{strat.tag}.jsonl", res["per_request"])
        if "histogram_log" in res:
            hist_path = out_dir / f"e2e_{strat.tag}_histograms.json"
            hist_path.write_text(json.dumps({
                "histogram_log": res["histogram_log"],
                "laplace_alpha": res["laplace_alpha"],
                "bin_size": res["bin_size"],
            }, indent=2))
            log.info("Histogram log saved to %s", hist_path)
        summary["strategies"][strat.tag] = {
            "total_time": res["total_time"],
            "total_capture_time": res["total_capture_time"],
            "wall_time": res["wall_time"],
            "hits": res["hits"],
            "tokens_saved": res["tokens_saved"],
            "tokens_total": res["tokens_total"],
            "cache_stats": res["cache_stats"],
        }
        summary_path.write_text(json.dumps(summary, indent=2))

    log.info("Done. Results saved to %s", out_dir)

if __name__ == "__main__":
    main()
