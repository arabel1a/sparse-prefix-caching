"""End-to-end prefix caching evaluation on real conversation traces.

Runs checkpoint placement strategies with FIFO prefix cache.
All checkpoint strategies store both GDN recurrent states and attention KV.
"""
import spase_cache
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from spase_cache.checkpoint_cache import (
    prefill_from_checkpoint,
)
from spase_cache.datasets.base import Dataset
from spase_cache.utils import (
    setup_output_dir,
    resolve_strategies,
    make_model,
    prefill_baseline,
    _model_device,
    _sync_device,
    prefill_and_capture_at,
    PrefixCache,
    _save_jsonl,
)
from spase_cache.strategies import checkpoint_positions, HistogramTracker

log = logging.getLogger(__name__)


def _make_dataset(cfg) -> Dataset:
    cls = hydra.utils.get_class(cfg.data._target_)
    return cls(cfg=cfg.data)


def _is_histogram_strategy(stype):
    return stype in ("histogram_frozen", "histogram_periodic", "histogram_exp_decay")


def _make_histogram_tracker(strategy, max_len):
    """Create a HistogramTracker from strategy config."""
    stype = strategy.type
    mode = {'histogram_frozen': 'frozen',
            'histogram_periodic': 'periodic',
            'histogram_exp_decay': 'exp_decay'}[stype]
    budget = strategy.n_blocks
    gamma = strategy.get('gamma', 0.99)
    replan_interval = strategy.get('replan_interval', 100)
    bin_size = strategy.get('bin_size', 1)
    log.info("histogram %s [%s]: budget=%d, bin_size=%d, max_len=%d",
             strategy.tag, stype, budget, bin_size, max_len)
    return HistogramTracker(max_len, budget, mode=mode, gamma=gamma,
                            replan_interval=replan_interval, bin_size=bin_size)


def warmup_cache(model, dataset, requests, vocab_size, strategy,
                 kv_budget, gdn_budget, cache=None, progress=False,
                 histogram_tracker=None):
    """Run requests through the model to fill the cache, without timing."""
    dev = _model_device(model)
    uses_cache = strategy.type != "no_cache"
    is_hist = _is_histogram_strategy(strategy.type)
    if cache is None:
        cache = PrefixCache(kv_budget, gdn_budget)

    for req in tqdm(requests, desc=f"{strategy.tag} warmup", disable=not progress):
        conv_id = dataset.conv_id(req)
        all_toks = dataset.get_tokens(req)
        input_ids = torch.tensor([all_toks], dtype=torch.long) % vocab_size
        seq_len = input_ids.shape[1]

        cached_store, match_len = cache.find_best_prefix(conv_id, input_ids[0])
        input_ids = input_ids.to(dev)
        if is_hist and histogram_tracker is not None:
            histogram_tracker.observe(match_len)

        if cached_store is not None:
            prefill_from_checkpoint(model, input_ids, cached_store)
        else:
            prefill_baseline(model, input_ids)
        _sync_device(dev)
        positions = checkpoint_positions(seq_len, histogram_tracker=histogram_tracker, **strategy)
        store = prefill_and_capture_at(model, input_ids, positions)
        _sync_device(dev)
        store.to("cpu")
        cache.put(conv_id, store)

    if is_hist and histogram_tracker is not None:
        if strategy.type == 'histogram_frozen':
            histogram_tracker.freeze()

    return cache if uses_cache else PrefixCache(kv_budget, gdn_budget)


def simulate(model, dataset, requests, vocab_size, strategy,
             kv_budget, gdn_budget, cache=None, progress=False,
             histogram_tracker=None):
    """Run all requests through the model with given caching strategy."""
    dev = _model_device(model)
    uses_cache = strategy.type != "no_cache"
    is_hist = _is_histogram_strategy(strategy.type)
    online_hist = is_hist and strategy.type != 'histogram_frozen'
    if cache is None and uses_cache:
        cache = PrefixCache(kv_budget, gdn_budget)
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

        if cached_store is not None:
            hit = True
            _sync_device(dev)
            t0 = time.perf_counter()
            prefill_from_checkpoint(model, input_ids, cached_store)
            _sync_device(dev)
            dt = time.perf_counter() - t0
            kv_len = min(cached_store.kv_len, seq_len)
            reusable_kv = kv_len
            ckpt = cached_store.best_checkpoint(min(kv_len, seq_len))
            if ckpt:
                tokens_saved = ckpt.position
                reusable_gdn = ckpt.position
        else:
            _sync_device(dev)
            t0 = time.perf_counter()
            prefill_baseline(model, input_ids)
            _sync_device(dev)
            dt = time.perf_counter() - t0

        capture_s = 0.0
        if uses_cache:
            positions = checkpoint_positions(seq_len, histogram_tracker=histogram_tracker, **strategy)
            _sync_device(dev)
            cap_t0 = time.perf_counter()
            store = prefill_and_capture_at(model, input_ids, positions)
            _sync_device(dev)
            capture_s = time.perf_counter() - cap_t0
            store.to("cpu")
            cache.put(conv_id, store)
        _sync_device(dev)

        per_request.append({
            "conv_id": str(conv_id), "seq_len": seq_len,
            "time_s": dt, "capture_s": capture_s, "hit": hit,
            "tokens_saved": tokens_saved,
            "reusable_kv": reusable_kv, "reusable_gdn": reusable_gdn,
            "prefix_match": match_len,
            "n_cache_entries": cache.n_entries if cache else 0,
            "cache_kv_bytes": cache.kv_used if cache else 0,
            "cache_gdn_bytes": cache.gdn_used if cache else 0,
        })

    wall_time = time.perf_counter() - wall_t0
    return {
        "total_time": sum(e["time_s"] for e in per_request),
        "total_capture_time": sum(e["capture_s"] for e in per_request),
        "wall_time": wall_time,
        "hits": sum(1 for e in per_request if e["hit"]),
        "tokens_saved": sum(e["tokens_saved"] for e in per_request),
        "tokens_total": sum(e["seq_len"] for e in per_request),
        "cache_entries": cache.n_entries if cache else 0,
        "per_request": per_request,
    }


def run_strategy(model, dataset, strat, train_requests, test_requests,
                 vocab_size, kv_budget, gdn_budget,
                 max_seq_len, progress):
    """Run a single strategy end-to-end. All caches are local and freed on return."""
    hist_tracker = None
    if _is_histogram_strategy(strat.type):
        hist_tracker = _make_histogram_tracker(strat, max_seq_len)

    log.info("Strategy: %s — warming cache on train split...", strat.tag)
    cache = warmup_cache(model, dataset, train_requests, vocab_size,
                         strat, kv_budget, gdn_budget, progress=progress,
                         histogram_tracker=hist_tracker)
    log.info("Strategy: %s — evaluating on test split...", strat.tag)
    res = simulate(model, dataset, test_requests, vocab_size,
                   strat, kv_budget, gdn_budget, cache=cache, progress=progress,
                   histogram_tracker=hist_tracker)

    log.info("  %s: prefill=%.1fs capture=%.1fs wall=%.1fs",
             strat.tag, res["total_time"], res["total_capture_time"], res["wall_time"])
    if res["hits"] > 0:
        log.info("    hits: %d/%d (%.1f%%)",
                 res["hits"], len(test_requests), res["hits"] / len(test_requests) * 100)
    return res


@hydra.main(config_path=r"../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = setup_output_dir(cfg, "benchmark_e2e")
    resolve_strategies(cfg)
    model = make_model(cfg)
    config = model.config
    e2e = cfg.benchmark_e2e
    strategies = list(cfg.strategies)
    kv_budget = int(e2e.kv_budget_gb * 1e9)
    gdn_budget = int(e2e.gdn_budget_gb * 1e9)
    progress = e2e.get("progress", True)
    vocab_size = config.vocab_size

    # Load dataset (interleaving happens inside)
    dataset = _make_dataset(cfg)
    dataset.load(seed=cfg.seed)
    train_requests, test_requests = dataset.train_test_split()

    total_tokens = sum(len(dataset.get_tokens(r)) for r in test_requests)
    log.info("Train: %d, Test: %d requests, test tokens: %d, KV budget: %.1f GB, GDN budget: %.1f GB",
             len(train_requests), len(test_requests), total_tokens,
             e2e.kv_budget_gb, e2e.gdn_budget_gb)

    summary = {
        "model_name": cfg.model.name,
        "n_train_requests": len(train_requests),
        "n_test_requests": len(test_requests),
        "total_tokens": total_tokens,
        "train_frac": cfg.data.get("train_frac", 0.5),
        "kv_budget_gb": e2e.kv_budget_gb,
        "gdn_budget_gb": e2e.gdn_budget_gb,
        "strategies": {},
        "strategy_styles": OmegaConf.to_container(cfg.strategies, resolve=True),
    }
    summary_path = out_dir / "e2e_summary.json"

    # initial warmup to alleviate gpu clock control, etc
    print("Warming up cores")
    cache = warmup_cache(model, dataset, train_requests[:e2e.warmup_seqs], vocab_size,
                   strategies[0], kv_budget, gdn_budget, progress=progress)
    simulate(model, dataset, test_requests[:e2e.warmup_seqs], vocab_size,
                    strategies[0], kv_budget, gdn_budget, cache=cache, progress=progress)

    for strat in strategies:
        res = run_strategy(model, dataset, strat, train_requests, test_requests,
                           vocab_size, kv_budget, gdn_budget,
                           cfg.data.max_seq_len, progress)

        _save_jsonl(out_dir / f"e2e_{strat.tag}.jsonl", res["per_request"])
        summary["strategies"][strat.tag] = {
            "total_time": res["total_time"],
            "total_capture_time": res["total_capture_time"],
            "wall_time": res["wall_time"],
            "hits": res["hits"],
            "tokens_saved": res["tokens_saved"],
            "tokens_total": res["tokens_total"],
            "cache_entries": res["cache_entries"],
        }
        summary_path.write_text(json.dumps(summary, indent=2))

    log.info("Done. Results saved to %s", out_dir)

if __name__ == "__main__":
    main()
