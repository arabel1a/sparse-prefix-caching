"""End-to-end prefix caching evaluation on real conversation traces.

Runs checkpoint placement strategies with FIFO prefix cache.
All checkpoint strategies store both GDN recurrent states and attention KV.
"""
import spase_cache
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm

from spase_cache.checkpoint_cache import (
    prefill_from_checkpoint,
)
from spase_cache.utils import (
    setup_output_dir,
    interleave as _interleave_util,
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

def load_requests(processed_path):
    """Load pre-tokenized requests from prepare_data.py parquet output.

    Returns (requests, conv_tokens) where:
      requests: list of (url, turn, n_msgs) lightweight descriptors
      conv_tokens: dict url -> list of list[int] (per-message token lists)

    Tensors are NOT materialized here; build them lazily in simulate().
    """
    import polars as pl
    df = pl.read_parquet(processed_path)

    conv_tokens = {}
    conv_msg_indices = {}
    for url in df["url"].unique(maintain_order=True).to_list():
        conv = df.filter(pl.col("url") == url).sort("message_index")
        conv_tokens[url] = conv["tokens"].to_list()
        conv_msg_indices[url] = conv["message_index"].to_list()

    user_rows = df.filter(pl.col("role") == "user").sort("ts")
    requests = []
    turn_counter = {}
    for url, msg_idx in zip(
        user_rows["url"].to_list(), user_rows["message_index"].to_list()
    ):
        n_msgs = conv_msg_indices[url].index(msg_idx) + 1
        turn = turn_counter.get(url, 0)
        turn_counter[url] = turn + 1
        requests.append((url, turn, n_msgs))

    log.info("Loaded %d requests from %s", len(requests), processed_path)
    return requests, conv_tokens


def interleave(requests, seed):
    """
    Interleave conversations with Poisson arrival times, preserving turn order.
    This makes ny benchmark closer to real high-load system. Since the datset I
    an using is crowdsourced, its timestamp almost never overlap. This will lead
    to 100% cache hit -- each turn in conversation orrives right after previous.
    """
    by_conv = defaultdict(list)
    for req in requests:
        by_conv[req[0]].append(req)

    rng = np.random.RandomState(seed)
    conv_ids = list(by_conv.keys())
    rng.shuffle(conv_ids)

    queues = {cid: list(by_conv[cid]) for cid in conv_ids}
    arrival = {cid: rng.exponential(1.0) for cid in conv_ids}
    ordered = []
    while queues:
        cid = min(queues, key=lambda c: arrival[c])
        ordered.append(queues[cid].pop(0))
        if queues[cid]:
            arrival[cid] += rng.exponential(1.0)
        else:
            del queues[cid]
            del arrival[cid]
    return ordered


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
    exclude_full_hits = strategy.get('exclude_full_hits', True)
    log.info("histogram %s [%s]: budget=%d, bin_size=%d, max_len=%d, exclude_full=%s",
             strategy.tag, stype, budget, bin_size, max_len, exclude_full_hits)
    return HistogramTracker(max_len, budget, mode=mode, gamma=gamma,
                            replan_interval=replan_interval, bin_size=bin_size,
                            exclude_full_hits=exclude_full_hits)


def _get_overlap_depth(cache, conv_id, turn, seq_len):
    """Get the overlap depth for a request (how many prefix tokens are cached)."""
    cached_store, _ = cache.find_best_prefix(conv_id, turn)
    if cached_store is not None:
        return min(cached_store.kv_len, seq_len), cached_store
    return 0, None


def warmup_cache(model, requests, conv_tokens, vocab_size, strategy,
                 kv_budget, gdn_budget, cache=None, progress=False,
                 histogram_tracker=None):
    """Run requests through the model to fill the cache, without timing.

    For no_cache strategy, still runs prefill_baseline on all train requests
    so that MPS/CUDA caches and memory allocators are fully warmed before
    the timed evaluation phase.

    For histogram strategies, also observes overlap depths and builds the
    histogram. Returns (cache, histogram_tracker).
    """
    dev = _model_device(model)
    uses_cache = strategy.type != "no_cache"
    is_hist = _is_histogram_strategy(strategy.type)
    if cache is None:
        cache = PrefixCache(kv_budget, gdn_budget)

    for conv_id, turn, n_msgs in tqdm(requests, desc=f"{strategy.tag} warmup", disable=not progress):
        all_toks = [t for toks in conv_tokens[conv_id][:n_msgs] for t in toks]
        input_ids = torch.tensor([all_toks], dtype=torch.long).to(dev) % vocab_size
        seq_len = input_ids.shape[1]

        # Observe overlap depth for histogram strategies
        overlap_depth, cached_store = _get_overlap_depth(cache, conv_id, turn, seq_len)
        if is_hist and histogram_tracker is not None:
            full_hit = cached_store is not None and overlap_depth >= cached_store.kv_len
            histogram_tracker.observe(overlap_depth, is_full_hit=full_hit)

        if cached_store is not None:
            prefill_from_checkpoint(model, input_ids, cached_store)
        else:
            prefill_baseline(model, input_ids)
        _sync_device(dev)
        positions = checkpoint_positions(seq_len, histogram_tracker=histogram_tracker, **strategy)
        store = prefill_and_capture_at(model, input_ids, positions)
        _sync_device(dev)
        store.to("cpu")
        cache.put((conv_id, turn), store)

    # For frozen mode, solve DP once after all warmup data
    if is_hist and histogram_tracker is not None:
        if strategy.type == 'histogram_frozen':
            histogram_tracker.freeze()

    return cache if uses_cache else PrefixCache(kv_budget, gdn_budget)


def simulate(model, requests, conv_tokens, vocab_size, strategy,
             kv_budget, gdn_budget, cache=None, progress=False,
             histogram_tracker=None):
    """Run all requests through the model with given caching strategy.

    KV cache and GDN checkpoints have separate budgets. KV cache alone cannot
    skip compute (attention needs Q from GDN FFN hidden states), but is required
    so attention can attend to history without re-encoding old tokens.

    For histogram_periodic/histogram_exp_decay, continues observing overlaps
    and replanning during the test phase.
    """
    dev = _model_device(model)
    uses_cache = strategy.type != "no_cache"
    is_hist = _is_histogram_strategy(strategy.type)
    online_hist = is_hist and strategy.type != 'histogram_frozen'
    if cache is None and uses_cache:
        cache = PrefixCache(kv_budget, gdn_budget)
    per_request = []
    wall_t0 = time.perf_counter()

    for conv_id, turn, n_msgs in tqdm(requests, desc=strategy.tag, disable=not progress):
        all_toks = [t for toks in conv_tokens[conv_id][:n_msgs] for t in toks]
        input_ids = torch.tensor([all_toks], dtype=torch.long).to(dev) % vocab_size
        seq_len = input_ids.shape[1]

        hit = False
        tokens_saved = 0
        cached_store = None
        cached_turn = -1
        reusable_kv = 0
        reusable_gdn = 0

        if uses_cache:
            cached_store, cached_turn = cache.find_best_prefix(conv_id, turn)

        # Observe overlap for online histogram strategies
        if online_hist and histogram_tracker is not None:
            overlap = min(cached_store.kv_len, seq_len) if cached_store else 0
            full_hit = cached_store is not None and overlap >= cached_store.kv_len
            histogram_tracker.observe(overlap, is_full_hit=full_hit)

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

        # Capture checkpoints for cache (or dummy forward to keep device warm)
        capture_s = 0.0
        if uses_cache:
            positions = checkpoint_positions(seq_len, histogram_tracker=histogram_tracker, **strategy)
            _sync_device(dev)
            cap_t0 = time.perf_counter()
            store = prefill_and_capture_at(model, input_ids, positions)
            _sync_device(dev)
            capture_s = time.perf_counter() - cap_t0
            store.to("cpu")
            cache.put((conv_id, turn), store)
        _sync_device(dev)

        per_request.append({
            "conv_id": str(conv_id), "turn": turn, "seq_len": seq_len,
            "time_s": dt, "capture_s": capture_s, "hit": hit,
            "tokens_saved": tokens_saved,
            "reusable_kv": reusable_kv, "reusable_gdn": reusable_gdn,
            "cached_turn": cached_turn,
            "turn_gap": turn - cached_turn if hit else -1,
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


def run_strategy(model, strat, train_requests, test_requests,
                 conv_tokens, vocab_size, kv_budget, gdn_budget,
                 max_seq_len, progress):
    """Run a single strategy end-to-end. All caches are local and freed on return."""
    hist_tracker = None
    if _is_histogram_strategy(strat.type):
        hist_tracker = _make_histogram_tracker(strat, max_seq_len)

    log.info("Strategy: %s — warming cache on train split...", strat.tag)
    cache = warmup_cache(model, train_requests, conv_tokens, vocab_size,
                         strat, kv_budget, gdn_budget, progress=progress,
                         histogram_tracker=hist_tracker)
    log.info("Strategy: %s — evaluating on test split...", strat.tag)
    res = simulate(model, test_requests, conv_tokens, vocab_size,
                   strat, kv_budget, gdn_budget, cache=cache, progress=progress,
                   histogram_tracker=hist_tracker)

    log.info("  %s: prefill=%.1fs capture=%.1fs wall=%.1fs",
             strat.tag, res["total_time"], res["total_capture_time"], res["wall_time"])
    if res["hits"] > 0:
        n_test = len(test_requests)
        n_hits = res["hits"]
        consecutive = sum(1 for e in res["per_request"] if e["turn_gap"] == 1)
        non_consecutive = sum(1 for e in res["per_request"] if e["hit"] and e["turn_gap"] > 1)
        log.info("    hits: %d/%d (%.1f%%) — consecutive: %d (%.1f%%), non-consecutive: %d (%.1f%%)",
                 n_hits, n_test, n_hits / n_test * 100,
                 consecutive, consecutive / n_test * 100,
                 non_consecutive, non_consecutive / n_test * 100)
        if non_consecutive > 0:
            nc_kv = [e["reusable_kv"] for e in res["per_request"] if e["hit"] and e["turn_gap"] > 1]
            log.info("    non-consecutive hit overlap: mean=%.0f, median=%.0f tokens",
                     np.mean(nc_kv), np.median(nc_kv))
    return res


@hydra.main(config_path=r"../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = setup_output_dir(cfg, "benchmark_e2e")
    model = make_model(cfg)
    dev = _model_device(model)
    config = model.config
    e2e = cfg.benchmark_e2e
    strategies = list(cfg.strategies)
    kv_budget = int(e2e.kv_budget_gb * 1e9)
    gdn_budget = int(e2e.gdn_budget_gb * 1e9)
    progress = e2e.get("progress", True)

    # Load and interleave requests
    requests, conv_tokens = load_requests(cfg.data.processed)
    requests = interleave(requests, cfg.seed)
    vocab_size = config.vocab_size

    # Train/test split
    train_frac = cfg.data.get("train_frac", 0.5)
    n_train = int(len(requests) * train_frac)
    train_requests = requests[:n_train]
    test_requests = requests[n_train:]

    total_tokens = sum(
        sum(len(t) for t in conv_tokens[url][:n_msgs])
        for url, _, n_msgs in test_requests
    )
    log.info("Train: %d, Test: %d requests, test tokens: %d, KV budget: %.1f GB, GDN budget: %.1f GB",
             len(train_requests), len(test_requests), total_tokens,
             e2e.kv_budget_gb, e2e.gdn_budget_gb)

    from omegaconf import OmegaConf
    summary = {
        "model_name": cfg.model.name,
        "n_train_requests": len(train_requests),
        "n_test_requests": len(test_requests),
        "total_tokens": total_tokens,
        "train_frac": train_frac,
        "kv_budget_gb": e2e.kv_budget_gb,
        "gdn_budget_gb": e2e.gdn_budget_gb,
        "strategies": {},
        "strategy_styles": OmegaConf.to_container(cfg.strategies, resolve=True),
    }
    summary_path = out_dir / "e2e_summary.json"
    
    # initial warmup to allevaite gpu clock control, etc
    print("Warming up cores")
    cache = warmup_cache(model, train_requests[:e2e.warmup_seqs], conv_tokens, vocab_size,
                   strategies[0], kv_budget, gdn_budget, progress=progress)
    simulate(model, test_requests[:e2e.warmup_seqs], conv_tokens, vocab_size,
                    strategies[0], kv_budget, gdn_budget, cache=cache, progress=progress)


    for strat in strategies:
        res = run_strategy(model, strat, train_requests, test_requests,
                           conv_tokens, vocab_size, kv_budget, gdn_budget,
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
