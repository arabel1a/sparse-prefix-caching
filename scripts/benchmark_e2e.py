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
    make_model,
    prefill_baseline,
    _model_device,
    _sync_device,
    prefill_and_capture_at,
    PrefixCache,
    _save_jsonl,
)
from spase_cache.strategies import checkpoint_positions

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


def simulate(model, requests, conv_tokens, vocab_size, strategy, cache_budget,
             progress=False):
    """Run all requests through the model with given caching strategy.

    All checkpoint strategies store both GDN recurrent states and attention KV
    cache, since GDN-only checkpointing cannot skip FFN compute (the GA layer
    needs hidden states produced by GDN FFNs).
    """
    dev = _model_device(model)
    uses_cache = strategy.tag != "no_cache"
    cache = PrefixCache(cache_budget) if uses_cache else None
    per_request = []

    for conv_id, turn, n_msgs in tqdm(requests, desc=strategy.tag, disable=not progress):
        all_toks = [t for toks in conv_tokens[conv_id][:n_msgs] for t in toks]
        input_ids = torch.tensor([all_toks], dtype=torch.long).to(dev) % vocab_size
        seq_len = input_ids.shape[1]

        hit = False
        tokens_saved = 0
        cached_store = None

        if uses_cache:
            cached_store, _ = cache.find_best_prefix(conv_id, turn)

        if cached_store is not None:
            hit = True
            _sync_device(dev)
            t0 = time.perf_counter()
            prefill_from_checkpoint(model, input_ids, cached_store)
            _sync_device(dev)
            dt = time.perf_counter() - t0
            ckpt = cached_store.best_checkpoint(seq_len)
            if ckpt:
                tokens_saved = ckpt.position
        else:
            _sync_device(dev)
            t0 = time.perf_counter()
            prefill_baseline(model, input_ids)
            _sync_device(dev)
            dt = time.perf_counter() - t0

        # Capture checkpoints for cache
        if uses_cache:
            positions = checkpoint_positions(seq_len, **strategy)
            store = prefill_and_capture_at(model, input_ids, positions)
            size = store.memory_bytes()
            store.to("cpu")
            cache.put((conv_id, turn), store, size)

        per_request.append({
            "conv_id": str(conv_id), "turn": turn, "seq_len": seq_len,
            "time_s": dt, "hit": hit, "tokens_saved": tokens_saved,
        })

    return {
        "total_time": sum(e["time_s"] for e in per_request),
        "hits": sum(1 for e in per_request if e["hit"]),
        "tokens_saved": sum(e["tokens_saved"] for e in per_request),
        "tokens_total": sum(e["seq_len"] for e in per_request),
        "cache_entries": cache.n_entries if cache else 0,
        "per_request": per_request,
    }


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = setup_output_dir(cfg, "benchmark_e2e")
    model = make_model(cfg)
    dev = _model_device(model)
    config = model.config
    e2e = cfg.benchmark_e2e
    strategies = list(cfg.strategies)
    cache_budget = int(e2e.cache_budget_gb * 1e9)
    progress = e2e.get("progress", True)

    # Load and interleave requests
    requests, conv_tokens = load_requests(cfg.data.processed)
    requests = interleave(requests, cfg.seed)
    vocab_size = config.vocab_size

    total_tokens = sum(
        sum(len(t) for t in conv_tokens[url][:n_msgs])
        for url, _, n_msgs in requests
    )
    log.info("Total requests: %d, tokens: %d, cache budget: %.1f GB",
             len(requests), total_tokens, e2e.cache_budget_gb)

    # Warmup
    log.info("Warming up...")
    dummy = torch.randint(0, config.vocab_size, (1, 512)).to(dev)
    for _ in range(3):
        prefill_baseline(model, dummy)
    _sync_device(dev)

    from omegaconf import OmegaConf
    summary = {
        "model_name": cfg.model.name,
        "n_requests": len(requests),
        "total_tokens": total_tokens,
        "cache_budget_gb": e2e.cache_budget_gb,
        "strategies": {},
        "strategy_styles": OmegaConf.to_container(cfg.strategies, resolve=True),
    }
    summary_path = out_dir / "e2e_summary.json"

    for strat in strategies:
        log.info("Strategy: %s", strat.tag)
        res = simulate(model, requests, conv_tokens, vocab_size,
                       strat, cache_budget, progress=progress)

        log.info("  %s: total=%.1fs", strat.tag, res["total_time"])
        if res["hits"] > 0:
            log.info("    hits: %d/%d (%.1f%%)",
                     res["hits"], len(requests), res["hits"] / len(requests) * 100)

        _save_jsonl(out_dir / f"e2e_{strat.tag}.jsonl", res["per_request"])
        summary["strategies"][strat.tag] = {
            "total_time": res["total_time"],
            "hits": res["hits"],
            "tokens_saved": res["tokens_saved"],
            "tokens_total": res["tokens_total"],
            "cache_entries": res["cache_entries"],
        }
        summary_path.write_text(json.dumps(summary, indent=2))

    log.info("Done. Results saved to %s", out_dir)

if __name__ == "__main__":
    main()
