"""
End-to-end prefix caching evaluation on real ShareGPT conversation traces.

## Evaluation scheme

**Data preparation:**
1. Load ShareGPT conversations from CSV (each row = one turn with role + text).
3. Take `n_conversations` longest (default 100). Crop their length to max_seq_len
4. For each remaining conversation with T turns, generate T prefix requests:
   - request 0: turn 1 only           (user)
   - request 1: turns 1–2             (user, llm)
   - request 2: turns 1–3             (user, llm, user)
   - ...
   - request T-1: turns 1–T           (full conversation)
   Each request is the concatenation of all turns up to that point, separated
   by role tags (e.g. "<|user|> ... <|llm|> ..."). This is how requests are
   processed in real multi-turn scenarios.

**Simulation:**
1. Conversations are randomly interleaved (seed-controlled), but turns within
   each conversation arrive in order (turn 1 before turn 2, etc.). This models
   realistic multi-tenant traffic where a later turn cannot arrive before its
   predecessor.
2. Requests are processed one at a time (no batching).
3. A fixed memory budget (`cache_budget_bytes`) is available for caching.
4. For each caching strategy, a simple FIFO cache is maintained:
   - On processing a request, check if any prefix of the request's token
     sequence is already cached (i.e. a previous request from the same
     conversation with fewer turns).
   - If a cache hit is found, compute only the remaining (uncached) tokens.
   - After processing, store the request's cache entry (GDN state + KV for
     logarithmic/block strategies, or just KV for attention-only).
   - If the cache is full, evict the oldest entry.

**Strategies compared:**
1. **No caching:** Always recompute from scratch (baseline).
2. **Attention-only KV cache:** Skip attention layers (simulates cached KV).
3. **Block hybrid B=16, no KV:** Cache GDN states at block boundaries only.
4. **Logarithmic, no KV:** Cache GDN states at 2^i positions only.
5. **Block hybrid + attention:** Cache GDN at block boundaries + KV.
6. **Logarithmic + attention:** Cache GDN at 2^i positions + KV.

Strategies 2–4 are measured with attention layers disabled. For 3 and 4, the
per-request attention cost (= no_cache − attn_only time) is added back to give
the realistic total, since attention must still recompute all N tokens (no KV).

**Metrics:**
- Total prefill time across all requests.
- Cache hit rate (fraction of requests with a usable cache entry).
- GDN tokens saved: tokens skipped via GDN recurrent state checkpoints.

**Walltime vs reported time:**
On cache hits the strategy must re-capture checkpoints for the new (longer)
sequence. This should be done with similar time to just forward pass 
through the rest of the sequence. That's why I measure just a forward time. 

Current kernels does not allow to e.g. extract GDN intermediate states. Instead, 
to update the cache (e.g. in block-boundary strategy) I need to run MANY small
kernel calls (e.g. for just 16 tokens). This time is NOT included in the reported 
"total_time" (only the inference pass is timed).


**Simplifications:**
- No cache block alignment / overlap detection — each conversation's cache is
  stored as a single entry keyed by conversation ID + number of turns.
- FIFO eviction (no LRU/LFU).
- Single-request processing (no batching).
- Random weights (shapes from Qwen3.5-0.8B, single layer group);
  real tokenizer is used for realistic token counts, IDs mapped to model vocab via modulo.
- Requests arrive randomly (but keeping order within conversations). 
"""

import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm

from checkpoint_cache import (
    apply_patch,
    make_model,
    prefill_baseline,
    prefill_and_capture,
    prefill_from_checkpoint,
    PrefixCheckpointStore,
    _model_device,
    _sync_device,
    _checkpoint_positions,
)
from contextlib import nullcontext
import matplotlib.pyplot as plt
from benchmark_baselines import prefill_and_capture_at, _block_positions, disable_attention_layers


# ---------------------------------------------------------------------------
# Truncate store to fit budget: drop GDN checkpoints from the smallest
# positions first until the store fits.  KV cache is never dropped since
# it is shared by all strategies that use caching.
# ---------------------------------------------------------------------------
def _truncate_store(store, budget_bytes):
    """Remove GDN checkpoints (smallest positions first) until store fits in budget.
    Returns (store, new_size) or (None, 0) if even KV-only exceeds budget."""
    size = store.memory_bytes()
    if size <= budget_bytes:
        return store, size
    # Drop checkpoints smallest-first
    positions = sorted(store.checkpoints.keys())
    for pos in positions:
        ckpt = store.checkpoints.pop(pos)
        for t in ckpt.recurrent_states.values():
            size -= t.nelement() * t.element_size()
        for t in ckpt.conv_states.values():
            size -= t.nelement() * t.element_size()
        if size <= budget_bytes:
            return store, size
    # KV-only still too big
    if size > budget_bytes:
        return None, 0
    return store, size


# ---------------------------------------------------------------------------
# Simple FIFO cache with memory budget
# ---------------------------------------------------------------------------
class PrefixCache:
    def __init__(self, budget_bytes):
        self.budget = budget_bytes
        self.used = 0
        self.entries = OrderedDict()  # key -> (store, size_bytes)

    def get(self, key):
        if key in self.entries:
            return self.entries[key][0]
        return None

    def find_best_prefix(self, conv_id, n_turns):
        """Find the longest cached prefix for this conversation with <= n_turns."""
        best = None
        best_turns = 0
        for t in range(n_turns, 0, -1):
            k = (conv_id, t)
            if k in self.entries:
                return self.entries[k][0], t
        return None, 0

    def put(self, key, store, size_bytes):
        if size_bytes > self.budget:
            store, size_bytes = _truncate_store(store, self.budget)
            if store is None:
                return
        # Evict oldest until we have room
        while self.used + size_bytes > self.budget and self.entries:
            _, (_, evicted_size) = self.entries.popitem(last=False)
            self.used -= evicted_size
        self.entries[key] = (store, size_bytes)
        self.used += size_bytes

    @property
    def n_entries(self):
        return len(self.entries)


# ---------------------------------------------------------------------------
# Build requests from conversations
# ---------------------------------------------------------------------------
def build_requests(cfg):
    """Load conversations, filter, and generate prefix requests.
    Selects the longest conversations (by total char count) up to n_conversations."""
    e2e = cfg.benchmark_e2e
    df = pd.read_csv(e2e.data_path)

    # Compute total character count per conversation
    total_chars = df.groupby('url')['plain_text'].apply(lambda x: x.astype(str).str.len().sum())
    longest = total_chars.sort_values(ascending=False).head(e2e.n_conversations).index
    df = df[df['url'].isin(longest)]

    print(f"total {df['plain_text'].apply(lambda x: len(str(x))).sum()} chars")
    
    
    # Build prefix requests
    requests = []  # list of (conv_id, n_turns, token_text)
    for conv_id in df['url'].unique():
        conv = df[df['url'] == conv_id].sort_values('message_index').head(e2e.max_rounds)
        turns = []
        for _, row in conv.iterrows():
            role = row['role']
            text = str(row['plain_text'])
            turns.append(f"<|{role}|> {text}")

        for t in range(1, len(turns) + 1):
            prefix_text = "\n".join(turns[:t])
            requests.append((conv_id, t, prefix_text))

    print(f"Loaded {len(df['url'].unique())} conversations")
    return requests


def tokenize_requests(requests, vocab_size, tokenizer_name="Qwen/Qwen3.5-0.8B", max_seq_len=None):
    """Tokenize requests using the real HF tokenizer for realistic token
    counts and prefix-sharing patterns."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    assert tokenizer.vocab_size <= vocab_size, (
        f"Tokenizer vocab ({tokenizer.vocab_size}) > model vocab ({vocab_size}). "
        f"Set model.vocab_size >= {tokenizer.vocab_size} in config."
    )
    tokenized = []
    last_len = {}
    for conv_id, n_turns, text in requests:
        # truncation may result in strange situations where I have several exactly same prompts 
        # in my request (if non-last round exceeds the len). This patch avoids that:
        if last_len.get(conv_id,0) >= max_seq_len:
            continue
        enc = tokenizer(
                text, 
                add_special_tokens=False, 
                return_tensors="pt",
                max_length=max_seq_len,
                truncation=True,
                truncation_side="left",
        )["input_ids"]

        tokenized.append((conv_id, n_turns, enc))
        last_len[conv_id] = enc.shape[1]
    print(len(tokenized))
    return tokenized


# ---------------------------------------------------------------------------
# Run simulation for one strategy
# ---------------------------------------------------------------------------
def simulate(model, requests, strategy, cache_budget, block_size=16, progress=False):
    """Run all requests through the model with given caching strategy.

    Strategies:
      no_cache    — full model, no caching (baseline)
      attn_only   — attention disabled, no caching (= GDN+FFN baseline)
      block / log — attention disabled, GDN checkpoints cached
      block_and_attn  / log_and_attn      — full model, GDN + KV cached

    For *_no_attn strategies the measured time is GDN-only; the caller adds
    back per-request attention cost derived from (no_cache − attn_only).
    """
    dev = _model_device(model)
    cache = PrefixCache(cache_budget)

    skip_attention = strategy in ("attn_only", "block", "log")
    no_caching = strategy in ("no_cache", "attn_only")

    total_time = 0.0
    hits = 0
    tokens_saved = 0
    tokens_total = 0
    per_request = []

    ctx = disable_attention_layers(model) if skip_attention else nullcontext()

    with ctx:
        for i, (conv_id, n_turns, input_ids) in enumerate(tqdm(requests, desc=strategy, disable=not progress)):
            input_ids = input_ids.to(dev)
            seq_len = input_ids.shape[1]
            tokens_total += seq_len

            req_hit = False
            req_tokens_saved = 0

            if no_caching:
                _sync_device(dev)
                t0 = time.perf_counter()
                prefill_baseline(model, input_ids)
                _sync_device(dev)
                dt = time.perf_counter() - t0
                total_time += dt
                per_request.append({
                    "conv_id": str(conv_id), "n_turns": n_turns, "seq_len": seq_len,
                    "time_s": dt, "hit": False, "tokens_saved": 0,
                })
                continue

            # --- Caching strategies ---
            cached_store, cached_turns = cache.find_best_prefix(conv_id, n_turns)

            if strategy in ("block_and_attn", "block"):
                ckpt_positions = _block_positions(seq_len, block_size)
            else:  # log_and_attn / log
                ckpt_positions = _checkpoint_positions(seq_len)

            if cached_store is not None:
                req_hit = True
                hits += 1
                cached_store.to(dev)
                _sync_device(dev)
                t0 = time.perf_counter()
                prefill_from_checkpoint(model, input_ids, cached_store)
                _sync_device(dev)
                dt = time.perf_counter() - t0
                total_time += dt
                cached_store.to("cpu")

                ckpt = cached_store.best_checkpoint(seq_len)
                if ckpt:
                    req_tokens_saved = ckpt.position
                    tokens_saved += req_tokens_saved

                # Re-capture checkpoints for the new longer sequence (untimed)
                store = prefill_and_capture_at(model, input_ids, ckpt_positions)
            else:
                # Cache miss: timed forward, then capture (untimed)
                _sync_device(dev)
                t0 = time.perf_counter()
                prefill_baseline(model, input_ids)
                _sync_device(dev)
                dt = time.perf_counter() - t0
                total_time += dt

                store = prefill_and_capture_at(model, input_ids, ckpt_positions)

            # no-attn strategies don't store KV
            if skip_attention:
                store.kv_cache_keys = {}
                store.kv_cache_values = {}

            size = store.memory_bytes()
            store.to("cpu")
            cache.put((conv_id, n_turns), store, size)

            per_request.append({
                "conv_id": str(conv_id), "n_turns": n_turns, "seq_len": seq_len,
                "time_s": dt, "hit": req_hit, "tokens_saved": req_tokens_saved,
            })

    return {
        "total_time": total_time,
        "hits": hits,
        "tokens_saved": tokens_saved,
        "tokens_total": tokens_total,
        "cache_entries": cache.n_entries,
        "per_request": per_request,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    apply_patch()
    model = make_model(cfg)
    dev = _model_device(model)
    config = model.config
    e2e = cfg.benchmark_e2e

    # Build and tokenize requests
    requests = build_requests(cfg)
    tokenizer_name = cfg.model.get("tokenizer", "Qwen/Qwen3.5-0.8B")
    max_seq_len = e2e.get("max_seq_len", None)
    tokenized = tokenize_requests(requests, config.vocab_size, tokenizer_name=tokenizer_name,
                                  max_seq_len=max_seq_len)


    # Interleave conversations randomly, but keep turn order within each conversation.
    from collections import defaultdict
    by_conv = defaultdict(list)
    for req in tokenized:
        by_conv[req[0]].append(req)

    rng = np.random.RandomState(e2e.seed)
    conv_ids = list(by_conv.keys())
    rng.shuffle(conv_ids)

    queues = {cid: list(by_conv[cid]) for cid in conv_ids}
    arrival = {cid: rng.exponential(1.0) for cid in conv_ids}
    tokenized_ordered = []
    while queues:
        cid = min(queues, key=lambda c: arrival[c])
        tokenized_ordered.append(queues[cid].pop(0))
        if queues[cid]:
            arrival[cid] += rng.exponential(1.0)
        else:
            del queues[cid]
            del arrival[cid]
    tokenized = tokenized_ordered

    print(f"\nTotal requests: {len(tokenized)}")
    total_tokens = sum(ids.shape[1] for _, _, ids in tokenized)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Cache budget: {e2e.cache_budget_gb:.1f} GB = {e2e.cache_budget_gb * 1e9:.0f} bytes")

    cache_budget = int(e2e.cache_budget_gb * 1e9)
    B = e2e.block_size
    progress = e2e.progress

    # Run all 6 direct simulations
    run_strategies = [
        "no_cache", "attn_only",
        "block", "log",
        "block_and_attn", "log_and_attn",
    ]
    results = {}

    # Warmup
    print("\nWarming up...")
    dummy = torch.randint(0, config.vocab_size, (1, 512)).to(dev)
    for _ in range(3):
        prefill_baseline(model, dummy)
    with disable_attention_layers(model):
        for _ in range(3):
            prefill_baseline(model, dummy)
    _sync_device(dev)

    for strat in run_strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strat}")
        print(f"{'='*60}")
        res = simulate(model, tokenized, strat, cache_budget, B, progress=progress)
        results[strat] = res
        suffix = " (GDN-only, attn cost added later)" if strat in ("block", "log") else ""
        print(f"  Total time: {res['total_time']:.1f}s{suffix}")
        if res['hits'] > 0:
            print(f"  Cache hits: {res['hits']}/{len(tokenized)} "
              f"({res['hits']/len(tokenized)*100:.1f}%)")
        if res['tokens_saved'] > 0:
            print(f"  Tokens saved: {res['tokens_saved']:,}/{res['tokens_total']:,} "
                  f"({res['tokens_saved']/res['tokens_total']*100:.1f}%)")

    # Derive *_no_attn real times: GDN-only measured time + per-request attention cost.
    # attn_cost_i = no_cache_i - attn_only_i  (both are non-caching runs, so paired)
    for strat in ["block", "log"]:
        for i in range(len(tokenized)):
            attn_cost = max(
                results["no_cache"]["per_request"][i]["time_s"]
                - results["attn_only"]["per_request"][i]["time_s"],
                0,
            )
            results[strat]["per_request"][i]["time_s"] += attn_cost
        results[strat]["total_time"] = sum(
            e["time_s"] for e in results[strat]["per_request"]
        )

    # Summary table
    report = ["no_cache", "attn_only", "block", "log", "block_and_attn", "log_and_attn"]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'Time (s)':>10} {'Speedup':>8} {'Hit rate':>10} {'GDN saved':>10}")
    print("-" * 60)
    t_base = results['no_cache']['total_time']
    for strat in report:
        r = results[strat]
        speedup = t_base / r['total_time'] if r['total_time'] > 0 else float('inf')
        has_cache = strat not in ('no_cache', 'attn_only')
        hit_rate = r['hits'] / len(tokenized) * 100 if has_cache else 0
        tok_saved = r['tokens_saved'] / r['tokens_total'] * 100 if r['tokens_total'] > 0 and has_cache else 0
        print(f"{strat:<20} {r['total_time']:>10.1f} {speedup:>7.2f}x {hit_rate:>9.1f}% {tok_saved:>9.1f}%")

    # Save per-request logs
    out_dir = Path(e2e.get("output_dir", "outputs"))
    plots_path = Path(e2e.boxplots)
    out_dir.mkdir(parents=True, exist_ok=True)
    for strat in report:
        out_path = out_dir / f"e2e_{strat}.jsonl"
        with open(out_path, "w") as f:
            for entry in results[strat]["per_request"]:
                f.write(json.dumps(entry) + "\n")
    print(f"\nPer-request logs saved to {out_dir}/e2e_*.jsonl")

    # --- Boxplots ---
    labels = {
        "no_cache":      "No cache",
        "attn_only":     "Attn-only\nKV cache",
        "block": f"Block\nB={B}",
        "log":   "Log",
        "block_and_attn":    "Block\n+Attn",
        "log_and_attn":      "Log\n+Attn",
    }
    colors = {
        "no_cache":      "black",
        "attn_only":     "tab:red",
        "block": "tab:blue",
        "log":   "tab:orange",
        "block_and_attn":    "tab:green",
        "log_and_attn":      "tab:purple",
    }
    times_ms = {
        s: np.array([e["time_s"] for e in results[s]["per_request"]]) * 1000
        for s in report
    }
    baseline_ms = times_ms["no_cache"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: per-request time distribution
    bp1 = ax1.boxplot(
        [times_ms[s] for s in report],
        labels=[labels[s] for s in report],
        patch_artist=True,
        showfliers=True,
        flierprops=dict(markersize=2, alpha=0.5),
    )
    for patch, s in zip(bp1["boxes"], report):
        patch.set_facecolor(colors[s])
        patch.set_alpha(0.6)
    ax1.set_ylabel("Per-request prefill time (ms)")
    ax1.set_title(f"Time distribution — {cfg.model.name}")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: per-request speedup
    speedup_strats = [s for s in report if s != "no_cache"]
    speedups = {
        s: baseline_ms / np.maximum(times_ms[s], 1e-6)
        for s in speedup_strats
    }
    bp2 = ax2.boxplot(
        [speedups[s] for s in speedup_strats],
        labels=[labels[s] for s in speedup_strats],
        patch_artist=True,
        showfliers=True,
        flierprops=dict(markersize=2, alpha=0.5),
    )
    for patch, s in zip(bp2["boxes"], speedup_strats):
        patch.set_facecolor(colors[s])
        patch.set_alpha(0.6)
    ax2.axhline(y=1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax2.set_ylabel("Speedup vs no-cache")
    ax2.set_title(f"Speedup distribution — {cfg.model.name}")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(plots_path, dpi=200, bbox_inches="tight")
    print(f"Saved boxplots to {plots_path}")

    # --- Time vs context length ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

    for s in report:
        seq_lens = np.array([e["seq_len"] for e in results[s]["per_request"]])
        t_ms = times_ms[s]
        order = np.argsort(seq_lens)
        ax3.scatter(seq_lens, t_ms, s=6, alpha=0.3, color=colors[s], label=labels[s])

    ax3.set_xlabel("Context length (tokens)")
    ax3.set_ylabel("Prefill time (ms)")
    ax3.set_yscale("log")
    ax3.set_title(f"Time vs context length — {cfg.model.name}")
    ax3.legend(fontsize=7, loc="upper left")
    ax3.grid(True, alpha=0.3)

    for s in speedup_strats:
        seq_lens = np.array([e["seq_len"] for e in results[s]["per_request"]])
        sp = speedups[s]
        ax4.scatter(seq_lens, sp, s=6, alpha=0.3, color=colors[s], label=labels[s])

    ax4.axhline(y=1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax4.set_xlabel("Context length (tokens)")
    ax4.set_ylabel("Speedup vs no-cache")
    ax4.set_yscale("log")
    ax4.set_title(f"Speedup vs context length — {cfg.model.name}")
    ax4.legend(fontsize=7, loc="upper left")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    scatter_path = out_dir / "e2e_time_vs_length.png"
    plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
    print(f"Saved time-vs-length to {scatter_path}")


if __name__ == "__main__":
    main()
