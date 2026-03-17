"""Empirical benchmark of all prefix caching strategies.

Measures 6 strategies matching the theoretical FLOP plot:
1. No caching (baseline)
2. Attention-only KV cache (skip attention, recompute GDN+FFN)
3. Block hybrid B=16, no attention cache
4. Logarithmic, no attention cache
5. Attention + Block hybrid
6. Attention + Logarithmic

**Simplification 1:** Strategies 2,5,6 skips the attention layers entirely 
instead of real reading from cache, since hf does not seem to support this.
Modifying transformer's cache behavior would involve modifying attention kernel.
"""
import hybrid_logarithmic_cache
import gc
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import hydra
from omegaconf import DictConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

from hybrid_logarithmic_cache.checkpoint_cache import (
    apply_patch,
    make_model,
    prefill_baseline,
    prefill_from_checkpoint,
    PrefixCheckpointStore,
    RecurrentCheckpoint,
    _model_device,
    _get_linear_layers,
    _get_attention_layers,
    _sync_device,
    _checkpoint_positions,
)


def _free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def _block_positions(seq_len, block_size):
    return list(range(block_size, seq_len + 1, block_size))

def prefill_and_capture_at(model, input_ids, ckpt_positions):
    """Run prefill capturing GDN+conv states and attention KV at specified positions."""
    device = _model_device(model)
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    config = model.config
    linear_layers = _get_linear_layers(config)
    attn_layers = _get_attention_layers(config)

    ckpt_positions = sorted(set(p for p in ckpt_positions if 0 < p <= seq_len))
    store = PrefixCheckpointStore(prefix_tokens=input_ids.clone())
    boundaries = sorted(set([0] + ckpt_positions + [seq_len]))

    cache = Qwen3_5DynamicCache(config=config)
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        with torch.no_grad():
            out = model(
                input_ids=input_ids[:, start:end],
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(start, end, device=device),
            )
        cache = out.past_key_values

        if end in ckpt_positions:
            store.checkpoints[end] = RecurrentCheckpoint(
                position=end,
                recurrent_states={
                    li: cache.recurrent_states[li].clone()
                    for li in linear_layers if cache.recurrent_states[li] is not None
                },
                conv_states={
                    li: cache.conv_states[li].clone()
                    for li in linear_layers if cache.conv_states[li] is not None
                },
            )

    for li in attn_layers:
        if cache.key_cache[li] is not None:
            store.kv_cache_keys[li] = cache.key_cache[li].clone()
            store.kv_cache_values[li] = cache.value_cache[li].clone()
    return store


@contextmanager
def disable_attention_layers(model):
    """Skip attention decoder layers entirely (pass-through)."""
    attn_layers = _get_attention_layers(model.config)
    saved = {}

    def _identity(hidden_states, *args, **kwargs):
        return hidden_states

    for li in attn_layers:
        saved[li] = model.layers[li].forward
        model.layers[li].forward = _identity
    try:
        yield
    finally:
        for li, fn in saved.items():
            model.layers[li].forward = fn


def _time(n_runs, dev, fn, *args):
    _sync_device(dev)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(*args)
    _sync_device(dev)
    return (time.perf_counter() - t0) / n_runs


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    apply_patch()
    model = make_model(cfg)
    dev = _model_device(model)
    config = model.config

    bb = cfg.benchmark_baselines
    seq_lens = list(bb.seq_lens)
    B = bb.block_size
    n_runs = bb.n_runs

    keys = ['no_cache', 'attn_only', 'block', 'log', 'block_and_attn', 'log_and_attn']
    results = {k: [] for k in keys}
    cache_sizes = {k: [] for k in keys}

    # Warmup: use smallest and largest seq_len to cover kernel JIT range
    print("Warming up...")
    for N in [seq_lens[0], seq_lens[-1]]:
        torch.manual_seed(bb.seed)
        ids = torch.randint(0, config.vocab_size, (1, N)).to(dev)
        prefill_baseline(model, ids)
        del ids; _free_gpu()
        ids = torch.randint(0, config.vocab_size, (1, N)).to(dev)
        with disable_attention_layers(model):
            prefill_baseline(model, ids)
        del ids; _free_gpu()
    _sync_device(dev)
    print("Warmup done.\n")

    for N in seq_lens:
        torch.manual_seed(bb.seed)
        _free_gpu()

        # Per-N warmup
        input_ids = torch.randint(0, config.vocab_size, (1, N))
        prefill_baseline(model, input_ids)
        _free_gpu()

        # Capture checkpoints — offload each to CPU immediately
        input_ids = torch.randint(0, config.vocab_size, (1, N))
        log_store = prefill_and_capture_at(model, input_ids, _checkpoint_positions(N))
        log_store.to("cpu"); _free_gpu()
        block_store = prefill_and_capture_at(model, input_ids, _block_positions(N, B))
        block_store.to("cpu"); _free_gpu()

        # Cache sizes (bytes) — computed on CPU tensors, same values
        kv_bytes = sum(t.nelement() * t.element_size() for s in [log_store]
                       for t in list(s.kv_cache_keys.values()) + list(s.kv_cache_values.values()))
        def _gdn_bytes(store):
            return sum(t.nelement() * t.element_size()
                       for ckpt in store.checkpoints.values()
                       for t in list(ckpt.recurrent_states.values()) + list(ckpt.conv_states.values()))
        log_gdn_bytes = _gdn_bytes(log_store)
        block_gdn_bytes = _gdn_bytes(block_store)
        log_ckpt = log_store.best_checkpoint(N)
        block_ckpt = block_store.best_checkpoint(N)

        # 1. No cache
        t_no_cache = _time(n_runs, dev, prefill_baseline, model, input_ids)
        _free_gpu()

        # 2. Attention-only KV cache = skip attention layers, keep GDN+FFN
        with disable_attention_layers(model):
            t_attn_only = _time(n_runs, dev, prefill_baseline, model, input_ids)
        _free_gpu()

        t_attn_cost = max(t_no_cache - t_attn_only, 0)

        # 3. Block hybrid + attention: resume from block boundary, full pipeline
        block_store.to(dev)
        t_block_and_attn = _time(n_runs, dev, prefill_from_checkpoint, model, input_ids, block_store)
        _free_gpu()

        # 5. Block (no attn) = GDN-only for remaining tokens + attention cost for all N
        with disable_attention_layers(model):
            t_block_gdn = _time(n_runs, dev, prefill_from_checkpoint, model, input_ids, block_store)
        t_block = t_block_gdn + t_attn_cost
        block_store.to("cpu"); _free_gpu()

        # 4. Logarithmic + attention: resume from 2^i, full pipeline
        log_store.to(dev)
        t_log_and_attn = _time(n_runs, dev, prefill_from_checkpoint, model, input_ids, log_store)
        _free_gpu()

        # 6. Log (no attn) = GDN-only for remaining tokens + attention cost for all N
        with disable_attention_layers(model):
            t_log_gdn = _time(n_runs, dev, prefill_from_checkpoint, model, input_ids, log_store)
        t_log = t_log_gdn + t_attn_cost
        log_store.to("cpu"); _free_gpu()

        for k, v in zip(keys, [t_no_cache, t_attn_only, t_block, t_log, t_block_and_attn, t_log_and_attn]):
            results[k].append(v)

        cache_sizes['no_cache'].append(0)
        cache_sizes['attn_only'].append(kv_bytes)
        cache_sizes['block'].append(block_gdn_bytes)
        cache_sizes['log'].append(log_gdn_bytes)
        cache_sizes['block_and_attn'].append(kv_bytes + block_gdn_bytes)
        cache_sizes['log_and_attn'].append(kv_bytes + log_gdn_bytes)

        print(
            f"N={N:5d} | "
            f"no_cache {t_no_cache*1000:7.1f}ms | "
            f"attn_only {t_attn_only*1000:7.1f}ms | "
            f"block(B={B}) {t_block*1000:7.1f}ms | "
            f"log {t_log*1000:7.1f}ms | "
            f"block+attn {t_block_and_attn*1000:7.1f}ms (skip {block_ckpt.position if block_ckpt else 0}) | "
            f"log+attn {t_log_and_attn*1000:7.1f}ms (skip {log_ckpt.position if log_ckpt else 0})"
        )

        del log_store, block_store; _free_gpu()

    # --- Theoretical FLOPs and cache sizes ---
    m = cfg.model
    d, d_ff = m.hidden_size, m.intermediate_size
    n_q, n_kv, d_a = m.num_attention_heads, m.num_key_value_heads, m.head_dim
    n_v, n_qk, d_h = m.linear_num_value_heads, m.linear_num_key_heads, m.linear_value_head_dim
    gdn_per_group = m.gdn_layers
    n_layers = m.gdn_layers + m.ga_layers
    bpe = 2  # bf16

    flop_ffn = n_layers * 3 * d * d_ff
    flop_ga_proj = 3 * d * n_q * d_a + 2 * d * n_kv * d_a
    flop_gdn_proj = gdn_per_group * (3 * d * n_v * d_h + 2 * d * n_qk * d_h + 2 * d * n_v)
    flop_gdn_rec = gdn_per_group * n_v * d_h ** 2
    flop_ga_quad_per_tok = n_q * d_a
    flop_ga_linear = flop_ffn / n_layers + flop_ga_proj
    flop_gdn_recompute = flop_ffn * gdn_per_group / n_layers + flop_gdn_proj + flop_gdn_rec

    N_arr = np.array(seq_lens, dtype=float)

    # Theoretical recomputation FLOPs per strategy
    theo = {}
    theo['no_cache'] = N_arr * (flop_ga_linear + flop_gdn_recompute) + N_arr**2 * flop_ga_quad_per_tok
    theo['attn_only'] = N_arr * flop_gdn_recompute
    saved_attn = N_arr**2 * flop_ga_quad_per_tok + N_arr * flop_ga_linear
    saved_block = (N_arr - N_arr % B) * flop_gdn_recompute
    saved_log = (2.0 ** np.floor(np.log2(N_arr))) * flop_gdn_recompute
    theo['block'] = theo['no_cache'] - saved_block
    theo['log'] = theo['no_cache'] - saved_log
    theo['block_and_attn'] = np.maximum(theo['no_cache'] - saved_attn - saved_block, 0)
    theo['log_and_attn'] = np.maximum(theo['no_cache'] - saved_attn - saved_log, 0)

    # Scale theoretical FLOPs to empirical ms (least-squares fit on no_cache line)
    emp_ms = np.array(results['no_cache']) * 1000
    theo_nc = theo['no_cache']
    flop_scale = np.dot(emp_ms, theo_nc) / np.dot(theo_nc, theo_nc)

    # Theoretical cache sizes (bytes)
    attn_kv_per_token = n_kv * d_a * 2 * bpe
    gdn_state_per_ckpt = gdn_per_group * n_v * d_h**2 * bpe
    conv_dim = 2 * n_qk * d_h + n_v * d_h
    conv_state_per_ckpt = gdn_per_group * conv_dim * m.linear_conv_kernel_dim * bpe
    per_ckpt = gdn_state_per_ckpt + conv_state_per_ckpt

    n_block_ckpts = np.floor(N_arr / B)
    n_log_ckpts = np.floor(np.log2(N_arr)) + 1

    theo_cache = {}
    theo_cache['no_cache'] = np.zeros_like(N_arr)
    theo_cache['attn_only'] = N_arr * attn_kv_per_token
    theo_cache['block'] = n_block_ckpts * per_ckpt
    theo_cache['log'] = n_log_ckpts * per_ckpt
    theo_cache['block_and_attn'] = theo_cache['attn_only'] + theo_cache['block']
    theo_cache['log_and_attn'] = theo_cache['attn_only'] + theo_cache['log']

    # --- Plot ---
    to_ms = lambda a: np.array(a) * 1000
    to_mb = lambda a: np.array(a) / 1024 / 1024

    line_specs = [
        ('no_cache',      'No caching',               'black',      'o', '-',  2),
        ('attn_only',     'Attention-only KV cache',   'tab:red',    's', '-',  2),
        ('block', f'Block hybrid B={B} (CUDA)','tab:blue',   '^', '-',  1.5),
        ('log',   'Logarithmic',               'tab:orange', 'v', '-',  2),
        ('block_and_attn',    'Attention + Block hybrid',   'tab:green',  'D', '-',  2),
        ('log_and_attn',      'Attention + Logarithmic',    'tab:purple', 'p', '-',  2),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: latency (solid empirical, dashed theoretical)
    for key, label, color, marker, ls, lw in line_specs:
        ax1.plot(N_arr, to_ms(results[key]), marker=marker, ls=ls, label=label,
                 color=color, lw=lw, markersize=5)
        ax1.plot(N_arr, theo[key] * flop_scale, ls='--', color=color, lw=1, alpha=0.6)

    # Add one dashed legend entry
    ax1.plot([], [], ls='--', color='gray', lw=1, alpha=0.6, label='Theoretical (scaled FLOPs)')
    ax1.set_xlabel("Cached prefix length")
    ax1.set_ylabel("Prefill latency (ms)")
    ax1.set_title(f"Empirical vs theoretical latency — single {cfg.model.name} layer group")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(seq_lens) * 1.05)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Right: cache size (solid empirical, dashed theoretical)
    for key, label, color, marker, ls, lw in line_specs:
        if key == 'no_cache':
            continue  # zero line, skip
        ax2.plot(N_arr, to_mb(cache_sizes[key]), marker=marker, ls=ls, label=label,
                 color=color, lw=lw, markersize=5)
        ax2.plot(N_arr, to_mb(theo_cache[key]), ls='--', color=color, lw=1, alpha=0.6)

    ax2.plot([], [], ls='--', color='gray', lw=1, alpha=0.6, label='Theoretical')
    ax2.set_xlabel("Cached sequence length")
    ax2.set_ylabel("Cache size (MB, single layer group)")
    ax2.set_yscale("log", base=2)
    ax2.set_title(f"Empirical vs theoretical cache size — single {cfg.model.name} layer group")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    out = bb.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot to {out}")

if __name__ == "__main__":
    main()
