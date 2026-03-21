"""Generate plots from saved benchmark results.

Usage (hydra):
  python plot_results.py output_dir=path/to/run
  python plot_results.py output_dir=path/to/run plot._target_=plot_results.plot_overlap

The plot._target_ selects which plot function to call.
Available targets: plot_results.plot_all (default), plot_results.plot_single,
                   plot_results.plot_e2e, plot_results.plot_overlap, plot_results.plot_tradeoff
"""
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _build_style_map(strategies_cfg):
    """Build {tag: (label, color, marker, ls, lw)} from hydra strategies config."""
    out = {}
    for s in strategies_cfg:
        out[s.tag] = (
            s.label,
            s.color,
            s.marker,
            s.get("linestyle", "-"),
            s.get("linewidth", 2),
        )
    return out


def _spec(key, style_map):
    if key in style_map:
        return style_map[key]
    return key, 'gray', 'x', '-', 1


# ---------------------------------------------------------------------------
# Baselines plot
# ---------------------------------------------------------------------------
def plot_single(out_dir, root_dir=None, style_map=None, **_kw):
    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    path = root_dir / "benchmark_single" / "baselines_results.json"
    if not path.exists():
        print(f"  skipping baselines plot ({path} not found)")
        return

    data = json.loads(path.read_text())
    seq_lens = data["seq_lens"]
    strategies = data["strategies"]
    m = data["model_params"]
    model_name = data["model_name"]

    N_arr = np.array(seq_lens, dtype=float)

    # --- Theoretical FLOPs ---
    d, d_ff = m["hidden_size"], m["intermediate_size"]
    n_q, n_kv, d_a = m["num_attention_heads"], m["num_key_value_heads"], m["head_dim"]
    n_v, n_qk, d_h = m["linear_num_value_heads"], m["linear_num_key_heads"], m["linear_value_head_dim"]
    gdn_per_group = m["gdn_layers"]
    n_layers = m["gdn_layers"] + m["ga_layers"]
    bpe = 2

    flop_ffn = n_layers * 3 * d * d_ff
    flop_ga_proj = 3 * d * n_q * d_a + 2 * d * n_kv * d_a
    flop_gdn_proj = gdn_per_group * (3 * d * n_v * d_h + 2 * d * n_qk * d_h + 2 * d * n_v)
    flop_gdn_rec = gdn_per_group * n_v * d_h ** 2
    flop_ga_quad_per_tok = n_q * d_a
    flop_ga_linear = flop_ffn / n_layers + flop_ga_proj
    flop_gdn_recompute = flop_ffn * gdn_per_group / n_layers + flop_gdn_proj + flop_gdn_rec

    # All checkpoint strategies assume KV cache is always stored alongside GDN
    # checkpoints, since without KV the GA layer needs hidden states from GDN
    # FFNs and we cannot skip any GDN compute for the prefix.
    flop_per_tok = flop_ga_linear + flop_gdn_recompute

    # Theoretical FLOPs per strategy — keyed by tag
    styles = data.get("strategy_styles", [])
    styles_by_tag = {s["tag"]: s for s in styles}
    theo = {}
    for tag in strategies:
        s_cfg = styles_by_tag.get(tag, {})
        stype = s_cfg.get("type", tag)  # fallback for old data without type
        if stype == "no_cache":
            theo[tag] = N_arr * flop_per_tok + N_arr**2 * flop_ga_quad_per_tok
        elif stype == "kv_only":
            theo[tag] = N_arr * flop_gdn_recompute
        elif s_cfg.get("block_size"):
            B = s_cfg["block_size"]
            n_tail = N_arr % B
            theo[tag] = n_tail * flop_per_tok + n_tail * N_arr * flop_ga_quad_per_tok
        elif stype in ("diadic", "dyadic", "log"):
            n_tail = N_arr - 2.0 ** np.floor(np.log2(N_arr))
            theo[tag] = n_tail * flop_per_tok + n_tail * N_arr * flop_ga_quad_per_tok
        elif stype == "sqrt":
            n_tail = N_arr % np.floor(np.sqrt(N_arr)).astype(int)
            theo[tag] = n_tail * flop_per_tok + n_tail * N_arr * flop_ga_quad_per_tok

    # Scale theoretical to empirical
    if 'no_cache' in strategies:
        emp_ms = np.array(strategies['no_cache']['times_s']) * 1000
        theo_nc = theo['no_cache']
        flop_scale = np.dot(emp_ms, theo_nc) / np.dot(theo_nc, theo_nc)
    else:
        flop_scale = 0

    # Theoretical cache sizes
    attn_kv_per_token = n_kv * d_a * 2 * bpe
    gdn_state_per_ckpt = gdn_per_group * n_v * d_h**2 * bpe
    conv_dim = 2 * n_qk * d_h + n_v * d_h
    conv_state_per_ckpt = gdn_per_group * conv_dim * m["linear_conv_kernel_dim"] * bpe
    per_ckpt = gdn_state_per_ckpt + conv_state_per_ckpt

    theo_cache = {}
    for tag in strategies:
        s_cfg = styles_by_tag.get(tag, {})
        stype = s_cfg.get("type", tag)
        if stype == "no_cache":
            theo_cache[tag] = np.zeros_like(N_arr)
        elif stype == "kv_only":
            theo_cache[tag] = N_arr * attn_kv_per_token
        elif s_cfg.get("block_size"):
            B = s_cfg["block_size"]
            n_ckpts = np.floor(N_arr / B)
            theo_cache[tag] = N_arr * attn_kv_per_token + n_ckpts * per_ckpt
        elif stype in ("diadic", "dyadic", "log"):
            n_ckpts = np.floor(np.log2(N_arr)) + 1
            theo_cache[tag] = N_arr * attn_kv_per_token + n_ckpts * per_ckpt
        elif stype == "sqrt":
            n_ckpts = np.floor(np.sqrt(N_arr))
            theo_cache[tag] = N_arr * attn_kv_per_token + n_ckpts * per_ckpt

    # --- Plot ---
    to_ms = lambda a: np.array(a) * 1000
    to_mb = lambda a: np.array(a) / 1024 / 1024

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for key in strategies:
        label, color, marker, ls, lw = _spec(key, style_map)
        ax1.plot(N_arr, to_ms(strategies[key]['times_s']), marker=marker, ls=ls,
                 label=label, color=color, lw=lw, markersize=5)
        if key in theo and flop_scale:
            ax1.plot(N_arr, theo[key] * flop_scale, ls='--', color=color, lw=1, alpha=0.6)

    if flop_scale:
        ax1.plot([], [], ls='--', color='gray', lw=1, alpha=0.6, label='Theoretical (scaled FLOPs)')
    ax1.set_xlabel("Cached prefix length")
    ax1.set_ylabel("Prefill latency (ms)")
    ax1.set_title(f"Empirical vs theoretical latency — single {model_name} layer group")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(seq_lens) * 1.05)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for key in strategies:
        if key == 'no_cache':
            continue
        label, color, marker, ls, lw = _spec(key, style_map)
        ax2.plot(N_arr, to_mb(strategies[key]['cache_bytes']), marker=marker, ls=ls,
                 label=label, color=color, lw=lw, markersize=5)
        if key in theo_cache:
            ax2.plot(N_arr, to_mb(theo_cache[key]), ls='--', color=color, lw=1, alpha=0.6)

    ax2.plot([], [], ls='--', color='gray', lw=1, alpha=0.6, label='Theoretical')
    ax2.set_xlabel("Cached sequence length")
    ax2.set_ylabel("Cache size (MB, single layer group)")
    ax2.set_yscale("log", base=2)
    ax2.set_title(f"Empirical vs theoretical cache size — single {model_name} layer group")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    out = out_dir / "benchmark_single.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# E2E plots
# ---------------------------------------------------------------------------
def _load_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def plot_e2e(out_dir, root_dir=None, style_map=None, **_kw):
    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    data_dir = root_dir / "benchmark_e2e"
    summary_path = data_dir / "e2e_summary.json"
    if not summary_path.exists():
        print(f"  skipping e2e plots ({summary_path} not found)")
        return

    summary = json.loads(summary_path.read_text())
    model_name = summary["model_name"]
    strat_keys = list(summary["strategies"].keys())

    # Load per-request data
    per_request = {}
    for strat in strat_keys:
        jsonl = data_dir / f"e2e_{strat}.jsonl"
        if jsonl.exists():
            per_request[strat] = _load_jsonl(jsonl)

    if not per_request:
        print("  no JSONL files found, skipping e2e plots")
        return

    report = [s for s in strat_keys if s in per_request]
    times_ms = {s: np.array([e["time_s"] for e in per_request[s]]) * 1000 for s in report}
    baseline_ms = times_ms.get("no_cache")

    # --- Boxplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    box_labels = {s: _spec(s, style_map)[0].replace(' ', '\n', 1) for s in report}

    bp1 = ax1.boxplot(
        [times_ms[s] for s in report],
        labels=[box_labels[s] for s in report],
        patch_artist=True, showfliers=True,
        flierprops=dict(markersize=2, alpha=0.5),
    )
    for patch, s in zip(bp1["boxes"], report):
        patch.set_facecolor(_spec(s, style_map)[1])
        patch.set_alpha(0.6)
    ax1.set_ylabel("Per-request prefill time (ms)")
    ax1.set_title(f"Time distribution — {model_name}")
    ax1.grid(True, alpha=0.3, axis="y")

    if baseline_ms is not None:
        speedup_strats = [s for s in report if s != "no_cache"]
        speedups = {s: baseline_ms / np.maximum(times_ms[s], 1e-6) for s in speedup_strats}

        bp2 = ax2.boxplot(
            [speedups[s] for s in speedup_strats],
            labels=[box_labels[s] for s in speedup_strats],
            patch_artist=True, showfliers=True,
            flierprops=dict(markersize=2, alpha=0.5),
        )
        for patch, s in zip(bp2["boxes"], speedup_strats):
            patch.set_facecolor(_spec(s, style_map)[1])
            patch.set_alpha(0.6)
        ax2.axhline(y=1.0, color="black", ls="--", lw=1, alpha=0.5)
        ax2.set_ylabel("Speedup vs no-cache")
        ax2.set_title(f"Speedup distribution — {model_name}")
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    boxplots_path = out_dir / "e2e_boxplots.png"
    plt.savefig(boxplots_path, dpi=200, bbox_inches="tight")
    print(f"  saved {boxplots_path}")
    plt.close()

    # --- Time vs context length ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

    for s in report:
        label, color, *_ = _spec(s, style_map)
        seq_lens = np.array([e["seq_len"] for e in per_request[s]])
        ax3.scatter(seq_lens, times_ms[s], s=6, alpha=0.3, color=color, label=label)

    ax3.set_xlabel("Context length (tokens)")
    ax3.set_ylabel("Prefill time (ms)")
    ax3.set_yscale("log")
    ax3.set_title(f"Time vs context length — {model_name}")
    ax3.legend(fontsize=7, loc="upper left")
    ax3.grid(True, alpha=0.3)

    if baseline_ms is not None:
        for s in speedup_strats:
            label, color, *_ = _spec(s, style_map)
            seq_lens = np.array([e["seq_len"] for e in per_request[s]])
            ax4.scatter(seq_lens, speedups[s], s=6, alpha=0.3, color=color, label=label)

        ax4.axhline(y=1.0, color="black", ls="--", lw=1, alpha=0.5)
        ax4.set_xlabel("Context length (tokens)")
        ax4.set_ylabel("Speedup vs no-cache")
        ax4.set_yscale("log")
        ax4.set_title(f"Speedup vs context length — {model_name}")
        ax4.legend(fontsize=7, loc="upper left")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    scatter_path = out_dir / "e2e_time_vs_length.png"
    plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
    print(f"  saved {scatter_path}")
    plt.close()

    # Print summary table
    if "no_cache" in summary["strategies"]:
        t_base = summary["strategies"]["no_cache"]["total_time"]
    else:
        t_base = None
    n_req = summary.get("n_test_requests", summary.get("n_requests", 1))

    print(f"\n  {'Strategy':<30} {'Time (s)':>10} {'Speedup':>8} {'Hit rate':>10} {'GDN saved':>10}")
    print(f"  {'-'*70}")
    for strat in report:
        s = summary["strategies"].get(strat, {})
        label = _spec(strat, style_map)[0]
        t = s.get("total_time", 0)
        speedup = t_base / t if t_base and t > 0 else 0
        has_cache = strat != "no_cache"
        hit_rate = s.get("hits", 0) / n_req * 100 if has_cache else 0
        tok_total = s.get("tokens_total", 1)
        tok_saved = s.get("tokens_saved", 0) / tok_total * 100 if has_cache and tok_total > 0 else 0
        print(f"  {label:<30} {t:>10.1f} {speedup:>7.2f}x {hit_rate:>9.1f}% {tok_saved:>9.1f}%")


# ---------------------------------------------------------------------------
# Overlap distribution plot
# ---------------------------------------------------------------------------
def plot_overlap(out_dir, root_dir=None, **_kw):
    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    path = root_dir / "prepare_data" / "overlap_lcp.json"
    if not path.exists():
        log.info("skipping overlap plot (%s not found)", path)
        return

    data = json.loads(path.read_text())
    lcp = np.array(data["lcp_lengths"])
    n_total = data.get("n_requests", data.get("n_test_requests", len(lcp)))
    n_conversations = data["n_conversations"]
    lcp_pos = lcp[lcp > 0]

    if len(lcp_pos) == 0:
        log.warning("no cache hits, skipping overlap plot")
        return

    from fitter import Fitter

    distributions = ["gamma", "lognorm", "beta", "expon", "weibull_min", "norm"]
    f = Fitter(lcp_pos, distributions=distributions)
    f.fit()

    plt.figure(figsize=(10, 6))
    f.summary()
    plt.title(f"Distribution of Longest Common Prefixes ({n_conversations} conversations, {n_total} requests)")
    plt.xlabel("Prefix Match Length (Tokens)")
    plt.ylabel("Density")

    out = out_dir / "overlap_distribution.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    log.info("saved %s", out)
    plt.close()

    best_dist = f.get_best(method="sumsquare_error")
    best_name = list(best_dist.keys())[0]
    best_params = list(best_dist.values())[0]
    n_hits = len(lcp_pos)
    log.info("Best fit: %s, params: %s", best_name, best_params)
    log.info("Hit rate: %d/%d (%.1f%%)", n_hits, n_total, n_hits / n_total * 100)
    log.info("Mean LCP (hits only): %.1f tokens", lcp_pos.mean())
    log.info("Median LCP (hits only): %.1f tokens", np.median(lcp_pos))


# ---------------------------------------------------------------------------
# Speedup vs cache size trade-off
# ---------------------------------------------------------------------------
def plot_tradeoff(out_dir, root_dir=None, style_map=None, **_kw):
    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    path = root_dir / "benchmark_single" / "baselines_results.json"
    if not path.exists():
        print(f"  skipping tradeoff plot ({path} not found)")
        return

    data = json.loads(path.read_text())
    seq_lens = np.array(data["seq_lens"], dtype=float)
    strategies = data["strategies"]
    model_name = data["model_name"]

    if "no_cache" not in strategies:
        print("  skipping tradeoff plot (no_cache baseline missing)")
        return

    baseline_times = np.array(strategies["no_cache"]["times_s"])
    to_mb = lambda a: np.array(a) / 1024 / 1024

    fig, ax = plt.subplots(figsize=(10, 7))

    for key in strategies:
        if key == "no_cache":
            continue
        label, color, marker, ls, lw = _spec(key, style_map)
        times = np.array(strategies[key]["times_s"])
        speedup = np.mean(baseline_times / np.maximum(times, 1e-12))
        cache_mb = np.mean(to_mb(strategies[key]["cache_bytes"]))
        ax.scatter(cache_mb, speedup, s=120, color=color, marker=marker,
                   zorder=3, edgecolors="black", linewidths=0.5)
        ax.annotate(label, (cache_mb, speedup), textcoords="offset points",
                    xytext=(8, 4), fontsize=8)

    ax.set_xlabel("Mean cache size (MB, single layer group)")
    ax.set_ylabel("Mean speedup vs no-cache")
    ax.set_title(f"Speedup vs cache size trade-off — {model_name}")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    plt.tight_layout()
    out = out_dir / "tradeoff.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Composite targets
# ---------------------------------------------------------------------------
def plot_all(out_dir, root_dir=None, style_map=None, **_kw):
    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    plot_single(out_dir, root_dir=root_dir, style_map=style_map)
    plot_tradeoff(out_dir, root_dir=root_dir, style_map=style_map)
    plot_e2e(out_dir, root_dir=root_dir, style_map=style_map)
    plot_overlap(out_dir, root_dir=root_dir)


# ---------------------------------------------------------------------------
# Hydra entry point — dispatches via plot._target_
# ---------------------------------------------------------------------------
@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    from spase_cache.utils import setup_output_dir
    root_dir = Path(cfg.output_dir)
    out_dir = setup_output_dir(cfg, "plot_results")
    style_map = _build_style_map(cfg.strategies)

    print(f"Plotting results from {root_dir} into {out_dir}")
    hydra.utils.call(cfg.plot_results, out_dir=out_dir, root_dir=root_dir, style_map=style_map)
    print("Done.")


if __name__ == "__main__":
    main()
