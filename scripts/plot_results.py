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
# Trie diagnostics — 6-panel figure
# ---------------------------------------------------------------------------
def plot_trie_diagnostics(out_dir, root_dir=None, style_map=None, **_kw):
    """Diagnostic plots for understanding prefix trie structure and cache behavior."""
    from collections import defaultdict

    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    data_dir = root_dir / "benchmark_e2e"
    summary_path = data_dir / "e2e_summary.json"
    if not summary_path.exists():
        print(f"  skipping trie diagnostics ({summary_path} not found)")
        return

    summary = json.loads(summary_path.read_text())
    model_name = summary["model_name"]
    strat_keys = list(summary["strategies"].keys())

    per_request = {}
    for strat in strat_keys:
        jsonl = data_dir / f"e2e_{strat}.jsonl"
        if jsonl.exists():
            per_request[strat] = _load_jsonl(jsonl)
    if not per_request:
        print("  no JSONL files found, skipping trie diagnostics")
        return

    # Pick a reference strategy with cache for overlap distribution
    ref_strat = None
    for s in strat_keys:
        if s != "no_cache" and s in per_request:
            if "balanced" in s or "block" in s:
                ref_strat = s
                break
    if ref_strat is None:
        ref_strat = next(s for s in strat_keys if s != "no_cache" and s in per_request)
    ref = per_request[ref_strat]

    fig = plt.figure(figsize=(21, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)])

    # ---- Panel (0,0): Overlap depth histogram ----
    ax = axes[0, 0]
    overlaps = np.array([d["reusable_kv"] for d in ref])
    hit_overlaps = overlaps[overlaps > 0]
    if len(hit_overlaps):
        ax.hist(hit_overlaps, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(np.median(hit_overlaps), color="red", ls="--", lw=1.5,
                   label=f"median = {np.median(hit_overlaps):.0f}")
        ax.axvline(np.mean(hit_overlaps), color="orange", ls="--", lw=1.5,
                   label=f"mean = {np.mean(hit_overlaps):.0f}")
        ax.legend(fontsize=8)
    miss_frac = (overlaps == 0).sum() / len(overlaps)
    ax.set_xlabel("Overlap depth (tokens)")
    ax.set_ylabel("Count")
    ax.set_title(f"Overlap distribution (hits only, {miss_frac:.0%} misses hidden)")

    # ---- Panel (0,1): Overlap depth vs seq_len scatter ----
    ax = axes[0, 1]
    seq_lens = np.array([d["seq_len"] for d in ref])
    hits_mask = np.array([d["hit"] for d in ref])
    max_val = max(seq_lens.max(), overlaps.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, lw=1, label="overlap = seq_len")
    ax.scatter(seq_lens[~hits_mask], overlaps[~hits_mask], s=4, alpha=0.2,
               color="gray", label=f"miss ({(~hits_mask).sum()})", rasterized=True)
    ax.scatter(seq_lens[hits_mask], overlaps[hits_mask], s=4, alpha=0.3,
               color="steelblue", label=f"hit ({hits_mask.sum()})", rasterized=True)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Overlap depth (tokens)")
    ax.set_title("Overlap depth vs sequence length")
    ax.legend(fontsize=8)

    # ---- Panel (0,2): Token savings CDF ----
    ax = axes[0, 2]
    report = [s for s in strat_keys if s in per_request and s != "no_cache"]
    for s in report:
        label, color, marker, ls, lw = _spec(s, style_map)
        saved = np.sort([d["reusable_gdn"] for d in per_request[s]])
        cdf = np.arange(1, len(saved) + 1) / len(saved)
        ax.plot(saved, cdf, color=color, ls=ls, lw=max(lw, 1.5), label=label)
    ax.set_xlabel("GDN tokens reusable per request")
    ax.set_ylabel("CDF")
    ax.set_title("Token savings CDF by strategy")
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ---- Panel (1,0): Per-conversation prefix growth ----
    ax = axes[1, 0]
    conv_data = defaultdict(list)
    for d in ref:
        conv_data[d["conv_id"]].append((d["turn"], d["seq_len"]))

    max_turn = max(t for entries in conv_data.values() for t, _ in entries)
    lens_by_turn = [[] for _ in range(max_turn + 1)]
    for entries in conv_data.values():
        for t, s in entries:
            lens_by_turn[t].append(s)

    valid_turns = [t for t in range(max_turn + 1) if len(lens_by_turn[t]) >= 5]
    if valid_turns:
        p25 = [np.percentile(lens_by_turn[t], 25) for t in valid_turns]
        p50 = [np.percentile(lens_by_turn[t], 50) for t in valid_turns]
        p75 = [np.percentile(lens_by_turn[t], 75) for t in valid_turns]
        p90 = [np.percentile(lens_by_turn[t], 90) for t in valid_turns]
        counts = [len(lens_by_turn[t]) for t in valid_turns]

        ax.fill_between(valid_turns, p25, p75, alpha=0.3, color="steelblue", label="25–75th pct")
        ax.plot(valid_turns, p50, color="steelblue", lw=2, label="median")
        ax.plot(valid_turns, p90, color="steelblue", ls="--", lw=1, alpha=0.7, label="90th pct")

        ax2_twin = ax.twinx()
        ax2_twin.bar(valid_turns, counts, alpha=0.12, color="gray")
        ax2_twin.set_ylabel("# requests at turn", color="gray", fontsize=8)
        ax2_twin.tick_params(axis="y", labelcolor="gray", labelsize=7)

    ax.set_xlabel("Turn index")
    ax.set_ylabel("Context length (tokens)")
    ax.set_title("Context growth per conversation turn")
    ax.legend(fontsize=8, loc="upper left")

    # ---- Panel (1,1): Cache residency over time ----
    ax = axes[1, 1]
    has_cache_field = "n_cache_entries" in ref[0]
    req_idx = np.arange(len(ref))
    hits_arr = np.array([d["hit"] for d in ref])

    if has_cache_field:
        n_entries = [d["n_cache_entries"] for d in ref]
        ax.plot(req_idx, n_entries, color="steelblue", lw=1, label="cache entries")
        ax.set_ylabel("Cache entries")
        # secondary axis: cache memory
        if "cache_kv_bytes" in ref[0]:
            ax_mem = ax.twinx()
            kv_mb = np.array([d["cache_kv_bytes"] for d in ref]) / 1e9
            gdn_mb = np.array([d["cache_gdn_bytes"] for d in ref]) / 1e9
            ax_mem.plot(req_idx, kv_mb, color="tab:red", lw=0.8, alpha=0.6, label="KV (GB)")
            ax_mem.plot(req_idx, gdn_mb, color="tab:green", lw=0.8, alpha=0.6, label="GDN (GB)")
            ax_mem.set_ylabel("Cache memory (GB)", fontsize=8)
            ax_mem.legend(fontsize=7, loc="center right")
    else:
        cum_hits = np.cumsum(hits_arr)
        ax.plot(req_idx, cum_hits, color="steelblue", lw=1, label="cumulative hits")
        ax.set_ylabel("Cumulative hits")

    # Rug plot for hit/miss along bottom
    for idx in req_idx[hits_arr]:
        ax.axvline(idx, ymin=0, ymax=0.02, color="green", alpha=0.15, lw=0.3)
    for idx in req_idx[~hits_arr]:
        ax.axvline(idx, ymin=0, ymax=0.02, color="red", alpha=0.3, lw=0.3)

    ax.set_xlabel("Request index (green=hit, red=miss)")
    ax.set_title(f"Cache residency over time ({ref_strat})")
    ax.legend(fontsize=8, loc="upper left")

    # ---- Panel (1,2): Prefix sharing heatmap ----
    ax = axes[1, 2]
    sharing_path = root_dir / "prepare_data" / "prefix_sharing.json"
    if sharing_path.exists():
        sharing = json.loads(sharing_path.read_text())
        lcp_matrix = np.array(sharing["lcp_matrix"], dtype=np.float32)
        n_convs = len(lcp_matrix)

        # Cluster by similarity
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            max_lcp = lcp_matrix.max()
            dist = max_lcp - lcp_matrix
            np.fill_diagonal(dist, 0)
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method="average")
            order = leaves_list(Z)
            lcp_matrix = lcp_matrix[np.ix_(order, order)]
        except ImportError:
            pass  # plot unsorted if scipy unavailable

        im = ax.imshow(lcp_matrix, cmap="viridis", aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="LCP (tokens)", shrink=0.8)
        ax.set_title(f"Pairwise prefix sharing ({n_convs} convs, clustered)")
        ax.set_xlabel("Conversation")
        ax.set_ylabel("Conversation")
    else:
        ax.text(0.5, 0.5,
                "prefix_sharing.json not found\n\n"
                "Run prepare_data with\ncompute_prefix_sharing target",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
        ax.set_title("Prefix sharing heatmap")

    # ---- Panel (2,0): Cross-turn hit analysis ----
    ax = axes[2, 0]
    has_turn_gap = "turn_gap" in ref[0]
    if has_turn_gap:
        gaps = np.array([d["turn_gap"] for d in ref if d["hit"]])
        consecutive = (gaps == 1).sum()
        non_consec = (gaps > 1).sum()
        max_gap = gaps.max() if len(gaps) else 1
        bins = np.arange(0.5, max_gap + 1.5, 1)
        ax.hist(gaps, bins=bins, color="steelblue", alpha=0.7, edgecolor="white")
        ax.set_xlabel("Turn gap (1 = consecutive)")
        ax.set_ylabel("Count")
        ax.set_title(f"Hit turn gaps — consecutive: {consecutive}, skip-turn: {non_consec}")
        # Annotate fractions
        n_total = len(ref)
        ax.text(0.97, 0.95,
                f"consecutive: {consecutive/n_total:.1%} of all\n"
                f"skip-turn: {non_consec/n_total:.1%} of all",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    else:
        ax.text(0.5, 0.5, "turn_gap not in data\n(re-run benchmark_e2e)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title("Hit turn gaps")

    # ---- Panel (2,1): Non-consecutive hit overlap distribution ----
    ax = axes[2, 1]
    if has_turn_gap:
        nc_data = [d for d in ref if d["hit"] and d["turn_gap"] > 1]
        consec_data = [d for d in ref if d["hit"] and d["turn_gap"] == 1]
        if consec_data:
            ax.hist([d["reusable_kv"] for d in consec_data], bins=40,
                    alpha=0.5, color="steelblue", edgecolor="white", label=f"consecutive ({len(consec_data)})")
        if nc_data:
            ax.hist([d["reusable_kv"] for d in nc_data], bins=40,
                    alpha=0.7, color="tab:orange", edgecolor="white", label=f"skip-turn ({len(nc_data)})")
            nc_kv = [d["reusable_kv"] for d in nc_data]
            ax.axvline(np.median(nc_kv), color="tab:orange", ls="--", lw=1.5)
        ax.set_xlabel("Overlap depth (tokens)")
        ax.set_ylabel("Count")
        ax.set_title("Overlap by hit type")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "turn_gap not in data",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title("Overlap by hit type")

    # ---- Panel (2,2): unused ----
    axes[2, 2].set_visible(False)

    fig.suptitle(f"Prefix trie diagnostics — {model_name}", fontsize=14, y=0.99)
    out = out_dir / "trie_diagnostics.png"
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
    plot_trie_diagnostics(out_dir, root_dir=root_dir, style_map=style_map)


# ---------------------------------------------------------------------------
# Hydra entry point — dispatches via plot._target_
# ---------------------------------------------------------------------------
@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    from spase_cache.utils import setup_output_dir, resolve_strategies
    root_dir = Path(cfg.output_dir)
    out_dir = setup_output_dir(cfg, "plot_results")
    resolve_strategies(cfg)
    style_map = _build_style_map(cfg.strategies)

    print(f"Plotting results from {root_dir} into {out_dir}")
    hydra.utils.call(cfg.plot_results, out_dir=out_dir, root_dir=root_dir, style_map=style_map)
    print("Done.")


if __name__ == "__main__":
    main()
