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
    """Build {tag: (label, color, marker, ls, lw, family_label)} from hydra strategies config."""
    out = {}
    for s in strategies_cfg:
        out[s.tag] = (
            s.label,
            s.color,
            s.marker,
            s.linestyle,
            s.linewidth,
            s.family_label,
        )
    return out


def _spec(key, style_map):
    """Return (label, color, marker, ls, lw) for a strategy tag."""
    if key in style_map:
        return style_map[key][:5]
    return key, 'gray', 'x', '-', 1


def _family_label(key, style_map):
    """Return family_label for a strategy tag (for legend grouping)."""
    if key in style_map and len(style_map[key]) > 5:
        return style_map[key][5]
    return _spec(key, style_map)[0]


def _build_style_index(summary):
    """Build {tag: style_dict} from strategy_styles in summary JSON."""
    return {s["tag"]: s for s in summary["strategy_styles"]}


def _sort_strategies_by_budget(strat_keys, style_index=None):
    """Sort strategies so that same-budget strategies are adjacent.

    Uses n_blocks from strategy_styles config when available.
    """
    FAMILY_ORDER = {
        "balanced_fix_nblocks": 0, "histogram_frozen": 1, "histogram_exp_decay": 2,
        "block": 3, "log": 4, "kv_only": -1, "no_cache": -2,
    }

    def _sort_key(s):
        cfg = style_index[s]
        stype = cfg["type"]
        family = FAMILY_ORDER.get(stype, 5)
        n_blocks = cfg.get("n_blocks", 0)
        block_size = cfg.get("block_size", 0)

        if family <= 2:
            return (n_blocks, family, s)
        elif family < 0:
            return (family, 0, s)
        else:
            return (1000 + block_size, family, s)

    return sorted(strat_keys, key=_sort_key)


def _group_strategies_by_budget(strat_keys, style_index=None):
    """Group strategies into (budget_label, [strats]) for grid layout.

    Reads n_blocks from strategy config. Strategies sharing the same n_blocks
    are placed in the same row so they can be compared side by side.
    """
    from collections import OrderedDict

    BUDGETED_TYPES = {"balanced_fix_nblocks", "histogram_frozen", "histogram_exp_decay"}
    FAMILY_ORDER = {"balanced_fix_nblocks": 0, "histogram_frozen": 1, "histogram_exp_decay": 2}

    budgeted = OrderedDict()
    others = []

    for s in strat_keys:
        cfg = style_index[s]
        stype = cfg["type"]
        if stype in BUDGETED_TYPES:
            budgeted.setdefault(cfg["n_blocks"], []).append(s)
        else:
            others.append(s)

    groups = []
    for budget, strats in sorted(budgeted.items()):
        strats.sort(key=lambda s: FAMILY_ORDER[style_index[s]["type"]] if style_index[s]["type"] in FAMILY_ORDER else 9)
        groups.append((f"{budget} blocks", strats))

    for s in others:
        groups.append((s, [s]))

    return groups


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
    styles_by_tag = {s["tag"]: s for s in data["strategy_styles"]}
    theo = {}
    for tag in strategies:
        s_cfg = styles_by_tag[tag]
        stype = s_cfg["type"]
        if stype == "no_cache":
            theo[tag] = N_arr * flop_per_tok + N_arr**2 * flop_ga_quad_per_tok
        elif stype == "kv_only":
            theo[tag] = N_arr * flop_gdn_recompute
        elif stype == "balanced_fix_blocksize":
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
        s_cfg = styles_by_tag[tag]
        stype = s_cfg["type"]
        if stype == "no_cache":
            theo_cache[tag] = np.zeros_like(N_arr)
        elif stype == "kv_only":
            theo_cache[tag] = N_arr * attn_kv_per_token
        elif stype == "balanced_fix_blocksize":
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

    sidx = _build_style_index(summary)
    report = _sort_strategies_by_budget([s for s in strat_keys if s in per_request], sidx)
    # keep no_cache first
    if "no_cache" in report:
        report.remove("no_cache")
        report = ["no_cache"] + report
    times_ms = {s: np.array([e["time_s"] for e in per_request[s]]) * 1000 for s in report}
    baseline_ms = times_ms["no_cache"]

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
    ax1.tick_params(axis="x", rotation=45, labelsize=7)

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
        ax2.tick_params(axis="x", rotation=45, labelsize=7)

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
    n_req = summary["n_test_requests"]

    print(f"\n  {'Strategy':<30} {'Time (s)':>10} {'Speedup':>8} {'Hit rate':>10} {'GDN saved':>10}")
    print(f"  {'-'*70}")
    for strat in report:
        s = summary["strategies"][strat]
        label = _spec(strat, style_map)[0]
        t = s["total_time"]
        speedup = t_base / t if t_base and t > 0 else 0
        has_cache = strat != "no_cache"
        hit_rate = s["hits"] / n_req * 100 if has_cache else 0
        tok_total = s["tokens_total"]
        tok_saved = s["tokens_saved"] / tok_total * 100 if has_cache and tok_total > 0 else 0
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
    n_total = data["n_requests"]
    n_conversations = data["n_conversations"]
    lcp_pos = lcp[lcp > 0]

    if len(lcp_pos) == 0:
        log.warning("no cache hits, skipping overlap plot")
        return

    from fitter import Fitter

    distributions = ["gamma", "lognorm", "beta", "expon", "weibull_min", "norm"]
    f = Fitter(lcp_pos, distributions=distributions, bins=20)
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
        times = np.array(strategies[key]["times_s"])
        if np.all(times == 0):
            continue  # strategy not supported in benchmark_single
        label, color, marker, ls, lw = _spec(key, style_map)
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
    has_turn = "turn" in ref[0]
    if has_turn:
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
    else:
        ax.text(0.5, 0.5, "turn data not available",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
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

        if n_convs < 2:
            ax.text(0.5, 0.5, f"Only {n_convs} conversation(s)\n(need ≥2 for heatmap)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title("Prefix sharing heatmap")
        else:
            # Cluster by similarity
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            max_lcp = lcp_matrix.max()
            dist = max_lcp - lcp_matrix
            np.fill_diagonal(dist, 0)
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method="average")
            order = leaves_list(Z)
            lcp_matrix = lcp_matrix[np.ix_(order, order)]

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

    # ---- Panel (2,2): Compressed prefix trie ----
    ax = axes[2, 2]
    trie_path = root_dir / "prepare_data" / "trie.json"
    if trie_path.exists():
        trie_data = json.loads(trie_path.read_text())
        nodes = trie_data["nodes"]
        edges = trie_data["edges"]

        if len(nodes) > 500:
            ax.text(0.5, 0.5, f"Trie too large to plot\n({len(nodes)} nodes)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title("Compressed prefix trie")
        else:
            import networkx as nx
            G = nx.DiGraph()
            for n in nodes:
                G.add_node(n["id"])
            edge_labels = {}
            for e in edges:
                G.add_edge(e["src"], e["dst"])
                edge_labels[(e["src"], e["dst"])] = str(e["length"])

            # hierarchical layout using depth
            depth_map = {n["id"]: n["depth"] for n in nodes}
            # group nodes by depth, spread horizontally
            from collections import defaultdict as _dd
            by_depth = _dd(list)
            for n in nodes:
                by_depth[n["depth"]].append(n["id"])

            pos = {}
            max_depth = max(depth_map.values()) if depth_map else 0
            for depth, nids in by_depth.items():
                for i, nid in enumerate(nids):
                    x = (i - (len(nids) - 1) / 2)
                    y = -depth  # root at top
                    pos[nid] = (x, y)

            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=15, node_color="steelblue", alpha=0.8)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", alpha=0.5,
                                   arrows=True, arrowsize=5, width=0.5)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                         font_size=5, font_color="tab:red", label_pos=0.5)
            ax.set_title(f"Compressed prefix trie ({len(nodes)} nodes, {len(edges)} edges)")
            ax.set_aspect("equal")
    else:
        ax.text(0.5, 0.5, "trie.json not found\n\nRun prepare_data",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title("Compressed prefix trie")

    fig.suptitle(f"Prefix trie diagnostics — {model_name}", fontsize=14, y=0.99)
    out = out_dir / "trie_diagnostics.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 1: True data LCP over rounds per conversation (strategy-independent)
# ---------------------------------------------------------------------------
def plot_lcp_over_rounds(out_dir, root_dir=None, style_map=None, **_kw):
    """Plot true prefix_match (data-level LCP) vs within-conversation round index.

    prefix_match is identical across all strategies — it depends only on the data.
    We use any single cached strategy as reference to extract the values.
    """
    from collections import defaultdict

    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    data_dir = root_dir / "benchmark_e2e"
    summary_path = data_dir / "e2e_summary.json"
    if not summary_path.exists():
        print(f"  skipping lcp_over_rounds ({summary_path} not found)")
        return

    summary = json.loads(summary_path.read_text())
    model_name = summary["model_name"]
    strat_keys = list(summary["strategies"].keys())

    # Pick any cached strategy — prefix_match is the same for all
    ref_strat = None
    for s in strat_keys:
        if s not in ("no_cache",):
            jsonl = data_dir / f"e2e_{s}.jsonl"
            if jsonl.exists():
                ref_strat = s
                break
    if ref_strat is None:
        return

    entries = _load_jsonl(data_dir / f"e2e_{ref_strat}.jsonl")

    # Group by conv_id, assign within-conversation round index
    conv_entries = defaultdict(list)
    for e in entries:
        conv_entries[e["conv_id"]].append(e)

    convs = list(conv_entries.keys())[:100]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: LCP vs within-conversation round
    cmap = plt.cm.tab20
    for ci, cid in enumerate(convs):
        es = conv_entries[cid]
        rounds = list(range(len(es)))
        lcp = [e["prefix_match"] for e in es]
        color = cmap(ci % 20)
        ax1.plot(rounds, lcp, color=color, alpha=0.4, lw=1.0, marker=".", markersize=3)

    ax1.set_xlabel("Round (within conversation)")
    ax1.set_ylabel("Prefix match length (tokens)")
    ax1.set_title(f"LCP over rounds ({len(convs)} conversations)")
    ax1.grid(True, alpha=0.3)

    # Right: LCP as fraction of seq_len
    for ci, cid in enumerate(convs):
        es = conv_entries[cid]
        rounds = list(range(len(es)))
        lcp_frac = [e["prefix_match"] / e["seq_len"] if e["seq_len"] > 0 else 0 for e in es]
        color = cmap(ci % 20)
        ax1_r = ax2
        ax1_r.plot(rounds, lcp_frac, color=color, alpha=0.4, lw=1.0, marker=".", markersize=3)

    ax2.set_xlabel("Round (within conversation)")
    ax2.set_ylabel("LCP / seq_len")
    ax2.set_title("LCP as fraction of sequence length")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"True data LCP per conversation round — {model_name}", fontsize=12)
    plt.tight_layout()
    out = out_dir / "lcp_over_rounds.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: Checkpoint positions for one conv across strategies
# ---------------------------------------------------------------------------
def plot_checkpoint_positions(out_dir, root_dir=None, style_map=None, **_kw):
    """For one conversation, show checkpoint positions per round for each strategy."""
    from collections import defaultdict

    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    data_dir = root_dir / "benchmark_e2e"
    summary_path = data_dir / "e2e_summary.json"
    if not summary_path.exists():
        print(f"  skipping checkpoint_positions ({summary_path} not found)")
        return

    summary = json.loads(summary_path.read_text())
    model_name = summary["model_name"]
    strat_keys = list(summary["strategies"].keys())

    # Load all strategies
    per_request = {}
    for strat in strat_keys:
        jsonl = data_dir / f"e2e_{strat}.jsonl"
        if jsonl.exists():
            per_request[strat] = _load_jsonl(jsonl)

    show = [s for s in strat_keys if s in per_request and s != "no_cache" and s != "kv_only"]
    if not show:
        return

    # Pick the conv_id with the most requests (from first strategy with cache)
    ref = per_request[show[0]]
    conv_counts = defaultdict(int)
    for e in ref:
        conv_counts[e["conv_id"]] += 1
    target_conv = max(conv_counts, key=conv_counts.get)

    sidx = _build_style_index(summary)
    groups = _group_strategies_by_budget(show, sidx)
    # Only keep groups whose strategies exist in per_request
    groups = [(lbl, [s for s in strats if s in per_request])
              for lbl, strats in groups]
    groups = [(lbl, strats) for lbl, strats in groups if strats]

    nrows = len(groups)
    ncols = max(len(strats) for _, strats in groups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for row, (group_label, strats) in enumerate(groups):
        for col in range(ncols):
            ax = axes[row][col]
            if col >= len(strats):
                ax.set_visible(False)
                continue

            strat = strats[col]
            entries = [e for e in per_request[strat] if e["conv_id"] == target_conv]
            label, color, *_ = _spec(strat, style_map)

            for round_idx, e in enumerate(entries):
                seq_len = e["seq_len"]
                positions = e["added_positions"]
                ax.barh(round_idx, seq_len, height=0.6, color=color, alpha=0.15)
                if positions:
                    ax.scatter(positions, [round_idx] * len(positions),
                               s=8, color=color, zorder=3, alpha=0.7)
                pm = e["prefix_match"]
                ax.plot([pm, pm], [round_idx - 0.3, round_idx + 0.3],
                        zorder=999, color="red", lw=1.2, alpha=1.0)

            ax.set_title(label, fontsize=8)
            ax.set_yticks(range(len(entries)))
            ax.set_yticklabels([str(i) for i in range(len(entries))], fontsize=6)
            ax.grid(True, alpha=0.2, axis="x")
            ax.invert_yaxis()

            if col == 0:
                ax.set_ylabel(group_label, fontsize=9)
            if row == nrows - 1:
                ax.set_xlabel("Token position")

    fig.suptitle(f"Checkpoint positions — conv: {target_conv}\n"
                 f"(dots = checkpoints, red line = prefix match) — {model_name}",
                 fontsize=10)
    plt.tight_layout()
    out = out_dir / "checkpoint_positions.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Why distribution-aware methods lose — GDN gap analysis
# ---------------------------------------------------------------------------
def plot_gdn_gap(out_dir, root_dir=None, style_map=None, **_kw):
    """Show how much GDN reuse each strategy wastes relative to KV prefix match.

    prefix_match is identical across strategies (determined by data).
    reusable_gdn depends on checkpoint placement.
    The gap = prefix_match - reusable_gdn = wasted GDN recompute.
    """
    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    data_dir = root_dir / "benchmark_e2e"
    summary_path = data_dir / "e2e_summary.json"
    if not summary_path.exists():
        print(f"  skipping gdn_gap ({summary_path} not found)")
        return

    summary = json.loads(summary_path.read_text())
    model_name = summary["model_name"]
    strat_keys = list(summary["strategies"].keys())

    per_request = {}
    for strat in strat_keys:
        jsonl = data_dir / f"e2e_{strat}.jsonl"
        if jsonl.exists():
            per_request[strat] = _load_jsonl(jsonl)

    sidx = _build_style_index(summary)
    show = _sort_strategies_by_budget(
        [s for s in strat_keys if s in per_request and s not in ("no_cache", "kv_only")], sidx)
    if not show:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Boxplot of GDN efficiency = reusable_gdn / prefix_match
    efficiencies = {}
    for s in show:
        eff = []
        for e in per_request[s]:
            pm = e["prefix_match"]
            if pm > 0:
                eff.append(e["reusable_gdn"] / pm)
        efficiencies[s] = eff

    labels_eff = [_spec(s, style_map)[0].replace(' ', '\n', 1) for s in show]
    colors_eff = [_spec(s, style_map)[1] for s in show]
    bp = ax1.boxplot(
        [efficiencies[s] for s in show],
        labels=labels_eff, patch_artist=True, showfliers=False,
        flierprops=dict(markersize=2, alpha=0.5),
    )
    for patch, c in zip(bp["boxes"], colors_eff):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax1.set_ylabel("GDN efficiency (reusable_gdn / prefix_match)")
    ax1.set_title("Checkpoint placement efficiency")
    ax1.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.4)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="x", rotation=45, labelsize=7)

    # Panel 2: Scatter — number of checkpoints vs GDN efficiency
    for s in show:
        label, color, marker, *_ = _spec(s, style_map)
        n_ckpts = []
        effs = []
        for e in per_request[s]:
            pm = e["prefix_match"]
            if pm > 0:
                n_ckpts.append(len(e["added_positions"]))
                effs.append(e["reusable_gdn"] / pm)
        ax2.scatter(n_ckpts, effs, s=10, alpha=0.3, color=color, label=label)
    ax2.set_xlabel("# checkpoints placed")
    ax2.set_ylabel("GDN efficiency")
    ax2.set_title("More checkpoints → better GDN coverage?")
    ax2.legend(fontsize=6, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.4)

    # Panel 3: Gap (prefix_match - reusable_gdn) as fraction of seq_len
    gap_fracs = {}
    for s in show:
        gaps = []
        for e in per_request[s]:
            sl = e["seq_len"]
            gap = e["prefix_match"] - e["reusable_gdn"]
            gaps.append(gap / sl if sl > 0 else 0)
        gap_fracs[s] = gaps

    bp3 = ax3.boxplot(
        [gap_fracs[s] for s in show],
        labels=labels_eff, patch_artist=True, showfliers=False,
    )
    for patch, c in zip(bp3["boxes"], colors_eff):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax3.set_ylabel("Wasted GDN fraction (gap / seq_len)")
    ax3.set_title("GDN recompute waste per request")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.tick_params(axis="x", rotation=45, labelsize=7)

    fig.suptitle(f"GDN gap analysis — why distribution-aware may underperform — {model_name}",
                 fontsize=11)
    plt.tight_layout()
    out = out_dir / "gdn_gap_analysis.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 4: Cache size breakdown — KV vs GDN over time per strategy
# ---------------------------------------------------------------------------
def plot_cache_breakdown(out_dir, root_dir=None, style_map=None, **_kw):
    """Plot cache_kv_bytes and cache_gdn_bytes over request index for each strategy."""
    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    data_dir = root_dir / "benchmark_e2e"
    summary_path = data_dir / "e2e_summary.json"
    if not summary_path.exists():
        print(f"  skipping cache_breakdown ({summary_path} not found)")
        return

    summary = json.loads(summary_path.read_text())
    model_name = summary["model_name"]
    strat_keys = list(summary["strategies"].keys())

    per_request = {}
    for strat in strat_keys:
        jsonl = data_dir / f"e2e_{strat}.jsonl"
        if jsonl.exists():
            per_request[strat] = _load_jsonl(jsonl)

    show = [s for s in strat_keys if s in per_request and s not in ("no_cache",)]
    if not show:
        return

    sidx = _build_style_index(summary)
    groups = _group_strategies_by_budget(show, sidx)
    groups = [(lbl, [s for s in strats if s in per_request])
              for lbl, strats in groups]
    groups = [(lbl, strats) for lbl, strats in groups if strats]

    nrows = len(groups)
    ncols = max(len(strats) for _, strats in groups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                             squeeze=False, sharey=True)

    first_ax = None
    for row, (group_label, strats) in enumerate(groups):
        for col in range(ncols):
            ax = axes[row][col]
            if col >= len(strats):
                ax.set_visible(False)
                continue

            strat = strats[col]
            entries = per_request[strat]
            label, color, *_ = _spec(strat, style_map)

            req_idx = np.arange(len(entries))
            kv_gb = np.array([e["cache_kv_bytes"] for e in entries]) / 1e9
            gdn_gb = np.array([e["cache_gdn_bytes"] for e in entries]) / 1e9
            n_entries = np.array([e["n_cache_entries"] for e in entries])

            ax.fill_between(req_idx, 0, kv_gb, alpha=0.4, color="tab:red", label="KV cache")
            ax.fill_between(req_idx, kv_gb, kv_gb + gdn_gb, alpha=0.4, color="tab:green", label="GDN ckpts")
            ax.set_title(label, fontsize=9)
            ax.grid(True, alpha=0.3)

            ax2t = ax.twinx()
            ax2t.plot(req_idx, n_entries, color="gray", lw=0.8, alpha=0.5, ls="--")
            ax2t.set_ylabel("# entries", fontsize=7, color="gray")
            ax2t.tick_params(axis="y", labelcolor="gray", labelsize=6)

            if col == 0:
                ax.set_ylabel(f"{group_label}\nGB", fontsize=8)
            if first_ax is None:
                ax.legend(fontsize=7, loc="upper left")
                first_ax = ax

    fig.suptitle(f"Cache memory breakdown (KV vs GDN) — {model_name}", fontsize=12)
    plt.tight_layout()
    out = out_dir / "cache_breakdown.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Histogram before/after Laplace smoothing
# ---------------------------------------------------------------------------
def plot_histograms(out_dir, root_dir=None, style_map=None, **_kw):
    """Plot raw vs Laplace-smoothed histograms from histogram strategy solves."""
    from sparse_prefix_caching.strategies import laplace_smoothing

    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    data_dir = root_dir / "benchmark_e2e"
    summary_path = data_dir / "e2e_summary.json"
    if not summary_path.exists():
        print(f"  skipping histograms plot ({summary_path} not found)")
        return

    summary = json.loads(summary_path.read_text())
    model_name = summary["model_name"]

    # Find all histogram log files
    hist_files = sorted(data_dir.glob("e2e_*_histograms.json"))
    if not hist_files:
        print("  no histogram log files found, skipping")
        return

    for hf in hist_files:
        tag = hf.stem.replace("e2e_", "").replace("_histograms", "")
        data = json.loads(hf.read_text())
        entries = data["histogram_log"]
        alpha = data["laplace_alpha"]
        bin_size = data["bin_size"]

        if not entries:
            continue

        n = len(entries)
        fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n), squeeze=False)

        label = _spec(tag, style_map)[0] if style_map else tag

        for i, entry in enumerate(entries):
            counts = np.array(entry["counts"])
            smoothed = laplace_smoothing(counts, alpha)
            n_obs = entry["n_obs"]
            x = np.arange(len(counts)) * bin_size

            # Trim to last nonzero + some margin
            last_nz = max(np.argwhere(counts > 0).max().item() if counts.max() > 0 else 0,
                          np.argwhere(smoothed > 0).max().item() if smoothed.max() > 0 else 0)
            trim = min(last_nz + 10, len(counts))

            ax_raw = axes[i, 0]
            ax_raw.bar(x[:trim], counts[:trim], width=bin_size * 0.8,
                       color="steelblue", alpha=0.7, edgecolor="white")
            ax_raw.set_title(f"Raw (solve #{i+1}, n_obs={n_obs})", fontsize=9)
            ax_raw.set_ylabel("Count")
            ax_raw.grid(True, alpha=0.3, axis="y")

            ax_sm = axes[i, 1]
            ax_sm.bar(x[:trim], smoothed[:trim], width=bin_size * 0.8,
                      color="tab:orange", alpha=0.7, edgecolor="white")
            ax_sm.set_title(f"Laplace smoothed (α={alpha})", fontsize=9)
            ax_sm.set_ylabel("Count")
            ax_sm.grid(True, alpha=0.3, axis="y")

            if i == n - 1:
                ax_raw.set_xlabel("Token position")
                ax_sm.set_xlabel("Token position")

        fig.suptitle(f"Histogram evolution — {label} — {model_name}", fontsize=12)
        plt.tight_layout()
        out = out_dir / f"histograms_{tag}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  saved {out}")
        plt.close()


# ---------------------------------------------------------------------------
# Scatter: tokens to process vs wallclock time
# ---------------------------------------------------------------------------

def plot_tokens_vs_time(out_dir, root_dir=None, style_map=None, **_kw):
    """Scatter plot: tokens to process (seq_len - tokens_saved) vs wallclock time."""
    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    data_dir = root_dir / "benchmark_e2e"
    summary_path = data_dir / "e2e_summary.json"
    if not summary_path.exists():
        print(f"  skipping tokens_vs_time plot ({summary_path} not found)")
        return

    summary = json.loads(summary_path.read_text())
    model_name = summary["model_name"]
    sidx = _build_style_index(summary)

    SKIP_TYPES = {"no_cache"}

    # Precompute max prefix_match across all strategies for sizing
    max_match_global = 0
    for tag in summary["strategies"]:
        jsonl_path = data_dir / f"e2e_{tag}.jsonl"
        if not jsonl_path.exists():
            continue
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                if e["hit"] and e["prefix_match"] > max_match_global:
                    max_match_global = e["prefix_match"]

    fig, ax = plt.subplots(figsize=(8, 6))

    seen_families = set()
    for tag, _stats in summary["strategies"].items():
        cfg = sidx[tag]
        if cfg["type"] in SKIP_TYPES:
            continue
        jsonl_path = data_dir / f"e2e_{tag}.jsonl"
        if not jsonl_path.exists():
            continue
        entries = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        if not entries:
            continue

        _, color, marker, _ls, _lw = _spec(tag, style_map)
        fam = _family_label(tag, style_map)

        hits = [e for e in entries if e["hit"]]
        misses = [e for e in entries if not e["hit"]]

        # Misses: small grey dots
        if misses:
            miss_label = "miss" if "miss" not in seen_families else None
            seen_families.add("miss")
            ax.scatter([e["seq_len"] - e["tokens_saved"] for e in misses],
                       [e["time_s"] for e in misses],
                       color="grey", marker=".", s=15, alpha=0.4,
                       label=miss_label, zorder=2)

        # Hits: colored by family, size proportional to match_len
        if hits:
            label = fam if fam not in seen_families else None
            seen_families.add(fam)
            match_lens = [e["prefix_match"] for e in hits]
            max_match = max(max_match_global, 1)
            sizes = [10 + 120 * (m / max_match) for m in match_lens]
            ax.scatter([e["seq_len"] - e["tokens_saved"] for e in hits],
                       [e["time_s"] for e in hits],
                       color=color, marker=marker, s=sizes, alpha=0.6,
                       label=label, zorder=3)

    ax.set_xlabel("Tokens to process (seq_len − tokens_saved)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"Tokens to process vs wall-clock time — {model_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = out_dir / "tokens_vs_time.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pareto front — tokens saved vs n_blocks for each strategy family
# ---------------------------------------------------------------------------

def _pareto_reference_lines(root_dir):
    """Compute reference lines for Pareto plots.

    Returns (n_convs, perfect_total_savings, perfect_total_time) or None if data unavailable.
    - n_convs: number of unique conversations (vertical line)
    - perfect_total_savings: total speedup if all prefix matches are fully cached
    - perfect_total_time: total time if all prefix matches are fully cached
    """
    # n_convs from overlap_lcp.json
    lcp_path = root_dir / "prepare_data" / "overlap_lcp.json"
    n_convs = None
    if lcp_path.exists():
        lcp_data = json.loads(lcp_path.read_text())
        n_convs = lcp_data.get("n_conversations")

    # Perfect model from any strategy's JSONL (they all see the same prefix_match)
    data_dir = root_dir / "benchmark_e2e"
    perfect_total_savings = None
    perfect_total_time = None
    for jsonl_path in sorted(data_dir.glob("e2e_*.jsonl")):
        tag = jsonl_path.stem.removeprefix("e2e_")
        if tag == "no_cache":
            continue
        entries = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        if not entries or "prefix_match" not in entries[0]:
            continue
        total_tokens = sum(e["seq_len"] for e in entries)
        total_recompute = sum(e["seq_len"] - e["prefix_match"] for e in entries)
        perfect_total_savings = total_tokens / total_recompute if total_recompute > 0 else float("inf")
        perfect_total_time = sum((e["seq_len"] - e["prefix_match"]) * 0.05 for e in entries)
        break  # all strategies see the same prefix_match values

    return n_convs, perfect_total_savings, perfect_total_time


def _add_reference_lines(ax, n_convs, perfect_value, y_metric):
    """Add n_convs vertical line and perfect-model horizontal line to an axis."""
    if n_convs is not None:
        ax.axvline(n_convs, color="red", linestyle=":", linewidth=1.5, alpha=0.7,
                   label=f"n_convs = {n_convs}", zorder=2)
    if perfect_value is not None:
        ax.axhline(perfect_value, color="blue", linestyle=":", linewidth=1.5, alpha=0.7,
                   label=f"perfect = {perfect_value:.2f}" if y_metric == "savings" else f"perfect = {perfect_value:.1f}s",
                   zorder=2)


def _load_jsonl_stats(data_dir, tag):
    """Load per-request JSONL and return (avg_gdn_bytes, avg_n_blocks, avg_speedup, total_speedup, total_time_s)."""
    jsonl_path = data_dir / f"e2e_{tag}.jsonl"
    if not jsonl_path.exists():
        return 0, 0, 1.0, 1.0, 0.0
    entries = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not entries:
        return 0, 0, 1.0, 1.0, 0.0
    avg_gdn = sum(e["cache_gdn_bytes"] for e in entries) / len(entries)
    avg_nb = sum(len(e["added_positions"]) for e in entries) / len(entries)
    avg_speedup = sum(
        e["seq_len"] / (e["seq_len"] - e["tokens_saved"])
        for e in entries if e["seq_len"] - e["tokens_saved"] > 0
    ) / len(entries)
    total_tokens = sum(e["seq_len"] for e in entries)
    total_recompute = sum(e["seq_len"] - e["tokens_saved"] for e in entries)
    total_speedup = total_tokens / total_recompute if total_recompute > 0 else 1.0
    total_time_s = sum(e["time_s"] for e in entries)
    return avg_gdn, avg_nb, avg_speedup, total_speedup, total_time_s


def plot_pareto_broken(out_dir, root_dir=None, style_map=None, break_at=35, reference_lines=False, **_kw):
    """Pareto front with a broken x-axis: [0, break_at] then [break_at+gap, max].

    ``break_at`` is the n_blocks threshold separating the two panels.
    Strategies with avg_n_blocks <= break_at go to the left panel,
    the rest go to the right panel.
    """
    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    data_dir = root_dir / "benchmark_e2e"
    summary_path = data_dir / "e2e_summary.json"
    if not summary_path.exists():
        print(f"  skipping pareto_broken plot ({summary_path} not found)")
        return

    summary = json.loads(summary_path.read_text())
    model_name = summary["model_name"]
    sidx = _build_style_index(summary)

    SKIP_TYPES = {"kv_only", "no_cache"}

    # Collect: type -> [(tag, avg_n_blocks, saved%, avg_gdn, avg_speedup, total_speedup, total_time_s)]
    families = {}
    for tag, stats in summary["strategies"].items():
        cfg = sidx[tag]
        stype = cfg["type"]
        if stype in SKIP_TYPES:
            continue
        tok_saved_frac = stats["tokens_saved"] / stats["tokens_total"] * 100 if stats["tokens_total"] > 0 else 0
        avg_gdn, avg_nb, avg_speedup, total_speedup, total_time_s = _load_jsonl_stats(data_dir, tag)
        families.setdefault(stype, []).append((tag, avg_nb, tok_saved_frac, avg_gdn, avg_speedup, total_speedup, total_time_s))

    if not families:
        print("  no budgeted strategies found, skipping pareto_broken plot")
        return

    # Split points into left (<=break_at) and right (>break_at)
    all_right_x = []
    for points in families.values():
        for p in points:
            if p[1] > break_at:
                all_right_x.append(p[1])

    if not all_right_x:
        # Nothing beyond break_at — fall back to normal pareto
        plot_pareto(out_dir, root_dir=root_dir, style_map=style_map, reference_lines=reference_lines, **_kw)
        return

    right_lo = min(all_right_x) * 0.9
    right_hi = max(all_right_x) * 1.1

    if reference_lines:
        n_convs, perfect_total_savings, perfect_total_time = _pareto_reference_lines(root_dir)
    else:
        n_convs = perfect_total_savings = perfect_total_time = None

    # --- Helper to draw one metric on a broken-axis figure ---
    def _draw_broken(ylabel, title_suffix, y_idx, out_name, y_metric="savings"):
        fig, (ax_l, ax_r) = plt.subplots(
            1, 2, sharey=True, figsize=(10, 6),
            gridspec_kw={"width_ratios": [3, 1], "wspace": 0.08},
        )

        for stype, points in sorted(families.items()):
            points.sort(key=lambda x: x[1])
            rep_tag = points[0][0]
            _, color, marker, ls, lw = _spec(rep_tag, style_map)
            fam = _family_label(rep_tag, style_map)

            xs = [p[1] for p in points]
            ys = [p[y_idx] for p in points]

            # Plot the full series on BOTH axes (matplotlib clips to xlim)
            ax_l.plot(xs, ys, color=color, marker=marker, ls=ls, lw=lw,
                      markersize=8, label=fam, zorder=3)
            ax_r.plot(xs, ys, color=color, marker=marker, ls=ls, lw=lw,
                      markersize=8, zorder=3)

        ax_l.set_xlim(0, break_at * 1.05)
        ax_r.set_xlim(right_lo, right_hi)

        # Diagonal break marks on axes
        d = 0.015
        kwargs = dict(transform=ax_l.transAxes, color="k", clip_on=False, lw=1)
        ax_l.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax_l.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        kwargs["transform"] = ax_r.transAxes
        ax_r.plot((-d, +d), (-d, +d), **kwargs)
        ax_r.plot((-d, +d), (1 - d, 1 + d), **kwargs)

        # Draw "//" break indicator between panels for each family that spans the break
        for stype, points in sorted(families.items()):
            points.sort(key=lambda x: x[1])
            xs = [p[1] for p in points]
            ys = [p[y_idx] for p in points]
            has_left = any(x <= break_at for x in xs)
            has_right = any(x > break_at for x in xs)
            if not (has_left and has_right):
                continue
            rep_tag = points[0][0]
            _, color, _, _, _ = _spec(rep_tag, style_map)
            # Find the last left point and first right point
            last_left_y = [y for x, y in zip(xs, ys) if x <= break_at][-1]
            first_right_y = [y for x, y in zip(xs, ys) if x > break_at][0]
            # Interpolate y at the midpoint of the gap (in figure coords)
            mid_y = (last_left_y + first_right_y) / 2
            # Place "//" text in the gap using figure-level coordinates
            # Convert data y to figure y via ax_l
            fig_y = ax_l.transData.transform((0, mid_y))[1]
            fig_y = fig.transFigure.inverted().transform((0, fig_y))[1]
            # x position: midpoint between the two axes
            bbox_l = ax_l.get_position()
            bbox_r = ax_r.get_position()
            fig_x = (bbox_l.x1 + bbox_r.x0) / 2
            fig.text(fig_x, fig_y, "//", ha="center", va="center",
                     fontsize=11, fontweight="bold", color=color, rotation=15)

        ax_l.spines["right"].set_visible(False)
        ax_r.spines["left"].set_visible(False)
        ax_r.tick_params(left=False)

        ax_l.set_xlabel("Avg number of GDN checkpoints")
        ax_l.set_ylabel(ylabel)
        ax_r.set_xlabel("")
        fig.suptitle(f"Pareto front: {title_suffix}", fontsize=13)

        # Reference lines
        perfect_val = perfect_total_savings if y_metric == "savings" else perfect_total_time
        _add_reference_lines(ax_l, n_convs, perfect_val, y_metric)
        _add_reference_lines(ax_r, n_convs, perfect_val, y_metric)

        # Legend from left axis only (it has all families)
        ax_l.legend(fontsize=9)

        ax_l.grid(True, alpha=0.3)
        ax_r.grid(True, alpha=0.3)

        out_path = out_dir / out_name
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  saved {out_path}")
        plt.close(fig)

    # Panel: total savings
    _draw_broken("Total savings (\u00d7)", "total savings vs checkpoint budget", 5, "pareto_broken_total_savings.png", y_metric="savings")
    # Panel: total time
    _draw_broken("Total processing time (s)", f"total time vs checkpoint budget \u2014 {model_name}", 6, "pareto_broken_total_time.png", y_metric="time")

    # Print table (same as plot_pareto)
    print(f"\n  {'Strategy':<30} {'Avg n_blk':>9} {'Saved%':>8} {'Avg spd':>8} {'Tot spd':>8} {'Tot time':>10} {'Avg GDN (MB)':>12}")
    print(f"  {'-'*91}")
    for stype, points in sorted(families.items()):
        points.sort(key=lambda x: x[1])
        for tag, nb, saved, gdn, avg_spd, tot_spd, tot_time in points:
            label = _spec(tag, style_map)[0]
            print(f"  {label:<30} {nb:>9.1f} {saved:>7.1f}% {avg_spd:>7.2f}\u00d7 {tot_spd:>7.2f}\u00d7 {tot_time:>9.1f}s {gdn/1024/1024:>11.2f}")


def plot_pareto(out_dir, root_dir=None, style_map=None, reference_lines=False, **_kw):
    """Plot Pareto front: tokens_saved (%) vs avg GDN cache size."""
    out_dir = Path(out_dir)
    root_dir = Path(root_dir) if root_dir else out_dir
    data_dir = root_dir / "benchmark_e2e"
    summary_path = data_dir / "e2e_summary.json"
    if not summary_path.exists():
        print(f"  skipping pareto plot ({summary_path} not found)")
        return

    summary = json.loads(summary_path.read_text())
    model_name = summary["model_name"]
    sidx = _build_style_index(summary)

    SKIP_TYPES = {"kv_only", "no_cache"}

    # Collect: type -> [(tag, avg_n_blocks, saved%, avg_gdn_bytes, avg_speedup, total_speedup, total_time_s)]
    families = {}

    for tag, stats in summary["strategies"].items():
        cfg = sidx[tag]
        stype = cfg["type"]
        if stype in SKIP_TYPES:
            continue
        tok_saved_frac = stats["tokens_saved"] / stats["tokens_total"] * 100 if stats["tokens_total"] > 0 else 0
        avg_gdn, avg_nb, avg_speedup, total_speedup, total_time_s = _load_jsonl_stats(data_dir, tag)
        nb = avg_nb
        families.setdefault(stype, []).append((tag, nb, tok_saved_frac, avg_gdn, avg_speedup, total_speedup, total_time_s))

    if not families:
        print("  no budgeted strategies found, skipping pareto plot")
        return

    if reference_lines:
        n_convs, perfect_total_savings, perfect_total_time = _pareto_reference_lines(root_dir)
    else:
        n_convs = perfect_total_savings = perfect_total_time = None

    # --- Panel 1: tokens_saved% vs avg n_blocks ---
    # fig1, ax1 = plt.subplots(figsize=(8, 6))
    # for stype, points in sorted(families.items()):
        # points.sort(key=lambda x: x[1])
        # rep_tag = points[0][0]
        # _, color, marker, ls, lw = _spec(rep_tag, style_map)
        # fam = _family_label(rep_tag, style_map)
        # ax1.plot([p[1] for p in points], [p[2] for p in points],
                 # color=color, marker=marker, ls=ls, lw=lw,
                 # markersize=8, label=fam, zorder=3)
    # ax1.set_xlabel("Avg number of GDN checkpoints")
    # ax1.set_ylabel("Tokens saved (%)")
    # ax1.set_title(f"Pareto front: savings vs checkpoint budget — {model_name}")
    # ax1.legend(fontsize=9)
    # ax1.grid(True, alpha=0.3)
    # out1 = out_dir / "pareto_nblocks.png"
    # fig1.savefig(out1, dpi=200, bbox_inches="tight")
    # print(f"  saved {out1}")
    # plt.close(fig1)
 
    # --- Panel 2: tokens_saved% vs avg GDN cache size (MB) ---
    # fig2, ax2 = plt.subplots(figsize=(8, 6))
    # for stype, points in sorted(families.items()):
        # points.sort(key=lambda x: x[1])
        # rep_tag = points[0][0]
        # _, color, marker, ls, lw = _spec(rep_tag, style_map)
        # fam = _family_label(rep_tag, style_map)
        # gdn_mb = [p[3] / 1024 / 1024 for p in points]
        # saved = [p[2] for p in points]
        # ax2.plot(gdn_mb, saved, color=color, marker=marker, ls=ls, lw=lw,
                 # markersize=8, label=fam, zorder=3)
    # ax2.set_xlabel("Avg GDN cache size (MB)")
    # ax2.set_ylabel("Tokens saved (%)")
    # ax2.set_title(f"Pareto front: savings vs GDN memory — {model_name}")
    # ax2.legend(fontsize=9)
    # ax2.grid(True, alpha=0.3)
    # out2 = out_dir / "pareto_gdn_mb.png"
    # fig2.savefig(out2, dpi=200, bbox_inches="tight")
    # print(f"  saved {out2}")
    # plt.close(fig2)
 
    # --- Panel 3: avg theoretical speedup vs avg n_blocks ---
    # fig3, ax3 = plt.subplots(figsize=(8, 6))
    # for stype, points in sorted(families.items()):
        # points.sort(key=lambda x: x[1])
        # rep_tag = points[0][0]
        # _, color, marker, ls, lw = _spec(rep_tag, style_map)
        # fam = _family_label(rep_tag, style_map)
        # ax3.plot([p[1] for p in points], [p[4] for p in points],
                 # color=color, marker=marker, ls=ls, lw=lw,
                 # markersize=8, label=fam, zorder=3)
    # ax3.set_xlabel("Avg number of GDN checkpoints")
    # ax3.set_ylabel("Avg per-request speedup (×)")
    # ax3.set_title(f"Pareto front: avg speedup vs checkpoint budget — {model_name}")
    # ax3.legend(fontsize=9)
    # ax3.grid(True, alpha=0.3)
    # out3 = out_dir / "pareto_speedup.png"
    # fig3.savefig(out3, dpi=200, bbox_inches="tight")
    # print(f"  saved {out3}")
    # plt.close(fig3)
 
    # --- Panel 4: total speedup vs avg n_blocks ---
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    for stype, points in sorted(families.items()):
        points.sort(key=lambda x: x[1])
        rep_tag = points[0][0]
        _, color, marker, ls, lw = _spec(rep_tag, style_map)
        fam = _family_label(rep_tag, style_map)
        ax4.plot([p[1] for p in points], [p[5] for p in points],
                 color=color, marker=marker, ls=ls, lw=lw,
                 markersize=8, label=fam, zorder=3)
    ax4.set_xlabel("Avg number of GDN checkpoints")
    ax4.set_ylabel("Total savings (×)")
    ax4.set_title(f"Pareto front: total savings vs checkpoint budget")
    _add_reference_lines(ax4, n_convs, perfect_total_savings, "savings")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    out4 = out_dir / "pareto_total_savings.png"
    fig4.savefig(out4, dpi=200, bbox_inches="tight")
    print(f"  saved {out4}")
    plt.close(fig4)

    # --- Panel 5: total processing time vs avg n_blocks ---
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    for stype, points in sorted(families.items()):
        points.sort(key=lambda x: x[1])
        rep_tag = points[0][0]
        _, color, marker, ls, lw = _spec(rep_tag, style_map)
        fam = _family_label(rep_tag, style_map)
        ax5.plot([p[1] for p in points], [p[6] for p in points],
                 color=color, marker=marker, ls=ls, lw=lw,
                 markersize=8, label=fam, zorder=3)
    ax5.set_xlabel("Avg number of GDN checkpoints")
    ax5.set_ylabel("Total processing time (s)")
    ax5.set_title(f"Pareto front: total time vs checkpoint budget — {model_name}")
    _add_reference_lines(ax5, n_convs, perfect_total_time, "time")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    out5 = out_dir / "pareto_total_time.png"
    fig5.savefig(out5, dpi=200, bbox_inches="tight")
    print(f"  saved {out5}")
    plt.close(fig5)

    # Print table
    print(f"\n  {'Strategy':<30} {'Avg n_blk':>9} {'Saved%':>8} {'Avg spd':>8} {'Tot spd':>8} {'Tot time':>10} {'Avg GDN (MB)':>12}")
    print(f"  {'-'*91}")
    for stype, points in sorted(families.items()):
        points.sort(key=lambda x: x[1])
        for tag, nb, saved, gdn, avg_spd, tot_spd, tot_time in points:
            label = _spec(tag, style_map)[0]
            print(f"  {label:<30} {nb:>9.1f} {saved:>7.1f}% {avg_spd:>7.2f}× {tot_spd:>7.2f}× {tot_time:>9.1f}s {gdn/1024/1024:>11.2f}")


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
    plot_lcp_over_rounds(out_dir, root_dir=root_dir, style_map=style_map)
    plot_checkpoint_positions(out_dir, root_dir=root_dir, style_map=style_map)
    plot_gdn_gap(out_dir, root_dir=root_dir, style_map=style_map)
    plot_cache_breakdown(out_dir, root_dir=root_dir, style_map=style_map)
    plot_histograms(out_dir, root_dir=root_dir, style_map=style_map)
    plot_tokens_vs_time(out_dir, root_dir=root_dir, style_map=style_map)
    plot_pareto(out_dir, root_dir=root_dir, style_map=style_map, **_kw)

# ---------------------------------------------------------------------------
# Hydra entry point — dispatches via plot._target_
# ---------------------------------------------------------------------------
@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    from sparse_prefix_caching.utils import setup_output_dir, resolve_strategies
    root_dir = Path(cfg.output_dir)
    out_dir = setup_output_dir(cfg, "plot_results")
    resolve_strategies(cfg)
    style_map = _build_style_map(cfg.strategies)

    print(f"Plotting results from {root_dir} into {out_dir}")
    hydra.utils.call(cfg.plot_results, out_dir=out_dir, root_dir=root_dir, style_map=style_map)
    print("Done.")


if __name__ == "__main__":
    main()
