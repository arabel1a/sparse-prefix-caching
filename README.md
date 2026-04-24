# Sparse Prefix Caching for Hybrid LLMs

Experiment codebase for sparse checkpoint placement in hybrid (attention + state-space) LLM serving. Instead of caching recurrent state at every token or every block boundary, we store checkpoints at a **sparse** set of positions and recompute the gap on cache hit.

The key idea: recurrent layers (e.g. GatedDeltaNet in Qwen3.5) can resume from a single stored state, so we don't need dense per-token caching. But recurrent state is expensive ($d_\text{head}^2$ per head vs $d_\text{head}$ for attention KV), so fewer checkpoints = more sequences fit in cache = higher hit rate.

## Checkpoint placement strategies

All strategies store the full attention KV cache; they differ only in where recurrent (GDN) state checkpoints are placed. Positions are clipped to block boundaries for kernel compatibility.

| Strategy | Config type | Description |
|---|---|---|
| No cache | `no_cache` | No checkpoints, full recompute |
| Block | `block` | Checkpoint every `block_size` tokens |
| Balanced | `balanced` | `n_blocks` evenly spaced checkpoints |
| Sqrt | `sqrt` | $\sqrt{L}$-spaced checkpoints |
| Log/Diadic | `log` | Checkpoints at $1, 2, 4, 8, \ldots$ |
| **DP-optimal** | `hist_exp_decay` | Solves an $O(NM)$ DP on the empirical overlap distribution with exponential reweighting for drift |

## Datasets

Experiments target workloads where requests share long prefixes:

- **QuALITY / NarrativeQA** -- multiple questions about the same long document
- **System Prompts** -- real system prompts from major LLM providers combined with ShareGPT user queries

Requests within each group are interleaved with Poisson arrival to simulate realistic concurrent load.

## Reproducing

```bash
uv sync

for cfg in quality_sim quality_clean system_prompts_clean system_prompts_sim narrativeqa_sim; do
  bash runall.sh -cn $cfg
done
```

`*_sim` configs run with `dry_run=True` (no GPU, synthetic timing). `*_clean` configs run real inference on a single Qwen3.5-0.8B layer group.

Outputs (plots, JSONL logs) go to `outputs/<dataset>/<run_name>/`.
