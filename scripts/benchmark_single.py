"""Empirical benchmark of prefix caching strategies (compute only).

Attention cost is measured once per sequence length and added to checkpoint strategies.
Results saved to baselines_results.json.
"""
import spase_cache
import json
import logging

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from spase_cache.checkpoint_cache import (
    prefill_from_checkpoint,
    gdn_cache_bytes,
)
from spase_cache.utils import (
    setup_output_dir,
    make_model,
    prefill_baseline,
    _model_device,
    _sync_device,
    gpu_mb,
    free_gpu,
    reset_peak_memory,
    prefill_and_capture_at,
    disable_attention_layers,
    time_fn,
    warmup,
)
from spase_cache.strategies import checkpoint_positions

log = logging.getLogger(__name__)

def _save_results(path, data):
    path.write_text(json.dumps(data, indent=2))

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = setup_output_dir(cfg, "benchmark_single")
    model = make_model(cfg)
    dev = _model_device(model)
    config = model.config

    bb = cfg.benchmark_baselines
    seq_lens = list(bb.seq_lens)
    B = cfg.baseline.block_size
    n_runs = bb.n_runs
    strategies = list(bb.strategies)

    results = {k: [] for k in strategies}
    cache_sizes = {k: [] for k in strategies}
    completed_seq_lens = []

    out_path = out_dir / "baselines_results.json"
    model_params = OmegaConf.to_container(cfg.model, resolve=True)

    # Warmup todo: move to common
    log.info("Warming up...")
    warmup(model, seq_lens[0])
    warmup(model, seq_lens[-1])
    _sync_device(dev)
    log.info("Warmup done. gpu=%.0fMB", gpu_mb())

    for N in seq_lens:
        torch.manual_seed(cfg.seed)
        free_gpu()
        log.info("=== N=%d  gpu=%.0fMB ===", N, gpu_mb())

        input_ids = torch.randint(0, config.vocab_size, (1, N)).to(dev)

        # Per-N warmup
        warmup(model, N)
        free_gpu()

        # Measure full no_cache and attention cost
        input_ids = torch.randint(0, config.vocab_size, (1, N)).to(dev)
        t_full = time_fn(n_runs, dev, prefill_baseline, model, input_ids)
        with disable_attention_layers(model):
            t_gdn_only = time_fn(n_runs, dev, prefill_baseline, model, input_ids)
        attn_cost = max(t_full - t_gdn_only, 0)
        # todo: add kvcache size
        # todo: actually it shpuld skip ffns too, co add ffn calculation
        times = {"no_cache": t_full}
        bytes_map = {"no_cache": 0}

        # Checkpoint strategies: measure GDN-only time, add attn cost
        for strat in strategies:
            if strat == "no_cache":
                continue

            positions = checkpoint_positions(strat, N, B)
            store = prefill_and_capture_at(model, input_ids, positions)
            bytes_map[strat] = gdn_cache_bytes(store)

            store.to(dev)
            with disable_attention_layers(model):
                t = time_fn(n_runs, dev, prefill_from_checkpoint, model, input_ids, store)
            store.to("cpu")
            times[strat] = t + attn_cost
            times[strat + "+attn_cache"] = t

            del store; free_gpu()
            reset_peak_memory()

        for k in strategies:
            results[k].append(times.get(k, 0))
            cache_sizes[k].append(bytes_map.get(k, 0))

        completed_seq_lens.append(N)

        parts = [f"N={N:5d}"]
        for k in strategies:
            parts.append(f"{k} {times.get(k, 0)*1000:7.1f}ms")
        log.info(" | ".join(parts))

        _save_results(out_path, {
            "model_name": cfg.model.name,
            "block_size": B,
            "seq_lens": completed_seq_lens,
            "strategies": {k: {"times_s": results[k], "cache_bytes": cache_sizes[k]} for k in strategies},
            "model_params": model_params,
        })
        log.info("Saved results to %s", out_path)

    log.info("Done. Results saved to %s", out_path)

if __name__ == "__main__":
    main()
