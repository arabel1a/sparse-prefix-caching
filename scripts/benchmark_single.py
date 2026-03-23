"""Empirical benchmark of prefix caching strategies.

All checkpoint strategies store both GDN recurrent states and attention KV cache.
Results saved to baselines_results.json.
"""
import spase_cache
import json
import logging
import time

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from spase_cache.checkpoint_cache import (
    prefill_from_checkpoint,
)
from spase_cache.utils import (
    setup_output_dir,
    resolve_strategies,
    make_model,
    prefill_baseline,
    _model_device,
    _sync_device,
    gpu_mb,
    free_gpu,
    reset_peak_memory,
    prefill_and_capture_at,
    time_fn,
    warmup,
)
from spase_cache.strategies import checkpoint_positions

log = logging.getLogger(__name__)

def _save_results(path, data):
    path.write_text(json.dumps(data, indent=2))

@hydra.main(config_path=r'../conf', config_name='config', version_base="1.3")
def main(cfg: DictConfig):
    out_dir = setup_output_dir(cfg, "benchmark_single")
    resolve_strategies(cfg)
    model = make_model(cfg)
    dev = _model_device(model)
    config = model.config

    bb = cfg.benchmark_baselines
    seq_lens = list(bb.seq_lens)
    n_runs = bb.n_runs
    strategies = list(cfg.strategies)

    all_tags = [s.tag for s in strategies]
    results = {t: [] for t in all_tags}
    capture_times = {t: [] for t in all_tags}
    cache_sizes = {t: [] for t in all_tags}
    completed_seq_lens = []
    wall_t0 = time.perf_counter()

    out_path = out_dir / "baselines_results.json"
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    strategy_styles = OmegaConf.to_container(cfg.strategies, resolve=True)

    # Warmup
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

        # Measure full no_cache baseline
        input_ids = torch.randint(0, config.vocab_size, (1, N)).to(dev)
        t_full = time_fn(n_runs, dev, prefill_baseline, model, input_ids)
        times = {"no_cache": t_full}
        cap_times = {"no_cache": 0.0}
        bytes_map = {"no_cache": 0}

        # Checkpoint strategies: always include KV cache
        for strat in strategies:
            if strat.type == "no_cache":
                continue
            if strat.type.startswith("histogram_"):
                continue  # histogram strategies need trace data, skip in single-seq benchmark

            positions = checkpoint_positions(N, **strat)
            _sync_device(dev)
            cap_t0 = time.perf_counter()
            store = prefill_and_capture_at(model, input_ids, positions)
            _sync_device(dev)
            cap_times[strat.tag] = time.perf_counter() - cap_t0
            bytes_map[strat.tag] = store.memory_bytes()
            store.to("cpu")

            times[strat.tag] = time_fn(n_runs, dev, prefill_from_checkpoint, model, input_ids, store)

            del store; free_gpu()
            reset_peak_memory()

        for t in all_tags:
            results[t].append(times.get(t, 0))
            capture_times[t].append(cap_times.get(t, 0.0))
            cache_sizes[t].append(bytes_map.get(t, 0))

        completed_seq_lens.append(N)

        parts = [f"N={N:5d}"]
        for t in all_tags:
            parts.append(f"{t} {times.get(t, 0)*1000:7.1f}ms cap={cap_times.get(t, 0)*1000:5.1f}ms")
        log.info(" | ".join(parts))
        wall_time = time.perf_counter() - wall_t0
        _save_results(out_path, {
            "model_name": cfg.model.name,
            "seq_lens": completed_seq_lens,
            "wall_time": wall_time,
            "strategies": {t: {"times_s": results[t], "capture_times_s": capture_times[t], "cache_bytes": cache_sizes[t]} for t in all_tags},
            "model_params": model_params,
            "strategy_styles": strategy_styles,
        })
        log.info("Saved results to %s", out_path)

    log.info("Done. Results saved to %s", out_path)

if __name__ == "__main__":
    main()
