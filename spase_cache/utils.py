"""Shared utilities for benchmark scripts.

Everything that benchmark scripts both need lives here
so the scripts stay independent of each other.
"""
import gc
import json
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache, Qwen3_5TextModel

from spase_cache.checkpoint_cache import (
    PrefixCheckpointStore,
    RecurrentCheckpoint,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output directory & logging
# ---------------------------------------------------------------------------
def setup_output_dir(cfg, task: str):
    """Create output dir and configure file + console logging. Returns Path.

    Each script gets its own subfolder: {output_dir}/{task}/.
    The overwrite check applies to the task subfolder.
    """
    from hydra.core.hydra_config import HydraConfig

    if cfg.output_dir:
        root_dir = Path(cfg.output_dir)
    else:
        root_dir = Path(HydraConfig.get().runtime.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    out_dir = root_dir / task
    if not cfg.get("overwrite", True) and out_dir.exists():
        raise FileExistsError(f"Output dir already exists and overwrite=False: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, cfg.get("log_level", "INFO").upper(), logging.INFO)

    file_handler = logging.FileHandler(out_dir / "run.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    return out_dir

def _save_jsonl(path, entries):
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")



# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def gpu_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _sync_device(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def _model_device(model: Qwen3_5TextModel) -> torch.device:
    return next(model.parameters()).device


def _get_linear_layers(config: Qwen3_5TextConfig) -> list[int]:
    return [i for i, lt in enumerate(config.layer_types) if lt == "linear_attention"]


def _get_attention_layers(config: Qwen3_5TextConfig) -> list[int]:
    return [i for i, lt in enumerate(config.layer_types) if lt == "full_attention"]


def make_model_config(cfg) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        intermediate_size=cfg.model.intermediate_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        num_key_value_heads=cfg.model.num_key_value_heads,
        head_dim=cfg.model.head_dim,
        linear_conv_kernel_dim=cfg.model.linear_conv_kernel_dim,
        linear_key_head_dim=cfg.model.linear_key_head_dim,
        linear_value_head_dim=cfg.model.linear_value_head_dim,
        linear_num_key_heads=cfg.model.linear_num_key_heads,
        linear_num_value_heads=cfg.model.linear_num_value_heads,
        max_position_embeddings=cfg.model.max_position_embeddings,
        torch_dtype=cfg.dtype,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.model.rope_theta},
    )

def make_model(cfg) -> Qwen3_5TextModel:
    model_config = make_model_config(cfg)
    model = Qwen3_5TextModel(model_config).eval()
    model.set_attn_implementation("sdpa")
    model.to(cfg.device)
    return model


# ---------------------------------------------------------------------------
# Chunked prefill
#
# HF materializes a full causal mask when Q_len != K_len. For large contexts
# it is very spacy -- (Q_len × K_len × 4B). Chunking keeps the mask small.
#
#
# Also chunked prefill is closer to real serving systems than prefilling
# everything at once; for systems like vLLM it allows efficient interleaving
# of prefill and decode requests in continuous batching.
# ---------------------------------------------------------------------------
def chunked_prefill(
    model: Qwen3_5TextModel,
    input_ids: torch.Tensor,
    resume_pos,
    cache,
    max_chunk: int = 4096,
) -> tuple[torch.Tensor, Qwen3_5DynamicCache]:

    device = _model_device(model)
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    pos = resume_pos
    output = None
    while pos < seq_len:
        end = min(pos + max_chunk, seq_len)
        with torch.no_grad():
            output = model(
                input_ids=input_ids[:, pos:end],
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(pos, end, device=device),
            )
        cache = output.past_key_values
        pos = end
    return output.last_hidden_state, cache

def prefill_baseline(model, input_ids):
    return chunked_prefill(model, input_ids, 0, Qwen3_5DynamicCache(config=model.config))

def warmup(model, N):
    dev = _model_device(model)
    ids = torch.randint(0, model.config.vocab_size, (1, N)).to(dev)
    prefill_baseline(model, ids)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def time_fn(n_runs, dev, fn, *args):
    _sync_device(dev)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        ret = fn(*args)
        del ret
    _sync_device(dev)
    return (time.perf_counter() - t0) / n_runs

# ---------------------------------------------------------------------------
# Prefill with capture at arbitrary positions
# ---------------------------------------------------------------------------
def prefill_and_capture_at(model, input_ids, ckpt_positions):
    """Run prefill capturing GDN+conv states and attention KV at specified positions.

    Checkpoints are moved to CPU immediately after cloning to avoid
    accumulating all of them on GPU (block-B=16 at 32K = 2048 ckpts ≈ 3.3GB).
    """
    device = _model_device(model)
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    config = model.config
    linear_layers = _get_linear_layers(config)
    attn_layers = _get_attention_layers(config)

    ckpt_positions = sorted(set(p for p in ckpt_positions if 0 < p <= seq_len))
    store = PrefixCheckpointStore(prefix_tokens=input_ids.clone().cpu())
    boundaries = sorted(set([0] + ckpt_positions + [seq_len]))

    log.info("capture %d ckpts for seq_len=%d, boundaries=%d segments, gpu=%.0fMB",
             len(ckpt_positions), seq_len, len(boundaries) - 1, gpu_mb())

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
                    li: cache.recurrent_states[li].clone().cpu()
                    for li in linear_layers if cache.recurrent_states[li] is not None
                },
                conv_states={
                    li: cache.conv_states[li].clone().cpu()
                    for li in linear_layers if cache.conv_states[li] is not None
                },
            )

    log.info("capture done, cloning KV, gpu=%.0fMB", gpu_mb())
    for li in attn_layers:
        if cache.key_cache[li] is not None:
            store.kv_cache_keys[li] = cache.key_cache[li].clone().cpu()
            store.kv_cache_values[li] = cache.value_cache[li].clone().cpu()
    return store


# ---------------------------------------------------------------------------
# Attention layer disabler (context manager)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# FIFO prefix cache with memory budget
# ---------------------------------------------------------------------------
def _truncate_store(store, budget_bytes):
    """Remove GDN checkpoints (smallest positions first) until store fits in budget.
    Returns (store, new_size) or (None, 0) if even KV-only exceeds budget."""
    size = store.memory_bytes()
    if size <= budget_bytes:
        return store, size
    positions = sorted(store.checkpoints.keys())
    for pos in positions:
        ckpt = store.checkpoints.pop(pos)
        for t in ckpt.recurrent_states.values():
            size -= t.nelement() * t.element_size()
        for t in ckpt.conv_states.values():
            size -= t.nelement() * t.element_size()
        if size <= budget_bytes:
            return store, size
    if size > budget_bytes:
        return None, 0
    return store, size


class PrefixCache:
    def __init__(self, budget_bytes):
        self.budget = budget_bytes
        self.used = 0
        self.entries = OrderedDict()  # key -> (store, size_bytes)

    def get(self, key):
        if key in self.entries:
            return self.entries[key][0]
        return None

    def find_best_prefix(self, conv_id, turn):
        """Find the longest cached prefix for this conversation with turn index < turn."""
        for t in range(turn - 1, -1, -1):
            k = (conv_id, t)
            if k in self.entries:
                return self.entries[k][0], t
        return None, -1

    def put(self, key, store, size_bytes):
        if size_bytes > self.budget:
            store, size_bytes = _truncate_store(store, self.budget)
            if store is None:
                return
        while self.used + size_bytes > self.budget and self.entries:
            _, (_, evicted_size) = self.entries.popitem(last=False)
            self.used -= evicted_size
        self.entries[key] = (store, size_bytes)
        self.used += size_bytes

    @property
    def n_entries(self):
        return len(self.entries)
