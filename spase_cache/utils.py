"""Shared utilities for benchmark scripts.

Everything that benchmark scripts both need lives here
so the scripts stay independent of each other.
"""
import gc
import shutil
import json
import logging
import time
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval, use_cache=True)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache, Qwen3_5TextModel

from spase_cache.checkpoint_cache import (
    PrefixCheckpointStore,
    RecurrentCheckpoint,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy resolution
# ---------------------------------------------------------------------------
_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "conf" / "strategy"


def resolve_strategies(cfg):
    """Resolve strategy dict {tag: {overrides}} into a list of configs.

    Uses OmegaConf.load/merge so base.yaml defaults and ${} interpolations
    work natively. ``_base_`` override selects which YAML to load (defaults
    to the tag).
    """
    base_cfg = OmegaConf.load(_STRATEGY_DIR / "base.yaml") if (_STRATEGY_DIR / "base.yaml").exists() else {}

    resolved = []
    for tag, overrides in cfg.strategies.items():
        # Resolve top-level interpolations (e.g. ${n_blocks_when_kvcache_equals_gdncache})
        overrides = OmegaConf.to_container(overrides, resolve=True) if overrides else {}
        name = overrides.pop("_base_", tag)

        yaml_path = _STRATEGY_DIR / f"{name}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Strategy config not found: {yaml_path}")

        strat_cfg = OmegaConf.load(yaml_path)
        if "defaults" in strat_cfg:
            del strat_cfg["defaults"]

        merged = OmegaConf.to_container(OmegaConf.merge(base_cfg, strat_cfg, overrides, {"tag": tag}))
        # Resolve ${key} references to sibling keys within each strategy dict
        for k, v in merged.items():
            if isinstance(v, str) and "${" in v:
                merged[k] = v.replace("${", "{").format_map(merged)
        resolved.append(merged)

    cfg.strategies = OmegaConf.create(resolved)
    log.info("resolved %d strategies: %s", len(resolved), [s["tag"] for s in resolved])


# ---------------------------------------------------------------------------
# Request interleaving
# ---------------------------------------------------------------------------
def interleave(requests, seed):
    """Interleave conversations with Poisson arrival times, preserving turn order.

    Simulates concurrent load: crowdsourced timestamps almost never overlap,
    leading to trivial 100% cache hits. Interleaving with exponential
    inter-arrival times produces realistic cache miss patterns.

    Args:
        requests: list of tuples where first element is conversation id.
        seed: random seed for reproducibility.
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
    if out_dir.exists():
        if not cfg.get("overwrite", True): 
            raise FileExistsError(f"Output dir already exists and overwrite=False: {out_dir}")
        else:
             shutil.rmtree(out_dir)
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


def gdn_checkpoint_bytes(config: Qwen3_5TextConfig) -> int:
    """Bytes for one GDN checkpoint (all linear-attention layers in one group).

    Per layer: recurrent state n_v * d_h * d_h  +  conv state n_v * d_h * (K-1).
    Summed over all linear-attention layers.
    Formula from README: 3 * n_v * d_h^2 (dominant term, ignoring conv).
    """
    n_v = config.linear_num_value_heads
    d_h = config.linear_value_head_dim
    k = config.linear_conv_kernel_dim
    n_linear = len(_get_linear_layers(config))
    elem = 2 if str(config.torch_dtype) in ("float16", "torch.float16", "bfloat16", "torch.bfloat16") else 4
    rec_elems = n_v * d_h * d_h
    conv_elems = n_v * d_h * (k - 1)
    return n_linear * (rec_elems + conv_elems) * elem


def kv_per_token_bytes(config: Qwen3_5TextConfig) -> int:
    """Bytes of attention KV cache per token (all attention layers in one group).

    Per layer per token: 2 * n_kv * d_a (key + value).
    """
    n_kv = config.num_key_value_heads
    d_a = config.head_dim
    n_attn = len(_get_attention_layers(config))
    elem = 2 if str(config.torch_dtype) in ("float16", "torch.float16", "bfloat16", "torch.bfloat16") else 4
    return n_attn * 2 * n_kv * d_a * elem


def compute_r(config: Qwen3_5TextConfig) -> float:
    """Ratio of per-checkpoint GDN size to per-token KV size.

    r = gdn_checkpoint_bytes / kv_per_token_bytes.
    From README: r = 3 * n_v * d_h^2 / (2 * n_kv * d_a)  (element-count ratio,
    same as byte ratio since both use the same dtype).
    For 0.8B: r=768, for 27B: r=4608.
    """
    ckpt_b = gdn_checkpoint_bytes(config)
    kv_b = kv_per_token_bytes(config)
    return ckpt_b / kv_b


def max_checkpoints_for_budget(config: Qwen3_5TextConfig, gdn_budget_bytes: int) -> int:
    """Maximum number of GDN checkpoints that fit in a given byte budget."""
    return gdn_budget_bytes // gdn_checkpoint_bytes(config)


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
    store = PrefixCheckpointStore(prefix_tokens=input_ids[0].clone().cpu())
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
def _truncate_gdn(store, gdn_budget):
    """Remove GDN checkpoints (smallest positions first) until GDN fits in budget."""
    gdn_size = store.gdn_bytes()
    if gdn_size <= gdn_budget:
        return store
    positions = sorted(store.checkpoints.keys())
    for pos in positions:
        ckpt = store.checkpoints.pop(pos)
        for t in ckpt.recurrent_states.values():
            gdn_size -= t.nelement() * t.element_size()
        for t in ckpt.conv_states.values():
            gdn_size -= t.nelement() * t.element_size()
        if gdn_size <= gdn_budget:
            return store
    return store  # all checkpoints removed


def _prefix_match_len(stored_tokens, input_ids):
    """Length of common prefix between stored token tensor and input_ids tensor.

    Both are 1-D int tensors. Comparison is vectorized.
    """
    n = min(len(stored_tokens), len(input_ids))
    if n == 0:
        return 0
    match = stored_tokens[:n] == input_ids[:n]
    # first mismatch position (or n if all match)
    mismatches = torch.where(~match)[0]
    return n if len(mismatches) == 0 else mismatches[0].item()


class PrefixCache:
    """FIFO prefix cache with separate KV and GDN budgets.

    Entries are keyed by (conv_id, seq_id) internally. Lookup finds the
    entry for a given conv_id whose prefix_tokens best matches the query.
    """

    def __init__(self, kv_budget_bytes, gdn_budget_bytes):
        self.kv_budget = kv_budget_bytes
        self.gdn_budget = gdn_budget_bytes
        self.kv_used = 0
        self.gdn_used = 0
        self.entries = OrderedDict()  # (conv_id, seq_id) -> (store, kv_bytes, gdn_bytes)
        self._conv_entries = defaultdict(list)  # conv_id -> list of (conv_id, seq_id) keys
        self._next_id = 0

    def find_best_prefix(self, conv_id, input_ids):
        """Find cached entry for conv_id with longest token prefix match.

        Args:
            conv_id: conversation / document id.
            input_ids: 1-D token tensor for the current request.

        Returns (store, match_len) or (None, 0).
        """
        best_store = None
        best_len = 0
        for key in self._conv_entries.get(conv_id, []):
            if key not in self.entries:
                continue
            store = self.entries[key][0]
            if store.prefix_tokens is None:
                continue
            ml = _prefix_match_len(store.prefix_tokens, input_ids)
            if ml > best_len:
                best_len = ml
                best_store = store
        return best_store, best_len

    def put(self, conv_id, store, _size_bytes=None):
        _truncate_gdn(store, self.gdn_budget)

        kv_b = store.kv_bytes()
        gdn_b = store.gdn_bytes()

        if kv_b > self.kv_budget:
            return

        while self.entries and (
            self.kv_used + kv_b > self.kv_budget or
            self.gdn_used + gdn_b > self.gdn_budget
        ):
            evicted_key, (_, evicted_kv, evicted_gdn) = self.entries.popitem(last=False)
            self.kv_used -= evicted_kv
            self.gdn_used -= evicted_gdn
            # clean up conv index
            ecid = evicted_key[0]
            if ecid in self._conv_entries:
                try:
                    self._conv_entries[ecid].remove(evicted_key)
                except ValueError:
                    pass
                if not self._conv_entries[ecid]:
                    del self._conv_entries[ecid]

        key = (conv_id, self._next_id)
        self._next_id += 1
        self.entries[key] = (store, kv_b, gdn_b)
        self.kv_used += kv_b
        self.gdn_used += gdn_b
        self._conv_entries[conv_id].append(key)

    @property
    def n_entries(self):
        return len(self.entries)

    @property
    def used(self):
        return self.kv_used + self.gdn_used
