"""
Core fucntionality of GDN checkpointing.

Requires monkey-patching Qwen3_5GatedDeltaNet.forward() because the upstream HF
implementation always passes initial_state=None during prefill (seq_len > 1).
Our patch passes the cached recurrent_state when available.
See patches.py for details
"""

from dataclasses import dataclass, field

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DynamicCache,
    Qwen3_5TextModel,
)


@dataclass
class RecurrentCheckpoint:
    position: int
    recurrent_states: dict[int, torch.Tensor]  # layer_idx → (B, H, Dk, Dv)
    conv_states: dict[int, torch.Tensor]        # layer_idx → (B, C, K-1)

    def to(self, device):
        self.recurrent_states = {k: v.to(device) for k, v in self.recurrent_states.items()}
        self.conv_states = {k: v.to(device) for k, v in self.conv_states.items()}
        return self


@dataclass
class PrefixCheckpointStore:
    """Stores recurrent state checkpoints at choosen positions for a token prefix."""
    prefix_tokens: torch.Tensor | None = None
    checkpoints: dict[int, RecurrentCheckpoint] = field(default_factory=dict)
    kv_cache_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    kv_cache_values: dict[int, torch.Tensor] = field(default_factory=dict)

    def best_checkpoint(self, seq_len: int) -> RecurrentCheckpoint | None:
        best = None
        for pos in sorted(self.checkpoints.keys()):
            if pos <= seq_len:
                best = self.checkpoints[pos]
        return best

    @property
    def num_checkpoints(self) -> int:
        return len(self.checkpoints)

    def gdn_bytes(self) -> int:
        total = 0
        for ckpt in self.checkpoints.values():
            for t in ckpt.recurrent_states.values():
                total += t.nelement() * t.element_size()
            for t in ckpt.conv_states.values():
                total += t.nelement() * t.element_size()
        return total

    def kv_bytes(self) -> int:
        total = 0
        for t in self.kv_cache_keys.values():
            total += t.nelement() * t.element_size()
        for t in self.kv_cache_values.values():
            total += t.nelement() * t.element_size()
        return total

    @property
    def kv_len(self) -> int:
        """Sequence length covered by stored KV cache."""
        for t in self.kv_cache_keys.values():
            return t.shape[2]  # (B, H, L, D)
        return 0

    def memory_bytes(self) -> int:
        return self.gdn_bytes() + self.kv_bytes()

    def to(self, device):
        if self.prefix_tokens is not None:
            self.prefix_tokens = self.prefix_tokens.to(device)
        for ckpt in self.checkpoints.values():
            ckpt.to(device)
        self.kv_cache_keys = {k: v.to(device) for k, v in self.kv_cache_keys.items()}
        self.kv_cache_values = {k: v.to(device) for k, v in self.kv_cache_values.items()}
        return self


# ---------------------------------------------------------------------------
# Prefill from checkpoint
# ---------------------------------------------------------------------------
def prefill_from_checkpoint(
    model: Qwen3_5TextModel,
    input_ids: torch.Tensor,
    store: PrefixCheckpointStore,
    max_chunk: int = 4096,
    match_len: int | None = None,
) -> tuple[torch.Tensor, Qwen3_5DynamicCache]:
    """Resume prefill from best GDN checkpoint, recomputing full model for tail.

    To produce GDN input at layer i, we need attention output at layer i-1.
    Attention output requires Q (from current hidden states), not just cached
    K,V.  So all FFNs and attention must recompute from the checkpoint/last kv
    onward.

    Therefore: full model runs from checkpoint position m to seq_len.
    If no GDN checkpoint exists, this is equivalent to no_cache.

    match_len: number of leading tokens that actually match the stored prefix.
        KV cache and GDN checkpoints beyond this point are stale (computed for
        different tokens) and must not be used.
    """
    from spase_cache.utils import _model_device, _get_linear_layers, _get_attention_layers, prefill_baseline, chunked_prefill

    device = _model_device(model)
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    config = model.config
    linear_layers = _get_linear_layers(config)
    attn_layers = _get_attention_layers(config)

    # Determine available KV cache length, clipped to actual prefix match
    kv_len = min(store.kv_len, seq_len)
    if match_len is not None:
        kv_len = min(kv_len, match_len)

    # Best GDN checkpoint that doesn't exceed KV coverage.
    # If GDN checkpoint is ahead of KV, attention would miss positions
    # kv_len..gdn_pos, so we can only use checkpoints within KV range.
    ckpt = store.best_checkpoint(min(kv_len, seq_len))

    if ckpt is None:
        # No usable GDN checkpoint (kv_only, or all checkpoints beyond KV range).
        # KV cache alone can't skip compute: recomputing from 0 overwrites cached
        # KV via cache_position scatter, so seeding it is pointless.
        return prefill_baseline(model, input_ids)

    resume_pos = ckpt.position

    if resume_pos >= seq_len and kv_len >= seq_len:
        # Full cache hit — no compute needed, return cached state
        cache = Qwen3_5DynamicCache(config=config)
        for li in linear_layers:
            if li in ckpt.recurrent_states:
                cache.recurrent_states[li] = ckpt.recurrent_states[li].to(device)
            if li in ckpt.conv_states:
                cache.conv_states[li] = ckpt.conv_states[li].to(device)
        for li in attn_layers:
            if li in store.kv_cache_keys:
                cache.key_cache[li] = store.kv_cache_keys[li][:, :, :seq_len, :].to(device)
                cache.value_cache[li] = store.kv_cache_values[li][:, :, :seq_len, :].to(device)
        dummy = torch.zeros(1, 1, config.hidden_size, device=device)
        return dummy, cache

    # Build cache with restored GDN states + KV up to resume position.
    # Full model runs from resume_pos: all layers (GDN, attention, FFN)
    # must recompute because each layer's output feeds into the next.
    cache = Qwen3_5DynamicCache(config=config)
    for li in linear_layers:
        if li in ckpt.recurrent_states:
            cache.recurrent_states[li] = ckpt.recurrent_states[li].to(device)
        if li in ckpt.conv_states:
            cache.conv_states[li] = ckpt.conv_states[li].to(device)
    for li in attn_layers:
        if li in store.kv_cache_keys:
            cache.key_cache[li] = store.kv_cache_keys[li][:, :, :resume_pos, :].to(device)
            cache.value_cache[li] = store.kv_cache_values[li][:, :, :resume_pos, :].to(device)

    # Recompute full model for tail tokens
    return chunked_prefill(model, input_ids, resume_pos, cache, max_chunk=max_chunk)

# ---------------------------------------------------------------------------
# Correctness test - differences in attention outputs
# ---------------------------------------------------------------------------
def test_correctness(cfg):
    from spase_cache.utils import make_model, prefill_baseline, _get_linear_layers, _get_attention_layers, prefill_and_capture_at
    from spase_cache.strategies import checkpoint_positions

    print("=" * 60)
    print("CORRECTNESS TEST")
    print("=" * 60)

    model = make_model(cfg)
    config = model.config

    torch.manual_seed(cfg.test.seed)
    seq_len = cfg.test.seq_len
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

    linear_layers = _get_linear_layers(config)
    attn_layers = _get_attention_layers(config)

    hidden_baseline, cache_baseline = prefill_baseline(model, input_ids)

    # Capture checkpoints using logarithmic positions
    positions = checkpoint_positions(seq_len, tag="log")
    store = prefill_and_capture_at(model, input_ids, positions)

    hidden_restored, cache_restored = prefill_from_checkpoint(model, input_ids, store)

    max_state_diff = 0.0
    for li in linear_layers:
        s1 = cache_baseline.recurrent_states[li]
        s3 = cache_restored.recurrent_states[li]
        d_res = (s1 - s3).abs().max().item()
        max_state_diff = max(max_state_diff, d_res)
        print(f"Layer {li} GDN state diff: restored={d_res:.2e}")

    max_kv_diff = 0.0
    for li in attn_layers:
        dk = (cache_baseline.key_cache[li] - cache_restored.key_cache[li]).abs().max().item()
        dv = (cache_baseline.value_cache[li] - cache_restored.value_cache[li]).abs().max().item()
        max_kv_diff = max(max_kv_diff, dk, dv)
        print(f"Layer {li} attn KV diff: K={dk:.2e}  V={dv:.2e}")

    ckpt = store.best_checkpoint(seq_len)
    print(f"\nCheckpoints: {sorted(store.checkpoints.keys())}")
    print(f"Tokens skipped: {ckpt.position}/{seq_len} ({ckpt.position/seq_len*100:.0f}%)")
    print(f"Checkpoint memory: {store.memory_bytes()/1024:.1f} KB")

    tol = cfg.test.tolerance
    passed = max_state_diff < tol and max_kv_diff < tol
    print(f"\nPASS: {passed} (max_state_diff={max_state_diff:.2e}, max_kv_diff={max_kv_diff:.2e})")
    return passed
