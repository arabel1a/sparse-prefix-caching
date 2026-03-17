"""
Prototype: 2^i recurrent state prefix caching for Qwen3.5 hybrid models.

Demonstrates the core idea: cache GDN recurrent states at power-of-2 positions
during prefill, then reuse the nearest checkpoint on subsequent requests with
the same prefix, skipping computation for the cached portion.

Requires monkey-patching Qwen3_5GatedDeltaNet.forward() because the upstream HF
implementation always passes initial_state=None during prefill (seq_len > 1).
Our patch passes the cached recurrent_state when available.
"""

import os

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DynamicCache,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextModel,
    apply_mask_to_padding_states,
)


# ---------------------------------------------------------------------------
# Monkey-patch: pass initial_state during chunked prefill when cache exists
# ---------------------------------------------------------------------------
def _patched_gdn_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: Qwen3_5DynamicCache | None = None,
    cache_position: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    K = self.conv_kernel_size

    use_precomputed_states = (
        cache_params is not None
        and cache_params.has_previous_state
        and seq_len == 1
        and cache_position is not None
    )

    recurrent_state = None
    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv, conv_state,
            self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation,
        )
    else:
        is_continuation = (
            cache_params is not None
            and conv_state is not None
            and cache_position is not None
            and cache_position[0] > 0
        )

        if is_continuation:
            # PATCH: use saved conv_state as left context instead of zero-padding.
            # conv_state has shape (B, C, K) — last K pre-activation values.
            # We need last K-1 as context for the conv.
            context = conv_state[:, :, -(K - 1):]
            combined = torch.cat([context, mixed_qkv], dim=-1)  # (B, C, K-1 + seq_len)
            # Run depthwise conv1d with NO padding, output len = K-1+seq_len-K+1 = seq_len
            conv_out = F.conv1d(combined, self.conv1d.weight, self.conv1d.bias, groups=self.conv_dim)
            mixed_qkv_post = F.silu(conv_out)
            # Save new conv state: last K raw pre-activation values
            all_raw = torch.cat([conv_state, self.in_proj_qkv(hidden_states).transpose(1, 2)], dim=-1)
            cache_params.conv_states[self.layer_idx] = all_raw[:, :, -K:]
            mixed_qkv = mixed_qkv_post
        else:
            if cache_params is not None:
                # Save last K pre-activation values as conv state
                cache_params.conv_states[self.layer_idx] = F.pad(mixed_qkv, (K - mixed_qkv.shape[-1], 0))
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if not use_precomputed_states:
        # PATCH: pass recurrent_state as initial_state when continuing from checkpoint
        initial = recurrent_state if (cache_params is not None and recurrent_state is not None) else None
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=initial,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def apply_patch():
    """Replace GatedDeltaNet.forward with our patched version."""
    Qwen3_5GatedDeltaNet.forward = _patched_gdn_forward


# ---------------------------------------------------------------------------
# Checkpoint store
# ---------------------------------------------------------------------------
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
    """Stores recurrent state checkpoints at 2^i positions for a token prefix."""
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

    def memory_bytes(self) -> int:
        total = 0
        for ckpt in self.checkpoints.values():
            for t in ckpt.recurrent_states.values():
                total += t.nelement() * t.element_size()
            for t in ckpt.conv_states.values():
                total += t.nelement() * t.element_size()
        for t in self.kv_cache_keys.values():
            total += t.nelement() * t.element_size()
        for t in self.kv_cache_values.values():
            total += t.nelement() * t.element_size()
        return total

    def to(self, device):
        if self.prefix_tokens is not None:
            self.prefix_tokens = self.prefix_tokens.to(device)
        for ckpt in self.checkpoints.values():
            ckpt.to(device)
        self.kv_cache_keys = {k: v.to(device) for k, v in self.kv_cache_keys.items()}
        self.kv_cache_values = {k: v.to(device) for k, v in self.kv_cache_values.items()}
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _model_device(model: Qwen3_5TextModel) -> torch.device:
    return next(model.parameters()).device


def _get_linear_layers(config: Qwen3_5TextConfig) -> list[int]:
    return [i for i, lt in enumerate(config.layer_types) if lt == "linear_attention"]


def _get_attention_layers(config: Qwen3_5TextConfig) -> list[int]:
    return [i for i, lt in enumerate(config.layer_types) if lt == "full_attention"]


def _checkpoint_positions(seq_len: int) -> list[int]:
    positions = []
    i = 0
    while (1 << i) <= seq_len:
        positions.append(1 << i)
        i += 1
    return positions


def _sync_device(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Prefill with checkpoint capture
# ---------------------------------------------------------------------------
def prefill_and_capture(
    model: Qwen3_5TextModel,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, Qwen3_5DynamicCache, PrefixCheckpointStore]:
    """Run full prefill in segments, capturing GDN states at 2^i positions."""
    device = _model_device(model)
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    config = model.config
    linear_layers = _get_linear_layers(config)
    attn_layers = _get_attention_layers(config)

    ckpt_positions = _checkpoint_positions(seq_len)
    store = PrefixCheckpointStore(prefix_tokens=input_ids.clone())

    boundaries = [0] + ckpt_positions
    if boundaries[-1] < seq_len:
        boundaries.append(seq_len)

    cache = Qwen3_5DynamicCache(config=config)

    for seg_idx in range(len(boundaries) - 1):
        start = boundaries[seg_idx]
        end = boundaries[seg_idx + 1]
        segment_ids = input_ids[:, start:end]
        cache_position = torch.arange(start, end, device=device)

        with torch.no_grad():
            output = model(
                input_ids=segment_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
        cache = output.past_key_values

        if end in ckpt_positions:
            ckpt = RecurrentCheckpoint(
                position=end,
                recurrent_states={
                    li: cache.recurrent_states[li].clone()
                    for li in linear_layers
                    if cache.recurrent_states[li] is not None
                },
                conv_states={
                    li: cache.conv_states[li].clone()
                    for li in linear_layers
                    if cache.conv_states[li] is not None
                },
            )
            store.checkpoints[end] = ckpt

    for li in attn_layers:
        if cache.key_cache[li] is not None:
            store.kv_cache_keys[li] = cache.key_cache[li].clone()
            store.kv_cache_values[li] = cache.value_cache[li].clone()

    return output.last_hidden_state, cache, store


# ---------------------------------------------------------------------------
# Prefill from checkpoint
# ---------------------------------------------------------------------------
def prefill_from_checkpoint(
    model: Qwen3_5TextModel,
    input_ids: torch.Tensor,
    store: PrefixCheckpointStore,
) -> tuple[torch.Tensor, Qwen3_5DynamicCache]:
    """
        Resume prefill from best checkpoint, computing only the tail tokens.
    """
    device = _model_device(model)
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    config = model.config
    linear_layers = _get_linear_layers(config)
    attn_layers = _get_attention_layers(config)

    ckpt = store.best_checkpoint(seq_len)

    if ckpt is None:
        # No GDN checkpoints — must recompute all GDN layers from scratch.
        # KV cache cannot reduce HF prefill without a custom attention kernel
        # (HF appends new KV on each forward, so pre-filling KV + passing all
        # tokens doubles the sequence length seen by attention).
        return prefill_baseline(model, input_ids)

    resume_pos = ckpt.position

    if resume_pos >= seq_len:
        # Full cache hit — no compute needed, return cached state
        cache = Qwen3_5DynamicCache(config=config)
        for li in linear_layers:
            if li in ckpt.recurrent_states:
                cache.recurrent_states[li] = ckpt.recurrent_states[li].clone()
            if li in ckpt.conv_states:
                cache.conv_states[li] = ckpt.conv_states[li].clone()
        for li in attn_layers:
            if li in store.kv_cache_keys:
                cache.key_cache[li] = store.kv_cache_keys[li][:, :, :seq_len, :].clone()
                cache.value_cache[li] = store.kv_cache_values[li][:, :, :seq_len, :].clone()
        dummy = torch.zeros(1, 1, config.hidden_size, device=device)
        return dummy, cache

    # Verify token match
    assert torch.equal(store.prefix_tokens[0, :resume_pos], input_ids[0, :resume_pos])

    # Build cache with restored states
    cache = Qwen3_5DynamicCache(config=config)
    for li in linear_layers:
        if li in ckpt.recurrent_states:
            cache.recurrent_states[li] = ckpt.recurrent_states[li].clone()
        if li in ckpt.conv_states:
            cache.conv_states[li] = ckpt.conv_states[li].clone()
    for li in attn_layers:
        if li in store.kv_cache_keys:
            cache.key_cache[li] = store.kv_cache_keys[li][:, :, :resume_pos, :].clone()
            cache.value_cache[li] = store.kv_cache_values[li][:, :, :resume_pos, :].clone()

    remaining_ids = input_ids[:, resume_pos:]
    cache_position = torch.arange(resume_pos, seq_len, device=device)

    with torch.no_grad():
        output = model(
            input_ids=remaining_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
    return output.last_hidden_state, output.past_key_values


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def prefill_baseline(
    model: Qwen3_5TextModel,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, Qwen3_5DynamicCache]:
    device = _model_device(model)
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    cache = Qwen3_5DynamicCache(config=model.config)
    cache_position = torch.arange(0, seq_len, device=device)
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
    return output.last_hidden_state, output.past_key_values


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------
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
    fa_failed=False
    try:
        import flash_attn
        model_config.update("attn_implementation", "flash_attention_2")
    except Exception:
        print("failed to flash_attention_2")
        fa_failed=True

    # --- construct model (DO NOT pass attn_implementation here) ---
    model = Qwen3_5TextModel(model_config).eval()

    # --- fallback if flash attention is not actually usable ---
    if fa_failed and cfg.device == "cuda":
        print("using xformers")
        from torch.nn.attention import sdpa_kernel, SDPBackend
        xformers_backend = SDPBackend.EFFICIENT_ATTENTION
        
        def wrapped_forward(orig_forward):
            def new_forward(*args, **kwargs):
                with sdpa_kernel(xformers_backend):
                    return orig_forward(*args, **kwargs)
            return new_forward

        model.forward = wrapped_forward(model.forward)


    model.to(cfg.device)
    return model
# ---------------------------------------------------------------------------
# Correctness test - differences in attention outputs
# ---------------------------------------------------------------------------
def test_correctness(cfg):
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
    hidden_capture, cache_capture, store = prefill_and_capture(model, input_ids)
    hidden_restored, cache_restored = prefill_from_checkpoint(model, input_ids, store)

    max_state_diff = 0.0
    for li in linear_layers:
        s1 = cache_baseline.recurrent_states[li]
        s2 = cache_capture.recurrent_states[li]
        s3 = cache_restored.recurrent_states[li]
        d_cap = (s1 - s2).abs().max().item()
        d_res = (s1 - s3).abs().max().item()
        max_state_diff = max(max_state_diff, d_cap, d_res)
        print(f"Layer {li} GDN state diff: capture={d_cap:.2e}  restored={d_res:.2e}")

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


# ---------------------------------------------------------------------------
# Profiled benchmark
# ---------------------------------------------------------------------------
def profile_benchmark(cfg):
    print("\n" + "=" * 60)
    print("PROFILED BENCHMARK")
    print("=" * 60)

    model = make_model(cfg)
    config = model.config
    dev = _model_device(model)

    seq_len = cfg.profile.seq_len
    torch.manual_seed(cfg.profile.seed)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

    # Warm up + capture
    prefill_baseline(model, input_ids)
    _, _, store = prefill_and_capture(model, input_ids)
    ckpt = store.best_checkpoint(seq_len)
    print(f"seq_len={seq_len}, checkpoint at {ckpt.position}, "
          f"recompute {seq_len - ckpt.position} tokens")

    trace_dir = Path(cfg.profile.trace_dir)
    trace_dir.mkdir(exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    n_warmup = cfg.profile.n_warmup
    n_active = cfg.profile.n_active

    # --- Baseline profile ---
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=0, warmup=n_warmup, active=n_active, repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir / "baseline")),
        record_shapes=cfg.profile.record_shapes,
        profile_memory=cfg.profile.profile_memory,
        with_stack=cfg.profile.with_stack,
    ) as prof:
        for _ in range(n_warmup + n_active):
            prefill_baseline(model, input_ids)
            _sync_device(dev)
            prof.step()

    print("\n--- Baseline top ops (CPU time) ---")
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=cfg.profile.profiler_row_limit,
    ))

    # --- Cached profile ---
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=0, warmup=n_warmup, active=n_active, repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir / "cached")),
        record_shapes=cfg.profile.record_shapes,
        profile_memory=cfg.profile.profile_memory,
        with_stack=cfg.profile.with_stack,
    ) as prof:
        for _ in range(n_warmup + n_active):
            prefill_from_checkpoint(model, input_ids, store)
            _sync_device(dev)
            prof.step()

    print("\n--- Cached top ops (CPU time) ---")
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=cfg.profile.profiler_row_limit,
    ))

    print(f"\nTraces saved to {trace_dir.resolve()}/")
    print("View with: tensorboard --logdir traces/")

