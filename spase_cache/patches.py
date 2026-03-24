from contextlib import contextmanager
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DynamicCache,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextModel,
    apply_mask_to_padding_states,
)
import torch
import torch.nn.functional as F
import logging

# TODO: move my chunked prefill here

# ---------------------------------------------------------------------------
# Enforce non-eager attention.
# I need it because eager implementation allows only 8K context compared to
# 64K on my 8G VRAM. By default, torch silently falls back to eager if there
# are no suitable kernels, e.g. if using pre-Ampere device and GQA. There is
# another patch that reuses MHA kernels for GQA, they are designed to work
# together.
# ---------------------------------------------------------------------------
def enforce_efficient_attention():
    if torch.cuda.is_available():
        logging.warning("Disabled fallback to eager attention. Your model will crush if there are no suitable kernels")
        from torch.nn.attention import _sdpa_kernel, SDPBackend
        _sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
            set_priority=True,
        )



# ---------------------------------------------------------------------------
# Monkey-patch: expand K and V to match Q for GQA models. I need this because
# there are no efficient kernels for pre-Ampere devices
# ---------------------------------------------------------------------------
_original_sdpa = F.scaled_dot_product_attention

def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=None):
    bq, hq, sq, dq = query.shape
    bk, hk, sk, dk = key.shape
    if hq != hk:
        num_rep = hq // hk
        key = key.unsqueeze(2).expand(bk, hk, num_rep, sk, dk).reshape(bk, hq, sk, dk)
        value = value.unsqueeze(2).expand(bk, hk, num_rep, sk, dk).reshape(bk, hq, sk, dk)

    return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=None, enable_gqa=False)

def apply_sdpa_patch():
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        logging.warning("Monkey-patching SDPA for Group Query Attention reusing MHA kernel with pre-Ampre devices.")
        F.scaled_dot_product_attention = _patched_sdpa


# ---------------------------------------------------------------------------
# GDN chunk-state capture mechanism.
#
# When enabled, the patched GDN forward uses chunk_gated_delta_rule_with_states
# and extracts recurrent states at requested 64-token chunk boundaries.
# States are moved to CPU immediately to avoid GPU memory buildup.
# ---------------------------------------------------------------------------
GDN_CHUNK_SIZE = 64

_capture_enabled = False
_capture_positions = set()   # absolute positions (multiples of 64) to capture
_captured_states = {}        # layer_idx -> {position: (recurrent_state, conv_state) on CPU}


@contextmanager
def capture_gdn_states(positions):
    """Enable GDN chunk-state capture at given positions during forward pass.

    Positions are rounded down to GDN_CHUNK_SIZE boundaries.
    Yields a dict: {layer_idx: {position: state_tensor}}.
    """
    global _capture_enabled, _capture_positions, _captured_states
    _capture_enabled = True
    _capture_positions = set(
        (p // GDN_CHUNK_SIZE) * GDN_CHUNK_SIZE for p in positions if p > 0
    )
    _captured_states = {}
    try:
        yield _captured_states
    finally:
        _capture_enabled = False


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
    # Keep raw pre-activation QKV for conv_state capture (before conv overwrites it)
    raw_qkv = mixed_qkv if _capture_enabled else None
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

        if _capture_enabled:
            core_attn_out, last_recurrent_state, h = chunk_gated_delta_rule_with_states(
                query, key, value, g=g, beta=beta,
                initial_state=initial,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
            # Extract recurrent + conv states at requested chunk boundaries.
            # raw_qkv is the pre-activation QKV tensor (B, C, seq_len) captured
            # before the conv overwrites mixed_qkv. For continuation chunks,
            # prepend the previous conv_state to get a complete raw stream.
            offset = cache_position[0].item() if cache_position is not None else 0
            if _capture_positions:
                if self.layer_idx not in _captured_states:
                    _captured_states[self.layer_idx] = {}

                # Build full raw QKV stream for conv_state extraction
                if is_continuation and conv_state is not None:
                    full_raw = torch.cat([conv_state, raw_qkv], dim=-1)
                    raw_offset = offset - K  # conv_state covers [offset-K, offset)
                else:
                    full_raw = raw_qkv
                    raw_offset = offset

                for i in range(h.shape[1]):
                    abs_pos = offset + i * GDN_CHUNK_SIZE
                    if abs_pos in _capture_positions:
                        rec = h[:, i].cpu()
                        # conv_state at abs_pos = last K raw values ending at abs_pos
                        local_end = abs_pos - raw_offset
                        local_start = max(local_end - K, 0)
                        cs = full_raw[:, :, local_start:local_end].cpu()
                        if cs.shape[-1] < K:
                            cs = F.pad(cs, (K - cs.shape[-1], 0))
                        _captured_states[self.layer_idx][abs_pos] = (rec, cs)
            del h
        else:
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


def apply_patched_gdn_forward():
    """Replace GatedDeltaNet.forward with our patched version."""
    logging.warning("Monkey-patching Qwen3.5 GDN forward")
    Qwen3_5GatedDeltaNet.forward = _patched_gdn_forward


# ---------------------------------------------------------------------------
# Inference-only GDN forward that returns intermediate chunk states.
#
# The standard chunk_gated_delta_rule computes h[B, NT, H, K, V] internally
# (one state per 64-token chunk boundary) but discards it. This function
# calls the same Triton kernels directly, skipping the autograd wrapper,
# and returns h alongside the normal outputs.
#
# h[:, i, ...] is the recurrent state *before* processing chunk i, i.e.
# the state after tokens [0 .. i*64). So h[:, 0, ...] is the initial state
# and the final state equals h after the last chunk.
# ---------------------------------------------------------------------------

def chunk_gated_delta_rule_with_states(
    q, k, v, g, beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
):
    """Inference-only chunk_gated_delta_rule that also returns chunk-boundary states.

    Returns:
        o:           [B, T, H, V]  — same as chunk_gated_delta_rule
        final_state: [N, H, K, V]  — same as chunk_gated_delta_rule
        h:           [B, NT, H, K, V] — recurrent state at each chunk boundary (every 64 tokens)
    """
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_fwd_o
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.index import prepare_chunk_indices

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu)
        if cu_seqlens is not None else None
    )

    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)

    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g, beta=beta,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    o = chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    return o.to(q.dtype), final_state, h
