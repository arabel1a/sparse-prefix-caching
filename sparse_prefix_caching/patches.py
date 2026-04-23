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


def apply_patched_gdn_forward():
    """Replace GatedDeltaNet.forward with our patched version."""
    logging.warning("Monkey-patching Qwen3.5 GDN forward")
    Qwen3_5GatedDeltaNet.forward = _patched_gdn_forward
