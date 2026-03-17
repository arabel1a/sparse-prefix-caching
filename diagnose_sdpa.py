"""Diagnose which SDPA backend is actually used and whether Q≠K OOMs."""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

device = "cuda"
dtype = torch.float16

# Mimic the model's attention shapes: 8 Q heads, 2 KV heads expanded to 8
n_heads = 8
head_dim = 256

def check(q_len, k_len, label):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q = torch.randn(1, n_heads, q_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, n_heads, k_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, n_heads, k_len, head_dim, device=device, dtype=dtype)
    mem_before = torch.cuda.memory_allocated()

    try:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(q_len == k_len), enable_gqa=False)
        peak = torch.cuda.max_memory_allocated()
        delta_mb = (peak - mem_before) / 1024**2
        qk_materialized_mb = n_heads * q_len * k_len * 4 / 1024**2  # float32
        print(f"  {label}: OK. peak delta={delta_mb:.0f}MB  (full QK^T would be {qk_materialized_mb:.0f}MB)")
    except Exception as e:
        print(f"  {label}: FAILED — {e}")
    del q, k, v
    torch.cuda.empty_cache()

print("=== xformers EFFICIENT_ATTENTION backend ===")
print("\n1) Q_len == K_len (like test_len.py single-pass):")
check(4096, 4096, "4K×4K")
check(16384, 16384, "16K×16K")

print("\n2) Q_len != K_len (like segmented prefill with KV cache):")
check(4096, 8192, "Q=4K, K=8K")
check(4096, 16384, "Q=4K, K=16K")
check(4096, 32768, "Q=4K, K=32K")
check(16384, 32768, "Q=16K, K=32K")

print("\n3) Q_len != K_len WITHOUT forcing xformers (auto backend selection):")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
q = torch.randn(1, n_heads, 4096, head_dim, device=device, dtype=dtype)
k = torch.randn(1, n_heads, 16384, head_dim, device=device, dtype=dtype)
v = torch.randn(1, n_heads, 16384, head_dim, device=device, dtype=dtype)
mem_before = torch.cuda.memory_allocated()
try:
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=False)
    peak = torch.cuda.max_memory_allocated()
    delta_mb = (peak - mem_before) / 1024**2
    print(f"  auto Q=4K,K=16K: OK. peak delta={delta_mb:.0f}MB")
except Exception as e:
    print(f"  auto Q=4K,K=16K: FAILED — {e}")

print("\n4) GPU memory summary:")
print(f"  Total: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
