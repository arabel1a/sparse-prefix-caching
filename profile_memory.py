"""Profile peak VRAM usage of Qwen3.5-0.8B forward pass.

Compares eager vs sdpa attention, and shows theoretical memory breakdown.
Run on CUDA: python profile_memory.py
"""
import torch
import gc

def mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

def reset_peak():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def profile_forward(seq_len, attn_impl="eager", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoConfig

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

    # Count attention layers
    n_attn = sum(1 for lt in config.layer_types if lt == "full_attention")
    n_linear = sum(1 for lt in config.layer_types if lt == "linear_attention")
    print(f"\nArchitecture: {config.num_hidden_layers} layers "
          f"({n_linear} GDN + {n_attn} full_attention)")
    print(f"  num_attention_heads={config.num_attention_heads}, "
          f"num_kv_heads={config.num_key_value_heads}, head_dim={config.head_dim}")

    config._attn_implementation = attn_impl
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to("cuda").eval()

    after_load = mem_mb()
    print(f"  Model weights: {after_load:.0f} MB")

    reset_peak()

    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device="cuda")
    with torch.no_grad():
        output = model(input_ids)
    del output

    peak = mem_mb()
    print(f"  seq_len={seq_len}, attn={attn_impl}, dtype={dtype}")
    print(f"  Peak activation memory: {peak:.0f} MB")

    # Theoretical breakdown
    N = seq_len
    n_heads = config.num_attention_heads
    bpe = 4  # QK^T is always fp32 (softmax upcast)
    qk_per_layer = n_heads * N * N * bpe / 1024**2
    mask = N * N * bpe / 1024**2
    print(f"\n  --- Theoretical (eager) ---")
    print(f"  QK^T per attn layer: {qk_per_layer:.0f} MB (fp32)")
    print(f"  Softmax output (same shape): {qk_per_layer:.0f} MB")
    print(f"  4D causal mask: {mask:.0f} MB")
    print(f"  Peak from attention alone: ~{qk_per_layer * 2 + mask:.0f} MB "
          f"(QK^T + softmax + mask, 1 layer at a time)")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return peak


if __name__ == "__main__":
    for seq_len in [2048, 4096, 8192, 16384]:
        print(f"\n{'='*60}")
        print(f"Seq len: {seq_len}")
        print(f"{'='*60}")

        for attn in ["eager", "sdpa"]:
            profile_forward(seq_len, attn_impl=attn)
