"""Verify that the patched GDN forward + capture produces identical hiddens.

Compares three modes:
  1. Vanilla (unpatched) Qwen3.5 GDN forward
  2. Patched forward with capture disabled (default path)
  3. Patched forward with capture enabled (chunk_gated_delta_rule_with_states)

All three must produce the same final hidden states.

Run:
    uv run python scripts/test_capture_correctness.py device=cuda
    uv run python scripts/test_capture_correctness.py device=mps
"""
import sys
import torch
import hydra
from omegaconf import DictConfig

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DynamicCache,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextModel,
)

# Save the ORIGINAL forward before spase_cache patches it on import
_original_gdn_forward = Qwen3_5GatedDeltaNet.forward

# Now import spase_cache (applies monkey-patches)
import spase_cache  # noqa: E402
from spase_cache.patches import capture_gdn_states, _patched_gdn_forward
from spase_cache.utils import make_model, prefill_baseline


def run_forward(model, input_ids):
    """Run a full prefill and return the final hidden states."""
    cache = Qwen3_5DynamicCache(config=model.config)
    out = model(input_ids=input_ids, past_key_values=cache)
    return out.last_hidden_state


@hydra.main(config_path=r'../conf', config_name='config', version_base="1.3")
def main(cfg: DictConfig):
    dev = cfg.device
    tol = cfg.test.tolerance
    seq_len = cfg.test.seq_len
    print(f"device={dev}  seq_len={seq_len}  tol={tol}")

    model = make_model(cfg)
    vocab_size = model.config.vocab_size

    torch.manual_seed(cfg.seed)
    input_ids = torch.randint(0, vocab_size, (1, seq_len)).to(dev)

    # --- Mode 1: vanilla (unpatched) forward ---
    Qwen3_5GatedDeltaNet.forward = _original_gdn_forward
    with torch.no_grad():
        h_vanilla = run_forward(model, input_ids).cpu()

    # --- Mode 2: patched forward, capture disabled ---
    Qwen3_5GatedDeltaNet.forward = _patched_gdn_forward
    with torch.no_grad():
        h_patched = run_forward(model, input_ids).cpu()

    # --- Mode 3: patched forward, capture enabled ---
    positions = list(range(0, seq_len, 64))  # every chunk boundary
    with torch.no_grad():
        with capture_gdn_states(positions) as captured:
            h_capture = run_forward(model, input_ids).cpu()

    # --- Compare ---
    diff_patched = (h_vanilla - h_patched).abs().max().item()
    diff_capture = (h_vanilla - h_capture).abs().max().item()

    print(f"vanilla vs patched (no capture):  max_diff = {diff_patched:.2e}")
    print(f"vanilla vs patched (with capture): max_diff = {diff_capture:.2e}")

    ok = True
    if diff_patched > tol:
        print(f"FAIL: patched (no capture) differs by {diff_patched:.2e} > {tol:.2e}")
        ok = False
    if diff_capture > tol:
        print(f"FAIL: patched (with capture) differs by {diff_capture:.2e} > {tol:.2e}")
        ok = False

    # Also verify captured states are non-empty
    if not captured:
        print("FAIL: no states were captured")
        ok = False
    else:
        n_layers = len(captured)
        n_positions = len(next(iter(captured.values())))
        print(f"Captured states: {n_layers} layers x {n_positions} positions")
        # Verify each captured state has (recurrent, conv) tuple
        for layer_idx, layer_states in captured.items():
            for pos, (rec, conv) in layer_states.items():
                assert rec.shape[-2] == model.config.linear_key_head_dim, \
                    f"Bad recurrent shape at layer {layer_idx} pos {pos}: {rec.shape}"
                assert conv.shape[-1] == model.config.linear_conv_kernel_dim, \
                    f"Bad conv shape at layer {layer_idx} pos {pos}: {conv.shape}"

    if ok:
        print("PASS")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
