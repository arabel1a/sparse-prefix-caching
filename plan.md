# Plan: FLA Intermediate State Extraction Research

## Status: Pre-research phase

### 2026-03-24: Research — can FLA kernels return states at arbitrary positions?

**Goal:** Determine feasibility of modifying flash-linear-attention to output recurrent
states at user-specified positions during a single forward pass, eliminating the need
for separate re-capture runs in `benchmark_e2e.py`.

See `fla_intermediate_states_report.md` for the full report.

### 2026-03-24: Implement Approach A — expose chunk-boundary states

Added `chunk_gated_delta_rule_with_states()` to `spase_cache/patches.py`.
This is an inference-only function that calls the same Triton kernels as
`chunk_gated_delta_rule` but returns the intermediate `h[B, NT, H, K, V]`
tensor (one state per 64-token chunk boundary). No FLA source changes needed.

### 2026-03-25: Integrate single-pass capture into benchmarks

Wired `chunk_gated_delta_rule_with_states` through the full stack:

1. **patches.py**: Added `capture_gdn_states(positions)` context manager +
   module-level capture mechanism. Modified `_patched_gdn_forward` to use
   `chunk_gated_delta_rule_with_states` when capture is enabled, extracting
   states at requested 64-token boundaries and moving them to CPU immediately.

2. **utils.py**: Added `build_store_from_captures()` that builds a
   `PrefixCheckpointStore` from captured GDN chunk states + final KV cache.
   Merges prefix checkpoints from existing store on cache hits.

3. **benchmark_e2e.py**: `warmup_cache` and `simulate` now do single-pass
   prefill+capture. `capture_s` is 0 — states come free from the forward pass.
   Old two-pass pattern (timed prefill + untimed `prefill_and_capture_at`) eliminated.

4. **benchmark_single.py**: Same single-pass approach. Capture time now equals
   baseline prefill time (same forward pass), making the comparison fair.

Conv states are now captured exactly: raw_qkv (pre-activation QKV projection)
is saved before the conv1d overwrites it, and the last K values up to each
checkpoint position are sliced out. For continuation chunks, the previous
conv_state is prepended to cover positions near the chunk boundary.

### 2026-03-25: Add capture correctness test

Added `scripts/test_capture_correctness.py` — compares hidden states from three
forward modes: vanilla (unpatched) GDN, patched with capture off, patched with
capture on. All must match within tolerance. Also validates captured state shapes.

### 2026-04-24: Fix chunk_gated_delta_rule_with_states to match FLA semantics

`chunk_gated_delta_rule_with_states` in patches.py was diverging from FLA's real
`chunk_gated_delta_rule_fwd`, producing errors and wrong outputs. Fixes:

- Force `.contiguous()` on q, k, v, g, beta, initial_state before calling
  downstream kernels (FLA's autograd path does this via `@input_guard`; my
  manual wrapper bypassed it, which caused `l2norm_fwd` to fail with a stride
  error when the model's `repeat_interleave` produced non-contiguous q/k).
- Pass `use_exp2=True` (the FLA default) through to `chunk_local_cumsum`,
  `chunk_gated_delta_rule_fwd_intra`, `chunk_gated_delta_rule_fwd_h`, and
  `chunk_fwd_o`, plus `scale=RCP_LN2` on the cumsum. Without this the gate
  values were on the wrong log base.

Result: `python scripts/test_capture_correctness.py -cn quality_clean device=cuda`
passes inside the sparse-prefix-caching container. Max diff between vanilla and
captured forward = 5.76e-4 (under the 1e-3 tolerance).

### 2026-04-24: Fix conv1d mismatch in _patched_gdn_forward

Test failed at longer seq_len (1.99e-3 at seq=512). Root cause: vanilla
`Qwen3_5GatedDeltaNet.forward` uses `self.causal_conv1d_fn` (Mamba's fused CUDA
kernel) when available, but the patched non-continuation branch in
`_patched_gdn_forward` was hardcoded to `F.silu(self.conv1d(...))`. The two are
numerically distinct; per-token conv differences fed the recurrence and grew
with T (linear-ish: 5.76e-4 @ 100 → 1.99e-3 @ 512). Confirmed
`causal_conv1d_fn` is installed inside the container.

Fix: mirror vanilla's branch in `patches.py` — call `self.causal_conv1d_fn` on
the non-continuation path, fall back to `F.silu(self.conv1d(...))` only when
it's unavailable. After fix, max_diff = 0.00e+00 at seq=512 and seq=2048.
