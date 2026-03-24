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
