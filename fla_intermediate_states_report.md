# Can FLA Return Recurrent States at Arbitrary Positions?

## Context

You're running Qwen3.5 with GatedDeltaNet (GDN) layers via FLA. Currently in
`benchmark_e2e.py`, prefill timing and checkpoint capture are separate passes because
`transformers` only returns the final recurrent state. You want FLA to return states at
chosen positions in a single pass.

## TL;DR

**Yes, it's doable.** FLA already has all the machinery — the chunk-mode kernel
(`chunk_fwd_kernel_h`) computes and stores intermediate states at every chunk boundary
(every 64 tokens). KDA even has a `return_intermediate_states` parameter as precedent.
The gap is: (1) these chunk-boundary states are internal, not returned to the user, and
(2) you need states at *arbitrary* positions, not just chunk boundaries.

Three approaches, ranked by effort:

| Approach | Effort | Perf overhead | Precision |
|----------|--------|---------------|-----------|
| A. Just expose chunk-boundary states | ~20 LOC | ~0 | Chunk-aligned only (every 64 tok) |
| B. Fused recurrent with state snapshots | ~80 LOC Triton | Small (extra stores) | Exact positions |
| C. Chunk states + intra-chunk fused recurrent fixup | ~120 LOC | Small | Exact positions |

---

## How FLA Kernels Work (the relevant parts)

### Two modes of operation

1. **Chunk mode** (`chunk_gla`, `chunk_gated_delta_rule`, etc.)
   - Splits sequence into BT=64-token chunks
   - Computes inter-chunk state `h[B, NS, H, K, V]` where NS = ceil(T/64)
   - State at chunk boundary `i` = recurrent state after processing tokens [0..i*64)
   - These states are used internally by the output kernel, then discarded
   - Only `ht` (final state) is returned if `output_final_state=True`

2. **Fused recurrent mode** (`fused_recurrent_fwd_kernel`)
   - Token-by-token loop: `for _ in range(T): h = h*decay + k*v`
   - State `b_h` lives in registers, never written to memory except final
   - Used for short sequences (T <= 64) and generation

### Where states live in chunk mode

In `fla/ops/common/chunk_h.py`, the kernel `chunk_fwd_kernel_h`:
```
for i_t in range(NT):           # iterate over 64-token chunks
    if i_t % NTS == 0:          # at split boundaries (default: every chunk)
        tl.store(p_h, b_h)     # write state to h[B, NS, H, K, V]
    # ... process chunk, update b_h ...
```

With default `split_size = chunk_size = 64`, **every** chunk boundary state is
materialized in the `h` tensor. This tensor is allocated, filled, used by the output
kernel, and then freed — never returned.

### KDA precedent: `return_intermediate_states`

`fla/ops/kda/chunk.py` already has:
```python
return_intermediate_states: bool = False
```
When True (inference only), it skips deleting `h` and returns it as a third output.
This is exactly what you need, but only implemented for KDA, not for
`chunk_gated_delta_rule` which Qwen3.5 GDN uses.

---

## Approach A: Expose Chunk-Boundary States (Easiest)

**What:** Add `return_intermediate_states` to `chunk_gated_delta_rule`, following the
KDA pattern. The `h` tensor is already computed — just don't delete it.

**Where to modify:**
- `fla/ops/gated_delta_rule/chunk.py` (or wherever `chunk_gated_delta_rule` lives)
- Add `return_intermediate_states=False` parameter
- When True, return `h` alongside `(o, final_state)`

**Precision:** States at every 64-token boundary. Your checkpoint positions would be
rounded to multiples of 64. For a 4096-token sequence, you get 64 possible checkpoint
positions.

**Overhead:** Zero additional compute. The `h` tensor is already computed. You just
keep the reference instead of freeing it. Memory cost: `B * NS * H * K * V * dtype`
which is already allocated during the forward pass anyway.

**Limitation:** If your optimal checkpoint positions don't align to 64-token boundaries,
you lose some granularity. But given that your strategies already work with discrete
block boundaries, this might be perfectly fine.

**Code sketch (Python wrapper only — no Triton changes):**
```python
# In chunk_gated_delta_rule forward:
h, v_new, final_state = chunk_gated_delta_rule_fwd_h(...)
# ... compute output ...
if not return_intermediate_states:
    h = None  # existing behavior: free memory
return o, final_state, h  # h is [B, NS, H, K, V], NS = ceil(T/64)
```

---

## Approach B: Fused Recurrent with Snapshot Positions (Most Precise)

**What:** Modify `fused_recurrent_fwd_kernel` to accept a sorted list of positions and
write `b_h` to an output buffer when the loop counter hits one of those positions.

**Where to modify:** `fla/ops/common/fused_recurrent.py`

**Triton changes:**
```python
# New parameter: positions tensor of shape [max_checkpoints], padded with -1
# New output: h_snapshots of shape [N, max_checkpoints, H, K, V]

for t in range(0, T):
    # ... existing decay + update logic ...
    if t == positions[next_checkpoint_idx]:
        tl.store(p_snapshot, b_h)
        next_checkpoint_idx += 1
```

**Complication on SM75:** The fused recurrent kernel loops token-by-token. On a 2080
Super (SM75, Turing), you have:
- 48 KB shared memory per SM (vs 164 KB on Ampere+)
- No async copies (cp.async)
- `check_shared_mem()` returns False on SM75, so BK/BV lists use [16, 32] instead
  of [32, 64] — smaller tiles, more kernel launches

The token loop means this kernel is **slow for long sequences** (that's why chunk mode
exists). However, for *inference-only* state capture where you just need the states (not
training gradients), this is actually clean.

**Overhead:** One extra `tl.store` per checkpoint position per head. Negligible compared
to the O(T) loop.

**Precision:** Exact token-level positions. You get the state after processing token t,
for any t you want.

---

## Approach C: Chunk States + Intra-Chunk Fixup (Best of Both)

**What:** Use chunk-boundary states from Approach A, then for the few positions that
fall mid-chunk, run a short fused-recurrent segment (up to 64 tokens) starting from the
nearest preceding chunk state.

**Implementation:**
1. Get `h[B, NS, H, K, V]` from the chunk kernel (Approach A)
2. For each desired position `p`:
   - `chunk_idx = p // 64`
   - `offset = p % 64`
   - If `offset == 0`: state is `h[:, chunk_idx, ...]`, done
   - Else: run fused recurrent from `h[:, chunk_idx, ...]` for `offset` tokens

**Overhead:** One short fused-recurrent call per non-aligned checkpoint. If you have
N_ckpt checkpoints and they don't align, worst case N_ckpt * 63 extra tokens processed.
In practice, if you align your strategy to 64-token boundaries, overhead is zero.

**This is the cleanest production approach** because:
- Chunk kernel runs at full speed (no modifications to the hot path)
- Fixup is a small post-processing step
- You can choose to skip fixup and just round positions to chunk boundaries

---

## SM75 (2080 Super) Considerations

- FLA works on SM75 but with reduced tile sizes (`check_shared_mem()` → False → BK,BV
  in [16, 32] instead of [32, 64])
- Triton on SM75: no TMA, no wgmma, no cp.async — but basic `tl.load`/`tl.store`/
  `tl.dot` all work fine
- The chunk kernel should work as-is. The fused recurrent kernel definitely works
  (simpler memory pattern)
- Main performance difference: smaller tiles mean more iterations in the K,V dimension
  loops, but the state-capture overhead is independent of tile size

---

## Recommendation

**Start with Approach A.** It requires zero Triton changes — just plumb
`return_intermediate_states` through the Python wrapper of `chunk_gated_delta_rule`,
mirroring what KDA already does. You get states every 64 tokens with zero overhead.

Then in your `benchmark_e2e.py`, instead of the separate `prefill_and_capture_at` call,
you extract states directly from the `h` tensor returned by the patched forward pass.
Your checkpoint strategies would quantize positions to multiples of 64, which is likely
fine given typical sequence lengths of 2K-8K.

If 64-token granularity isn't enough, upgrade to Approach C — use the chunk states as
anchors and run tiny fused-recurrent segments for the few positions that need sub-chunk
precision.

---

## Key Files Reference

| File | What it does |
|------|-------------|
| `fla/ops/common/chunk_h.py:32` | `chunk_fwd_kernel_h` — computes and stores inter-chunk states |
| `fla/ops/common/chunk_h.py:269` | `chunk_fwd_h()` — Python wrapper, allocates `h[B,NS,H,K,V]` |
| `fla/ops/common/fused_recurrent.py:26` | `fused_recurrent_fwd_kernel` — token-by-token loop |
| `fla/ops/kda/chunk.py:37` | KDA's `return_intermediate_states` — existing precedent |
| `fla/ops/kda/chunk_fwd.py:127` | Where `h` is conditionally kept vs deleted |
| `fla/ops/gla/chunk.py:1290` | `chunk_gla()` — GLA chunk entry point (similar pattern to GDN) |
| Your `spase_cache/patches.py:135` | Where you call `chunk_gated_delta_rule` |
