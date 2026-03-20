"""Caching strategy for prefix checkpoint evaluation."""
import math

STRATEGIES = [
    "no_cache",
    "kv_only",
    "balanced_fix_blocksize",
    "balanced_fix_nblocks",
    "sqrt",
    "dyadic"
]

def balanced_positions(seq_len, block_size=None, n_blocks=None):
    assert (block_size is None) != (n_blocks is None)
    if n_blocks is not None:
        block_size = seq_len // n_blocks
    return list(range(block_size, seq_len + 1, block_size))

def sqrt_positions(seq_len):
    block_size = int(math.sqrt(seq_len))
    return balanced_positions(seq_len, block_size=block_size)

def diadic_positions(seq_len: int) -> list[int]:
    positions = []
    i = 0
    while (1 << i) <= seq_len:
        positions.append(1 << i)
        i += 1
    return positions

def checkpoint_positions(seq_len, *, tag, block_size=None, n_blocks=None, **_ignored):
    """Return list of positions where GDN checkpoints should be captured.

    Accepts the full strategy config dict as kwargs
    (e.g. ``checkpoint_positions(seq_len, **strategy)``).
    """
    if tag in ("no_cache", "kv_only"):
        return []
    if tag in ("block", "balanced_fix_blocksize"):
        return balanced_positions(seq_len, block_size=block_size)
    if tag == "balanced_fix_nblocks":
        return balanced_positions(seq_len, n_blocks=n_blocks)
    if tag == "sqrt":
        return sqrt_positions(seq_len)
    if tag in ("log", "dyadic", "diadic"):
        return diadic_positions(seq_len)
    raise ValueError(f"Unknown strategy: {tag}")
