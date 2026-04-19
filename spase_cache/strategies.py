"""Caching strategy for prefix checkpoint evaluation."""
import logging
import math
from functools import lru_cache
import numpy as np

log = logging.getLogger(__name__)

STRATEGIES = [
    "no_cache",
    "kv_only",
    "balanced_fix_blocksize",
    "balanced_fix_nblocks",
    "sqrt",
    "dyadic",
    "histogram_frozen",
    "histogram_periodic",
    "histogram_exp_decay",
]

def balanced_positions(seq_len, block_size=None, n_blocks=None):
    assert (block_size is None) != (n_blocks is None)
    if n_blocks is not None:
        block_size = max(1, seq_len // (n_blocks + 1))
        return list(range(block_size, seq_len + 1, block_size))[:n_blocks]
    return list(range(block_size, seq_len + 1, block_size))

def sqrt_positions(seq_len):
    block_size = int(math.sqrt(seq_len))
    return balanced_positions(seq_len, block_size=block_size)

def diadic_positions(seq_len: int, start_at) -> list[int]:
    positions = []
    i = 0
    while (1 << i) <= seq_len:
        if (1 << i) >= start_at:
            positions.append(1 << i)
        i += 1
    return positions

def checkpoint_positions(seq_len, *, type, block_size=None, n_blocks=None, start_at=0, skip=0,
                         histogram_tracker=None, save_last=False, kernel_block_size=64, **_ignored):
    """Return list of positions where GDN checkpoints should be captured.

    Dispatches on `type` (the strategy type), not `tag` (the unique ID).
    Accepts the full strategy config dict as kwargs
    (e.g. ``checkpoint_positions(seq_len, **strategy)``).
    """
    if type == "no_cache":
        return []

    positions = []
    if type == "kv_only":
        pass  # no GDN checkpoints by default
    elif seq_len < skip:
        pass
    elif type in ("block", "balanced_fix_blocksize"):
        positions = balanced_positions(seq_len, block_size=block_size)
    elif type == "balanced_fix_nblocks":
        positions = balanced_positions(seq_len, n_blocks=n_blocks)
    elif type == "sqrt":
        positions = sqrt_positions(seq_len)
    elif type in ("log", "dyadic", "diadic"):
        positions = diadic_positions(seq_len, start_at)
    elif type in ("histogram_frozen", "histogram_periodic", "histogram_exp_decay"):
        if histogram_tracker is None:
            raise ValueError(f"Strategy type {type} requires a histogram_tracker")
        positions = histogram_tracker.get_positions(seq_len)
    else:
        raise ValueError(f"Unknown strategy type: {type}")

    if save_last and seq_len not in positions:
        positions.append(seq_len)
    return sorted({(x // kernel_block_size) * kernel_block_size for x in positions})

# ---------------------------------------------------------------------------
# DP-optimal checkpoint placement (Corollary 1 from the paper)
# ---------------------------------------------------------------------------
def solve_dp(hist, budget):
    """Solve the DP forward pass (vectorized).

    Given overlap histogram `hist` (array of length N+1 where hist[t] = weight
    of overlap depth t, t=0..N), and checkpoint budget M, fills the DP table.

    Returns (all_back, N, M) where all_back[m][j] stores the optimal split
    point for m checkpoints covering positions 1..j. Returns None if no
    meaningful solution (empty hist or zero budget).
    """
    N = len(hist) - 1  # hist[0..N], positions 1..N
    M = min(budget, N)
    if N <= 0 or M <= 0:
        return None

    p = np.array(hist, dtype=np.float64)
    total = p.sum()
    if total < 1e-12:
        return None
    p = p / total

    cum_p = np.zeros(N + 2)
    cum_pt = np.zeros(N + 2)
    t_idx = np.arange(N + 1, dtype=np.float64)
    cum_p[1:] = np.cumsum(p)
    cum_pt[1:] = np.cumsum(p * t_idx)
    cpt_j = cum_pt[1:]
    cp_j = cum_p[1:]

    dp_prev = cpt_j.copy()
    all_back = [None]

    for m in range(1, M + 1):
        dp_curr = np.full(N + 1, np.inf)
        back_curr = np.zeros(N + 1, dtype=np.intp)
        dp_curr[0] = 0.0
        for j in range(1, N + 1):
            s_range = np.arange(1, j + 1)
            costs = (dp_prev[s_range - 1]
                     + cpt_j[j] - cum_pt[s_range]
                     - s_range * (cp_j[j] - cum_p[s_range]))
            idx = np.argmin(costs)
            dp_curr[j] = costs[idx]
            back_curr[j] = s_range[idx]
        dp_prev = dp_curr
        all_back.append(back_curr)

    return all_back, N, M


def backtrack(all_back, M, j):
    """Backtrack from dp[M, j] to recover checkpoint positions (bin indices)."""
    positions = []
    for m in range(M, 0, -1):
        if j <= 0:
            break
        s = int(all_back[m][j])
        positions.append(s)
        j = s - 1
    positions.sort()
    return positions


def _bin_index(value, bin_size):
    return value // bin_size


def _bin_to_pos(bin_idx, bin_size):
    """Convert bin index back to token position (use bin upper edge)."""
    return bin_idx * bin_size


def laplace_smoothing(counts, alpha):
    """Apply Laplace smoothing up to the last non-zero bin."""
    smoothed = counts.copy()
    nz = np.argwhere(smoothed > 0)
    if len(nz) == 0:
        return smoothed
    max_bin_pos = nz.max().item()
    smoothed[:max_bin_pos] += alpha
    return smoothed


class HistogramTracker:
    """Tracks overlap depth histogram for distribution-aware checkpointing.

    Three modes:
    - 'frozen': accumulate during warmup, solve DP once, freeze
    - 'periodic': re-solve DP every `replan_interval` observations
    - 'exp_decay': exponential decay weighting (gamma^{n-1-i}), re-solve every `replan_interval`

    bin_size > 1 bins overlap depths into groups, making the histogram smoother
    and the DP faster. Returned positions are multiples of bin_size.
    """

    def __init__(self, max_len, budget, mode='frozen', gamma=0.99,
                 replan_interval=100, alpha=1., bin_size=1,
                 adaptive_backtrack=True):
        self.max_len = max_len
        self.budget = budget
        self.mode = mode
        self.gamma = gamma
        self.alpha = alpha # laplace smoothing
        self.replan_interval = replan_interval
        self.bin_size = max(1, bin_size)
        self.adaptive_backtrack = adaptive_backtrack

        n_bins = max_len // self.bin_size + 1
        self.counts = np.zeros(n_bins, dtype=np.float64)
        self.n_obs = 0
        self._dp_result = None  # (all_back, N, M) from solve_dp
        self._fixed_positions = None  # used when adaptive_backtrack=False
        self._dirty = True
        self._n_solves = 0
        self.histogram_log = []  # list of {n_obs, counts}

    def observe(self, overlap_depth):
        """Record an observed overlap depth."""
        b = min(_bin_index(max(int(overlap_depth), 0), self.bin_size),
                len(self.counts) - 1)
        if self.mode == 'exp_decay':
            self.counts *= self.gamma
        self.counts[b] += 1.0
        self.n_obs += 1
        self._dirty = True

    def solve(self):
        """Solve DP on current (binned) histogram and store full DP tables."""
        self.histogram_log.append({
            "n_obs": self.n_obs,
            "counts": self.counts.copy(),
        })
        smoothed = laplace_smoothing(self.counts, self.alpha)
        self._dp_result = solve_dp(smoothed, self.budget)
        if self._dp_result is None:
            log.info("DP solve: degenerate histogram, using balanced fallback")
        else:
            _, N, M = self._dp_result
            positions = self._backtrack_bins(N)
            token_positions = [_bin_to_pos(b, self.bin_size) for b in positions]
            log.info("DP solved: %d ckpts, n_obs=%d, bin_size=%d, positions=%s",
                     len(positions), self.n_obs, self.bin_size, token_positions)
            if not self.adaptive_backtrack:
                self._fixed_positions = token_positions
        self._backtrack_bins.cache_clear()
        self._n_solves += 1
        self._dirty = False

    @lru_cache(maxsize=256)
    def _backtrack_bins(self, j_bin):
        """Backtrack from bin index j_bin, returning bin-level positions."""
        all_back, N, M = self._dp_result
        j = min(j_bin, N)
        return backtrack(all_back, M, j)

    def get_positions(self, seq_len):
        """Get checkpoint positions for a request of given length."""
        if self._dp_result is None:
            # No DP solution yet — use balanced fallback until first solve
            block = max(1, seq_len // (self.budget + 1))
            return list(range(block, seq_len + 1, block))[:self.budget]

        should_solve = False
        if self.mode == 'frozen':
            pass
        elif self.mode in ('periodic', 'exp_decay'):
            if self._dirty and self.n_obs % self.replan_interval == 0:
                should_solve = True

        if should_solve:
            self.solve()

        if not self.adaptive_backtrack and self._fixed_positions is not None:
            return [p for p in self._fixed_positions if 0 < p <= seq_len]

        j_bin = seq_len // self.bin_size
        bin_positions = self._backtrack_bins(j_bin)
        return [_bin_to_pos(b, self.bin_size) for b in bin_positions if 0 < b <= j_bin]

    def freeze(self):
        """Solve and freeze — no further updates to positions."""
        self.solve()
        self.mode = 'frozen'

