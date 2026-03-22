"""Caching strategy for prefix checkpoint evaluation."""
import logging
import math
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
        block_size = seq_len // n_blocks
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


# ---------------------------------------------------------------------------
# DP-optimal checkpoint placement (Corollary 1 from the paper)
# ---------------------------------------------------------------------------
def solve_dp(hist, budget):
    """Solve the optimal checkpoint placement DP (vectorized).

    Given overlap histogram `hist` (array of length N+1 where hist[t] = weight
    of overlap depth t, t=0..N), and checkpoint budget M, returns list of optimal
    checkpoint positions (0-indexed bin indices).

    DP recurrence (paper Corollary 1):
        dp[0, j] = sum_{t=1}^{j} p_t * t
        dp[m, j] = min_{1<=s<=j} (dp[m-1, s-1] + w(s, j))
    where w(s, j) = sum_{t=s}^{j} p_t * (t - s)

    Returns sorted list of checkpoint positions (1-indexed into bins).
    """
    N = len(hist) - 1  # hist[0..N], positions 1..N
    M = min(budget, N)  # can't place more checkpoints than positions
    if N <= 0 or M <= 0:
        return []

    p = np.array(hist, dtype=np.float64)
    total = p.sum()
    if total < 1e-12:
        block = max(1, N // (M + 1))
        return list(range(block, N + 1, block))[:M]
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

    positions = []
    j = N
    for m in range(M, 0, -1):
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
                 replan_interval=100, bin_size=1, exclude_full_hits=True):
        self.max_len = max_len
        self.budget = budget
        self.mode = mode
        self.gamma = gamma
        self.replan_interval = replan_interval
        self.bin_size = max(1, bin_size)
        self.exclude_full_hits = exclude_full_hits

        n_bins = max_len // self.bin_size + 1
        self.counts = np.zeros(n_bins, dtype=np.float64)
        self.n_obs = 0
        self.n_skipped_full = 0
        self._positions = None  # cached DP solution (in token positions)
        self._dirty = True

    def observe(self, overlap_depth, is_full_hit=False):
        """Record an observed overlap depth.

        If exclude_full_hits is set and is_full_hit is True, the observation
        is skipped — full hits are already handled by the seq_len checkpoint
        and would bias the DP toward high positions.
        """
        if self.exclude_full_hits and is_full_hit:
            self.n_skipped_full += 1
            return
        b = min(_bin_index(max(int(overlap_depth), 0), self.bin_size),
                len(self.counts) - 1)
        if self.mode == 'exp_decay':
            self.counts *= self.gamma
        self.counts[b] += 1.0
        self.n_obs += 1
        self._dirty = True

    def solve(self):
        """Solve DP on current (binned) histogram and cache the result."""
        bin_positions = solve_dp(self.counts, self.budget)
        self._positions = [_bin_to_pos(b, self.bin_size) for b in bin_positions]
        self._dirty = False
        log.info("DP solved: %d ckpts, n_obs=%d (skipped %d full hits), bin_size=%d, positions=%s",
                 len(self._positions), self.n_obs, self.n_skipped_full,
                 self.bin_size, self._positions)

    def get_positions(self, seq_len):
        """Get checkpoint positions for a request of given length.

        For frozen mode, defers DP solve until freeze() is called.
        Note: save_last is handled by checkpoint_positions(), not here.
        """
        should_solve = False
        if self._positions is None:
            if self.mode == 'frozen':
                return []  # no DP positions yet; save_last handled by caller
            should_solve = True
        elif self.mode in ('periodic', 'exp_decay'):
            if self._dirty and self.n_obs % self.replan_interval == 0:
                should_solve = True

        if should_solve:
            self.solve()

        return [p for p in self._positions if 0 < p <= seq_len]

    def freeze(self):
        """Solve and freeze — no further updates to positions."""
        self.solve()
        self.mode = 'frozen'


def checkpoint_positions(seq_len, *, type, block_size=None, n_blocks=None, start_at=0, skip=0,
                         save_last=False, histogram_tracker=None, **_ignored):
    """Return list of positions where GDN checkpoints should be captured.

    Dispatches on `type` (the strategy type), not `tag` (the unique ID).
    Accepts the full strategy config dict as kwargs
    (e.g. ``checkpoint_positions(seq_len, **strategy)``).

    If save_last is True, always includes seq_len as a checkpoint position.
    This is useful for any strategy: the endpoint checkpoint is "free" in that
    it guarantees full reuse on consecutive-turn hits.
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
    return sorted(set(positions))
