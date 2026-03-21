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
        if (1 << i) < start_at:
            continue
        positions.append(1 << i)
        i += 1
    return positions


# ---------------------------------------------------------------------------
# DP-optimal checkpoint placement (Corollary 1 from the paper)
# ---------------------------------------------------------------------------
def solve_dp(hist, budget):
    """Solve the optimal checkpoint placement DP (vectorized).

    Given overlap histogram `hist` (array of length N+1 where hist[t] = probability
    of overlap depth t, t=0..N), and checkpoint budget M, returns list of optimal
    checkpoint positions.

    DP recurrence (paper Corollary 1):
        dp[0, j] = sum_{t=1}^{j} p_t * t
        dp[m, j] = min_{1<=s<=j} (dp[m-1, s-1] + w(s, j))
    where w(s, j) = sum_{t=s}^{j} p_t * (t - s)

    Returns sorted list of checkpoint positions (1-indexed).
    """
    N = len(hist) - 1  # hist[0..N], positions 1..N
    M = budget
    if N <= 0 or M <= 0:
        return []

    # Normalize histogram
    p = np.array(hist, dtype=np.float64)
    total = p.sum()
    if total < 1e-12:
        block = max(1, N // (M + 1))
        return list(range(block, N + 1, block))[:M]
    p = p / total

    # Precompute prefix sums for w(s, j) = sum_{t=s}^{j} p_t*(t-s)
    # w(s,j) = cum_pt[j+1] - cum_pt[s] - s * (cum_p[j+1] - cum_p[s])
    cum_p = np.zeros(N + 2)
    cum_pt = np.zeros(N + 2)
    t_idx = np.arange(N + 1, dtype=np.float64)
    cum_p[1:] = np.cumsum(p)
    cum_pt[1:] = np.cumsum(p * t_idx)

    # Precompute w(s, j) matrix — w[s, j] for 0 <= s, j <= N
    # w[s, j] = (cum_pt[j+1] - cum_pt[s]) - s * (cum_p[j+1] - cum_p[s])
    # Shape: (N+1, N+1), only upper triangle (s <= j) is meaningful
    s_arr = np.arange(N + 1, dtype=np.float64)
    # cum_pt[j+1] for j=0..N → cum_pt[1..N+1]
    cpt_j = cum_pt[1:]  # shape (N+1,)
    cp_j = cum_p[1:]    # shape (N+1,)
    # w[s, j] = cpt_j[j] - cum_pt[s] - s * (cp_j[j] - cum_p[s])
    # Vectorize: for each j, compute over all s at once
    # But full matrix is O(N^2) memory — fine for N up to ~20k

    # Base case: m=0, dp[0, j] = w(0, j) = cum_pt[j+1]
    dp_prev = cpt_j.copy()  # dp_prev[j] = w(0, j) for j=0..N

    # Store backtrack: list of arrays
    all_back = [None]  # index 0 unused (m=0 has no backtrack)

    for m in range(1, M + 1):
        dp_curr = np.full(N + 1, np.inf)
        back_curr = np.zeros(N + 1, dtype=np.intp)
        dp_curr[0] = 0.0
        # For each j, minimize over s in [1..j]:
        #   cost(s) = dp_prev[s-1] + w(s, j)
        #   w(s, j) = cpt_j[j] - cum_pt[s] - s*(cp_j[j] - cum_p[s])
        # Vectorized: for each j, build cost[1..j] and take argmin
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

    # Backtrack to recover positions
    positions = []
    j = N
    for m in range(M, 0, -1):
        s = int(all_back[m][j])
        positions.append(s)
        j = s - 1
    positions.sort()
    return positions


class HistogramTracker:
    """Tracks overlap depth histogram for distribution-aware checkpointing.

    Three modes:
    - 'frozen': accumulate during warmup, solve DP once, freeze
    - 'periodic': re-solve DP every `replan_interval` observations
    - 'exp_decay': exponential decay weighting (gamma^{n-1-i}), re-solve every `replan_interval`
    """

    def __init__(self, max_len, budget, mode='frozen', gamma=0.99, replan_interval=100):
        self.max_len = max_len
        self.budget = budget
        self.mode = mode
        self.gamma = gamma
        self.replan_interval = replan_interval

        # Histogram bins: index = overlap depth (0..max_len)
        self.counts = np.zeros(max_len + 1, dtype=np.float64)
        self.n_obs = 0
        self._positions = None  # cached DP solution
        self._dirty = True      # needs re-solving

    def observe(self, overlap_depth):
        """Record an observed overlap depth."""
        depth = min(max(int(overlap_depth), 0), self.max_len)
        if self.mode == 'exp_decay':
            # Decay all previous counts, then add new observation
            self.counts *= self.gamma
            self.counts[depth] += 1.0
        else:
            self.counts[depth] += 1.0
        self.n_obs += 1
        self._dirty = True

    def solve(self):
        """Solve DP on current histogram and cache the result."""
        self._positions = solve_dp(self.counts, self.budget)
        self._dirty = False
        log.info("DP solved: %d checkpoints, n_obs=%d, positions=%s",
                 len(self._positions), self.n_obs, self._positions)

    def get_positions(self, seq_len):
        """Get checkpoint positions for a request of given length.

        Solves DP if needed (based on mode and replan schedule).
        Returns positions <= seq_len.
        """
        should_solve = False
        if self._positions is None:
            should_solve = True
        elif self.mode == 'frozen':
            # Only solve once (after warmup)
            pass
        elif self.mode in ('periodic', 'exp_decay'):
            if self._dirty and self.n_obs % self.replan_interval == 0:
                should_solve = True

        if should_solve:
            self.solve()

        return [p for p in self._positions if p <= seq_len]

    def freeze(self):
        """Solve and freeze — no further updates to positions."""
        self.solve()
        self.mode = 'frozen'


def checkpoint_positions(seq_len, *, type, block_size=None, n_blocks=None, start_at=0, skip=0,
                         histogram_tracker=None, **_ignored):
    """Return list of positions where GDN checkpoints should be captured.

    Dispatches on `type` (the strategy type), not `tag` (the unique ID).
    Accepts the full strategy config dict as kwargs
    (e.g. ``checkpoint_positions(seq_len, **strategy)``).
    """
    if type in ("no_cache", "kv_only") or seq_len < skip:
        return []
    if type in ("block", "balanced_fix_blocksize"):
        return balanced_positions(seq_len, block_size=block_size)
    if type == "balanced_fix_nblocks":
        return balanced_positions(seq_len, n_blocks=n_blocks)
    if type == "sqrt":
        return sqrt_positions(seq_len)
    if type in ("log", "dyadic", "diadic"):
        return diadic_positions(seq_len, start_at)
    if type in ("histogram_frozen", "histogram_periodic", "histogram_exp_decay"):
        if histogram_tracker is None:
            raise ValueError(f"Strategy type {type} requires a histogram_tracker")
        return histogram_tracker.get_positions(seq_len)
    raise ValueError(f"Unknown strategy type: {type}")
