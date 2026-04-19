"""Synthetic dataset with controlled overlap distribution.

Generates sequences on-the-fly with exact control over:
  - n_convs: number of conversations (anchor strings)
  - seq_len: fixed length of each sequence
  - overlap distribution: how much prefix each request shares with its conversation's anchor

Each conversation has an "anchor" sequence of fixed tokens. Requests copy
the first L tokens from the anchor (L ~ overlap_distribution), then fill
the rest with a unique delimiter token to guarantee no spurious hash collisions.

No tokenizer or disk storage needed — tokens are synthetic integers.
"""
import logging

import numpy as np
from omegaconf import DictConfig

from spase_cache.datasets.base import Dataset

log = logging.getLogger(__name__)


class SyntheticDataset(Dataset):
    """Synthetic dataset with controllable overlap distribution."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._tokens = {}  # (conv_id, idx) -> list[int]

    def prepare(self, tokenizer) -> None:
        pass  # nothing to prepare — generated on the fly

    def _load(self) -> None:
        cfg = self.cfg
        n_convs = cfg.n_convs
        seq_len = cfg.seq_len
        n_requests = cfg.n_requests
        seed = cfg.seed

        rng = np.random.RandomState(seed)

        # Step 0: create anchor sequences — each conv gets a unique base token
        # Anchor i is filled with token value i. Values in [0, n_convs).
        anchors = np.zeros((n_convs, seq_len), dtype=np.int32)
        for i in range(n_convs):
            anchors[i, :] = i

        # Delimiter tokens: one per request, guaranteed outside [0, n_convs)
        delimiter_base = n_convs

        self._tokens = {}
        self._requests = []

        for req_idx in range(n_requests):
            conv = req_idx % n_convs

            # Sample overlap length
            if cfg.overlap_dist == "uniform":
                L = rng.randint(cfg.overlap_min, cfg.overlap_max + 1)
            elif cfg.overlap_dist == "normal":
                L = int(np.clip(rng.normal(cfg.overlap_mu, cfg.overlap_sigma),
                                cfg.overlap_min, cfg.overlap_max))
            elif cfg.overlap_dist == "fixed":
                L = cfg.overlap_length
            else:
                raise ValueError(f"Unknown overlap_dist: {cfg.overlap_dist}")

            L = int(np.clip(L, 0, seq_len))

            # Build token sequence: first L tokens from anchor, rest = unique delimiter
            tokens = np.empty(seq_len, dtype=np.int32)
            tokens[:L] = anchors[conv, :L]
            delimiter = delimiter_base + req_idx
            tokens[L:] = delimiter

            key = (str(conv), req_idx)
            self._tokens[key] = tokens.tolist()
            self._requests.append(key)

        log.info(
            "Generated %d synthetic requests: %d convs, seq_len=%d, overlap_dist=%s",
            n_requests, n_convs, seq_len, cfg.overlap_dist,
        )

    def conv_id(self, request) -> str:
        return request[0]

    def get_tokens(self, request) -> list[int]:
        return self._tokens[request]
