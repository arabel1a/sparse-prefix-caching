"""Abstract dataset for prefix caching benchmarks."""
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from spase_cache.utils import interleave as _interleave

log = logging.getLogger(__name__)


class Dataset(ABC):
    """Base class for all benchmark datasets.

    Subclasses must implement:
      - prepare(tokenizer): filter raw data, tokenize, save parquet
      - _load(): load prepared parquet, populate _requests and internal state
      - conv_id(request): extract grouping key (conversation/document id)
      - get_tokens(request): return full token list for a request

    After load(seed), .requests is the final interleaved order (if configured).
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._requests = None

    # -- abstract interface --------------------------------------------------

    @abstractmethod
    def prepare(self, tokenizer) -> None:
        """Filter raw data, tokenize, write parquet to cfg.processed."""

    @abstractmethod
    def _load(self) -> None:
        """Load prepared parquet. Populate self._requests and internal token storage."""

    @abstractmethod
    def conv_id(self, request) -> str:
        """Grouping key for interleaving and cache lookup."""

    @abstractmethod
    def get_tokens(self, request) -> list[int]:
        """Full token sequence for a single request."""

    # -- common functionality ------------------------------------------------

    def load(self, seed: int = 42) -> None:
        """Load data, then interleave if configured. Call once."""
        self._load()
        if self.cfg.get("interleave", False):
            log.info("Interleaving %d requests (seed=%d)...", len(self._requests), seed)
            self._requests = _interleave(self._requests, seed)

    @property
    def requests(self) -> list:
        return self._requests

    def train_test_split(self, train_frac=None):
        if train_frac is None:
            train_frac = self.cfg.get("train_frac", 0.5)
        n_train = int(len(self._requests) * train_frac)
        return self._requests[:n_train], self._requests[n_train:]
