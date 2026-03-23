"""Tokenize raw conversation data and compute prefix overlap statistics.

Usage:
  python prepare_data.py
  python prepare_data.py prepare._target_=prepare_data.tokenize
  python prepare_data.py prepare._target_=prepare_data.compute_overlap
"""
import json
import logging
from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from spase_cache.datasets.base import Dataset
from spase_cache.utils import setup_output_dir

log = logging.getLogger(__name__)


_HASH_PRIME = 2654435761
_HASH_MASK = (1 << 63) - 1


def _make_dataset(cfg) -> Dataset:
    """Instantiate the dataset from config (uses _target_ in data config)."""
    cls = hydra.utils.get_class(cfg.data._target_)
    return cls(cfg=cfg.data)


def tokenize(out_dir, full_cfg, **_kw):
    """Filter raw data, tokenize, and save parquet."""
    dataset = _make_dataset(full_cfg)
    tokenizer = AutoTokenizer.from_pretrained(full_cfg.model.tokenizer, use_fast=True)
    dataset.prepare(tokenizer)


def compute_overlap(out_dir, full_cfg, **_kw):
    """Load tokenized parquet and compute prefix overlap distribution."""
    out_dir = Path(out_dir)
    dataset = _make_dataset(full_cfg)
    dataset.load(seed=full_cfg.seed)

    requests = dataset.requests
    n_conversations = len(set(dataset.conv_id(r) for r in requests))
    log.info("%d requests from %d conversations", len(requests), n_conversations)

    train_requests, test_requests = dataset.train_test_split()
    log.info("Train: %d requests, Test: %d requests", len(train_requests), len(test_requests))

    seen = set()

    log.info("Warming prefix cache with train split...")
    for req in tqdm(train_requests, desc="train"):
        tokens = dataset.get_tokens(req)
        h = 0
        for t in tokens:
            h = (h * _HASH_PRIME + int(t) + 1) & _HASH_MASK
            seen.add(h)

    lcp_lengths = []
    log.info("Simulating non-deleting prefix cache on test split...")
    for req in tqdm(test_requests, desc="test"):
        tokens = dataset.get_tokens(req)
        h = 0
        lcp = 0
        matched = True
        for t in tokens:
            h = (h * _HASH_PRIME + int(t) + 1) & _HASH_MASK
            if matched and h in seen:
                lcp += 1
            else:
                matched = False
            seen.add(h)
        lcp_lengths.append(lcp)

    overlap_path = out_dir / "overlap_lcp.json"
    overlap_path.write_text(json.dumps({
        "lcp_lengths": lcp_lengths,
        "n_requests": len(test_requests),
        "n_train_requests": len(train_requests),
        "n_conversations": n_conversations,
    }))

    n_hits = sum(1 for l in lcp_lengths if l > 0)
    log.info("%d/%d test requests had cache hits (%.1f%%)",
             n_hits, len(lcp_lengths), n_hits / len(lcp_lengths) * 100)
    log.info("Overlap results saved to %s", overlap_path)


def prepare_all(out_dir, full_cfg, **_kw):
    """Run tokenize then compute_overlap."""
    tokenize(out_dir, full_cfg)
    compute_overlap(out_dir, full_cfg)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = setup_output_dir(cfg, "prepare_data")
    fn = hydra.utils.get_method(cfg.prepare_data._target_)
    fn(out_dir=out_dir, full_cfg=cfg)


if __name__ == "__main__":
    main()
