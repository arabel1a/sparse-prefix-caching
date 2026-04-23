"""OpenAssistant OASST1 conversation trees.

Each tree has multiple root-to-leaf paths that share early turns.
Tree = conversation, each root-to-leaf path = one request.
Paths sharing early turns share a genuine token prefix.
"""
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from sparse_prefix_caching.datasets.base import Dataset

log = logging.getLogger(__name__)


def _build_trees(ds):
    msgs = {}
    children = defaultdict(list)
    roots = []
    for row in ds:
        mid = row["message_id"]
        pid = row["parent_id"]
        msgs[mid] = row
        if pid is None:
            roots.append(mid)
        else:
            children[pid].append(mid)
    trees = defaultdict(list)
    for mid in roots:
        trees[msgs[mid]["message_tree_id"]].append(mid)
    return msgs, children, trees


def _enumerate_paths(mid, msgs, children):
    kids = children[mid]
    if not kids:
        yield [mid]
        return
    for child in kids:
        for suffix in _enumerate_paths(child, msgs, children):
            yield [mid] + suffix


def _format_path(path, msgs):
    parts = []
    for mid in path:
        role = msgs[mid]["role"]
        label = "Human" if role == "prompter" else "Assistant"
        parts.append(f"{label}: {msgs[mid]['text']}")
    return "\n\n".join(parts)


class Osst1Dataset(Dataset):
    """Conversation tree dataset from OpenAssistant OASST1."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._tokens = {}
        assert False, outdated

    def prepare(self, tokenizer) -> None:
        from datasets import load_dataset

        cfg = self.cfg
        ds = load_dataset("OpenAssistant/oasst1", split=cfg.split)

        msgs, children, trees = _build_trees(ds)

        rows = []
        for tree_id, root_ids in trees.items():
            all_paths = []
            for rid in root_ids:
                all_paths.extend(_enumerate_paths(rid, msgs, children))
            if len(all_paths) < cfg.min_paths:
                continue
            for i, path in enumerate(all_paths):
                text = _format_path(path, msgs)
                rows.append({"tree_id": tree_id, "path_idx": i, "text": text})

        log.info("OASST1: %d trees, %d paths (min_paths=%d)",
                 len(set(r["tree_id"] for r in rows)), len(rows), cfg.min_paths)

        if len(rows) > cfg.max_rows:
            rows = rows[:cfg.max_rows]
            log.info("Capped to %d rows (max_rows)", cfg.max_rows)

        texts = [r["text"] for r in rows]
        all_tokens = []
        for i in tqdm(range(0, len(texts), cfg.tokenizer_chunk_size), desc="tokenizing"):
            chunk = texts[i:i + cfg.tokenizer_chunk_size]
            for ids in tokenizer(chunk, add_special_tokens=False, truncation=True,
                                 max_length=cfg.max_seq_len)["input_ids"]:
                all_tokens.append(np.array(ids, dtype=np.int32))

        df = pl.DataFrame({
            "tree_id": [r["tree_id"] for r in rows],
            "path_idx": [r["path_idx"] for r in rows],
            "tokens": all_tokens,
            "n_tokens": [len(t) for t in all_tokens],
        })

        n_before = len(df)
        df = df.filter(pl.col("n_tokens") >= cfg.min_seq_len)
        log.info("Dropped %d paths shorter than %d tokens", n_before - len(df), cfg.min_seq_len)

        out_path = Path(cfg.processed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        log.info("Saved %d paths to %s", len(df), out_path)

    def _load(self) -> None:
        df = pl.read_parquet(self.cfg.processed)
        self._tokens = {}
        self._requests = []
        for tree_id, path_idx, tokens in zip(
            df["tree_id"].to_list(), df["path_idx"].to_list(), df["tokens"].to_list()
        ):
            key = (tree_id, path_idx)
            self._tokens[key] = tokens
            self._requests.append(key)
        log.info("Loaded %d paths from %s", len(self._requests), self.cfg.processed)

    def conv_id(self, request) -> str:
        return request[0]  # tree_id

    def get_tokens(self, request) -> list[int]:
        return self._tokens[request]
