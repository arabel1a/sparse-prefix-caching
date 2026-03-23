"""SQuAD dataset — context paragraphs with multiple QA pairs.

Each context paragraph is a "conversation", each QA pair is a "request".
Texts in the same group share the context prefix but diverge at the question.

text = context + " Question: " + question + " Answer: " + answer
"""
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from spase_cache.datasets.base import Dataset

log = logging.getLogger(__name__)


class SquadDataset(Dataset):
    """QA dataset from SQuAD — shared context prefix, diverging questions."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._tokens = {}  # (group_id, idx) -> list[int]

    # -- prepare ---------------------------------------------------------------

    def prepare(self, tokenizer) -> None:
        from datasets import load_dataset

        cfg = self.cfg
        split = cfg.get("split", "train")
        ds = load_dataset("rajpurkar/squad", split=split)

        # Group by context paragraph
        groups: dict[str, list[dict]] = {}
        for row in ds:
            ctx = row["context"]
            groups.setdefault(ctx, []).append(row)

        min_questions = cfg.get("min_questions", 3)
        rows = []
        group_id = 0
        for ctx, qas in groups.items():
            if len(qas) < min_questions:
                continue
            for i, row in enumerate(qas):
                answer = row["answers"]["text"][0] if row["answers"]["text"] else ""
                text = f"{ctx} Question: {row['question']} Answer: {answer}"
                rows.append({"group_id": group_id, "idx": i, "text": text})
            group_id += 1

        log.info("SQuAD: %d groups, %d QA pairs (min_questions=%d)", group_id, len(rows), min_questions)

        # Tokenize
        chunk_size = cfg.get("tokenizer_chunk_size", 64)
        texts = [r["text"] for r in rows]
        all_tokens = []
        for i in tqdm(range(0, len(texts), chunk_size), desc="tokenizing"):
            chunk = texts[i:i + chunk_size]
            for ids in tokenizer(chunk, add_special_tokens=False, truncation=True,
                                 max_length=cfg.max_seq_len)["input_ids"]:
                all_tokens.append(np.array(ids, dtype=np.int32))

        df = pl.DataFrame({
            "group_id": [r["group_id"] for r in rows],
            "idx": [r["idx"] for r in rows],
            "tokens": all_tokens,
            "n_tokens": [len(t) for t in all_tokens],
        })

        min_seq_len = cfg.get("min_seq_len", 0)
        if min_seq_len > 0:
            n_before = len(df)
            df = df.filter(pl.col("n_tokens") >= min_seq_len)
            log.info("Dropped %d rows shorter than %d tokens", n_before - len(df), min_seq_len)

        df = df.head(cfg.get("max_rows", len(df)))

        out_path = Path(cfg.processed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        log.info("Saved %d QA pairs to %s", len(df), out_path)

    # -- load / access ---------------------------------------------------------

    def _load(self) -> None:
        df = pl.read_parquet(self.cfg.processed)
        self._tokens = {}
        self._requests = []
        for gid, idx, tokens in zip(
            df["group_id"].to_list(), df["idx"].to_list(), df["tokens"].to_list()
        ):
            key = (gid, idx)
            self._tokens[key] = tokens
            self._requests.append(key)
        log.info("Loaded %d QA pairs from %s", len(self._requests), self.cfg.processed)

    def conv_id(self, request) -> str:
        return str(request[0])  # group_id

    def get_tokens(self, request) -> list[int]:
        return self._tokens[request]
