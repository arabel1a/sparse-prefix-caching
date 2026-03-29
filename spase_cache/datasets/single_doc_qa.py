"""Single-document, multiple-questions QA datasets.

All datasets here share the same structure:
  - One long document is a "conversation" (conv_id = group_id)
  - Each QA pair is a "request" sharing the document prefix
  - text = document + " Question: " + question + " Answer: " + answer

Subclasses only override _load_raw() to return list[dict] with keys:
  group_id (int), idx (int), text (str)
"""
import logging
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from spase_cache.datasets.base import Dataset

log = logging.getLogger(__name__)


class SingleDocQADataset(Dataset):
    """Base for single-document multiple-question QA datasets."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._tokens = {}  # (group_id, idx) -> list[int]

    @abstractmethod
    def _load_raw(self) -> list[dict]:
        """Return list of {"group_id": int, "idx": int, "text": str}."""

    # -- prepare ---------------------------------------------------------------

    def prepare(self, tokenizer) -> None:
        cfg = self.cfg
        rows = self._load_raw()

        max_rows = cfg.get("max_rows", len(rows))
        if len(rows) > max_rows:
            rows = rows[:max_rows]
            log.info("Capped to %d rows (max_rows)", max_rows)

        # Tokenize
        chunk_size = cfg.get("tokenizer_chunk_size", 16)
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

        out_path = Path(cfg.processed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        log.info("Saved %d rows to %s", len(df), out_path)

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
        log.info("Loaded %d rows from %s", len(self._requests), self.cfg.processed)

    def conv_id(self, request) -> str:
        return str(request[0])

    def get_tokens(self, request) -> list[int]:
        return self._tokens[request]


# ---------------------------------------------------------------------------
# Concrete datasets
# ---------------------------------------------------------------------------


class SquadDataset(SingleDocQADataset):
    """SQuAD — context paragraphs with multiple QA pairs."""

    def _load_raw(self) -> list[dict]:
        from datasets import load_dataset

        cfg = self.cfg
        ds = load_dataset("rajpurkar/squad", split=cfg.get("split", "train"))

        groups: dict[str, list[dict]] = {}
        for row in ds:
            groups.setdefault(row["context"], []).append(row)

        min_q = cfg.get("min_questions", 3)
        rows, gid = [], 0
        for ctx, qas in groups.items():
            if len(qas) < min_q:
                continue
            for i, row in enumerate(qas):
                answer = row["answers"]["text"][0] if row["answers"]["text"] else ""
                text = f"{ctx} Question: {row['question']} Answer: {answer}"
                rows.append({"group_id": gid, "idx": i, "text": text})
            gid += 1
        log.info("SQuAD: %d groups, %d QA pairs", gid, len(rows))
        return rows


class QuALITYDataset(SingleDocQADataset):
    """QuALITY — long-document multiple-choice QA (~2k-8k token passages).

    Uses tasksource/QuALITY (parquet, natively supported). One row per question,
    each with full `article` text. gold_label is 1-indexed (1-4).
    We group by article_id; each question becomes a request sharing the article prefix.
    """

    def _load_raw(self) -> list[dict]:
        from datasets import load_dataset

        cfg = self.cfg
        ds = load_dataset("tasksource/QuALITY", split=cfg.get("split", "train"))

        # Group by article_id
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in ds:
            groups[row["article_id"]].append(row)

        min_q = cfg.get("min_questions", 3)
        rows, gid = [], 0
        for article_id, qas in groups.items():
            if len(qas) < min_q:
                continue
            article = qas[0]["article"]
            for i, row in enumerate(qas):
                options = row["options"]
                opts_str = " ".join(f"({chr(65+j)}) {o}" for j, o in enumerate(options))
                label = row["gold_label"] - 1  # 1-indexed -> 0-indexed
                answer = options[label] if 0 <= label < len(options) else ""
                text = f"{article} Question: {row['question']} Options: {opts_str} Answer: {answer}"
                rows.append({"group_id": gid, "idx": i, "text": text})
            gid += 1
        log.info("QuALITY: %d articles, %d questions", gid, len(rows))
        return rows

class MuLDDataset(SingleDocQADataset):
    """MuLD — Multitask Long Document Benchmark (>10k token documents).

    Loads raw bz2-compressed JSONL from ghomasHudson/muld HF repo.
    Each line: {"input": str, "output": str|list[str], "metadata": str}.
    We group by input text (the document); each output becomes a request.
    """

    _SUBSET_FILES = {
        "NarrativeQA": "narrativeqa",
        "HotpotQA": "hotpotqa",
        "Character Archetype Classification": "character_id",
        "OpenSubtitles": "opensubtitles",
        "AO3 Style Change Detection": "style_change",
        "VLSP": "vlsp",
    }

    def _load_raw(self) -> list[dict]:
        import bz2
        import json
        from huggingface_hub import hf_hub_download

        cfg = self.cfg
        subset = cfg.get("subset", "NarrativeQA")
        split = cfg.get("split", "train")
        file_key = self._SUBSET_FILES[subset]
        filename = f"data/{file_key}_{split}.json.bz2"

        path = hf_hub_download("ghomasHudson/muld", filename, repo_type="dataset")

        # Parse JSONL from bz2
        all_rows = []
        with bz2.open(path, "rt") as f:
            for line in f:
                row = json.loads(line)
                if not isinstance(row["output"], list):
                    row["output"] = [row["output"]]
                all_rows.append(row)

        # Group by input text
        groups: dict[str, list[str]] = defaultdict(list)
        for row in all_rows:
            for out in row["output"]:
                groups[row["input"]].append(out)

        min_q = cfg.get("min_questions", 2)
        rows, gid = [], 0
        for inp, outputs in groups.items():
            if len(outputs) < min_q:
                continue
            for i, out in enumerate(outputs):
                text = f"{inp} Answer: {out}"
                rows.append({"group_id": gid, "idx": i, "text": text})
            gid += 1
        log.info("MuLD/%s: %d documents, %d instances", subset, gid, len(rows))
        return rows
