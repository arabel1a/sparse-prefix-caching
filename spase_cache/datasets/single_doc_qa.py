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

        if len(rows) > cfg.max_rows:
            rows = rows[:cfg.max_rows]
            log.info("Capped to %d rows (max_rows)", cfg.max_rows)

        # Tokenize
        chunk_size = cfg.tokenizer_chunk_size
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

        n_before = len(df)
        df = df.filter(pl.col("n_tokens") >= cfg.min_seq_len)
        log.info("Dropped %d rows shorter than %d tokens", n_before - len(df), cfg.min_seq_len)

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
        ds = load_dataset("rajpurkar/squad", split=cfg.split)

        groups: dict[str, list[dict]] = {}
        for row in ds:
            groups.setdefault(row["context"], []).append(row)

        rows, gid = [], 0
        for ctx, qas in groups.items():
            if len(qas) < cfg.min_questions:
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
        ds = load_dataset("tasksource/QuALITY", split=cfg.split)

        # Group by article_id
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in ds:
            groups[row["article_id"]].append(row)

        rows, gid = [], 0
        for article_id, qas in groups.items():
            if len(qas) < cfg.min_questions:
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

    For NarrativeQA the input is "Question ï»¿Document" or "Question <html>..."
    We split question from document, group by document, and construct:
      text = document + " Question: " + question + " Answer: " + answer
    """

    _SUBSET_FILES = {
        "NarrativeQA": "narrativeqa",
        "HotpotQA": "hotpotqa",
        "Character Archetype Classification": "character_id",
        "OpenSubtitles": "opensubtitles",
        "AO3 Style Change Detection": "style_change",
        "VLSP": "vlsp",
    }

    @staticmethod
    def _split_question_doc(inp: str) -> tuple[str, str]:
        """Split MuLD input into (question, document).

        NarrativeQA format: "Question? ï»¿Document..." or "Question? <html>..."
        """
        if "\ufeff" in inp:
            q, doc = inp.split("\ufeff", 1)
            return q.strip(), doc.strip()
        if "<html>" in inp:
            idx = inp.index("<html>")
            return inp[:idx].strip(), inp[idx:]
        # Fallback: split at first '?'
        if "?" in inp:
            idx = inp.index("?") + 1
            return inp[:idx].strip(), inp[idx:].strip()
        return "", inp

    def _load_raw(self) -> list[dict]:
        import bz2
        import hashlib
        import json
        from huggingface_hub import hf_hub_download

        cfg = self.cfg
        subset = cfg.subset
        split = cfg.split
        file_key = self._SUBSET_FILES[subset]
        filename = f"data/{file_key}_{split}.json.bz2"

        path = hf_hub_download("ghomasHudson/muld", filename, repo_type="dataset")

        # Parse JSONL from bz2, group by document
        # doc_hash -> {"doc": str, "qas": [(question, answer)]}
        doc_groups: dict[str, dict] = {}
        n_read = 0
        with bz2.open(path, "rt") as f:
            for line in tqdm(f, total=cfg.max_rows, desc="pre-filtering"):
                if n_read >= cfg.max_rows:
                    break
                row = json.loads(line)
                outputs = row["output"] if isinstance(row["output"], list) else [row["output"]]
                question, doc = self._split_question_doc(row["input"])
                doc_hash = hashlib.md5(doc.encode()).hexdigest()

                if doc_hash not in doc_groups:
                    doc_groups[doc_hash] = {"doc": doc, "qas": []}
                for out in outputs:
                    doc_groups[doc_hash]["qas"].append((question, out))
                    n_read += 1

        rows, gid = [], 0
        for j, (doc_hash, grp) in enumerate(doc_groups.items()):
            if j >= cfg.max_convs: break
            if len(grp["qas"]) < cfg.min_questions: continue
            doc = grp["doc"]
            for i, (question, answer) in enumerate(grp["qas"]):
                if i > cfg.max_questions: break
                text = f"{doc} Question: {question} Answer: {answer}"
                rows.append({"group_id": gid, "idx": i, "text": text})
            gid += 1
        log.info("MuLD/%s: %d documents, %d instances (read %d raw rows)",
                 subset, gid, len(rows), n_read)
        return rows
