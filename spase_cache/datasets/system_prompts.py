"""System prompt datasets — shared system prompt prefix with varied user queries.

Each system prompt is a "conversation". Requests within a conversation share
the system prompt as a long common prefix, followed by different user queries
sampled from ShareGPT.

  text = system_prompt + "\n\nUser: " + user_query
"""
import logging
from abc import abstractmethod
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from spase_cache.datasets.base import Dataset

log = logging.getLogger(__name__)


class SystemPromptDataset(Dataset):
    """Base for system-prompt-prefix datasets."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._tokens = {}  # (prompt_id, query_idx) -> list[int]

    @abstractmethod
    def _load_raw(self) -> list[dict]:
        """Return list of {"group_id": str, "idx": int, "text": str}."""

    # -- prepare ---------------------------------------------------------------

    def prepare(self, tokenizer) -> None:
        cfg = self.cfg
        rows = self._load_raw()

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
        for group_id, idx, tokens in zip(
            df["group_id"].to_list(), df["idx"].to_list(), df["tokens"].to_list()
        ):
            key = (group_id, idx)
            self._tokens[key] = tokens
            self._requests.append(key)
        log.info("Loaded %d rows from %s", len(self._requests), self.cfg.processed)

    def conv_id(self, request) -> str:
        return str(request[0])

    def get_tokens(self, request) -> list[int]:
        return self._tokens[request]


# ---------------------------------------------------------------------------
# Leaked system prompts + ShareGPT user queries
# ---------------------------------------------------------------------------

def _load_system_prompts(prompts_dir: Path, min_chars: int, max_chars: int) -> list[tuple[str, str]]:
    """Load .md files recursively, return list of (name, text)."""
    prompts = []
    for md in sorted(prompts_dir.rglob("*.md")):
        # skip meta files and old/raw directories
        rel = md.relative_to(prompts_dir)
        parts = rel.parts
        if any(p.lower() in ("old", "raw", ".git", ".github") for p in parts):
            continue
        if md.name.lower() == "readme.md":
            continue
        text = md.read_text(errors="replace").strip()
        if len(text) < min_chars or len(text) > max_chars:
            continue
        name = str(rel).replace("/", "_").removesuffix(".md")
        prompts.append((name, text))
    return prompts


def _load_user_queries(sharegpt_path: str, max_queries: int) -> list[str]:
    """Load first-turn user messages from ShareGPT CSV."""
    df = pl.read_csv(sharegpt_path)
    df = df.filter(pl.col("role") == "user").sort(["url", "message_index"])
    # take first user message per conversation
    first_turns = df.group_by("url").first()
    queries = first_turns["plain_text"].drop_nulls().to_list()
    # filter out very short queries
    queries = [q for q in queries if len(q) > 10]
    if len(queries) > max_queries:
        queries = queries[:max_queries]
    return queries


class LeakedPromptsDataset(SystemPromptDataset):
    """System prompts from leaked sources + ShareGPT user queries."""

    def _load_raw(self) -> list[dict]:
        cfg = self.cfg
        prompts = _load_system_prompts(Path(cfg.prompts_dir), cfg.min_prompt_chars, cfg.max_prompt_chars)
        log.info("Loaded %d system prompts from %s", len(prompts), cfg.prompts_dir)

        if len(prompts) > cfg.max_convs:
            prompts = prompts[:cfg.max_convs]

        queries = _load_user_queries(cfg.sharegpt_path, cfg.queries_per_prompt * len(prompts))
        log.info("Loaded %d user queries from %s", len(queries), cfg.sharegpt_path)

        rows = []
        for gid, (name, system_prompt) in enumerate(prompts):
            # round-robin assign queries to prompts
            prompt_queries = queries[gid::len(prompts)][:cfg.queries_per_prompt]
            for i, query in enumerate(prompt_queries):
                text = f"{system_prompt}\n\nUser: {query}"
                rows.append({"group_id": name, "idx": i, "text": text})

        log.info("LeakedPrompts: %d prompts, %d requests", len(prompts), len(rows))
        return rows
