"""NVIDIA Nemotron-SWE-v1 multi-turn SWE conversations.

Each row is a session with growing conversation prefix.
Sessions for the same repo share system prompt + tool schemas as prefix.

Each row = one conversation (keyed by index), steps within = requests.
"""
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from spase_cache.datasets.base import Dataset

log = logging.getLogger(__name__)


def _extract_calls(row):
    messages = row["messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)

    tools = row.get("tools", "")
    if isinstance(tools, str) and tools:
        tool_prefix = f"Tools:\n{tools}\n\n"
    elif isinstance(tools, list):
        tool_prefix = f"Tools:\n{json.dumps(tools)}\n\n"
    else:
        tool_prefix = ""

    calls = []
    context = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            content = json.dumps(msg.get("tool_calls", ""))
        context.append(f"{role}: {content}")
        if role == "assistant":
            prompt = tool_prefix + "\n\n".join(context[:-1])
            calls.append(prompt)
    return calls


class NemotronSweDataset(Dataset):
    """SWE trajectories from Nemotron-SWE-v1."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._tokens = {}
        assert False, "outdated"

    def prepare(self, tokenizer) -> None:
        from datasets import load_dataset

        cfg = self.cfg
        ds = load_dataset("nvidia/Nemotron-SWE-v1", split=cfg.split, streaming=True)

        rows = []
        n_sessions = 0
        for idx, row in enumerate(ds):
            if n_sessions >= cfg.max_sessions:
                break
            if len(rows) >= cfg.max_rows:
                break
            repo = row.get("repo", f"session_{idx}")
            calls = _extract_calls(row)
            if not calls:
                continue
            n_sessions += 1
            for step, prompt in enumerate(calls):
                rows.append({"repo": repo, "session_idx": idx, "step": step, "text": prompt})

        log.info("Nemotron SWE: %d sessions, %d calls, %d repos",
                 len(set(r["session_idx"] for r in rows)), len(rows),
                 len(set(r["repo"] for r in rows)))

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
            "repo": [r["repo"] for r in rows],
            "session_idx": [r["session_idx"] for r in rows],
            "step": [r["step"] for r in rows],
            "tokens": all_tokens,
            "n_tokens": [len(t) for t in all_tokens],
        })

        n_before = len(df)
        df = df.filter(pl.col("n_tokens") >= cfg.min_seq_len)
        log.info("Dropped %d calls shorter than %d tokens", n_before - len(df), cfg.min_seq_len)

        out_path = Path(cfg.processed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        log.info("Saved %d calls to %s", len(df), out_path)

    def _load(self) -> None:
        df = pl.read_parquet(self.cfg.processed)
        self._tokens = {}
        self._requests = []
        for repo, session_idx, step, tokens in zip(
            df["repo"].to_list(), df["session_idx"].to_list(),
            df["step"].to_list(), df["tokens"].to_list()
        ):
            key = (repo, session_idx, step)
            self._tokens[key] = tokens
            self._requests.append(key)
        log.info("Loaded %d calls from %s", len(self._requests), self.cfg.processed)

    def conv_id(self, request) -> str:
        return f"{request[0]}_{request[1]}"  # repo_sessionIdx

    def get_tokens(self, request) -> list[int]:
        return self._tokens[request]
