"""Agentic trajectory datasets — multi-step LLM calls with growing context.

All datasets here share the same structure:
  - One trajectory/session is a "conversation" (conv_id + session_idx)
  - Each LLM call is a "request" with prefix = all previous messages
  - Subclasses override _load_raw() to return list[dict] with keys:
    conv_id (str), session_idx (int), step (int), text (str)
"""
import json
import logging
from abc import abstractmethod
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from sparse_prefix_caching.datasets.base import Dataset

log = logging.getLogger(__name__)


class AgenticDataset(Dataset):
    """Base for agentic trajectory datasets."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._tokens = {}  # (conv_id, session_idx, step) -> list[int]

    @abstractmethod
    def _load_raw(self) -> list[dict]:
        """Return list of {"conv_id": str, "session_idx": int, "step": int, "text": str}."""

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
            "conv_id": [r["conv_id"] for r in rows],
            "session_idx": [r["session_idx"] for r in rows],
            "step": [r["step"] for r in rows],
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
        for conv_id, session_idx, step, tokens in zip(
            df["conv_id"].to_list(), df["session_idx"].to_list(),
            df["step"].to_list(), df["tokens"].to_list()
        ):
            key = (conv_id, session_idx, step)
            self._tokens[key] = tokens
            self._requests.append(key)
        log.info("Loaded %d rows from %s", len(self._requests), self.cfg.processed)

    def conv_id(self, request) -> str:
        return f"{request[0]}_{request[1]}"

    def get_tokens(self, request) -> list[int]:
        return self._tokens[request]


# ---------------------------------------------------------------------------
# SWE-agent trajectories (nebius)
# ---------------------------------------------------------------------------

def _extract_swe_agent_calls(row):
    traj = json.loads(row["trajectory"]) if isinstance(row["trajectory"], str) else row["trajectory"]
    calls = []
    context = []
    for msg in traj:
        role = msg.get("role", "")
        text = msg.get("content", msg.get("text", ""))
        if not text:
            continue
        context.append(f"{role}: {text}")
        if role in ("assistant", "ai"):
            calls.append("\n\n".join(context[:-1]))
    return calls


class SweAgentDataset(AgenticDataset):
    """SWE-agent trajectories from nebius/SWE-agent-trajectories."""

    def _load_raw(self) -> list[dict]:
        from datasets import load_dataset

        cfg = self.cfg
        ds = load_dataset("nebius/SWE-agent-trajectories", split=cfg.split, streaming=True)

        rows = []
        n_sessions = 0
        for idx, row in enumerate(ds):
            if n_sessions >= cfg.max_sessions:
                break
            if len(rows) >= cfg.max_rows:
                break
            instance_id = row["instance_id"]
            model = row.get("model_name", "unknown")
            calls = _extract_swe_agent_calls(row)
            if not calls:
                continue
            n_sessions += 1
            for step, prompt in enumerate(calls):
                rows.append({
                    "conv_id": instance_id,
                    "session_idx": idx,
                    "step": step,
                    "text": prompt,
                })

        log.info("SWE-agent: %d sessions, %d calls",
                 len(set(r["session_idx"] for r in rows)), len(rows))
        return rows


# ---------------------------------------------------------------------------
# Nemotron-SWE (nvidia)
# ---------------------------------------------------------------------------

def _extract_nemotron_calls(row):
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


class NemotronSweDataset(AgenticDataset):
    """SWE trajectories from nvidia/Nemotron-SWE-v1."""

    def _load_raw(self) -> list[dict]:
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
            calls = _extract_nemotron_calls(row)
            if not calls:
                continue
            n_sessions += 1
            for step, prompt in enumerate(calls):
                rows.append({
                    "conv_id": repo,
                    "session_idx": idx,
                    "step": step,
                    "text": prompt,
                })

        log.info("Nemotron SWE: %d sessions, %d calls, %d repos",
                 len(set(r["session_idx"] for r in rows)), len(rows),
                 len(set(r["conv_id"] for r in rows)))
        return rows
