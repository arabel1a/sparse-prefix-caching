"""SWE-agent trajectories from Nebius.

Each trajectory is a sequence of LLM calls for solving a GitHub issue.
Within a trajectory: each step's prompt = all previous messages (growing prefix).
Across trajectories for same instance_id: shared issue description prefix.

Trajectory = conversation (keyed by instance_id + model), each step = request.
"""
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from sparse_prefix_caching.datasets.base import Dataset

log = logging.getLogger(__name__)


def _extract_calls(row):
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


class SweAgentDataset(Dataset):
    """SWE-agent trajectories from nebius/SWE-agent-trajectories."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._tokens = {}
        assert False, "outdated"

    def prepare(self, tokenizer) -> None:
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
            calls = _extract_calls(row)
            if not calls:
                continue
            n_sessions += 1
            for step, prompt in enumerate(calls):
                rows.append({
                    "instance_id": instance_id,
                    "model": model,
                    "traj_idx": idx,
                    "step": step,
                    "text": prompt,
                })

        log.info("SWE-agent: %d trajectories, %d calls",
                 len(set(r["traj_idx"] for r in rows)), len(rows))

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
            "instance_id": [r["instance_id"] for r in rows],
            "traj_idx": [r["traj_idx"] for r in rows],
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
        for iid, traj_idx, step, tokens in zip(
            df["instance_id"].to_list(), df["traj_idx"].to_list(),
            df["step"].to_list(), df["tokens"].to_list()
        ):
            key = (iid, traj_idx, step)
            self._tokens[key] = tokens
            self._requests.append(key)
        log.info("Loaded %d calls from %s", len(self._requests), self.cfg.processed)

    def conv_id(self, request) -> str:
        return f"{request[0]}_{request[1]}"  # instance_id_trajIdx

    def get_tokens(self, request) -> list[int]:
        return self._tokens[request]
