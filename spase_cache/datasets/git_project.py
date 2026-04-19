"""Git project dataset — full-repo snapshots at each commit.

Each commit produces a snapshot: all tracked files (filtered by extension)
concatenated with delimiters. Successive commits share a long common prefix,
making this ideal for prefix caching benchmarks.

conv_id is always the repo name (single "conversation" = the repo's history).
Each request = one commit snapshot.
"""
import logging
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from spase_cache.datasets.base import Dataset

log = logging.getLogger(__name__)

DELIMITER = "=" * 60


def _run(cmd: str, cwd: Path) -> str:
    return subprocess.check_output(cmd, shell=True, cwd=cwd).decode(errors="replace").strip()


def _get_commits(repo_dir: Path) -> list[str]:
    out = _run("git log --reverse --format=%H", repo_dir)
    return out.splitlines() if out else []


def _get_tree_files(repo_dir: Path, commit: str, extensions: set[str]) -> list[str]:
    out = _run(f"git ls-tree -r --name-only {commit}", repo_dir)
    if not out:
        return []
    return [f for f in out.splitlines() if Path(f).suffix in extensions]


def _get_renames(repo_dir: Path, parent: str, child: str) -> dict[str, str]:
    out = _run(f"git diff --diff-filter=R -M --name-status {parent} {child}", repo_dir)
    renames = {}
    for line in out.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].startswith("R"):
            renames[parts[1]] = parts[2]
    return renames


def _get_file_content(repo_dir: Path, commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=repo_dir
    ).decode(errors="replace")


def _build_snapshot(repo_dir: Path, commit: str, ordered_files: list[str]) -> str:
    parts = []
    for fpath in ordered_files:
        content = _get_file_content(repo_dir, commit, fpath)
        parts.append(f"{DELIMITER}\n# FILE: {fpath}\n{DELIMITER}\n{content}")
    return "\n".join(parts)


class GitProjectDataset(Dataset):
    """Revision dataset from a git repository's commit history."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._tokens = {}  # commit_hash -> list[int]

    # -- prepare ---------------------------------------------------------------

    def prepare(self, tokenizer) -> None:
        cfg = self.cfg
        repo_dir = Path(cfg.fetch.repo_dir).resolve()
        extensions = set(cfg.fetch.extensions)
        commits = _get_commits(repo_dir)
        limit = min(cfg.max_revisions, cfg.max_rows)
        commits = commits[:limit]
        log.info("Processing %d commits from %s", len(commits), repo_dir)

        # Build stable-ordered snapshots
        ordered_files: list[str] = []
        rows = []

        for i, commit in enumerate(tqdm(commits, desc="building snapshots")):
            current_files = set(_get_tree_files(repo_dir, commit, extensions))

            if i == 0:
                ordered_files = sorted(current_files)
            else:
                prev = commits[i - 1]
                renames = _get_renames(repo_dir, prev, commit)
                for j, f in enumerate(ordered_files):
                    if f in renames:
                        ordered_files[j] = renames[f]
                ordered_files = [f for f in ordered_files if f in current_files]
                existing = set(ordered_files)
                new_files = sorted(current_files - existing)
                ordered_files.extend(new_files)

            snapshot = _build_snapshot(repo_dir, commit, ordered_files)
            rows.append({"commit": commit, "snapshot": snapshot})

        if len(rows) > cfg.max_rows:
            rows = rows[:cfg.max_rows]
            log.info("Capped to %d rows (max_rows)", cfg.max_rows)

        log.info("Tokenizing %d snapshots...", len(rows))
        snapshots = [r["snapshot"] for r in rows]
        all_tokens = []
        for i in tqdm(range(0, len(snapshots), cfg.tokenizer_chunk_size), desc="tokenizing"):
            chunk = snapshots[i:i + cfg.tokenizer_chunk_size]
            for ids in tokenizer(chunk, add_special_tokens=False, truncation=True,
                                 max_length=cfg.max_seq_len)["input_ids"]:
                all_tokens.append(np.array(ids, dtype=np.int32))

        # Build parquet
        df = pl.DataFrame({
            "commit": [r["commit"] for r in rows],
            "tokens": all_tokens,
            "n_tokens": [len(t) for t in all_tokens],
        })

        # Drop short snapshots
        n_before = len(df)
        df = df.filter(pl.col("n_tokens") >= cfg.min_seq_len)
        log.info("Dropped %d snapshots shorter than %d tokens", n_before - len(df), cfg.min_seq_len)

        out_path = Path(cfg.processed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        log.info("Saved %d snapshots to %s", len(df), out_path)

    # -- load / access ---------------------------------------------------------

    def _load(self) -> None:
        df = pl.read_parquet(self.cfg.processed)
        self._tokens = {}
        self._requests = []
        for commit, tokens in zip(df["commit"].to_list(), df["tokens"].to_list()):
            self._tokens[commit] = tokens
            self._requests.append(commit)
        log.info("Loaded %d commit snapshots from %s", len(self._requests), self.cfg.processed)

    def conv_id(self, request) -> str:
        # All commits belong to a single "conversation" (the repo)
        return self.cfg.name

    def get_tokens(self, request) -> list[int]:
        return self._tokens[request]
