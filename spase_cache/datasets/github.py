"""GitHub file revision dataset — edit histories of individual files.

Each file (repo+path) is a "conversation", each commit version is a "request".
Successive versions share a long common prefix, good for prefix caching benchmarks.

Raw data: data/github/{repo_slug}__{file_slug}.jsonl
Each line: {"commit": str, "timestamp": str, "message": str, "text": str}
"""
import json
import logging
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from spase_cache.datasets.base import Dataset

log = logging.getLogger(__name__)


def _slug(repo: str, filepath: str) -> str:
    return f"{repo.replace('/', '_')}__{filepath.replace('/', '_')}"


def _run(cmd: str, cwd: str = None, timeout: int = 300) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    return r.stdout.strip()


def _fetch_file_history(repo: str, filepath: str, clone_dir: str, max_commits: int = 1000) -> list[dict]:
    """Clone repo (blobless) and extract all versions of a file, oldest-first."""
    repo_url = f"https://github.com/{repo}.git"
    repo_dir = Path(clone_dir) / repo.replace("/", "_")

    if not repo_dir.exists():
        log.info("  Cloning %s (filter=blob:none)...", repo)
        subprocess.run(
            f"git clone --filter=blob:none --no-checkout {repo_url} {repo_dir}",
            shell=True, check=True, capture_output=True, timeout=120,
        )

    log_output = _run(
        f'git log --follow --reverse --format="%H||%aI||%s" -- "{filepath}"',
        cwd=str(repo_dir), timeout=120,
    )
    if not log_output:
        log.warning("  No commits found for %s", filepath)
        return []

    lines = log_output.strip().split("\n")
    log.info("  %d commits touching %s", len(lines), filepath)

    if len(lines) > max_commits:
        step = len(lines) / max_commits
        indices = [int(i * step) for i in range(max_commits)]
        lines = [lines[i] for i in indices]
        log.info("  Sampled down to %d commits", len(lines))

    results = []
    for line in lines:
        parts = line.split("||", 2)
        if len(parts) != 3:
            continue
        commit, timestamp, message = parts
        content = _run(f'git show {commit}:"{filepath}"', cwd=str(repo_dir), timeout=30)
        if not content:
            continue
        results.append({
            "commit": commit,
            "timestamp": timestamp,
            "message": message,
            "text": content,
        })

    return results


class GitHubDataset(Dataset):
    """Revision dataset from GitHub file edit histories."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._tokens = {}  # (slug, commit) -> list[int]

    # -- prepare ---------------------------------------------------------------

    def prepare(self, tokenizer) -> None:
        cfg = self.cfg
        raw_dir = Path(cfg.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Fetch if needed
        if not cfg.get("skip_fetch", True):
            targets = list(cfg.fetch.targets)
            max_commits = cfg.fetch.get("max_commits", 1000)
            min_commits = cfg.fetch.get("min_commits", 50)
            min_words = cfg.fetch.get("min_words", 3000)
            clone_dir = cfg.fetch.get("clone_dir") or tempfile.mkdtemp(prefix="gh_edit_history_")
            log.info("Clone dir: %s", clone_dir)

            for target in targets:
                repo, filepath = target["repo"], target["file"]
                slug = _slug(repo, filepath)
                outpath = raw_dir / f"{slug}.jsonl"

                if outpath.exists():
                    n = sum(1 for _ in open(outpath))
                    log.info("[skip] %s:%s -> %s (%d revisions)", repo, filepath, outpath, n)
                    continue

                log.info("[fetch] %s:%s", repo, filepath)
                history = _fetch_file_history(repo, filepath, clone_dir, max_commits)

                kept = [h for h in history if len(h["text"].split()) >= min_words]
                log.info("  %d versions -> %d with >= %d words", len(history), len(kept), min_words)

                if len(kept) < min_commits:
                    log.info("  Skipping: only %d versions (need %d)", len(kept), min_commits)
                    continue

                with open(outpath, "w") as f:
                    for h in kept:
                        f.write(json.dumps(h) + "\n")

        # Read all JSONL files
        jsonl_files = sorted(raw_dir.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files in {raw_dir}. Set skip_fetch=false to download.")

        all_rows = []
        for jf in jsonl_files:
            slug = jf.stem
            with open(jf) as f:
                for line in f:
                    rec = json.loads(line)
                    all_rows.append({
                        "slug": slug,
                        "commit": rec["commit"],
                        "text": rec["text"],
                    })

        log.info("Read %d revisions from %d files", len(all_rows), len(jsonl_files))

        # Limit revisions per file
        max_revisions = cfg.get("max_revisions", 1000)
        by_file = defaultdict(list)
        for row in all_rows:
            by_file[row["slug"]].append(row)
        limited_rows = []
        for slug in sorted(by_file):
            limited_rows.extend(by_file[slug][:max_revisions])

        # Tokenize
        log.info("Tokenizing %d revisions...", len(limited_rows))
        chunk_size = cfg.get("tokenizer_chunk_size", 16)
        texts = [r["text"] for r in limited_rows]
        all_tokens = []
        for i in tqdm(range(0, len(texts), chunk_size), desc="tokenizing"):
            chunk = texts[i:i + chunk_size]
            for ids in tokenizer(chunk, add_special_tokens=False, truncation=True,
                                 max_length=cfg.max_seq_len)["input_ids"]:
                all_tokens.append(np.array(ids, dtype=np.int32))

        df = pl.DataFrame({
            "slug": [r["slug"] for r in limited_rows],
            "commit": [r["commit"] for r in limited_rows],
            "tokens": all_tokens,
            "n_tokens": [len(t) for t in all_tokens],
        })

        # Drop short
        min_seq_len = cfg.get("min_seq_len", 0)
        if min_seq_len > 0:
            n_before = len(df)
            df = df.filter(pl.col("n_tokens") >= min_seq_len)
            log.info("Dropped %d revisions shorter than %d tokens", n_before - len(df), min_seq_len)

        df = df.head(cfg.get("max_rows", len(df)))

        out_path = Path(cfg.processed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        log.info("Saved %d revisions to %s", len(df), out_path)

    # -- load / access ---------------------------------------------------------

    def _load(self) -> None:
        df = pl.read_parquet(self.cfg.processed)
        self._tokens = {}
        self._requests = []
        for slug, commit, tokens in zip(
            df["slug"].to_list(), df["commit"].to_list(), df["tokens"].to_list()
        ):
            key = (slug, commit)
            self._tokens[key] = tokens
            self._requests.append(key)
        log.info("Loaded %d revisions from %s", len(self._requests), self.cfg.processed)

    def conv_id(self, request) -> str:
        return request[0]  # file slug

    def get_tokens(self, request) -> list[int]:
        return self._tokens[request]
