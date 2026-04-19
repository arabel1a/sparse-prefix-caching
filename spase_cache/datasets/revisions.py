"""Revision-based datasets — successive versions of the same document.

All datasets here share the same structure:
  - One document/file/repo is a "conversation" (conv_id)
  - Each revision is a "request" sharing a long prefix with neighbors
  - Subclasses override _load_raw() to return list[dict] with keys:
    conv_id (str), rev_id (str), text (str)
"""
import json
import logging
import subprocess
import tempfile
import time
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import requests as http_requests
import yaml
from omegaconf import DictConfig
from tqdm.auto import tqdm

from spase_cache.datasets.base import Dataset

log = logging.getLogger(__name__)


class RevisionDataset(Dataset):
    """Base for revision-based datasets."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._tokens = {}  # (conv_id, rev_id) -> list[int]

    @abstractmethod
    def _load_raw(self) -> list[dict]:
        """Return list of {"conv_id": str, "rev_id": str, "text": str}."""

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
            "rev_id": [r["rev_id"] for r in rows],
            "tokens": all_tokens,
            "n_tokens": [len(t) for t in all_tokens],
        })

        n_before = len(df)
        df = df.filter(pl.col("n_tokens") >= cfg.min_seq_len)
        log.info("Dropped %d rows shorter than %d tokens", n_before - len(df), cfg.min_seq_len)

        n_before = len(df)
        df = df.filter(pl.col("n_tokens") < cfg.max_seq_len)
        if n_before - len(df) > 0:
            log.info("Dropped %d truncated rows (>= %d tokens)", n_before - len(df), cfg.max_seq_len)

        out_path = Path(cfg.processed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        log.info("Saved %d rows to %s", len(df), out_path)

    # -- load / access ---------------------------------------------------------

    def _load(self) -> None:
        df = pl.read_parquet(self.cfg.processed)
        self._tokens = {}
        self._requests = []
        for conv_id, rev_id, tokens in zip(
            df["conv_id"].to_list(), df["rev_id"].to_list(), df["tokens"].to_list()
        ):
            key = (conv_id, rev_id)
            self._tokens[key] = tokens
            self._requests.append(key)
        log.info("Loaded %d rows from %s", len(self._requests), self.cfg.processed)

    def conv_id(self, request) -> str:
        return str(request[0])

    def get_tokens(self, request) -> list[int]:
        return self._tokens[request]


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_HEADERS = {"User-Agent": "EditHistoryResearch/1.0 (research@example.com)"}


def _wiki_slug(title: str) -> str:
    return title.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def _fetch_revisions(title: str, max_revisions: int = 500) -> list[dict]:
    """Fetch up to max_revisions most recent revisions, returned oldest-first."""
    import mwparserfromhell

    def wiki_to_plain(wikitext: str) -> str:
        return mwparserfromhell.parse(wikitext).strip_code()

    revisions = []
    params = {
        "action": "query", "titles": title,
        "prop": "revisions", "rvprop": "ids|timestamp|content",
        "rvslots": "main", "rvlimit": 50, "rvdir": "older", "format": "json",
    }

    while len(revisions) < max_revisions:
        for attempt in range(5):
            resp = http_requests.get(_WIKI_API, params=params, timeout=30, headers=_WIKI_HEADERS)
            if resp.status_code == 200:
                break
            time.sleep(2 ** attempt)
        else:
            log.warning("Failed to fetch %s after 5 retries, stopping at %d revisions",
                        title, len(revisions))
            break

        data = resp.json()
        page = next(iter(data["query"]["pages"].values()))
        if "revisions" not in page:
            break

        for rev in page["revisions"]:
            content = rev["slots"]["main"].get("*", "")
            revisions.append({
                "rev_id": rev["revid"],
                "timestamp": rev["timestamp"],
                "text": wiki_to_plain(content),
            })
            if len(revisions) >= max_revisions:
                break

        if "continue" not in data or len(revisions) >= max_revisions:
            break
        params["rvcontinue"] = data["continue"]["rvcontinue"]
        time.sleep(0.5)

    revisions.reverse()
    return revisions


class WikipediaDataset(RevisionDataset):
    """Revision dataset from Wikipedia article edit histories."""

    def _load_raw(self) -> list[dict]:
        cfg = self.cfg
        raw_dir = Path(cfg.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Fetch if needed
        if not cfg.skip_fetch:
            for title in cfg.fetch.articles:
                slug = _wiki_slug(title)
                outpath = raw_dir / f"{slug}.jsonl"
                if outpath.exists():
                    n = sum(1 for _ in open(outpath))
                    log.info("[skip] %s: %s already exists (%d revisions)", title, outpath, n)
                    continue
                log.info("[fetch] %s (up to %d revisions)...", title, cfg.fetch.max_revisions)
                revisions = _fetch_revisions(title, cfg.fetch.max_revisions)
                kept = [r for r in revisions
                        if cfg.fetch.min_words <= len(r["text"].split()) <= cfg.fetch.max_words]
                log.info("  %d total -> %d with %d-%d words",
                         len(revisions), len(kept), cfg.fetch.min_words, cfg.fetch.max_words)
                with open(outpath, "w") as f:
                    for r in kept:
                        f.write(json.dumps(r) + "\n")

        # Read all JSONL files
        jsonl_files = sorted(raw_dir.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files in {raw_dir}. Set skip_fetch=false to download.")

        all_rows = []
        for jf in jsonl_files[:cfg.max_convs]:
            slug = jf.stem
            with open(jf) as f:
                for line in f:
                    rec = json.loads(line)
                    all_rows.append({"slug": slug, "rev_id": rec["rev_id"], "text": rec["text"]})

        log.info("Read %d revisions from %d articles", len(all_rows), len(jsonl_files))

        # Limit revisions per article
        by_article = defaultdict(list)
        for row in all_rows:
            by_article[row["slug"]].append(row)

        rows = []
        for slug in sorted(by_article):
            for r in by_article[slug][:cfg.max_revisions]:
                rows.append({"conv_id": slug, "rev_id": str(r["rev_id"]), "text": r["text"]})

        log.info("WikipediaDataset: %d articles, %d revisions", len(by_article), len(rows))
        return rows


# ---------------------------------------------------------------------------
# GitHub file revisions
# ---------------------------------------------------------------------------

def _gh_slug(repo: str, filepath: str) -> str:
    return f"{repo.replace('/', '_')}__{filepath.replace('/', '_')}"


def _git_run(cmd: str, cwd: str = None, timeout: int = 300) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    return r.stdout.strip()


def _fetch_file_history(repo: str, filepath: str, clone_dir: str,
                        max_commits: int = 1000, timeout=120) -> list[dict]:
    """Clone repo (blobless) and extract all versions of a file, oldest-first."""
    repo_url = f"https://github.com/{repo}.git"
    repo_dir = Path(clone_dir) / repo.replace("/", "_")

    if not repo_dir.exists():
        log.info("  Cloning %s (filter=blob:none)...", repo)
        subprocess.run(
            f"git clone --filter=blob:none --no-checkout {repo_url} {repo_dir}",
            shell=True, check=True, capture_output=True, timeout=timeout,
        )

    log_output = _git_run(
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
        content = _git_run(f'git show {commit}:"{filepath}"', cwd=str(repo_dir), timeout=timeout)
        if not content:
            continue
        results.append({
            "commit": commit, "timestamp": timestamp, "message": message, "text": content,
        })

    return results


class GitHubDataset(RevisionDataset):
    """Revision dataset from GitHub file edit histories."""

    def _load_raw(self) -> list[dict]:
        cfg = self.cfg
        raw_dir = Path(cfg.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Fetch if needed
        if not cfg.skip_fetch:
            if cfg.fetch.targets_file and Path(cfg.fetch.targets_file).exists():
                targets = yaml.safe_load(Path(cfg.fetch.targets_file).read_text()) or []
                log.info("Loaded %d targets from %s", len(targets), cfg.fetch.targets_file)
            else:
                targets = list(cfg.fetch.targets)
            clone_dir = cfg.fetch.clone_dir or tempfile.mkdtemp(prefix="gh_edit_history_")
            log.info("Clone dir: %s", clone_dir)

            for target in targets:
                repo, filepath = target["repo"], target["file"]
                slug = _gh_slug(repo, filepath)
                outpath = raw_dir / f"{slug}.jsonl"

                if outpath.exists():
                    n = sum(1 for _ in open(outpath))
                    log.info("[skip] %s:%s -> %s (%d revisions)", repo, filepath, outpath, n)
                    continue

                log.info("[fetch] %s:%s", repo, filepath)
                history = _fetch_file_history(
                    repo, filepath, clone_dir, cfg.fetch.max_commits, cfg.fetch.timeout)

                kept = [h for h in history
                        if cfg.fetch.min_words <= len(h["text"].split()) <= cfg.fetch.max_words]
                log.info("  %d versions -> %d with %d-%d words",
                         len(history), len(kept), cfg.fetch.min_words, cfg.fetch.max_words)

                if len(kept) < cfg.fetch.min_commits:
                    log.info("  Skipping: only %d versions (need %d)", len(kept), cfg.fetch.min_commits)
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
                    all_rows.append({"slug": slug, "commit": rec["commit"], "text": rec["text"]})

        log.info("Read %d revisions from %d files", len(all_rows), len(jsonl_files))

        # Limit per file, then cap conversations
        by_file = defaultdict(list)
        for row in all_rows:
            by_file[row["slug"]].append(row)

        rows = []
        for slug in sorted(list(by_file.keys())[:cfg.max_convs]):
            for r in by_file[slug][:cfg.max_revisions]:
                rows.append({"conv_id": slug, "rev_id": r["commit"], "text": r["text"]})

        log.info("GitHubDataset: %d files, %d revisions", len(by_file), len(rows))
        return rows


# ---------------------------------------------------------------------------
# Git project (full-repo snapshots)
# ---------------------------------------------------------------------------

_SNAPSHOT_DELIM = "=" * 60


def _git_project_run(cmd: str, cwd: Path) -> str:
    return subprocess.check_output(cmd, shell=True, cwd=cwd).decode(errors="replace").strip()


def _get_commits(repo_dir: Path) -> list[str]:
    out = _git_project_run("git log --reverse --format=%H", repo_dir)
    return out.splitlines() if out else []


def _get_tree_files(repo_dir: Path, commit: str, extensions: set[str]) -> list[str]:
    out = _git_project_run(f"git ls-tree -r --name-only {commit}", repo_dir)
    if not out:
        return []
    return [f for f in out.splitlines() if Path(f).suffix in extensions]


def _get_renames(repo_dir: Path, parent: str, child: str) -> dict[str, str]:
    out = _git_project_run(f"git diff --diff-filter=R -M --name-status {parent} {child}", repo_dir)
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
        parts.append(f"{_SNAPSHOT_DELIM}\n# FILE: {fpath}\n{_SNAPSHOT_DELIM}\n{content}")
    return "\n".join(parts)


class GitProjectDataset(RevisionDataset):
    """Revision dataset from a git repository's commit history."""

    def _load_raw(self) -> list[dict]:
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
            rows.append({"conv_id": cfg.name, "rev_id": commit, "text": snapshot})

        log.info("GitProjectDataset: %d snapshots", len(rows))
        return rows
