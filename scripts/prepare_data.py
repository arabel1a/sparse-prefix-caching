"""Tokenize raw data and compute prefix overlap statistics.

Supports two data formats:
  - conversation (ShareGPT): multi-turn chat CSV, tokens concatenated across turns
  - revision (GitHub/Wikipedia/git_project): document edit histories as JSONL,
    each entry is the full document at that revision

Usage:
  python prepare_data.py                                          # default dataset
  python prepare_data.py --config-name=toy data=github            # github dataset, toy scale
  python prepare_data.py prepare_data._target_=prepare_data.tokenize
"""
import json
import logging
import subprocess
import tempfile
import time as _time
from pathlib import Path

import numpy as np
import polars as pl
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from spase_cache.utils import setup_output_dir, interleave, resolve_strategies

log = logging.getLogger(__name__)


_HASH_PRIME = 2654435761
_HASH_MASK = (1 << 63) - 1


def _is_revision(cfg):
    return cfg.data.get("format", "conversation") == "revision"


# ---------------------------------------------------------------------------
# Fetchers — download raw data into raw_dir as JSONL
# ---------------------------------------------------------------------------
def fetch_github(cfg):
    """Clone repos and extract file edit histories into JSONL files."""
    fetch = cfg.data.fetch
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    clone_dir = fetch.get("clone_dir") or tempfile.mkdtemp(prefix="gh_edit_history_")
    log.info("GitHub clone dir: %s", clone_dir)

    for target in fetch.targets:
        repo, filepath = target["repo"], target["file"]
        slug = f"{repo.replace('/', '_')}__{filepath.replace('/', '_')}"
        outpath = raw_dir / f"{slug}.jsonl"

        if outpath.exists():
            n = sum(1 for _ in open(outpath))
            log.info("[skip] %s:%s -> %s (%d revisions)", repo, filepath, outpath, n)
            continue

        log.info("[fetch] %s:%s", repo, filepath)
        repo_url = f"https://github.com/{repo}.git"
        repo_dir = Path(clone_dir) / repo.replace("/", "_")

        if not repo_dir.exists():
            log.info("  Cloning %s (filter=blob:none)...", repo)
            subprocess.run(
                f"git clone --filter=blob:none --no-checkout {repo_url} {repo_dir}",
                shell=True, check=True, capture_output=True, timeout=120,
            )

        log_output = subprocess.run(
            f'git log --follow --reverse --format="%H||%aI||%s" -- "{filepath}"',
            shell=True, capture_output=True, text=True, cwd=str(repo_dir), timeout=120,
        ).stdout.strip()

        if not log_output:
            log.warning("  No commits found for %s", filepath)
            continue

        lines = log_output.strip().split("\n")
        log.info("  %d commits touching %s", len(lines), filepath)

        max_commits = fetch.get("max_commits", 1000)
        if len(lines) > max_commits:
            step = len(lines) / max_commits
            indices = [int(i * step) for i in range(max_commits)]
            lines = [lines[i] for i in indices]

        results = []
        for line in lines:
            parts = line.split("||", 2)
            if len(parts) != 3:
                continue
            commit, timestamp, message = parts
            content = subprocess.run(
                f'git show {commit}:"{filepath}"',
                shell=True, capture_output=True, text=True, cwd=str(repo_dir), timeout=30,
            ).stdout.strip()
            if not content:
                continue
            results.append({"commit": commit, "timestamp": timestamp, "message": message, "text": content})

        min_words = fetch.get("min_words", 3000)
        kept = [h for h in results if len(h["text"].split()) >= min_words]
        log.info("  %d versions -> %d with >= %d words", len(results), len(kept), min_words)

        min_commits = fetch.get("min_commits", 50)
        if len(kept) < min_commits:
            log.info("  Skipping: only %d versions (need %d)", len(kept), min_commits)
            continue

        with open(outpath, "w") as f:
            for h in kept:
                f.write(json.dumps(h) + "\n")
        log.info("  -> %s", outpath)


def fetch_wikipedia(cfg):
    """Fetch article revision histories from Wikipedia API into JSONL files."""
    import requests as http_requests
    import mwparserfromhell

    fetch = cfg.data.fetch
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    API = "https://en.wikipedia.org/w/api.php"
    HEADERS = {"User-Agent": "EditHistoryResearch/1.0 (research@example.com)"}

    for title in fetch.articles:
        slug = title.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        outpath = raw_dir / f"{slug}.jsonl"

        if outpath.exists():
            n = sum(1 for _ in open(outpath))
            log.info("[skip] %s: %s already exists (%d revisions)", title, outpath, n)
            continue

        max_revisions = fetch.get("max_revisions", 500)
        log.info("[fetch] %s (up to %d revisions)...", title, max_revisions)

        revisions = []
        params = {
            "action": "query", "titles": title, "prop": "revisions",
            "rvprop": "ids|timestamp|content", "rvslots": "main",
            "rvlimit": 50, "rvdir": "older", "format": "json",
        }

        while len(revisions) < max_revisions:
            for attempt in range(5):
                resp = http_requests.get(API, params=params, timeout=30, headers=HEADERS)
                if resp.status_code == 200:
                    break
                _time.sleep(2 ** attempt)
            else:
                log.warning("  Failed after 5 retries, stopping at %d revisions", len(revisions))
                break

            data = resp.json()
            page = next(iter(data["query"]["pages"].values()))
            if "revisions" not in page:
                break

            for rev in page["revisions"]:
                content = rev["slots"]["main"].get("*", "")
                parsed = mwparserfromhell.parse(content)
                revisions.append({
                    "rev_id": rev["revid"],
                    "timestamp": rev["timestamp"],
                    "text": parsed.strip_code(),
                })
                if len(revisions) >= max_revisions:
                    break

            if "continue" not in data or len(revisions) >= max_revisions:
                break
            params["rvcontinue"] = data["continue"]["rvcontinue"]
            _time.sleep(0.5)

        revisions.reverse()  # chronological order

        min_words = fetch.get("min_words", 6000)
        kept = [r for r in revisions if len(r["text"].split()) >= min_words]
        log.info("  %d total -> %d with >= %d words", len(revisions), len(kept), min_words)

        with open(outpath, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        log.info("  -> %s", outpath)


def fetch_git_project(cfg):
    """Build full-project snapshots at each commit of a git repo."""
    fetch = cfg.data.fetch
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = Path(fetch.get("repo_dir", ".")).resolve()
    extensions = set(fetch.get("extensions", [".py"]))
    DELIMITER = "=" * 60

    outpath = raw_dir / "snapshots.jsonl"
    if outpath.exists():
        n = sum(1 for _ in open(outpath))
        log.info("[skip] %s already exists (%d snapshots)", outpath, n)
        return

    def _run(cmd):
        return subprocess.check_output(cmd, shell=True, cwd=repo_dir).decode(errors="replace").strip()

    commits = _run("git log --reverse --format=%H").splitlines()
    max_revisions = cfg.data.get("max_revisions", 1000)
    if len(commits) > max_revisions:
        step = len(commits) / max_revisions
        commits = [commits[int(i * step)] for i in range(max_revisions)]

    log.info("Processing %d commits from %s", len(commits), repo_dir)

    ordered_files = []
    results = []

    for i, commit in enumerate(commits):
        out = _run(f"git ls-tree -r --name-only {commit}")
        current_files = set(f for f in out.splitlines() if Path(f).suffix in extensions) if out else set()

        if i == 0:
            ordered_files = sorted(current_files)
        else:
            prev = commits[i - 1]
            rename_out = _run(f"git diff --diff-filter=R -M --name-status {prev} {commit}")
            renames = {}
            for line in rename_out.splitlines():
                parts = line.split("\t")
                if len(parts) == 3 and parts[0].startswith("R"):
                    renames[parts[1]] = parts[2]

            for j, f in enumerate(ordered_files):
                if f in renames:
                    ordered_files[j] = renames[f]
            ordered_files = [f for f in ordered_files if f in current_files]
            new_files = sorted(current_files - set(ordered_files))
            ordered_files.extend(new_files)

        # Build snapshot
        parts = []
        for fpath in ordered_files:
            content = subprocess.run(
                ["git", "show", f"{commit}:{fpath}"],
                cwd=repo_dir, capture_output=True,
            ).stdout.decode(errors="replace")
            parts.append(f"{DELIMITER}\n# FILE: {fpath}\n{DELIMITER}\n{content}")
        snapshot = "\n".join(parts)

        timestamp = _run(f"git show -s --format=%aI {commit}")
        results.append({
            "commit": commit,
            "timestamp": timestamp,
            "text": snapshot,
        })

        if (i + 1) % 50 == 0:
            log.info("  [%d/%d] %s — %d files, %d chars", i + 1, len(commits), commit[:7], len(ordered_files), len(snapshot))

    with open(outpath, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info("  -> %s (%d snapshots)", outpath, len(results))


def fetch(out_dir, full_cfg, **_kw):
    """Fetch raw data. Dispatcher based on data config."""
    fmt = full_cfg.data.get("format", "conversation")
    if fmt == "conversation":
        log.info("Conversation format — no fetch needed (raw CSV expected at %s)", full_cfg.data.raw_path)
        return

    if full_cfg.data.get("skip_fetch", False):
        log.info("skip_fetch=True, skipping fetch step")
        return

    raw_dir = Path(full_cfg.data.raw_dir)
    if raw_dir.exists() and any(raw_dir.iterdir()):
        log.info("Raw data already exists at %s, skipping fetch (set skip_fetch=false and delete dir to re-fetch)", raw_dir)
        return

    # Determine fetcher from raw_dir name or explicit type
    raw_dir_name = raw_dir.name
    if "github" in raw_dir_name:
        fetch_github(full_cfg)
    elif "wikipedia" in raw_dir_name or "wiki" in raw_dir_name:
        fetch_wikipedia(full_cfg)
    elif "git_project" in raw_dir_name or "git" in raw_dir_name:
        fetch_git_project(full_cfg)
    else:
        raise ValueError(f"Unknown revision dataset type for raw_dir={raw_dir}")


# ---------------------------------------------------------------------------
# Filter & tokenize — conversation format (ShareGPT)
# ---------------------------------------------------------------------------
def _filter(raw_path, max_rounds, max_rows):
    log.info(f"Loading {raw_path}...")
    df = pl.read_csv(raw_path)
    df = df.with_columns(
        pl.col("message_create_time").str.replace("ts:", "").cast(pl.Float64).alias("ts")
    )

    # drop rows without timestamp
    df = df.filter(pl.col("ts").is_not_null())

    # keep only first max_rounds messages per conversation
    df = df.sort(["url", "message_index"])
    df = df.with_columns(
        pl.col("message_index").rank("ordinal").over("url").alias("_rank")
    ).filter(pl.col("_rank") <= max_rounds).drop("_rank")

    # limit total rows
    df = df.head(max_rows)

    log.info(f"Filtered: {df.n_unique('url')} conversations, {len(df)} rows")
    return df


def tokenize(out_dir, full_cfg, **_kw):
    """Filter raw data, tokenize, and save parquet."""
    if _is_revision(full_cfg):
        return tokenize_revisions(out_dir, full_cfg)

    df = _filter(full_cfg.data.raw_path, full_cfg.data.max_rounds, full_cfg.data.max_rows)

    log.info(f"Loading tokenizer {full_cfg.model.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(full_cfg.model.tokenizer, use_fast=True)

    # format text as '<|role|> text' with \n separator between messages
    df = df.with_columns(
        (pl.col("message_index") == pl.col("message_index").min().over("url")).alias("_is_first")
    )
    df = df.with_columns(
        (pl.when(pl.col("_is_first"))
         .then(pl.lit("<|") + pl.col("role") + pl.lit("|> ") + pl.col("plain_text").fill_null(""))
         .otherwise(pl.lit("\n<|") + pl.col("role") + pl.lit("|> ") + pl.col("plain_text").fill_null(""))
        ).alias("_formatted")
    ).drop("_is_first")

    # tokenize in chunks, converting to numpy immediately to avoid Python int overhead
    # (Python int = 28 bytes each vs numpy int32 = 4 bytes)
    log.info("Tokenizing messages...")
    chunk_size = full_cfg.prepare_data.tokenizer_chunk_size
    formatted = df["_formatted"]
    all_tokens = []
    for i in tqdm(range(0, len(formatted), chunk_size), desc="tokenizing chunks"):
        chunk_texts = formatted[i:i + chunk_size].to_list()
        for ids in tokenizer(chunk_texts, add_special_tokens=False)["input_ids"]:
            all_tokens.append(np.array(ids, dtype=np.int32))
        del chunk_texts
    df = df.drop("_formatted", "plain_text").with_columns(pl.Series("tokens", all_tokens))
    del all_tokens

    # cumulative token count per conversation for truncation
    df = df.with_columns(
        pl.col("tokens").list.len().cum_sum().over("url").alias("cum_tokens")
    )
    n_before = len(df)
    df = df.filter(pl.col("cum_tokens") <= full_cfg.data.max_seq_len)
    log.info(f"Tokenized {len(df)} messages ({n_before - len(df)} truncated)")

    # skip conversations whose total length is below min_seq_len
    min_seq_len = full_cfg.data.get("min_seq_len", 0)
    if min_seq_len > 0:
        conv_total = df.group_by("url").agg(pl.col("tokens").list.len().sum().alias("total_tokens"))
        short_convs = conv_total.filter(pl.col("total_tokens") < min_seq_len)["url"]
        n_convs_before = df.n_unique("url")
        df = df.filter(~pl.col("url").is_in(short_convs))
        log.info(f"Dropped {n_convs_before - df.n_unique('url')} conversations shorter than {min_seq_len} tokens")

    prepared_path = Path(full_cfg.data.processed)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(prepared_path)
    log.info(f"Saved {len(df)} messages to {prepared_path}")


# ---------------------------------------------------------------------------
# Filter & tokenize — revision format (GitHub/Wikipedia/git_project)
# ---------------------------------------------------------------------------
def tokenize_revisions(out_dir, full_cfg, **_kw):
    """Load JSONL revision files from raw_dir, tokenize, save parquet.

    Output parquet has same schema as conversation format:
      url, message_index, role, ts, tokens

    For revisions:
      - url = document identifier (filename slug)
      - message_index = revision number (0, 1, 2, ...)
      - role = "user" (all revisions are treated as requests)
      - ts = synthetic timestamp for ordering
      - tokens = full tokenized text of this revision
    """
    raw_dir = Path(full_cfg.data.raw_dir)
    log.info("Loading revision JSONL files from %s...", raw_dir)

    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files in {raw_dir}")

    log.info(f"Loading tokenizer {full_cfg.model.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(full_cfg.model.tokenizer, use_fast=True)
    chunk_size = full_cfg.prepare_data.tokenizer_chunk_size
    max_revisions = full_cfg.data.get("max_revisions", 1000)
    max_seq_len = full_cfg.data.max_seq_len
    min_seq_len = full_cfg.data.get("min_seq_len", 0)
    max_rows = full_cfg.data.max_rows

    rows = []
    ts_counter = 0.0

    for jsonl_path in jsonl_files:
        doc_id = jsonl_path.stem
        with open(jsonl_path) as f:
            revisions = [json.loads(line) for line in f]

        if len(revisions) > max_revisions:
            step = len(revisions) / max_revisions
            revisions = [revisions[int(i * step)] for i in range(max_revisions)]

        log.info("  %s: %d revisions", doc_id, len(revisions))

        # tokenize in chunks
        texts = [r["text"] for r in revisions]
        all_tok = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            for ids in tokenizer(chunk, add_special_tokens=False)["input_ids"]:
                all_tok.append(np.array(ids, dtype=np.int32))

        for rev_idx, tok in enumerate(all_tok):
            # truncate individual revisions to max_seq_len
            if len(tok) > max_seq_len:
                tok = tok[:max_seq_len]
            if len(tok) < min_seq_len:
                continue
            rows.append({
                "url": doc_id,
                "message_index": rev_idx,
                "role": "user",
                "ts": ts_counter,
                "tokens": tok.tolist(),
            })
            ts_counter += 1.0

        if len(rows) >= max_rows:
            rows = rows[:max_rows]
            break

    df = pl.DataFrame(rows)
    log.info("Total: %d revisions across %d documents", len(df), df.n_unique("url"))

    prepared_path = Path(full_cfg.data.processed)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(prepared_path)
    log.info(f"Saved {len(df)} revisions to {prepared_path}")


# ---------------------------------------------------------------------------
# Overlap & prefix sharing analysis (format-aware)
# ---------------------------------------------------------------------------
def _get_request_tokens(conv_tokens, url, n_msgs, is_rev):
    """Get the token sequence for a request. For revisions, use only current revision."""
    if is_rev:
        return list(conv_tokens[url][1][n_msgs - 1])
    # conversation: concatenate turns 0..n_msgs
    return [t for toks in conv_tokens[url][1][:n_msgs] for t in toks]


def train_test_split(requests, train_frac):
    """Split requests into train (first train_frac) and test (the rest)."""
    n_train = int(len(requests) * train_frac)
    return requests[:n_train], requests[n_train:]


def compute_overlap(out_dir, full_cfg, **_kw):
    """Load tokenized parquet and compute prefix overlap distribution."""
    out_dir = Path(out_dir)
    is_rev = _is_revision(full_cfg)
    prepared_path = Path(full_cfg.data.processed)
    log.info(f"Loading {prepared_path}...")
    df = pl.read_parquet(prepared_path)

    log.info("Computing prefix overlap distribution...")
    user_rows = df.filter(pl.col("role") == "user").sort("ts")
    n_conversations = user_rows.n_unique("url")
    log.info(f"{len(user_rows)} requests from {n_conversations} conversations")

    # build compact lookup: url -> (msg_indices, token_arrays)
    conv_tokens = {}
    for url in df["url"].unique(maintain_order=True).to_list():
        conv = df.filter(pl.col("url") == url).sort("message_index")
        conv_tokens[url] = (
            conv["message_index"].to_list(),
            [np.array(t, dtype=np.int32) for t in conv["tokens"].to_list()],
        )

    iter_urls = user_rows["url"].to_list()
    iter_msg_indices = user_rows["message_index"].to_list()
    del user_rows, df

    requests = list(zip(iter_urls, iter_msg_indices))
    if full_cfg.data.get("interleave", False):
        log.info("Interleaving requests with Poisson arrivals (seed=%d)...", full_cfg.seed)
        requests = interleave(requests, full_cfg.seed)

    train_requests, test_requests = train_test_split(
        requests, full_cfg.data.get("train_frac", 0.5))
    log.info("Train: %d requests, Test: %d requests", len(train_requests), len(test_requests))

    seen = set()

    # warm cache with train split
    log.info("Warming prefix cache with train split...")
    for url, msg_idx in tqdm(train_requests, desc="train"):
        msg_indices, token_lists = conv_tokens[url]
        n_msgs = msg_indices.index(msg_idx) + 1
        toks = _get_request_tokens(conv_tokens, url, n_msgs, is_rev)
        h = 0
        for t in toks:
            h = (h * _HASH_PRIME + int(t) + 1) & _HASH_MASK
            seen.add(h)

    # evaluate on test split
    lcp_lengths = []
    log.info("Simulating non-deleting prefix cache on test split...")
    for url, msg_idx in tqdm(test_requests, desc="test"):
        msg_indices, token_lists = conv_tokens[url]
        n_msgs = msg_indices.index(msg_idx) + 1
        toks = _get_request_tokens(conv_tokens, url, n_msgs, is_rev)

        h = 0
        lcp = 0
        matched = True

        for t in toks:
            h = (h * _HASH_PRIME + int(t) + 1) & _HASH_MASK
            if matched and h in seen:
                lcp += 1
            else:
                matched = False
            seen.add(h)

        lcp_lengths.append(lcp)

    overlap_path = out_dir / "overlap_lcp.json"
    overlap_path.write_text(json.dumps({
        "lcp_lengths": lcp_lengths,
        "n_requests": len(test_requests),
        "n_train_requests": len(train_requests),
        "n_conversations": n_conversations,
    }))

    n_hits = sum(1 for l in lcp_lengths if l > 0)
    log.info(f"{n_hits}/{len(lcp_lengths)} test requests had cache hits "
             f"({n_hits/len(lcp_lengths)*100:.1f}%)")
    log.info(f"Overlap results saved to {overlap_path}")


def compute_prefix_sharing(out_dir, full_cfg, n_sample=200, **_kw):
    """Sample conversations and compute pairwise LCP matrix for trie diagnostics."""
    out_dir = Path(out_dir)
    is_rev = _is_revision(full_cfg)
    df = pl.read_parquet(full_cfg.data.processed)

    urls = df["url"].unique(maintain_order=True).to_list()
    rng = np.random.RandomState(full_cfg.seed)
    if len(urls) > n_sample:
        urls = rng.choice(urls, n_sample, replace=False).tolist()

    conv_seqs = []
    conv_lens = []
    for url in urls:
        conv = df.filter(pl.col("url") == url).sort("message_index")
        if is_rev:
            # for revisions, use the last revision as the representative sequence
            tokens = np.array(conv["tokens"].to_list()[-1], dtype=np.int32)
        else:
            tokens = np.concatenate([np.array(t, dtype=np.int32) for t in conv["tokens"].to_list()])
        conv_seqs.append(tokens)
        conv_lens.append(len(tokens))

    n = len(conv_seqs)
    lcp_matrix = np.zeros((n, n), dtype=np.int32)

    for i in tqdm(range(n), desc="prefix sharing"):
        lcp_matrix[i, i] = conv_lens[i]
        for j in range(i + 1, n):
            min_len = min(conv_lens[i], conv_lens[j])
            if min_len == 0:
                continue
            matches = conv_seqs[i][:min_len] == conv_seqs[j][:min_len]
            lcp = int(np.argmin(matches)) if not matches.all() else min_len
            lcp_matrix[i, j] = lcp
            lcp_matrix[j, i] = lcp

    sharing_path = out_dir / "prefix_sharing.json"
    sharing_path.write_text(json.dumps({
        "lcp_matrix": lcp_matrix.tolist(),
        "conv_lens": conv_lens,
        "n_conversations": n,
    }))
    log.info("Prefix sharing matrix (%d×%d) saved to %s", n, n, sharing_path)


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------
def prepare_all(out_dir, full_cfg, **_kw):
    """Run full pipeline: fetch -> tokenize -> overlap -> prefix sharing."""
    fetch(out_dir, full_cfg)
    tokenize(out_dir, full_cfg)
    compute_overlap(out_dir, full_cfg)
    compute_prefix_sharing(out_dir, full_cfg)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = setup_output_dir(cfg, "prepare_data")
    resolve_strategies(cfg)
    fn = hydra.utils.get_method(cfg.prepare_data._target_)
    fn(out_dir=out_dir, full_cfg=cfg)


if __name__ == "__main__":
    main()
