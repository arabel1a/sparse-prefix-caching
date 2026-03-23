"""Wikipedia revision dataset — article edit histories.

Each article is a "conversation", each revision is a "request".
Successive revisions of the same article share a long common prefix,
making this ideal for prefix caching benchmarks.

Raw data: data/wikipedia/{article_slug}.jsonl
Each line: {"rev_id": int, "timestamp": str, "text": str}
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests as http_requests
from omegaconf import DictConfig
from tqdm.auto import tqdm

from spase_cache.datasets.base import Dataset

log = logging.getLogger(__name__)

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "EditHistoryResearch/1.0 (research@example.com)"}


def _slug(title: str) -> str:
    return title.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def _fetch_revisions(title: str, max_revisions: int = 500) -> list[dict]:
    """Fetch up to max_revisions most recent revisions, returned oldest-first."""
    try:
        import mwparserfromhell
    except ImportError:
        raise ImportError("pip install mwparserfromhell  (needed for wikipedia fetch)")

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
            resp = http_requests.get(API, params=params, timeout=30, headers=HEADERS)
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


class WikipediaDataset(Dataset):
    """Revision dataset from Wikipedia article edit histories."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._tokens = {}  # (slug, rev_id) -> list[int]

    # -- prepare ---------------------------------------------------------------

    def prepare(self, tokenizer) -> None:
        cfg = self.cfg
        raw_dir = Path(cfg.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Fetch if needed
        if not cfg.get("skip_fetch", True):
            articles = list(cfg.fetch.articles)
            max_rev = cfg.fetch.get("max_revisions", 500)
            min_words = cfg.fetch.get("min_words", 6000)
            for title in articles:
                slug = _slug(title)
                outpath = raw_dir / f"{slug}.jsonl"
                if outpath.exists():
                    n = sum(1 for _ in open(outpath))
                    log.info("[skip] %s: %s already exists (%d revisions)", title, outpath, n)
                    continue
                log.info("[fetch] %s (up to %d revisions)...", title, max_rev)
                revisions = _fetch_revisions(title, max_rev)
                kept = [r for r in revisions if len(r["text"].split()) >= min_words]
                log.info("  %d total -> %d with >= %d words", len(revisions), len(kept), min_words)
                with open(outpath, "w") as f:
                    for r in kept:
                        f.write(json.dumps(r) + "\n")

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
                        "rev_id": rec["rev_id"],
                        "text": rec["text"],
                    })

        log.info("Read %d revisions from %d articles", len(all_rows), len(jsonl_files))

        # Tokenize
        max_revisions = cfg.get("max_revisions", 500)
        # Group by article, limit revisions per article
        from collections import defaultdict
        by_article = defaultdict(list)
        for row in all_rows:
            by_article[row["slug"]].append(row)
        limited_rows = []
        for slug in sorted(by_article):
            limited_rows.extend(by_article[slug][:max_revisions])

        max_rows = cfg.get("max_rows", len(limited_rows))
        if len(limited_rows) > max_rows:
            limited_rows = limited_rows[:max_rows]
            log.info("Capped to %d rows (max_rows)", max_rows)

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
            "rev_id": [r["rev_id"] for r in limited_rows],
            "tokens": all_tokens,
            "n_tokens": [len(t) for t in all_tokens],
        })

        # Drop short
        min_seq_len = cfg.get("min_seq_len", 0)
        if min_seq_len > 0:
            n_before = len(df)
            df = df.filter(pl.col("n_tokens") >= min_seq_len)
            log.info("Dropped %d revisions shorter than %d tokens", n_before - len(df), min_seq_len)

        out_path = Path(cfg.processed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        log.info("Saved %d revisions to %s", len(df), out_path)

    # -- load / access ---------------------------------------------------------

    def _load(self) -> None:
        df = pl.read_parquet(self.cfg.processed)
        self._tokens = {}
        self._requests = []
        for slug, rev_id, tokens in zip(
            df["slug"].to_list(), df["rev_id"].to_list(), df["tokens"].to_list()
        ):
            key = (slug, rev_id)
            self._tokens[key] = tokens
            self._requests.append(key)
        log.info("Loaded %d revisions from %s", len(self._requests), self.cfg.processed)

    def conv_id(self, request) -> str:
        return request[0]  # article slug

    def get_tokens(self, request) -> list[int]:
        return self._tokens[request]
