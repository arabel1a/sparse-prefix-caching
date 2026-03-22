"""
Fetch full revision histories of long Wikipedia articles via the MediaWiki API.

Output: dataset/wikipedia/{article_slug}.jsonl
Each line: {"rev_id": int, "timestamp": str, "text": str}
Revisions are ordered oldest-first.

Usage:
    python dataset/fetch_wikipedia.py                     # default article list
    python dataset/fetch_wikipedia.py --articles "Python (programming language)" "Climate change"
    python dataset/fetch_wikipedia.py --max-revisions 500
"""

import argparse
import json
import time
from pathlib import Path

import mwparserfromhell
import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "EditHistoryResearch/1.0 (research@example.com)"}

# Articles known to be long (8-16K+ tokens) with many edits
DEFAULT_ARTICLES = [
    "United States",
    "World War II",
    "Climate change",
    "Python (programming language)",
    "Russia",
    "History of China",
    "Evolution",
    "Solar System",
    "Roman Empire",
    "Machine learning",
]


def wiki_to_plain(wikitext: str) -> str:
    parsed = mwparserfromhell.parse(wikitext)
    return parsed.strip_code()


def fetch_revisions(title: str, max_revisions: int = 1000) -> list[dict]:
    """Fetch up to max_revisions most recent revisions for a given article, returned oldest-first."""
    revisions = []
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "ids|timestamp|content",
        "rvslots": "main",
        "rvlimit": 50,  # API max per request
        "rvdir": "older",  # newest first, we reverse at the end
        "format": "json",
    }

    while len(revisions) < max_revisions:
        for attempt in range(5):
            resp = requests.get(API, params=params, timeout=30, headers=HEADERS)
            if resp.status_code == 200:
                break
            time.sleep(2 ** attempt)
        else:
            print(f"  Failed to fetch after 5 retries, stopping at {len(revisions)} revisions")
            break

        data = resp.json()
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))

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

        time.sleep(0.5)  # be nice to the API

    revisions.reverse()  # back to chronological order (oldest first)
    return revisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", nargs="+", default=DEFAULT_ARTICLES)
    parser.add_argument("--max-revisions", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="dataset/wikipedia")
    parser.add_argument("--min-tokens-approx", type=int, default=6000,
                        help="Skip revisions shorter than this (word-approx)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for title in args.articles:
        slug = title.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        outpath = out / f"{slug}.jsonl"

        if outpath.exists():
            n = sum(1 for _ in open(outpath))
            print(f"[skip] {title}: {outpath} already exists ({n} revisions)")
            continue

        print(f"[fetch] {title} (up to {args.max_revisions} revisions)...")
        revisions = fetch_revisions(title, args.max_revisions)

        # Filter: keep only revisions where the article is long enough
        min_words = args.min_tokens_approx  # rough: 1 word ≈ 1.3 tokens
        kept = [r for r in revisions if len(r["text"].split()) >= min_words]
        print(f"  {len(revisions)} total -> {len(kept)} with >= {min_words} words")

        with open(outpath, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")

        print(f"  -> {outpath}")

    print("Done.")


if __name__ == "__main__":
    main()
