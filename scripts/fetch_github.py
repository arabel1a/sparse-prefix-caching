"""
Extract full edit histories of individual files from GitHub repositories.

For each target file: clones the repo (shallow history of that file only),
extracts every version, saves as JSONL.

Output: dataset/github/{repo_slug}__{file_slug}.jsonl
Each line: {"commit": str, "timestamp": str, "message": str, "text": str}
Commits are ordered oldest-first.

Usage:
    python dataset/fetch_github.py                  # default targets
    python dataset/fetch_github.py --targets '{"repo":"python/cpython","file":"Lib/test/test_typing.py"}'
    python dataset/fetch_github.py --min-commits 100

Requires: git (shell)
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

# Files known to be long-lived, heavily edited, and 8K+ tokens
DEFAULT_TARGETS = [
    {"repo": "python/cpython", "file": "Lib/test/test_typing.py"},
    {"repo": "python/cpython", "file": "Lib/test/test_asyncio/test_tasks.py"},
    {"repo": "python/cpython", "file": "Doc/whatsnew/3.12.rst"},
    {"repo": "pandas-dev/pandas", "file": "pandas/core/frame.py"},
    {"repo": "django/django", "file": "django/db/models/query.py"},
    {"repo": "rust-lang/rust", "file": "compiler/rustc_parse/src/parser/expr.rs"},
    {"repo": "torvalds/linux", "file": "kernel/sched/fair.c"},
    {"repo": "tensorflow/tensorflow", "file": "tensorflow/python/ops/math_ops.py"},
]


def run(cmd: str, cwd: str = None, timeout: int = 300) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    return r.stdout.strip()


def extract_file_history(repo: str, filepath: str, clone_dir: str, max_commits: int = 1000) -> list[dict]:
    repo_url = f"https://github.com/{repo}.git"
    repo_dir = Path(clone_dir) / repo.replace("/", "_")

    if not repo_dir.exists():
        print(f"  Cloning {repo} (filter=blob:none for speed)...")
        subprocess.run(
            f"git clone --filter=blob:none --no-checkout {repo_url} {repo_dir}",
            shell=True, check=True, capture_output=True, timeout=120,
        )

    # Get all commits touching this file, oldest first
    log_output = run(
        f'git log --follow --reverse --format="%H||%aI||%s" -- "{filepath}"',
        cwd=str(repo_dir),
        timeout=120,
    )

    if not log_output:
        print(f"  No commits found for {filepath}")
        return []

    lines = log_output.strip().split("\n")
    print(f"  {len(lines)} commits touching {filepath}")

    if len(lines) > max_commits:
        # sample evenly
        step = len(lines) / max_commits
        indices = [int(i * step) for i in range(max_commits)]
        lines = [lines[i] for i in indices]
        print(f"  Sampled down to {len(lines)} commits")

    results = []
    for line in lines:
        parts = line.split("||", 2)
        if len(parts) != 3:
            continue
        commit, timestamp, message = parts

        # Get file content at this commit
        content = run(f'git show {commit}:"{filepath}"', cwd=str(repo_dir), timeout=30)
        if not content:
            # File might not exist at this commit (renamed, etc.)
            continue

        results.append({
            "commit": commit,
            "timestamp": timestamp,
            "message": message,
            "text": content,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=None,
                        help="JSON objects with 'repo' and 'file' keys")
    parser.add_argument("--clone-dir", type=str, default=None,
                        help="Where to clone repos (default: temp dir)")
    parser.add_argument("--out-dir", type=str, default="dataset/github")
    parser.add_argument("--min-commits", type=int, default=50,
                        help="Skip files with fewer commits than this")
    parser.add_argument("--min-words", type=int, default=3000,
                        help="Skip revisions shorter than this many words")
    parser.add_argument("--max-commits", type=int, default=1000)
    args = parser.parse_args()

    if args.targets:
        targets = [json.loads(t) for t in args.targets]
    else:
        targets = DEFAULT_TARGETS

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    clone_dir = args.clone_dir or tempfile.mkdtemp(prefix="gh_edit_history_")
    print(f"Clone dir: {clone_dir}")

    for target in targets:
        repo, filepath = target["repo"], target["file"]
        slug = f"{repo.replace('/', '_')}__{filepath.replace('/', '_')}"
        outpath = out / f"{slug}.jsonl"

        if outpath.exists():
            n = sum(1 for _ in open(outpath))
            print(f"[skip] {repo}:{filepath} -> {outpath} ({n} revisions)")
            continue

        print(f"[fetch] {repo}:{filepath}")
        try:
            history = extract_file_history(repo, filepath, clone_dir, args.max_commits)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Filter by length
        kept = [h for h in history if len(h["text"].split()) >= args.min_words]
        print(f"  {len(history)} versions -> {len(kept)} with >= {args.min_words} words")

        if len(kept) < args.min_commits:
            print(f"  Skipping: only {len(kept)} versions (need {args.min_commits})")
            continue

        with open(outpath, "w") as f:
            for h in kept:
                f.write(json.dumps(h) + "\n")

        print(f"  -> {outpath}")

    print(f"\nDone. Repos cloned in: {clone_dir}")
    print("You can delete the clone dir when finished.")


if __name__ == "__main__":
    main()
