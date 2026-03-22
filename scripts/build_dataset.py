"""Build a dataset of full project states at each commit.

Each commit produces one file named <commit_hash>.txt containing all tracked
files concatenated with filename delimiters.  File ordering is stable across
commits: new files are appended, deleted files are removed in-place, and
renames update the path without changing position.
"""

import subprocess
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent / "hybrid-logarithmic-cache"
OUTPUT_DIR = Path(__file__).parent / "dataset"
DELIMITER = "=" * 60


def run(cmd: str, cwd: Path = REPO_DIR) -> str:
    return subprocess.check_output(cmd, shell=True, cwd=cwd).decode(errors="replace").strip()


def get_commits() -> list[str]:
    """Return commit hashes in chronological order (oldest first)."""
    out = run("git log --reverse --format=%H")
    return out.splitlines()


INCLUDE_EXTENSIONS = {".py"}


def get_tree_files(commit: str) -> list[str]:
    """Return sorted list of .py files in a commit's tree."""
    out = run(f"git ls-tree -r --name-only {commit}")
    if not out:
        return []
    return [f for f in out.splitlines() if Path(f).suffix in INCLUDE_EXTENSIONS]


def get_renames(parent: str, child: str) -> dict[str, str]:
    """Return {old_path: new_path} for renames between two commits."""
    out = run(f"git diff --diff-filter=R -M --name-status {parent} {child}")
    renames = {}
    for line in out.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        # format: R<percent>\told\tnew
        if len(parts) == 3 and parts[0].startswith("R"):
            renames[parts[1]] = parts[2]
    return renames


def get_file_content(commit: str, path: str) -> str:
    """Get file content at a specific commit."""
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{path}"],
            cwd=REPO_DIR,
        ).decode(errors="replace")
    except subprocess.CalledProcessError:
        return ""


def build_snapshot(commit: str, ordered_files: list[str]) -> str:
    """Build concatenated snapshot for a commit."""
    parts = []
    for fpath in ordered_files:
        content = get_file_content(commit, fpath)
        parts.append(f"{DELIMITER}\n# FILE: {fpath}\n{DELIMITER}\n{content}")
    return "\n".join(parts)


def main():
    commits = get_commits()
    print(f"Processing {len(commits)} commits from {REPO_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Stable ordered list of files, maintained across commits
    ordered_files: list[str] = []

    prev_snapshot = None

    for i, commit in enumerate(commits):
        current_files = set(get_tree_files(commit))

        if i == 0:
            # First commit: just use sorted order
            ordered_files = sorted(current_files)
        else:
            prev = commits[i - 1]
            renames = get_renames(prev, commit)

            # 1. Apply renames in-place
            for j, f in enumerate(ordered_files):
                if f in renames:
                    ordered_files[j] = renames[f]

            # 2. Remove deleted files (not in current tree)
            ordered_files = [f for f in ordered_files if f in current_files]

            # 3. Append new files (in current tree but not in ordered list)
            existing = set(ordered_files)
            new_files = sorted(current_files - existing)
            ordered_files.extend(new_files)

        # Build and write snapshot
        snapshot = build_snapshot(commit, ordered_files)
        out_path = OUTPUT_DIR / f"{commit}.txt"
        out_path.write_text(snapshot)
        short = commit[:7]

        # Compute common prefix length with previous snapshot
        prefix_info = ""
        if prev_snapshot is not None:
            common = 0
            for a, b in zip(prev_snapshot, snapshot):
                if a != b:
                    break
                common += 1
            pct = common / max(len(prev_snapshot), 1) * 100
            prefix_info = f", common prefix: {common}/{len(snapshot)} chars ({pct:.0f}%)"
        prev_snapshot = snapshot

        print(f"  [{i+1}/{len(commits)}] {short} — {len(ordered_files)} files, {len(snapshot)} chars{prefix_info}")

    print(f"\nDone. {len(commits)} snapshots written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
