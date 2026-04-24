"""Discover GitHub files suitable for revision-based benchmarking.

Finds files in popular repos that:
  1. Have enough commits (--min-commits, default 50)
  2. Current size falls in a word-count window (--min-words / --max-words)

Outputs a YAML list of {repo, file} targets to stdout or --output file.

Usage:
    uv run python scripts/discover_github_files.py --output conf/data/github_targets.yaml
    uv run python scripts/discover_github_files.py --repos python/cpython django/django --max-files-per-repo 20
"""
import argparse
import json
import subprocess
import tempfile
import logging
from pathlib import Path
from tqdm import tqdm
import yaml

log = logging.getLogger(__name__)

DEFAULT_REPOS = [
    "python/cpython",
    "django/django",
    "pallets/flask",
    "psf/requests",
    "encode/httpx",
    "torvalds/linux",
    "rust-lang/rust",
    "golang/go",
    "nodejs/node",
    "tensorflow/tensorflow",
    "pytorch/pytorch",
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
    "numpy/numpy",
    "matplotlib/matplotlib",
    "ansible/ansible",
    "home-assistant/core",
    "saltstack/salt",
    "celery/celery",
    "pallets/jinja",
    "sqlalchemy/sqlalchemy",
    "encode/starlette",
    "tiangolo/fastapi",
    "pytest-dev/pytest",
    "psf/black",
    "sympy/sympy",
    "scrapy/scrapy",
    "bokeh/bokeh",
    "ipython/ipython",
    "httpie/cli",
]


def _run(cmd, cwd=None, timeout=300):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    return r.stdout.strip()


def clone_repo(repo, clone_dir):
    repo_dir = Path(clone_dir) / repo.replace("/", "_")
    if not repo_dir.exists():
        log.info("Cloning %s (blobless)...", repo)
        subprocess.run(
            f"git clone --filter=blob:none --no-checkout https://github.com/{repo}.git {repo_dir}",
            shell=True, check=True, capture_output=True, timeout=180,
        )
    return repo_dir


def find_files_with_many_commits(repo_dir, min_commits, extensions):
    """Find files with >= min_commits commits using git shortlog."""
    ext_filter = " ".join(f'"*.{e}"' for e in extensions)
    # Use git log on HEAD only (not --all) to avoid timeout on huge repos.
    # --diff-filter=M is slow on blobless clones, so just list touched files.
    cmd = f"git log -n 10000 --name-only --format= -- {ext_filter}"
    output = _run(cmd, cwd=str(repo_dir), timeout=600)
    if not output:
        return {}

    counts = {}
    for line in tqdm(output.split("\n")):
        line = line.strip()
        if line:
            counts[line] = counts.get(line, 0) + 1

    return {f: c for f, c in counts.items() if c >= min_commits}


def check_file_size(repo_dir, filepath, min_words, max_words):
    """Check if the current version of a file has word count in range."""
    content = _run(f'git show HEAD:"{filepath}"', cwd=str(repo_dir), timeout=30)
    if not content:
        return None
    words = len(content.split())
    if min_words <= words <= max_words:
        return words
    return None


def discover_targets(
    repos, clone_dir, min_commits, min_words, max_words,
    max_files_per_repo, extensions, max_total,
):
    targets = []
    for repo in repos:
        if len(targets) >= max_total:
            break
        log.info("Processing %s...", repo)
        try:
            repo_dir = clone_repo(repo, clone_dir)
        except Exception as e:
            log.warning("Failed to clone %s: %s", repo, e)
            continue

        file_commits = find_files_with_many_commits(repo_dir, min_commits, extensions)
        # Sort by commit count descending
        ranked = sorted(file_commits.items(), key=lambda x: -x[1])
        log.info("  %d files with >= %d commits", len(ranked), min_commits)

        added = 0
        for filepath, n_commits in ranked:
            if added >= max_files_per_repo or len(targets) >= max_total:
                break
            words = check_file_size(repo_dir, filepath, min_words, max_words)
            if words is not None:
                targets.append({"repo": repo, "file": filepath, "commits": n_commits, "words": words})
                log.info("    + %s (%d commits, %d words)", filepath, n_commits, words)
                added += 1

        log.info("  Added %d files from %s", added, repo)

    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", nargs="+", default=None, help="Repos to scan (default: built-in list)")
    parser.add_argument("--clone-dir", default=None, help="Dir to clone repos into (default: tempdir)")
    parser.add_argument("--min-commits", type=int, default=50)
    parser.add_argument("--min-words", type=int, default=2000)
    parser.add_argument("--max-words", type=int, default=6000)
    parser.add_argument("--max-files-per-repo", type=int, default=15)
    parser.add_argument("--max-total", type=int, default=200)
    parser.add_argument("--extensions", nargs="+", default=["py", "c", "rs", "go", "js", "ts", "java", "rb"])
    parser.add_argument("--output", "-o", default=None, help="Output YAML file (default: stdout)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    repos = args.repos or DEFAULT_REPOS
    clone_dir = args.clone_dir or tempfile.mkdtemp(prefix="gh_discover_")
    log.info("Clone dir: %s", clone_dir)

    targets = discover_targets(
        repos, clone_dir, args.min_commits, args.min_words, args.max_words,
        args.max_files_per_repo, args.extensions, args.max_total,
    )

    # Strip metadata for output
    output = [{"repo": t["repo"], "file": t["file"]} for t in targets]

    log.info("Discovered %d files total", len(output))

    yaml_str = yaml.dump(output, default_flow_style=False, sort_keys=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(yaml_str)
        log.info("Written to %s", args.output)
    else:
        print(yaml_str)


if __name__ == "__main__":
    main()
