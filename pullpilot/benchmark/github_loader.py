"""Build a REAL benchmark by pulling bug-fix commits from GitHub.

For a bug-fix commit, the commit's version of a file is the FIXED code and its
parent is the BUGGY code. We reverse that into a PR that re-introduces the bug
(fixed -> buggy), with the touched lines as ground truth.

Auto-discovery scans a repo's recent commits, keeps the ones whose message looks
like a bug fix and that change a single small .py file, and builds a dataset.

    # auto-find bug fixes in a repo (recommended):
    python -m pullpilot.benchmark.github_loader --owner pallets --repo flask \
        --discover 8 --out data/examples/real.json

    # or one commit you already know is a bug fix:
    python -m pullpilot.benchmark.github_loader --owner OWNER --repo REPO \
        --sha FIX_SHA --path path/to/file.py --out data/examples/real.json

Set GITHUB_TOKEN (a free Personal Access Token) to avoid the 60/hour limit.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

from .build_dataset import BenchmarkPR, make_buggy_pr, save_dataset

_API = "https://api.github.com"
_RAW = "https://raw.githubusercontent.com"
_RETRIES = 4

BUGFIX_KEYWORDS = (
    "fix", "bug", "error", "crash", "regression", "incorrect", "wrong",
    "broken", "typo", "fixes", "fixed", "resolve", "fault", "fail",
)


def _headers() -> dict:
    h = {"User-Agent": "pullpilot", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _open(url: str) -> bytes:
    last = None
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(url, headers=_headers())
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 429, 500, 502, 503) and attempt < _RETRIES - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError as e:
            last = e
            if attempt < _RETRIES - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    raise last  # type: ignore


def _get_json(url: str) -> dict:
    return json.loads(_open(url).decode())


def _get_text(url: str) -> str:
    return _open(url).decode("utf-8", errors="replace")


def list_commits(owner: str, repo: str, per_page: int = 100) -> list:
    return _get_json(f"{_API}/repos/{owner}/{repo}/commits?per_page={per_page}")


def commit_info(owner: str, repo: str, sha: str) -> dict:
    return _get_json(f"{_API}/repos/{owner}/{repo}/commits/{sha}")


def fetch_file(owner: str, repo: str, sha: str, path: str) -> str:
    return _get_text(f"{_RAW}/{owner}/{repo}/{sha}/{path}")


def build_pr_from_commit(owner: str, repo: str, fix_sha: str, path: str,
                         pr_id: Optional[str] = None) -> BenchmarkPR:
    info = commit_info(owner, repo, fix_sha)
    parents = info.get("parents", [])
    if not parents:
        raise ValueError("commit has no parent")
    parent_sha = parents[0]["sha"]
    fixed = fetch_file(owner, repo, fix_sha, path)       # after fix
    buggy = fetch_file(owner, repo, parent_sha, path)    # before fix (buggy)
    title = (info.get("commit", {}).get("message", "") or "").splitlines()[0][:80]
    pr_id = pr_id or f"{repo}-{fix_sha[:7]}"
    return make_buggy_pr(pr_id, path, buggy, fixed, category="real",
                         title=title, description=f"{owner}/{repo}@{fix_sha[:7]}")


def _looks_like_bugfix(message: str) -> bool:
    first = (message or "").lower().splitlines()[0] if message else ""
    return any(k in first for k in BUGFIX_KEYWORDS)


def discover_bugfix_commits(owner: str, repo: str, n: int = 8, scan: int = 100,
                            only_bugfix: bool = True,
                            max_changes: int = 40) -> List[Tuple[str, str]]:
    """Scan recent commits; keep bug-fix-looking ones that modify a single small
    .py file. Returns (sha, path) pairs."""
    commits = list_commits(owner, repo, per_page=min(scan, 100))
    found: List[Tuple[str, str]] = []
    for c in commits:
        msg = c.get("commit", {}).get("message", "")
        if only_bugfix and not _looks_like_bugfix(msg):
            continue
        try:
            info = commit_info(owner, repo, c["sha"])
        except Exception:
            continue
        if not info.get("parents"):
            continue
        py = [f for f in info.get("files", [])
              if f.get("filename", "").endswith(".py")
              and f.get("status") == "modified"
              and 1 <= f.get("changes", 999) <= max_changes
              and "test" not in f.get("filename", "").lower()]
        if not py:
            continue
        py.sort(key=lambda f: f.get("changes", 999))
        found.append((c["sha"], py[0]["filename"]))
        if len(found) >= n:
            break
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--sha")
    ap.add_argument("--path")
    ap.add_argument("--discover", type=int, default=0,
                    help="auto-find N bug-fix commits")
    ap.add_argument("--scan", type=int, default=100,
                    help="how many recent commits to scan (max 100)")
    ap.add_argument("--any", action="store_true",
                    help="don't require a bug-fix-looking message")
    ap.add_argument("--out", default="data/examples/real.json")
    args = ap.parse_args()

    prs: List[BenchmarkPR] = []
    if args.discover:
        pairs = discover_bugfix_commits(args.owner, args.repo, args.discover,
                                        scan=args.scan, only_bugfix=not args.any)
        if not pairs:
            print("No matching commits found. Try a different repo, raise --scan, "
                  "or add --any to drop the bug-fix message filter.")
        for sha, path in pairs:
            try:
                pr = build_pr_from_commit(args.owner, args.repo, sha, path)
                prs.append(pr)
                print(f"  + {pr.id}  {path}  gt_lines={pr.ground_truth_lines}  \"{pr.title[:50]}\"")
            except Exception as exc:
                print(f"  ! skip {sha[:7]} {path}: {exc}")
    else:
        if not (args.sha and args.path):
            ap.error("provide --sha and --path, or --discover N")
        pr = build_pr_from_commit(args.owner, args.repo, args.sha, args.path)
        prs.append(pr)
        print(f"  + {pr.id}  {args.path}  gt_lines={pr.ground_truth_lines}")

    if prs:
        save_dataset(prs, os.path.abspath(args.out))
        print(f"\nwrote {len(prs)} real PR(s) to {args.out}")
    else:
        print("no PRs built")


if __name__ == "__main__":
    main()
