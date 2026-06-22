"""Run PullPilot on a live pull request and post the review as a PR comment.
This is the GitHub Action integration: in CI it reads the PR from the event
context, reviews it, and comments back.

    # locally (dry run prints the comment instead of posting):
    python -m pullpilot.github_review --owner OWNER --repo REPO --pr 42 \
        --provider anthropic --dry-run

    # in CI: reads GITHUB_REPOSITORY + PR_NUMBER + GITHUB_TOKEN from env, posts.
    python -m pullpilot.github_review --provider anthropic --post

Needs GITHUB_TOKEN (with pull-requests:write to post) and an LLM key.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

from .diff_parser import parse_diff
from .engines import LLMEngine, StaticAnalysisEngine
from .providers import get_provider
from .reviewer import PullRequest, Reviewer
from .schema import Review, Severity

_API = "https://api.github.com"
_RAW = "https://raw.githubusercontent.com"
_RETRIES = 3

_SEV_MARK = {
    Severity.CRITICAL: "🔴 critical",
    Severity.MAJOR: "🟠 major",
    Severity.MINOR: "🟡 minor",
    Severity.INFO: "🔵 info",
}


def _headers(accept: str = "application/vnd.github+json") -> dict:
    h = {"User-Agent": "pullpilot", "Accept": accept}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _request(url: str, method: str = "GET", data: Optional[bytes] = None,
             accept: str = "application/vnd.github+json") -> bytes:
    last = None
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(url, data=data, method=method, headers=_headers(accept))
            if data is not None:
                req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 429, 500, 502, 503) and attempt < _RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError as e:
            last = e
            if attempt < _RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise last  # type: ignore


def get_pull(owner: str, repo: str, number: int) -> dict:
    return json.loads(_request(f"{_API}/repos/{owner}/{repo}/pulls/{number}"))


def get_pull_diff(owner: str, repo: str, number: int) -> str:
    return _request(f"{_API}/repos/{owner}/{repo}/pulls/{number}",
                    accept="application/vnd.github.v3.diff").decode("utf-8", "replace")


def fetch_file(owner: str, repo: str, sha: str, path: str) -> str:
    return _request(f"{_RAW}/{owner}/{repo}/{sha}/{path}",
                    accept="text/plain").decode("utf-8", "replace")


def post_comment(owner: str, repo: str, number: int, body: str) -> dict:
    payload = json.dumps({"body": body}).encode()
    return json.loads(_request(
        f"{_API}/repos/{owner}/{repo}/issues/{number}/comments",
        method="POST", data=payload))


def build_pr(owner: str, repo: str, number: int) -> PullRequest:
    pull = get_pull(owner, repo, number)
    head_sha = pull["head"]["sha"]
    diff = get_pull_diff(owner, repo, number)
    post_files = {}
    for fc in parse_diff(diff).files:
        try:
            post_files[fc.path] = fetch_file(owner, repo, head_sha, fc.path)
        except Exception:
            pass  # deleted/binary/unfetchable file -> review the diff alone
    return PullRequest(diff=diff, post_files=post_files,
                       title=pull.get("title", ""), description=pull.get("body") or "")


def format_review_markdown(review: Review) -> str:
    lines = ["## 🤖 PullPilot review", "", review.summary or "_(no summary)_", ""]
    issues = review.sorted_issues()
    if not issues:
        lines.append("✅ No issues found in the changed lines.")
    else:
        n_verified = sum(1 for i in issues if i.verified)
        lines.append(f"**{len(issues)} finding(s)**"
                     + (f", {n_verified} machine-verified" if n_verified else "") + ":")
        lines.append("")
        for i in issues:
            badge = "✅ verified" if i.verified else "💭 inferred"
            q = "❓ " if i.is_question else ""
            lines.append(
                f"- {_SEV_MARK.get(i.severity, i.severity.value)} · `{i.file}:"
                f"{i.line_start}-{i.line_end}` · _{i.type.value}_ · {badge} · "
                f"conf {i.confidence:.0%}")
            lines.append(f"  {q}{i.explanation}")
            if i.suggested_fix:
                lines.append(f"  - _suggested fix:_ {i.suggested_fix}")
        lines.append("")
    lines.append("---")
    lines.append("_PullPilot is an automated assistant; final review judgement "
                 "rests with a human._")
    return "\n".join(lines)


def _resolve_target(args) -> Tuple[str, str, int]:
    owner, repo, number = args.owner, args.repo, args.pr
    if (not owner or not repo) and os.environ.get("GITHUB_REPOSITORY"):
        owner, repo = os.environ["GITHUB_REPOSITORY"].split("/", 1)
    if number is None and os.environ.get("PR_NUMBER"):
        number = int(os.environ["PR_NUMBER"])
    if not (owner and repo and number):
        raise SystemExit("need --owner/--repo/--pr or GITHUB_REPOSITORY + PR_NUMBER")
    return owner, repo, number


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner")
    ap.add_argument("--repo")
    ap.add_argument("--pr", type=int)
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--verify", action="store_true", help="also run linter/tests")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--post", action="store_true", help="post the review as a PR comment")
    g.add_argument("--dry-run", action="store_true", help="print the review instead")
    args = ap.parse_args()

    owner, repo, number = _resolve_target(args)
    pr = build_pr(owner, repo, number)

    engine = (StaticAnalysisEngine() if args.provider == "static"
              else LLMEngine(get_provider(args.provider)))
    review = Reviewer(engine, use_context=True, verify=args.verify).review(pr)
    body = format_review_markdown(review)

    if args.post:
        post_comment(owner, repo, number, body)
        print(f"posted review to {owner}/{repo}#{number} ({len(review.issues)} findings)")
    else:
        print(body)


if __name__ == "__main__":
    main()
