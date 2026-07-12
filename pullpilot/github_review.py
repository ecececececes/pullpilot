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
import re
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

from .diff_parser import parse_diff
from .engines import LLMEngine, StaticAnalysisEngine
from .providers import get_provider
from .reviewer import PullRequest, Reviewer
from .schema import Issue, Review, Severity

_API = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
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


def build_pr(owner: str, repo: str, number: int) -> Tuple[PullRequest, str]:
    """Returns the PullRequest plus the head commit sha (needed to anchor
    inline review comments)."""
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
                       title=pull.get("title", ""), description=pull.get("body") or ""), head_sha


# operators or a leading Python keyword mark a fix as replacement code
_CODE_HINT = re.compile(
    r"[=()\[\]:+\-*/%<>]"
    r"|^\s*(return|if|for|while|def|class|import|from|raise|yield|assert|with)\b")


def _suggestion_block(issue: Issue) -> str:
    """Render suggested_fix as a GitHub ```suggestion block when it looks like
    replacement code (one-click applicable); prose fixes render as plain text.
    Only valid inside line-anchored review comments."""
    fix = (issue.suggested_fix or "").rstrip()
    if not fix.strip():
        return ""
    lines = fix.splitlines()
    if any(l.startswith(("+", "-")) for l in lines):
        code = [l[1:] for l in lines if l.startswith("+")]
        if code:
            return "\n```suggestion\n" + "\n".join(code) + "\n```"
        return f"\n_Suggested fix:_ {fix.strip()}"
    stripped = fix.strip()
    if "\n" not in stripped and not stripped.endswith(".") and _CODE_HINT.search(stripped):
        # keep original indentation: the block replaces the whole line
        return "\n```suggestion\n" + fix.lstrip("\n") + "\n```"
    return f"\n_Suggested fix:_ {stripped}"


def _finding_header(issue: Issue) -> str:
    badge = "✅ verified" if issue.verified else "💭 inferred"
    return (f"{_SEV_MARK.get(issue.severity, issue.severity.value)} · "
            f"_{issue.type.value}_ · {badge} · conf {issue.confidence:.0%}")


def _inline_comment_body(issue: Issue) -> str:
    q = "❓ " if issue.is_question else ""
    return f"**{_finding_header(issue)}**\n\n{q}{issue.explanation}{_suggestion_block(issue)}"


def build_inline_review(review: Review, diff: str, head_sha: str) -> dict:
    """Build a Reviews-API payload: findings on lines GitHub can anchor to
    (lines present in the diff) become inline comments with suggestion blocks;
    the rest fold into the review body so nothing is dropped."""
    changed = {fc.path: set(fc.affected_lines) for fc in parse_diff(diff).files}
    comments, leftover = [], []
    for i in review.sorted_issues():
        if i.line_end in changed.get(i.file, set()):
            c = {"path": i.file, "line": i.line_end, "side": "RIGHT",
                 "body": _inline_comment_body(i)}
            if i.line_start < i.line_end and i.line_start in changed[i.file]:
                c["start_line"] = i.line_start
                c["start_side"] = "RIGHT"
            comments.append(c)
        else:
            leftover.append(i)

    body = ["## 🤖 PullPilot review", "", review.summary or "_(no summary)_", ""]
    if comments:
        body.append(f"**{len(comments)} finding(s)** commented inline.")
    if leftover:
        body.append(f"**{len(leftover)} finding(s)** outside the diff's comment range:")
        body.append("")
        for i in leftover:
            q = "❓ " if i.is_question else ""
            body.append(f"- {_finding_header(i)} · `{i.file}:{i.line_start}-{i.line_end}`")
            body.append(f"  {q}{i.explanation}")
            if i.suggested_fix:
                body.append(f"  - _suggested fix:_ {i.suggested_fix}")
    if not review.issues:
        body.append("✅ No issues found in the changed lines.")
    body += ["", "---", "_PullPilot is an automated assistant; final review "
             "judgement rests with a human._"]
    return {"commit_id": head_sha, "event": "COMMENT",
            "body": "\n".join(body), "comments": comments}


def post_review(owner: str, repo: str, number: int, payload: dict) -> dict:
    return json.loads(_request(
        f"{_API}/repos/{owner}/{repo}/pulls/{number}/reviews",
        method="POST", data=json.dumps(payload).encode()))


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
    pr, head_sha = build_pr(owner, repo, number)

    engine = (StaticAnalysisEngine() if args.provider == "static"
              else LLMEngine(get_provider(args.provider)))
    review = Reviewer(engine, use_context=True, verify=args.verify).review(pr)

    if args.post:
        payload = build_inline_review(review, pr.diff, head_sha)
        try:
            post_review(owner, repo, number, payload)
            print(f"posted inline review to {owner}/{repo}#{number} "
                  f"({len(payload['comments'])} inline, {len(review.issues)} total findings)")
        except urllib.error.HTTPError as e:
            # Reviews API can 422 (e.g. stale sha, unanchorable line);
            # degrade to the single-comment format rather than lose the review.
            print(f"inline review failed ({e.code}), falling back to a comment")
            post_comment(owner, repo, number, format_review_markdown(review))
            print(f"posted review to {owner}/{repo}#{number} ({len(review.issues)} findings)")
    else:
        print(format_review_markdown(review))


if __name__ == "__main__":
    main()
