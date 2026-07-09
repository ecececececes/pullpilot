"""The review pipeline: a fixed, deterministic sequence (not a free-roaming
agent), parameterised by an engine (LLM or static analysis).

    parse diff -> retrieve context -> engine.review(...) -> calibrate
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .calibration import calibrate
from .context_retriever import (
    ContextRetriever,
    NoContextRetriever,
    PythonASTRetriever,
)
from .diff_parser import ParsedDiff, parse_diff
from .engines import PullRequestData, ReviewEngine
from .schema import _SEVERITY_RANK, Review


def fallback_summary(parsed: ParsedDiff, review: Review) -> str:
    """Plain-language summary built from the diff and findings, used whenever
    an engine returns a blank one so every review ships with a summary."""
    files = [fc.path for fc in parsed.files]
    if files:
        shown = ", ".join(files[:3]) + (f" and {len(files) - 3} more" if len(files) > 3 else "")
        n_added = sum(len(fc.added_lines) for fc in parsed.files)
        n_removed = sum(len(fc.removed_lines) for fc in parsed.files)
        change = (f"This PR touches {len(files)} file(s) ({shown}), "
                  f"adding {n_added} line(s) and removing {n_removed}.")
    else:
        change = "This PR's diff could not be parsed into per-file changes."

    if not review.issues:
        return f"{change} The review found no issues in the changed lines."
    by_sev = sorted(
        {i.severity for i in review.issues}, key=lambda s: _SEVERITY_RANK[s])
    parts = ", ".join(
        f"{sum(1 for i in review.issues if i.severity == s)} {s.value}" for s in by_sev)
    n_verified = sum(1 for i in review.issues if i.verified)
    verified = f", {n_verified} machine-verified" if n_verified else ""
    return (f"{change} The review found {len(review.issues)} issue(s) "
            f"({parts}){verified}.")


@dataclass
class PullRequest:
    diff: str
    post_files: Dict[str, str] = field(default_factory=dict)  # path -> post-change source
    title: str = ""
    description: str = ""
    tests: str = ""          # optional test code to run for verification


class Reviewer:
    def __init__(self, engine: ReviewEngine, use_context: bool = True,
                 verify: bool = False):
        self.engine = engine
        self.use_context = use_context
        self.verify = verify
        self.retriever: ContextRetriever = (
            PythonASTRetriever() if use_context else NoContextRetriever()
        )

    def review(self, pr: PullRequest) -> Review:
        try:
            parsed = parse_diff(pr.diff)
        except Exception:
            # If parsing fails, use empty parsed (LLM reads raw diff)
            parsed = parse_diff("")
        changed_by_file = {fc.path: set(fc.affected_lines) for fc in parsed.files}

        blocks = []
        for fc in parsed.files:
            src = pr.post_files.get(fc.path)
            if src:
                ctx = self.retriever.retrieve(src, fc.affected_lines)
                if ctx:
                    blocks.append(f"## {fc.path}\n{ctx}")
        context = "\n\n".join(blocks)

        data = PullRequestData(pr.diff, pr.post_files, pr.title, pr.description)
        review = self.engine.review(data, changed_by_file, context)
        review = calibrate(review, changed_by_file)

        if self.verify:
            review = self._add_verified_findings(pr, review)
        if not review.summary.strip():
            review.summary = fallback_summary(parsed, review)
        return review

    def _add_verified_findings(self, pr: PullRequest, review: Review) -> Review:
        from .verification import run_linter, run_tests
        verified = []
        for path, src in pr.post_files.items():
            verified.extend(run_linter(src, path))
        if pr.tests:
            verified.extend(run_tests(pr.post_files, pr.tests))

        # De-dup on (file, line, explanation) rather than (line, source): the
        # static engine and run_linter both run pyflakes independently and can
        # report the identical message with different `source` values, which
        # would otherwise show up as two cards for one real bug. A verified
        # finding is strictly better than an inferred one, so it replaces
        # (rather than just skips next to) any matching model-sourced finding.
        def key(i):
            return (i.file, i.line_start, i.explanation)

        kept = list(review.issues)
        seen = {key(i) for i in kept}
        for v in verified:
            k = key(v)
            if k in seen:
                kept = [i for i in kept if key(i) != k]
            kept.append(v)
            seen.add(k)

        review.issues = kept
        return Review(summary=review.summary, issues=review.sorted_issues())
