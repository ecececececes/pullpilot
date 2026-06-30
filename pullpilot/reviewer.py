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
from .diff_parser import parse_diff
from .engines import PullRequestData, ReviewEngine
from .schema import Review


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
        return review

    def _add_verified_findings(self, pr: PullRequest, review: Review) -> Review:
        from .verification import run_linter, run_tests
        verified = []
        for path, src in pr.post_files.items():
            verified.extend(run_linter(src, path))
        if pr.tests:
            verified.extend(run_tests(pr.post_files, pr.tests))
        # de-dup verified vs model findings on (line, source)
        seen = {(i.line_start, i.source) for i in review.issues}
        for v in verified:
            if (v.line_start, v.source) not in seen:
                review.issues.append(v)
                seen.add((v.line_start, v.source))
        return Review(summary=review.summary, issues=review.sorted_issues())
