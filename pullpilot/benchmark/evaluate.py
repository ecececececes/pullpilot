"""Defect-detection metrics.

An issue "hits" a planted defect if a ground-truth line falls within the issue's
reported line range, expanded by a small tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..schema import Issue, Review
from .build_dataset import BenchmarkPR


def _issue_hits(issue: Issue, gt_lines: List[int], tol: int) -> bool:
    return any(issue.line_start - tol <= gt <= issue.line_end + tol for gt in gt_lines)


@dataclass
class Metrics:
    n_buggy: int
    n_clean: int
    detected: int                     # buggy PRs with >=1 issue hitting the defect
    recall: float                     # detected / n_buggy
    precision: float                  # hitting issues / all issues on buggy PRs
    exact_localization: float         # exact-line hits / hitting issues
    false_alarms_per_clean_pr: float


def per_pr_results(prs, reviews, tol: int = 2):
    """Per-PR detail for reporting: detection + issue count for each PR."""
    rows = []
    for p in prs:
        review = reviews[p.id]
        if p.label == "buggy":
            detected = any(_issue_hits(i, p.ground_truth_lines, tol) for i in review.issues)
        else:
            detected = None  # n/a for clean PRs
        rows.append({
            "id": p.id,
            "category": p.category,
            "label": p.label,
            "title": p.title,
            "detected": detected,
            "n_issues": len(review.issues),
            "ground_truth": p.ground_truth_lines,
            "issues": review.issues,
        })
    return rows


def evaluate(
    prs: List[BenchmarkPR],
    reviews: Dict[str, Review],
    tol: int = 2,
    exact_tol: int = 0,
) -> Metrics:
    n_buggy = sum(1 for p in prs if p.label == "buggy")
    n_clean = sum(1 for p in prs if p.label == "clean")

    detected = 0
    total_issues_on_buggy = 0
    hitting_issues = 0
    exact_hits = 0
    clean_alarms = 0

    for p in prs:
        review = reviews[p.id]
        if p.label == "buggy":
            total_issues_on_buggy += len(review.issues)
            any_hit = False
            for issue in review.issues:
                if _issue_hits(issue, p.ground_truth_lines, tol):
                    hitting_issues += 1
                    any_hit = True
                    if _issue_hits(issue, p.ground_truth_lines, exact_tol):
                        exact_hits += 1
            if any_hit:
                detected += 1
        else:
            clean_alarms += len(review.issues)

    return Metrics(
        n_buggy=n_buggy,
        n_clean=n_clean,
        detected=detected,
        recall=detected / n_buggy if n_buggy else 0.0,
        precision=hitting_issues / total_issues_on_buggy if total_issues_on_buggy else 0.0,
        exact_localization=exact_hits / hitting_issues if hitting_issues else 0.0,
        false_alarms_per_clean_pr=clean_alarms / n_clean if n_clean else 0.0,
    )
