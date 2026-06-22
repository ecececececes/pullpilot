"""Precision-first calibration. Treated as a core deliverable, not an afterthought.

- drops findings below a confidence floor (noise),
- enforces line grounding: an issue must sit within (or near) the changed lines,
- converts low-confidence findings into questions for the human reviewer,
- sorts so the highest-severity, highest-confidence issues come first.
"""
from __future__ import annotations

from typing import Dict, Set

from .schema import Review


def calibrate(
    review: Review,
    changed_lines_by_file: Dict[str, Set[int]],
    min_confidence: float = 0.35,
    question_threshold: float = 0.6,
    line_tolerance: int = 3,
) -> Review:
    kept = []
    for issue in review.issues:
        if issue.verified:
            kept.append(issue)  # machine-verified facts bypass calibration
            continue

        if issue.confidence < min_confidence:
            continue  # noise floor

        changed = changed_lines_by_file.get(issue.file)
        if changed:
            grounded = any(
                issue.line_start - line_tolerance <= cl <= issue.line_end + line_tolerance
                for cl in changed
            )
            if not grounded:
                continue  # not anchored to the change -> likely invented

        if issue.confidence < question_threshold:
            issue.is_question = True

        kept.append(issue)

    review.issues = kept
    return Review(summary=review.summary, issues=review.sorted_issues())
