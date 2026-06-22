"""Build a defect-detection benchmark by REVERSING real fix commits.

Trick: take the FIXED version of a file as the pre-existing clean code, and
construct a PR whose diff re-introduces the bug (fixed -> buggy). The lines that
diff touches are the ground-truth defect location. Real defects, exact labels,
zero manual annotation.

Populate pairs from BugsInPy / Defects4J checkouts, or any fix commit:
`git show <fix_sha>` is the fix; its parent is the buggy version.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass
from typing import List


@dataclass
class BenchmarkPR:
    id: str
    file: str
    diff: str                    # fixed -> buggy (the PR under review)
    post_source: str             # buggy version (post-change), for context retrieval
    ground_truth_lines: List[int]
    label: str                   # "buggy" or "clean"
    category: str = ""           # bug type, e.g. "off_by_one"
    title: str = ""
    description: str = ""


def _unified(before: str, after: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _ground_truth_lines(before: str, after: str) -> List[int]:
    """1-indexed line numbers in `after` where the change manifests.

    For replace/insert: the changed lines themselves. For a pure deletion (a bug
    introduced by removing code, e.g. a dropped None-check) there is no added
    line, so we anchor to the surviving line that now sits at the deletion point.
    """
    after_len = len(after.splitlines())
    sm = difflib.SequenceMatcher(a=before.splitlines(), b=after.splitlines())
    gt: List[int] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            gt.extend(range(j1 + 1, j2 + 1))
        elif tag == "delete":
            anchor = min(j1 + 1, after_len) if after_len else 1
            gt.append(anchor)
    return sorted(set(gt))


def make_buggy_pr(
    pr_id: str, file: str, buggy_source: str, fixed_source: str,
    category: str = "", title: str = "", description: str = "",
) -> BenchmarkPR:
    diff = _unified(fixed_source, buggy_source, file)
    gt = _ground_truth_lines(fixed_source, buggy_source)
    return BenchmarkPR(
        id=pr_id, file=file, diff=diff, post_source=buggy_source,
        ground_truth_lines=gt, label="buggy", category=category,
        title=title, description=description,
    )


def make_clean_pr(
    pr_id: str, file: str, before: str, after: str,
    title: str = "", description: str = "",
) -> BenchmarkPR:
    diff = _unified(before, after, file)
    return BenchmarkPR(
        id=pr_id, file=file, diff=diff, post_source=after,
        ground_truth_lines=[], label="clean", title=title, description=description,
    )


def save_dataset(prs: List[BenchmarkPR], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(p) for p in prs], f, indent=2)


def load_dataset(path: str) -> List[BenchmarkPR]:
    with open(path) as f:
        return [BenchmarkPR(**d) for d in json.load(f)]
