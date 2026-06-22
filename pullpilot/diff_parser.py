"""Parse a unified diff into structured, addressable units.

Downstream analysis reasons over discrete changed regions rather than an
undifferentiated blob of text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from unidiff import PatchSet


@dataclass
class FileChange:
    path: str                          # target (post-change) path
    added_lines: List[int] = field(default_factory=list)    # new-file line numbers
    removed_lines: List[int] = field(default_factory=list)   # old-file line numbers
    affected_lines: List[int] = field(default_factory=list)  # post-file band touched by hunks
    raw: str = ""


@dataclass
class ParsedDiff:
    files: List[FileChange] = field(default_factory=list)


def _strip_prefix(path: str) -> str:
    for pre in ("a/", "b/"):
        if path.startswith(pre):
            return path[len(pre):]
    return path


def parse_diff(diff_text: str) -> ParsedDiff:
    patch = PatchSet(diff_text)
    files: List[FileChange] = []
    for pf in patch:
        path = _strip_prefix(pf.target_file or pf.source_file or "")
        added, removed, affected = [], [], []
        for hunk in pf:
            for line in hunk:
                if line.is_added and line.target_line_no:
                    added.append(line.target_line_no)
                elif line.is_removed and line.source_line_no:
                    removed.append(line.source_line_no)
                if line.target_line_no:  # added + context: a band in the new file
                    affected.append(line.target_line_no)
        files.append(
            FileChange(
                path=path,
                added_lines=added,
                removed_lines=removed,
                affected_lines=sorted(set(affected)),
                raw=str(pf),
            )
        )
    return ParsedDiff(files=files)
