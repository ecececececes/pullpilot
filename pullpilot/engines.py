"""Pluggable review engines behind one interface.

  * LLMEngine            - prompt -> structured JSON (needs a provider/API key)
  * StaticAnalysisEngine - real AST + pyflakes detection (works offline)

The Reviewer orchestrates parsing + context retrieval, then delegates to an
engine, so the benchmark harness is identical across both.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Dict, Set

from . import prompts
from .providers import Provider
from .schema import Review
from .static_analysis import analyze


class PullRequestData:
    """Lightweight container passed to engines (avoids a circular import)."""

    def __init__(self, diff: str, post_files: Dict[str, str], title: str, description: str):
        self.diff = diff
        self.post_files = post_files
        self.title = title
        self.description = description


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


class ReviewEngine(ABC):
    name = "base"

    @abstractmethod
    def review(self, pr: PullRequestData,
               changed_by_file: Dict[str, Set[int]], context: str) -> Review:
        ...


class LLMEngine(ReviewEngine):
    name = "llm"

    def __init__(self, provider: Provider):
        self.provider = provider
        self.name = f"llm:{provider.name}"

    def review(self, pr, changed_by_file, context):
        system = prompts.system_prompt()
        user = prompts.user_prompt(pr.title, pr.description, pr.diff, context)
        raw = self.provider.complete(system, user)
        try:
            return Review.model_validate(extract_json(raw))
        except Exception as exc:
            return Review(summary=f"(parse error: {exc})", issues=[])


class StaticAnalysisEngine(ReviewEngine):
    name = "static"

    def review(self, pr, changed_by_file, context):
        issues = []
        for path, src in pr.post_files.items():
            issues.extend(analyze(src, changed_by_file.get(path, set()), path))
        summary = (
            f"Static analysis of {', '.join(pr.post_files) or 'the diff'}: "
            f"{len(issues)} issue(s) found in changed lines."
        )
        return Review(summary=summary, issues=issues)
