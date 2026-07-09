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
    """Pull the review object out of model output. Models wrap JSON in fences,
    echo the schema/template before their answer, or append prose — so scan
    every balanced object and prefer the one shaped like a review."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]

    decoder = json.JSONDecoder()
    candidates = []
    i = text.find("{")
    while i != -1:
        try:
            obj, consumed = decoder.raw_decode(text[i:])
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
        i = text.find("{", i + consumed)
    if not candidates:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")

    def is_review(o: dict) -> bool:
        # an echoed schema/template has $defs/properties; a real answer has
        # an issues list (possibly empty) and no schema keys
        return (isinstance(o.get("issues"), list)
                and "$defs" not in o and "properties" not in o)

    for obj in candidates:
        if is_review(obj):
            return obj
    raise ValueError(
        f"model output contained JSON but no review object: {text[:200]!r}")


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
        # summary left blank: the Reviewer fills in the shared diff-aware one
        return Review(issues=issues)
