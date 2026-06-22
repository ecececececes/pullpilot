"""Retrieve grounding context for changed lines.

Two implementations let you run the central ablation:
  * PythonASTRetriever - enclosing function/class bodies + referenced symbol defs
  * NoContextRetriever  - the bare-diff baseline (returns nothing)

Deterministic, no vector DB. For other languages swap in a tree-sitter based
retriever behind the same interface.
"""
from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Set


class ContextRetriever(ABC):
    @abstractmethod
    def retrieve(self, source: str, changed_lines: Iterable[int]) -> str:
        ...


class NoContextRetriever(ContextRetriever):
    """Ablation baseline: the model sees only the diff."""

    def retrieve(self, source: str, changed_lines: Iterable[int]) -> str:
        return ""


class PythonASTRetriever(ContextRetriever):
    def retrieve(self, source: str, changed_lines: Iterable[int]) -> str:
        changed: Set[int] = set(changed_lines)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ""  # unparseable post-change file -> fall back to no context
        lines = source.splitlines()
        blocks = []
        seen: Set[tuple] = set()

        # 1. Enclosing function / class definitions that the change falls inside.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", start) or start
                if any(start <= ln <= end for ln in changed):
                    key = (start, end)
                    if key not in seen:
                        seen.add(key)
                        snippet = "\n".join(lines[start - 1:end])
                        blocks.append(
                            f"# enclosing {type(node).__name__} "
                            f"'{node.name}' (lines {start}-{end})\n{snippet}"
                        )

        # 2. Top-level definitions of symbols referenced on the changed lines.
        referenced = self._names_on_lines(tree, changed)
        defs = self._toplevel_defs(tree, lines)
        for name in sorted(referenced):
            block = defs.get(name)
            if block and block not in blocks:
                blocks.append(block)

        return "\n\n".join(blocks)

    @staticmethod
    def _names_on_lines(tree: ast.AST, changed: Set[int]) -> Set[str]:
        names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and getattr(node, "lineno", None) in changed:
                names.add(node.id)
        return names

    @staticmethod
    def _toplevel_defs(tree: ast.Module, lines) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", start) or start
                out[node.name] = (
                    f"# definition of '{node.name}' (lines {start}-{end})\n"
                    + "\n".join(lines[start - 1:end])
                )
        return out
