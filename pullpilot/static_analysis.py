"""A real, working static analyser. Detects a meaningful subset of defects via
AST patterns plus pyflakes, grounded to the changed lines of the diff.

This is genuinely "the system working" with no API key, and doubles as:
  * the tool-augmented / linter baseline from the proposal, and
  * a strong comparison point the LLM has to beat.

It deliberately catches pattern-based bugs (mutable defaults, == None, bare
except, identity-compare-to-literal, unguarded .get() subscript, resource
leaks, pyflakes findings) and deliberately MISSES purely semantic bugs
(off-by-one, wrong operator, swapped args) — that gap is the motivation for an
LLM reviewer.
"""
from __future__ import annotations

import ast
from typing import List, Set

from .schema import Issue, IssueType, Severity


def _issue(file, line, type_, sev, conf, expl, fix=None) -> Issue:
    return Issue(
        file=file, line_start=line, line_end=line, type=type_,
        severity=sev, confidence=conf, explanation=expl, suggested_fix=fix,
    )


class _AstChecks(ast.NodeVisitor):
    def __init__(self, file: str):
        self.file = file
        self.issues: List[Issue] = []
        self._safe_open_lines: Set[int] = set()

    # --- mutable default arguments ---
    def visit_FunctionDef(self, node: ast.FunctionDef):
        defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d]
        for default in defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.issues.append(_issue(
                    self.file, default.lineno, IssueType.BUG, Severity.MAJOR, 0.9,
                    "Mutable default argument: the same object is shared across "
                    "calls and persists between them.",
                    "Default to None and build the container inside the function.",
                ))
        self._check_none_then_subscript(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore

    # --- comparisons ---
    def visit_Compare(self, node: ast.Compare):
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.NotEq)) and \
                    isinstance(comp, ast.Constant) and comp.value is None:
                self.issues.append(_issue(
                    self.file, node.lineno, IssueType.STYLE, Severity.MINOR, 0.85,
                    "Comparison to None should use 'is' / 'is not'.",
                    "Use 'is None' or 'is not None'.",
                ))
            if isinstance(op, (ast.Is, ast.IsNot)) and \
                    isinstance(comp, ast.Constant) and \
                    isinstance(comp.value, (int, float, str, bytes)):
                self.issues.append(_issue(
                    self.file, node.lineno, IssueType.BUG, Severity.MAJOR, 0.8,
                    "Using 'is' to compare with a literal relies on object "
                    "identity and is unreliable.",
                    "Use '==' for value comparison.",
                ))
        self.generic_visit(node)

    # --- bare / overbroad except ---
    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.type is None:
            self.issues.append(_issue(
                self.file, node.lineno, IssueType.BUG, Severity.MAJOR, 0.7,
                "Bare 'except:' swallows everything, including "
                "KeyboardInterrupt and SystemExit.",
                "Catch a specific exception type.",
            ))
        self.generic_visit(node)

    # --- resource leak: open() not in a with-statement ---
    def visit_With(self, node: ast.With):
        for item in node.items:
            expr = item.context_expr
            if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) \
                    and expr.func.id == "open":
                self._safe_open_lines.add(expr.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "open" \
                and node.lineno not in self._safe_open_lines:
            self.issues.append(_issue(
                self.file, node.lineno, IssueType.BUG, Severity.MINOR, 0.6,
                "File opened without a 'with' block; it may not be closed if an "
                "error occurs.",
                "Use 'with open(...) as f:'.",
            ))
        self.generic_visit(node)

    # --- unguarded subscript of a .get() result (possible None) ---
    def _check_none_then_subscript(self, fn: ast.AST):
        got_from_get: dict[str, int] = {}
        guarded: Set[str] = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) \
                    and isinstance(n.value.func, ast.Attribute) \
                    and n.value.func.attr == "get" \
                    and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                got_from_get[n.targets[0].id] = n.lineno
            if isinstance(n, ast.If):
                for sub in ast.walk(n.test):
                    if isinstance(sub, ast.Name):
                        guarded.add(sub.id)
        for n in ast.walk(fn):
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) \
                    and n.value.id in got_from_get and n.value.id not in guarded:
                self.issues.append(_issue(
                    self.file, n.lineno, IssueType.BUG, Severity.MAJOR, 0.65,
                    "Value from .get() may be None and is subscripted without a "
                    "None check.",
                    "Check for None before indexing the result.",
                ))


def _pyflakes_issues(source: str, file: str) -> List[Issue]:
    try:
        from pyflakes.checker import Checker
    except Exception:
        return []
    try:
        tree = ast.parse(source, filename=file)
    except SyntaxError:
        return []
    checker = Checker(tree, filename=file)
    out: List[Issue] = []
    for m in checker.messages:
        text = m.message % m.message_args
        cls = type(m).__name__
        type_ = IssueType.BUG if "Undefined" in cls else IssueType.STYLE
        sev = Severity.MAJOR if "Undefined" in cls else Severity.MINOR
        out.append(_issue(file, m.lineno, type_, sev, 0.75, f"pyflakes: {text}"))
    return out


def analyze(source: str, changed_lines: Set[int], file: str,
            tolerance: int = 2) -> List[Issue]:
    """Return issues whose line falls within (or near) the changed lines."""
    try:
        tree = ast.parse(source, filename=file)
    except SyntaxError:
        return []
    checks = _AstChecks(file)
    checks.visit(tree)
    all_issues = checks.issues + _pyflakes_issues(source, file)

    if not changed_lines:
        return all_issues
    grounded = [
        i for i in all_issues
        if any(abs(i.line_start - cl) <= tolerance for cl in changed_lines)
    ]
    # de-duplicate (line, explanation)
    seen, unique = set(), []
    for i in grounded:
        key = (i.line_start, i.explanation)
        if key not in seen:
            seen.add(key)
            unique.append(i)
    return unique
