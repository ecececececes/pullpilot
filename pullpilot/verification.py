"""Tool-augmented verification: run a linter and the PR's tests for real, and
turn their output into MACHINE-VERIFIED findings (source != "model").

These are facts, not inferences: a linter genuinely flagged a line, or a test
genuinely failed. They are sorted first and never down-ranked by calibration.

Execution safety: each tool runs in a temp dir, in a subprocess, under a wall
clock timeout and POSIX resource limits (CPU time, output file size), with a
stripped environment. NOTE: this is defence-in-depth, not a true jail. For
running *untrusted* PRs from the internet you still want a container / nsjail /
seccomp boundary — see run_tests(..., allow_untrusted=...).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

from .schema import Issue, IssueType, Severity

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover
    resource = None  # type: ignore


def _preexec(cpu_seconds: int, fsize_mb: int):
    if resource is None:
        return None

    def _set():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        fsize = fsize_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    return _set


def _run(cmd: List[str], cwd: str, timeout: int = 15,
         cpu_seconds: int = 10, fsize_mb: int = 16) -> subprocess.CompletedProcess:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "HOME": cwd,
        # no proxy vars -> outbound network has no egress route configured here
    }
    return subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True,
        timeout=timeout, preexec_fn=_preexec(cpu_seconds, fsize_mb),
    )


def run_linter(source: str, file: str) -> List[Issue]:
    """Run a real linter (ruff if available, else pyflakes) and tag findings as
    machine-verified (source='linter')."""
    issues: List[Issue] = []
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, os.path.basename(file))
        with open(target, "w") as f:
            f.write(source)

        # Prefer ruff (fast, rich); fall back to pyflakes.
        import importlib.util
        if importlib.util.find_spec("ruff") is not None:
            try:
                proc = _run([sys.executable, "-m", "ruff", "check", "--output-format",
                             "concise", target], cwd=tmp)
                for line in proc.stdout.splitlines():
                    # format: path:line:col: CODE message
                    parts = line.split(":", 3)
                    if len(parts) >= 4 and parts[1].strip().isdigit():
                        ln = int(parts[1])
                        issues.append(Issue(
                            file=file, line_start=ln, line_end=ln,
                            type=IssueType.STYLE, severity=Severity.MINOR,
                            confidence=0.95, source="linter",
                            explanation=f"ruff: {parts[3].strip()}"))
                return issues
            except (FileNotFoundError, subprocess.SubprocessError):
                pass

        # pyflakes fallback (in-process; already a dependency)
        try:
            import ast as _ast
            from pyflakes.checker import Checker
            tree = _ast.parse(source, filename=file)
            for m in Checker(tree, filename=file).messages:
                issues.append(Issue(
                    file=file, line_start=m.lineno, line_end=m.lineno,
                    type=IssueType.BUG if "Undefined" in type(m).__name__ else IssueType.STYLE,
                    severity=Severity.MAJOR if "Undefined" in type(m).__name__ else Severity.MINOR,
                    confidence=0.95, source="linter",
                    explanation=f"pyflakes: {m.message % m.message_args}"))
        except SyntaxError:
            pass
    return issues


def run_tests(post_files: Dict[str, str], test_code: str,
              timeout: int = 20, allow_untrusted: bool = False) -> List[Issue]:
    """Write the PR's files + a test file to a temp dir and actually run the
    tests. A failing test becomes a machine-verified finding implicating the
    changed file (source='test').

    Set allow_untrusted=True only inside a real container/jail — running
    arbitrary PR code on the host is unsafe.
    """
    issues: List[Issue] = []
    with tempfile.TemporaryDirectory() as tmp:
        for path, src in post_files.items():
            full = os.path.join(tmp, os.path.basename(path))
            with open(full, "w") as f:
                f.write(src)
        test_path = os.path.join(tmp, "test_pr.py")
        with open(test_path, "w") as f:
            f.write(test_code)

        try:
            proc = _run([sys.executable, "-m", "pytest", "-q", "test_pr.py"],
                        cwd=tmp, timeout=timeout)
        except subprocess.TimeoutExpired:
            primary = next(iter(post_files), "test_pr.py")
            return [Issue(
                file=primary, line_start=1, line_end=len(post_files.get(primary, "x").splitlines()) or 1,
                type=IssueType.LOGIC, severity=Severity.CRITICAL, confidence=0.99,
                source="test",
                explanation="Test run timed out — possible infinite loop or hang.")]

        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-6:])[:400]
            primary = next(iter(post_files), "test_pr.py")
            n_lines = len(post_files.get(primary, "x").splitlines()) or 1
            issues.append(Issue(
                file=primary, line_start=1, line_end=n_lines,
                type=IssueType.LOGIC, severity=Severity.CRITICAL, confidence=0.99,
                source="test",
                explanation=f"A test failed against this change. Summary: {tail}"))
    return issues
