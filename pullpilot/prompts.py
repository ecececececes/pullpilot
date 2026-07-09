"""Prompt templates. This is the file you will iterate on most.

The calibration step enforces precision mechanically, but the prompt is your
first line of defence against noise.
"""
from __future__ import annotations

import json

SYSTEM_PROMPT = """You are PullPilot, a precise code-review assistant.
You review a single pull-request diff and report only issues you can ground in
specific changed lines.

Principles (PRECISION OVER RECALL):
- Only report an issue supported by the lines shown. Do NOT speculate about code
  you cannot see.
- Prefer a few high-confidence issues over many weak ones.
- If you are not confident, lower the confidence and phrase the explanation as a
  question; the system will mark it for the human reviewer.
- Every issue MUST reference a real line range within the changed lines.
- Use the retrieved context to avoid inventing issues and to catch problems that
  depend on surrounding code.
- ALWAYS fill "summary" with 1-3 plain-language sentences: what the PR changes
  and your overall assessment. Never leave it empty, even when there are no
  issues.

Issue types: likely_bug, logic_error, style_violation, missing_tests, security_smell
Severities: critical, major, minor, informational

Respond with ONLY a JSON object in exactly this shape — it is a template, fill
in your own values, do NOT echo it back (no markdown, no prose, no schema):
{schema}

"type" must be one of the issue types above; "severity" one of the severities;
"confidence" a number between 0 and 1; "suggested_fix" replacement code or null.
"""

USER_TEMPLATE = """PR title: {title}
PR description: {description}

=== DIFF ===
{diff}

=== RETRIEVED CONTEXT ===
{context}

Review the diff. Respond with the JSON object only."""


# A compact fill-in template instead of Review.model_json_schema(): small
# models parrot a raw JSON-Schema dump back (breaking parsing), and the full
# schema wastes tokens without improving adherence.
_RESPONSE_TEMPLATE = {
    "summary": "1-3 plain-language sentences: what the PR changes and your overall assessment",
    "issues": [{
        "file": "path/to/file.py",
        "line_start": 3,
        "line_end": 4,
        "type": "likely_bug",
        "severity": "major",
        "confidence": 0.9,
        "explanation": "what is wrong and why",
        "suggested_fix": "replacement code, or null",
    }],
}


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(schema=json.dumps(_RESPONSE_TEMPLATE, indent=1))


def user_prompt(title: str, description: str, diff: str, context: str) -> str:
    return USER_TEMPLATE.format(
        title=title or "(none)",
        description=description or "(none)",
        diff=diff,
        context=context or "(no context retrieved)",
    )
