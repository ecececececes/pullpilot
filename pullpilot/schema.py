"""The fixed review schema. Constraining model output to this makes reviews
machine-parseable and directly comparable against ground truth."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class IssueType(str, Enum):
    BUG = "likely_bug"
    LOGIC = "logic_error"
    STYLE = "style_violation"
    TESTING = "missing_tests"
    SECURITY = "security_smell"


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "informational"


_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.MAJOR: 1,
    Severity.MINOR: 2,
    Severity.INFO: 3,
}


class Issue(BaseModel):
    file: str
    line_start: int
    line_end: int
    type: IssueType
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    suggested_fix: Optional[str] = None
    is_question: bool = False  # set by calibration when confidence is low
    source: str = "model"      # "model" | "linter" | "test" — provenance / verification

    @property
    def verified(self) -> bool:
        return self.source in ("linter", "test")

    @field_validator("line_end")
    @classmethod
    def _end_after_start(cls, v: int, info):
        start = info.data.get("line_start")
        if start is not None and v < start:
            return start
        return v


class Review(BaseModel):
    summary: str = ""
    issues: List[Issue] = Field(default_factory=list)

    def sorted_issues(self) -> List[Issue]:
        """Verified facts first, then highest severity, then highest confidence."""
        return sorted(
            self.issues,
            key=lambda i: (0 if i.verified else 1, _SEVERITY_RANK[i.severity], -i.confidence),
        )
