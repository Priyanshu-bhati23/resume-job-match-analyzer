from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RequirementMatch(BaseModel):
    requirement: str
    score: float = Field(ge=0, le=1, description="Calibrated similarity to the best resume line")
    evidence: str | None = Field(description="The resume line that matched best, if any is close enough")
    status: Literal["strong", "partial", "missing"]


class SkillReport(BaseModel):
    matched: list[str]
    missing: list[str]
    extra: list[str] = Field(description="Skills on the resume the posting does not ask for")
    coverage: float | None = Field(description="matched / required; null when the posting names no known skills")


class Suggestion(BaseModel):
    category: Literal["missing_skill", "weak_evidence", "wording", "structure"]
    priority: Literal["high", "medium", "low"]
    title: str = Field(min_length=3, max_length=120)
    detail: str = Field(min_length=10, max_length=600)
    resume_excerpt: str | None = Field(default=None, description="Verbatim line from the resume this refers to")
    rewrite: str | None = Field(default=None, description="Suggested rewording of resume_excerpt")


class SuggestionSet(BaseModel):
    summary: str = Field(min_length=10, max_length=600)
    suggestions: list[Suggestion] = Field(max_length=8)


class ScoreBreakdown(BaseModel):
    overall: int = Field(ge=0, le=100)
    semantic: int = Field(ge=0, le=100)
    skills: int | None = Field(ge=0, le=100)
    weights: dict[str, float]


class AnalysisResponse(BaseModel):
    score: ScoreBreakdown
    skills: SkillReport
    requirements: list[RequirementMatch]
    summary: str
    suggestions: list[Suggestion]
    meta: dict
