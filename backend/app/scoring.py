"""Resume-to-job fit scoring.

Two independent signals, combined with fixed, visible weights:

  semantic  For every requirement line in the posting, find the most similar
            line in the resume (cosine similarity of embeddings). The score is
            the mean of those best matches - i.e. requirement coverage, not
            whole-document similarity, so a long resume is not rewarded for bulk.
  skills    Share of the posting's named skills that also appear on the resume.

The numbers are a screening heuristic, not a hiring prediction.
"""

from __future__ import annotations

import numpy as np

from .embeddings import Embedder
from .parsing import split_units
from .schemas import RequirementMatch, ScoreBreakdown, SkillReport
from .skills import extract_skills

WEIGHTS = {"semantic": 0.6, "skills": 0.4}
STRONG, PARTIAL = 0.6, 0.3


def calibrate(cosine: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((cosine - low) / (high - low), 0.0, 1.0)


def match_requirements(resume: str, job: str, embedder: Embedder) -> list[RequirementMatch]:
    # Four words minimum on the posting side drops headings like "About the role".
    requirements, lines = split_units(job, min_words=4, max_units=40), split_units(resume)
    if not requirements or not lines:
        return []
    vectors = embedder.embed(requirements + lines)
    similarity = vectors[: len(requirements)] @ vectors[len(requirements) :].T
    best = similarity.argmax(axis=1)
    scores = calibrate(similarity.max(axis=1), *embedder.calibration)
    matches = []
    for requirement, index, score in zip(requirements, best, scores, strict=True):
        status = "strong" if score >= STRONG else "partial" if score >= PARTIAL else "missing"
        matches.append(
            RequirementMatch(
                requirement=requirement,
                score=round(float(score), 3),
                evidence=lines[int(index)] if status != "missing" else None,
                status=status,
            )
        )
    return matches


def skill_report(resume: str, job: str) -> SkillReport:
    required, have = extract_skills(job), set(extract_skills(resume))
    matched = [s for s in required if s in have]
    return SkillReport(
        matched=matched,
        missing=[s for s in required if s not in have],
        extra=[s for s in extract_skills(resume) if s not in required],
        coverage=round(len(matched) / len(required), 3) if required else None,
    )


def score(requirements: list[RequirementMatch], skills: SkillReport) -> ScoreBreakdown:
    semantic = float(np.mean([r.score for r in requirements])) if requirements else 0.0
    if skills.coverage is None:
        # The posting names no skills we recognise: do not invent a skills score.
        weights, overall = {"semantic": 1.0, "skills": 0.0}, semantic
    else:
        weights = WEIGHTS
        overall = weights["semantic"] * semantic + weights["skills"] * skills.coverage
    return ScoreBreakdown(
        overall=round(overall * 100),
        semantic=round(semantic * 100),
        skills=None if skills.coverage is None else round(skills.coverage * 100),
        weights=weights,
    )
