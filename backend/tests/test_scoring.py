import numpy as np
import pytest

from app import scoring
from app.embeddings import HashingEmbedder, OpenAIEmbedder
from app.schemas import RequirementMatch, SkillReport

from .helpers import JOB, RESUME

EMBEDDER = HashingEmbedder()


def test_requirements_are_matched_to_their_best_evidence():
    matches = {m.requirement: m for m in scoring.match_requirements(RESUME, JOB, EMBEDDER)}

    apis = matches["Develop REST APIs in Python with FastAPI"]
    assert apis.status == "strong" and apis.evidence.startswith("Developed REST APIs in Python")

    rag = matches["Build retrieval augmented generation pipelines with FAISS or Pinecone"]
    assert rag.status in ("strong", "partial") and "retrieval augmented generation pipeline" in rag.evidence

    tuning = matches["Experience fine-tuning models with PyTorch"]
    assert tuning.status == "missing" and tuning.evidence is None and tuning.score < 0.3


def test_headings_in_the_posting_are_not_scored_as_requirements():
    requirements = [m.requirement for m in scoring.match_requirements(RESUME, JOB, EMBEDDER)]
    assert "AI Engineer" not in requirements and "About the role" not in requirements


def test_skill_report_splits_matched_missing_and_extra():
    report = scoring.skill_report(RESUME, JOB)
    assert report.matched == ["Python", "SQL", "LLM", "RAG", "FAISS", "FastAPI", "REST APIs", "Docker"]
    assert report.missing == ["Pinecone", "Fine-tuning", "PyTorch", "Kubernetes", "AWS"]
    assert "PostgreSQL" in report.extra and report.coverage == round(8 / 13, 3)


def test_overall_score_is_the_documented_weighted_sum():
    requirements = [
        RequirementMatch(requirement="a b c d", score=1.0, evidence="x", status="strong"),
        RequirementMatch(requirement="e f g h", score=0.5, evidence="y", status="partial"),
    ]
    skills = SkillReport(matched=["Python"], missing=["AWS", "Rust", "Go"], extra=[], coverage=0.25)
    result = scoring.score(requirements, skills)
    assert (result.semantic, result.skills) == (75, 25)
    assert result.overall == round((0.6 * 0.75 + 0.4 * 0.25) * 100) == 55


def test_posting_without_known_skills_is_scored_on_semantics_alone():
    requirements = [RequirementMatch(requirement="a b c d", score=0.8, evidence="x", status="strong")]
    result = scoring.score(requirements, SkillReport(matched=[], missing=[], extra=["Python"], coverage=None))
    assert result.overall == 80 and result.skills is None and result.weights == {"semantic": 1.0, "skills": 0.0}


def test_a_matching_resume_outscores_an_unrelated_one():
    chef = (
        "Head chef with ten years of experience running a busy restaurant kitchen.\n"
        "- Designed seasonal menus and managed a team of twelve cooks\n"
        "- Reduced food waste by a third through better inventory planning\n"
    )

    def overall(resume):
        return scoring.score(
            scoring.match_requirements(resume, JOB, EMBEDDER), scoring.skill_report(resume, JOB)
        ).overall

    assert overall(RESUME) > 45 and overall(chef) < 10


def test_empty_inputs_score_zero_instead_of_crashing():
    assert scoring.match_requirements("", JOB, EMBEDDER) == []
    assert scoring.score([], SkillReport(matched=[], missing=[], extra=[], coverage=None)).overall == 0


def test_calibration_clips_to_the_unit_interval():
    out = scoring.calibrate(np.array([-0.2, 0.05, 0.25, 0.45, 0.9]), 0.05, 0.45)
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])


def test_openai_embedder_batches_and_restores_order():
    import httpx

    def handler(request):
        import json

        inputs = json.loads(request.content)["input"]
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(inputs)]
        return httpx.Response(200, json={"data": list(reversed(data))})  # API order is not guaranteed

    embedder = OpenAIEmbedder("embed-test", "key", transport=httpx.MockTransport(handler))
    vectors = embedder.embed(["a", "bbbb"])
    assert vectors.shape == (2, 2) and np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert vectors[0, 0] < vectors[1, 0]  # row 0 is still "a"
