import pytest
from fastapi.testclient import TestClient

from app.main import app, get_embedder, get_llm

from .helpers import JOB, RESUME, FakeLLM, make_pdf


@pytest.fixture
def client():
    app.dependency_overrides[get_llm] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_analyze_pasted_text(client):
    response = client.post("/api/analyze", data={"job_description": JOB, "resume_text": RESUME})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["score"]["overall"] <= 100 and body["score"]["weights"] == {"semantic": 0.6, "skills": 0.4}
    assert "Kubernetes" in body["skills"]["missing"] and "FastAPI" in body["skills"]["matched"]
    assert {r["status"] for r in body["requirements"]} <= {"strong", "partial", "missing"}
    assert body["suggestions"] and body["meta"] == {
        "embedder": "hashing-4096",
        "suggestions_by": "rules",
        "resume_chars": len(RESUME.strip()),
    }


def test_analyze_uploaded_pdf(client):
    pdf = make_pdf([line for line in RESUME.splitlines() if line.strip()])
    response = client.post(
        "/api/analyze",
        data={"job_description": JOB},
        files={"resume_file": ("resume.pdf", pdf, "application/pdf")},
    )
    assert response.status_code == 200
    assert "FAISS" in response.json()["skills"]["matched"]


def test_llm_suggestions_are_used_when_a_model_is_configured(client):
    reply = (
        '{"summary": "Good backend fit with a cloud gap.", "suggestions": [{"category": "missing_skill", '
        '"priority": "high", "title": "Mention Kubernetes if you have used it", '
        '"detail": "The posting asks for Kubernetes on AWS; add it only if it is true.", '
        '"resume_excerpt": null, "rewrite": null}]}'
    )
    app.dependency_overrides[get_llm] = lambda: FakeLLM(reply)
    body = client.post("/api/analyze", data={"job_description": JOB, "resume_text": RESUME}).json()
    assert body["meta"]["suggestions_by"] == "fake-model"
    assert body["summary"] == "Good backend fit with a cloud gap."


@pytest.mark.parametrize(
    ("data", "files", "detail"),
    [
        ({"job_description": JOB}, None, "provide a resume"),
        ({"job_description": JOB, "resume_text": "too short"}, None, "resume is too short"),
        ({"job_description": "short", "resume_text": RESUME}, None, "job description is too short"),
        (
            {"job_description": JOB},
            {"resume_file": ("cv.docx", b"PK\x03\x04", "application/zip")},
            "unsupported file type",
        ),
    ],
)
def test_invalid_requests_are_rejected_with_a_reason(client, data, files, detail):
    response = client.post("/api/analyze", data=data, files=files)
    assert response.status_code == 422 and detail in response.json()["detail"]


def test_missing_job_description_is_a_validation_error(client):
    assert client.post("/api/analyze", data={"resume_text": RESUME}).status_code == 422


def test_embedder_dependency_can_be_swapped(client):
    class Constant:
        name, calibration = "constant", (0.0, 1.0)

        def embed(self, texts):
            import numpy as np

            return np.ones((len(texts), 4), dtype="float32") / 2.0

    app.dependency_overrides[get_embedder] = lambda: Constant()
    body = client.post("/api/analyze", data={"job_description": JOB, "resume_text": RESUME}).json()
    assert body["meta"]["embedder"] == "constant" and body["score"]["semantic"] == 100
