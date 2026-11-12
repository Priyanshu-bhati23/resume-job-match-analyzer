"""FastAPI application."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__, embeddings, llm, scoring, suggestions
from .parsing import MAX_UPLOAD_BYTES, ParseError, extract_text, split_units
from .schemas import AnalysisResponse

MAX_RESUME_CHARS, MAX_JOB_CHARS, MIN_CHARS = 60_000, 30_000, 80

app = FastAPI(title="Resume & Job-Match Analyzer", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_embedder() -> embeddings.Embedder:
    return embeddings.from_env()


def get_llm() -> llm.LLM | None:
    return llm.from_env()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(
    job_description: str = Form(...),
    resume_text: str | None = Form(None),
    resume_file: UploadFile | None = File(None),
    embedder: embeddings.Embedder = Depends(get_embedder),
    model: llm.LLM | None = Depends(get_llm),
) -> AnalysisResponse:
    if resume_file is not None and resume_file.filename:
        data = resume_file.file.read(MAX_UPLOAD_BYTES + 1)
        try:
            resume = extract_text(resume_file.filename, data)
        except ParseError as exc:
            raise HTTPException(422, str(exc)) from exc
    elif resume_text and resume_text.strip():
        resume = resume_text
    else:
        raise HTTPException(422, "provide a resume file or resume text")

    resume, job = resume.strip()[:MAX_RESUME_CHARS], job_description.strip()[:MAX_JOB_CHARS]
    if len(resume) < MIN_CHARS:
        raise HTTPException(422, "the resume is too short to analyse")
    if len(job) < MIN_CHARS:
        raise HTTPException(422, "the job description is too short to analyse")

    requirements = scoring.match_requirements(resume, job, embedder)
    skills = scoring.skill_report(resume, job)
    advice, source = suggestions.generate(model, resume, job, split_units(resume), requirements, skills)
    return AnalysisResponse(
        score=scoring.score(requirements, skills),
        skills=skills,
        requirements=requirements,
        summary=advice.summary,
        suggestions=advice.suggestions,
        meta={"embedder": embedder.name, "suggestions_by": source, "resume_chars": len(resume)},
    )


# In the Docker image the built React app sits next to the backend and is served from "/".
_DIST = Path(os.getenv("FRONTEND_DIST", Path(__file__).parent.parent.parent / "frontend" / "dist"))
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
