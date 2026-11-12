import pytest

from app.parsing import ParseError, extract_text, split_units
from app.skills import extract_skills

from .helpers import RESUME, make_pdf


def test_pdf_text_is_extracted():
    pdf = make_pdf(["Built a RAG pipeline (FAISS) for 2,000 users", "Developed REST APIs in Python"])
    text = extract_text("resume.pdf", pdf)
    assert "RAG pipeline (FAISS)" in text and "REST APIs in Python" in text


def test_pdf_is_detected_by_content_not_just_extension():
    assert "RAG pipeline" in extract_text("resume.bin", make_pdf(["Built a RAG pipeline with FAISS and FastAPI here"]))


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("cv.pdf", b"%PDF-1.4 this is not really a pdf", "could not read"),
        ("cv.pdf", make_pdf(["hi"]), "no text found"),
        ("cv.docx", b"PK\x03\x04", "unsupported file type"),
        ("cv.txt", b"\xff\xfe\x00bad", "not UTF-8"),
        ("cv.txt", b"x" * (5 * 1024 * 1024 + 1), "larger than 5 MB"),
    ],
)
def test_bad_uploads_get_a_clear_error(name, data, message):
    with pytest.raises(ParseError, match=message):
        extract_text(name, data)


def test_units_strip_bullets_and_skip_headings():
    units = split_units(RESUME)
    assert "EXPERIENCE" not in units and "SKILLS" not in units
    assert "Developed REST APIs in Python and deployed them with Docker and GitHub Actions" in units
    assert not any(u.startswith(("-", "•")) for u in units)


def test_wrapped_pdf_lines_are_rejoined():
    wrapped = (
        "• Built and maintained full-stack web applications: user interfaces, backend APIs, database integration and\n"
        "performance fixes.\n"
        "\x7f Worked on CI/CD pipelines and cloud deployment\n"
    )
    units = split_units(wrapped)
    assert units[0].endswith("database integration and performance fixes")
    assert units[1] == "Worked on CI/CD pipelines and cloud deployment"  # control-char bullet removed


def test_units_are_deduplicated_and_capped():
    assert split_units("Same line repeated here\nSame line repeated here\n") == ["Same line repeated here"]
    assert len(split_units("\n".join(f"Unique line number {i} here" for i in range(300)), max_units=50)) == 50


def test_skills_use_aliases_and_canonical_names():
    text = "Shipped k8s services in golang; built retrieval-augmented generation with a vector database and postgres."
    assert extract_skills(text) == ["Go", "RAG", "Vector search", "PostgreSQL", "Kubernetes"]


def test_skill_matching_avoids_common_false_positives():
    prose = "We go to market fast, rest on weekends, express ideas clearly and spark joy. Swift decisions matter."
    assert extract_skills(prose) == ["Swift"]  # capitalised "Swift" is ambiguous; the others must not match
    assert "Java" not in extract_skills("Expert in JavaScript")
    assert "C++" in extract_skills("Wrote C++ and C# services") and "C#" in extract_skills("Wrote C++ and C# services")


def test_skills_survive_pdf_line_breaks():
    assert "Vector search" in extract_skills("embeddings, vector\nsearch (FAISS)")
