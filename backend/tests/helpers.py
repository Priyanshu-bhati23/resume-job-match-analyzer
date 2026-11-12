"""Fixtures shared by the tests."""

from __future__ import annotations

RESUME = """Asha Rao
asha@example.com | Bengaluru

SUMMARY
Backend engineer who builds LLM-powered products: RAG pipelines and agents shipped to real users.

EXPERIENCE
Acme Labs | Software Engineer
- Built a retrieval augmented generation pipeline with FAISS and FastAPI serving 2,000 daily users
- Developed REST APIs in Python and deployed them with Docker and GitHub Actions
- Maintained PostgreSQL schemas and wrote the pytest suite

SKILLS
Python, SQL, FastAPI, LangGraph, FAISS, Docker, PostgreSQL
"""

JOB = """AI Engineer

About the role
We are hiring an engineer to build LLM applications.

Responsibilities
- Build retrieval augmented generation pipelines with FAISS or Pinecone
- Develop REST APIs in Python with FastAPI
- Deploy services with Docker and Kubernetes on AWS

Requirements
- Strong Python and SQL skills
- Experience fine-tuning models with PyTorch
"""


def make_pdf(lines: list[str]) -> bytes:
    """A minimal single-page PDF containing `lines` as text (valid xref table included)."""
    ops = ["BT", "/F1 11 Tf", "72 760 Td", "14 TL"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({escaped}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


class FakeLLM:
    name = "fake-model"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, system, messages):
        self.calls.append((system, list(messages)))
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply
