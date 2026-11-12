"""Turn an uploaded resume into text, and text into scoreable units."""

from __future__ import annotations

import io
import re

from pypdf import PdfReader

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PAGES = 15
# Bullet glyphs often survive PDF extraction as control or private-use characters.
_BULLET = re.compile(r"^\s*(?:[-*•▪●◦‣·\x7f\uf0b7\uf0a7]|\d+[.)])\s+")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_LOWER_START = re.compile(r"^[a-z(]")
WRAP_WIDTH = 85  # a line at least this long that lacks end punctuation was wrapped by the PDF


class ParseError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ParseError("file is larger than 5 MB")
    name = filename.lower()
    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise ParseError("PDF is password protected")
            pages = reader.pages[:MAX_PAGES]
            text = "\n".join(page.extract_text() or "" for page in pages)
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError("could not read this PDF") from exc
        if len(text.strip()) < 40:
            raise ParseError("no text found in the PDF - it may be a scanned image; paste the text instead")
        return text
    if name.endswith((".txt", ".md")):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ParseError("text file is not UTF-8") from exc
    raise ParseError("unsupported file type: upload a PDF, .txt or .md, or paste the text")


def split_units(text: str, min_words: int = 3, max_units: int = 120) -> list[str]:
    """Split a resume or job description into lines worth comparing.

    Joins wrapped lines back together, strips bullet markers, splits long
    paragraphs into sentences, and drops headings and fragments.
    """
    merged: list[str] = []
    for raw in text.replace("\r", "").split("\n"):
        if not raw.strip():
            merged.append("")
            continue
        is_bullet = bool(_BULLET.match(raw))
        line = _CONTROL.sub("", _BULLET.sub("", raw)).strip()
        if line.isupper() and len(line.split()) <= 4:
            merged.append("")  # section heading ("EXPERIENCE"): a hard break, and not a unit itself
            continue
        previous = merged[-1] if merged else ""
        wrapped = _LOWER_START.match(line) or len(previous) >= WRAP_WIDTH
        if previous and not is_bullet and wrapped and not previous.endswith((".", ":", ";")):
            merged[-1] = f"{previous} {line}"
        else:
            merged.append(line)

    units: list[str] = []
    for block in merged:
        for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z])", block):
            sentence = " ".join(sentence.split()).strip(" .;")
            if len(sentence.split()) >= min_words and sentence not in units:
                units.append(sentence)
    return units[:max_units]
