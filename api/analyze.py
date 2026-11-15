"""Vercel entry point: the FastAPI backend as a Python function. The React app is served
as static files from frontend/dist, and calls /api/analyze on the same origin."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.main import app  # noqa: F401
