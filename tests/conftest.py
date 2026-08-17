"""Shared test configuration: deterministic, headless, and offline by default."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

