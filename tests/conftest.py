"""Shared test configuration: deterministic, headless, and offline by default."""

import os
import sys
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")

_DEFAULT_WARNING_FORMAT = warnings.formatwarning


def _relative_path(path):
    try:
        return os.path.relpath(path)
    except ValueError:  # Different Windows drives have no relative representation.
        return os.path.basename(path)


def _relative_warning_text(message):
    text = str(message)
    roots = {
        os.getcwd(),
        os.path.expanduser("~"),
        sys.prefix,
        os.environ.get("TMPDIR", os.environ.get("TEMP", "")),
    }
    for root in sorted(filter(None, roots), key=len, reverse=True):
        absolute = os.path.abspath(root)
        text = text.replace(absolute, _relative_path(absolute))
    return text


def _format_relative_warning(message, category, filename, lineno, line=None):
    """Keep pytest and CI warning locations relative to the invocation directory."""
    return _DEFAULT_WARNING_FORMAT(
        _relative_warning_text(message),
        category,
        _relative_path(filename),
        lineno,
        line,
    )


warnings.formatwarning = _format_relative_warning
