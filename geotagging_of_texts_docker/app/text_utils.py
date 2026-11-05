from __future__ import annotations
import re


def split_text(text: str) -> list[str]:
    """Split plain extracted text on blank lines while trimming layout whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pieces = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n+", text)]
    return [p for p in pieces if p]
