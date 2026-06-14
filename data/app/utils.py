# app/utils.py — Shared utility helpers

import re


def sanitize_text(text: str) -> str:
    """
    Basic sanitization: strip leading/trailing whitespace and collapse
    multiple consecutive whitespace characters into a single space.
    """
    if not text:
        return ''
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def truncate_text(text: str, max_length: int = 5000) -> str:
    """
    Truncate *text* to *max_length* characters to avoid processing
    extremely long inputs.
    """
    if len(text) > max_length:
        return text[:max_length]
    return text


def format_confidence(value: float) -> str:
    """Return a confidence value as a formatted percentage string, e.g. '87.3%'."""
    return f'{value:.1f}%'
