"""Basic request validation and guardrails for the API wrapper."""

from __future__ import annotations

from fastapi import HTTPException

from helpers.text_processing import norm_space


def clean_question(question: str, max_chars: int = 2500) -> str:
    value = norm_space(question)
    if not value:
        raise HTTPException(status_code=400, detail="question must not be empty")
    if len(value) > max_chars:
        raise HTTPException(status_code=400, detail=f"question is longer than {max_chars} chars")
    return value


def require_enabled(enabled: bool, feature_name: str) -> None:
    if not enabled:
        raise HTTPException(status_code=403, detail=f"{feature_name} is disabled by configuration")
