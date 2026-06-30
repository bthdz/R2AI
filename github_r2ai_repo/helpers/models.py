"""Pydantic request/response models for the production API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class BatchQuestionRequest(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=100)
    top_k: int = Field(default=5, ge=1, le=20)


class CrawlRequest(BaseModel):
    csv_path: str | None = None
    out_dir: str | None = None
    start: int = Field(default=0, ge=0)
    limit: int | None = Field(default=None, ge=1)
    workers: int = Field(default=1, ge=1, le=8)
    sleep: float | None = Field(default=None, ge=0)
    timeout: int | None = Field(default=None, ge=1)
    retries: int | None = Field(default=None, ge=0)
    no_r2ai: bool = False


class SubmissionRow(BaseModel):
    id: int
    question: str
    answer: str
    relevant_docs: list[str] = Field(default_factory=list)
    relevant_articles: list[str] = Field(default_factory=list)


class AnswerResult(BaseModel):
    query: str
    matched_id: int | None
    score: float
    answer: str
    relevant_docs: list[str]
    relevant_articles: list[str]
    candidates: list[dict[str, Any]]


class ApiResponse(BaseModel):
    status: str = "ok"
    data: Any = None
    message: str | None = None

