"""FastAPI server for the R2AI production wrapper."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from helpers.config import load_settings
from helpers.constants import APP_NAME, APP_VERSION
from helpers.guardrails import clean_question, require_enabled
from helpers.models import ApiResponse, BatchQuestionRequest, CrawlRequest, QuestionRequest
from helpers.pipeline import R2AIPipeline


settings = load_settings()
pipeline = R2AIPipeline(settings)

app = FastAPI(title=APP_NAME, version=APP_VERSION)
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> ApiResponse:
    return ApiResponse(data=pipeline.health())


@app.get("/api/config")
def api_config() -> ApiResponse:
    return ApiResponse(
        data={
            "app": APP_NAME,
            "version": APP_VERSION,
            "results_path": str(settings.results_path),
            "data_dir": str(settings.data_dir),
            "enable_crawl_endpoint": settings.enable_crawl_endpoint,
        }
    )


@app.post("/api/answer")
def answer(request: QuestionRequest) -> ApiResponse:
    question = clean_question(request.question)
    return ApiResponse(data=pipeline.answer(question, top_k=request.top_k))


@app.post("/api/batch-answer")
def batch_answer(request: BatchQuestionRequest) -> ApiResponse:
    results = [pipeline.answer(clean_question(question), top_k=request.top_k) for question in request.questions]
    return ApiResponse(data=results)


@app.post("/api/crawl-vbpl")
def crawl_vbpl(request: CrawlRequest) -> ApiResponse:
    require_enabled(settings.enable_crawl_endpoint, "VBPL crawler endpoint")
    result = pipeline.run_vbpl_crawler(request)
    status = "ok" if result["returncode"] == 0 else "error"
    return ApiResponse(status=status, data=result)
