"""Production API pipeline over precomputed R2AI artifacts.

The heavy competition pipeline is intentionally kept in ``colab_pipeline``.
This module is a lightweight serving layer for the final ``results.json`` and
for invoking the existing VBPL crawler in ``run.py`` when explicitly enabled.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from helpers.config import Settings
from helpers.decorators import timed
from helpers.models import AnswerResult, CrawlRequest, SubmissionRow
from helpers.text_processing import norm_space, tokenize_vi


class R2AIPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._rows: list[SubmissionRow] | None = None
        self._token_cache: dict[int, set[str]] = {}

    def health(self) -> dict[str, Any]:
        return {
            "results_path": str(self.settings.results_path),
            "results_exists": self.settings.results_path.exists(),
            "full_corpus_path": str(self.settings.full_corpus_path),
            "full_corpus_exists": self.settings.full_corpus_path.exists(),
            "bm25_db_path": str(self.settings.bm25_db_path),
            "bm25_db_exists": self.settings.bm25_db_path.exists(),
            "faiss_index_path": str(self.settings.faiss_index_path),
            "faiss_index_exists": self.settings.faiss_index_path.exists(),
            "faiss_meta_path": str(self.settings.faiss_meta_path),
            "faiss_meta_exists": self.settings.faiss_meta_path.exists(),
            "hyde_path": str(self.settings.hyde_path),
            "hyde_exists": self.settings.hyde_path.exists(),
            "rule_plan_path": str(self.settings.rule_plan_path),
            "rule_plan_exists": self.settings.rule_plan_path.exists(),
            "retrieval_candidates_path": str(self.settings.retrieval_candidates_path),
            "retrieval_candidates_exists": self.settings.retrieval_candidates_path.exists(),
            "graph_path": str(self.settings.graph_path),
            "graph_exists": self.settings.graph_path.exists(),
            "crawl_input_path": str(self.settings.crawl_input_path),
            "crawl_input_exists": self.settings.crawl_input_path.exists(),
            "enable_crawl_endpoint": self.settings.enable_crawl_endpoint,
            "device": self.settings.device,
        }

    def load_rows(self, force: bool = False) -> list[SubmissionRow]:
        if self._rows is not None and not force:
            return self._rows
        if not self.settings.results_path.exists():
            self._rows = []
            return self._rows

        raw = json.loads(self.settings.results_path.read_text(encoding="utf-8"))
        rows: list[SubmissionRow] = []
        for item in raw:
            rows.append(
                SubmissionRow(
                    id=int(item.get("id", 0)),
                    question=norm_space(item.get("question", "")),
                    answer=norm_space(item.get("answer", "")),
                    relevant_docs=[norm_space(x) for x in item.get("relevant_docs", []) if norm_space(x)],
                    relevant_articles=[
                        norm_space(x) for x in item.get("relevant_articles", []) if norm_space(x)
                    ],
                )
            )
        self._rows = rows
        self._token_cache = {}
        return rows

    def _row_tokens(self, row: SubmissionRow) -> set[str]:
        if row.id not in self._token_cache:
            text = " ".join([row.question, row.answer, " ".join(row.relevant_articles)])
            self._token_cache[row.id] = set(tokenize_vi(text, limit=120))
        return self._token_cache[row.id]

    @timed
    def answer(self, question: str, top_k: int = 5) -> dict[str, Any]:
        rows = self.load_rows()
        query = norm_space(question)
        if not rows:
            return AnswerResult(
                query=query,
                matched_id=None,
                score=0.0,
                answer="",
                relevant_docs=[],
                relevant_articles=[],
                candidates=[],
            ).dict()

        query_tokens = set(tokenize_vi(query, limit=80))
        exact_key = norm_space(query).lower()
        scored: list[tuple[float, SubmissionRow]] = []

        for row in rows:
            if row.question.lower() == exact_key:
                scored.append((1.0, row))
                continue
            row_tokens = self._row_tokens(row)
            if not query_tokens or not row_tokens:
                score = 0.0
            else:
                overlap = len(query_tokens & row_tokens)
                score = overlap / max(1, len(query_tokens))
            scored.append((score, row))

        scored.sort(key=lambda item: (item[0], -item[1].id), reverse=True)
        best_score, best = scored[0]
        candidates = [
            {
                "id": row.id,
                "score": round(score, 5),
                "question": row.question,
                "answer_head": row.answer[:300],
            }
            for score, row in scored[:top_k]
        ]
        return AnswerResult(
            query=query,
            matched_id=best.id,
            score=round(best_score, 5),
            answer=best.answer,
            relevant_docs=best.relevant_docs,
            relevant_articles=best.relevant_articles,
            candidates=candidates,
        ).dict()

    def run_vbpl_crawler(self, request: CrawlRequest) -> dict[str, Any]:
        run_py = self.settings.project_root / "run.py"
        if not run_py.exists():
            raise FileNotFoundError(f"Cannot find crawler entrypoint: {run_py}")

        csv_path = Path(request.csv_path) if request.csv_path else self.settings.crawl_input_path
        out_dir = Path(request.out_dir) if request.out_dir else self.settings.crawl_output_dir

        cmd = [
            sys.executable,
            str(run_py),
            "--csv",
            str(csv_path),
            "--out",
            str(out_dir),
            "--start",
            str(request.start),
            "--workers",
            str(request.workers),
        ]
        if request.limit is not None:
            cmd.extend(["--limit", str(request.limit)])
        if request.sleep is not None:
            cmd.extend(["--sleep", str(request.sleep)])
        if request.timeout is not None:
            cmd.extend(["--timeout", str(request.timeout)])
        if request.retries is not None:
            cmd.extend(["--retries", str(request.retries)])
        if request.no_r2ai:
            cmd.append("--no-r2ai")

        completed = subprocess.run(
            cmd,
            cwd=str(self.settings.project_root),
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "command": cmd,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "output_dir": str(out_dir),
        }
