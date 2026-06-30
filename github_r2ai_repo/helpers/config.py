"""Environment driven configuration for the production API wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    results_path: Path
    full_corpus_path: Path
    bm25_db_path: Path
    faiss_index_path: Path
    faiss_meta_path: Path
    hyde_path: Path
    rule_plan_path: Path
    retrieval_candidates_path: Path
    graph_path: Path
    crawl_input_path: Path
    crawl_output_dir: Path
    enable_crawl_endpoint: bool
    device: str
    project_root: Path


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = Path(os.getenv("R2AI_DATA_DIR", os.getenv("R2AI_BASE_DIR", project_root / "data")))

    return Settings(
        data_dir=data_dir,
        results_path=Path(os.getenv("R2AI_RESULTS_PATH", data_dir / "results.json")),
        full_corpus_path=Path(
            os.getenv(
                "R2AI_FULL_CORPUS_PATH",
                data_dir / "processed/chunks/all_chunks_safe_20260622_012757_repaired_min80_plus_vbpl100_20260626_154852.jsonl",
            )
        ),
        bm25_db_path=Path(
            os.getenv(
                "R2AI_BM25_DB_PATH",
                data_dir / "processed/index/bm25_fts5_legal_plus_vbpl100_20260626_154852.sqlite",
            )
        ),
        faiss_index_path=Path(
            os.getenv(
                "R2AI_FAISS_INDEX_PATH",
                data_dir / "processed/index/qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216.index",
            )
        ),
        faiss_meta_path=Path(
            os.getenv(
                "R2AI_FAISS_META_PATH",
                data_dir / "processed/index/qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216_meta.jsonl",
            )
        ),
        hyde_path=Path(
            os.getenv(
                "R2AI_HYDE_PATH",
                data_dir / "processed/query_plans/r2ai_stage1_hyde_qwen25_15b_v1.jsonl",
            )
        ),
        rule_plan_path=Path(
            os.getenv(
                "R2AI_RULE_PLAN_PATH",
                data_dir / "processed/query_plans/r2ai_stage1_query_plans_hybrid_rule_llm_planner_v1_utf8fixed.jsonl",
            )
        ),
        retrieval_candidates_path=Path(
            os.getenv(
                "R2AI_RETRIEVAL_CANDIDATES_PATH",
                data_dir / "processed/retrieval/r2ai_stage1_hybrid_candidates_bm25plus_vbpl100_qwen3plus_20260626_170709.jsonl",
            )
        ),
        graph_path=Path(
            os.getenv(
                "R2AI_GRAPH_PATH",
                data_dir / "processed/graphs/vbpl_relation_graph_exact_v2.jsonl",
            )
        ),
        crawl_input_path=Path(os.getenv("R2AI_CRAWL_INPUT", data_dir / "vbpl_excel_exports")),
        crawl_output_dir=Path(os.getenv("R2AI_CRAWL_OUT", data_dir / "raw/vbpl_crawled")),
        enable_crawl_endpoint=_bool_env("R2AI_ENABLE_CRAWL_ENDPOINT", False),
        device=os.getenv("R2AI_DEVICE", "cpu"),
        project_root=project_root,
    )
