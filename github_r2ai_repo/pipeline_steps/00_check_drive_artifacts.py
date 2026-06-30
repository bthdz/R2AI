from __future__ import annotations

import os
from pathlib import Path


def default_base_dir() -> Path:
    env = os.environ.get("R2AI_BASE_DIR") or os.environ.get("R2AI_DATA_DIR")
    if env:
        return Path(env)
    colab_dir = Path("/content/drive/MyDrive/R2AI/Law")
    if colab_dir.exists():
        return colab_dir
    return Path(__file__).resolve().parents[1] / "data"


BASE_DIR = default_base_dir()

REQUIRED = [
    "R2AIStage1DATA.json",
    "processed/chunks/all_chunks_safe_20260622_012757_repaired_min80_plus_vbpl100_20260626_154852.jsonl",
    "processed/index/bm25_fts5_legal_plus_vbpl100_20260626_154852.sqlite",
    "processed/index/qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216.index",
    "processed/index/qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216_meta.jsonl",
    "processed/query_plans/r2ai_stage1_hyde_qwen25_15b_v1.jsonl",
    "processed/query_plans/r2ai_stage1_query_plans_hybrid_rule_llm_planner_v1_utf8fixed.jsonl",
    "processed/retrieval/r2ai_stage1_hybrid_candidates_bm25plus_vbpl100_qwen3plus_20260626_170709.jsonl",
    "processed/graphs/vbpl_relation_graph_exact_v2.jsonl",
]

RAW_DIRS = [
    "raw/vbpl_crawled",
    "raw/vbpl_crawled_0",
    "raw/vbpl_crawled_1",
    "raw/vbpl_crawled_2",
    "raw/vbpl_crawled_3",
    "raw/dvc_tthb",
    "raw/supplemental_clean",
]


def size_mb(path: Path) -> str:
    if path.is_dir():
        return "dir"
    return f"{path.stat().st_size / (1024 ** 2):.1f} MB"


def main() -> None:
    print("R2AI_BASE_DIR:", BASE_DIR)
    missing = []
    for rel in REQUIRED:
        path = BASE_DIR / rel
        ok = path.exists()
        print(f"{'OK  ' if ok else 'MISS'} {rel} ({size_mb(path) if ok else 'missing'})")
        if not ok:
            missing.append(rel)

    print("\nRaw folders for full rebuild:")
    for rel in RAW_DIRS:
        path = BASE_DIR / rel
        print(f"{'OK  ' if path.exists() else 'SKIP'} {rel}")

    if missing:
        raise SystemExit("Missing required artifacts:\n" + "\n".join(missing))


if __name__ == "__main__":
    main()
