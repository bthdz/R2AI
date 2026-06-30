# Thu muc du lieu runtime

Thu muc `data/` dung cho Docker/local. Hay tai cac file da chia se tu Google
Drive va dat dung cau truc sau:

```text
data/
  R2AIStage1DATA.json
  results.json                         # submission tot nhat de API tra loi nhanh
  raw/
    vbpl_crawled/
    vbpl_crawled_0/
    vbpl_crawled_1/
    vbpl_crawled_2/
    vbpl_crawled_3/
    dvc_tthb/
    supplemental_clean/
  vbpl_excel_exports/                  # neu muon crawl lai tu file excel/csv
  processed/
    chunks/
      all_chunks_safe_20260622_012757_repaired_min80_plus_vbpl100_20260626_154852.jsonl
    index/
      bm25_fts5_legal_plus_vbpl100_20260626_154852.sqlite
      qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216.index
      qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216_meta.jsonl
      qwen3_8b_dim1024_sharded_v1/     # neu can rebuild/kiem tra embedding shards
    query_plans/
      r2ai_stage1_hyde_qwen25_15b_v1.jsonl
      r2ai_stage1_query_plans_hybrid_rule_llm_planner_v1_utf8fixed.jsonl
    retrieval/
      r2ai_stage1_hybrid_candidates_bm25plus_vbpl100_qwen3plus_20260626_170709.jsonl
    graphs/
      vbpl_relation_graph_exact_v2.jsonl
```

Kiem tra nhanh:

```bash
python pipeline_steps/00_check_drive_artifacts.py
```

Trong Docker, thu muc nay duoc mount vao `/app/data`.

