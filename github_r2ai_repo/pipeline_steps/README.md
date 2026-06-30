# Pipeline step scripts

Thu muc nay tach pipeline thanh cac diem chay ro rang.

- `00_check_drive_artifacts.py`: kiem tra cac file da giu lai trong Google Drive hoac thu muc `data/` local.
- `01_crawl_vbpl.py`: chay lai crawl du lieu VBPL khi muon rebuild tu dau.
- `02_run_from_kept_artifacts.py`: chay nhanh tu cac artifact tot nhat da giu lai: chunk, BM25, FAISS, HyDE, retrieval candidates.
- `03_run_full_rebuild.py`: rebuild lai tu raw/crawl den submission. Chay rat lau va can GPU.
- `04_rebuild_best_artifacts.py`: rebuild tu dau roi copy output ve dung ten artifact tot nhat dang duoc pipeline artifact-first su dung.
- `10_ingest_normalize_chunk.py`: ingest raw folders, normalize va tao chunk.
- `20_build_indexes.py`: build BM25 SQLite FTS5 va FAISS tu chunk.
- `30_query_retrieve.py`: tao HyDE/query expansion va hybrid candidates.
- `40_rerank_answer_submit.py`: rerank, sinh answer/evidence va export submission.

Mac dinh Colab dung:

```bash
python pipeline_steps/00_check_drive_artifacts.py
python pipeline_steps/02_run_from_kept_artifacts.py
```

Neu muon rebuild tu dau va tao lai cac file artifact tot nhat:

```bash
python pipeline_steps/04_rebuild_best_artifacts.py --overwrite
```

Neu muon rebuild ca artifact phu `rule_plan` va `graph`:

```bash
python pipeline_steps/04_rebuild_best_artifacts.py --overwrite --rebuild-support-artifacts
```

Neu chi muon promote output da co san tu mot run cu:

```bash
python pipeline_steps/04_rebuild_best_artifacts.py --run-id rebuild_best_artifacts_v1 --promote-only --overwrite
```

Neu muon chay tung buoc rebuild:

```bash
python pipeline_steps/10_ingest_normalize_chunk.py
python pipeline_steps/20_build_indexes.py
python pipeline_steps/30_query_retrieve.py
python pipeline_steps/40_rerank_answer_submit.py
```

Mac dinh local/Docker dung `R2AI_BASE_DIR=/app/data`.
