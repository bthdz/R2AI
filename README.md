# R2AI Law RAG

Hệ thống RAG cho hỏi đáp pháp luật Việt Nam. Repo này phục vụ hai mục tiêu:

1. Chạy lại pipeline trên Google Colab từ các artifact tốt nhất đã giữ trên Google Drive.
2. Chạy API local bằng Docker để demo/tra cứu từ `results.json` và kiểm tra artifact.

Các file dữ liệu lớn không commit lên GitHub. Người chạy cần tải dữ liệu/checkpoint từ Google Drive theo đúng cấu trúc bên dưới.

Link dữ liệu/checkpoint:

```text
https://drive.google.com/drive/folders/1EcQwg30qoVF3b2Cfkte-t1bh1ReJQ2nf?usp=drive_link
```

## Cấu Trúc Repo

```text
r2ai-law-rag/
  app/
    server.py                         # FastAPI server
    static/                           # UI tĩnh
  helpers/
    config.py                         # cấu hình bằng biến môi trường
    constants.py
    decorators.py
    guardrails.py
    models.py                         # schema request/response
    pipeline.py                       # serving layer + wrapper crawler
    text_processing.py
  pipeline_steps/
    00_check_drive_artifacts.py        # kiểm tra đủ artifact
    01_crawl_vbpl.py                   # crawl VBPL bằng run.py
    02_run_from_kept_artifacts.py      # chạy Colab từ artifact tốt nhất
    03_run_full_rebuild.py             # rebuild từ raw/crawl
    04_rebuild_best_artifacts.py       # rebuild rồi promote về tên artifact tốt nhất
    10_ingest_normalize_chunk.py       # ingest, normalize, chunk
    20_build_indexes.py                # build BM25 và FAISS
    30_query_retrieve.py               # HyDE/query + hybrid retrieval
    40_rerank_answer_submit.py         # rerank, LLM evidence, submission
  data/                                # dữ liệu local, không commit
  run.py                               # crawler VBPL
  full_colab_pipeline.py               # full pipeline RAG trên Colab
  Dockerfile
  docker-compose.yml
  requirements.txt                     # dependencies cho API/Docker
  requirements-colab.txt               # dependencies cho Colab pipeline
  .env.example
  README.md
```

## Dữ Liệu Cần Tải

Trên Google Drive, thư mục chuẩn là:

```text
/content/drive/MyDrive/R2AI/Law/
```

Khi chạy local/Docker, tải cùng nội dung đó vào:

```text
./data/
```

Cấu trúc tối thiểu:

```text
data/
  R2AIStage1DATA.json
  results.json                         # kết quả tốt nhất để API serve
  raw/
    vbpl_crawled/
    vbpl_crawled_0/
    vbpl_crawled_1/
    vbpl_crawled_2/
    vbpl_crawled_3/
    dvc_tthb/
    supplemental_clean/
  vbpl_excel_exports/
  processed/
    chunks/
      all_chunks_safe_20260622_012757_repaired_min80_plus_vbpl100_20260626_154852.jsonl
    index/
      qwen3_8b_dim1024_sharded_v1/
      bm25_fts5_legal_plus_vbpl100_20260626_154852.sqlite
      qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216.index
      qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216_meta.jsonl
    query_plans/
      r2ai_stage1_hyde_qwen25_15b_v1.jsonl
      r2ai_stage1_query_plans_hybrid_rule_llm_planner_v1_utf8fixed.jsonl
    retrieval/
      r2ai_stage1_hybrid_candidates_bm25plus_vbpl100_qwen3plus_20260626_170709.jsonl
    graphs/
      vbpl_relation_graph_exact_v2.jsonl
```

Link tải dữ liệu/checkpoint:

```text
https://drive.google.com/drive/folders/1EcQwg30qoVF3b2Cfkte-t1bh1ReJQ2nf?usp=drive_link
```

## Mô Hình Và Checkpoint

Pipeline dùng các checkpoint public sau:

| Thành phần | Checkpoint |
| --- | --- |
| Embedding | `Qwen/Qwen3-Embedding-8B` |
| HyDE/query expansion | `Qwen/Qwen2.5-3B-Instruct`, thử nghiệm thêm `Qwen/Qwen2.5-1.5B-Instruct` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Answer/evidence filtering | `Qwen/Qwen2.5-7B-Instruct` |

FAISS/BM25/artifact đã build được chia sẻ trong thư mục dữ liệu. Nếu muốn không tải lại model từ Hugging Face, có thể cache checkpoint trong Drive/HF cache riêng và mount vào runtime.

## Chạy Trên Google Colab Từ Artifact Tốt Nhất

Cách này là đường chạy khuyến nghị để reproduce kết quả nhanh hơn. Nó bỏ qua crawl, chunk, build BM25/FAISS và retrieval vì các file tốt nhất đã có trong Drive.

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content/drive/MyDrive/R2AI/Law/github_r2ai_repo
!pip -q install -r requirements-colab.txt

!python pipeline_steps/00_check_drive_artifacts.py
!python pipeline_steps/02_run_from_kept_artifacts.py
```

Output được ghi vào:

```text
/content/drive/MyDrive/R2AI/Law/processed/full_pipeline/full_rag_v1/
  08_rerank/
  09_answer_evidence/
  10_submissions/
```

File nộp nằm trong:

```text
processed/full_pipeline/full_rag_v1/10_submissions/
```

## Chạy Lại Từ Đầu Trên Colab

Chạy từ đầu rất lâu và nên dùng A100. Nếu chỉ có T4, nên dùng artifact-first ở phần trên.

1. Crawl VBPL nếu muốn crawl lại:

```bash
python pipeline_steps/01_crawl_vbpl.py \
  --csv /content/drive/MyDrive/R2AI/Law/vbpl_excel_exports \
  --out /content/drive/MyDrive/R2AI/Law/raw/vbpl_crawled \
  --workers 2
```

2. Rebuild toàn bộ pipeline:

```bash
python pipeline_steps/03_run_full_rebuild.py
```

3. Rebuild từ đầu và xuất lại đúng các file artifact tốt nhất mà chế độ artifact-first sử dụng:

```bash
python pipeline_steps/04_rebuild_best_artifacts.py --overwrite
```

Lệnh trên sẽ chạy full pipeline rồi copy output về các đường dẫn chuẩn:

```text
processed/chunks/all_chunks_safe_20260622_012757_repaired_min80_plus_vbpl100_20260626_154852.jsonl
processed/index/bm25_fts5_legal_plus_vbpl100_20260626_154852.sqlite
processed/index/qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216.index
processed/index/qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216_meta.jsonl
processed/query_plans/r2ai_stage1_hyde_qwen25_15b_v1.jsonl
processed/retrieval/r2ai_stage1_hybrid_candidates_bm25plus_vbpl100_qwen3plus_20260626_170709.jsonl
processed/submissions/rebuilt_best/results.json
```

Hai artifact phụ `rule_plan` và `graph` mặc định giữ bản đã chia sẻ sẵn. Nếu muốn rebuild cả hai:

```bash
python pipeline_steps/04_rebuild_best_artifacts.py --overwrite --rebuild-support-artifacts
```

Nếu muốn copy cả embedding shards:

```bash
python pipeline_steps/04_rebuild_best_artifacts.py --overwrite --promote-shards
```

Nếu đã có output trong `processed/full_pipeline/rebuild_best_artifacts_v1/` và chỉ muốn copy về tên chuẩn:

```bash
python pipeline_steps/04_rebuild_best_artifacts.py --promote-only --overwrite
```

Hoặc chạy từng cụm bước:

```bash
python pipeline_steps/10_ingest_normalize_chunk.py
python pipeline_steps/20_build_indexes.py
python pipeline_steps/30_query_retrieve.py
python pipeline_steps/40_rerank_answer_submit.py
```

Các stage chính:

```text
crawl/ingest -> normalize -> chunk -> BM25 FTS5 -> FAISS
-> HyDE/query planning -> hybrid retrieval -> rerank
-> LLM answer/evidence -> submission variants
```

## Chạy Local Bằng Docker

1. Clone repo.

2. Tải dữ liệu từ Google Drive về `./data/` đúng cấu trúc ở mục “Dữ Liệu Cần Tải”.

3. Đặt file kết quả tốt nhất thành:

```text
data/results.json
```

4. Tạo `.env`:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

5. Chạy Docker:

```bash
docker compose up --build
```

6. Kiểm tra:

```bash
curl http://localhost:8000/health
```

Mở UI:

```text
http://localhost:8000
```

Gọi API:

```bash
curl -X POST http://localhost:8000/api/answer \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Người lao động được nghỉ việc riêng khi nào?\",\"top_k\":5}"
```

## Chạy Không Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

## Schema Output

`results.json`:

```json
[
  {
    "id": 1,
    "question": "...",
    "answer": "...",
    "relevant_docs": ["mã văn bản|tên văn bản"],
    "relevant_articles": ["mã văn bản|tên văn bản|Điều 1"]
  }
]
```

## Ghi Chú Vận Hành

- Docker/API không rebuild index. Nó serve kết quả và hỗ trợ endpoint crawler nếu bật `R2AI_ENABLE_CRAWL_ENDPOINT=true`.
- Full RAG cần GPU và chạy qua `full_colab_pipeline.py` hoặc các script trong `pipeline_steps/`.
- Các file nặng gồm chunk JSONL, BM25 SQLite và FAISS index phải để ngoài GitHub, chia sẻ bằng Drive.
