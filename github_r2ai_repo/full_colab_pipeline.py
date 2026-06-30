"""
Full R2AI Law RAG pipeline for Colab.

Stages:
  1. Crawl / ingest legal documents.
  2. Normalize + dedupe documents/articles.
  3. Chunk legal articles.
  4. Build BM25 SQLite FTS5.
  5. Build dense FAISS index.
  6. Generate query expansion + HyDE.
  7. Hybrid retrieval: BM25 + dense + RRF.
  8. Cross-encoder rerank.
  9. LLM answer + evidence selection.
 10. Build submission variants.

Run in Colab:
  %cd /content/drive/MyDrive/R2AI/Law/github_r2ai_repo
  !python pipeline_steps/02_run_from_kept_artifacts.py

Important:
  - Every expensive stage writes checkpoints.
  - Use RUN_STAGES near the top to resume only the missing part.
  - Default mode reuses the best kept artifacts in /content/drive/MyDrive/R2AI/Law.
  - For a full rebuild, run pipeline_steps/03_run_full_rebuild.py.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# =============================================================================
# 0) CONFIG
# =============================================================================


@dataclass
class PipelineConfig:
    base_dir: Path = field(default_factory=lambda: Path(os.environ.get("R2AI_BASE_DIR", "/content/drive/MyDrive/R2AI/Law")))
    work_dir: Path = field(default_factory=lambda: Path(os.environ.get("R2AI_WORK_DIR", "/content/r2ai_full_pipeline")))
    run_id: str = field(default_factory=lambda: os.environ.get("R2AI_RUN_ID", "full_rag_v1"))  # keep stable for resume

    question_file: Path | None = None
    web_mapping_file: Path | None = None

    # Default mode for reproduction: reuse the best artifacts kept on Google
    # Drive. Set use_kept_artifacts=False when rebuilding from raw data.
    use_kept_artifacts: bool = True
    kept_chunk_file: Path | None = None
    kept_bm25_db: Path | None = None
    kept_faiss_index: Path | None = None
    kept_faiss_meta: Path | None = None
    kept_hyde_file: Path | None = None
    kept_rule_plan_file: Path | None = None
    kept_candidates_file: Path | None = None
    kept_graph_file: Path | None = None

    # Existing crawl folders; each can contain documents.jsonl/articles.jsonl.
    raw_dirs: list[str] = field(
        default_factory=lambda: [
            "raw/vbpl_crawled",
            "raw/vbpl_crawled_0",
            "raw/vbpl_crawled_1",
            "raw/vbpl_crawled_2",
            "raw/vbpl_crawled_3",
            "raw/dvc_tthb",
            "raw/supplemental_clean",
        ]
    )

    # Data choices.
    skip_expired_docs: bool = True
    crawl_max_urls: int | None = None
    crawl_sleep_sec: float = 0.4
    request_timeout_sec: int = 25

    # Chunking.
    chunk_target_chars: int = 2200
    chunk_max_chars: int = 3600
    chunk_overlap_paragraphs: int = 1
    min_chunk_chars: int = 80

    # Embedding / FAISS.
    embed_model: str = "Qwen/Qwen3-Embedding-8B"
    embed_batch_size: int = 32
    embed_shard_size: int = 50000
    dense_topk: int = 60

    # Query expansion / HyDE.
    hyde_model: str = "Qwen/Qwen2.5-3B-Instruct"
    hyde_batch_size: int = 16
    hyde_max_new_tokens: int = 160

    # Retrieval.
    bm25_topk: int = 80
    final_candidate_topk: int = 80
    rrf_k: int = 60

    # Reranker.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_batch_size: int = 96
    rerank_max_pair_chars: int = 1600

    # LLM answer/evidence.
    answer_model: str = "Qwen/Qwen2.5-7B-Instruct"
    answer_batch_size: int = 1
    answer_context_articles: int = 5
    answer_article_chars: int = 1000
    answer_max_new_tokens: int = 320

    # Submission variants.
    submission_variants: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "p4_d2": {"max_articles": 4, "max_docs": 2, "max_per_doc": 2},
            "p5_d3": {"max_articles": 5, "max_docs": 3, "max_per_doc": 2},
            "p3_d2_precision": {"max_articles": 3, "max_docs": 2, "max_per_doc": 2},
            "p6_d3_recall": {"max_articles": 6, "max_docs": 3, "max_per_doc": 3},
        }
    )

    def __post_init__(self) -> None:
        self.question_file = self.question_file or self.base_dir / "R2AIStage1DATA.json"
        self.web_mapping_file = self.web_mapping_file or self.base_dir / "r2ai_question_based_web_mapping_report.xlsx"
        self.kept_chunk_file = self.kept_chunk_file or self.base_dir / "processed/chunks/all_chunks_safe_20260622_012757_repaired_min80_plus_vbpl100_20260626_154852.jsonl"
        self.kept_bm25_db = self.kept_bm25_db or self.base_dir / "processed/index/bm25_fts5_legal_plus_vbpl100_20260626_154852.sqlite"
        self.kept_faiss_index = self.kept_faiss_index or self.base_dir / "processed/index/qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216.index"
        self.kept_faiss_meta = self.kept_faiss_meta or self.base_dir / "processed/index/qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216_meta.jsonl"
        self.kept_hyde_file = self.kept_hyde_file or self.base_dir / "processed/query_plans/r2ai_stage1_hyde_qwen25_15b_v1.jsonl"
        self.kept_rule_plan_file = self.kept_rule_plan_file or self.base_dir / "processed/query_plans/r2ai_stage1_query_plans_hybrid_rule_llm_planner_v1_utf8fixed.jsonl"
        self.kept_candidates_file = self.kept_candidates_file or self.base_dir / "processed/retrieval/r2ai_stage1_hybrid_candidates_bm25plus_vbpl100_qwen3plus_20260626_170709.jsonl"
        self.kept_graph_file = self.kept_graph_file or self.base_dir / "processed/graphs/vbpl_relation_graph_exact_v2.jsonl"

    def _use_kept(self, path: Path | None) -> bool:
        return bool(self.use_kept_artifacts and path and Path(path).exists())

    @property
    def out_dir(self) -> Path:
        return self.base_dir / "processed" / "full_pipeline" / self.run_id

    @property
    def report_dir(self) -> Path:
        return self.base_dir / "reports" / self.run_id

    @property
    def crawl_docs_raw(self) -> Path:
        return self.out_dir / "01_crawl" / "documents_raw.jsonl"

    @property
    def crawl_articles_raw(self) -> Path:
        return self.out_dir / "01_crawl" / "articles_raw.jsonl"

    @property
    def normalized_docs(self) -> Path:
        return self.out_dir / "02_normalized" / "documents_normalized.jsonl"

    @property
    def normalized_articles(self) -> Path:
        return self.out_dir / "02_normalized" / "articles_normalized.jsonl"

    @property
    def chunk_file(self) -> Path:
        if self._use_kept(self.kept_chunk_file):
            return Path(self.kept_chunk_file)
        return self.out_dir / "03_chunks" / "chunks.jsonl"

    @property
    def bm25_db(self) -> Path:
        if self._use_kept(self.kept_bm25_db):
            return Path(self.kept_bm25_db)
        return self.out_dir / "04_bm25" / "bm25_fts5.sqlite"

    @property
    def dense_dir(self) -> Path:
        return self.out_dir / "05_dense"

    @property
    def faiss_index(self) -> Path:
        if self._use_kept(self.kept_faiss_index):
            return Path(self.kept_faiss_index)
        return self.dense_dir / "dense.index"

    @property
    def faiss_meta(self) -> Path:
        if self._use_kept(self.kept_faiss_meta):
            return Path(self.kept_faiss_meta)
        return self.dense_dir / "dense_meta.jsonl"

    @property
    def hyde_file(self) -> Path:
        if self._use_kept(self.kept_hyde_file):
            return Path(self.kept_hyde_file)
        return self.out_dir / "06_query" / "query_hyde.jsonl"

    @property
    def question_vectors(self) -> Path:
        return self.out_dir / "06_query" / "question_vectors.npy"

    @property
    def candidates_file(self) -> Path:
        if self._use_kept(self.kept_candidates_file):
            return Path(self.kept_candidates_file)
        return self.out_dir / "07_retrieval" / "hybrid_candidates.jsonl"

    @property
    def rerank_file(self) -> Path:
        return self.out_dir / "08_rerank" / "reranked_candidates.jsonl"

    @property
    def evidence_file(self) -> Path:
        return self.out_dir / "09_answer_evidence" / "answer_evidence.jsonl"

    @property
    def submission_dir(self) -> Path:
        return self.out_dir / "10_submissions"


CFG = PipelineConfig()

RUN_STAGES = {
    "install": False,
    "mount_drive": True,
    "crawl_or_ingest": False,       # set True if you want to crawl mapping URLs
    "normalize": False,
    "chunk": False,
    "build_bm25": False,
    "build_faiss": False,
    "query_hyde": False,
    "retrieve": False,
    "rerank": True,
    "answer_evidence": True,
    "build_submission": True,
}


# =============================================================================
# 1) COMMON UTILS
# =============================================================================


LEGAL_ID_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)*\b", re.I)
LEGAL_ID_SHORT_RE = re.compile(r"\b\d{1,4}/[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)+\b", re.I)
ARTICLE_RE = re.compile(r"(?=Điều\s+([0-9]+[a-zA-Z]?|toan_van)\.?\s)", re.I)
CJK_RE = re.compile(r"[\u3400-\u9fff]")


def install_deps() -> None:
    packages = [
        "requests",
        "beautifulsoup4",
        "lxml",
        "pandas",
        "openpyxl",
        "tqdm",
        "ftfy",
        "numpy",
        "faiss-cpu",
        "sentence-transformers",
        "transformers",
        "accelerate",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *packages])


def mount_drive_if_colab() -> None:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as exc:
        print("Drive mount skipped:", exc)


def ensure_dirs(cfg: PipelineConfig) -> None:
    for p in [
        cfg.out_dir,
        cfg.report_dir,
        cfg.crawl_docs_raw.parent,
        cfg.crawl_articles_raw.parent,
        cfg.normalized_docs.parent,
        cfg.chunk_file.parent,
        cfg.bm25_db.parent,
        cfg.dense_dir,
        cfg.hyde_file.parent,
        cfg.candidates_file.parent,
        cfg.rerank_file.parent,
        cfg.evidence_file.parent,
        cfg.submission_dir,
        cfg.work_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def validate_kept_artifacts(cfg: PipelineConfig) -> None:
    """Fail early when reproduction mode is enabled and kept artifacts are missing."""
    if not cfg.use_kept_artifacts:
        return

    required = {
        "questions": cfg.question_file,
        "chunks": cfg.kept_chunk_file,
        "bm25_sqlite": cfg.kept_bm25_db,
        "faiss_index": cfg.kept_faiss_index,
        "faiss_meta": cfg.kept_faiss_meta,
        "hyde": cfg.kept_hyde_file,
        "retrieval_candidates": cfg.kept_candidates_file,
    }
    optional = {
        "rule_query_plan": cfg.kept_rule_plan_file,
        "relation_graph": cfg.kept_graph_file,
    }

    print("\nKEPT ARTIFACT CHECK")
    missing: list[str] = []
    for name, path in required.items():
        ok = bool(path and Path(path).exists())
        size = f"{Path(path).stat().st_size / (1024 ** 2):.1f} MB" if ok else "missing"
        print(f"  {'OK' if ok else 'MISS'} {name}: {path} ({size})")
        if not ok:
            missing.append(f"{name}: {path}")
    for name, path in optional.items():
        ok = bool(path and Path(path).exists())
        size = f"{Path(path).stat().st_size / (1024 ** 2):.1f} MB" if ok else "optional missing"
        print(f"  {'OK' if ok else 'SKIP'} {name}: {path} ({size})")

    if missing:
        raise FileNotFoundError(
            "Missing required kept artifacts. Put the kept Google Drive files in the documented paths, "
            "or set CFG.use_kept_artifacts=False and enable rebuild stages.\n"
            + "\n".join(missing)
        )


def norm_space(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def vi_fold(s: Any) -> str:
    import unicodedata

    x = norm_space(s).lower()
    x = unicodedata.normalize("NFD", x)
    x = "".join(ch for ch in x if unicodedata.category(ch) != "Mn")
    return x.replace("đ", "d")


def repair_text(s: Any) -> str:
    text = norm_space(s)
    try:
        from ftfy import fix_text

        text = fix_text(text)
    except Exception:
        pass
    return norm_space(text)


def extract_legal_id(s: Any) -> str:
    text = norm_space(s)
    m = LEGAL_ID_RE.search(text) or LEGAL_ID_SHORT_RE.search(text)
    return m.group(0).upper().replace("Ð", "Đ") if m else ""


def infer_doc_type(title: str) -> str:
    t = vi_fold(title)
    if "bo luat" in t:
        return "Bộ luật"
    if re.search(r"\bluat\b", t):
        return "Luật"
    if "nghi dinh" in t:
        return "Nghị định"
    if "thong tu" in t:
        return "Thông tư"
    if "nghi quyet" in t:
        return "Nghị quyết"
    if "quyet dinh" in t:
        return "Quyết định"
    if "phap lenh" in t:
        return "Pháp lệnh"
    return ""


def article_label(no: Any) -> str:
    x = norm_space(no)
    x = re.sub(r"^điều\s+", "", x, flags=re.I)
    x = re.sub(r"^dieu\s+", "", x, flags=re.I)
    return f"Điều {x}" if x else ""


def doc_key(legal_id: str, title: str) -> str:
    return legal_id or sha1_text(title)[:16]


def article_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            norm_space(row.get("legal_id") or extract_legal_id(row.get("title"))),
            vi_fold(row.get("title")),
            norm_space(row.get("article_no")),
        ]
    )


def article_citation(row: dict[str, Any]) -> str:
    code = norm_space(row.get("legal_id") or extract_legal_id(row.get("title")))
    title = norm_space(row.get("title"))
    no = norm_space(row.get("article_no"))
    if not no:
        no = norm_space(row.get("article_label")).replace("Điều ", "")
    label = article_label(no)
    return f"{code}|{title}|{label}" if code else f"{title}|{label}"


def doc_citation(row: dict[str, Any]) -> str:
    code = norm_space(row.get("legal_id") or extract_legal_id(row.get("title")))
    title = norm_space(row.get("title"))
    return f"{code}|{title}" if code else title


def iter_jsonl(path: Path, strict: bool = False) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip().lstrip("\ufeff")
            if not s:
                continue
            if not s.startswith("{"):
                if strict:
                    raise ValueError(f"Bad JSONL line {path}:{line_no}")
                continue
            try:
                obj = json.loads(s)
            except Exception:
                if strict:
                    raise
                continue
            if isinstance(obj, dict):
                yield obj


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_questions(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj if isinstance(obj, list) else (obj.get("data") or obj.get("questions") or obj.get("rows"))
    out = []
    for i, x in enumerate(rows, start=1):
        out.append(
            {
                "id": int(x.get("id", i)),
                "question": repair_text(x.get("question") or x.get("query") or x.get("text")),
            }
        )
    return out


def load_seen_ids(path: Path) -> set[int]:
    return {int(x["id"]) for x in iter_jsonl(path) if "id" in x}


def looks_local(title: str) -> bool:
    t = vi_fold(title)
    return bool(
        re.search(
            r"\b(ubnd|hdnd|uy ban nhan dan|hoi dong nhan dan|tinh|thanh pho|huyen|quan|xa|phuong|dia ban)\b",
            t,
        )
    )


def question_allows_local(q: str) -> bool:
    qf = vi_fold(q)
    return bool(
        re.search(
            r"(dia phuong|dia ban|tinh|thanh pho|ubnd|hdnd|uy ban|hoi dong nhan dan|cap tinh|cap huyen|cap xa)",
            qf,
        )
    )


def boilerplate_article(title: str) -> bool:
    t = vi_fold(title)
    return any(
        x in t
        for x in [
            "pham vi dieu chinh",
            "doi tuong ap dung",
            "giai thich tu ngu",
            "hieu luc thi hanh",
            "trach nhiem thi hanh",
            "dieu khoan thi hanh",
            "to chuc thuc hien",
        ]
    )


def question_needs_boilerplate(q: str) -> bool:
    qf = vi_fold(q)
    return bool(
        re.search(
            r"(pham vi|doi tuong|ap dung|giai thich|khai niem|hieu luc|trach nhiem|thi hanh|to chuc|nguyen tac)",
            qf,
        )
    )


def doc_type_bonus(doc_type: str, title: str) -> float:
    t = vi_fold(f"{doc_type} {title}")
    if "bo luat" in t:
        return 0.10
    if re.search(r"\bluat\b", t):
        return 0.08
    if "nghi dinh" in t:
        return 0.04
    if "thong tu" in t:
        return 0.02
    if "quyet dinh" in t:
        return -0.03
    return 0.0


# =============================================================================
# 2) CRAWL / INGEST
# =============================================================================


def extract_urls_from_mapping(path: Path) -> list[str]:
    if not path.exists():
        return []
    import pandas as pd

    urls: list[str] = []
    if path.suffix.lower() in [".xlsx", ".xls"]:
        sheets = pd.read_excel(path, sheet_name=None)
        frames = list(sheets.values())
    else:
        frames = [pd.read_csv(path)]

    for df in frames:
        for col in df.columns:
            for value in df[col].dropna().astype(str).tolist():
                for url in re.findall(r"https?://[^\s\"'<>]+", value):
                    urls.append(url.rstrip(").,;"))
    return sorted(set(urls))


def html_to_text_and_title(html: str, url: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)
    text = soup.get_text("\n", strip=True)
    text = repair_text(text)
    return text, repair_text(title or url)


def split_articles_from_text(text: str) -> list[dict[str, str]]:
    text = repair_text(text)
    matches = list(re.finditer(r"Điều\s+([0-9]+[a-zA-Z]?|toan_van)\.?\s+([^\n]{0,180})", text, flags=re.I))
    if not matches:
        return [{"article_no": "toan_van", "article_title": "Toàn văn", "text": text}]

    out = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = repair_text(text[start:end])
        no = repair_text(m.group(1))
        title = repair_text(m.group(2))
        out.append({"article_no": no, "article_title": title, "text": body})
    return out


def crawl_url(url: str, timeout: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 R2AI legal research crawler; contact: local",
        "Accept-Language": "vi,en;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    text, title = html_to_text_and_title(resp.text, url)
    legal_id = extract_legal_id(title) or extract_legal_id(text)
    dockey = doc_key(legal_id, title)
    doc = {
        "doc_id": dockey,
        "legal_id": legal_id,
        "title": title,
        "doc_type": infer_doc_type(title),
        "url": url,
        "source": "web",
        "raw_text_len": len(text),
    }
    articles = []
    for a in split_articles_from_text(text):
        articles.append({**doc, **a})
    return doc, articles


def ingest_existing_raw_dirs(cfg: PipelineConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    docs: list[dict[str, Any]] = []
    articles: list[dict[str, Any]] = []
    for rel in cfg.raw_dirs:
        d = cfg.base_dir / rel
        if not d.exists():
            continue
        for name in ["documents.jsonl", "documents (1).jsonl", "documents (2).jsonl"]:
            p = d / name
            if p.exists():
                docs.extend(iter_jsonl(p))
        for name in ["articles.jsonl", "articles.json", "articles (1).jsonl", "articles (2).jsonl"]:
            p = d / name
            if not p.exists():
                continue
            if p.suffix == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, list):
                    articles.extend(obj)
                elif isinstance(obj, dict):
                    articles.extend(obj.get("data") or obj.get("articles") or [])
            else:
                articles.extend(iter_jsonl(p))
    return docs, articles


def stage_crawl_or_ingest(cfg: PipelineConfig) -> None:
    ensure_dirs(cfg)
    seen_urls_path = cfg.out_dir / "01_crawl" / "seen_urls.json"
    seen_urls = set(json.loads(seen_urls_path.read_text(encoding="utf-8"))) if seen_urls_path.exists() else set()

    # Always ingest existing raw dirs first; this is cheap and checkpointed by dedupe later.
    existing_docs, existing_articles = ingest_existing_raw_dirs(cfg)
    if existing_docs or existing_articles:
        print("ingest existing raw:", len(existing_docs), "docs", len(existing_articles), "articles")
        for x in existing_docs:
            append_jsonl(cfg.crawl_docs_raw, x)
        for x in existing_articles:
            append_jsonl(cfg.crawl_articles_raw, x)

    urls = extract_urls_from_mapping(cfg.web_mapping_file)
    if cfg.crawl_max_urls:
        urls = urls[: cfg.crawl_max_urls]
    print("crawl urls:", len(urls), "already seen:", len(seen_urls))

    ok = 0
    fail = 0
    for i, url in enumerate(urls, start=1):
        if url in seen_urls:
            continue
        try:
            doc, articles = crawl_url(url, cfg.request_timeout_sec)
            append_jsonl(cfg.crawl_docs_raw, doc)
            for a in articles:
                append_jsonl(cfg.crawl_articles_raw, a)
            ok += 1
        except Exception as exc:
            fail += 1
            append_jsonl(cfg.out_dir / "01_crawl" / "crawl_errors.jsonl", {"url": url, "error": repr(exc)})
        seen_urls.add(url)
        if i % 20 == 0:
            write_json(seen_urls_path, sorted(seen_urls))
            print(f"crawl {i}/{len(urls)} ok={ok} fail={fail}")
        time.sleep(cfg.crawl_sleep_sec)
    write_json(seen_urls_path, sorted(seen_urls))
    print("crawl done", {"ok": ok, "fail": fail})


# =============================================================================
# 3) NORMALIZE + DEDUPE
# =============================================================================


def normalize_doc(raw: dict[str, Any]) -> dict[str, Any] | None:
    title = repair_text(raw.get("title") or raw.get("ten_van_ban") or raw.get("name"))
    if not title:
        return None
    legal_id = repair_text(raw.get("legal_id") or raw.get("so_ky_hieu") or raw.get("code") or extract_legal_id(title))
    url = repair_text(raw.get("url") or raw.get("source_url"))
    doc_type = repair_text(raw.get("doc_type") or raw.get("loai_van_ban") or infer_doc_type(title))
    status = repair_text(raw.get("status") or raw.get("tinh_trang") or raw.get("hieu_luc"))
    dockey = repair_text(raw.get("doc_id") or doc_key(legal_id, title))
    return {
        "doc_id": dockey,
        "legal_id": legal_id,
        "title": title,
        "doc_type": doc_type,
        "url": url,
        "status": status,
        "source": repair_text(raw.get("source") or "vbpl"),
    }


def normalize_article(raw: dict[str, Any], doc_lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    title = repair_text(raw.get("title") or raw.get("ten_van_ban") or raw.get("doc_title"))
    legal_id = repair_text(raw.get("legal_id") or raw.get("so_ky_hieu") or extract_legal_id(title) or extract_legal_id(raw.get("text")))
    dockey = repair_text(raw.get("doc_id") or doc_key(legal_id, title))
    doc = doc_lookup.get(dockey) or {}
    title = title or doc.get("title", "")
    if not title:
        return None

    text = repair_text(raw.get("text") or raw.get("content") or raw.get("body"))
    if not text:
        return None

    article_no = repair_text(raw.get("article_no") or raw.get("dieu") or raw.get("article") or "")
    article_title = repair_text(raw.get("article_title") or raw.get("ten_dieu") or raw.get("heading") or "")

    if not article_no:
        m = re.search(r"Điều\s+([0-9]+[a-zA-Z]?|toan_van)\.?", text, flags=re.I)
        article_no = m.group(1) if m else "toan_van"

    if not article_title:
        m = re.search(r"Điều\s+[0-9]+[a-zA-Z]?\.\s*([^\n]{0,160})", text, flags=re.I)
        article_title = repair_text(m.group(1)) if m else article_label(article_no)

    return {
        "doc_id": dockey,
        "legal_id": legal_id or doc.get("legal_id", ""),
        "title": title,
        "doc_type": repair_text(raw.get("doc_type") or doc.get("doc_type") or infer_doc_type(title)),
        "url": repair_text(raw.get("url") or doc.get("url") or ""),
        "status": repair_text(raw.get("status") or doc.get("status") or ""),
        "source": repair_text(raw.get("source") or doc.get("source") or "vbpl"),
        "article_no": article_no,
        "article_label": article_label(article_no),
        "article_title": article_title,
        "text": text,
    }


def stage_normalize(cfg: PipelineConfig) -> None:
    ensure_dirs(cfg)
    if cfg.normalized_docs.exists() and cfg.normalized_articles.exists():
        print("normalize skipped, exists:", cfg.normalized_articles)
        return

    raw_docs, raw_articles = ingest_existing_raw_dirs(cfg)
    raw_docs.extend(iter_jsonl(cfg.crawl_docs_raw))
    raw_articles.extend(iter_jsonl(cfg.crawl_articles_raw))

    docs_by_key: dict[str, dict[str, Any]] = {}
    for raw in raw_docs:
        doc = normalize_doc(raw)
        if not doc:
            continue
        if cfg.skip_expired_docs and "het hieu luc" in vi_fold(doc.get("status")):
            continue
        key = doc.get("doc_id") or doc_key(doc.get("legal_id", ""), doc.get("title", ""))
        docs_by_key[key] = doc

    seen_articles = set()
    article_count = 0
    skipped_dup = 0
    cfg.normalized_docs.parent.mkdir(parents=True, exist_ok=True)
    with cfg.normalized_docs.open("w", encoding="utf-8") as f:
        for doc in sorted(docs_by_key.values(), key=lambda x: (x.get("legal_id", ""), x.get("title", ""))):
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with cfg.normalized_articles.open("w", encoding="utf-8") as f:
        for raw in raw_articles:
            row = normalize_article(raw, docs_by_key)
            if not row:
                continue
            if cfg.skip_expired_docs and "het hieu luc" in vi_fold(row.get("status")):
                continue
            key = article_key(row)
            text_hash = sha1_text(row.get("text", ""))
            dedupe_key = key or text_hash
            if dedupe_key in seen_articles or text_hash in seen_articles:
                skipped_dup += 1
                continue
            seen_articles.add(dedupe_key)
            seen_articles.add(text_hash)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            article_count += 1

    report = {
        "docs": len(docs_by_key),
        "articles": article_count,
        "skipped_dup": skipped_dup,
        "normalized_docs": str(cfg.normalized_docs),
        "normalized_articles": str(cfg.normalized_articles),
    }
    write_json(cfg.report_dir / "02_normalize_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


# =============================================================================
# 4) CHUNKING
# =============================================================================


def split_paragraphs(text: str) -> list[str]:
    raw = re.split(r"(?:\n+|(?<=\.)\s+(?=\d+\.|[a-z]\)))", text)
    return [repair_text(x) for x in raw if repair_text(x)]


def chunk_text(text: str, target_chars: int, max_chars: int, overlap_paragraphs: int) -> list[str]:
    text = repair_text(text)
    if len(text) <= max_chars:
        return [text]
    paragraphs = split_paragraphs(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paragraphs:
        add_len = len(p) + 1
        if cur and cur_len + add_len > target_chars:
            chunks.append(repair_text(" ".join(cur)))
            cur = cur[-overlap_paragraphs:] if overlap_paragraphs else []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(p)
        cur_len += add_len
        if cur_len >= max_chars:
            chunks.append(repair_text(" ".join(cur)))
            cur = cur[-overlap_paragraphs:] if overlap_paragraphs else []
            cur_len = sum(len(x) + 1 for x in cur)
    if cur:
        chunks.append(repair_text(" ".join(cur)))
    return chunks


def stage_chunk(cfg: PipelineConfig) -> None:
    ensure_dirs(cfg)
    if cfg.chunk_file.exists():
        print("chunk skipped, exists:", cfg.chunk_file)
        return

    rows = 0
    skipped_short = 0
    seen_chunk_hash = set()
    with cfg.chunk_file.open("w", encoding="utf-8") as out:
        for art in iter_jsonl(cfg.normalized_articles, strict=True):
            pieces = chunk_text(
                art["text"],
                target_chars=cfg.chunk_target_chars,
                max_chars=cfg.chunk_max_chars,
                overlap_paragraphs=cfg.chunk_overlap_paragraphs,
            )
            for idx, piece in enumerate(pieces):
                if len(piece) < cfg.min_chunk_chars:
                    skipped_short += 1
                    continue
                h = sha1_text(piece)
                if h in seen_chunk_hash:
                    continue
                seen_chunk_hash.add(h)
                cid = f"vbpl:{doc_key(art.get('legal_id',''), art.get('title',''))}:article:{art.get('article_no')}:chunk:{idx}"
                obj = {
                    "chunk_id": cid,
                    "source": art.get("source", "vbpl"),
                    "doc_id": art.get("doc_id"),
                    "legal_id": art.get("legal_id"),
                    "doc_type": art.get("doc_type"),
                    "title": art.get("title"),
                    "article_no": art.get("article_no"),
                    "article_label": article_label(art.get("article_no")),
                    "article_title": art.get("article_title"),
                    "url": art.get("url"),
                    "text": piece,
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                rows += 1
                if rows % 50000 == 0:
                    print("chunks:", rows)
    report = {"chunks": rows, "skipped_short": skipped_short, "chunk_file": str(cfg.chunk_file)}
    write_json(cfg.report_dir / "03_chunk_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


# =============================================================================
# 5) BM25 SQLITE FTS5
# =============================================================================


def stage_build_bm25(cfg: PipelineConfig) -> None:
    ensure_dirs(cfg)
    if cfg.bm25_db.exists():
        print("bm25 skipped, exists:", cfg.bm25_db)
        return
    local_db = cfg.work_dir / cfg.bm25_db.name
    if local_db.exists():
        local_db.unlink()
    conn = sqlite3.connect(str(local_db))
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute(
        """
        CREATE TABLE chunks (
          rowid INTEGER PRIMARY KEY,
          chunk_id TEXT,
          source TEXT,
          doc_id TEXT,
          legal_id TEXT,
          doc_type TEXT,
          title TEXT,
          article_no TEXT,
          article_label TEXT,
          article_title TEXT,
          url TEXT,
          text TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
          title_fts,
          legal_fts,
          article_fts,
          body_fts,
          content='chunks',
          content_rowid='rowid'
        )
        """
    )
    batch = []
    fts_batch = []
    rows = 0
    for obj in iter_jsonl(cfg.chunk_file, strict=True):
        rows += 1
        batch.append(
            (
                rows,
                obj.get("chunk_id"),
                obj.get("source"),
                obj.get("doc_id"),
                obj.get("legal_id"),
                obj.get("doc_type"),
                obj.get("title"),
                obj.get("article_no"),
                obj.get("article_label"),
                obj.get("article_title"),
                obj.get("url"),
                obj.get("text"),
            )
        )
        fts_batch.append(
            (
                rows,
                obj.get("title", ""),
                obj.get("legal_id", ""),
                f"{obj.get('article_label','')} {obj.get('article_title','')}",
                obj.get("text", ""),
            )
        )
        if len(batch) >= 5000:
            cur.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            cur.executemany("INSERT INTO chunks_fts(rowid,title_fts,legal_fts,article_fts,body_fts) VALUES (?,?,?,?,?)", fts_batch)
            conn.commit()
            batch.clear()
            fts_batch.clear()
            print("bm25 rows:", rows)
    if batch:
        cur.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        cur.executemany("INSERT INTO chunks_fts(rowid,title_fts,legal_fts,article_fts,body_fts) VALUES (?,?,?,?,?)", fts_batch)
    conn.commit()
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
    conn.commit()
    conn.close()
    shutil.copy2(local_db, cfg.bm25_db)
    report = {"rows": rows, "bm25_db": str(cfg.bm25_db)}
    write_json(cfg.report_dir / "04_bm25_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


# =============================================================================
# 6) DENSE FAISS
# =============================================================================


def load_sentence_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    device = "cuda" if _torch_cuda_available() else "cpu"
    return SentenceTransformer(model_name, device=device, trust_remote_code=True)


def _torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def stage_build_faiss(cfg: PipelineConfig) -> None:
    ensure_dirs(cfg)
    if cfg.faiss_index.exists() and cfg.faiss_meta.exists():
        print("faiss skipped, exists:", cfg.faiss_index)
        return
    import faiss
    import numpy as np

    shard_dir = cfg.dense_dir / "embed_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # First pass: make embedding shards. Resume by counting rows already
    # materialized in shard metadata files, then skipping that many chunks.
    model = load_sentence_model(cfg.embed_model)
    buffer_text: list[str] = []
    buffer_meta: list[dict[str, Any]] = []
    existing_shards = sorted(shard_dir.glob("shard_*.npy"))
    existing_meta = sorted(shard_dir.glob("shard_*.jsonl"))
    existing_rows = sum(1 for p in existing_meta for _ in iter_jsonl(p, strict=True))
    shard_id = (max(int(p.stem.split("_")[1]) for p in existing_shards) + 1) if existing_shards else 0
    total = 0

    def flush_shard() -> None:
        nonlocal shard_id, total, buffer_text, buffer_meta
        if not buffer_text:
            return
        npy = shard_dir / f"shard_{shard_id:05d}.npy"
        meta = shard_dir / f"shard_{shard_id:05d}.jsonl"
        if npy.exists() and meta.exists():
            print("skip existing shard", shard_id)
        else:
            embs = model.encode(
                buffer_text,
                batch_size=cfg.embed_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ).astype("float32")
            np.save(npy, embs)
            with meta.open("w", encoding="utf-8") as f:
                for m in buffer_meta:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            print("saved shard", shard_id, "rows", len(buffer_text))
        total += len(buffer_text)
        shard_id += 1
        buffer_text = []
        buffer_meta = []

    if existing_rows:
        print("existing embed shard rows:", existing_rows, "next shard:", shard_id)

    for row_no, obj in enumerate(iter_jsonl(cfg.chunk_file, strict=True), start=1):
        if row_no <= existing_rows:
            continue
        text = "\n".join(
            [
                norm_space(obj.get("title")),
                norm_space(obj.get("article_label")),
                norm_space(obj.get("article_title")),
                norm_space(obj.get("text")),
            ]
        )
        buffer_text.append(text)
        buffer_meta.append(obj)
        if len(buffer_text) >= cfg.embed_shard_size:
            flush_shard()
    flush_shard()

    del model
    gc.collect()

    # Merge shards into FAISS.
    index = None
    meta_out = cfg.faiss_meta.open("w", encoding="utf-8")
    vectors = 0
    for npy in sorted(shard_dir.glob("shard_*.npy")):
        arr = np.load(npy).astype("float32")
        if index is None:
            index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        meta_path = npy.with_suffix(".jsonl")
        for m in iter_jsonl(meta_path, strict=True):
            meta_out.write(json.dumps(m, ensure_ascii=False) + "\n")
        vectors += arr.shape[0]
        print("merged", npy.name, "total", vectors)
    meta_out.close()
    if index is None:
        raise RuntimeError("No dense shards found.")
    faiss.write_index(index, str(cfg.faiss_index))
    report = {"vectors": vectors, "faiss_index": str(cfg.faiss_index), "faiss_meta": str(cfg.faiss_meta)}
    write_json(cfg.report_dir / "05_faiss_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


# =============================================================================
# 7) QUERY EXPANSION / HYDE
# =============================================================================


def make_hyde_prompt(question: str) -> str:
    return (
        "Bạn là trợ lý truy hồi văn bản pháp luật Việt Nam.\n"
        "Viết JSON hợp lệ gồm query_expansion và hyde_answer.\n"
        "query_expansion: 5-10 cụm từ khóa pháp lý liên quan.\n"
        "hyde_answer: đoạn trả lời giả định 2-4 câu để tìm điều luật.\n"
        "Không bịa số điều, số văn bản, mức tiền hoặc thời hạn nếu không chắc.\n"
        "Chỉ trả JSON, không markdown.\n\n"
        f"Câu hỏi: {question}"
    )


def parse_json_object(text: str) -> dict[str, Any]:
    s = norm_space(text)
    s = re.sub(r"^```json\s*|\s*```$", "", s, flags=re.I)
    l, r = s.find("{"), s.rfind("}")
    if l >= 0 and r > l:
        try:
            return json.loads(s[l : r + 1])
        except Exception:
            return {}
    return {}


def stage_query_hyde(cfg: PipelineConfig) -> None:
    ensure_dirs(cfg)
    questions = load_questions(cfg.question_file)
    done = load_seen_ids(cfg.hyde_file)
    pending = [q for q in questions if q["id"] not in done]
    print("hyde done:", len(done), "pending:", len(pending))
    if not pending:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.hyde_model, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hyde_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    t0 = time.time()
    for start in range(0, len(pending), cfg.hyde_batch_size):
        batch = pending[start : start + cfg.hyde_batch_size]
        prompts = []
        for q in batch:
            messages = [
                {"role": "system", "content": "Bạn chỉ trả JSON hợp lệ bằng tiếng Việt."},
                {"role": "user", "content": make_hyde_prompt(q["question"])},
            ]
            prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=cfg.hyde_max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for q, full in zip(batch, out):
            raw = tok.decode(full[input_len:], skip_special_tokens=True)
            obj = parse_json_object(raw)
            rec = {
                "id": q["id"],
                "question": q["question"],
                "query_expansion": obj.get("query_expansion") or "",
                "hyde_answer": obj.get("hyde_answer") or norm_space(raw)[:800],
            }
            append_jsonl(cfg.hyde_file, rec)
        if (start // cfg.hyde_batch_size + 1) % 10 == 0:
            print(f"hyde {min(start + cfg.hyde_batch_size, len(pending))}/{len(pending)} elapsed={(time.time()-t0)/60:.1f}m")
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# 8) HYBRID RETRIEVAL
# =============================================================================


STOPWORDS = {
    "và",
    "hoặc",
    "thì",
    "là",
    "của",
    "cho",
    "với",
    "trong",
    "khi",
    "nếu",
    "theo",
    "được",
    "bị",
    "có",
    "không",
    "như",
    "nào",
    "gì",
    "về",
    "cần",
    "phải",
    "sẽ",
    "một",
    "các",
    "những",
    "để",
    "tại",
    "từ",
    "đến",
    "ra",
    "sao",
    "này",
    "đó",
}


def vi_terms(text: str, limit: int = 32) -> list[str]:
    toks = re.findall(r"[0-9A-Za-zÀ-ỹĐđ/\.-]+", norm_space(text))
    legal_ids = LEGAL_ID_RE.findall(text) + LEGAL_ID_SHORT_RE.findall(text)
    out = []
    seen = set()
    for t in legal_ids + toks:
        low = t.lower().strip(".,;:/")
        if len(low) < 2 or low in STOPWORDS:
            continue
        if low not in seen:
            seen.add(low)
            out.append(t)
    return out[:limit]


def sqlite_quote_token(t: str) -> str:
    return '"' + t.replace('"', '""') + '"'


def make_fts_query(text: str, limit: int = 32) -> str:
    terms = vi_terms(text, limit=limit)
    return " OR ".join(sqlite_quote_token(t) for t in terms)


def sqlite_row_to_candidate(row: dict[str, Any], rank: int, source: str, score: float) -> dict[str, Any]:
    article_no = norm_space(row.get("article_no"))
    base = {
        "chunk_id": row.get("chunk_id"),
        "source": row.get("source") or "vbpl",
        "doc_id": row.get("doc_id"),
        "legal_id": row.get("legal_id") or extract_legal_id(row.get("title")),
        "doc_type": row.get("doc_type"),
        "title": row.get("title"),
        "article_no": article_no,
        "article_label": article_label(article_no),
        "article_title": row.get("article_title"),
        "url": row.get("url"),
        "text": row.get("text"),
    }
    base["doc"] = doc_citation(base)
    base["citation"] = article_citation(base)
    base["retrieval_source"] = source
    base["retrieval_rank"] = rank
    base["retrieval_score"] = float(score)
    return base


def bm25_search(conn: sqlite3.Connection, text: str, topk: int) -> list[dict[str, Any]]:
    match_q = make_fts_query(text)
    if not match_q:
        return []
    sql = """
    SELECT chunks.*, bm25(chunks_fts) AS bm25_score
    FROM chunks_fts
    JOIN chunks ON chunks.rowid = chunks_fts.rowid
    WHERE chunks_fts MATCH ?
    ORDER BY bm25(chunks_fts)
    LIMIT ?
    """
    try:
        rows = [dict(r) for r in conn.execute(sql, (match_q, topk)).fetchall()]
    except sqlite3.OperationalError:
        short = make_fts_query(text, limit=14)
        rows = [dict(r) for r in conn.execute(sql, (short, topk)).fetchall()] if short else []
    return [sqlite_row_to_candidate(r, i, "bm25", -float(r.get("bm25_score") or 0)) for i, r in enumerate(rows, start=1)]


def load_meta_list(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, strict=True))


def rrf_merge(lists: list[tuple[str, list[dict[str, Any]], float]], k: int, topk: int, question: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source_name, items, weight in lists:
        for rank, item in enumerate(items, start=1):
            cit = item.get("citation") or article_citation(item)
            if not cit:
                continue
            add = weight / (k + rank)
            cur = merged.get(cit)
            if not cur:
                cur = {**item, "citation": cit, "doc": item.get("doc") or doc_citation(item), "rrf_score": 0.0, "sources": []}
                merged[cit] = cur
            cur["rrf_score"] += add
            cur["sources"].append({"source": source_name, "rank": rank, "score": item.get("retrieval_score")})

    for item in merged.values():
        bonus = 0.0
        bonus += doc_type_bonus(item.get("doc_type", ""), item.get("title", ""))
        if looks_local(item.get("title", "")) and not question_allows_local(question):
            bonus -= 0.08
        if boilerplate_article(item.get("article_title", "")) and not question_needs_boilerplate(question):
            bonus -= 0.03
        if not extract_legal_id(item.get("title", "")) and not item.get("legal_id"):
            bonus -= 0.05
        item["hybrid_score"] = item["rrf_score"] + bonus
    out = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)
    return out[:topk]


def stage_retrieve(cfg: PipelineConfig) -> None:
    ensure_dirs(cfg)
    questions = load_questions(cfg.question_file)
    done = load_seen_ids(cfg.candidates_file)
    pending = [q for q in questions if q["id"] not in done]
    print("retrieval done:", len(done), "pending:", len(pending))
    if not pending:
        return

    import faiss
    import numpy as np

    hyde_by_id = {int(x["id"]): x for x in iter_jsonl(cfg.hyde_file)}
    meta = load_meta_list(cfg.faiss_meta)
    index = faiss.read_index(str(cfg.faiss_index))
    embedder = load_sentence_model(cfg.embed_model)

    # Encode pending query texts in one pass.
    query_texts = []
    for q in pending:
        h = hyde_by_id.get(q["id"], {})
        exp = h.get("query_expansion", "")
        hyde = h.get("hyde_answer", "")
        query_texts.append(norm_space(f"{q['question']}\n{exp}\n{hyde}"))
    qvecs = embedder.encode(
        query_texts,
        batch_size=cfg.embed_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    conn = sqlite3.connect(str(cfg.bm25_db))
    conn.row_factory = sqlite3.Row
    t0 = time.time()
    for i, q in enumerate(pending):
        h = hyde_by_id.get(q["id"], {})
        search_text = norm_space(f"{q['question']}\n{h.get('query_expansion','')}\n{h.get('hyde_answer','')}")
        bm25_items = bm25_search(conn, search_text, cfg.bm25_topk)

        distances, ids = index.search(qvecs[i : i + 1], cfg.dense_topk)
        dense_items = []
        for rank, (idx, dist) in enumerate(zip(ids[0], distances[0]), start=1):
            if idx < 0:
                continue
            item = {**meta[int(idx)]}
            item["doc"] = doc_citation(item)
            item["citation"] = article_citation(item)
            item["retrieval_source"] = "dense"
            item["retrieval_rank"] = rank
            item["retrieval_score"] = float(dist)
            dense_items.append(item)

        candidates = rrf_merge(
            [("bm25", bm25_items, 1.0), ("dense", dense_items, 1.0)],
            cfg.rrf_k,
            cfg.final_candidate_topk,
            q["question"],
        )
        append_jsonl(
            cfg.candidates_file,
            {"id": q["id"], "question": q["question"], "query_text": search_text, "candidates": candidates},
        )
        if (i + 1) % 100 == 0:
            print(f"retrieve {i+1}/{len(pending)} elapsed={(time.time()-t0)/60:.1f}m")
    conn.close()
    del embedder, index, meta
    gc.collect()


# =============================================================================
# 9) RERANK
# =============================================================================


def stage_rerank(cfg: PipelineConfig) -> None:
    ensure_dirs(cfg)
    questions = load_questions(cfg.question_file)
    cand_by_id = {int(x["id"]): x for x in iter_jsonl(cfg.candidates_file)}
    done = load_seen_ids(cfg.rerank_file)
    pending = [q for q in questions if q["id"] in cand_by_id and q["id"] not in done]
    print("rerank done:", len(done), "pending:", len(pending))
    if not pending:
        return
    from sentence_transformers import CrossEncoder

    device = "cuda" if _torch_cuda_available() else "cpu"
    reranker = CrossEncoder(cfg.reranker_model, device=device)
    t0 = time.time()
    for idx, q in enumerate(pending, start=1):
        cands = cand_by_id[q["id"]].get("candidates", [])
        pairs = []
        for c in cands:
            text = "\n".join(
                [
                    norm_space(c.get("title")),
                    norm_space(c.get("article_label")),
                    norm_space(c.get("article_title")),
                    norm_space(c.get("text")),
                ]
            )[: cfg.rerank_max_pair_chars]
            pairs.append([q["question"], text])
        scores = reranker.predict(pairs, batch_size=cfg.rerank_batch_size, show_progress_bar=False) if pairs else []
        ranked = []
        for c, score in zip(cands, scores):
            bonus = doc_type_bonus(c.get("doc_type", ""), c.get("title", ""))
            if looks_local(c.get("title", "")) and not question_allows_local(q["question"]):
                bonus -= 0.18
            if boilerplate_article(c.get("article_title", "")) and not question_needs_boilerplate(q["question"]):
                bonus -= 0.08
            c = {**c, "rerank_score": float(score), "final_score": float(score) + bonus}
            ranked.append(c)
        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        append_jsonl(cfg.rerank_file, {"id": q["id"], "question": q["question"], "candidates": ranked})
        if idx % 100 == 0:
            print(f"rerank {idx}/{len(pending)} elapsed={(time.time()-t0)/60:.1f}m")
    del reranker
    gc.collect()


# =============================================================================
# 10) ANSWER + EVIDENCE SELECTOR
# =============================================================================


def make_answer_prompt(question: str, context: str) -> str:
    return f"""
Bạn là trợ lý pháp luật Việt Nam.

Nhiệm vụ:
1. Trả lời câu hỏi chỉ dựa trên CONTEXT.
2. Chọn các mục A1..A8 thật sự được dùng làm căn cứ trực tiếp cho câu trả lời.
3. Loại bỏ mục chỉ cùng chủ đề, mục địa phương nếu câu hỏi không hỏi địa phương, mục hiệu lực/trách nhiệm thi hành nếu không trực tiếp trả lời.
4. Thường chỉ chọn 1-4 mục.
5. Chỉ trả JSON hợp lệ, không markdown, không dùng tiếng Trung.

Schema:
{{"answer":"...", "used_articles":["A1","A3"]}}

CÂU HỎI:
{question}

CONTEXT:
{context}
""".strip()


def make_answer_context(cands: list[dict[str, Any]], cfg: PipelineConfig) -> tuple[list[dict[str, Any]], str]:
    items = []
    seen = set()
    for c in cands:
        cit = c.get("citation") or article_citation(c)
        if not cit or cit in seen:
            continue
        seen.add(cit)
        label = f"A{len(items)+1}"
        item = {**c, "label": label, "citation": cit, "doc": c.get("doc") or doc_citation(c)}
        items.append(item)
        if len(items) >= cfg.answer_context_articles:
            break
    blocks = []
    for it in items:
        blocks.append(
            f"[{it['label']}]\n"
            f"CITATION: {it['citation']}\n"
            f"TÊN VĂN BẢN: {it.get('title','')}\n"
            f"TÊN ĐIỀU: {it.get('article_label','')} {it.get('article_title','')}\n"
            f"NỘI DUNG: {norm_space(it.get('text',''))[:cfg.answer_article_chars]}"
        )
    return items, "\n\n".join(blocks)


def stage_answer_evidence(cfg: PipelineConfig) -> None:
    ensure_dirs(cfg)
    questions = load_questions(cfg.question_file)
    rerank_by_id = {int(x["id"]): x for x in iter_jsonl(cfg.rerank_file)}
    done = load_seen_ids(cfg.evidence_file)
    pending = [q for q in questions if q["id"] in rerank_by_id and q["id"] not in done]
    print("evidence done:", len(done), "pending:", len(pending))
    if not pending:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.answer_model, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.answer_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    t0 = time.time()
    for start in range(0, len(pending), cfg.answer_batch_size):
        batch = pending[start : start + cfg.answer_batch_size]
        prompts = []
        metas = []
        for q in batch:
            cands = rerank_by_id[q["id"]].get("candidates", [])
            items, ctx = make_answer_context(cands, cfg)
            messages = [
                {"role": "system", "content": "Bạn chỉ trả JSON hợp lệ bằng tiếng Việt."},
                {"role": "user", "content": make_answer_prompt(q["question"], ctx)},
            ]
            prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            metas.append((q, items))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=cfg.answer_max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for full, (q, items) in zip(out, metas):
            raw = tok.decode(full[input_len:], skip_special_tokens=True)
            obj = parse_json_object(raw)
            answer = repair_text(obj.get("answer") or "")
            if not answer or CJK_RE.search(answer):
                answer = ""
            label_map = {x["label"].upper(): x for x in items}
            labels = []
            for x in obj.get("used_articles") or []:
                lab = norm_space(x).strip("[]").upper()
                if lab.isdigit():
                    lab = "A" + lab
                if lab in label_map and lab not in labels:
                    labels.append(lab)
            if not labels and items:
                labels = [items[0]["label"].upper()]
            used = [label_map[x] for x in labels if x in label_map]
            append_jsonl(
                cfg.evidence_file,
                {
                    "id": q["id"],
                    "question": q["question"],
                    "answer": answer,
                    "used_labels": labels,
                    "used_articles": [x["citation"] for x in used],
                    "used_docs": list(dict.fromkeys(x["doc"] for x in used)),
                    "raw": raw[:2000],
                },
            )
        if (start + len(batch)) % 50 == 0:
            print(f"evidence {start+len(batch)}/{len(pending)} elapsed={(time.time()-t0)/60:.1f}m")
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# 11) SUBMISSION
# =============================================================================


def select_from_rerank(row: dict[str, Any], cfg_variant: dict[str, int]) -> tuple[list[str], list[str]]:
    arts = []
    docs = []
    seen_a = set()
    seen_d = set()
    per_doc = Counter()
    for c in row.get("candidates", []):
        cit = c.get("citation") or article_citation(c)
        doc = c.get("doc") or doc_citation(c)
        if not cit or cit in seen_a:
            continue
        if doc not in seen_d and len(seen_d) >= cfg_variant["max_docs"]:
            continue
        if per_doc[doc] >= cfg_variant["max_per_doc"]:
            continue
        seen_a.add(cit)
        arts.append(cit)
        if doc not in seen_d:
            seen_d.add(doc)
            docs.append(doc)
        per_doc[doc] += 1
        if len(arts) >= cfg_variant["max_articles"]:
            break
    return docs, arts


def stage_build_submission(cfg: PipelineConfig) -> None:
    ensure_dirs(cfg)
    questions = load_questions(cfg.question_file)
    rerank_by_id = {int(x["id"]): x for x in iter_jsonl(cfg.rerank_file)}
    evidence_by_id = {int(x["id"]): x for x in iter_jsonl(cfg.evidence_file)}
    summary = {}
    for name, vcfg in cfg.submission_variants.items():
        rows = []
        for q in questions:
            ev = evidence_by_id.get(q["id"], {})
            answer = repair_text(ev.get("answer") or "")
            used_arts = [norm_space(x) for x in ev.get("used_articles", []) if norm_space(x)]
            used_docs = [norm_space(x) for x in ev.get("used_docs", []) if norm_space(x)]
            if not used_arts:
                used_docs, used_arts = select_from_rerank(rerank_by_id.get(q["id"], {}), vcfg)
            else:
                # apply variant caps even when LLM selected evidence.
                capped_arts = []
                capped_docs = []
                seen_d = set()
                per_doc = Counter()
                for cit in used_arts:
                    doc = "|".join(cit.split("|")[:2])
                    if doc not in seen_d and len(seen_d) >= vcfg["max_docs"]:
                        continue
                    if per_doc[doc] >= vcfg["max_per_doc"]:
                        continue
                    capped_arts.append(cit)
                    if doc not in seen_d:
                        seen_d.add(doc)
                        capped_docs.append(doc)
                    per_doc[doc] += 1
                    if len(capped_arts) >= vcfg["max_articles"]:
                        break
                used_arts, used_docs = capped_arts, capped_docs
            rows.append(
                {
                    "id": int(q["id"]),
                    "question": q["question"],
                    "answer": answer,
                    "relevant_docs": used_docs,
                    "relevant_articles": used_arts,
                }
            )
        out_json = cfg.submission_dir / name / "results.json"
        out_zip = cfg.submission_dir / f"submission_{name}.zip"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(out_json, arcname="results.json")
        ac = [len(r["relevant_articles"]) for r in rows]
        dc = [len(r["relevant_docs"]) for r in rows]
        bad_answer_chars = sum(1 for r in rows if CJK_RE.search(r.get("answer", "")))
        summary[name] = {
            "json": str(out_json),
            "zip": str(out_zip),
            "article_count_min_avg_max": [min(ac), round(sum(ac) / len(ac), 3), max(ac)],
            "doc_count_min_avg_max": [min(dc), round(sum(dc) / len(dc), 3), max(dc)],
            "bad_answer_chars": bad_answer_chars,
        }
    write_json(cfg.report_dir / "10_submission_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# =============================================================================
# 12) MAIN
# =============================================================================


def main(cfg: PipelineConfig = CFG) -> None:
    print("BASE_DIR:", cfg.base_dir)
    print("OUT_DIR:", cfg.out_dir)
    print("USE_KEPT_ARTIFACTS:", cfg.use_kept_artifacts)
    print("RUN_STAGES:", json.dumps(RUN_STAGES, ensure_ascii=False))

    if RUN_STAGES["install"]:
        install_deps()
    if RUN_STAGES["mount_drive"]:
        mount_drive_if_colab()

    ensure_dirs(cfg)
    validate_kept_artifacts(cfg)

    if RUN_STAGES["crawl_or_ingest"]:
        stage_crawl_or_ingest(cfg)
    if RUN_STAGES["normalize"]:
        stage_normalize(cfg)
    if RUN_STAGES["chunk"]:
        stage_chunk(cfg)
    if RUN_STAGES["build_bm25"]:
        stage_build_bm25(cfg)
    if RUN_STAGES["build_faiss"]:
        stage_build_faiss(cfg)
    if RUN_STAGES["query_hyde"]:
        stage_query_hyde(cfg)
    if RUN_STAGES["retrieve"]:
        stage_retrieve(cfg)
    if RUN_STAGES["rerank"]:
        stage_rerank(cfg)
    if RUN_STAGES["answer_evidence"]:
        stage_answer_evidence(cfg)
    if RUN_STAGES["build_submission"]:
        stage_build_submission(cfg)


if __name__ == "__main__":
    main()
