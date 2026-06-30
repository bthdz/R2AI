from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import full_colab_pipeline as pipeline  # noqa: E402


BEST_ARTIFACTS = {
    "chunks": "processed/chunks/all_chunks_safe_20260622_012757_repaired_min80_plus_vbpl100_20260626_154852.jsonl",
    "bm25": "processed/index/bm25_fts5_legal_plus_vbpl100_20260626_154852.sqlite",
    "faiss_index": "processed/index/qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216.index",
    "faiss_meta": "processed/index/qwen3_8b_dim1024_merged_plus_vbpl100_20260626_163216_meta.jsonl",
    "hyde": "processed/query_plans/r2ai_stage1_hyde_qwen25_15b_v1.jsonl",
    "rule_plan": "processed/query_plans/r2ai_stage1_query_plans_hybrid_rule_llm_planner_v1_utf8fixed.jsonl",
    "retrieval": "processed/retrieval/r2ai_stage1_hybrid_candidates_bm25plus_vbpl100_qwen3plus_20260626_170709.jsonl",
    "graph": "processed/graphs/vbpl_relation_graph_exact_v2.jsonl",
    "shards": "processed/index/qwen3_8b_dim1024_sharded_v1",
}

LEGAL_ID_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)*\b", re.I)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path: Path, with_hash: bool = False) -> dict:
    info = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return info
    if path.is_file():
        info["bytes"] = path.stat().st_size
        if with_hash:
            info["sha256"] = sha256_file(path)
    else:
        files = [p for p in path.rglob("*") if p.is_file()]
        info["files"] = len(files)
        info["bytes"] = sum(p.stat().st_size for p in files)
    return info


def copy_file(src: Path, dst: Path, overwrite: bool) -> dict:
    if not src.exists():
        return {"src": str(src), "dst": str(dst), "copied": False, "reason": "missing source"}
    if dst.exists() and not overwrite:
        return {"src": str(src), "dst": str(dst), "copied": False, "reason": "destination exists"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"src": str(src), "dst": str(dst), "copied": True}


def copy_dir(src: Path, dst: Path, overwrite: bool) -> dict:
    if not src.exists():
        return {"src": str(src), "dst": str(dst), "copied": False, "reason": "missing source"}
    if dst.exists():
        if not overwrite:
            return {"src": str(src), "dst": str(dst), "copied": False, "reason": "destination exists"}
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return {"src": str(src), "dst": str(dst), "copied": True}


def norm_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("questions", data) if isinstance(data, dict) else data
    out = []
    for item in rows:
        qid = item.get("id", item.get("question_id", item.get("qid")))
        question = item.get("question", item.get("query", item.get("text", "")))
        if qid is not None and question:
            out.append({"id": int(qid), "question": norm_space(question)})
    return out


def terms(text: str, limit: int = 40) -> list[str]:
    raw = re.sub(r"[^0-9A-Za-zÀ-ỹ%/.-]+", " ", text).split()
    seen = set()
    out = []
    stop = {"cua", "cho", "voi", "trong", "theo", "phap", "luat", "dieu", "khoan", "quy", "dinh"}
    for token in raw:
        t = token.strip(".,;:").lower()
        if len(t) < 3 or t in stop or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= limit:
            break
    return out


def build_rule_plan_artifact(cfg, dst: Path, overwrite: bool) -> dict:
    if dst.exists() and not overwrite:
        return {"dst": str(dst), "built": False, "reason": "destination exists"}
    hyde_path = cfg.out_dir / "06_query" / "query_hyde.jsonl"
    if not cfg.question_file.exists() or not hyde_path.exists():
        return {"dst": str(dst), "built": False, "reason": "missing questions or hyde source"}

    hyde_by_id = {int(x["id"]): x for x in iter_jsonl(hyde_path) if "id" in x}
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with dst.open("w", encoding="utf-8") as out:
        for q in load_questions(cfg.question_file):
            h = hyde_by_id.get(q["id"], {})
            expansion = norm_space(h.get("query_expansion", ""))
            hyde_answer = norm_space(h.get("hyde_answer", ""))
            plan = {
                "id": q["id"],
                "question": q["question"],
                "method": "hybrid_rule_llm_planner_v1_utf8fixed_rebuilt",
                "queries": [x for x in [q["question"], expansion, hyde_answer] if x],
                "must_terms": terms(q["question"], limit=24),
                "legal_codes": sorted(set(LEGAL_ID_RE.findall(q["question"] + " " + hyde_answer))),
                "hyde_answer": hyde_answer,
            }
            out.write(json.dumps(plan, ensure_ascii=False) + "\n")
            rows += 1
    return {"dst": str(dst), "built": True, "rows": rows}


def build_graph_artifact(cfg, dst: Path, overwrite: bool) -> dict:
    if dst.exists() and not overwrite:
        return {"dst": str(dst), "built": False, "reason": "destination exists"}
    chunk_path = cfg.out_dir / "03_chunks" / "chunks.jsonl"
    if not chunk_path.exists():
        return {"dst": str(dst), "built": False, "reason": "missing chunk source"}

    dst.parent.mkdir(parents=True, exist_ok=True)
    seen_articles = set()
    rows = 0
    with dst.open("w", encoding="utf-8") as out:
        for obj in iter_jsonl(chunk_path):
            legal_id = norm_space(obj.get("legal_id"))
            article_no = norm_space(obj.get("article_no"))
            title = norm_space(obj.get("title"))
            key = (legal_id, article_no, title)
            if legal_id and article_no and key not in seen_articles:
                seen_articles.add(key)
                out.write(
                    json.dumps(
                        {
                            "type": "article_node",
                            "source": obj.get("source"),
                            "doc_id": obj.get("doc_id"),
                            "legal_id": legal_id,
                            "title": title,
                            "article_no": article_no,
                            "article_title": obj.get("article_title"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows += 1
            text = " ".join([title, norm_space(obj.get("article_title")), norm_space(obj.get("text"))])
            for target_code in sorted(set(LEGAL_ID_RE.findall(text))):
                if not legal_id or target_code.upper() == legal_id.upper():
                    continue
                out.write(
                    json.dumps(
                        {
                            "type": "mention_edge",
                            "src_legal_id": legal_id,
                            "src_article_no": article_no,
                            "dst_legal_id": target_code.upper(),
                            "relation": "mentions_legal_id",
                            "src_chunk_id": obj.get("chunk_id"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows += 1
    return {"dst": str(dst), "built": True, "rows": rows}


def set_full_rebuild_stages() -> None:
    pipeline.CFG.use_kept_artifacts = False
    pipeline.RUN_STAGES.update(
        {
            "install": False,
            "mount_drive": True,
            "crawl_or_ingest": True,
            "normalize": True,
            "chunk": True,
            "build_bm25": True,
            "build_faiss": True,
            "query_hyde": True,
            "retrieve": True,
            "rerank": True,
            "answer_evidence": True,
            "build_submission": True,
        }
    )


def promote_artifacts(args: argparse.Namespace) -> dict:
    cfg = pipeline.CFG
    base_dir = cfg.base_dir

    sources = {
        "chunks": cfg.out_dir / "03_chunks" / "chunks.jsonl",
        "bm25": cfg.out_dir / "04_bm25" / "bm25_fts5.sqlite",
        "faiss_index": cfg.out_dir / "05_dense" / "dense.index",
        "faiss_meta": cfg.out_dir / "05_dense" / "dense_meta.jsonl",
        "hyde": cfg.out_dir / "06_query" / "query_hyde.jsonl",
        "retrieval": cfg.out_dir / "07_retrieval" / "hybrid_candidates.jsonl",
        "shards": cfg.out_dir / "05_dense" / "embed_shards",
    }
    destinations = {name: base_dir / rel for name, rel in BEST_ARTIFACTS.items()}

    actions = {}
    for name in ["chunks", "bm25", "faiss_index", "faiss_meta", "hyde", "retrieval"]:
        actions[name] = copy_file(sources[name], destinations[name], overwrite=args.overwrite)

    if args.rebuild_support_artifacts:
        actions["rule_plan"] = build_rule_plan_artifact(cfg, destinations["rule_plan"], overwrite=args.overwrite)
        actions["graph"] = build_graph_artifact(cfg, destinations["graph"], overwrite=args.overwrite)
    else:
        actions["rule_plan"] = {
            "dst": str(destinations["rule_plan"]),
            "built": False,
            "reason": "kept existing shared artifact; pass --rebuild-support-artifacts to rebuild",
        }
        actions["graph"] = {
            "dst": str(destinations["graph"]),
            "built": False,
            "reason": "kept existing shared artifact; pass --rebuild-support-artifacts to rebuild",
        }

    if args.promote_shards:
        actions["shards"] = copy_dir(sources["shards"], destinations["shards"], overwrite=args.overwrite)
    else:
        actions["shards"] = {
            "src": str(sources["shards"]),
            "dst": str(destinations["shards"]),
            "copied": False,
            "reason": "skipped; pass --promote-shards to copy embedding shard folder",
        }

    variant_dir = cfg.submission_dir / args.best_variant
    best_results = variant_dir / "results.json"
    best_zip = cfg.submission_dir / f"submission_{args.best_variant}.zip"
    submit_dir = base_dir / "processed" / "submissions" / "rebuilt_best"
    actions["best_results"] = copy_file(best_results, submit_dir / "results.json", overwrite=args.overwrite)
    if best_zip.exists():
        actions["best_zip"] = copy_file(best_zip, submit_dir / f"submission_{args.best_variant}.zip", overwrite=args.overwrite)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": str(base_dir),
        "run_id": cfg.run_id,
        "out_dir": str(cfg.out_dir),
        "best_variant": args.best_variant,
        "actions": actions,
        "artifacts": {
            name: file_info(path, with_hash=args.hash)
            for name, path in destinations.items()
            if (name != "shards" or args.promote_shards)
        },
        "submission": file_info(submit_dir / "results.json", with_hash=args.hash),
    }
    manifest_path = base_dir / "processed" / "rebuild_best_artifacts_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the full pipeline from raw/crawled data, then promote outputs "
            "to the canonical best-artifact paths used by artifact-first reproduction."
        )
    )
    parser.add_argument("--run-id", default="rebuild_best_artifacts_v1", help="Pipeline run id under processed/full_pipeline.")
    parser.add_argument("--best-variant", default="p5_d3", help="Submission variant to promote as rebuilt_best/results.json.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing canonical artifacts.")
    parser.add_argument("--promote-only", action="store_true", help="Skip pipeline execution and only promote existing run outputs.")
    parser.add_argument("--promote-shards", action="store_true", help="Also copy embedding shards to processed/index/qwen3_8b_dim1024_sharded_v1.")
    parser.add_argument("--rebuild-support-artifacts", action="store_true", help="Also rebuild rule_plan and graph artifacts from rebuilt HyDE/chunks.")
    parser.add_argument("--hash", action="store_true", help="Compute sha256 hashes in manifest. Slow for large artifacts.")
    args = parser.parse_args()

    pipeline.CFG.run_id = args.run_id

    if not args.promote_only:
        set_full_rebuild_stages()
        pipeline.main(pipeline.CFG)

    manifest = promote_artifacts(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nCanonical best artifacts are under:", pipeline.CFG.base_dir / "processed")
    print("Rebuilt best submission:", pipeline.CFG.base_dir / "processed/submissions/rebuilt_best/results.json")


if __name__ == "__main__":
    main()
