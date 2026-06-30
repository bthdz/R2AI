from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import full_colab_pipeline as pipeline  # noqa: E402


def main() -> None:
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
    pipeline.main(pipeline.CFG)


if __name__ == "__main__":
    main()

