from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import full_colab_pipeline as pipeline  # noqa: E402


def main() -> None:
    pipeline.CFG.use_kept_artifacts = True
    pipeline.RUN_STAGES.update(
        {
            "install": False,
            "mount_drive": True,
            "crawl_or_ingest": False,
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
    )
    pipeline.main(pipeline.CFG)


if __name__ == "__main__":
    main()

