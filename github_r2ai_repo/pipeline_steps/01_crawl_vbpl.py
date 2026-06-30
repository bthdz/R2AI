from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(os.environ.get("R2AI_BASE_DIR", os.environ.get("R2AI_DATA_DIR", "/content/drive/MyDrive/R2AI/Law")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VBPL crawler entrypoint.")
    parser.add_argument(
        "--csv",
        default=str(BASE_DIR / "vbpl_excel_exports"),
        help="CSV/XLSX file or folder exported from VBPL.",
    )
    parser.add_argument(
        "--out",
        default=str(BASE_DIR / "raw" / "vbpl_crawled"),
        help="Output crawl folder.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start row.")
    parser.add_argument("--limit", type=int, default=None, help="Optional crawl limit.")
    parser.add_argument("--workers", type=int, default=1, help="Crawler worker count.")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "run.py"),
        "--csv",
        args.csv,
        "--out",
        args.out,
        "--start",
        str(args.start),
        "--workers",
        str(args.workers),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    print(" ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
