"""Download dataset tiers into rag/data (see brief section 7).

  python scripts/download_data.py --tier a        # Enginuity Task 2 (~6.6 MB)
  python scripts/download_data.py --tier b-small  # data-7.pdf only (1.7 MB) for routing practice
  python scripts/download_data.py --tier b        # all 10 Task 1 PDFs (~122 MB) + human-validated set
  python scripts/download_data.py --tier vidore   # ViDoRe v3 Industrial labels + 2 checklist PDFs (text gold)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DATA = Path(__file__).resolve().parent.parent / "data"
REPO = "enginuity2025/enginuity-bench"

TIERS = {
    "a": ["task-2-question-answering/*"],
    "b-small": ["task-1-component-identification/pdfs/data-7.pdf"],
    "b": ["task-1-component-identification/pdfs/*", "task-1-component-identification/human-validation-set/*"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=sorted(TIERS) + ["vidore"], default="a")
    a = ap.parse_args()
    if a.tier == "vidore":
        path = snapshot_download(
            "vidore/vidore_v3_industrial", repo_type="dataset",
            revision="233d20721f4deb392a09a86cca01761adbc91157", local_dir=str(DATA / "vidore"),
            allow_patterns=["queries/*", "qrels/*", "documents_metadata/*", "pdfs/metadata.csv",
                            "pdfs/TO_00-25-172CL-1.pdf", "pdfs/TO_00-25-172CL-3.pdf"])
        print(f"downloaded ViDoRe labels + PDFs -> {path}")
        return 0
    path = snapshot_download(REPO, repo_type="dataset", allow_patterns=TIERS[a.tier],
                             local_dir=str(DATA / "enginuity"))
    print(f"downloaded tier {a.tier} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
