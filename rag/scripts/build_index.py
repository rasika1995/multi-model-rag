"""Run the indexing pipeline: ingest > enrich > embed > upsert (clean rebuild).

Usage (from rag/):  python scripts/build_index.py
Settings come from rag/.env (CORPORA, PDF_PATHS, EMBED_MODEL, VECTOR_BACKEND, ...).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, settings  # noqa: E402
from app.openrouter import UpstreamError  # noqa: E402
from app.pipeline import build_index  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    s = settings()
    try:
        s.require_key()
        build_index(s)
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}")
        return 2
    except UpstreamError as e:
        print(f"MODEL ERROR: {e}\nRun scripts/check_models.py and change the slug in rag/.env.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
