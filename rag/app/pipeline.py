"""build_index: ingest > enrich > embed > upsert, then write the index report.

The index report records which embedding model built the index, unit counts
per type, rejected units, failed OCR pages and timings. It feeds Layer C and
the model-mismatch check. Every build resets the collection (clean rebuild).
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import Settings, settings
from .enrich import enrich
from .ingest import load_all, page_class_counts
from .openrouter import ModelClient, get_client, global_calls, reset_global_calls
from .schemas import Unit
from .store import index_dir_size, make_store

log = logging.getLogger(__name__)


def embed_units(units: list[Unit], client: ModelClient, s: Settings) -> list[list[float]]:
    """Search text for everything; figure images directly when EMBED_IMAGES is on."""
    vecs: list[list[float] | None] = [None] * len(units)
    text_idx = [i for i, u in enumerate(units) if not (s.embed_images and u.type == "figure")]
    img_idx = [i for i, u in enumerate(units) if s.embed_images and u.type == "figure"]
    if text_idx:
        for i, v in zip(text_idx, client.embed_texts([units[i].search_text for i in text_idx])):
            vecs[i] = v
    if img_idx:
        for i, v in zip(img_idx, client.embed_images([s.data_dir / units[i].artefact.path for i in img_idx])):
            vecs[i] = v
    return vecs  # type: ignore[return-value]


def build_index(s: Settings | None = None, client_factory: Callable[[], ModelClient] = get_client,
                progress: Callable[[str], None] = print) -> dict:
    s = s or settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    reset_global_calls()
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    store = make_store(s)
    store.ping()

    progress(f"[1/4] ingest corpora={','.join(s.corpora)}")
    t = time.perf_counter()
    loaded = load_all(s, client_factory)
    timings["ingest_s"] = round(time.perf_counter() - t, 2)
    (s.data_dir / "review_queue.json").write_text(json.dumps(loaded.review_queue, indent=2), encoding="utf-8")
    (s.data_dir / "page_routing.json").write_text(json.dumps(loaded.pages, indent=2), encoding="utf-8")
    progress(f"      units={dict(Counter(u.type for u in loaded.units))} pages={page_class_counts(loaded.pages)} "
             f"ocr_review_queue={len(loaded.review_queue)}")

    progress("[2/4] enrich (captions cached in data/cache/captions)")
    t = time.perf_counter()
    enriched = enrich(loaded.units, client_factory, s, progress=lambda m: progress("      " + m))
    timings["enrich_s"] = round(time.perf_counter() - t, 2)

    progress(f"[3/4] embed {len(enriched.units)} units with {s.embed_model} (batches of {s.embed_batch_size})")
    t = time.perf_counter()
    vectors = embed_units(enriched.units, client_factory(), s) if enriched.units else []
    timings["embed_s"] = round(time.perf_counter() - t, 2)
    if not vectors:
        raise RuntimeError("nothing to index: every unit was rejected (see data/index_report.json)")
    dim = len(vectors[0])

    progress(f"[4/4] upsert into {s.vector_backend}:{s.collection} (dim={dim}), provenance checked at write")
    t = time.perf_counter()
    meta = {
        "embed_model": s.embed_model,
        "embed_images": s.embed_images,
        "caption_prompt_version": s.caption_prompt_version,
        "table_context": s.table_context,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    store.reset(dim, meta)
    write_rejects = store.upsert(enriched.units, vectors)
    timings["upsert_s"] = round(time.perf_counter() - t, 2)
    timings["total_s"] = round(time.perf_counter() - t0, 2)

    rejected = [r.model_dump() for r in loaded.rejected + enriched.rejected + write_rejects]
    indexed = [u for u in enriched.units if u.unit_id not in {r["unit_id"] for r in rejected}]
    report = {
        **meta,
        "dimensions": dim,
        "vector_backend": s.vector_backend,
        "collection": s.collection,
        "corpora": list(s.corpora),
        "pdf_paths": list(s.pdf_paths),
        "units_indexed": store.count(),
        "units_by_type": dict(Counter(u.type for u in indexed)),
        "rejected_count": len(rejected),
        "rejected_by_stage": dict(Counter(r["stage"] for r in rejected)),
        "rejected": rejected,
        "page_classes": page_class_counts(loaded.pages),
        "ocr_failed_pages": len(loaded.review_queue),
        "ocr_failed": loaded.review_queue[:200],
        "captions": {"called": enriched.captions_called, "from_cache": enriched.captions_from_cache,
                     "models": enriched.caption_models},
        "model_calls": global_calls(),
        "timings": timings,
        "index_size_bytes": store.size_bytes() or index_dir_size(s.index_dir),
        "artefacts_size_bytes": index_dir_size(s.artefact_dir),
    }
    s.index_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    progress(f"done: {report['units_indexed']} units {report['units_by_type']}, rejected {len(rejected)}, "
             f"calls {report['model_calls']}, {timings['total_s']}s -> {s.index_report_path}")
    return report


def read_index_report(s: Settings | None = None) -> dict | None:
    s = s or settings()
    p: Path = s.index_report_path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
