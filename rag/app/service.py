"""Query orchestration shared by the HTTP API and the in-process evaluator.

retrieve (timed) -> answer (timed) -> response with sources, timings and the
number of model calls this query made.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from .config import ConfigError, Settings, settings
from .diagnostics import read_last_check
from .generate import answer
from .openrouter import ModelClient, get_client, track_calls
from .pipeline import read_index_report
from .retrieve import Retriever, check_index
from .schemas import Answer, Hit, QueryResponse


class QueryService:
    def __init__(self, s: Settings | None = None, client_factory: Callable[[], ModelClient] = get_client):
        self.s = s or settings()
        self.client_factory = client_factory
        self._retriever: Retriever | None = None
        self._meta_mtime: float | None = None
        self._lock = threading.Lock()

    @property
    def retriever(self) -> Retriever:
        """Recreated when the index is rebuilt while the server runs."""
        with self._lock:
            p = self.s.index_report_path
            mtime = p.stat().st_mtime if p.exists() else None
            if self._retriever is None or mtime != self._meta_mtime:
                self._retriever = Retriever(self.s, client_factory=self.client_factory)
                self._meta_mtime = mtime
            return self._retriever

    def index_info(self) -> dict:
        rep = read_index_report(self.s) or {}
        return {k: rep.get(k) for k in ("embed_model", "dimensions", "embed_images", "vector_backend",
                                         "units_indexed", "units_by_type", "built_at", "caption_prompt_version",
                                         "table_context")} | {"lexical_weight": self.s.lexical_weight}

    def retrieve(self, question: str, k: int | None, type_filter: str | None) -> tuple[list[Hit], dict, dict]:
        with track_calls() as calls:
            t = time.perf_counter()
            hits = self.retriever.retrieve(question, k or self.s.top_k, type_filter)
            ms = (time.perf_counter() - t) * 1000
        return hits, {"retrieval": round(ms, 1)}, dict(calls)

    def retrieve_batch(self, questions: list[str], k: int, type_filter: str | None):
        with track_calls() as calls:
            t = time.perf_counter()
            res = self.retriever.retrieve_batch(questions, k, type_filter)
            ms = (time.perf_counter() - t) * 1000
        return res, {"retrieval": round(ms, 1)}, dict(calls)

    def query(self, question: str, k: int | None = None, type_filter: str | None = None,
              generate: bool = True) -> QueryResponse:
        t0 = time.perf_counter()
        with track_calls() as calls:
            t = time.perf_counter()
            hits = self.retriever.retrieve(question, k or self.s.top_k, type_filter)
            t_ret = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            if generate:
                ans = answer(question, hits, self.client_factory, self.s)
            else:
                sourced = [h for h in hits if h.sourced]
                if sourced:
                    sourced[0].cited = True
                    top = sourced[0]
                    ans = Answer(status="answered", answer_type=top.type, citation=top.citation,
                                 cited_unit_id=top.unit_id, artefact=top.artefact, artefact_url=top.artefact_url,
                                 text=f"Top sourced unit: {top.citation} (generation off).")
                else:
                    ans = Answer(status="abstained", text="No sourced unit retrieved.",
                                 abstain_reason="no retrieved unit has resolvable provenance")
            t_gen = (time.perf_counter() - t) * 1000
        return QueryResponse(
            question=question, answer=ans, sources=hits,
            timings_ms={"retrieval": round(t_ret, 1), "generation": round(t_gen, 1),
                        "total": round((time.perf_counter() - t0) * 1000, 1)},
            calls={**dict(calls), "total": sum(calls.values())},
            index=self.index_info(),
        )

    def health(self) -> dict:
        s = self.s
        problems: list[dict] = []
        try:
            s.require_key()
        except ConfigError as e:
            problems.append({"kind": "configuration", "message": str(e), "fix": "add OPENROUTER_API_KEY to rag/.env"})
        index = None
        try:
            store = self.retriever.store
            store.ping()
            index = check_index(store, s)
        except ConfigError as e:
            problems.append({"kind": "index", "message": str(e), "fix": "python scripts/build_index.py"})
        except Exception as e:  # e.g. Qdrant down
            problems.append({"kind": "vector_db", "message": str(e), "fix": "docker compose up -d, or VECTOR_BACKEND=memory"})
        check = read_last_check(s)
        configured = {s.embed_model, s.vision_model, s.text_model, s.figure_model}
        if not check:
            problems.append({"kind": "models", "message": "model readiness never checked",
                             "fix": "python scripts/check_models.py"})
        else:
            for r in check["results"]:
                if not r["ok"] and r["model"] in configured:
                    problems.append({"kind": "models", "message": f"{r['role']} model {r['model']} failed: {r['note']}",
                                     "fix": "change the slug in rag/.env (see section 6) and re-run check_models"})
            checked = {r["model"] for r in check["results"]}
            unchecked = configured - checked
            if unchecked:
                problems.append({"kind": "models", "message": f"slugs changed since last check: {sorted(unchecked)}",
                                 "fix": "python scripts/check_models.py"})
        return {
            "status": "healthy" if not problems else "degraded",
            "problems": problems,
            "index": self.index_info() if index else None,
            "models": {"embed": s.embed_model, "vision": s.vision_model, "text": s.text_model,
                       "figure": s.figure_model},
            "vector_backend": s.vector_backend,
            "model_check": check,
        }

    def artefact_path(self, unit_id: str) -> Path:
        u = self.retriever.store.get(unit_id)
        if u is None or not u.artefact.path:
            raise FileNotFoundError(unit_id)
        root = self.s.data_dir.resolve()
        p = (root / u.artefact.path).resolve()
        if root not in p.parents:  # guard against reading outside the data folder
            raise PermissionError(f"artefact path escapes the data folder: {u.artefact.path}")
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p
