"""Stage 5: embed the query, search (optional type filter), optional BM25 fusion.

The generator is never involved here, so Layer A can measure retrieval alone.
"""
from __future__ import annotations

import re
import threading
from typing import Callable

import numpy as np

from .config import ConfigError, Settings, settings
from .openrouter import ModelClient, get_client
from .schemas import Hit, Unit
from .store import StoreHit, VectorStore, make_store


class IndexMismatch(ConfigError):
    pass


class IndexMissing(ConfigError):
    pass


def check_index(store: VectorStore, s: Settings) -> dict:
    """Refuse to query an index built with a different embedder / dimension / figure mode."""
    meta = store.read_meta()
    if not meta:
        raise IndexMissing("No index found. Build it with: python scripts/build_index.py")
    problems = []
    if meta.get("embed_model") != s.embed_model:
        problems.append(f"index built with EMBED_MODEL={meta.get('embed_model')!r}, configured {s.embed_model!r}")
    if s.embed_dimensions and meta.get("dimensions") != s.embed_dimensions:
        problems.append(f"index dimensions {meta.get('dimensions')} != EMBED_DIMENSIONS {s.embed_dimensions}")
    if bool(meta.get("embed_images")) != bool(s.embed_images):
        problems.append(f"index figure mode embed_images={meta.get('embed_images')} != configured {s.embed_images}")
    if problems:
        raise IndexMismatch("; ".join(problems) + ". Rebuild the index: python scripts/build_index.py")
    return meta


_TOK = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    toks = _TOK.findall(text.lower())
    out = []
    for t in toks:
        out.append(t)
        if re.search(r"[-/.]", t):
            out += [p for p in re.split(r"[-/.]", t) if p]
    return out


class Retriever:
    def __init__(self, s: Settings | None = None, store: VectorStore | None = None,
                 client_factory: Callable[[], ModelClient] = get_client):
        self.s = s or settings()
        self.store = store or make_store(self.s)
        self.client_factory = client_factory
        self._bm25 = None
        self._bm25_units: list[Unit] = []
        self._lock = threading.Lock()

    # --------------------------------------------------------------- lexical
    def _lexical(self):
        with self._lock:
            if self._bm25 is None:
                from rank_bm25 import BM25Okapi

                self._bm25_units = self.store.all_units()
                corpus = [tokenize(u.search_text) or ["_"] for u in self._bm25_units]
                self._bm25 = BM25Okapi(corpus) if corpus else None
        return self._bm25

    def _fuse(self, question: str, vec_hits: list[StoreHit], k: int, type_filter: str | None):
        w = self.s.lexical_weight
        bm = self._lexical()
        if bm is None:
            return [(h.unit, h.score, h.score, None) for h in vec_hits[:k]]
        lex = np.asarray(bm.get_scores(tokenize(question)), dtype=float)
        idx = {u.unit_id: i for i, u in enumerate(self._bm25_units)}
        cand: dict[str, tuple[Unit, float | None]] = {h.unit.unit_id: (h.unit, h.score) for h in vec_hits}
        allowed = [i for i, u in enumerate(self._bm25_units) if not type_filter or u.type == type_filter]
        for i in sorted(allowed, key=lambda i: -lex[i])[: max(k * 2, 10)]:
            u = self._bm25_units[i]
            cand.setdefault(u.unit_id, (u, None))
        vmin = min((v for _, v in cand.values() if v is not None), default=0.0)
        vmax = max((v for _, v in cand.values() if v is not None), default=1.0)
        lmax = max(lex.max() if len(lex) else 0.0, 1e-9)
        rows = []
        for uid, (u, v) in cand.items():
            vn = 0.0 if v is None else (v - vmin) / ((vmax - vmin) or 1.0)
            ln = lex[idx[uid]] / lmax if uid in idx else 0.0
            rows.append((u, (1 - w) * vn + w * ln, v, float(lex[idx[uid]]) if uid in idx else 0.0))
        rows.sort(key=lambda r: -r[1])
        return rows[:k]

    # ---------------------------------------------------------------- public
    def search_vector(self, question: str, vector: list[float], k: int, type_filter: str | None = None) -> list[Hit]:
        fetch = max(k * 4, 20) if self.s.lexical_weight > 0 else k
        vh = self.store.search(vector, fetch, type_filter)
        if self.s.lexical_weight > 0:
            rows = self._fuse(question, vh, k, type_filter)
        else:
            rows = [(h.unit, h.score, h.score, None) for h in vh[:k]]
        return [to_hit(r, u, score, v, l) for r, (u, score, v, l) in enumerate(rows, 1)]

    def retrieve(self, question: str, k: int | None = None, type_filter: str | None = None) -> list[Hit]:
        check_index(self.store, self.s)
        vec = self.client_factory().embed_texts([question])[0]
        return self.search_vector(question, vec, k or self.s.top_k, type_filter)

    def retrieve_batch(self, questions: list[str], k: int, type_filter: str | None = None) -> list[list[Hit]]:
        """One embeddings call per batch of 16 questions, not one per question."""
        check_index(self.store, self.s)
        vecs = self.client_factory().embed_texts(questions)
        return [self.search_vector(q, v, k, type_filter) for q, v in zip(questions, vecs)]


def to_hit(rank: int, u: Unit, score: float, vscore: float | None = None, lscore: float | None = None) -> Hit:
    sourced = u.provenance.is_sourced()
    return Hit(
        rank=rank, unit_id=u.unit_id, type=u.type, score=round(float(score), 4),
        vector_score=None if vscore is None else round(float(vscore), 4),
        lexical_score=None if lscore is None else round(float(lscore), 4),
        sourced=sourced, citation=u.provenance.citation() if sourced else None,
        provenance=u.provenance, search_text=u.search_text, artefact=u.artefact,
        artefact_url=f"/artefact/{u.unit_id}" if u.artefact.path else None,
    )
