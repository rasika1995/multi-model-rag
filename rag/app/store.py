"""Vector-store interface with two backends: Qdrant and in-memory (persisted).

The write path is the provenance fence: a unit without document, page and
unit id is refused and logged, never stored. Index metadata (embedding model,
dimensions, figure mode) is saved next to the index so retrieval can refuse
to query with a mismatched model.
"""
from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ConfigError, Settings, settings
from .schemas import ProvenanceError, Rejected, Unit

log = logging.getLogger(__name__)


@dataclass
class StoreHit:
    unit: Unit
    score: float


class VectorStore(ABC):
    backend: str

    def __init__(self, s: Settings):
        self.s = s
        self.meta_path = s.index_dir / f"{self.backend}_{s.collection}_meta.json"

    # ---- metadata (embedding model recorded with the index) ---------------
    def read_meta(self) -> dict[str, Any] | None:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return None

    def write_meta(self, meta: dict[str, Any]) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- provenance fence ---------------------------------------------------
    @staticmethod
    def check(units: list[Unit], vectors: list[list[float]], dim: int) -> tuple[list[int], list[Rejected]]:
        ok, rejected = [], []
        for i, (u, v) in enumerate(zip(units, vectors)):
            try:
                u.validate_provenance()
            except ProvenanceError as e:
                log.warning("REJECTED at write: %s", e)
                rejected.append(Rejected(unit_id=u.unit_id, stage="upsert", reason=str(e),
                                         document_id=u.provenance.document_id, page=u.provenance.page))
                continue
            if len(v) != dim:
                rejected.append(Rejected(unit_id=u.unit_id, stage="upsert",
                                         reason=f"vector dim {len(v)} != index dim {dim}"))
                continue
            ok.append(i)
        return ok, rejected

    @abstractmethod
    def reset(self, dim: int, meta: dict[str, Any]) -> None: ...

    @abstractmethod
    def upsert(self, units: list[Unit], vectors: list[list[float]]) -> list[Rejected]: ...

    @abstractmethod
    def search(self, vector: list[float], k: int, type_filter: str | None = None) -> list[StoreHit]: ...

    @abstractmethod
    def get(self, unit_id: str) -> Unit | None: ...

    @abstractmethod
    def all_units(self) -> list[Unit]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def ping(self) -> None: ...

    def size_bytes(self) -> int:
        return 0


# =============================================================== in-memory
class MemoryStore(VectorStore):
    backend = "memory"

    def __init__(self, s: Settings):
        super().__init__(s)
        self.vec_path = s.index_dir / f"memory_{s.collection}.npy"
        self.units_path = s.index_dir / f"memory_{s.collection}_units.jsonl"
        self._units: list[Unit] = []
        self._by_id: dict[str, int] = {}
        self._mat: np.ndarray | None = None
        self._dim: int | None = None
        self._load()

    def _load(self) -> None:
        if self.vec_path.exists() and self.units_path.exists():
            self._mat = np.load(self.vec_path)
            self._units = [Unit.model_validate_json(l) for l in
                           self.units_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            self._by_id = {u.unit_id: i for i, u in enumerate(self._units)}
            self._dim = int(self._mat.shape[1]) if self._mat.size else (self.read_meta() or {}).get("dimensions")

    def _save(self) -> None:
        self.vec_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.vec_path, self._mat if self._mat is not None else np.zeros((0, self._dim or 0), np.float32))
        self.units_path.write_text("\n".join(u.model_dump_json() for u in self._units), encoding="utf-8")

    def reset(self, dim: int, meta: dict[str, Any]) -> None:
        self._units, self._by_id, self._dim = [], {}, dim
        self._mat = np.zeros((0, dim), dtype=np.float32)
        self.write_meta({**meta, "dimensions": dim, "backend": self.backend})
        self._save()

    def upsert(self, units: list[Unit], vectors: list[list[float]]) -> list[Rejected]:
        if self._dim is None or self._mat is None:
            raise ConfigError("index not initialised: run scripts/build_index.py")
        ok, rejected = self.check(units, vectors, self._dim)
        rows = []
        for i in ok:
            u, v = units[i], np.asarray(vectors[i], dtype=np.float32)
            v = v / (np.linalg.norm(v) or 1.0)
            if u.unit_id in self._by_id:
                self._mat[self._by_id[u.unit_id]] = v
                self._units[self._by_id[u.unit_id]] = u
            else:
                self._by_id[u.unit_id] = len(self._units) + len(rows)
                rows.append((u, v))
        if rows:
            self._units += [u for u, _ in rows]
            self._mat = np.vstack([self._mat, np.stack([v for _, v in rows])])
        self._save()
        return rejected

    def search(self, vector, k, type_filter=None):
        if self._mat is None or not len(self._units):
            return []
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self._mat @ q
        if type_filter:
            mask = np.array([u.type == type_filter for u in self._units])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [StoreHit(self._units[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def get(self, unit_id):
        i = self._by_id.get(unit_id)
        return self._units[i] if i is not None else None

    def all_units(self):
        return list(self._units)

    def count(self):
        return len(self._units)

    def ping(self):
        return None

    def size_bytes(self):
        return sum(p.stat().st_size for p in (self.vec_path, self.units_path) if p.exists())


# ================================================================== Qdrant
class QdrantStore(VectorStore):
    backend = "qdrant"

    def __init__(self, s: Settings):
        super().__init__(s)
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=s.qdrant_url, timeout=30)

    @staticmethod
    def pid(unit_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, unit_id))

    def ping(self) -> None:
        try:
            self.client.get_collections()
        except Exception as e:
            raise ConfigError(f"Qdrant not reachable at {self.s.qdrant_url}: {e}. "
                              "Start it with `docker compose up -d` or set VECTOR_BACKEND=memory.") from e

    def reset(self, dim, meta):
        from qdrant_client import models as m

        self.ping()
        if self.client.collection_exists(self.s.collection):
            self.client.delete_collection(self.s.collection)
        self.client.create_collection(self.s.collection, vectors_config=m.VectorParams(size=dim, distance=m.Distance.COSINE))
        for key, schema in (("type", m.PayloadSchemaType.KEYWORD), ("document_id", m.PayloadSchemaType.KEYWORD),
                            ("page", m.PayloadSchemaType.INTEGER)):
            self.client.create_payload_index(self.s.collection, field_name=key, field_schema=schema)
        self.write_meta({**meta, "dimensions": dim, "backend": self.backend})

    def upsert(self, units, vectors):
        from qdrant_client import models as m

        meta = self.read_meta() or {}
        dim = int(meta.get("dimensions") or len(vectors[0]))
        ok, rejected = self.check(units, vectors, dim)
        points = [m.PointStruct(
            id=self.pid(units[i].unit_id), vector=list(map(float, vectors[i])),
            payload={
                "document_id": units[i].provenance.document_id,
                "page": units[i].provenance.page,
                "unit_id": units[i].unit_id,
                "type": units[i].type,
                "unit": units[i].model_dump(mode="json"),
            },
        ) for i in ok]
        for start in range(0, len(points), 64):
            self.client.upsert(self.s.collection, points=points[start:start + 64], wait=True)
        return rejected

    def search(self, vector, k, type_filter=None):
        from qdrant_client import models as m

        flt = m.Filter(must=[m.FieldCondition(key="type", match=m.MatchValue(value=type_filter))]) if type_filter else None
        res = self.client.query_points(self.s.collection, query=list(map(float, vector)), limit=k,
                                       query_filter=flt, with_payload=True)
        return [StoreHit(Unit.model_validate(p.payload["unit"]), float(p.score)) for p in res.points]

    def get(self, unit_id):
        pts = self.client.retrieve(self.s.collection, ids=[self.pid(unit_id)], with_payload=True)
        return Unit.model_validate(pts[0].payload["unit"]) if pts else None

    def all_units(self):
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(self.s.collection, limit=256, offset=offset, with_payload=True)
            out += [Unit.model_validate(p.payload["unit"]) for p in pts]
            if offset is None:
                return out

    def count(self):
        if not self.client.collection_exists(self.s.collection):
            return 0
        return int(self.client.count(self.s.collection, exact=True).count)


def make_store(s: Settings | None = None) -> VectorStore:
    s = s or settings()
    if s.vector_backend == "qdrant":
        return QdrantStore(s)
    if s.vector_backend == "memory":
        return MemoryStore(s)
    raise ConfigError(f"VECTOR_BACKEND must be 'qdrant' or 'memory', got {s.vector_backend!r}")


def index_dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0
