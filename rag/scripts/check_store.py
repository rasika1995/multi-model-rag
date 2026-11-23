"""Vector DB connection test (E0.2): create, write, search and delete a scratch collection."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.schemas import Artefact, Provenance, Unit  # noqa: E402
from app.store import make_store  # noqa: E402


def main() -> int:
    s = settings().with_overrides(collection="connection_test")
    st = make_store(s)
    st.ping()
    st.reset(4, {"embed_model": "test"})
    u = Unit(unit_id="u1", type="text", search_text="x", artefact=Artefact(kind="text", text="x"),
             provenance=Provenance(document_id="d", page=1, unit_id="u1"))
    st.upsert([u], [[1, 0, 0, 0]])
    hits = st.search([1, 0, 0, 0], 1)
    assert hits and hits[0].unit.unit_id == "u1"
    if s.vector_backend == "qdrant":
        st.client.delete_collection("connection_test")
    else:
        for p in (st.vec_path, st.units_path, st.meta_path):
            p.unlink(missing_ok=True)
    print(f"OK: {s.vector_backend} create/write/search/delete works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
