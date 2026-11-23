from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import server
from app.pipeline import build_index
from app.schemas import Artefact, Provenance, Unit
from app.service import QueryService
from app.store import MemoryStore


@pytest.fixture
def api(s, sample_pdf, fake_client):
    build_index(s, progress=lambda m: None)
    server._service = QueryService(s, client_factory=lambda: fake_client)
    yield TestClient(server.app)
    server._service = None


def test_query_returns_artefact_citation_sources_timings(api):
    r = api.post("/query", json={"question": "hydraulic pump exploded view diagram", "type_filter": "figure"})
    assert r.status_code == 200, r.text
    body = r.json()
    a = body["answer"]
    assert a["status"] == "answered" and a["answer_type"] == "figure"
    assert a["citation"].startswith("sample > p.2 >")
    assert body["sources"] and any(h["cited"] for h in body["sources"])
    assert set(body["timings_ms"]) == {"retrieval", "generation", "total"}
    img = api.get(a["artefact_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"


def test_health_reports_degraded_without_model_check(api):
    h = api.get("/health").json()
    assert h["status"] == "degraded"
    assert any(p["kind"] == "models" for p in h["problems"])


def test_retrieve_batch(api):
    r = api.post("/retrieve/batch", json={"questions": ["torque", "pump"], "k": 2})
    assert r.status_code == 200 and len(r.json()["results"]) == 2


def test_artefact_path_guard(api, s):
    store: MemoryStore = server._service.retriever.store
    (s.data_dir.parent / "secret.txt").write_text("nope")
    evil = Unit(unit_id="evil", type="figure", search_text="x",
                artefact=Artefact(kind="image", path="../secret.txt"),
                provenance=Provenance(document_id="d", page=1, unit_id="evil"))
    store.upsert([evil], [[0.0] * len(store._mat[0])])
    assert api.get("/artefact/evil").status_code == 403


def test_index_mismatch_is_configuration_error(api, s):
    meta = json.loads(server._service.retriever.store.meta_path.read_text())
    meta["embed_model"] = "something-else"
    server._service.retriever.store.write_meta(meta)
    r = api.post("/query", json={"question": "pump"})
    assert r.status_code == 503 and r.json()["error"]["kind"] == "index_mismatch"


def test_upstream_error_is_distinct(api, fake_client):
    from app.openrouter import ModelUnavailable

    def boom(**kw):
        raise ModelUnavailable("dead/model:free", "slug retired", 404)

    fake_client.embed_texts = lambda texts, model=None: boom()
    r = api.post("/query", json={"question": "a brand new question"})
    assert r.status_code == 502
    assert r.json()["error"] == {"kind": "model_unavailable", "model": "dead/model:free",
                                 "message": "[dead/model:free] slug retired (HTTP 404)"}
