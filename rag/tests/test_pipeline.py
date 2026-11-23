from __future__ import annotations

import json

import pytest

from app.config import ConfigError, Settings
from app.generate import answer, highlight_rows
from app.ingest import PdfLoader, garbled_ratio, gate
from app.pipeline import build_index
from app.retrieve import IndexMismatch, Retriever, to_hit
from app.schemas import Artefact, Provenance, Unit
from app.store import MemoryStore
from eval.metrics import first_rank, mrr, recall_at_k


def unit(uid="u1", page=1, typ="text", **kw):
    return Unit(unit_id=uid, type=typ, search_text=kw.get("text", "hello world"),
                artefact=kw.get("artefact", Artefact(kind="text", text="hello world")),
                provenance=Provenance(document_id=kw.get("doc", "doc"), page=page, unit_id=uid))


# ------------------------------------------------------------------ config
def test_missing_key_stops_before_any_call():
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        Settings(openrouter_api_key="").require_key()
    with pytest.raises(ConfigError, match="spaces or quotes"):
        Settings(openrouter_api_key=" sk-x ").require_key()


# --------------------------------------------------------- provenance fence
def test_store_rejects_unit_without_page(s):
    st = MemoryStore(s)
    st.reset(3, {"embed_model": "fake-embed"})
    good, bad = unit("good"), unit("bad", page=None)
    rejected = st.upsert([good, bad], [[1, 0, 0], [0, 1, 0]])
    assert [r.unit_id for r in rejected] == ["bad"]
    assert "page" in rejected[0].reason
    assert st.count() == 1 and st.get("bad") is None


def test_rebuild_twice_same_count(s, sample_pdf, fake_client):
    r1 = build_index(s, progress=lambda m: None)
    r2 = build_index(s, progress=lambda m: None)
    assert r1["units_indexed"] == r2["units_indexed"] > 0


# ------------------------------------------------------------------ routing
def test_router_classifies_pages_and_stamps_provenance(s, sample_pdf):
    res = PdfLoader(s=s).load()
    classes = {p["page"]: p["class"] for p in res.pages}
    assert classes[1] in ("mixed", "native") and classes[3] == "scanned"
    types = {u.type for u in res.units}
    assert {"text", "table", "figure"} <= types
    for u in res.units:
        assert u.provenance.is_sourced(), u.unit_id
    table = next(u for u in res.units if u.type == "table")
    assert table.provenance.page == 1
    assert table.artefact.columns[0] == "Fastener" and ["Mounting bolt", "45", "3"] in table.artefact.rows
    fig = next(u for u in res.units if u.type == "figure" and u.provenance.page == 2)
    assert (s.data_dir / fig.artefact.path).exists()
    assert "Figure 2-1" in fig.meta["caption_hint"]
    # the scan has no OCR engine in tests -> review queue, never indexed as text
    assert any(q["page"] == 3 for q in res.review_queue)
    assert not any(u.type == "text" and u.provenance.page == 3 for u in res.units)


def test_ocr_gate():
    s = Settings()
    assert garbled_ratio("Remove the four mounting bolts P/N 8923317") < 0.2
    assert garbled_ratio("~~ ;; ,. ' l| ^^ %% x") > 0.5
    assert not gate("~~ ;; ,. ' l| ^^ %%", None, "t", s).passed
    assert gate("Remove the four mounting bolts and inspect the shaft seal for leaks.", 90, "t", s).passed
    assert not gate("Remove the four mounting bolts and inspect the shaft seal for leaks.", 20, "t", s).passed


# ------------------------------------------------------- index + retrieval
def test_end_to_end_index_and_retrieve(s, sample_pdf, fake_client):
    rep = build_index(s, progress=lambda m: None)
    assert rep["embed_model"] == "fake-embed"
    assert set(rep["units_by_type"]) == {"text", "table", "figure"}
    assert rep["ocr_failed_pages"] == 1
    saved = json.loads(s.index_report_path.read_text())
    assert saved["captions"]["called"] >= 1

    r = Retriever(s)
    hits = r.retrieve("torque value for the mounting bolt", k=3)
    assert hits and all(h.citation for h in hits)
    assert "table" in [h.type for h in hits]  # fake bag-of-words embedder: rank loosely
    figs = r.retrieve("hydraulic pump exploded view", k=3, type_filter="figure")
    assert figs and all(h.type == "figure" for h in figs)

    # caption cache: a rebuild makes no caption calls
    fake_client.calls.clear()
    build_index(s, progress=lambda m: None)
    assert not [c for c in fake_client.calls if c[0] == "caption"]


def test_model_mismatch_refused(s, sample_pdf, fake_client):
    build_index(s, progress=lambda m: None)
    other = s.with_overrides(embed_model="another-embedder")
    with pytest.raises(IndexMismatch, match="Rebuild"):
        Retriever(other).retrieve("anything")


def test_lexical_fusion_finds_exact_identifier(s, sample_pdf, fake_client):
    build_index(s, progress=lambda m: None)
    fused = Retriever(s.with_overrides(lexical_weight=0.6))
    hits = fused.retrieve("Drain plug torque", k=3)
    assert hits[0].type == "table"


# ------------------------------------------------------------------ answer
def test_abstains_when_nothing_sourced(s, fake_client):
    u = unit("orphan", page=None)
    hits = [to_hit(1, u, 0.9)]
    a = answer("q", hits, lambda: fake_client, s)
    assert a.status == "abstained" and a.citation is None and "page" in a.text
    assert answer("q", [], lambda: fake_client, s).status == "abstained"


def test_unsourced_top_hit_is_skipped(s, fake_client):
    orphan = unit("orphan", page=None)
    good = unit("good", page=4)
    a = answer("hello", [to_hit(1, orphan, 0.95), to_hit(2, good, 0.5)], lambda: fake_client, s)
    assert a.status == "answered" and a.cited_unit_id == "good" and a.citation == "doc > p.4 > good"
    assert a.dropped_unsourced == 1
    assert "[made-up-unit]" not in a.text and "made-up-unit" in (a.generation_error or "")


def test_table_answer_returns_rows_and_highlights(s, fake_client):
    t = unit("tab", typ="table", text="Fastener=Mounting bolt; Torque=45",
             artefact=Artefact(kind="table", columns=["Fastener", "Torque", "Item"],
                               rows=[["Mounting bolt", "45", "3"], ["Drain plug", "25", "6"]]))
    a = answer("torque for the drain plug, item 6", [to_hit(1, t, 0.8)], lambda: fake_client, s)
    assert a.answer_type == "table" and a.artefact.rows and a.highlight_rows == [1]


def test_highlight_rows_item_numbers():
    cols = ["item_no", "part_no", "description"]
    rows = [["28", "123", "O-RING"], ["29", "124", "SEAL,RING"], ["30", "125", "BOLT"], ["31", "126", "SEAL"]]
    assert highlight_rows("do items 28, 29, and 31 come with the liner?", cols, rows) == [0, 1, 3]


# ----------------------------------------------------------------- metrics
def test_metrics():
    assert first_rank(["a", "b", "c"], {"c"}) == 3
    ranks = [1, 3, None, 2]
    assert recall_at_k(ranks, 1) == 0.25 and recall_at_k(ranks, 3) == 0.75
    assert abs(mrr(ranks) - (1 + 1 / 3 + 0.5) / 4) < 1e-9
