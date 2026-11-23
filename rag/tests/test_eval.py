from __future__ import annotations

from app.pipeline import build_index, read_index_report
from eval import run_eval


def test_layers_a_b_c_and_report(s, sample_pdf, fake_client):
    build_index(s, progress=lambda m: None)
    eng = run_eval.LocalEngine()
    units = {u.unit_id: u for u in eng.svc.retriever.store.all_units()}
    fig = next(u for u in units if "figure" in u and "p002" in u)
    tab = next(u for u in units if "table" in u)
    qa = [
        {"id": "q01", "qtype": "figure", "question": "hydraulic pump exploded view", "answer": "see figure",
         "context": "", "gold_units": [fig, tab], "gold_by_type": {"figure": [fig], "table": [tab]}},
        {"id": "q02", "qtype": "table", "question": "drain plug torque", "answer": "25 Nm", "context": "",
         "gold_units": [tab], "gold_by_type": {"table": [tab]}},
    ]
    a = run_eval.layer_a(eng, qa, k=5)
    assert set(a["by_target_type"]) == {"figure", "table"}
    assert a["by_target_type"]["figure"]["n"] == 1 and "filtered" in a["by_target_type"]["figure"]
    b, rows = run_eval.layer_b(eng, qa, use_judge=False, pause_s=0)
    assert b["answered"] == 2 and b["citation_accuracy"] is not None
    c = run_eval.layer_c(rows, read_index_report(s), 20)
    assert c["latency_ms"]["total"]["p50"] >= 0
    rep = {"label": "t", "run_at": "now", "index": {}, "A": a, "B": b, "C": c, "judge_model": None}
    rep["compare"] = run_eval.compare(rep, rep)
    md = run_eval.to_markdown(rep)
    assert "Layer A" in md and "Wrong-source rate" in md and "Queries per minute" in md
