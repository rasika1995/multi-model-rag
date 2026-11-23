"""Build data/qa.json, the held-out gold set, from Enginuity Task 2.

Each record: id, question, gold answer, cleaned context, question type, and
gold units by type (the figure and its table(s)), plus gold document/pages.

Usage (from rag/):  python scripts/build_gold.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.ingest import EnginuityT2Loader  # noqa: E402

TABLE_Q = re.compile(
    r"part number|part no|p/n|stock number|national stock|\bnsn\b|how many|quantity|\bqty\b|"
    r"which item number|what item number|listed as|parts list say", re.I)


def clean_context(c: str) -> str:
    """Drop stray caption text from the next figure: '... Figure 03 of 10 · Fuel System Figure 27. ...'."""
    c = re.split(r"\s*Figure \d+ of \d+\b", c)[0]
    return re.sub(r"\s+", " ", c).strip()


def qtype(question: str) -> str:
    """'table' for direct lookups the parts list answers, else 'figure' (reasoning over the diagram)."""
    return "table" if TABLE_Q.search(question) else "figure"


def main() -> None:
    s = settings()
    root = s.data_dir / "enginuity" / "task-2-question-answering"
    raw = json.loads((root / "qa_dataset.json").read_text(encoding="utf-8"))
    out = []
    for i, r in enumerate(raw, 1):
        num = re.search(r"figure_(\d+)", r["figure"]).group(1)
        group = f"fig{num}"
        fig = EnginuityT2Loader.figure_uid(group)
        tabs = [EnginuityT2Loader.table_uid(group, int(Path(t).stem.rsplit("_", 1)[1])) for t in r["tables"]]
        doc = EnginuityT2Loader.doc_id(group)
        out.append({
            "id": f"q{i:02d}",
            "question": r["question"].replace("�", "-"),
            "answer": r["answer"],
            "context": clean_context(r["context"]),
            "qtype": qtype(r["question"]),
            "group": group,
            "gold_units": [fig, *tabs],
            "gold_by_type": {"figure": [fig], "table": tabs},
            "gold_pages": [{"document": doc, "page": p} for p in range(1, len(tabs) + 2)],
            "source": "enginuity-task2",
        })
    s.qa_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(out)} questions -> {s.qa_path}")
    print("question types:", dict(Counter(q["qtype"] for q in out)))
    print("NOTE: Enginuity has no prose gold, so 'text' recall needs a text-gold corpus (e.g. ViDoRe, tier C).")


if __name__ == "__main__":
    main()
