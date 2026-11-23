"""Tier C gold from ViDoRe v3 Industrial: page-level evidence with content types.

Step 1 (plan, no index needed): choose documents, print a PDF_PATHS value that
ingests only the gold pages plus neighbouring distractor pages (keeps the
captioning budget small):
    python scripts/build_gold_vidore.py --plan --docs TO_00-25-172CL-1,TO_00-25-172CL-3

Step 2: put that PDF_PATHS (and CORPORA=enginuity_t2,pdf) in rag/.env, build the index.

Step 3 (build): map every qrel (document, page, content type) to the indexed
units on that page and write data/qa_vidore.json (and data/qa_all.json merged
with the Enginuity set):
    python scripts/build_gold_vidore.py --build --docs TO_00-25-172CL-1,TO_00-25-172CL-3

A qrel whose content type has no extracted unit on its page gets a placeholder
gold id ("missing:..."), so an extraction miss counts as a retrieval miss
instead of silently disappearing from the metrics.

Requires data/vidore (see README) and data/vidore/corpus_pages.csv (corpus_id ->
doc/page map; created by --fetch-map, which reads only three columns remotely).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

REV = "233d20721f4deb392a09a86cca01761adbc91157"
CT_MAP = {"Text": "text", "Table": "table", "Infographic": "figure", "Image": "figure", "Chart": "figure",
          "Figure": "figure", "Diagram": "figure"}


def fetch_map(root: Path) -> None:
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    parts = [pq.read_table(fs.open(f), columns=["corpus_id", "doc_id", "page_number_in_doc"]).to_pandas()
             for f in sorted(fs.glob(f"datasets/vidore/vidore_v3_industrial@{REV}/corpus/*.parquet"))]
    pd.concat(parts).to_csv(root / "corpus_pages.csv", index=False)


def load(root: Path, docs: set[str]):
    q = pd.read_parquet(root / "queries")
    r = pd.read_parquet(root / "qrels")
    cp = pd.read_csv(root / "corpus_pages.csv")
    q = q[q.language == "english"]
    r = r.merge(cp, on="corpus_id")
    r["page"] = r.page_number_in_doc + 1  # ViDoRe pages are 0-based; ours are 1-based
    inside = r.groupby("query_id").doc_id.apply(lambda s: set(s) <= docs)
    keep = set(inside[inside].index)
    q = q[q.query_id.isin(keep)]
    r = r[r.query_id.isin(q.query_id)]
    return q, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="comma-separated ViDoRe doc ids (PDF stems) to use")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--fetch-map", action="store_true")
    ap.add_argument("--neighbours", type=int, default=1, help="distractor pages either side of a gold page")
    a = ap.parse_args()
    s = settings()
    root = s.data_dir / "vidore"
    if a.fetch_map or not (root / "corpus_pages.csv").exists():
        fetch_map(root)
    docs = {d.strip() for d in a.docs.split(",") if d.strip()}
    q, r = load(root, docs)
    print(f"{len(q)} English queries fully inside {sorted(docs)}; {len(r)} qrels")

    if a.plan:
        specs = []
        for d, g in r.groupby("doc_id"):
            pages = set()
            for p in g.page:
                pages.update(range(max(1, p - a.neighbours), p + a.neighbours + 1))
            specs.append(f"vidore/pdfs/{d}.pdf@" + ";".join(str(p) for p in sorted(pages)))
        print("\nPut this in rag/.env:\nCORPORA=enginuity_t2,pdf\nPDF_PATHS=" + ",".join(specs))
        return 0

    if a.build:
        from app.store import make_store

        units = make_store(s).all_units()
        by_page: dict[tuple[str, int, str], list[str]] = defaultdict(list)
        for u in units:
            by_page[(u.provenance.document_id, u.provenance.page, u.type)].append(u.unit_id)
        out, missing = [], Counter()
        for _, row in q.sort_values("query_id").iterrows():
            gold: dict[str, list[str]] = defaultdict(list)
            pages = []
            for _, g in r[r.query_id == row.query_id].iterrows():
                for ct in g.content_type:
                    t = CT_MAP.get(ct)
                    if not t:
                        continue
                    ids = by_page.get((g.doc_id, int(g.page), t))
                    if not ids:
                        ids = [f"missing:{g.doc_id}:p{int(g.page)}:{t}"]
                        missing[t] += 1
                    gold[t] += [i for i in ids if i not in gold[t]]
                pages.append({"document": g.doc_id, "page": int(g.page), "score": int(g.score)})
            if not gold:
                continue
            qt = next((CT_MAP[c] for c in row.content_type if c in CT_MAP), "text")
            out.append({
                "id": f"v{int(row.query_id):04d}", "question": row.query, "answer": row.answer or "",
                "context": "", "qtype": qt, "gold_by_type": dict(gold),
                "gold_units": sorted({u for v in gold.values() for u in v}), "gold_pages": pages,
                "source": "vidore-v3-industrial",
            })
        (s.data_dir / "qa_vidore.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        merged = json.loads(s.qa_path.read_text(encoding="utf-8")) if s.qa_path.exists() else []
        merged = [x for x in merged if x.get("source") != "vidore-v3-industrial"] + out
        (s.data_dir / "qa_all.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(f"wrote {len(out)} -> data/qa_vidore.json; {len(merged)} -> data/qa_all.json")
        print("question types:", dict(Counter(x["qtype"] for x in out)),
              "| gold targets with no extracted unit (counted as misses):", dict(missing))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
