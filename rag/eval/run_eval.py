"""One command for Layers A, B and C. Saves JSON + Markdown with per-question results.

Examples (from rag/):
  python -m eval.run_eval --layers A --label baseline                 # retrieval only, via the API
  python -m eval.run_eval --layers A --label fix1 --compare data/reports/baseline.json
  python -m eval.run_eval --layers B,C --sample 20 --label dev        # answers + ops on 20 questions
  python -m eval.run_eval --layers A,B,C --local --label final        # in-process, no server needed

Layer A batches question embeddings (16 per call) and never calls the generator.
Type-filtered runs reuse the cached question embeddings, so they cost no calls.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from eval.metrics import first_rank, ndcg_at_k, percentile, precision_at_k, summarise_ranks, token_f1  # noqa: E402

TYPES = ("text", "figure", "table")


# ================================================================ engines
class ApiEngine:
    def __init__(self, base: str):
        import httpx

        self.base = base.rstrip("/")
        self.http = httpx.Client(timeout=600)

    def _post(self, path, body):
        r = self.http.post(self.base + path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:500]}")
        return r.json()

    def retrieve_batch(self, qs, k, type_filter=None):
        out, calls = [], 0
        for i in range(0, len(qs), 16):
            r = self._post("/retrieve/batch", {"questions": qs[i:i + 16], "k": k, "type_filter": type_filter})
            out += r["results"]
            calls += sum(r["calls"].values())
        return out, calls

    def query(self, q, k=None):
        return self._post("/query", {"question": q, "k": k})

    def health(self):
        try:
            r = self.http.get(self.base + "/health")
            return r.json()
        except Exception as e:
            raise SystemExit(
                f"No RAG API answered at {self.base}/health ({e.__class__.__name__}). "
                "Start it (docker compose up -d) or pass --api http://127.0.0.1:8000, or use --local. "
                "On Windows, 'localhost' can hit another app (e.g. one running in WSL)."
            ) from None


class LocalEngine:
    def __init__(self):
        from app.service import QueryService

        self.svc = QueryService()

    def retrieve_batch(self, qs, k, type_filter=None):
        out, calls = [], 0
        for i in range(0, len(qs), 16):
            res, _, c = self.svc.retrieve_batch(qs[i:i + 16], k, type_filter)
            out += [[h.model_dump() for h in hits] for hits in res]
            calls += sum(c.values())
        return out, calls

    def query(self, q, k=None):
        return self.svc.query(q, k).model_dump()

    def health(self):
        return self.svc.health()


# ================================================================ Layer A
def layer_a(engine, qa: list[dict], k: int) -> dict:
    qs = [q["question"] for q in qa]
    mixed, calls = engine.retrieve_batch(qs, k)
    filtered = {}
    for t in TYPES:
        if any(q["gold_by_type"].get(t) for q in qa):
            filtered[t], c = engine.retrieve_batch(qs, k, t)
            calls += c
    per_q = []
    for q, hits in zip(qa, mixed):
        ids = [h["unit_id"] for h in hits]
        row = {"id": q["id"], "qtype": q["qtype"], "question": q["question"], "gold": q["gold_units"],
               "gold_types": [t for t in TYPES if q["gold_by_type"].get(t)],
               "top": [{"unit_id": h["unit_id"], "type": h["type"], "score": h["score"]} for h in hits[:5]],
               "rank_any": first_rank(ids, q["gold_units"]),
               "p@5": precision_at_k(ids, set(q["gold_units"]), 5),
               "ndcg@5": ndcg_at_k(ids, set(q["gold_units"]), 5)}
        for t in TYPES:
            g = q["gold_by_type"].get(t) or []
            row[f"rank_{t}"] = first_rank(ids, g) if g else None
        per_q.append(row)
    for t, res in filtered.items():
        for row, q, hits in zip(per_q, qa, res):
            g = q["gold_by_type"].get(t) or []
            row[f"rank_{t}_filtered"] = first_rank([h["unit_id"] for h in hits], g) if g else None

    by_target = {}
    for t in TYPES:
        rows = [r for r, q in zip(per_q, qa) if q["gold_by_type"].get(t)]
        if rows:
            by_target[t] = summarise_ranks([r[f"rank_{t}"] for r in rows])
            if f"rank_{t}_filtered" in rows[0]:
                by_target[t]["filtered"] = summarise_ranks([r[f"rank_{t}_filtered"] for r in rows])
    by_qtype = {}
    for t in sorted({q["qtype"] for q in qa}):
        rows = [r for r in per_q if r["qtype"] == t]
        by_qtype[t] = summarise_ranks([r["rank_any"] for r in rows])
    return {
        "k": k, "calls": calls,
        "by_target_type": by_target,
        "by_question_type": by_qtype,
        "overall_any_gold": summarise_ranks([r["rank_any"] for r in per_q]),
        "precision@5": round(sum(r["p@5"] for r in per_q) / len(per_q), 4),
        "ndcg@5": round(sum(r["ndcg@5"] for r in per_q) / len(per_q), 4),
        "per_question": per_q,
    }


# ================================================================ Layer B
JUDGE_PROMPT = """You are grading a maintenance assistant. Be strict and output JSON only.
Question: {question}
Reference answer: {gold}
Reference rationale: {context}
Assistant answer: {answer}
Source the assistant cited ({cited}):
{source}

Return {{"correct": 1 or 0, "faithful": 1 or 0, "reason": "<15 words"}} where
correct = the assistant answer agrees with the reference on the key facts (same yes/no decision,
same part/item numbers, same action); paraphrase is fine.
faithful = every factual claim in the assistant answer is supported by the cited source text."""


def judge(question, gold, context, answer, cited, source) -> dict:
    from app.openrouter import UpstreamError, get_client

    s = settings()
    try:
        out = get_client().chat(role="judge", model=s.judge_model, temperature=0.0, max_tokens=200,
                                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                                    question=question, gold=gold, context=context, answer=answer,
                                    cited=cited, source=source[:3000])}])
    except UpstreamError as e:
        return {"error": str(e)}
    m = re.search(r"\{.*\}", out, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        return {"correct": int(d.get("correct", 0)), "faithful": int(d.get("faithful", 0)),
                "reason": str(d.get("reason", ""))[:200]}
    except (ValueError, TypeError):
        return {"error": f"unparseable judge output: {out[:120]!r}"}


def layer_b(engine, qa: list[dict], use_judge: bool, pause_s: float) -> tuple[dict, list[dict]]:
    rows = []
    for i, q in enumerate(qa, 1):
        t = time.perf_counter()
        try:
            r = engine.query(q["question"])
        except RuntimeError as e:
            rows.append({"id": q["id"], "error": str(e)})
            print(f"  B {i}/{len(qa)} {q['id']} ERROR {e}")
            continue
        a = r["answer"]
        cited = a.get("cited_unit_id")
        src = next((h for h in r["sources"] if h["unit_id"] == cited), None)
        gold = set(q["gold_units"])
        gtype = q["qtype"]
        row = {
            "id": q["id"], "qtype": gtype, "status": a["status"], "answer": a["text"], "cited": cited,
            "answer_type": a.get("answer_type"), "cited_is_gold": cited in gold if cited else None,
            "artefact_correct": (cited in (q["gold_by_type"].get(gtype) or [])) if cited else None,
            "token_f1": round(token_f1(a["text"], q["answer"]), 3),
            "timings_ms": r["timings_ms"], "calls": r["calls"],
            "generation_error": a.get("generation_error"),
        }
        if use_judge and a["status"] == "answered" and not a.get("generation_error"):
            source = src["search_text"] if src else ""
            row["judge"] = judge(q["question"], q["answer"], q["context"], a["text"], a.get("citation"), source)
        rows.append(row)
        print(f"  B {i}/{len(qa)} {q['id']} {a['status']} cited={cited} gold={row['cited_is_gold']} "
              f"{r['timings_ms']['total']:.0f}ms")
        spent = time.perf_counter() - t
        if pause_s > spent:
            time.sleep(pause_s - spent)

    ok = [r for r in rows if "error" not in r]
    answered = [r for r in ok if r["status"] == "answered"]
    judged = [r for r in ok if "judge" in r and "error" not in r["judge"]]
    def mean(xs):
        xs = list(xs)
        return round(sum(xs) / len(xs), 4) if xs else None
    art = {}
    for t in TYPES:
        sub = [r for r in answered if r["qtype"] == t]
        if sub:
            art[t] = {"n": len(sub), "artefact_correct": mean(1 if r["artefact_correct"] else 0 for r in sub)}
    summary = {
        "n": len(qa), "errors": len(rows) - len(ok),
        "answered": len(answered), "abstained": len(ok) - len(answered),
        "answer_correctness_judge": mean(r["judge"]["correct"] for r in judged),
        "faithfulness_judge": mean(r["judge"]["faithful"] for r in judged),
        "judged_n": len(judged),
        "answer_token_f1": mean(r["token_f1"] for r in answered),
        "citation_accuracy": mean(1 if r["cited_is_gold"] else 0 for r in answered),
        "wrong_source_rate": mean(1 if not r["cited_is_gold"] else 0 for r in answered),
        "wrong_source_ids": [r["id"] for r in answered if not r["cited_is_gold"]],
        "artefact_correctness": mean(1 if r["artefact_correct"] else 0 for r in answered),
        "artefact_correctness_by_type": art,
        "generation_errors": sum(1 for r in ok if r.get("generation_error")),
    }
    return summary, rows


# ================================================================ Layer C
def layer_c(b_rows: list[dict], index_report: dict | None, rpm: int) -> dict:
    ok = [r for r in b_rows if "timings_ms" in r]
    ret = [r["timings_ms"]["retrieval"] for r in ok]
    gen = [r["timings_ms"]["generation"] for r in ok]
    tot = [r["timings_ms"]["total"] for r in ok]
    calls = [r["calls"].get("total", 0) for r in ok]
    cpq = sum(calls) / len(calls) if calls else None
    judge_calls = sum(1 for r in ok if "judge" in r)
    rep = index_report or {}
    return {
        "queries": len(ok),
        "latency_ms": {
            "retrieval": {"p50": round(percentile(ret, 50), 1), "p95": round(percentile(ret, 95), 1)},
            "generation": {"p50": round(percentile(gen, 50), 1), "p95": round(percentile(gen, 95), 1)},
            "total": {"p50": round(percentile(tot, 50), 1), "p95": round(percentile(tot, 95), 1)},
        },
        "calls_per_query": round(cpq, 2) if cpq is not None else None,
        "calls_per_query_note": "embedding calls served from cache count as 0; a fresh question costs embed + generate = 2",
        "judge_calls": judge_calls,
        "queries_per_minute_under_cap": round(rpm / max(cpq or 2, 1), 1),
        "queries_per_minute_fresh_questions": round(rpm / 2, 1),
        "rate_limit_rpm": rpm,
        "ingestion_seconds": (rep.get("timings") or {}).get("total_s"),
        "ingestion_calls": rep.get("model_calls"),
        "index_size_bytes": rep.get("index_size_bytes"),
        "artefacts_size_bytes": rep.get("artefacts_size_bytes"),
        "units_indexed": rep.get("units_indexed"),
    }


# ================================================================ report
def fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def to_markdown(rep: dict) -> str:
    L = [f"# Evaluation report: {rep['label']}", "", f"- run at: {rep['run_at']}",
         f"- index: `{json.dumps(rep.get('index'))}`", ""]
    if "A" in rep:
        a = rep["A"]
        L += ["## Layer A: retrieval by type (generator out of the loop)", "",
              f"k={a['k']}, embedding calls this run: {a['calls']}", "",
              "| Type (gold target) | n | Recall@1 | Recall@3 | Recall@5 | MRR | Stage |", "|---|---|---|---|---|---|---|"]
        for t in TYPES:
            m = a["by_target_type"].get(t)
            if m:
                L.append(f"| {t} | {m['n']} | {m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | "
                         f"{m['mrr']:.3f} | {rep['label']} |")
            else:
                L.append(f"| {t} | 0 | - | - | - | - | no gold of this type in the set |")
        L += ["", "Type-filtered search (same embeddings, filter applied inside the vector search):", "",
              "| Type | Recall@1 | Recall@5 | MRR |", "|---|---|---|---|"]
        for t, m in a["by_target_type"].items():
            f = m.get("filtered")
            if f:
                L.append(f"| {t} | {f['recall@1']:.3f} | {f['recall@5']:.3f} | {f['mrr']:.3f} |")
        L += ["", "By question type (any gold unit):", "", "| Question type | n | R@1 | R@5 | MRR |", "|---|---|---|---|---|"]
        for t, m in a["by_question_type"].items():
            L.append(f"| {t} | {m['n']} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['mrr']:.3f} |")
        L += ["", f"precision@5 = {a['precision@5']}, nDCG@5 = {a['ndcg@5']}", "",
              "### Misses (a gold target missing from the top 5)", ""]
        for t in ("figure", "table", "text"):
            miss = [r for r in a["per_question"] if t in r["gold_types"] and
                    (r.get(f"rank_{t}") is None or r[f"rank_{t}"] > 5)]
            if not miss:
                continue
            L.append(f"**{t}** ({len(miss)}): " + ", ".join(r["id"] for r in miss))
            for r in miss[:3]:
                L.append(f"- {r['id']}: {r['question'][:110]}")
                L.append(f"  - gold: {r['gold']} | top: " + ", ".join(f"{h['unit_id']}({h['score']})" for h in r["top"][:3]))
            L.append("")
    if "compare" in rep:
        c = rep["compare"]
        L += ["## Change vs " + c["against"], "", "| Type | Metric | Before | After | Delta |", "|---|---|---|---|---|"]
        for row in c["deltas"]:
            L.append(f"| {row['type']} | {row['metric']} | {row['before']:.3f} | {row['after']:.3f} | {row['delta']:+.3f} |")
        L += ["", f"Improved: {', '.join(c['improved']) or 'none'}", f"Worsened: {', '.join(c['worsened']) or 'none'}", ""]
    if "B" in rep:
        b = rep["B"]
        L += ["## Layers B and C (T4)", "", "| Layer | Metric | Value | How measured / sample size |", "|---|---|---|---|",
              f"| B | Answer correctness | {fmt(b['answer_correctness_judge'])} (token F1 {fmt(b['answer_token_f1'])}) | LLM judge `{rep.get('judge_model')}` temp 0, n={b['judged_n']}; F1 vs gold answer |",
              f"| B | Faithfulness (judge) | {fmt(b['faithfulness_judge'])} | judge vs cited source text, n={b['judged_n']}; five-answer UI check is manual |",
              f"| B | Citation accuracy | {fmt(b['citation_accuracy'])} | cited unit in gold units, answered n={b['answered']} |",
              f"| B | Wrong-source rate | {fmt(b['wrong_source_rate'])} | answered but cited a non-gold unit: {', '.join(b['wrong_source_ids']) or 'none'} |",
              f"| B | Artefact correctness | {fmt(b['artefact_correctness'])} | returned artefact is the gold unit of the question's type; by type {b['artefact_correctness_by_type']} |",
              f"| B | Abstained | {b['abstained']} / {b['n']} | generation errors: {b['generation_errors']} |"]
    if "C" in rep:
        c = rep["C"]
        lat = c["latency_ms"]
        L += [f"| C | Latency p50 / p95 retrieval | {lat['retrieval']['p50']} / {lat['retrieval']['p95']} ms | n={c['queries']} |",
              f"| C | Latency p50 / p95 generation | {lat['generation']['p50']} / {lat['generation']['p95']} ms | n={c['queries']} |",
              f"| C | Latency p50 / p95 total | {lat['total']['p50']} / {lat['total']['p95']} ms | |",
              f"| C | Calls per query | {c['calls_per_query']} | {c['calls_per_query_note']} |",
              f"| C | Ingestion time and index size | {c['ingestion_seconds']} s, {c['index_size_bytes']} B index, {c['artefacts_size_bytes']} B artefacts | calls: {c['ingestion_calls']} |",
              f"| C | Queries per minute under the cap | {c['queries_per_minute_under_cap']} (fresh questions: {c['queries_per_minute_fresh_questions']}) | cap {c['rate_limit_rpm']} req/min |"]
    return "\n".join(L) + "\n"


def compare(cur: dict, base: dict) -> dict:
    deltas, improved, worsened = [], [], []
    for t, m in cur["A"]["by_target_type"].items():
        bm = base.get("A", {}).get("by_target_type", {}).get(t)
        if not bm:
            continue
        for key in ("recall@1", "recall@3", "recall@5", "mrr"):
            deltas.append({"type": t, "metric": key, "before": bm[key], "after": m[key], "delta": m[key] - bm[key]})
    bq = {r["id"]: r for r in base.get("A", {}).get("per_question", [])}
    for r in cur["A"]["per_question"]:
        b = bq.get(r["id"])
        if not b:
            continue
        for t in TYPES:
            x, y = b.get(f"rank_{t}"), r.get(f"rank_{t}")
            if x == y:
                continue
            x2, y2 = x or 999, y or 999
            (improved if y2 < x2 else worsened).append(f"{r['id']}:{t}({x or '-'}->{y or '-'})")
    return {"against": base.get("label", "?"), "deltas": deltas, "improved": improved, "worsened": worsened}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="A")
    ap.add_argument("--label", default="run")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--local", action="store_true", help="run the pipeline in-process instead of via the API")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample", type=int, default=0, help="Layer B/C: random sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--pause", type=float, default=None, help="seconds per Layer B question (default from RPM)")
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--qa", type=Path, default=None)
    args = ap.parse_args()

    s = settings()
    layers = {x.strip().upper() for x in args.layers.split(",")}
    qa = json.loads((args.qa or s.qa_path).read_text(encoding="utf-8"))
    engine = LocalEngine() if args.local else ApiEngine(args.api)
    health = engine.health()
    rep: dict = {"label": args.label, "run_at": datetime.now().isoformat(timespec="seconds"),
                 "index": health.get("index"), "judge_model": None if args.no_judge else s.judge_model,
                 "n_questions": len(qa)}
    if "A" in layers:
        print(f"Layer A on {len(qa)} questions ...")
        rep["A"] = layer_a(engine, qa, args.k)
        if args.compare:
            rep["compare"] = compare(rep, json.loads(args.compare.read_text(encoding="utf-8")))
    b_rows: list[dict] = []
    if layers & {"B", "C"}:
        sample = list(qa)
        if args.sample:
            random.Random(args.seed).shuffle(sample)
            sample = sorted(sample[: args.sample], key=lambda q: q["id"])
        calls_per_q = 2 + (0 if args.no_judge else 1)
        pause = args.pause if args.pause is not None else 60.0 / s.requests_per_minute * calls_per_q
        print(f"Layer B on {len(sample)} questions (pause {pause:.1f}s/question) ...")
        rep["B"], b_rows = layer_b(engine, sample, not args.no_judge, pause)
        rep["B_per_question"] = b_rows
        rep["B"]["sample_ids"] = [q["id"] for q in sample]
    if "C" in layers:
        from app.pipeline import read_index_report

        rep["C"] = layer_c(b_rows, read_index_report(s), s.requests_per_minute)

    s.reports_dir.mkdir(parents=True, exist_ok=True)
    stem = s.reports_dir / f"{args.label}"
    stem.with_suffix(".json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = to_markdown(rep)
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"saved {stem.with_suffix('.json')} and .md")


if __name__ == "__main__":
    main()
