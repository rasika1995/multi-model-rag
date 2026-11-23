# Lab A evidence report: multimodal retrieval over technical manuals

Fill this in as you go. Numbers come from `rag/data/index_report.json`, `rag/data/model_check.json` and
`rag/data/reports/*.md` (the evaluator writes T2 and T4 rows for you).

## Stage 0: setup

### T1 Model readiness log

Paste the table printed by `python scripts/check_models.py`.

| Role | Slug | Call OK? | Time (ms) | Notes / error |
|---|---|---|---|---|
| Embeddings | | | | |
| Vision / captioning | | | | |
| Text generation | | | | |
| Figure answers | | | | |

Vector DB: [ ] Qdrant dashboard screenshot, or [ ] in-memory fallback confirmed (`scripts/check_store.py`).

### Prediction (write before any metric is run)

Which question type (text, figure or table) will retrieve worst, and why?

> _Your prediction here._

## Stage 1: routing

`data-7.pdf` (182 pages, TM 55-2840-241-23P), router run with OCR off:

| Page class | Pages |
|---|---|
| blank | 20 |
| native (text only) | 14 |
| mixed (text + figure/table) | 116 |
| scanned (image-only, no text layer) | 31 |
| scanned with a hidden text layer | 1 |

Regions emitted: 126 text chunks, 90 tables, 65 figures (raster images and clusters of vector drawings,
rendered as PNG crops). With OCR off, all 31 scanned pages went to the review queue and none were indexed.
Re-run with `OCR_ENGINE=vision` (or tesseract) to OCR them behind the gate, and record how many fail.

Observed extraction issue to keep in mind: on the parts-index pages, the table extractor merges several
printed rows into one cell (one cell holds five part numbers). That is the "right table, wrong value"
failure mode.

**First loader you would have reached for, and what it would drop** (two sentences):

> _e.g. "PyPDFLoader / page.get_text() would have produced one flat string per page: the 65 diagrams would
> vanish (they are vector drawings or images with no text) and the parts tables would lose their columns,
> so 'item 28' could no longer be tied to its part number."_

## Stage 2: enrichment

- Caption cache: `rag/data/cache/captions/` (one JSON per image + prompt + model)
- Rejected units with reasons: `rejected` in `rag/data/index_report.json`
- Figure baseline decision: **caption text** (`EMBED_IMAGES=false`), because ... _(your reason)_

Five-unit provenance spot-check:

| Unit id | Type | Document > page | Opened the source: correct page? | Caption names the system? |
|---|---|---|---|---|
| | | | | |

## Stage 3: Layer A baseline

### T2 Layer A: retrieval by type

| Type | n | Recall@1 | Recall@3 | Recall@5 | MRR | Stage |
|---|---|---|---|---|---|---|
| Text | | | | | | Baseline |
| Text | | | | | | After fix 1 |
| Text | | | | | | After fix 2 |
| Text | | | | | | Final |
| Figure | | | | | | Baseline |
| Figure | | | | | | After fix 1 |
| Figure | | | | | | After fix 2 |
| Figure | | | | | | Final |
| Table | | | | | | Baseline |
| Table | | | | | | After fix 1 |
| Table | | | | | | After fix 2 |
| Table | | | | | | Final |

Notes on the gold:
- Enginuity Task 2: each question's gold is its figure **and** its table(s), so figure and table recall are
  reported separately on the same 60 questions. Only 12 distinct figures sit behind 60 questions, so Layer A
  is coarse. Its table TSVs were produced by an LLM (only the 50-figure Task 1 set was human-verified).
- Enginuity has no page numbers: each figure group is one document, the figure is page 1 and its tables are
  pages 2..n. Stable and resolvable, but synthetic; say so when you report.
- Text recall comes from ViDoRe v3 Industrial (tier C). A qrel whose content type has no extracted unit on
  its page counts as a miss (`missing:` gold id), so extraction gaps show up in recall.

## Stage 4: failure modes (graded heaviest)

### T3 Failure-mode log

| # | Symptom and question IDs | Root cause | Change made (one lever) | Before | After |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |

Cost lever applied and calls saved: _..._

## Stage 5: answer UI

- [ ] Screenshot: figure answer with citation and sources panel
- [ ] Screenshot: table answer with highlighted rows, citation and sources panel

Five-answer faithfulness check (through the UI):

| Question id | Artefact shown | Gold artefact | Same? | Reads well but wrong figure? |
|---|---|---|---|---|
| | | | | |

## Stage 6: Layers B and C

### T4 summary

Copy from `rag/data/reports/<label>.md`.

| Layer | Metric | Value | How measured / sample size |
|---|---|---|---|
| B | Answer correctness | | |
| B | Faithfulness (five-answer check) | | |
| B | Citation accuracy | | |
| B | Wrong-source rate | | |
| B | Artefact correctness | | |
| C | Latency p50 / p95 (retrieval, generation) | | |
| C | Calls per query | | |
| C | Ingestion time and index size | | |
| C | Queries per minute under the cap | | |

## Stage 7: wrap

### Provenance and data-handling note

**How provenance travels.** A unit is stamped `document > page > unit id` when it is extracted
(`app/ingest.py`), together with a bounding box and source path for PDF regions. Units that cannot be
stamped are rejected at extraction. Enrichment copies the unit and never changes its provenance. The vector
store refuses any unit missing document, page or unit id at write time and logs it (`app/store.py`), and
the payload carries `document_id`, `page`, `unit_id` and `type` as filterable fields. At answer time, hits
without resolvable provenance are dropped before the answer type is chosen. If none are left, the system
abstains and says why (`app/generate.py`). Citations the model invents are stripped and flagged. The UI
shows the citation under every answer, and the sources panel lists every retrieved unit with its
provenance, marking unsourced hits as "dropped, never cited".

**OCR-failed pages.** Scanned pages are OCR'd and scored (garbled-token ratio, length, and Tesseract
confidence when available). Pages below the gate go to `data/review_queue.json` with the reason and are
never indexed. The count is in the index report.

**What to redact or restrict in a real manual set.** _(your answer: export-controlled data, serial numbers,
customer or site identifiers, personal data in sign-off blocks; access control per document; never
send them to a free shared endpoint. Production uses a private or self-hosted captioner and embedder so
the bytes stay in your tenancy.)_

### T5 demo checklist

- [ ] Text question: answer, citation and sources panel shown.
- [ ] Figure question: the actual figure image shown with page citation.
- [ ] Table question: the exact rows shown with page citation.
- [ ] A question the system should refuse: it abstains and says why.
- [ ] One known weakness shown honestly, with the number that proves it.
- [ ] Status badge shown healthy; one degraded state demonstrated (for example a wrong model slug).

### One-page reflection

Cover: enrichment and UI choices; the change that moved figure and table recall most; latency and cost
reality; your position on the grounding ladder (prompt, retrieve, fine-tune, distil); Stage 0 prediction
versus result; one limitation a polished demo would hide.
