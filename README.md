# Rich Answers, Not Just Text: multimodal retrieval over technical manuals

Course 3 · Day 2 · Lab A. Ask a maintenance question and get back the **real figure image or the exact
table rows**, with a `document > page > unit` citation and a sources panel. Retrieve by text, answer
with the artefact.

```
rag/   Python service: ingest > enrich > embed > index > retrieve > answer, plus evaluation
web/   Next.js (App Router, TypeScript) UI: a thin client over the API
```

Everything runs in Docker: **Qdrant** (vector DB), the **API** (`rag/`) and the **UI** (`web/`). You use
one terminal, in the repo root, and run every job (index build, evaluation) as a one-off container.
Files the jobs write (corpus, index report, caches, eval reports) land in `rag/data` on your machine.

| Service | URL |
|---|---|
| UI | <http://127.0.0.1:3000> |
| API health | <http://127.0.0.1:8000/health> |
| Qdrant dashboard | <http://127.0.0.1:6333/dashboard> |

> **Use `127.0.0.1`, not `localhost`.** On Windows, `localhost` can resolve to IPv6 `::1`, and another app
> (for example one running in WSL) may be listening there on ports 3000 and 8000. Don't run `uvicorn` or
> `npm run dev` on the host at the same time: they take the same ports as the containers.

## Prerequisites

- Docker Desktop, running
- An OpenRouter key from <https://openrouter.ai/keys>. Under 10 credits bought you get **50 free calls a
  day**; a full lab run needs about 200 to 300. Top up 10 credits (free models don't spend them) or share
  one funded key.

## First time only

```powershell
copy rag\.env.example rag\.env          # then paste your key into OPENROUTER_API_KEY and set VECTOR_BACKEND=qdrant
docker compose up -d --build            # builds and starts qdrant, api, web (a few minutes the first time)

docker compose run --rm api python scripts/download_data.py --tier a   # Enginuity Task 2, ~6.6 MB
docker compose run --rm api python scripts/build_gold.py               # data/qa.json: 60 gold questions
docker compose run --rm api python scripts/check_models.py             # T1: OK/FAIL per model role
docker compose run --rm api python scripts/check_store.py              # Qdrant create/write/search/delete
docker compose run --rm api python scripts/build_index.py              # ~12 caption + 2 embedding calls, cached after
docker compose run --rm api python -m pytest -q                        # wiring tests, no network
```

Open <http://127.0.0.1:3000>. The status badge (top right) should say **Healthy**.

## Every day after that

```powershell
docker compose up -d                    # start (the index is kept in the qdrant volume)
docker compose down                     # stop
docker compose ps                       # what is running
docker compose logs -f api              # API logs (use web or qdrant for the others)
```

## When you change something

| You changed | Run |
|---|---|
| `rag/.env`, a setting that needs **no** rebuild (e.g. `LEXICAL_WEIGHT`, `MIN_SCORE`, `TOP_K`) | `docker compose up -d --force-recreate api` |
| `rag/.env`, a setting that **does** need a rebuild (model, captions, table context, corpus) | `docker compose up -d --force-recreate api` then `docker compose run --rm api python scripts/build_index.py` |
| Python code in `rag/` | `docker compose up -d --build api` |
| UI code in `web/`, or the API address in `docker-compose.yml` | `docker compose up -d --build web`, then **Ctrl+F5** in the browser |

## Evaluate

```powershell
docker compose run --rm api python -m eval.run_eval --api http://api:8000 --layers A --label baseline
git add . ; git commit -m "Layer A baseline" ; git tag layer-a-baseline
# ...change ONE lever in rag/.env (see "Stage 4 levers"), apply it (table above), then:
docker compose run --rm api python -m eval.run_eval --api http://api:8000 --layers A --label fix1 --compare data/reports/baseline.json
docker compose run --rm api python -m eval.run_eval --api http://api:8000 --layers B,C --sample 20 --label dev
docker compose run --rm api python -m eval.run_eval --api http://api:8000 --layers A,B,C --label final
```

Reports land in `rag/data/reports/<label>.json|.md` with per-question results, the T2 and T4 tables, and
the list of misses for Stage 4 diagnosis. Layer A never calls the generator.

API endpoints: `POST /query`, `POST /retrieve`, `POST /retrieve/batch`, `GET /health`, `GET /examples`,
`GET /artefact/{unit_id}`, `GET /units/{unit_id}`. Configuration problems return 503 with
`error.kind` = `configuration | index_missing | index_mismatch`; model problems return 502 with
`error.kind` = `upstream | model_unavailable` and the model slug.

### Text gold (Tier C, ViDoRe v3 Industrial)

Enginuity has no prose questions, so text recall needs ViDoRe (about 20 extra caption calls):

```powershell
docker compose run --rm api python scripts/download_data.py --tier vidore
docker compose run --rm api python scripts/build_gold_vidore.py --plan --docs TO_00-25-172CL-1,TO_00-25-172CL-3
# paste the printed CORPORA and PDF_PATHS lines into rag/.env, then:
docker compose up -d --force-recreate api
docker compose run --rm api python scripts/build_index.py
docker compose run --rm api python scripts/build_gold_vidore.py --build --docs TO_00-25-172CL-1,TO_00-25-172CL-3
docker compose run --rm api python -m eval.run_eval --api http://api:8000 --layers A --label baseline-all --qa data/qa_all.json
```

### Without Docker (optional)

Python 3.11+ and Node 20+: in `rag/`, create a venv, `pip install -r requirements.txt`, set
`VECTOR_BACKEND=memory`, then `python scripts/build_index.py` and `uvicorn app.server:app --port 8000`;
in `web/`, `npm install` and `npm run dev`. Stop the Docker containers first so the ports are free.

### Behind a TLS-inspecting proxy

If `docker compose build` fails with SSL certificate errors, export your corporate root CA as PEM into
`rag/docker-certs/` and `web/docker-certs/` (any `*.crt` name; git-ignored) and build again.

## How it maps to the brief

| Stage / epic | Where |
|---|---|
| E0 config, model check, vector DB | `app/config.py`, `app/diagnostics.py`, `scripts/check_models.py`, `scripts/check_store.py` |
| E1 routing + OCR gate | `app/ingest.py` (`PdfLoader`: blank / native / mixed / scanned / scanned-with-text-layer; `gate()`; review queue in `data/review_queue.json`) |
| E2 figure + table extraction with provenance | `app/ingest.py` (raster images and vector-drawing clusters rendered to PNG; tables kept as columns + rows) |
| E3 captions, linearisation, rejects | `app/enrich.py` (cached by image hash + prompt + model, reasoning off, fallback models, generic-caption check) |
| E4 index, provenance fence, model record | `app/store.py` (memory + Qdrant), `app/pipeline.py`, `data/index_report.json` |
| E5 retrieval + API | `app/retrieve.py` (type filter inside the search, optional BM25 fusion), `app/server.py` |
| E6 answer with the artefact, abstain | `app/generate.py` |
| E7 UI | `web/components/*` |
| E8 evaluation | `eval/run_eval.py`, `eval/metrics.py` |
| E9 notes | `docs/REPORT.md` |

## Stage 4 levers (change one at a time)

| Lever | Setting | Rebuild? |
|---|---|---|
| Richer figure captions (model/variant terms) | `CAPTION_PROMPT_VERSION=v2` | yes |
| Table text gets its figure's title ("nearby heading") | `TABLE_CONTEXT=paired_figure` | yes |
| BM25 fusion for exact part numbers | `LEXICAL_WEIGHT=0.3` | no |
| Embed figure images instead of captions | `EMBED_IMAGES=true` | yes |
| Larger embedder | `EMBED_MODEL=nvidia/nemotron-3.5-embed-8b:free` | yes |
| Abstention threshold | `MIN_SCORE=0.25` | no |

## Data handling

Free endpoints are for **public datasets only**. Never send confidential manuals to them; production would
use a private or self-hosted captioner and embedder so the bytes stay on your side. See `docs/REPORT.md`.
