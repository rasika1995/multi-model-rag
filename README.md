# Rich Answers, Not Just Text: multimodal retrieval over technical manuals

Course 3 · Day 2 · Lab A. Ask a maintenance question and get back the **real figure image or the exact
table rows**, with a `document > page > unit` citation and a sources panel. Retrieve by text, answer
with the artefact.

```
rag/   Python service: ingest > enrich > embed > index > retrieve > answer, plus evaluation
web/   Next.js (App Router, TypeScript) UI: a thin client over the API
```

The recommended setup runs **Qdrant** (vector DB), the **API** (`rag/`) and the **UI** (`web/`) in Docker.
You can also run the API and UI directly on Windows without Docker; that setup uses the API's persistent
in-memory vector store instead of Qdrant. Corpus files, artefacts, caches and reports are stored in
`rag/data`; Docker stores its Qdrant index in a named volume, while the no-Docker index files are in
`rag/data`.

| Service | URL |
|---|---|
| UI | <http://127.0.0.1:3000> |
| API health | <http://127.0.0.1:8000/health> |
| Qdrant dashboard | <http://127.0.0.1:6333/dashboard> |

> **Use `127.0.0.1`, not `localhost`.** On Windows, `localhost` can resolve to IPv6 `::1`, and another app
> (for example one running in WSL) may be listening there on ports 3000 and 8000. If using Docker, don't
> also run `uvicorn` or `npm run dev` on the host: they use the same ports as the containers.

## Prerequisites

- Docker Desktop, running
- An OpenRouter key from <https://openrouter.ai/keys>. Under 10 credits bought you get **50 free calls a
  day**; a full lab run needs about 200 to 300. Top up 10 credits (free models don't spend them) or share
  one funded key.

## First time only

```powershell
copy rag\.env.example rag\.env     # then paste your key into OPENROUTER_API_KEY and set VECTOR_BACKEND=qdrant
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
# ...change ONE lever in rag/.env (see "Stage 4 levers"), apply it (table above), then:
docker compose run --rm api python -m eval.run_eval --api http://api:8000 --layers A --label fix1 --compare data/reports/baseline.json
docker compose run --rm api python -m eval.run_eval --api http://api:8000 --layers B,C --sample 20 --label dev
docker compose run --rm api python -m eval.run_eval --api http://api:8000 --layers A,B,C --label final
```

Reports land in `rag/data/reports/<label>.json|.md` with per-question results, the T2 and T4 tables, and
the list of misses for Stage 4 diagnosis. Open the `.md` file to read the results. Layer A never calls the
generator.

Things to know:

- **The eval does not change any tracked files.** `rag/data/` (reports included) and `rag/.env` are in
  `.gitignore`, so `git status` stays clean after a run. Look in `rag/data/reports/` for the output.
- **The same settings give the same numbers.** Re-running without changing `rag/.env` or the index
  reproduces the previous scores, because the embeddings are cached.
- **Use a new `--label` for every run.** Reusing a label overwrites that report.
- **Record which lever you changed for each label**, because `rag/.env` is not committed. The report header
  shows the index settings (`table_context`, `lexical_weight`, ...).
- **Tag your code only when you change Python code** between runs, so you can return to the version that
  produced a given report. The tag keeps code only, not reports or `.env`:

  ```powershell
  git add . ; git commit -m "Layer A baseline" ; git tag layer-a-baseline
  ```

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

This Windows/PowerShell setup runs the API and UI as local processes and uses `VECTOR_BACKEND=memory`.
The memory store persists its index files under `rag/data/index`; you do not need to start Qdrant or
install Docker.

#### Prerequisites

- Python 3.11 or newer (check with `python --version`)
- Node.js 20+ (includes `npm`)
- An OpenRouter key from <https://openrouter.ai/keys>

If you have already started the Docker stack, stop it first from the repository root with
`docker compose down` so ports 3000 and 8000 are available. You do not need this command if you have not
started Docker.

#### First-time setup

Open PowerShell in the repository root. Create the local environment files and edit them:

```powershell
copy rag\.env.example rag\.env
code rag\.env
```

In `rag\.env`, set `OPENROUTER_API_KEY` to your key, keep `VECTOR_BACKEND=memory`, and set the CORS
origins to:

```text
CORS_ORIGINS=http://127.0.0.1:3000,http://localhost:3000
```

Create and activate the Python environment, then install the dependencies:

```powershell
cd rag
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell says running scripts is disabled, use a Command Prompt (`cmd`) window instead; the same
commands work there.

Download the sample corpus, create its question set, check model access and the local store, then build
the index. Indexing uses OpenRouter and may make model calls; the first build also downloads the dataset.
The index, corpus, and generated files are kept in `rag\data`.

```powershell
python scripts\download_data.py --tier a
python scripts\build_gold.py
python scripts\check_models.py
python scripts\check_store.py
python scripts\build_index.py
python -m pytest -q
```

Start the API in this PowerShell window and leave it running:

```powershell
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Open a **second** PowerShell window at the repository root. Configure and start the UI:

```powershell
cd web
copy .env.example .env.local
code .env.local
npm install
npm run dev
```

In `web\.env.local`, set `NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000`. Open
<http://127.0.0.1:3000>; the status badge should say **Healthy**. You can also check the API at
<http://127.0.0.1:8000/health>.

#### Starting and stopping later

After the one-time setup, open two PowerShell windows from the repository root:

```powershell
# Window 1: start the API
cd rag
.venv\Scripts\activate
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

```powershell
# Window 2: start the UI
cd web
npm run dev
```

Stop each process with **Ctrl+C** in its own window. If you change Python code or `rag\.env`, restart
the API. If you change the index inputs or embedding/caption settings, stop the API, then from the
`rag\` terminal with the virtual environment activated, run `python scripts\build_index.py` and start
the API again.

#### Run evaluation without Docker

Keep the API running, then use a third PowerShell window:

```powershell
cd rag
.venv\Scripts\activate
python -m eval.run_eval --api http://127.0.0.1:8000 --layers A --label baseline
```

For any API command shown elsewhere as `docker compose run --rm api python ...`, run the script from
`rag\` with the virtual environment activated and `python` instead. For evaluation commands, use
`http://127.0.0.1:8000` as the API address instead of Docker's `http://api:8000`.

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
