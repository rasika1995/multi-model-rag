"""HTTP API used by both the UI and the evaluator.

POST /query  POST /retrieve  POST /retrieve/batch  GET /health  GET /examples
GET /artefact/{unit_id}  GET /units/{unit_id}

Errors are distinct: configuration problems -> 503 {"kind": "configuration"|...},
upstream model problems -> 502 {"kind": "upstream"|"model_unavailable", "model": ...}.

Run: uvicorn app.server:app --reload --port 8000   (from the rag/ folder)
"""
from __future__ import annotations

import json
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import ConfigError, settings
from .openrouter import UpstreamError
from .retrieve import IndexMismatch, IndexMissing
from .schemas import QueryRequest, QueryResponse, RetrieveBatchRequest
from .service import QueryService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Multimodal manual RAG", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=list(settings().cors_origins), allow_methods=["*"],
                   allow_headers=["*"])

_service: QueryService | None = None


def service() -> QueryService:
    global _service
    if _service is None:
        _service = QueryService()
    return _service


@app.exception_handler(ConfigError)
async def _config_error(_: Request, e: ConfigError):
    kind = "index_mismatch" if isinstance(e, IndexMismatch) else "index_missing" if isinstance(e, IndexMissing) \
        else "configuration"
    return JSONResponse(status_code=503, content={"error": {"kind": kind, "message": str(e)}})


@app.exception_handler(UpstreamError)
async def _upstream_error(_: Request, e: UpstreamError):
    return JSONResponse(status_code=502, content={"error": {"kind": e.kind, "model": e.model, "message": str(e)}})


@app.get("/health")
def health():
    return service().health()


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return service().query(req.question, req.k, req.type_filter, req.generate)


@app.post("/retrieve")
def retrieve(req: QueryRequest):
    hits, timings, calls = service().retrieve(req.question, req.k, req.type_filter)
    return {"question": req.question, "hits": [h.model_dump() for h in hits], "timings_ms": timings, "calls": calls}


@app.post("/retrieve/batch")
def retrieve_batch(req: RetrieveBatchRequest):
    res, timings, calls = service().retrieve_batch(req.questions, req.k, req.type_filter)
    return {"results": [[h.model_dump() for h in hits] for hits in res], "timings_ms": timings, "calls": calls}


DEFAULT_EXAMPLES = [
    {"question": "Show the turbocharger assembly diagram", "type": "figure"},
    {"question": "What is the part number for the oil filter element?", "type": "table"},
    {"question": "Which items in the water pump parts list are gaskets?", "type": "table"},
    {"question": "What is the torque for the flux capacitor on a 1985 DeLorean?", "type": "abstain"},
]


@app.get("/examples")
def examples():
    s = settings()
    out = list(DEFAULT_EXAMPLES)
    if s.qa_path.exists():
        qa = json.loads(s.qa_path.read_text(encoding="utf-8"))
        picked: dict[str, dict] = {}
        for q in qa:
            picked.setdefault(q["qtype"], {"question": q["question"], "type": q["qtype"], "id": q["id"]})
        out = list(picked.values()) + out
    return {"examples": out[:6]}


@app.get("/units/{unit_id}")
def unit(unit_id: str):
    u = service().retriever.store.get(unit_id)
    if u is None:
        raise HTTPException(404, f"unknown unit {unit_id}")
    return u.model_dump()


@app.get("/artefact/{unit_id}")
def artefact(unit_id: str):
    try:
        p = service().artefact_path(unit_id)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except FileNotFoundError:
        raise HTTPException(404, f"no artefact file for unit {unit_id}")
    return FileResponse(p)
