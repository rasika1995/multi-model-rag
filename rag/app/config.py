"""Single source of truth for configuration.

Nothing else in the code base reads the environment: every module asks
`get_settings()`. Values come from the process environment first, then from
`rag/.env` (untracked). Model slugs are *always* configuration, never code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values

RAG_ROOT = Path(__file__).resolve().parent.parent  # .../rag


class ConfigError(RuntimeError):
    """A local configuration problem (missing key, bad value, stale index)."""


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _int_or_none(v: str | None) -> int | None:
    if v is None or str(v).strip() == "":
        return None
    return int(v)


@dataclass(frozen=True)
class Settings:
    # --- OpenRouter -------------------------------------------------------
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embed_model: str = "nvidia/nemotron-3.5-embed-1b:free"
    embed_dimensions: int | None = 1024
    embed_images: bool = False  # stretch: embed figure images instead of captions
    embed_batch_size: int = 16
    vision_model: str = "nvidia/nemotron-3.5-super-vl:free"
    text_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    figure_model: str = "nvidia/nemotron-3.5-super-vl:free"
    judge_model: str = "nvidia/nemotron-3.5-lightning:free"
    vision_fallback_models: tuple[str, ...] = ("google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free")
    disable_reasoning: bool = True
    requests_per_minute: int = 20
    max_retries: int = 4
    request_timeout_s: float = 120.0

    # --- Vector store ------------------------------------------------------
    vector_backend: str = "memory"  # "qdrant" | "memory"
    qdrant_url: str = "http://localhost:6333"
    collection: str = "manuals"

    # --- Corpus and ingestion ---------------------------------------------
    data_dir: Path = field(default_factory=lambda: RAG_ROOT / "data")
    corpora: tuple[str, ...] = ("enginuity_t2",)
    pdf_paths: tuple[str, ...] = ()
    ocr_engine: str = "auto"  # auto | tesseract | vision | none
    ocr_max_garbled: float = 0.35
    ocr_min_confidence: float = 60.0
    ocr_min_chars: int = 40
    min_figure_px: int = 120
    min_figure_area: float = 0.04  # fraction of the page area
    text_chunk_chars: int = 900

    # --- Enrichment / retrieval levers (Stage 4 "change one thing") --------
    caption_prompt_version: str = "v1"
    table_context: str = "none"  # none | paired_figure
    lexical_weight: float = 0.0  # 0 = vector only; >0 fuses BM25
    min_score: float = 0.0  # abstain when the top sourced score is below this
    top_k: int = 5
    figure_max_side: int = 1600

    # --- API ---------------------------------------------------------------
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)

    # ---------------------------------------------------------------------
    @property
    def artefact_dir(self) -> Path:
        return self.data_dir / "artefacts"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def index_report_path(self) -> Path:
        return self.data_dir / "index_report.json"

    @property
    def qa_path(self) -> Path:
        return self.data_dir / "qa.json"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    def require_key(self) -> str:
        key = self.openrouter_api_key.strip()
        if not key:
            raise ConfigError(
                "OPENROUTER_API_KEY is not set. Put it in rag/.env (see .env.example) "
                "or export it. Create one at https://openrouter.ai/keys"
            )
        if key != self.openrouter_api_key or key.startswith(("'", '"')):
            raise ConfigError("OPENROUTER_API_KEY has surrounding spaces or quotes; remove them.")
        return key

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **kw)


def load_settings(env_file: Path | None = None) -> Settings:
    env_file = env_file or (RAG_ROOT / ".env")
    file_vals = dotenv_values(env_file) if env_file.exists() else {}

    def get(name: str, default=None):
        v = os.environ.get(name)
        if v is None:
            v = file_vals.get(name)
        return default if v is None else v

    d = Settings()
    data_dir = Path(get("DATA_DIR", str(d.data_dir)))
    if not data_dir.is_absolute():
        data_dir = (RAG_ROOT / data_dir).resolve()

    def csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        v = get(name)
        if v is None:
            return default
        return tuple(x.strip() for x in v.split(",") if x.strip())

    return Settings(
        openrouter_api_key=get("OPENROUTER_API_KEY", ""),
        openrouter_base_url=get("OPENROUTER_BASE_URL", d.openrouter_base_url),
        embed_model=get("EMBED_MODEL", d.embed_model),
        embed_dimensions=_int_or_none(get("EMBED_DIMENSIONS", str(d.embed_dimensions or ""))),
        embed_images=_bool(get("EMBED_IMAGES"), d.embed_images),
        embed_batch_size=int(get("EMBED_BATCH_SIZE", d.embed_batch_size)),
        vision_model=get("VISION_MODEL", d.vision_model),
        text_model=get("TEXT_MODEL", d.text_model),
        figure_model=get("FIGURE_MODEL", d.figure_model),
        judge_model=get("JUDGE_MODEL", d.judge_model),
        vision_fallback_models=csv("VISION_FALLBACK_MODELS", d.vision_fallback_models),
        disable_reasoning=_bool(get("DISABLE_REASONING"), d.disable_reasoning),
        requests_per_minute=int(get("REQUESTS_PER_MINUTE", d.requests_per_minute)),
        max_retries=int(get("MAX_RETRIES", d.max_retries)),
        request_timeout_s=float(get("REQUEST_TIMEOUT_S", d.request_timeout_s)),
        vector_backend=get("VECTOR_BACKEND", d.vector_backend).lower(),
        qdrant_url=get("QDRANT_URL", d.qdrant_url),
        collection=get("COLLECTION", d.collection),
        data_dir=data_dir,
        corpora=csv("CORPORA", d.corpora),
        pdf_paths=csv("PDF_PATHS", d.pdf_paths),
        ocr_engine=get("OCR_ENGINE", d.ocr_engine).lower(),
        ocr_max_garbled=float(get("OCR_MAX_GARBLED", d.ocr_max_garbled)),
        ocr_min_confidence=float(get("OCR_MIN_CONFIDENCE", d.ocr_min_confidence)),
        ocr_min_chars=int(get("OCR_MIN_CHARS", d.ocr_min_chars)),
        min_figure_px=int(get("MIN_FIGURE_PX", d.min_figure_px)),
        min_figure_area=float(get("MIN_FIGURE_AREA", d.min_figure_area)),
        text_chunk_chars=int(get("TEXT_CHUNK_CHARS", d.text_chunk_chars)),
        caption_prompt_version=get("CAPTION_PROMPT_VERSION", d.caption_prompt_version),
        table_context=get("TABLE_CONTEXT", d.table_context).lower(),
        lexical_weight=float(get("LEXICAL_WEIGHT", d.lexical_weight)),
        min_score=float(get("MIN_SCORE", d.min_score)),
        top_k=int(get("TOP_K", d.top_k)),
        figure_max_side=int(get("FIGURE_MAX_SIDE", d.figure_max_side)),
        cors_origins=csv("CORS_ORIGINS", d.cors_origins),
    )


_override: Settings | None = None


@lru_cache(maxsize=1)
def _loaded() -> Settings:
    return load_settings()


def settings() -> Settings:
    """The active settings (an override set by tests/scripts, else env + .env)."""
    return _override or _loaded()


def set_settings(s: Settings | None) -> None:
    """Replace the active settings (tests, CLI flags). None restores env loading."""
    global _override
    _override = s
    _loaded.cache_clear()
