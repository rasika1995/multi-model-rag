"""The only module that calls the model API (OpenRouter, OpenAI-compatible).

- Process-wide throttle so we stay under the free-tier requests-per-minute cap.
- Retries with exponential backoff for 429 / 5xx / timeouts.
- 404 "No endpoints found" -> ModelUnavailable (the slug is retired; not a code bug).
- Disk caches keyed by content so a re-run never pays twice.
- Call counters (global and per request) feed Layer C.
"""
from __future__ import annotations

import base64
import contextvars
import hashlib
import io
import json
import logging
import math
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import httpx
from PIL import Image

from .config import ConfigError, settings

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ errors
class UpstreamError(RuntimeError):
    """The model provider failed. Always names the model."""

    kind = "upstream"

    def __init__(self, model: str, message: str, status: int | None = None):
        self.model = model
        self.status = status
        super().__init__(f"[{model}] {message}" + (f" (HTTP {status})" if status else ""))


class ModelUnavailable(UpstreamError):
    kind = "model_unavailable"


class EmptyCompletion(UpstreamError):
    kind = "empty_completion"


# ----------------------------------------------------------------- counters
_global_calls: Counter = Counter()
_request_calls: contextvars.ContextVar[Counter | None] = contextvars.ContextVar(
    "request_calls", default=None
)
_lock = threading.Lock()


def _count(role: str) -> None:
    with _lock:
        _global_calls[role] += 1
    c = _request_calls.get()
    if c is not None:
        c[role] += 1


def global_calls() -> dict[str, int]:
    return dict(_global_calls)


def reset_global_calls() -> None:
    _global_calls.clear()


class track_calls:
    """Context manager: count model calls made inside the block."""

    def __enter__(self) -> Counter:
        self.counter: Counter = Counter()
        self._token = _request_calls.set(self.counter)
        return self.counter

    def __exit__(self, *exc) -> None:
        _request_calls.reset(self._token)


# -------------------------------------------------------------------- cache
class DiskCache:
    def __init__(self, root: Path, namespace: str):
        self.dir = root / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p if isinstance(p, bytes) else str(p).encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> Any | None:
        f = self.dir / f"{key}.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        f = self.dir / f"{key}.json"
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        tmp.replace(f)


# ------------------------------------------------------------------ helpers
def image_to_data_url(path: Path | str, max_side: int | None = None) -> str:
    """Load an image, downscale so the long side <= max_side, return a data URL."""
    max_side = max_side or settings().figure_max_side
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def normalise(vec: list[float], dims: int | None) -> list[float]:
    """Matryoshka-style truncation then L2 normalisation."""
    v = vec[:dims] if dims else vec
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# ------------------------------------------------------------- client API
class ModelClient(Protocol):
    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...

    def embed_images(self, paths: list[Path], *, model: str | None = None) -> list[list[float]]: ...

    def chat(
        self,
        *,
        role: str,
        model: str,
        messages: list[dict],
        max_tokens: int = 400,
        temperature: float = 0.0,
        reasoning_off: bool | None = None,
    ) -> str: ...


class _Throttle:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self, rpm: int) -> None:
        interval = 60.0 / max(1, rpm)
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + interval
        if delay > 0:
            time.sleep(delay)


_throttle = _Throttle()


class OpenRouterClient:
    def __init__(self) -> None:
        s = settings()
        self.s = s
        self._http = httpx.Client(
            base_url=s.openrouter_base_url,
            timeout=s.request_timeout_s,
            headers={
                "Authorization": f"Bearer {s.require_key()}",
                "X-Title": "multimodal-rag-lab",
            },
        )
        self.embed_cache = DiskCache(s.cache_dir, "embeddings")

    # ------------------------------------------------------------ transport
    def _post(self, path: str, body: dict, *, model: str, role: str) -> dict:
        s = self.s
        last: Exception | None = None
        for attempt in range(s.max_retries + 1):
            _throttle.wait(s.requests_per_minute)
            _count(role)
            try:
                r = self._http.post(path, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = UpstreamError(model, f"network error: {e.__class__.__name__}: {e}")
                self._backoff(attempt, None)
                continue
            data: dict = {}
            try:
                data = r.json()
            except ValueError:
                pass
            err = data.get("error") if isinstance(data, dict) else None
            msg = (err.get("message") if isinstance(err, dict) else None) or r.text[:300]
            if r.status_code == 200 and not err:
                return data
            status = r.status_code if r.status_code != 200 else (err or {}).get("code")
            if r.status_code == 404 or "No endpoints found" in str(msg):
                raise ModelUnavailable(model, f"slug retired or unavailable: {msg}", 404)
            if r.status_code in (401, 403):
                raise UpstreamError(model, f"authentication/permission problem: {msg}", r.status_code)
            if r.status_code == 429 or status == 429 or (isinstance(status, int) and status >= 500) \
                    or r.status_code >= 500:
                last = UpstreamError(model, f"transient: {msg}", status if isinstance(status, int) else None)
                self._backoff(attempt, r.headers.get("retry-after"))
                continue
            raise UpstreamError(model, str(msg), status if isinstance(status, int) else None)
        assert last is not None
        raise last

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> None:
        try:
            wait = float(retry_after) if retry_after else 0.0
        except ValueError:
            wait = 0.0
        time.sleep(max(wait, min(60.0, 2.0 * (2**attempt))))

    # ----------------------------------------------------------- embeddings
    def _embed_inputs(self, inputs: list[Any], model: str) -> list[list[float]]:
        dims = self.s.embed_dimensions
        body: dict[str, Any] = {"model": model, "input": inputs, "encoding_format": "float"}
        if dims:
            body["dimensions"] = dims
        try:
            data = self._post("/embeddings", body, model=model, role="embed")
        except UpstreamError as e:
            if dims and e.status == 400 and not isinstance(e, ModelUnavailable):
                body.pop("dimensions")  # model may not accept it: truncate locally instead
                data = self._post("/embeddings", body, model=model, role="embed")
            else:
                raise
        rows = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        if len(rows) != len(inputs):
            raise UpstreamError(model, f"expected {len(inputs)} embeddings, got {len(rows)}")
        return [normalise(r["embedding"], dims) for r in rows]

    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        model = model or self.s.embed_model
        out: list[list[float] | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            hit = self.embed_cache.get(DiskCache.key("text", model, self.s.embed_dimensions, t))
            if hit is not None:
                out[i] = hit
            else:
                todo.append(i)
        bs = self.s.embed_batch_size
        for start in range(0, len(todo), bs):
            idx = todo[start : start + bs]
            vecs = self._embed_inputs([texts[i] for i in idx], model)
            for i, v in zip(idx, vecs):
                out[i] = v
                self.embed_cache.set(DiskCache.key("text", model, self.s.embed_dimensions, texts[i]), v)
        return out  # type: ignore[return-value]

    def embed_images(self, paths: list[Path], *, model: str | None = None) -> list[list[float]]:
        model = model or self.s.embed_model
        out = []
        for p in paths:
            raw = Path(p).read_bytes()
            key = DiskCache.key("image", model, self.s.embed_dimensions, raw)
            hit = self.embed_cache.get(key)
            if hit is None:
                url = image_to_data_url(p)
                try:  # content-part form
                    hit = self._embed_inputs([{"content": [{"type": "image_url", "image_url": {"url": url}}]}], model)[0]
                except UpstreamError as e:
                    if e.status != 400:
                        raise
                    hit = self._embed_inputs([[{"type": "image_url", "image_url": {"url": url}}]], model)[0]
                self.embed_cache.set(key, hit)
            out.append(hit)
        return out

    # ----------------------------------------------------------------- chat
    def chat(
        self,
        *,
        role: str,
        model: str,
        messages: list[dict],
        max_tokens: int = 400,
        temperature: float = 0.0,
        reasoning_off: bool | None = None,
    ) -> str:
        reasoning_off = self.s.disable_reasoning if reasoning_off is None else reasoning_off
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if reasoning_off:
            body["reasoning"] = {"enabled": False, "exclude": True}
        try:
            data = self._post("/chat/completions", body, model=model, role=role)
        except UpstreamError as e:
            if reasoning_off and e.status == 400 and "reasoning" in str(e).lower():
                # Mandatory-reasoning model: give it a bigger budget instead.
                body.pop("reasoning")
                body["max_tokens"] = max_tokens * 4
                data = self._post("/chat/completions", body, model=model, role=role)
            else:
                raise
        choices = data.get("choices") or []
        content = (choices[0].get("message", {}).get("content") if choices else None) or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        content = content.strip()
        if not content:
            fin = choices[0].get("finish_reason") if choices else None
            raise EmptyCompletion(
                model,
                f"empty completion (finish_reason={fin}); reasoning may have used the token budget",
            )
        return content


# ---------------------------------------------------------------- registry
_client: ModelClient | None = None
_client_lock = threading.Lock()


def get_client() -> ModelClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenRouterClient()
        return _client


def set_client(c: ModelClient | None) -> None:
    """Swap in a fake client (tests) or force re-creation after a settings change."""
    global _client
    _client = c


__all__ = [
    "ConfigError",
    "DiskCache",
    "EmptyCompletion",
    "ModelClient",
    "ModelUnavailable",
    "OpenRouterClient",
    "UpstreamError",
    "get_client",
    "global_calls",
    "image_to_data_url",
    "normalise",
    "reset_global_calls",
    "set_client",
    "track_calls",
]
