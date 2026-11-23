"""Model readiness: one tiny call per configured role (template T1).

Embedding calls bypass the cache so a retired slug can't hide behind it.
Results are saved to data/model_check.json and shown by /health.
"""
from __future__ import annotations

import base64
import io
import json
import time
from datetime import datetime, timezone

from PIL import Image, ImageDraw

from .config import ConfigError, Settings, settings
from .openrouter import OpenRouterClient, UpstreamError


def _tiny_png() -> str:
    im = Image.new("RGB", (64, 64), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, 56, 56], outline="black", width=3)
    d.line([8, 8, 56, 56], fill="red", width=3)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run_checks(s: Settings | None = None, include_judge: bool = False) -> list[dict]:
    s = s or settings()
    s.require_key()  # stops before any call when the key is missing
    client = OpenRouterClient()
    img = _tiny_png()
    roles: list[tuple[str, str, callable]] = [
        ("Embeddings", s.embed_model, lambda: client._embed_inputs(["readiness check"], s.embed_model)),
        ("Vision / captioning", s.vision_model, lambda: client.chat(
            role="check", model=s.vision_model, max_tokens=50,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Name the shape and the line colour in five words."},
                {"type": "image_url", "image_url": {"url": img}}]}])),
        ("Text generation", s.text_model, lambda: client.chat(
            role="check", model=s.text_model, max_tokens=20,
            messages=[{"role": "user", "content": "Reply with the single word OK."}])),
    ]
    if s.figure_model != s.vision_model:
        roles.append(("Figure answers", s.figure_model, lambda: client.chat(
            role="check", model=s.figure_model, max_tokens=50,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this image in five words."},
                {"type": "image_url", "image_url": {"url": img}}]}])))
    if include_judge:
        roles.append(("Judge", s.judge_model, lambda: client.chat(
            role="check", model=s.judge_model, max_tokens=20,
            messages=[{"role": "user", "content": "Reply with the single word OK."}])))

    results = []
    for role, model, fn in roles:
        t = time.perf_counter()
        try:
            out = fn()
            note = (f"dim={len(out[0])}" if isinstance(out, list) else str(out)[:80].replace("\n", " "))
            results.append({"role": role, "model": model, "ok": True,
                            "ms": round((time.perf_counter() - t) * 1000), "note": note})
        except (UpstreamError, ConfigError) as e:
            results.append({"role": role, "model": model, "ok": False,
                            "ms": round((time.perf_counter() - t) * 1000), "note": str(e)[:300]})
    if s.figure_model == s.vision_model:
        v = next(r for r in results if r["role"] == "Vision / captioning")
        results.append({**v, "role": "Figure answers", "note": "same slug as vision: " + v["note"]})
    payload = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "results": results}
    s.data_dir.mkdir(parents=True, exist_ok=True)
    (s.data_dir / "model_check.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return results


def read_last_check(s: Settings | None = None) -> dict | None:
    s = s or settings()
    p = s.data_dir / "model_check.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def format_t1(results: list[dict]) -> str:
    lines = ["| Role | Slug | Call OK? | Time (ms) | Notes / error |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['role']} | `{r['model']}` | {'OK' if r['ok'] else 'FAIL'} | {r['ms']} | "
                     f"{r['note'].replace('|', '/')} |")
    return "\n".join(lines)
