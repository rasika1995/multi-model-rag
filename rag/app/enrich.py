"""Stage 3: make figures and tables retrievable without touching the artefact.

- Figures: vision-model caption (reasoning off, cached on disk by image hash +
  prompt version + model, at most one call per figure). Empty or generic
  captions are caught and rejected, never indexed.
- Tables: linearised column=value rows for SEARCH; the structured cells stay
  as the artefact, unchanged.
- Anything that cannot be made retrievable is rejected with a reason.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Settings, settings
from .openrouter import DiskCache, EmptyCompletion, ModelClient, ModelUnavailable, UpstreamError, image_to_data_url
from .schemas import Rejected, Unit

log = logging.getLogger(__name__)

CAPTION_PROMPTS = {
    # Baseline prompt from the brief.
    "v1": (
        "You are indexing a technical maintenance manual so a field engineer can find this figure "
        "with a text search. In about 100 words describe: the system or component shown; every "
        "visible label; callout/item numbers and any part numbers; the type of drawing (exploded "
        "view, wiring diagram, schematic, installation view); and what the diagram is for. Use the "
        "exact terms printed on the figure. Plain text, no preamble."
    ),
    # Stage 4 lever: richer identifiers (model/variant terms, assembly names, sub-assemblies).
    "v2": (
        "You are indexing a technical maintenance manual so a field engineer can find this figure "
        "with a text search. Write plain text, no preamble, about 150 words:\n"
        "1. First line: the exact figure title if printed, otherwise the assembly name.\n"
        "2. The main assembly and every sub-assembly or component you can identify (use common "
        "names a mechanic would type, plus the printed names: e.g. 'cylinder liner, O-ring, seal').\n"
        "3. All callout/item numbers and any part, model or variant numbers printed.\n"
        "4. Drawing type (exploded view, wiring diagram, schematic, installation) and what task it "
        "supports (removal, replacement, inspection, ordering parts)."
    ),
}

GENERIC_PATTERNS = re.compile(
    r"^(this is |the image shows |)(a |an )?(technical |engineering )?(diagram|drawing|image|figure|picture)\.?$",
    re.I,
)


def is_usable_caption(c: str | None) -> bool:
    if not c:
        return False
    c = c.strip()
    if len(c) < 60 or GENERIC_PATTERNS.match(c):
        return False
    return True


@dataclass
class EnrichResult:
    units: list[Unit] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    captions_from_cache: int = 0
    captions_called: int = 0
    caption_models: dict[str, int] = field(default_factory=dict)


def caption_figure(path: Path, client: ModelClient, s: Settings, hint: str = "") -> tuple[str, str, bool]:
    """Return (caption, model_used, from_cache). Raises UpstreamError if every model fails."""
    prompt = CAPTION_PROMPTS.get(s.caption_prompt_version, CAPTION_PROMPTS["v1"])
    if hint:
        prompt += f"\nText found next to or inside the figure (may help): {hint[:400]}"
    raw = path.read_bytes()
    cache = DiskCache(s.cache_dir, "captions")
    models = [s.vision_model, *[m for m in s.vision_fallback_models if m != s.vision_model]]
    last: Exception | None = None
    for model in models:
        key = DiskCache.key(model, prompt, raw)
        hit = cache.get(key)
        if hit and is_usable_caption(hit.get("caption")):
            return hit["caption"], model, True
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_url(path, s.figure_max_side)}},
        ]}]
        try:
            cap = client.chat(role="caption", model=model, messages=messages, max_tokens=500)
            if not is_usable_caption(cap):
                # Usually reasoning ate the budget, or the model was lazy: one retry with more room.
                cap = client.chat(role="caption", model=model, messages=messages, max_tokens=1500,
                                  reasoning_off=True)
        except (ModelUnavailable, EmptyCompletion, UpstreamError) as e:
            log.warning("caption failed on %s with %s: %s", path.name, model, e)
            last = e
            continue
        if is_usable_caption(cap):
            cap = re.sub(r"\s+", " ", cap).strip()
            cache.set(key, {"caption": cap, "model": model, "prompt_version": s.caption_prompt_version})
            return cap, model, False
        last = EmptyCompletion(model, f"unusable caption: {cap[:80]!r}")
    assert last is not None
    raise last if isinstance(last, UpstreamError) else UpstreamError(models[-1], str(last))


def linearise_table(u: Unit, context: str = "") -> str:
    a = u.artefact
    cols = [re.sub(r"\s+", " ", c).strip() or f"col{i+1}" for i, c in enumerate(a.columns or [])]
    lines = []
    cap = u.meta.get("caption") or ""
    if cap:
        lines.append(f"Table: {cap}")
    if context:
        lines.append(context)
    if cols:
        lines.append("Columns: " + " | ".join(cols))
    for r in a.rows or []:
        cells = [f"{c}={re.sub(r'\s+', ' ', v).strip()}" for c, v in zip(cols, r) if v and v.strip()]
        if cells:
            lines.append("; ".join(cells))
    text = "\n".join(lines)
    return text[:6000]


def enrich(units: list[Unit], client_factory: Callable[[], ModelClient], s: Settings | None = None,
           progress: Callable[[str], None] | None = None) -> EnrichResult:
    s = s or settings()
    out = EnrichResult()
    by_id = {u.unit_id: u for u in units}
    figures = [u for u in units if u.type == "figure"]

    # ---- figures first (tables may borrow the paired figure's title) -------
    captions: dict[str, str] = {}
    for i, u in enumerate(figures, 1):
        path = s.data_dir / (u.artefact.path or "")
        if not u.artefact.path or not path.exists():
            out.rejected.append(Rejected(unit_id=u.unit_id, stage="enrich", reason="figure image file missing",
                                         document_id=u.provenance.document_id, page=u.provenance.page))
            continue
        hint = " ".join(x for x in (u.meta.get("caption_hint"), u.meta.get("labels_text")) if x)
        try:
            cap, model, cached = caption_figure(path, client_factory(), s, hint)
        except ModelUnavailable:
            raise  # every model retired: stop the run, the fix is configuration
        except UpstreamError as e:
            out.rejected.append(Rejected(unit_id=u.unit_id, stage="enrich", reason=f"no usable caption: {e}",
                                         document_id=u.provenance.document_id, page=u.provenance.page))
            continue
        out.captions_from_cache += int(cached)
        out.captions_called += int(not cached)
        out.caption_models[model] = out.caption_models.get(model, 0) + 1
        captions[u.unit_id] = cap
        extra = u.meta.get("caption_hint") or ""
        u2 = u.model_copy(deep=True)
        u2.search_text = f"{cap}\n{extra}".strip()
        u2.meta.update(caption=cap, caption_model=model, caption_prompt=s.caption_prompt_version)
        out.units.append(u2)
        if progress:
            progress(f"caption {i}/{len(figures)} {u.unit_id} ({'cache' if cached else model})")

    # ---- tables and text ----------------------------------------------------
    for u in units:
        if u.type == "figure":
            continue
        if u.type == "table":
            if not u.artefact.rows:
                out.rejected.append(Rejected(unit_id=u.unit_id, stage="enrich", reason="table has no rows",
                                             document_id=u.provenance.document_id, page=u.provenance.page))
                continue
            ctx = ""
            if s.table_context == "paired_figure":
                fid = u.meta.get("paired_figure")
                cap = captions.get(fid or "", "")
                if cap:
                    ctx = "Parts list for the figure: " + re.split(r"(?<=[.!?])\s", cap, maxsplit=1)[0]
            u2 = u.model_copy(deep=True)
            u2.search_text = linearise_table(u, ctx)
            out.units.append(u2)
        else:
            if not (u.search_text or "").strip():
                out.rejected.append(Rejected(unit_id=u.unit_id, stage="enrich", reason="empty text",
                                             document_id=u.provenance.document_id, page=u.provenance.page))
                continue
            out.units.append(u)
    _ = by_id
    return out
