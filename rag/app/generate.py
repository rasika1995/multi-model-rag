"""Stage 6: answer with the artefact, never a paraphrase, never unsourced.

Question -> retrieve -> drop unsourced -> decide type of the top SOURCED hit
-> figure: real image + vision answer | table: exact rows + text answer |
text: span + text answer. No sourced hit left -> abstain and say why.
"""
from __future__ import annotations

import logging
import re
from typing import Callable

from .config import Settings, settings
from .openrouter import ModelClient, UpstreamError, image_to_data_url
from .schemas import Answer, Hit

log = logging.getLogger(__name__)

NOT_IN_SOURCE = "NOT IN SOURCE"

SYSTEM = (
    "You answer maintenance questions for a field engineer using ONLY the sources provided. "
    "Each source is tagged like [unit-id]. Rules: answer in at most 3 short sentences; cite the "
    "source tag in square brackets after each claim; quote part numbers, item numbers and values "
    "exactly; never invent a source tag. If the sources do not contain the answer, reply exactly "
    f"'{NOT_IN_SOURCE}' and nothing else."
)

_STOP = set("""the and for with that this from what which when where does have your into
are can how is it of on or to a an in be do i my me we you they them shown show diagram figure
table part parts list need should would could will there here these those about each item""".split())


def highlight_rows(question: str, columns: list[str], rows: list[list[str]], limit: int = 12) -> list[int]:
    """Rows of a table that the question is about (item numbers, part numbers, names)."""
    q = question.lower()
    nums = set(re.findall(r"\b\d{1,3}\b", q))
    ids = {t for t in re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)+|\b\d{4,}\b", q)}
    words = {w for w in re.findall(r"[a-z]{4,}", q) if w not in _STOP}
    words |= {w[:-1] for w in words if w.endswith("s") and len(w) > 4}
    item_col = next((i for i, c in enumerate(columns or []) if re.search(r"item", c, re.I)), None)
    scored = []
    for i, r in enumerate(rows):
        row = " ".join(r).lower()
        sc = 0
        if item_col is not None and item_col < len(r) and r[item_col].strip() in nums:
            sc += 3
        sc += sum(2 for t in ids if t in row)
        sc += sum(1 for w in words if re.search(rf"\b{re.escape(w)}", row))
        if sc:
            scored.append((sc, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[0][0] if scored else 0
    return sorted(i for sc, i in scored[:limit] if sc >= max(1, top // 2))


def _source_block(h: Hit, max_chars: int = 3500) -> str:
    return f"[{h.unit_id}] ({h.type}, {h.citation})\n{h.search_text[:max_chars]}"


def _check_citations(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    bad = [t for t in re.findall(r"\[([^\[\]\s]+)\]", text) if t not in allowed]
    for t in bad:
        text = text.replace(f"[{t}]", "")
    return text.strip(), bad


def answer(question: str, hits: list[Hit], client_factory: Callable[[], ModelClient],
           s: Settings | None = None) -> Answer:
    s = s or settings()
    if not hits:
        return Answer(status="abstained", text="I can't answer: nothing was retrieved for this question.",
                      abstain_reason="no units retrieved")
    sourced = [h for h in hits if h.sourced]
    dropped = len(hits) - len(sourced)
    if not sourced:
        return Answer(status="abstained", dropped_unsourced=dropped,
                      text=f"I can't give a sourced answer: {dropped} unit(s) were retrieved but none can name "
                           "the document and page it came from.",
                      abstain_reason="no retrieved unit has resolvable provenance")
    top = sourced[0]
    if top.score < s.min_score:
        return Answer(status="abstained", dropped_unsourced=dropped,
                      text=f"I can't answer confidently: the best sourced match ({top.citation}) scored "
                           f"{top.score:.2f}, below the relevance threshold {s.min_score:.2f}.",
                      abstain_reason="top sourced score below MIN_SCORE")

    top.cited = True
    context = [top] + [h for h in sourced[1:3]]
    allowed = {h.unit_id for h in context}
    base = dict(answer_type=top.type, citation=top.citation, cited_unit_id=top.unit_id,
                artefact=top.artefact, artefact_url=top.artefact_url, dropped_unsourced=dropped)
    hl: list[int] = []
    if top.type == "table" and top.artefact.rows:
        hl = highlight_rows(question, top.artefact.columns or [], top.artefact.rows)

    user_text = (f"Question: {question}\n\nSources:\n" + "\n\n".join(_source_block(h) for h in context) +
                 f"\n\nThe primary source is [{top.unit_id}]. Answer now.")
    try:
        if top.type == "figure" and top.artefact.path:
            model = s.figure_model
            content = [{"type": "text", "text": user_text + "\nThe image attached is the primary source figure."},
                       {"type": "image_url", "image_url": {"url": image_to_data_url(s.data_dir / top.artefact.path, s.figure_max_side)}}]
        else:
            model = s.text_model
            content = user_text
        text = client_factory().chat(role="generate", model=model, max_tokens=400,
                                     messages=[{"role": "system", "content": SYSTEM},
                                               {"role": "user", "content": content}])
    except UpstreamError as e:
        log.warning("generation failed: %s", e)
        return Answer(status="answered", highlight_rows=hl, model=None, generation_error=str(e),
                      text=f"The answer model is unavailable ({e.model}), so here is the retrieved source "
                           f"itself: {top.citation}.", **base)

    if NOT_IN_SOURCE.lower() in text.lower() and len(text) < 40:
        return Answer(status="abstained", highlight_rows=hl, model=model,
                      text=f"The closest sourced unit ({top.citation}) does not contain the answer, so I won't guess. "
                           "It is shown below for reference.",
                      abstain_reason="model reported the sources do not answer the question", **base)
    text, bad = _check_citations(text, allowed)
    if f"[{top.unit_id}]" not in text and not any(f"[{u}]" in text for u in allowed):
        text = f"{text} [{top.unit_id}]"
    ans = Answer(status="answered", text=text, highlight_rows=hl, model=model, **base)
    if bad:
        ans.generation_error = f"removed invented citation(s): {', '.join(bad)}"
    return ans
