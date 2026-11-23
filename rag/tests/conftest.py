"""Wiring tests: no network. A deterministic fake replaces OpenRouter."""
from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config as config_mod  # noqa: E402
from app import openrouter  # noqa: E402

DIM = 64


def bow_vector(text: str) -> list[float]:
    v = [0.0] * DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeClient:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def embed_texts(self, texts, *, model=None):
        self.calls.append(("embed", model or ""))
        return [bow_vector(t) for t in texts]

    def embed_images(self, paths, *, model=None):
        return [bow_vector(Path(p).stem) for p in paths]

    def chat(self, *, role, model, messages, max_tokens=400, temperature=0.0, reasoning_off=None):
        self.calls.append((role, model))
        if role == "caption":
            return ("Exploded view of the hydraulic pump assembly showing housing, impeller, shaft seal and "
                    "gasket with callout item numbers 1 to 6 for parts ordering.")
        if role == "ocr":
            return "Scanned page text about hydraulic pump removal and installation steps for the mechanic."
        content = messages[-1]["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        uid = re.search(r"The primary source is \[([^\]]+)\]", text).group(1)
        return f"Use the parts shown [{uid}] and also see [made-up-unit]."


@pytest.fixture
def fake_client():
    c = FakeClient()
    openrouter.set_client(c)
    yield c
    openrouter.set_client(None)


@pytest.fixture
def s(tmp_path):
    st = config_mod.Settings(openrouter_api_key="test-key", data_dir=tmp_path, vector_backend="memory",
                             embed_model="fake-embed", embed_dimensions=None, corpora=("pdf",),
                             pdf_paths=("raw/sample.pdf",), ocr_engine="none", min_figure_area=0.02)
    config_mod.set_settings(st)
    yield st
    config_mod.set_settings(None)


@pytest.fixture
def sample_pdf(s):
    """Three pages: native text + table, vector diagram with caption, image-only scan."""
    import pymupdf as fitz
    from PIL import Image

    raw = s.data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((72, 72), "SECTION 2 HYDRAULIC PUMP MAINTENANCE", fontsize=12)
    p1.insert_textbox(fitz.Rect(72, 90, 520, 200),
                      "Remove the hydraulic pump by disconnecting the inlet line and removing the four mounting "
                      "bolts. Inspect the shaft seal for leaks before reinstalling the pump on the gearbox pad.",
                      fontsize=10)
    p1.insert_text((72, 230), "Table 2-1. Torque values", fontsize=10)
    x0, y0, cw, rh = 72, 240, 150, 20
    cells = [["Fastener", "Torque Nm", "Item"], ["Mounting bolt", "45", "3"], ["Inlet fitting", "30", "5"],
             ["Drain plug", "25", "6"]]
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            rect = fitz.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            p1.draw_rect(rect, color=(0, 0, 0), width=0.8)
            p1.insert_text((rect.x0 + 4, rect.y1 - 6), val, fontsize=9)

    p2 = doc.new_page()
    for i in range(40):  # a "diagram": many vector strokes in one region
        p2.draw_line((100 + i * 8, 150), (300 + i * 4, 400), color=(0, 0, 0), width=0.5)
    p2.draw_circle((250, 280), 60, color=(0, 0, 0))
    p2.insert_text((120, 430), "Figure 2-1. Hydraulic pump exploded view", fontsize=10)

    img = raw / "scan.png"
    Image.new("RGB", (600, 800), (235, 235, 235)).save(img)
    p3 = doc.new_page()
    p3.insert_image(p3.rect, filename=str(img))

    path = raw / "sample.pdf"
    doc.save(path)
    return path
