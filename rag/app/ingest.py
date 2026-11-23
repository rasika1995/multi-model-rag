"""Stage 1-2: loaders, content-type router, OCR gate, figure/table extraction.

Every loader returns the same LoadResult so downstream code never changes
when the corpus does. Every unit is stamped document > page > unit id here,
at birth; units that cannot be stamped are rejected, not repaired later.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import Settings, settings
from .schemas import Artefact, Provenance, Rejected, Unit

log = logging.getLogger(__name__)


@dataclass
class LoadResult:
    units: list[Unit] = field(default_factory=list)
    pages: list[dict[str, Any]] = field(default_factory=list)  # per-page routing report
    rejected: list[Rejected] = field(default_factory=list)
    review_queue: list[dict[str, Any]] = field(default_factory=list)  # OCR-gate failures

    def extend(self, other: "LoadResult") -> None:
        self.units += other.units
        self.pages += other.pages
        self.rejected += other.rejected
        self.review_queue += other.review_queue


class Loader(Protocol):
    name: str

    def load(self) -> LoadResult: ...


def rel(path: Path, s: Settings) -> str:
    """Path relative to the data dir, with forward slashes (stored in payloads)."""
    return Path(path).resolve().relative_to(s.data_dir.resolve()).as_posix()


# ============================================================ Enginuity T2
class EnginuityT2Loader:
    """Enginuity Bench Task 2: 12 figure PNGs, 15 table PNGs + TSVs, 60 QA.

    The files are rendered manual pages with no page numbers, so each figure
    group is treated as one document: the figure is page 1 and its tables are
    pages 2..n, in the order the dataset lists them. This is stated in the
    provenance note; it is a synthetic but stable and resolvable page.
    """

    name = "enginuity_t2"

    def __init__(self, s: Settings | None = None):
        self.s = s or settings()
        self.root = self.s.data_dir / "enginuity" / "task-2-question-answering"

    @staticmethod
    def doc_id(group: str) -> str:
        return f"enginuity-t2-{group}"

    @staticmethod
    def figure_uid(group: str) -> str:
        return f"t2-{group}-figure"

    @staticmethod
    def table_uid(group: str, n: int) -> str:
        return f"t2-{group}-table{n}"

    def load(self) -> LoadResult:
        res = LoadResult()
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.root} not found. Run: python scripts/download_data.py --tier a"
            )
        for fig in sorted((self.root / "figures").glob("figure_*.*")):
            m = re.match(r"figure_(\d+)", fig.stem)
            if not m:
                continue
            num = m.group(1)
            group = f"fig{num}"
            doc = self.doc_id(group)
            fuid = self.figure_uid(group)
            res.units.append(
                Unit(
                    unit_id=fuid,
                    type="figure",
                    artefact=Artefact(kind="image", path=rel(fig, self.s)),
                    provenance=Provenance(
                        document_id=doc, page=1, unit_id=fuid, source_path=rel(fig, self.s)
                    ),
                    meta={"group": group, "corpus": self.name},
                )
            )
            pages = 1
            for timg in sorted((self.root / "tables").glob(f"table_{num}_*.png")):
                tn = int(timg.stem.rsplit("_", 1)[1])
                tuid = self.table_uid(group, tn)
                tsv = self.root / "table_data" / f"{timg.stem}.tsv"
                pages += 1
                if not tsv.exists():
                    res.rejected.append(
                        Rejected(unit_id=tuid, stage="extract", reason="table image has no TSV (no structure)",
                                 document_id=doc, page=pages)
                    )
                    continue
                columns, rows = read_tsv(tsv)
                res.units.append(
                    Unit(
                        unit_id=tuid,
                        type="table",
                        artefact=Artefact(kind="table", columns=columns, rows=rows, path=rel(timg, self.s)),
                        provenance=Provenance(
                            document_id=doc, page=pages, unit_id=tuid, source_path=rel(tsv, self.s)
                        ),
                        meta={"group": group, "paired_figure": fuid, "corpus": self.name},
                    )
                )
            res.pages.append({"document": doc, "pages": pages, "class": "pre-rendered"})
        return res


def read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f, delimiter="\t")]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return [], []
    header, body = rows[0], rows[1:]
    width = len(header)
    body = [(r + [""] * width)[:width] for r in body]
    return header, body


# ===================================================================== OCR
_SINGLE_OK = set("aAI&+-/")


def garbled_ratio(text: str) -> float:
    """Share of tokens that look like OCR noise. Digits and part numbers are fine."""
    toks = text.split()
    if not toks:
        return 1.0
    bad = 0
    for t in toks:
        alnum = sum(ch.isalnum() for ch in t)
        if alnum == 0:
            bad += 1
        elif (len(t) - alnum) / len(t) > 0.5:
            bad += 1
        elif len(t) == 1 and t.isalpha() and t not in _SINGLE_OK:
            bad += 1
    return bad / len(toks)


@dataclass
class OcrResult:
    text: str
    engine: str
    confidence: float | None
    garbled: float
    passed: bool
    reason: str = ""


def gate(text: str, confidence: float | None, engine: str, s: Settings) -> OcrResult:
    g = garbled_ratio(text)
    reasons = []
    if len(text.strip()) < s.ocr_min_chars:
        reasons.append(f"too little text ({len(text.strip())} < {s.ocr_min_chars} chars)")
    if g > s.ocr_max_garbled:
        reasons.append(f"garbled ratio {g:.2f} > {s.ocr_max_garbled}")
    if confidence is not None and confidence < s.ocr_min_confidence:
        reasons.append(f"mean confidence {confidence:.0f} < {s.ocr_min_confidence:.0f}")
    if "[UNREADABLE]" in text:
        reasons.append("model reported the page unreadable")
    return OcrResult(text, engine, confidence, g, not reasons, "; ".join(reasons))


def _tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


OCR_PROMPT = (
    "Transcribe every piece of text on this technical manual page exactly as printed, "
    "in reading order. Keep part numbers, item numbers and units verbatim. Do not describe "
    "the drawing and do not add commentary. If the page has no legible text, output "
    "[UNREADABLE]."
)


def ocr_page_image(png: Path, s: Settings, client_factory: Callable[[], Any] | None) -> OcrResult:
    engine = s.ocr_engine
    if engine in ("auto", "tesseract") and _tesseract_available():
        import pytesseract
        from PIL import Image

        data = pytesseract.image_to_data(Image.open(png), output_type=pytesseract.Output.DICT)
        words = [w for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
        confs = [float(c) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
        text = " ".join(words)
        conf = sum(confs) / len(confs) if confs else 0.0
        return gate(text, conf, "tesseract", s)
    if engine == "tesseract":
        return OcrResult("", "tesseract", None, 1.0, False, "tesseract not installed")
    if engine in ("auto", "vision") and client_factory is not None:
        from .openrouter import DiskCache, UpstreamError, image_to_data_url

        cache = DiskCache(s.cache_dir, "ocr")
        key = DiskCache.key(s.vision_model, OCR_PROMPT, png.read_bytes())
        text = cache.get(key)
        if text is None:
            try:
                text = client_factory().chat(
                    role="ocr",
                    model=s.vision_model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(png)}},
                    ]}],
                )
                cache.set(key, text)
            except UpstreamError as e:
                return OcrResult("", "vision", None, 1.0, False, f"OCR call failed: {e}")
        return gate(text, None, "vision", s)
    return OcrResult("", "none", None, 1.0, False, "no OCR engine available (install tesseract or set OCR_ENGINE=vision)")


# =================================================================== PDFs
def _norm_boiler(t: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", t.strip().lower()))


def _overlap(a, b) -> float:
    """Intersection area / area of a."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def _merge_rects(rects: list[list[float]]) -> list[list[float]]:
    rects = [list(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out: list[list[float]] = []
        for r in rects:
            for o in out:
                if _overlap(r, o) > 0.2 or _overlap(o, r) > 0.2:
                    o[:] = [min(o[0], r[0]), min(o[1], r[1]), max(o[2], r[2]), max(o[3], r[3])]
                    changed = True
                    break
            else:
                out.append(r)
        rects = out
    return rects


def parse_pdf_spec(spec: str) -> tuple[Path, list[int] | None]:
    """'to.pdf', 'to.pdf@1-40' or 'to.pdf@3;7-9;15' (1-based pages, inclusive ranges)."""
    if "@" in spec:
        p, rng = spec.rsplit("@", 1)
        pages: set[int] = set()
        for part in rng.split(";"):
            a, _, b = part.strip().partition("-")
            pages.update(range(int(a), int(b or a) + 1))
        return Path(p), sorted(pages)
    return Path(spec), None


class PdfLoader:
    """Route each page (blank / native / mixed / scanned) and emit regions."""

    name = "pdf"

    def __init__(self, specs: list[str] | None = None, s: Settings | None = None,
                 client_factory: Callable[[], Any] | None = None):
        self.s = s or settings()
        self.specs = specs if specs is not None else list(self.s.pdf_paths)
        self.client_factory = client_factory

    def load(self) -> LoadResult:
        res = LoadResult()
        for spec in self.specs:
            path, rng = parse_pdf_spec(spec)
            if not path.is_absolute():
                path = (self.s.data_dir / path) if (self.s.data_dir / path).exists() else path.resolve()
            res.extend(self.load_pdf(path, rng))
        return res

    # ------------------------------------------------------------------
    def load_pdf(self, path: Path, pages: list[int] | None = None) -> LoadResult:
        import pymupdf as fitz

        s = self.s
        res = LoadResult()
        doc = fitz.open(path)
        doc_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem)
        page_nums = [p - 1 for p in (pages or range(1, len(doc) + 1)) if 1 <= p <= len(doc)]
        fig_dir = s.artefact_dir / "figures"
        tab_dir = s.artefact_dir / "tables"
        page_dir = s.artefact_dir / "pages"
        for d in (fig_dir, tab_dir, page_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ---- pass 1: repeated header/footer lines --------------------------
        band_counts: Counter = Counter()
        nonblank = 0
        for i in page_nums:
            p = doc[i]
            h = p.rect.height
            seen = set()
            blocks = p.get_text("blocks")
            if blocks:
                nonblank += 1
            for b in blocks:
                if b[6] != 0:
                    continue
                if b[1] < 0.08 * h or b[3] > 0.92 * h:
                    seen.add(_norm_boiler(b[4]))
            band_counts.update(seen)
        boiler = {t for t, c in band_counts.items() if c >= 3 and c >= 0.2 * max(1, nonblank)}

        # ---- pass 2: route pages ------------------------------------------
        for i in page_nums:
            page = doc[i]
            pno = i + 1
            W, H = page.rect.width, page.rect.height
            A = W * H
            text = page.get_text().strip()
            imgs = page.get_image_info()
            img_rects = [list(im["bbox"]) for im in imgs]
            big = max([((r[2] - r[0]) * (r[3] - r[1])) / A for r in img_rects] or [0.0])
            drawings = page.get_drawings()
            info: dict[str, Any] = {
                "document": doc_id, "page": pno, "chars": len(text), "images": len(imgs),
                "largest_image_frac": round(big, 3), "drawings": len(drawings),
            }

            if len(text) < 10 and not imgs and len(drawings) < 5:
                info["class"] = "blank"
                res.pages.append(info)
                continue

            scanned = big >= 0.6 and len(text) < 200
            hidden_layer = big >= 0.6 and len(text) >= 200
            info["class"] = "scanned" if scanned else ("scanned_text_layer" if hidden_layer else "native")

            if scanned:
                self._scanned_page(page, doc_id, pno, page_dir, fig_dir, res, info)
                res.pages.append(info)
                continue

            # ---- tables ---------------------------------------------------
            table_rects: list[list[float]] = []
            n_tab = 0
            try:
                tabs = page.find_tables().tables  # ~0.5 s/page: the dominant ingestion cost
            except Exception as e:  # pragma: no cover - parser edge cases
                log.warning("find_tables failed on %s p%d: %s", doc_id, pno, e)
                tabs = []
            for t in tabs:
                raw = _table_cells(page, t)
                raw = [r for r in raw if any(r)]
                if len(raw) < 2 or max(len(r) for r in raw) < 2:
                    continue
                n_tab += 1
                uid = f"{doc_id}-p{pno:03d}-table{n_tab}"
                bbox = [round(x, 1) for x in t.bbox]
                table_rects.append(bbox)
                header = raw[0]
                body = [(r + [""] * len(header))[: len(header)] for r in raw[1:]]
                caption = self._nearby_caption(page, bbox, above=True)
                (tab_dir / f"{uid}.json").write_text(
                    json.dumps({"columns": header, "rows": body, "caption": caption}), encoding="utf-8"
                )
                png = tab_dir / f"{uid}.png"
                page.get_pixmap(clip=fitz.Rect(bbox), dpi=130).save(png)
                res.units.append(Unit(
                    unit_id=uid, type="table",
                    artefact=Artefact(kind="table", columns=header, rows=body, path=rel(png, s)),
                    provenance=Provenance(document_id=doc_id, page=pno, unit_id=uid, bbox=bbox,
                                          source_path=rel(path, s) if _inside(path, s) else path.name),
                    meta={"caption": caption, "corpus": self.name},
                ))

            # ---- figures: raster images + clusters of vector drawings -----
            min_side = s.min_figure_px * 0.75  # px at 96 dpi -> points
            cand = [r for r in img_rects]
            if len(drawings) >= 30:
                try:
                    cand += [list(r) for r in page.cluster_drawings(drawings=drawings)]
                except Exception:  # pragma: no cover
                    pass
            figs = []
            for r in _merge_rects(cand):
                w, h = r[2] - r[0], r[3] - r[1]
                if w < min_side or h < min_side or (w * h) / A < s.min_figure_area:
                    continue  # rules, icons, logos
                if (w * h) / A > 0.95:
                    continue  # page border
                if any(_overlap(r, t) > 0.5 for t in table_rects):
                    continue
                figs.append(r)
            for n, r in enumerate(figs, 1):
                uid = f"{doc_id}-p{pno:03d}-figure{n}"
                bbox = [round(x, 1) for x in r]
                png = fig_dir / f"{uid}.png"
                page.get_pixmap(clip=fitz.Rect(bbox), dpi=150).save(png)
                labels = " ".join(
                    b[4].replace("\n", " ").strip() for b in page.get_text("blocks")
                    if b[6] == 0 and _overlap(b[:4], bbox) > 0.8
                )[:600]
                res.units.append(Unit(
                    unit_id=uid, type="figure",
                    artefact=Artefact(kind="image", path=rel(png, s)),
                    provenance=Provenance(document_id=doc_id, page=pno, unit_id=uid, bbox=bbox,
                                          source_path=rel(path, s) if _inside(path, s) else path.name),
                    meta={"caption_hint": self._nearby_caption(page, bbox, above=False),
                          "labels_text": labels, "corpus": self.name},
                ))

            # ---- text: blocks outside figures/tables, minus boilerplate ---
            blocks = []
            for b in page.get_text("blocks", sort=True):
                if b[6] != 0:
                    continue
                t = re.sub(r"\s+", " ", b[4]).strip()
                if not t or _norm_boiler(b[4]) in boiler:
                    continue
                if any(_overlap(b[:4], r) > 0.6 for r in table_rects + figs):
                    continue
                blocks.append((t, list(b[:4])))
            if hidden_layer:
                ocr = gate(" ".join(t for t, _ in blocks), None, "text_layer", s)
                info["ocr"] = {"engine": "text_layer", "garbled": round(ocr.garbled, 3), "passed": ocr.passed}
                if not ocr.passed:
                    res.review_queue.append({"document": doc_id, "page": pno, "reason": ocr.reason,
                                             "engine": "text_layer"})
                    blocks = []
            n_text = self._emit_text(blocks, doc_id, pno, path, res)
            info.update(tables=n_tab, figures=len(figs), text_units=n_text)
            if info["class"] == "native" and (n_tab or figs):
                info["class"] = "mixed"
            res.pages.append(info)
        doc.close()
        return res

    # ------------------------------------------------------------------
    def _emit_text(self, blocks, doc_id, pno, path, res: LoadResult) -> int:
        s = self.s
        chunks: list[tuple[str, list[float]]] = []
        cur, box = "", None
        for t, b in blocks:
            heading = len(t) < 80 and t.upper() == t and any(c.isalpha() for c in t)
            if cur and (len(cur) + len(t) > s.text_chunk_chars or (heading and len(cur) > 300)):
                chunks.append((cur, box))
                cur, box = "", None
            cur = f"{cur}\n{t}" if cur else t
            box = b if box is None else [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
        if cur:
            chunks.append((cur, box))
        n = 0
        for text, bbox in chunks:
            if len(text) < 40:
                continue  # stray labels / page numbers
            n += 1
            uid = f"{doc_id}-p{pno:03d}-text{n}"
            res.units.append(Unit(
                unit_id=uid, type="text", search_text=text,
                artefact=Artefact(kind="text", text=text),
                provenance=Provenance(document_id=doc_id, page=pno, unit_id=uid,
                                      bbox=[round(x, 1) for x in bbox] if bbox else None,
                                      source_path=rel(path, s) if _inside(path, s) else path.name),
                meta={"corpus": self.name},
            ))
        return n

    def _scanned_page(self, page, doc_id, pno, page_dir, fig_dir, res: LoadResult, info) -> None:
        s = self.s
        png = page_dir / f"{doc_id}-p{pno:03d}.png"
        page.get_pixmap(dpi=150).save(png)
        ocr = ocr_page_image(png, s, self.client_factory)
        info["ocr"] = {"engine": ocr.engine, "confidence": ocr.confidence,
                       "garbled": round(ocr.garbled, 3), "passed": ocr.passed, "reason": ocr.reason}
        if not ocr.passed:
            res.review_queue.append({"document": doc_id, "page": pno, "reason": ocr.reason,
                                     "engine": ocr.engine, "image": rel(png, s)})
        else:
            self._emit_text([(re.sub(r"[ \t]+", " ", ocr.text).strip(), [0, 0, page.rect.width, page.rect.height])],
                            doc_id, pno, Path(page.parent.name), res)
        # A scan dominated by drawing (little text) is also a figure: keep the real page image.
        if len(ocr.text.strip()) < 600:
            uid = f"{doc_id}-p{pno:03d}-figure1"
            fpng = fig_dir / f"{uid}.png"
            fpng.write_bytes(png.read_bytes())
            res.units.append(Unit(
                unit_id=uid, type="figure",
                artefact=Artefact(kind="image", path=rel(fpng, s)),
                provenance=Provenance(document_id=doc_id, page=pno, unit_id=uid,
                                      bbox=[0, 0, page.rect.width, page.rect.height]),
                meta={"caption_hint": "", "labels_text": ocr.text[:600] if ocr.passed else "",
                      "corpus": self.name, "scanned": True},
            ))
            info["figures"] = 1

    @staticmethod
    def _nearby_caption(page, bbox, above: bool) -> str:
        best = ""
        for b in page.get_text("blocks"):
            if b[6] != 0:
                continue
            t = re.sub(r"\s+", " ", b[4]).strip()
            if not t or len(t) > 200:
                continue
            horiz = min(b[2], bbox[2]) - max(b[0], bbox[0]) > 0
            gap_below = b[1] - bbox[3]
            gap_above = bbox[1] - b[3]
            if horiz and (0 <= gap_below <= 40 or 0 <= gap_above <= 40):
                if re.match(r"(?i)(figure|fig\.|table)\s*[\w-]+", t):
                    return t
                best = best or t
        return best


def _table_cells(page, table) -> list[list[str]]:
    """Read each cell from its rectangle: keeps inter-word spaces that extract() can drop."""
    import pymupdf as fitz

    out = []
    try:
        for row in table.rows:
            out.append([re.sub(r"\s+", " ", page.get_textbox(fitz.Rect(c))).strip() if c else ""
                        for c in row.cells])
    except Exception:
        out = [[re.sub(r"\s+", " ", c or "").strip() for c in r] for r in table.extract()]
    return out


def _inside(path: Path, s: Settings) -> bool:
    try:
        Path(path).resolve().relative_to(s.data_dir.resolve())
        return True
    except ValueError:
        return False


# =============================================================== registry
def make_loaders(s: Settings | None = None, client_factory=None) -> list[Loader]:
    s = s or settings()
    out: list[Loader] = []
    for name in s.corpora:
        if name == "enginuity_t2":
            out.append(EnginuityT2Loader(s))
        elif name == "pdf":
            out.append(PdfLoader(s=s, client_factory=client_factory))
        else:
            raise ValueError(f"unknown corpus {name!r} in CORPORA (expected enginuity_t2, pdf)")
    return out


def load_all(s: Settings | None = None, client_factory=None) -> LoadResult:
    s = s or settings()
    res = LoadResult()
    for loader in make_loaders(s, client_factory):
        log.info("loading corpus %s", loader.name)
        res.extend(loader.load())
    # Provenance spot-check at birth: anything unstampable is rejected now.
    kept = []
    for u in res.units:
        missing = u.provenance.missing()
        if missing:
            res.rejected.append(Rejected(unit_id=u.unit_id, stage="extract",
                                         reason=f"missing provenance: {', '.join(missing)}",
                                         document_id=u.provenance.document_id, page=u.provenance.page))
        else:
            kept.append(u)
    res.units = kept
    return res


def page_class_counts(pages: list[dict]) -> dict[str, int]:
    return dict(Counter(p.get("class", "?") for p in pages))
