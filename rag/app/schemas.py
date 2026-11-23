"""Data backbone: the Unit, its Provenance, and the API request/response types.

A Unit separates what we SEARCH (search_text) from what we RETURN (artefact),
tied together by provenance: document > page > unit id.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

UnitType = Literal["text", "figure", "table"]


class ProvenanceError(ValueError):
    """A unit cannot name where it came from. Fail closed."""


class Provenance(BaseModel):
    document_id: str | None = None
    page: int | None = None  # 1-based
    unit_id: str | None = None
    bbox: list[float] | None = None  # x0, y0, x1, y1 in PDF points
    source_path: str | None = None  # original file, relative to data dir

    def missing(self) -> list[str]:
        out = []
        if not self.document_id:
            out.append("document_id")
        if self.page is None or not isinstance(self.page, int) or self.page < 1:
            out.append("page")
        if not self.unit_id:
            out.append("unit_id")
        return out

    def is_sourced(self) -> bool:
        return not self.missing()

    def citation(self) -> str:
        return f"{self.document_id} > p.{self.page} > {self.unit_id}"


class Artefact(BaseModel):
    """What we hand back. Exactly one of path / rows / text is the payload."""

    kind: Literal["image", "table", "text"]
    path: str | None = None  # image file, relative to data dir
    columns: list[str] | None = None
    rows: list[list[str]] | None = None
    text: str | None = None


class Unit(BaseModel):
    unit_id: str
    type: UnitType
    search_text: str = ""
    artefact: Artefact
    provenance: Provenance
    meta: dict[str, Any] = Field(default_factory=dict)

    def validate_provenance(self) -> None:
        missing = self.provenance.missing()
        if missing:
            raise ProvenanceError(f"unit {self.unit_id!r} missing provenance: {', '.join(missing)}")
        if self.provenance.unit_id != self.unit_id:
            raise ProvenanceError(
                f"unit {self.unit_id!r} provenance unit_id mismatch ({self.provenance.unit_id!r})"
            )


class Rejected(BaseModel):
    unit_id: str | None
    stage: str
    reason: str
    document_id: str | None = None
    page: int | None = None


# --------------------------------------------------------------------- API
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=50)
    type_filter: UnitType | None = None
    generate: bool = True


class RetrieveBatchRequest(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=50)
    type_filter: UnitType | None = None


class Hit(BaseModel):
    rank: int
    unit_id: str
    type: UnitType
    score: float
    vector_score: float | None = None
    lexical_score: float | None = None
    sourced: bool
    citation: str | None
    provenance: Provenance
    search_text: str
    artefact: Artefact
    artefact_url: str | None = None
    cited: bool = False


class Answer(BaseModel):
    status: Literal["answered", "abstained"]
    answer_type: UnitType | None = None
    text: str
    abstain_reason: str | None = None
    citation: str | None = None
    cited_unit_id: str | None = None
    artefact: Artefact | None = None
    artefact_url: str | None = None
    highlight_rows: list[int] = Field(default_factory=list)
    model: str | None = None
    generation_error: str | None = None
    dropped_unsourced: int = 0


class QueryResponse(BaseModel):
    question: str
    answer: Answer
    sources: list[Hit]
    timings_ms: dict[str, float]
    calls: dict[str, int]
    index: dict[str, Any]
