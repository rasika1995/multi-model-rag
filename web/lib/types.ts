// Mirrors rag/app/schemas.py. The UI holds no business logic: it renders what the API returns.

export type UnitType = "text" | "figure" | "table";

export interface Provenance {
  document_id: string | null;
  page: number | null;
  unit_id: string | null;
  bbox?: number[] | null;
  source_path?: string | null;
}

export interface Artefact {
  kind: "image" | "table" | "text";
  path?: string | null;
  columns?: string[] | null;
  rows?: string[][] | null;
  text?: string | null;
}

export interface Hit {
  rank: number;
  unit_id: string;
  type: UnitType;
  score: number;
  vector_score?: number | null;
  lexical_score?: number | null;
  sourced: boolean;
  citation: string | null;
  provenance: Provenance;
  search_text: string;
  artefact: Artefact;
  artefact_url?: string | null;
  cited: boolean;
}

export interface Answer {
  status: "answered" | "abstained";
  answer_type: UnitType | null;
  text: string;
  abstain_reason?: string | null;
  citation?: string | null;
  cited_unit_id?: string | null;
  artefact?: Artefact | null;
  artefact_url?: string | null;
  highlight_rows: number[];
  model?: string | null;
  generation_error?: string | null;
  dropped_unsourced: number;
}

export interface QueryResponse {
  question: string;
  answer: Answer;
  sources: Hit[];
  timings_ms: { retrieval: number; generation: number; total: number };
  calls: Record<string, number>;
  index: Record<string, unknown>;
}

export interface HealthProblem {
  kind: string;
  message: string;
  fix: string;
}

export interface Health {
  status: "healthy" | "degraded";
  problems: HealthProblem[];
  index: Record<string, unknown> | null;
  models: Record<string, string>;
  vector_backend: string;
}

export interface Example {
  question: string;
  type: string;
  id?: string;
}

export interface ApiErrorBody {
  kind: string;
  message: string;
  model?: string;
}
