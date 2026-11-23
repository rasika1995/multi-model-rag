import type { ApiErrorBody, Example, Health, QueryResponse, UnitType } from "./types";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(public status: number, public body: ApiErrorBody) {
    super(body.message);
  }

  /** A sentence an engineer can act on. */
  get readable(): string {
    const b = this.body;
    switch (b.kind) {
      case "model_unavailable":
        return `The model ${b.model} is unavailable (slug retired or down). Change it in rag/.env and re-run check_models.`;
      case "upstream":
      case "empty_completion":
        return `The model provider failed${b.model ? ` for ${b.model}` : ""}: ${b.message}`;
      case "index_missing":
        return "No index has been built yet. Run: python scripts/build_index.py";
      case "index_mismatch":
        return `The index was built with a different embedding model. ${b.message}`;
      case "configuration":
        return `Configuration problem: ${b.message}`;
      case "network":
        return `Cannot reach the API at ${API_BASE}. Is the service running (uvicorn app.server:app --port 8000)?`;
      default:
        return b.message || `Request failed (${this.status})`;
    }
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    });
  } catch {
    throw new ApiError(0, { kind: "network", message: "network error" });
  }
  if (!res.ok) {
    let body: ApiErrorBody = { kind: "http", message: `HTTP ${res.status}` };
    try {
      const j = await res.json();
      if (j?.error) body = j.error;
      else if (j?.detail) body = { kind: "http", message: typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail) };
    } catch {
      /* keep default */
    }
    throw new ApiError(res.status, body);
  }
  return res.json() as Promise<T>;
}

export const api = {
  query: (question: string, typeFilter: UnitType | null) =>
    request<QueryResponse>("/query", {
      method: "POST",
      body: JSON.stringify({ question, type_filter: typeFilter }),
    }),
  health: () => request<Health>("/health"),
  examples: () => request<{ examples: Example[] }>("/examples"),
  artefactUrl: (path: string) => `${API_BASE}${path}`,
};
