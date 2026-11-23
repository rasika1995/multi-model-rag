"use client";

import { FormEvent, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Example, UnitType } from "@/lib/types";

interface Props {
  busy: boolean;
  onAsk: (question: string, typeFilter: UnitType | null) => void;
}

export default function QueryBox({ busy, onAsk }: Props) {
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<"" | UnitType>("");
  const [examples, setExamples] = useState<Example[]>([]);

  useEffect(() => {
    api
      .examples()
      .then((r) => setExamples(r.examples))
      .catch(() => setExamples([]));
  }, []);

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    if (q.trim() && !busy) onAsk(q.trim(), filter || null);
  };

  return (
    <form className="querybox" onSubmit={submit}>
      <label htmlFor="q" className="sr-only">
        Question
      </label>
      <textarea
        id="q"
        value={q}
        rows={2}
        placeholder="Ask about a part, a diagram or a spec, e.g. “show the turbocharger assembly”"
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) submit();
        }}
      />
      <div className="qb-row">
        <select value={filter} onChange={(e) => setFilter(e.target.value as "" | UnitType)} aria-label="Content type">
          <option value="">Any content</option>
          <option value="figure">Figures only</option>
          <option value="table">Tables only</option>
          <option value="text">Text only</option>
        </select>
        <button type="submit" className="primary" disabled={busy || !q.trim()}>
          {busy ? "Searching…" : "Ask"}
        </button>
      </div>
      {examples.length > 0 && (
        <div className="examples">
          <span>Try:</span>
          {examples.map((ex, i) => (
            <button
              type="button"
              key={i}
              className="chip"
              disabled={busy}
              title={ex.id ? `${ex.id} · ${ex.type}` : ex.type}
              onClick={() => {
                setQ(ex.question);
                onAsk(ex.question, filter || null);
              }}
            >
              <span className={`tag tag-${ex.type}`}>{ex.type}</span>
              {ex.question.length > 90 ? ex.question.slice(0, 88) + "…" : ex.question}
            </button>
          ))}
        </div>
      )}
    </form>
  );
}
