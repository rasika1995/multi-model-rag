"use client";

import { useState } from "react";
import type { Hit } from "@/lib/types";

function SourceRow({ h }: { h: Hit }) {
  const [open, setOpen] = useState(false);
  return (
    <li className={`source ${h.cited ? "cited" : ""} ${h.sourced ? "" : "unsourced"}`}>
      <div className="source-head">
        <span className="rank">#{h.rank}</span>
        <span className={`tag tag-${h.type}`}>{h.type}</span>
        <span className="score" title={h.lexical_score != null ? `vector ${h.vector_score} · bm25 ${h.lexical_score}` : ""}>
          {h.score.toFixed(3)}
        </span>
        {h.cited && <span className="cited-mark">cited</span>}
      </div>
      {h.sourced ? (
        <code className="prov">{h.citation}</code>
      ) : (
        <code className="prov bad">no resolvable page: dropped, never cited</code>
      )}
      <button className="link" onClick={() => setOpen((o) => !o)}>
        {open ? "hide indexed text" : "show indexed text"}
      </button>
      {open && <pre className="snippet">{h.search_text}</pre>}
    </li>
  );
}

export default function SourcesPanel({ hits }: { hits: Hit[] }) {
  return (
    <aside className="panel sources" aria-label="Sources">
      <h2>Sources ({hits.length})</h2>
      <p className="hint">Everything retrieved for this answer, with provenance. Verify before you act.</p>
      <ol>
        {hits.map((h) => (
          <SourceRow key={h.unit_id} h={h} />
        ))}
      </ol>
    </aside>
  );
}
