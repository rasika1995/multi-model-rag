"use client";

import { api } from "@/lib/api";
import type { Artefact, QueryResponse } from "@/lib/types";

function TableArtefact({ a, highlight }: { a: Artefact; highlight: number[] }) {
  const hl = new Set(highlight);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {(a.columns || []).map((c, i) => (
              <th key={i}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(a.rows || []).map((r, i) => (
            <tr key={i} className={hl.has(i) ? "hl" : undefined}>
              {r.map((c, j) => (
                <td key={j}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function AnswerPanel({ res }: { res: QueryResponse }) {
  const a = res.answer;
  const art = a.artefact;
  const imgUrl = a.artefact_url ? api.artefactUrl(a.artefact_url) : null;
  const abstained = a.status === "abstained";

  return (
    <section className="panel answer" aria-label="Answer">
      <header className="answer-head">
        <span className={`pill pill-${a.status}`}>{abstained ? "Abstained" : "Answer"}</span>
        {a.answer_type && <span className={`tag tag-${a.answer_type}`}>{a.answer_type}</span>}
        <span className="meta">
          {Math.round(res.timings_ms.total)} ms · retrieval {Math.round(res.timings_ms.retrieval)} · generation{" "}
          {Math.round(res.timings_ms.generation)} · {res.calls.total ?? 0} model calls
        </span>
      </header>

      <p className="answer-text">{a.text}</p>
      {abstained && a.abstain_reason && <p className="reason">Why: {a.abstain_reason}</p>}
      {a.generation_error && <p className="warn">⚠ {a.generation_error}</p>}

      {a.citation && (
        <div className="citation">
          <span className="cite-label">{abstained ? "Closest source (not an answer)" : "Source"}</span>
          <code>{a.citation}</code>
        </div>
      )}

      {art?.kind === "image" && imgUrl && (
        <figure className="artefact">
          <a href={imgUrl} target="_blank" rel="noreferrer" title="Open full size">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={imgUrl} alt={`Figure ${a.citation}`} />
          </a>
          <figcaption>{a.citation} · click to open full size</figcaption>
        </figure>
      )}

      {art?.kind === "table" && (
        <figure className="artefact">
          <TableArtefact a={art} highlight={a.highlight_rows} />
          <figcaption>
            {a.citation}
            {a.highlight_rows.length > 0 && ` · ${a.highlight_rows.length} row(s) highlighted as relevant`}
            {imgUrl && (
              <>
                {" · "}
                <a href={imgUrl} target="_blank" rel="noreferrer">
                  view original page
                </a>
              </>
            )}
          </figcaption>
        </figure>
      )}

      {art?.kind === "text" && (
        <figure className="artefact">
          <blockquote>{art.text}</blockquote>
          <figcaption>{a.citation}</figcaption>
        </figure>
      )}
    </section>
  );
}
