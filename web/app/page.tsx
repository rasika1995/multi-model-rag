"use client";

import { useEffect, useRef, useState } from "react";
import AnswerPanel from "@/components/AnswerPanel";
import QueryBox from "@/components/QueryBox";
import SourcesPanel from "@/components/SourcesPanel";
import StatusBadge from "@/components/StatusBadge";
import { api, ApiError } from "@/lib/api";
import type { QueryResponse, UnitType } from "@/lib/types";

export default function Home() {
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => () => {
    if (timer.current) clearInterval(timer.current);
  }, []);

  const ask = async (question: string, typeFilter: UnitType | null) => {
    setBusy(true);
    setError(null);
    setElapsed(0);
    const t0 = Date.now();
    timer.current = setInterval(() => setElapsed((Date.now() - t0) / 1000), 200);
    try {
      setRes(await api.query(question, typeFilter));
    } catch (e) {
      setRes(null);
      setError(e instanceof ApiError ? e.readable : String(e));
    } finally {
      if (timer.current) clearInterval(timer.current);
      setBusy(false);
    }
  };

  return (
    <main>
      <header className="top">
        <div>
          <h1>Manual Finder</h1>
          <p className="sub">Ask a maintenance question. Get the real figure or table rows, with the page it came from.</p>
        </div>
        <StatusBadge />
      </header>

      <QueryBox busy={busy} onAsk={ask} />

      {busy && (
        <div className="loading" role="status">
          <span className="spinner" /> Retrieving and answering… {elapsed.toFixed(1)} s
          <span className="hint"> (free-tier models can take 10–30 s)</span>
        </div>
      )}
      {error && !busy && (
        <div className="error" role="alert">
          <strong>Request failed.</strong> {error}
        </div>
      )}
      {res && !busy && (
        <div className="results">
          <AnswerPanel res={res} />
          <SourcesPanel hits={res.sources} />
        </div>
      )}
      {!res && !busy && !error && (
        <p className="empty">Pick an example above or type a question. Answers always show their source and page.</p>
      )}
    </main>
  );
}
