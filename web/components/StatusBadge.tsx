"use client";

import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { Health } from "@/lib/types";

export default function StatusBadge() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setHealth(await api.health());
      setError(null);
    } catch (e) {
      setHealth(null);
      setError(e instanceof ApiError ? e.readable : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 30000);
    return () => clearInterval(t);
  }, [refresh]);

  const state = error ? "down" : health?.status ?? "checking";
  const label = { healthy: "Healthy", degraded: "Degraded", down: "API down", checking: "Checking…" }[state];
  const index = health?.index as { units_indexed?: number; embed_model?: string } | null | undefined;

  return (
    <div className="status">
      <button className={`badge badge-${state}`} onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className="dot" /> {label}
      </button>
      {open && (
        <div className="status-pop" role="dialog">
          {error && <p className="err">{error}</p>}
          {health && (
            <>
              {health.problems.length === 0 && <p>All configured models passed the last readiness check.</p>}
              {health.problems.map((p, i) => (
                <div key={i} className="problem">
                  <strong>{p.kind}</strong>: {p.message}
                  <div className="fix">Fix: {p.fix}</div>
                </div>
              ))}
              <dl>
                <dt>Vector DB</dt>
                <dd>{health.vector_backend}</dd>
                <dt>Index</dt>
                <dd>{index ? `${index.units_indexed} units · ${index.embed_model}` : "none"}</dd>
                {Object.entries(health.models).map(([k, v]) => (
                  <div key={k} className="dl-row">
                    <dt>{k}</dt>
                    <dd>{v}</dd>
                  </div>
                ))}
              </dl>
            </>
          )}
          <button className="link" onClick={refresh}>
            Re-check
          </button>
        </div>
      )}
    </div>
  );
}
