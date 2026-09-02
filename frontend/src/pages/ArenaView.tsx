import React, { useState, useEffect } from "react";
import type { ArenaBenchmarkResponse } from "../types";
import { runTournament } from "../api";
import { Trophy, Sparkles, RefreshCw } from "lucide-react";

interface ArenaViewProps {
  domain: string;
}

export const ArenaView: React.FC<ArenaViewProps> = ({ domain }) => {
  const [data, setData] = useState<ArenaBenchmarkResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [metricSort, setMetricSort] = useState<string>("ndcg_at_10");

  const runBenchmark = async () => {
    setLoading(true);
    try {
      const res = await runTournament(domain, undefined, 10, 25);
      setData(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runBenchmark();
  }, [domain]);

  const sortedModels = data ? [...data.models].sort((a: any, b: any) => b[metricSort] - a[metricSort]) : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
      {/* Header */}
      <div className="glass-panel" style={{ padding: "24px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "16px" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <Trophy size={22} color="#f59e0b" />
            <h2 style={{ fontSize: "1.3rem", fontWeight: "700" }}>Model Arena: Algorithmic Tournament</h2>
          </div>
          <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", marginTop: "4px" }}>
            Rigorous head-to-head empirical evaluation on holdout test splits (NDCG@10, MAP, Coverage, Diversity & Latency)
          </p>
        </div>

        <button className="btn btn-primary" onClick={runBenchmark} disabled={loading}>
          <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
          <span>{loading ? "Simulating Tournament..." : "Re-run Tournament"}</span>
        </button>
      </div>

      {data && (
        <>
          {/* Executive Summary Callout */}
          <div style={{ background: "rgba(245, 158, 11, 0.1)", border: "1px solid rgba(245, 158, 11, 0.3)", padding: "16px 20px", borderRadius: "10px", display: "flex", gap: "12px", alignItems: "center" }}>
            <Sparkles size={20} color="#f59e0b" style={{ flexShrink: 0 }} />
            <p style={{ fontSize: "0.88rem", color: "#fef3c7", lineHeight: 1.5 }}>
              {data.summary}
            </p>
          </div>

          {/* Tournament Metrics Table */}
          <div className="glass-panel" style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: "0.85rem" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border-color)", background: "rgba(0,0,0,0.3)" }}>
                  <th style={{ padding: "14px 18px", color: "var(--text-muted)", fontWeight: "600" }}>RANK & MODEL</th>
                  <th style={{ padding: "14px 18px", color: "var(--text-muted)", fontWeight: "600" }}>ORIGIN</th>
                  <th
                    style={{ padding: "14px 18px", color: metricSort === "ndcg_at_10" ? "#38bdf8" : "var(--text-muted)", cursor: "pointer" }}
                    onClick={() => setMetricSort("ndcg_at_10")}
                  >
                    NDCG@10 ⬍
                  </th>
                  <th
                    style={{ padding: "14px 18px", color: metricSort === "hit_rate_at_10" ? "#38bdf8" : "var(--text-muted)", cursor: "pointer" }}
                    onClick={() => setMetricSort("hit_rate_at_10")}
                  >
                    HIT RATE ⬍
                  </th>
                  <th
                    style={{ padding: "14px 18px", color: metricSort === "diversity_ils" ? "#38bdf8" : "var(--text-muted)", cursor: "pointer" }}
                    onClick={() => setMetricSort("diversity_ils")}
                  >
                    DIVERSITY (ILS) ⬍
                  </th>
                  <th
                    style={{ padding: "14px 18px", color: metricSort === "catalog_coverage" ? "#38bdf8" : "var(--text-muted)", cursor: "pointer" }}
                    onClick={() => setMetricSort("catalog_coverage")}
                  >
                    COVERAGE ⬍
                  </th>
                  <th
                    style={{ padding: "14px 18px", color: metricSort === "avg_latency_ms" ? "#38bdf8" : "var(--text-muted)", cursor: "pointer" }}
                    onClick={() => setMetricSort("avg_latency_ms")}
                  >
                    LATENCY ⬍
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedModels.map((m, idx) => {
                  const isTop = idx === 0;
                  return (
                    <tr
                      key={m.model_key}
                      style={{
                        borderBottom: "1px solid rgba(31, 41, 61, 0.4)",
                        background: isTop ? "rgba(245, 158, 11, 0.05)" : "transparent",
                      }}
                    >
                      <td style={{ padding: "14px 18px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                          <span style={{ fontWeight: "700", color: isTop ? "#f59e0b" : "var(--text-muted)", width: "20px" }}>
                            #{idx + 1}
                          </span>
                          <div>
                            <div style={{ fontWeight: "600", color: "#fff" }}>{m.model_name}</div>
                            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{m.generation}</div>
                          </div>
                        </div>
                      </td>
                      <td style={{ padding: "14px 18px" }}>
                        <span className="badge badge-purple" style={{ fontSize: "0.72rem" }}>
                          {m.company} ({m.year})
                        </span>
                      </td>
                      <td style={{ padding: "14px 18px", fontWeight: "700", color: "#34d399", fontFamily: "monospace" }}>
                        {m.ndcg_at_10.toFixed(3)}
                      </td>
                      <td style={{ padding: "14px 18px", color: "#38bdf8", fontFamily: "monospace" }}>
                        {(m.hit_rate_at_10 * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: "14px 18px", color: "#c084fc", fontFamily: "monospace" }}>
                        {(m.diversity_ils * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: "14px 18px", color: "#94a3b8", fontFamily: "monospace" }}>
                        {(m.catalog_coverage * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: "14px 18px", color: m.avg_latency_ms < 3.0 ? "#34d399" : "#f59e0b", fontFamily: "monospace" }}>
                        {m.avg_latency_ms.toFixed(2)} ms
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
};
