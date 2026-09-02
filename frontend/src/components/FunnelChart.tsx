import React from "react";
import type { StageMetric } from "../types";
import { ArrowDown, Zap } from "lucide-react";

interface FunnelChartProps {
  metrics: StageMetric[];
  totalLatencyMs: number;
}

export const FunnelChart: React.FC<FunnelChartProps> = ({ metrics, totalLatencyMs }) => {
  if (!metrics || metrics.length === 0) {
    return (
      <div style={{ padding: "32px", textAlign: "center", color: "var(--text-muted)" }}>
        Execute the pipeline to view real-time stage funnel analytics.
      </div>
    );
  }

  const colors = ["#8b5cf6", "#06b6d4", "#f59e0b", "#10b981", "#e50914"];

  return (
    <div className="glass-panel" style={{ padding: "24px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
        <div>
          <h3 style={{ fontSize: "1.1rem", fontWeight: "700" }}>5-Stage Production Pipeline Funnel</h3>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Live candidate reduction, algorithmic transformations, and latency per stage
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "rgba(16, 185, 129, 0.1)", border: "1px solid rgba(16, 185, 129, 0.3)", padding: "6px 14px", borderRadius: "8px" }}>
          <Zap size={16} color="#34d399" />
          <span style={{ fontSize: "0.85rem", fontWeight: "600", color: "#34d399" }}>
            Total Pipeline Latency: {totalLatencyMs.toFixed(1)} ms
          </span>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        {metrics.map((stage, idx) => {
          const color = colors[idx % colors.length];
          const pct = Math.max(15, Math.min(100, (stage.output_count / (metrics[0].input_count || 1)) * 100));

          return (
            <div key={stage.stage_number} style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <div
                style={{
                  background: "var(--bg-dark)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "10px",
                  padding: "14px 18px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  position: "relative",
                  overflow: "hidden",
                }}
              >
                {/* Background Progress Bar */}
                <div
                  style={{
                    position: "absolute",
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: `${pct}%`,
                    background: `${color}15`,
                    borderRight: `3px solid ${color}`,
                    transition: "width 0.6s cubic-bezier(0.16, 1, 0.3, 1)",
                  }}
                />

                <div style={{ position: "relative", zIndex: 2, display: "flex", alignItems: "center", gap: "14px" }}>
                  <div
                    style={{
                      width: "32px",
                      height: "32px",
                      borderRadius: "8px",
                      background: `${color}25`,
                      color: color,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontWeight: "700",
                      fontSize: "0.85rem",
                    }}
                  >
                    0{stage.stage_number}
                  </div>
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <span style={{ fontWeight: "600", fontSize: "0.95rem" }}>{stage.stage_name}</span>
                      <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>• {stage.description}</span>
                    </div>
                    <div style={{ display: "flex", gap: "12px", marginTop: "4px", fontSize: "0.78rem" }}>
                      <span style={{ color: "var(--text-muted)" }}>Input: <strong style={{ color: "#fff" }}>{stage.input_count}</strong></span>
                      <span style={{ color: color }}>Output: <strong>{stage.output_count} candidates</strong></span>
                    </div>
                  </div>
                </div>

                <div style={{ position: "relative", zIndex: 2, textAlign: "right" }}>
                  <span style={{ fontFamily: "monospace", fontSize: "0.85rem", color: "#38bdf8", fontWeight: "600" }}>
                    {stage.latency_ms.toFixed(2)} ms
                  </span>
                </div>
              </div>

              {idx < metrics.length - 1 && (
                <div style={{ display: "flex", justifyContent: "center", margin: "-4px 0" }}>
                  <ArrowDown size={14} color="var(--text-muted)" opacity={0.6} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
