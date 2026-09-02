import React from "react";
import type { ExplainResult } from "../types";
import { Brain, Sparkles, X } from "lucide-react";

interface AttentionHeatmapProps {
  explanation: ExplainResult | null;
  onClose: () => void;
}

export const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({ explanation, onClose }) => {
  if (!explanation) return null;

  const interactions = explanation.top_contributing_interactions || [];

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(0,0,0,0.75)",
        backdropFilter: "blur(6px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 100,
        padding: "20px",
      }}
      onClick={onClose}
    >
      <div
        className="glass-panel"
        style={{
          width: "100%",
          maxWidth: "650px",
          maxHeight: "90vh",
          overflowY: "auto",
          padding: "28px",
          position: "relative",
          border: "1px solid rgba(139, 92, 246, 0.4)",
          boxShadow: "0 16px 40px rgba(0,0,0,0.8)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: "absolute",
            top: "18px",
            right: "18px",
            background: "rgba(255,255,255,0.08)",
            border: "none",
            borderRadius: "50%",
            width: "32px",
            height: "32px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#fff",
            cursor: "pointer",
          }}
        >
          <X size={16} />
        </button>

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
          <div style={{ width: "36px", height: "36px", borderRadius: "8px", background: "rgba(139, 92, 246, 0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Brain size={20} color="#c084fc" />
          </div>
          <div>
            <h3 style={{ fontSize: "1.15rem", fontWeight: "700" }}>Recommendation Explanation</h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Algorithm: <strong style={{ color: "#38bdf8" }}>{explanation.algorithm.toUpperCase()}</strong> • Item: <strong style={{ color: "#fff" }}>{explanation.item_title}</strong>
            </p>
          </div>
        </div>

        {/* Natural Language Rationale */}
        <div
          style={{
            background: "rgba(139, 92, 246, 0.08)",
            border: "1px solid rgba(139, 92, 246, 0.25)",
            borderRadius: "10px",
            padding: "16px",
            marginBottom: "24px",
            display: "flex",
            gap: "12px",
          }}
        >
          <Sparkles size={18} color="#c084fc" style={{ flexShrink: 0, marginTop: "2px" }} />
          <p style={{ fontSize: "0.88rem", lineHeight: 1.6, color: "#e2e8f0" }}>
            {explanation.natural_language_explanation}
          </p>
        </div>

        {/* Attention Weights / Contributing Factors */}
        <div>
          <h4 style={{ fontSize: "0.92rem", fontWeight: "600", marginBottom: "12px", color: "var(--text-muted)" }}>
            Top Contributing Past Interactions (DIN Attention Activation)
          </h4>

          {interactions.length > 0 ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              {interactions.map((item, idx) => {
                const weight = item.attention_weight || item.similarity || 0.5;
                const weightPct = Math.round(weight * 100);

                return (
                  <div
                    key={idx}
                    style={{
                      background: "var(--bg-dark)",
                      border: "1px solid var(--border-color)",
                      borderRadius: "8px",
                      padding: "12px 14px",
                      display: "flex",
                      flexDirection: "column",
                      gap: "6px",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontSize: "0.88rem", fontWeight: "600" }}>{item.title}</span>
                      <span style={{ fontFamily: "monospace", fontSize: "0.82rem", fontWeight: "700", color: "#34d399" }}>
                        {weightPct}% Weight
                      </span>
                    </div>

                    <div style={{ width: "100%", height: "6px", background: "rgba(255,255,255,0.06)", borderRadius: "3px", overflow: "hidden" }}>
                      <div
                        style={{
                          width: `${weightPct}%`,
                          height: "100%",
                          background: "linear-gradient(to right, #8b5cf6, #10b981)",
                          borderRadius: "3px",
                          transition: "width 0.5s ease",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
              General collaborative affinity across latent preference clusters.
            </p>
          )}
        </div>

        {/* Feature Importance Breakdown */}
        {explanation.feature_importance && Object.keys(explanation.feature_importance).length > 0 && (
          <div style={{ marginTop: "24px" }}>
            <h4 style={{ fontSize: "0.92rem", fontWeight: "600", marginBottom: "12px", color: "var(--text-muted)" }}>
              Feature Importance Allocation
            </h4>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
              {Object.entries(explanation.feature_importance).map(([k, v]) => (
                <div key={k} style={{ background: "rgba(255,255,255,0.05)", padding: "6px 12px", borderRadius: "6px", fontSize: "0.78rem" }}>
                  <span style={{ color: "var(--text-muted)" }}>{k.replace(/_/g, " ")}: </span>
                  <strong style={{ color: "#38bdf8" }}>{(v * 100).toFixed(0)}%</strong>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
