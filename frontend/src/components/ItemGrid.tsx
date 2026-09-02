import React from "react";
import type { ItemScore } from "../types";
import { ThumbsUp, Play, FastForward, Sparkles } from "lucide-react";

interface ItemGridProps {
  items: ItemScore[];
  onExplain?: (item: ItemScore) => void;
  onInteract?: (item: ItemScore, eventType: string) => void;
  title?: string;
  subtitle?: string;
}

export const ItemGrid: React.FC<ItemGridProps> = ({
  items,
  onExplain,
  onInteract,
  title,
  subtitle,
}) => {
  if (!items || items.length === 0) {
    return (
      <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)", background: "var(--bg-card)", borderRadius: "12px", border: "1px dashed var(--border-color)" }}>
        No recommended items found for this selection.
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      {(title || subtitle) && (
        <div>
          {title && <h3 style={{ fontSize: "1.2rem", fontWeight: "700", color: "#fff" }}>{title}</h3>}
          {subtitle && <p style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>{subtitle}</p>}
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
          gap: "18px",
        }}
      >
        {items.map((item, idx) => {
          const scoreDisplay = (item.score * 100).toFixed(1);
          const isHighMatch = item.score > 0.75;

          return (
            <div
              key={item.item_id || idx}
              className="glass-panel"
              style={{
                borderRadius: "12px",
                overflow: "hidden",
                transition: "all 0.25s ease",
                position: "relative",
                display: "flex",
                flexDirection: "column",
                border: "1px solid var(--border-color)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "translateY(-4px)";
                e.currentTarget.style.borderColor = "rgba(229, 9, 20, 0.5)";
                e.currentTarget.style.boxShadow = "0 8px 24px rgba(0,0,0,0.5)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.borderColor = "var(--border-color)";
                e.currentTarget.style.boxShadow = "none";
              }}
            >
              {/* Poster Image Container */}
              <div style={{ position: "relative", width: "100%", height: "260px", background: "#1a2234" }}>
                <img
                  src={item.thumbnail_url || "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=400&q=80"}
                  alt={item.title}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                  loading="lazy"
                  onError={(e) => {
                    e.currentTarget.src = "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&q=80";
                  }}
                />

                {/* Dark Gradient Overlay */}
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    background: "linear-gradient(to top, rgba(10,13,20,0.95) 0%, rgba(10,13,20,0.2) 60%, transparent 100%)",
                  }}
                />

                {/* Rank Badge */}
                <div
                  style={{
                    position: "absolute",
                    top: "10px",
                    left: "10px",
                    background: "rgba(0, 0, 0, 0.75)",
                    backdropFilter: "blur(6px)",
                    border: "1px solid rgba(255,255,255,0.15)",
                    borderRadius: "6px",
                    padding: "2px 8px",
                    fontSize: "0.75rem",
                    fontWeight: "800",
                    color: "#fff",
                  }}
                >
                  #{item.rank || idx + 1}
                </div>

                {/* Score Tag */}
                <div
                  style={{
                    position: "absolute",
                    top: "10px",
                    right: "10px",
                    background: isHighMatch ? "rgba(16, 185, 129, 0.9)" : "rgba(6, 182, 212, 0.9)",
                    borderRadius: "6px",
                    padding: "2px 7px",
                    fontSize: "0.72rem",
                    fontWeight: "700",
                    color: "#000",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.4)",
                  }}
                >
                  {scoreDisplay}% Match
                </div>

                {/* Artwork Variant Tag (Netflix bandit) */}
                {item.artwork_variant_id && (
                  <div
                    style={{
                      position: "absolute",
                      bottom: "10px",
                      left: "10px",
                      background: "rgba(229, 9, 20, 0.85)",
                      borderRadius: "4px",
                      padding: "2px 6px",
                      fontSize: "0.65rem",
                      fontWeight: "700",
                      color: "#fff",
                    }}
                  >
                    ARTWORK: {item.artwork_variant_id.replace("variant_", "").toUpperCase()}
                  </div>
                )}
              </div>

              {/* Item Info Body */}
              <div style={{ padding: "14px", display: "flex", flexDirection: "column", flexGrow: 1, justifyContent: "space-between", gap: "10px" }}>
                <div>
                  <h4 style={{ fontSize: "0.95rem", fontWeight: "700", color: "#fff", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {item.title}
                  </h4>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "4px" }}>
                    <span className="badge badge-purple" style={{ fontSize: "0.68rem" }}>
                      {item.category || "General"}
                    </span>
                    {item.source_model && (
                      <span className="badge badge-cyan" style={{ fontSize: "0.68rem" }}>
                        {item.source_model}
                      </span>
                    )}
                  </div>
                </div>

                {/* Interactive Simulation Actions */}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderTop: "1px solid var(--border-color)", paddingTop: "10px" }}>
                  <div style={{ display: "flex", gap: "4px" }}>
                    <button
                      title="Simulate Click & Watch"
                      onClick={() => onInteract && onInteract(item, "watch_complete")}
                      style={{ background: "rgba(255,255,255,0.08)", border: "none", borderRadius: "6px", padding: "6px", color: "#fff", cursor: "pointer" }}
                    >
                      <Play size={14} />
                    </button>
                    <button
                      title="Simulate Thumbs Up"
                      onClick={() => onInteract && onInteract(item, "like")}
                      style={{ background: "rgba(255,255,255,0.08)", border: "none", borderRadius: "6px", padding: "6px", color: "#34d399", cursor: "pointer" }}
                    >
                      <ThumbsUp size={14} />
                    </button>
                    <button
                      title="Simulate Skip (< 10s)"
                      onClick={() => onInteract && onInteract(item, "skip")}
                      style={{ background: "rgba(255,255,255,0.08)", border: "none", borderRadius: "6px", padding: "6px", color: "#f87171", cursor: "pointer" }}
                    >
                      <FastForward size={14} />
                    </button>
                  </div>

                  {onExplain && (
                    <button
                      onClick={() => onExplain(item)}
                      style={{
                        background: "rgba(139, 92, 246, 0.15)",
                        border: "1px solid rgba(139, 92, 246, 0.4)",
                        borderRadius: "6px",
                        padding: "4px 8px",
                        color: "#c084fc",
                        fontSize: "0.72rem",
                        fontWeight: "600",
                        display: "flex",
                        alignItems: "center",
                        gap: "4px",
                        cursor: "pointer",
                      }}
                    >
                      <Sparkles size={12} />
                      <span>Why?</span>
                    </button>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
