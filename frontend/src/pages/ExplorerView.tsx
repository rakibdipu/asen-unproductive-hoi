import React, { useState } from "react";
import type { ItemScore } from "../types";
import { recommendSingle } from "../api";
import { ItemGrid } from "../components/ItemGrid";
import { Brain, Search, Sparkles } from "lucide-react";

interface ExplorerViewProps {
  domain: string;
}

export const ExplorerView: React.FC<ExplorerViewProps> = ({ domain }) => {
  const [query, setQuery] = useState<string>("Mind-bending sci-fi cyberpunk mystery with deep philosophical concepts");
  const [results, setResults] = useState<ItemScore[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const handleSearch = async () => {
    setLoading(true);
    try {
      const res = await recommendSingle("usr_0001", "llm_reranker", domain, 10, { query });
      setResults(res.items);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
      {/* Header */}
      <div className="glass-panel" style={{ padding: "24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
          <span style={{ fontSize: "1.5rem" }}>🛋️</span>
          <h2 style={{ fontSize: "1.3rem", fontWeight: "700" }}>Find a Distraction (Semantic Search)</h2>
        </div>
        <p style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
          কিছু একটা দেখে সময় নষ্ট করতে চান? মনের মতো প্লট, ফিলিং বা মুডের কথা লিখুন —
          আধুনিক জেনারেটিভ এআই মডেল আপনাকে সেরা ডিস্ট্র্যাকশন খুঁজে দেবে।
        </p>

        {/* Search Input Box */}
        <div style={{ display: "flex", gap: "10px", marginTop: "20px" }}>
          <div style={{ position: "relative", flexGrow: 1 }}>
            <Search size={16} color="var(--text-muted)" style={{ position: "absolute", left: "14px", top: "50%", transform: "translateY(-50%)" }} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="কী দিয়ে সময় নষ্ট করতে চান? (যেমন: Mind-bending sci-fi, intense crime thriller)..."
              style={{
                width: "100%",
                background: "var(--bg-dark)",
                border: "1px solid var(--border-color)",
                borderRadius: "10px",
                padding: "12px 14px 12px 42px",
                color: "#fff",
                fontSize: "0.9rem",
                outline: "none",
              }}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            />
          </div>
          <button className="btn btn-primary" onClick={handleSearch} disabled={loading}>
            <Sparkles size={16} />
            <span>{loading ? "Searching..." : "🛋️ Find a Distraction"}</span>
          </button>
        </div>

        {/* Quick Prompt Ideas */}
        <div style={{ display: "flex", gap: "8px", marginTop: "12px", flexWrap: "wrap", alignItems: "center" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Try concepts:</span>
          {[
            "Atmospheric dark crime neo-noir in Gotham",
            "High school anime romance with time travel",
            "Epic planetary desert sci-fi warfare",
            "Classic 90s mobster drama",
          ].map((prompt) => (
            <button
              key={prompt}
              onClick={() => {
                setQuery(prompt);
              }}
              style={{
                background: "rgba(255,255,255,0.05)",
                border: "1px solid var(--border-color)",
                borderRadius: "6px",
                padding: "4px 10px",
                fontSize: "0.75rem",
                color: "#94a3b8",
                cursor: "pointer",
              }}
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Results */}
      {results.length > 0 && (
        <ItemGrid
          items={results}
          title="Semantic Zero-Shot Recommendations"
          subtitle="Ranked by dense semantic textual embedding resonance"
        />
      )}
    </div>
  );
};
