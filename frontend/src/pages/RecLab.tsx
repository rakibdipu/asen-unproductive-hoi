import React, { useState, useEffect } from "react";
import type { UserProfile, ItemScore, ExplainResult } from "../types";
import { fetchUsers, recommendSingle, explainRecommendation, ingestEvent } from "../api";
import { ItemGrid } from "../components/ItemGrid";
import { AttentionHeatmap } from "../components/AttentionHeatmap";
import { Play, Clock } from "lucide-react";

interface RecLabProps {
  domain: string;
}

export const RecLab: React.FC<RecLabProps> = ({ domain }) => {
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [selectedUser, setSelectedUser] = useState<string>("usr_0001");
  const [selectedAlgo, setSelectedAlgo] = useState<string>("sasrec");
  const [topN, setTopN] = useState<number>(10);
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<ItemScore[]>([]);
  const [latency, setLatency] = useState<number>(0);
  const [explanation, setExplanation] = useState<ExplainResult | null>(null);

  const algorithms = [
    { id: "popularity", name: "Time-Decayed Popularity (Baseline)", gen: "Gen 1" },
    { id: "item_cf", name: "Item-Item CF (Amazon 2001)", gen: "Gen 1" },
    { id: "implicit_als", name: "Implicit ALS (Hu-Koren 2008)", gen: "Gen 2" },
    { id: "bpr_mf", name: "BPR-MF Pairwise Ranking (Rendle 2009)", gen: "Gen 2" },
    { id: "two_tower", name: "Two-Tower Deep Retrieval (YouTube 2016)", gen: "Gen 3" },
    { id: "wide_deep", name: "Wide & Deep Learning (Google 2016)", gen: "Gen 4" },
    { id: "deepfm", name: "DeepFM Shared Interactions (Huawei 2017)", gen: "Gen 4" },
    { id: "din", name: "DIN Target Attention (Alibaba 2018)", gen: "Gen 4" },
    { id: "mmoe", name: "MMoE Multi-Task Utility (Google 2018)", gen: "Gen 4" },
    { id: "sasrec", name: "SASRec Transformer Sequential (UCSD 2018)", gen: "Gen 5" },
    { id: "lightgcn", name: "LightGCN Graph Diffusion (NUS 2020)", gen: "Gen 5" },
    { id: "llm_reranker", name: "LLM Semantic Recommender (2026)", gen: "Gen 6" },
  ];

  useEffect(() => {
    fetchUsers(domain).then((data) => {
      setUsers(data);
      if (data.length > 0) {
        setSelectedUser(data[0].user_id);
      }
    });
  }, [domain]);

  const handleRun = async () => {
    setLoading(true);
    try {
      const res = await recommendSingle(selectedUser, selectedAlgo, domain, topN);
      setResults(res.items);
      setLatency(res.latency_ms);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedUser) {
      handleRun();
    }
  }, [selectedUser, selectedAlgo, domain]);

  const handleExplain = async (item: ItemScore) => {
    try {
      const exp = await explainRecommendation(selectedUser, item.item_id, selectedAlgo, domain);
      setExplanation(exp);
    } catch (e) {
      console.error(e);
    }
  };

  const handleInteract = async (item: ItemScore, eventType: string) => {
    await ingestEvent({
      event_id: `ev_${Date.now()}`,
      user_id: selectedUser,
      item_id: item.item_id,
      event_type: eventType,
      timestamp: Date.now() / 1000,
      artwork_variant_id: item.artwork_variant_id,
      context: { genre: item.category },
    });
    // Refresh to show real-time adaptation
    handleRun();
  };

  const currentUserObj = users.find((u) => u.user_id === selectedUser);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Control Header */}
      <div className="glass-panel" style={{ padding: "20px 24px" }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "18px", alignItems: "center", justifyContent: "space-between" }}>
          {/* User Select */}
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: "600" }}>ACTIVE USER PERSONA</label>
            <select
              value={selectedUser}
              onChange={(e) => setSelectedUser(e.target.value)}
              style={{
                background: "var(--bg-dark)",
                color: "#fff",
                border: "1px solid var(--border-color)",
                padding: "8px 12px",
                borderRadius: "8px",
                fontSize: "0.85rem",
                outline: "none",
                minWidth: "220px",
              }}
            >
              {users.map((u) => (
                <option key={u.user_id} value={u.user_id}>
                  {u.username} ({u.persona})
                </option>
              ))}
            </select>
          </div>

          {/* Algorithm Select */}
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: "600" }}>ALGORITHM ENGINE (12 MODELS)</label>
            <select
              value={selectedAlgo}
              onChange={(e) => setSelectedAlgo(e.target.value)}
              style={{
                background: "var(--bg-dark)",
                color: "#fff",
                border: "1px solid var(--border-color)",
                padding: "8px 12px",
                borderRadius: "8px",
                fontSize: "0.85rem",
                outline: "none",
                minWidth: "300px",
              }}
            >
              {algorithms.map((a) => (
                <option key={a.id} value={a.id}>
                  [{a.gen}] {a.name}
                </option>
              ))}
            </select>
          </div>

          {/* Top N Count */}
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <label style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: "600" }}>RECOMMENDATION COUNT</label>
            <div style={{ display: "flex", gap: "6px" }}>
              {[5, 10, 15, 20].map((n) => (
                <button
                  key={n}
                  onClick={() => setTopN(n)}
                  style={{
                    padding: "6px 12px",
                    borderRadius: "6px",
                    border: "1px solid var(--border-color)",
                    background: topN === n ? "#e50914" : "var(--bg-dark)",
                    color: topN === n ? "#fff" : "var(--text-muted)",
                    fontSize: "0.8rem",
                    cursor: "pointer",
                  }}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Trigger Button */}
          <div style={{ display: "flex", alignItems: "flex-end" }}>
            <button className="btn btn-primary" onClick={handleRun} disabled={loading}>
              <Play size={16} />
              <span>{loading ? "Wasting Time..." : "🛋️ Waste My Time"}</span>
            </button>
          </div>
        </div>

        {/* Selected User Persona Bar */}
        {currentUserObj && (
          <div style={{ marginTop: "16px", paddingTop: "14px", borderTop: "1px solid var(--border-color)", display: "flex", gap: "16px", alignItems: "center", flexWrap: "wrap", fontSize: "0.8rem" }}>
            <span style={{ color: "var(--text-muted)" }}>Target Persona: <strong style={{ color: "#38bdf8" }}>{currentUserObj.persona}</strong></span>
            <span style={{ color: "var(--text-muted)" }}>Affinity: <strong style={{ color: "#fff" }}>{Array.isArray(currentUserObj.fav_genres) ? currentUserObj.fav_genres.join(", ") : currentUserObj.fav_genres}</strong></span>
            <span style={{ color: "var(--text-muted)" }}>Activity: <span className="badge badge-green">{currentUserObj.activity_level.toUpperCase()}</span></span>
            {latency > 0 && (
              <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "4px", color: "#34d399", fontFamily: "monospace" }}>
                <Clock size={12} /> Latency: {latency.toFixed(2)} ms
              </span>
            )}
          </div>
        )}
      </div>

      {/* Recommended Items Grid */}
      <ItemGrid
        items={results}
        onExplain={handleExplain}
        onInteract={handleInteract}
        title={`Top Picks Generated by ${selectedAlgo.toUpperCase()}`}
        subtitle="Hover to inspect scores and variant artwork. Click 'Why?' to view attention heatmaps."
      />

      {/* Attention / Explanation Modal */}
      <AttentionHeatmap explanation={explanation} onClose={() => setExplanation(null)} />
    </div>
  );
};
