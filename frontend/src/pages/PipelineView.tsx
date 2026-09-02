import React, { useState, useEffect } from "react";
import type { UserProfile, PipelineResult, ExplainResult, ItemScore } from "../types";
import { fetchUsers, recommendPipeline, explainRecommendation, ingestEvent } from "../api";
import { FunnelChart } from "../components/FunnelChart";
import { ItemGrid } from "../components/ItemGrid";
import { AttentionHeatmap } from "../components/AttentionHeatmap";
import { Play } from "lucide-react";

interface PipelineViewProps {
  domain: string;
}

export const PipelineView: React.FC<PipelineViewProps> = ({ domain }) => {
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [selectedUser, setSelectedUser] = useState<string>("usr_0001");
  const [loading, setLoading] = useState<boolean>(false);
  const [pipelineData, setPipelineData] = useState<PipelineResult | null>(null);
  const [explanation, setExplanation] = useState<ExplainResult | null>(null);

  useEffect(() => {
    fetchUsers(domain).then((data) => {
      setUsers(data);
      if (data.length > 0) {
        setSelectedUser(data[0].user_id);
      }
    });
  }, [domain]);

  const handleRunPipeline = async () => {
    setLoading(true);
    try {
      const res = await recommendPipeline(selectedUser, domain);
      setPipelineData(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedUser) {
      handleRunPipeline();
    }
  }, [selectedUser, domain]);

  const handleExplain = async (item: ItemScore) => {
    try {
      const exp = await explainRecommendation(selectedUser, item.item_id, "din", domain);
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
    handleRunPipeline();
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      {/* Control Header */}
      <div className="glass-panel" style={{ padding: "20px 24px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "16px" }}>
        <div>
          <h2 style={{ fontSize: "1.25rem", fontWeight: "700" }}>End-to-End Production Pipeline</h2>
          <p style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
            Execute the complete multi-stage funnel: Retrieval ➔ Filtering ➔ Heavy Scoring ➔ MMR Diversity ➔ Slate & Artwork Bandit
          </p>
        </div>

        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
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
            }}
          >
            {users.map((u) => (
              <option key={u.user_id} value={u.user_id}>
                {u.username} ({u.persona})
              </option>
            ))}
          </select>

          <button className="btn btn-primary" onClick={handleRunPipeline} disabled={loading}>
            <Play size={16} />
            <span>{loading ? "Streaming Funnel..." : "Execute Pipeline"}</span>
          </button>
        </div>
      </div>

      {/* Funnel Visualizer */}
      {pipelineData && (
        <FunnelChart
          metrics={pipelineData.stage_metrics}
          totalLatencyMs={pipelineData.total_latency_ms}
        />
      )}

      {/* Dynamic Netflix-Style Rows */}
      {pipelineData && pipelineData.slate_page && (
        <div style={{ display: "flex", flexDirection: "column", gap: "36px" }}>
          <div style={{ borderBottom: "1px solid var(--border-color)", paddingBottom: "12px" }}>
            <h3 style={{ fontSize: "1.3rem", fontWeight: "800", color: "#fff" }}>
              Personalized Homepage Slate (Netflix Architecture)
            </h3>
            <p style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
              Each row generated with distinct algorithmic heuristics and personalized LinUCB artwork thumbnails.
            </p>
          </div>

          {pipelineData.slate_page.rows.map((row) => (
            <div key={row.row_id}>
              <ItemGrid
                items={row.items}
                onExplain={handleExplain}
                onInteract={handleInteract}
                title={row.title}
                subtitle={row.description}
              />
            </div>
          ))}
        </div>
      )}

      {/* Explanation Modal */}
      <AttentionHeatmap explanation={explanation} onClose={() => setExplanation(null)} />
    </div>
  );
};
