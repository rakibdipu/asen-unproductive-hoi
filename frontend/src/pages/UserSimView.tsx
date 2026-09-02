import React, { useState, useEffect } from "react";
import type { ItemScore } from "../types";
import { fetchItems, recommendSingle, ingestEvent } from "../api";
import { ItemGrid } from "../components/ItemGrid";
import { UserCheck, Activity } from "lucide-react";

interface UserSimViewProps {
  domain: string;
}

export const UserSimView: React.FC<UserSimViewProps> = ({ domain }) => {
  const selectedUser = "usr_0001";
  const [sessionEvents, setSessionEvents] = useState<any[]>([]);
  const [recommendations, setRecommendations] = useState<ItemScore[]>([]);
  const [catalogItems, setCatalogItems] = useState<any[]>([]);
  const [isAutoPlaying, setIsAutoPlaying] = useState<boolean>(false);

  useEffect(() => {
    fetchItems(domain).then(setCatalogItems);
  }, [domain]);

  const loadRecs = async () => {
    if (!selectedUser) return;
    try {
      const res = await recommendSingle(selectedUser, "sasrec", domain, 8);
      setRecommendations(res.items);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadRecs();
  }, [selectedUser, domain]);

  const simulateAction = async (item: any, actionType: string) => {
    const ev = {
      event_id: `sim_${Date.now()}`,
      user_id: selectedUser,
      item_id: item.item_id,
      event_type: actionType,
      timestamp: Date.now() / 1000,
      artwork_variant_id: item.artwork_variant_id,
      context: { title: item.title, genre: item.primary_genre || item.genre || item.category },
    };

    await ingestEvent(ev);
    setSessionEvents((prev) => [ev, ...prev.slice(0, 9)]);
    // Re-trigger sequential transformer (SASRec) recommendation with the updated session!
    await loadRecs();
  };

  // Auto-play simulation loop
  useEffect(() => {
    let timer: any = null;
    if (isAutoPlaying && catalogItems.length > 0) {
      timer = setInterval(() => {
        const item = catalogItems[Math.floor(Math.random() * catalogItems.length)];
        const actions = ["watch_complete", "like", "click", "skip"];
        const act = actions[Math.floor(Math.random() * actions.length)];
        simulateAction(item, act);
      }, 2500);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isAutoPlaying, catalogItems, selectedUser]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
      {/* Header */}
      <div className="glass-panel" style={{ padding: "24px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "16px" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <UserCheck size={22} color="#10b981" />
            <h2 style={{ fontSize: "1.3rem", fontWeight: "700" }}>Real-Time User Behavior Simulator (TikTok / YouTube Stream)</h2>
          </div>
          <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", marginTop: "4px" }}>
            Simulate micro-actions (skips, full watches, likes) and watch SASRec sequential embeddings adjust in real-time.
          </p>
        </div>

        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          <button
            className={`btn ${isAutoPlaying ? "btn-secondary" : "btn-primary"}`}
            onClick={() => setIsAutoPlaying(!isAutoPlaying)}
          >
            <Activity size={16} />
            <span>{isAutoPlaying ? "Stop Auto Simulation" : "Start Auto Simulation (Stream)"}</span>
          </button>
        </div>
      </div>

      {/* Simulator Workspace */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "24px" }}>
        {/* Left: Live Recommendations Feed */}
        <div>
          <ItemGrid
            items={recommendations}
            title="Real-Time Adapted Slate (SASRec Sequential Model)"
            subtitle="Automatically updates after every simulated interaction event"
            onInteract={(item, type) => simulateAction(item, type)}
          />
        </div>

        {/* Right: Live Session Interaction Feed */}
        <div className="glass-panel" style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "16px", height: "fit-content" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "#fff" }}>Live Session Event Log</h3>
          <p style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
            Streamed through WebSocket event bus to online feature store
          </p>

          <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxHeight: "450px", overflowY: "auto" }}>
            {sessionEvents.length === 0 ? (
              <span style={{ fontSize: "0.82rem", color: "var(--text-muted)", textAlign: "center", padding: "20px" }}>
                Click action icons on the cards or start Auto-Simulation.
              </span>
            ) : (
              sessionEvents.map((ev, i) => (
                <div
                  key={ev.event_id || i}
                  style={{
                    background: "var(--bg-dark)",
                    border: "1px solid var(--border-color)",
                    borderRadius: "8px",
                    padding: "10px",
                    fontSize: "0.8rem",
                    display: "flex",
                    flexDirection: "column",
                    gap: "4px",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span className="badge badge-cyan" style={{ fontSize: "0.7rem" }}>{ev.event_type}</span>
                    <span style={{ color: "var(--text-muted)", fontSize: "0.72rem", fontFamily: "monospace" }}>
                      {new Date(ev.timestamp * 1000).toLocaleTimeString()}
                    </span>
                  </div>
                  <strong style={{ color: "#fff", fontSize: "0.85rem" }}>
                    {ev.context?.title || ev.item_id}
                  </strong>
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
                    Genre: {ev.context?.genre || "General"}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
