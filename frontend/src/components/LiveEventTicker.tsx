import React, { useEffect, useState } from "react";
import type { UserInteractionEvent } from "../types";
import { Radio, Play, ThumbsUp, FastForward, Eye } from "lucide-react";

export const LiveEventTicker: React.FC = () => {
  const [events, setEvents] = useState<UserInteractionEvent[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: any = null;

    const connect = () => {
      ws = new WebSocket("ws://localhost:8000/api/events/ws");

      ws.onopen = () => {
        setConnected(true);
      };

      ws.onmessage = (msg) => {
        try {
          const parsed = JSON.parse(msg.data);
          setEvents((prev) => [parsed, ...prev.slice(0, 7)]);
        } catch (e) {
          // ignore ping
        }
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectTimeout = setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    connect();

    return () => {
      if (ws) ws.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, []);

  const getEventIcon = (type: string) => {
    switch (type) {
      case "watch_complete":
        return <Play size={12} color="#34d399" />;
      case "like":
        return <ThumbsUp size={12} color="#38bdf8" />;
      case "skip":
        return <FastForward size={12} color="#f87171" />;
      default:
        return <Eye size={12} color="#94a3b8" />;
    }
  };

  return (
    <div
      style={{
        background: "rgba(10, 13, 20, 0.9)",
        border: "1px solid var(--border-color)",
        borderRadius: "10px",
        padding: "10px 16px",
        display: "flex",
        alignItems: "center",
        gap: "16px",
        overflowX: "auto",
        whiteSpace: "nowrap",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "6px", flexShrink: 0 }}>
        <Radio size={14} color={connected ? "#10b981" : "#f59e0b"} className={connected ? "animate-pulse" : ""} />
        <span style={{ fontSize: "0.75rem", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.05em", color: connected ? "#34d399" : "#f59e0b" }}>
          Live Event Bus
        </span>
      </div>

      <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
        {events.length === 0 ? (
          <span style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontStyle: "italic" }}>
            Waiting for user interaction events... (Click or watch any card to broadcast)
          </span>
        ) : (
          events.map((ev, i) => (
            <div
              key={`${ev.event_id || 'ev'}_${i}_${ev.timestamp}`}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                background: "rgba(255,255,255,0.05)",
                padding: "3px 8px",
                borderRadius: "6px",
                fontSize: "0.75rem",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              {getEventIcon(ev.event_type)}
              <span style={{ fontWeight: "600", color: "#fff" }}>{ev.user_id}</span>
              <span style={{ color: "var(--text-muted)" }}>{ev.event_type}</span>
              <span style={{ color: "#38bdf8", fontFamily: "monospace" }}>{ev.item_id}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
