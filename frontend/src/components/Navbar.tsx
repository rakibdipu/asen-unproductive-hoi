import React from "react";
import {
  Film,
  Music,
  ShoppingBag,
  Flame,
  Layers,
  FlaskConical,
  BarChart3,
  Image,
  UserCheck,
  Brain,
  Activity
} from "lucide-react";

interface NavbarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  domain: string;
  setDomain: (domain: string) => void;
  serverOnline: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  setCurrentTab,
  domain,
  setDomain,
  serverOnline,
}) => {
  const tabs = [
    { id: "home", label: "🛋️ Procrastination Hub", icon: Activity },
    { id: "lab", label: "🧪 Algorithm Lab", icon: FlaskConical },
    { id: "pipeline", label: "⚡ 5-Stage Pipeline", icon: Layers },
    { id: "arena", label: "⚔️ Model Arena", icon: BarChart3 },
    { id: "artwork", label: "🎨 Artwork Bandit", icon: Image },
    { id: "user_sim", label: "📱 Live Feed Sim", icon: UserCheck },
    { id: "explorer", label: "🔍 Find a Distraction", icon: Brain },
  ];

  const domains = [
    { id: "movies", label: "Movies & Video (Netflix)", icon: Film },
    { id: "music", label: "Music & Audio (Spotify)", icon: Music },
    { id: "ecommerce", label: "E-Commerce (Amazon)", icon: ShoppingBag },
    { id: "short_feed", label: "Short Feed (TikTok)", icon: Flame },
  ];

  return (
    <header style={{ borderBottom: "1px solid var(--border-color)", background: "rgba(10, 13, 20, 0.95)", position: "sticky", top: 0, zIndex: 50, backdropFilter: "blur(10px)" }}>
      <div style={{ maxWidth: "1600px", margin: "0 auto", padding: "12px 24px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "16px" }}>
        
        {/* Logo & Brand */}
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{ width: "42px", height: "42px", borderRadius: "12px", background: "linear-gradient(135deg, #ff1a1a, #8b0000)", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 0 20px rgba(229, 9, 20, 0.6)", fontSize: "1.4rem", border: "1px solid rgba(255, 77, 77, 0.4)" }}>
            🛋️
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontSize: "1.3rem", fontWeight: "900", letterSpacing: "-0.01em", background: "linear-gradient(to right, #ffffff, #ff4d4d, #e50914)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", filter: "drop-shadow(0 0 12px rgba(229, 9, 20, 0.4))" }}>
                আসেন Unproductive হই
              </span>
              <span className="badge badge-red">🪫 1% BATTERY VIBE</span>
            </div>
            <p style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>
              Asen Unproductive Hoi • কাজের কাজ বাদ দিয়ে টাইম নষ্ট করার সেরা এআই ইঞ্জিন
            </p>
          </div>
        </div>

        {/* Domain Selector */}
        <div style={{ display: "flex", alignItems: "center", gap: "6px", background: "var(--bg-card)", padding: "4px", borderRadius: "10px", border: "1px solid var(--border-color)" }}>
          {domains.map((d) => {
            const Icon = d.icon;
            const active = domain === d.id;
            return (
              <button
                key={d.id}
                onClick={() => setDomain(d.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  padding: "6px 12px",
                  borderRadius: "8px",
                  border: "none",
                  fontSize: "0.8rem",
                  fontWeight: active ? "600" : "500",
                  cursor: "pointer",
                  background: active ? "rgba(229, 9, 20, 0.2)" : "transparent",
                  color: active ? "#ff4d4d" : "var(--text-muted)",
                  transition: "all 0.15s ease",
                }}
              >
                <Icon size={14} />
                <span>{d.label.split(" ")[0]}</span>
              </button>
            );
          })}
        </div>

        {/* Server Status Pill */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: serverOnline ? "#10b981" : "#ef4444", boxShadow: serverOnline ? "0 0 8px #10b981" : "none" }} />
          <span style={{ fontSize: "0.75rem", color: serverOnline ? "#34d399" : "#f87171", fontFamily: "monospace" }}>
            {serverOnline ? "API READY (Port 8000)" : "API CONNECTING..."}
          </span>
        </div>
      </div>

      {/* Navigation Sub-bar */}
      <div style={{ borderTop: "1px solid rgba(31, 41, 61, 0.6)", background: "rgba(17, 23, 38, 0.6)" }}>
        <div style={{ maxWidth: "1600px", margin: "0 auto", padding: "0 24px", display: "flex", gap: "8px", overflowX: "auto" }}>
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const active = currentTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setCurrentTab(tab.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  padding: "10px 16px",
                  border: "none",
                  borderBottom: active ? "2px solid #e50914" : "2px solid transparent",
                  background: "transparent",
                  color: active ? "#fff" : "var(--text-muted)",
                  fontWeight: active ? "600" : "500",
                  fontSize: "0.85rem",
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                  transition: "color 0.15s ease",
                }}
              >
                <Icon size={16} color={active ? "#e50914" : "currentColor"} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
