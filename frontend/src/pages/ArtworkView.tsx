import React, { useState, useEffect } from "react";
import { getBanditStatus, simulateBanditRound } from "../api";
import { Image, Sparkles, User, Play, RefreshCw } from "lucide-react";

export const ArtworkView: React.FC = () => {
  const [banditStatus, setBanditStatus] = useState<any>(null);
  const [selectedProfile, setSelectedProfile] = useState<string>("action_fan");
  const [lastDecision, setLastDecision] = useState<any>(null);
  const [simulationRounds, setSimulationRounds] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);

  const tasteProfiles = [
    {
      id: "action_fan",
      name: "Action Enthusiast",
      description: "Loves adrenaline, car chases, explosions and fast pacing.",
      affinities: { action: 0.9, scifi: 0.5, romance: 0.1, cinematic: 0.3 },
    },
    {
      id: "romance_lover",
      name: "Romance & Drama Lover",
      description: "Drawn to interpersonal chemistry, relationships, and emotional intimacy.",
      affinities: { action: 0.1, scifi: 0.2, romance: 0.95, cinematic: 0.4 },
    },
    {
      id: "scifi_geek",
      name: "Sci-Fi & Cyberpunk Fan",
      description: "Loves futuristic aesthetics, mind-bending concepts, and technology.",
      affinities: { action: 0.5, scifi: 0.95, romance: 0.1, cinematic: 0.7 },
    },
    {
      id: "cinephile",
      name: "Auteur / Cinematic Critic",
      description: "Values wide cinematography, visual composition, and art direction.",
      affinities: { action: 0.2, scifi: 0.4, romance: 0.3, cinematic: 0.95 },
    },
  ];

  const artworkVariants = [
    {
      id: "variant_action",
      label: "Action Focus Variant",
      desc: "High-contrast explosion / combat focus",
      img: "https://image.tmdb.org/t/p/w500/tXQvtRWfkUUnWJAn2tN3jERIUG.jpg",
    },
    {
      id: "variant_character",
      label: "Character Close-Up Variant",
      desc: "Intense emotional face portrait",
      img: "https://image.tmdb.org/t/p/w500/r84x4x93LbZ2gozISTBYVeq0gLZ.jpg",
    },
    {
      id: "variant_romantic",
      label: "Romantic / Warm Variant",
      desc: "Couple intimacy & warm color palette",
      img: "https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?w=600&q=80",
    },
    {
      id: "variant_cinematic",
      label: "Scenic Wide Landscape Variant",
      desc: "Atmospheric environment & scale",
      img: "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg",
    },
  ];

  const refreshStatus = async () => {
    try {
      const data = await getBanditStatus();
      setBanditStatus(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    refreshStatus();
  }, []);

  const handleSimulate = async (clicksCount: number = 1) => {
    setLoading(true);
    const prof = tasteProfiles.find((p) => p.id === selectedProfile)!;

    for (let i = 0; i < clicksCount; i++) {
      const res = await simulateBanditRound(
        "mov_0001",
        {
          name: prof.name,
          action_affinity: prof.affinities.action,
          scifi_affinity: prof.affinities.scifi,
          romance_affinity: prof.affinities.romance,
          cinematic_affinity: prof.affinities.cinematic,
        },
        true
      ) as any;
      setLastDecision(res.decision);
    }

    setSimulationRounds((prev) => prev + clicksCount);
    await refreshStatus();
    setLoading(false);
  };


  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
      {/* Header */}
      <div className="glass-panel" style={{ padding: "24px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "16px" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <Image size={22} color="#e50914" />
            <h2 style={{ fontSize: "1.3rem", fontWeight: "700" }}>Netflix Artwork Personalization (LinUCB Bandit)</h2>
          </div>
          <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", marginTop: "4px" }}>
            Contextual Bandit dynamically tests and selects the optimal visual poster tailored to each user's taste profile.
          </p>
        </div>

        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          <span style={{ fontSize: "0.85rem", color: "#34d399", fontWeight: "600" }}>
            Total Bandit Interactions: {simulationRounds}
          </span>
          <button className="btn btn-secondary" onClick={refreshStatus}>
            <RefreshCw size={14} />
            <span>Refresh Weights</span>
          </button>
        </div>
      </div>

      {/* Interactive Taste Profile Selector */}
      <div className="glass-panel" style={{ padding: "20px" }}>
        <h3 style={{ fontSize: "0.95rem", fontWeight: "700", marginBottom: "12px", color: "var(--text-muted)" }}>
          1. SELECT USER PERSONA PROFILE
        </h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "12px" }}>
          {tasteProfiles.map((p) => {
            const active = selectedProfile === p.id;
            return (
              <div
                key={p.id}
                onClick={() => setSelectedProfile(p.id)}
                style={{
                  background: active ? "rgba(229, 9, 20, 0.15)" : "var(--bg-dark)",
                  border: `1px solid ${active ? "#e50914" : "var(--border-color)"}`,
                  borderRadius: "10px",
                  padding: "14px",
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <User size={16} color={active ? "#ff4d4d" : "var(--text-muted)"} />
                  <strong style={{ fontSize: "0.9rem", color: active ? "#fff" : "var(--text-main)" }}>{p.name}</strong>
                </div>
                <p style={{ fontSize: "0.76rem", color: "var(--text-muted)", marginTop: "6px" }}>{p.description}</p>
                <div style={{ display: "flex", gap: "6px", marginTop: "8px", flexWrap: "wrap" }}>
                  <span className="badge badge-red" style={{ fontSize: "0.68rem" }}>Action: {(p.affinities.action * 100).toFixed(0)}%</span>
                  <span className="badge badge-purple" style={{ fontSize: "0.68rem" }}>Romance: {(p.affinities.romance * 100).toFixed(0)}%</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Simulation Controls */}
        <div style={{ display: "flex", gap: "12px", marginTop: "16px", alignItems: "center" }}>
          <button className="btn btn-primary" onClick={() => handleSimulate(1)} disabled={loading}>
            <Play size={16} />
            <span>Simulate 1 View</span>
          </button>
          <button className="btn btn-secondary" onClick={() => handleSimulate(10)} disabled={loading}>
            <Play size={16} />
            <span>Simulate 10 User Views (Converge)</span>
          </button>
          {lastDecision && (
            <span style={{ fontSize: "0.85rem", color: "#38bdf8", marginLeft: "auto" }}>
              Current Choice: <strong style={{ color: "#fff" }}>{lastDecision.selected_arm.toUpperCase()}</strong> (UCB Score: {lastDecision.total_score.toFixed(3)})
            </span>
          )}
        </div>
      </div>

      {/* 4 Artwork Arms Visual Comparison */}
      <div>
        <h3 style={{ fontSize: "1.1rem", fontWeight: "700", marginBottom: "14px" }}>
          2. THE 4 ARTWORK CANDIDATE ARMS FOR "INCEPTION"
        </h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "18px" }}>
          {artworkVariants.map((arm) => {
            const isChosen = lastDecision?.selected_arm === arm.id;
            const stats = banditStatus?.arm_statistics?.[arm.id] || { pull_count: 0, win_rate: 0 };

            return (
              <div
                key={arm.id}
                className="glass-panel"
                style={{
                  borderRadius: "12px",
                  overflow: "hidden",
                  border: isChosen ? "2px solid #e50914" : "1px solid var(--border-color)",
                  position: "relative",
                  boxShadow: isChosen ? "0 0 20px rgba(229, 9, 20, 0.4)" : "none",
                }}
              >
                {isChosen && (
                  <div
                    style={{
                      position: "absolute",
                      top: "10px",
                      left: "10px",
                      zIndex: 10,
                      background: "#e50914",
                      color: "#fff",
                      fontSize: "0.72rem",
                      fontWeight: "800",
                      padding: "3px 8px",
                      borderRadius: "6px",
                      display: "flex",
                      alignItems: "center",
                      gap: "4px",
                    }}
                  >
                    <Sparkles size={12} />
                    SELECTED BY BANDIT
                  </div>
                )}

                <div style={{ width: "100%", height: "280px" }}>
                  <img
                    src={arm.img}
                    alt={arm.label}
                    style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    onError={(e) => {
                      e.currentTarget.src = "https://images.unsplash.com/photo-1514565131-fce0801e5785?w=600&q=80";
                    }}
                  />
                </div>

                <div style={{ padding: "16px" }}>
                  <h4 style={{ fontSize: "0.95rem", fontWeight: "700", color: "#fff" }}>{arm.label}</h4>
                  <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "2px" }}>{arm.desc}</p>

                  <div style={{ marginTop: "12px", paddingTop: "10px", borderTop: "1px solid var(--border-color)", display: "flex", justifyContent: "space-between", fontSize: "0.8rem" }}>
                    <span style={{ color: "var(--text-muted)" }}>Impressions: <strong style={{ color: "#fff" }}>{stats.pull_count}</strong></span>
                    <span style={{ color: "var(--text-muted)" }}>Win Rate: <strong style={{ color: "#34d399" }}>{(stats.win_rate * 100).toFixed(1)}%</strong></span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
