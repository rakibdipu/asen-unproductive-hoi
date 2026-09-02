import React from "react";
import {
  Layers,
  Zap,
  TrendingUp,
  Cpu,
  BarChart,
  ShieldCheck,
  PlayCircle,
  ArrowRight
} from "lucide-react";

interface HomeProps {
  onNavigate: (tab: string) => void;
  domain: string;
}

export const Home: React.FC<HomeProps> = ({ onNavigate, domain }) => {
  const stats = [
    { label: "Algorithms Integrated", value: "12 Models", desc: "Gen 1 (CF) to Gen 6 (LLM)", icon: Cpu, color: "#8b5cf6" },
    { label: "Production Pipeline", value: "5 Stages", desc: "Retrieval to Slate Generation", icon: Layers, color: "#e50914" },
    { label: "Vector Search Latency", value: "< 2.5 ms", desc: "Batch PyTorch Inner-Product ANN", icon: Zap, color: "#06b6d4" },
    { label: "Multi-Domain Adapters", value: "4 Domains", desc: "Movies, Music, E-Com, Short Feed", icon: TrendingUp, color: "#10b981" },
  ];

  const algorithmFamilies = [
    {
      gen: "Gen 1 • Classic Heuristic",
      models: "Popularity (Time-Decayed), Item-Item Collaborative Filtering (Amazon 2001)",
      desc: "Fast, interpretable, zero-shot baselines for cold start and immediate co-occurrence.",
    },
    {
      gen: "Gen 2 • Latent Factor MF",
      models: "Implicit ALS (Hu-Koren 2008), BPR-MF (Rendle 2009)",
      desc: "Matrix factorization with confidence weighting and pairwise ranking optimization.",
    },
    {
      gen: "Gen 3 • Deep Retrieval",
      models: "Two-Tower DNN (YouTube Covington et al. 2016) + In-Memory ANN",
      desc: "Dual neural towers mapping users and catalog items into a shared vector metric space.",
    },
    {
      gen: "Gen 4 • Deep Ranking",
      models: "Wide & Deep (Google), DeepFM (Huawei), DIN (Alibaba), MMoE Multi-Task (Google)",
      desc: "Second-order feature crosses, target-aware local attention, and multi-objective utility.",
    },
    {
      gen: "Gen 5 • Sequential & Graph",
      models: "SASRec Transformer (UCSD 2018), LightGCN (NUS 2020)",
      desc: "Self-attention causal sequence modeling and multi-hop bipartite graph neighborhood diffusion.",
    },
    {
      gen: "Gen 6 • Bandits & LLM",
      models: "LinUCB Bandit (Netflix Artwork 2018), LLM Generative Recommender (2026)",
      desc: "Contextual bandit artwork exploration/exploitation and in-context semantic explanations.",
    },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      {/* Hero Banner */}
      <div
        className="glass-panel"
        style={{
          padding: "36px 40px",
          borderRadius: "16px",
          background: "linear-gradient(135deg, rgba(229, 9, 20, 0.25) 0%, rgba(139, 0, 0, 0.15) 50%, rgba(10, 13, 20, 0.95) 100%)",
          border: "1px solid rgba(229, 9, 20, 0.4)",
          boxShadow: "0 0 35px rgba(229, 9, 20, 0.2)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "24px",
        }}
      >
        <div style={{ maxWidth: "750px" }}>
          <div style={{ display: "flex", gap: "8px", marginBottom: "12px" }}>
            <span className="badge badge-red">🛋️ PROCRASTINATION ENGINE</span>
            <span className="badge badge-red" style={{ background: "rgba(255, 26, 26, 0.2)", color: "#ff4d4d", border: "1px solid #ff4d4d" }}>🪫 1% BATTERY VIBE</span>
          </div>
          <h1 style={{ fontSize: "2.5rem", fontWeight: "900", lineHeight: 1.2, letterSpacing: "-0.02em" }}>
            আসেন Unproductive হই
          </h1>
          <p style={{ marginTop: "12px", color: "var(--text-muted)", fontSize: "1.05rem", lineHeight: 1.6 }}>
            কাজের মাঝে একটু বিরতি দরকার? আসতেই পারেন! YouTube, Netflix, TikTok ও Spotify-র ২০ বছরের
            আধুনিক এআই অ্যালগরিদম একত্রিত করে তৈরি — আপনার সময় সফলভাবে নষ্ট করার এক মাস্টারপিস রিকমেন্ডেশন সিস্টেম।
          </p>
          <div style={{ display: "flex", gap: "14px", marginTop: "24px" }}>
            <button className="btn btn-primary" onClick={() => onNavigate("pipeline")}>
              <PlayCircle size={18} />
              <span>🛋️ Start Procrastinating (Pipeline)</span>
            </button>
            <button className="btn btn-secondary" onClick={() => onNavigate("arena")}>
              <BarChart size={18} />
              <span>⚔️ Battle of Unproductive Models</span>
            </button>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "10px", minWidth: "260px" }}>
          <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid var(--border-color)", padding: "14px 18px", borderRadius: "10px" }}>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Current Domain</span>
            <div style={{ fontSize: "1.1rem", fontWeight: "700", color: "#fff", textTransform: "capitalize" }}>
              {domain} Experience
            </div>
          </div>
          <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid var(--border-color)", padding: "14px 18px", borderRadius: "10px" }}>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Pipeline Status</span>
            <div style={{ fontSize: "1.1rem", fontWeight: "700", color: "#34d399", display: "flex", alignItems: "center", gap: "6px" }}>
              <ShieldCheck size={18} /> Fully Calibrated
            </div>
          </div>
        </div>
      </div>

      {/* KPI Stats Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "18px" }}>
        {stats.map((s, i) => {
          const Icon = s.icon;
          return (
            <div key={i} className="glass-panel" style={{ padding: "20px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
                <span style={{ fontSize: "0.82rem", color: "var(--text-muted)", fontWeight: "500" }}>{s.label}</span>
                <div style={{ width: "32px", height: "32px", borderRadius: "8px", background: `${s.color}20`, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <Icon size={18} color={s.color} />
                </div>
              </div>
              <div style={{ fontSize: "1.8rem", fontWeight: "800", color: "#fff" }}>{s.value}</div>
              <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "4px" }}>{s.desc}</p>
            </div>
          );
        })}
      </div>

      {/* Algorithm Generations Matrix */}
      <div className="glass-panel" style={{ padding: "28px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
          <div>
            <h2 style={{ fontSize: "1.3rem", fontWeight: "700" }}>Integrated Algorithm Generations</h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
              The full evolution of recommendation algorithms active in this unified platform
            </p>
          </div>
          <button className="btn btn-secondary" onClick={() => onNavigate("lab")}>
            <span>Try in Algorithm Lab</span>
            <ArrowRight size={14} />
          </button>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "16px" }}>
          {algorithmFamilies.map((fam, idx) => (
            <div
              key={idx}
              style={{
                background: "var(--bg-dark)",
                border: "1px solid var(--border-color)",
                borderRadius: "10px",
                padding: "16px",
                display: "flex",
                flexDirection: "column",
                gap: "8px",
              }}
            >
              <span style={{ fontSize: "0.78rem", fontWeight: "700", color: "#38bdf8", textTransform: "uppercase" }}>
                {fam.gen}
              </span>
              <h4 style={{ fontSize: "0.95rem", fontWeight: "600", color: "#fff" }}>{fam.models}</h4>
              <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", lineHeight: 1.5 }}>{fam.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
