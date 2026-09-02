import { useState, useEffect } from "react";
import { Navbar } from "./components/Navbar";
import { LiveEventTicker } from "./components/LiveEventTicker";
import { Home } from "./pages/Home";
import { RecLab } from "./pages/RecLab";
import { PipelineView } from "./pages/PipelineView";
import { ArenaView } from "./pages/ArenaView";
import { ArtworkView } from "./pages/ArtworkView";
import { UserSimView } from "./pages/UserSimView";
import { ExplorerView } from "./pages/ExplorerView";
import { fetchHealth } from "./api";

export function App() {
  const [currentTab, setCurrentTab] = useState<string>("home");
  const [domain, setDomain] = useState<string>("movies");
  const [serverOnline, setServerOnline] = useState<boolean>(false);

  useEffect(() => {
    const checkStatus = () => {
      fetchHealth()
        .then(() => setServerOnline(true))
        .catch(() => setServerOnline(false));
    };
    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <Navbar
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        domain={domain}
        setDomain={setDomain}
        serverOnline={serverOnline}
      />

      <main style={{ maxWidth: "1600px", margin: "0 auto", width: "100%", padding: "24px", display: "flex", flexDirection: "column", gap: "24px", flexGrow: 1 }}>
        {/* Real-time Streaming WebSocket Ticker */}
        <LiveEventTicker />

        {/* Dynamic Page Router */}
        {currentTab === "home" && <Home onNavigate={setCurrentTab} domain={domain} />}
        {currentTab === "lab" && <RecLab domain={domain} />}
        {currentTab === "pipeline" && <PipelineView domain={domain} />}
        {currentTab === "arena" && <ArenaView domain={domain} />}
        {currentTab === "artwork" && <ArtworkView />}
        {currentTab === "user_sim" && <UserSimView domain={domain} />}
        {currentTab === "explorer" && <ExplorerView domain={domain} />}
      </main>

      {/* Footer */}
      <footer style={{ borderTop: "1px solid var(--border-color)", padding: "16px 24px", textAlign: "center", fontSize: "0.78rem", color: "var(--text-muted)" }}>
        আসেন Unproductive হই (Asen Unproductive Hoi) • The Ultimate AI Procrastination & Distraction Engine • Powered by 12 Production Models
      </footer>
    </div>
  );
}

export default App;
