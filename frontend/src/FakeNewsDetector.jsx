import { useState, useEffect, useCallback } from "react";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const VERDICT_CONFIG = {
  FAKE:       { color: "#ff3b3b", bg: "#1a0000", border: "#ff3b3b", icon: "⚠", glow: "0 0 30px #ff3b3b55", label: "FAKE" },
  REAL:       { color: "#00e676", bg: "#001a07", border: "#00e676", icon: "✓", glow: "0 0 30px #00e67655", label: "REAL" },
  UNVERIFIED: { color: "#ffaa00", bg: "#1a0e00", border: "#ffaa00", icon: "?", glow: "0 0 30px #ffaa0055", label: "UNVERIFIED" },
  NO_DATA:    { color: "#555",    bg: "#111",    border: "#333",    icon: "–", glow: "none",               label: "NO DATA" },
  UNKNOWN:    { color: "#555",    bg: "#111",    border: "#333",    icon: "–", glow: "none",               label: "UNKNOWN" },
};

const DEMO_ARTICLES = [
  {
    title: "Scientists Discover That Drinking Coffee Cures All Cancers, Study Shows",
    body: "Researchers at an unnamed university claim that drinking 47 cups of coffee per day eliminates all forms of cancer. The study, published on a personal blog, found 100% effectiveness in a sample size of 3 people.",
    verdict: "FAKE", confidence: 0.91,
    bert: { label: "FAKE", confidence: 0.88, probabilities: { REAL: 0.06, FAKE: 0.88, UNVERIFIED: 0.06 } },
    factcheck: { verdict: "FAKE", confidence: 0.94, summary: "Snopes rated False (2023-08-12); PolitiFact rated Pants on Fire (2023-08-14)" },
    graph: { score: 0.78, features: { num_nodes: 1240, bot_ratio: 0.62, early_spread_ratio: 0.81, max_depth: 3, max_breadth: 890 } },
  },
  {
    title: "Federal Reserve Raises Interest Rates by 0.25% at March Meeting",
    body: "The Federal Reserve voted unanimously to raise its benchmark interest rate by a quarter percentage point on Wednesday, continuing its campaign to bring inflation back to its 2% target.",
    verdict: "REAL", confidence: 0.89,
    bert: { label: "REAL", confidence: 0.85, probabilities: { REAL: 0.85, FAKE: 0.09, UNVERIFIED: 0.06 } },
    factcheck: { verdict: "REAL", confidence: 0.92, summary: "AP Fact Check confirmed (2024-03-20); Reuters verified (2024-03-20)" },
    graph: { score: 0.14, features: { num_nodes: 340, bot_ratio: 0.08, early_spread_ratio: 0.22, max_depth: 7, max_breadth: 45 } },
  },
  {
    title: "New Study Links 5G Towers to Mysterious Illness Spreading Across Europe",
    body: "Reports are emerging from multiple European countries of an unexplained illness that residents believe may be connected to recently installed 5G infrastructure.",
    verdict: "UNVERIFIED", confidence: 0.61,
    bert: { label: "UNVERIFIED", confidence: 0.58, probabilities: { REAL: 0.22, FAKE: 0.36, UNVERIFIED: 0.42 } },
    factcheck: { verdict: "UNVERIFIED", confidence: 0.55, summary: "No authoritative fact-checks found; WHO statement pending" },
    graph: { score: 0.51, features: { num_nodes: 680, bot_ratio: 0.31, early_spread_ratio: 0.48, max_depth: 5, max_breadth: 210 } },
  },
];

function ConfidenceBar({ value, color, label }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setWidth(Math.max(0, Math.min(1, value)) * 100), 120);
    return () => clearTimeout(t);
  }, [value]);

  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3, fontSize: 11, color: "#666" }}>
        <span>{label}</span>
        <span style={{ color }}>{(Math.max(0, Math.min(1, value)) * 100).toFixed(0)}%</span>
      </div>
      <div style={{ height: 4, background: "#1a1a1a", borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${width}%`, background: color, borderRadius: 2,
          transition: "width 0.9s cubic-bezier(0.4,0,0.2,1)"
        }} />
      </div>
    </div>
  );
}

function RadialScore({ score, label, color }) {
  const safeScore = Math.max(0, Math.min(1, score || 0));
  const r = 28, circ = 2 * Math.PI * r;
  const [dash, setDash] = useState(circ);
  useEffect(() => {
    const t = setTimeout(() => setDash(circ * (1 - safeScore)), 120);
    return () => clearTimeout(t);
  }, [safeScore, circ]);

  return (
    <div style={{ textAlign: "center", minWidth: 70 }}>
      <svg width={70} height={70} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={35} cy={35} r={r} fill="none" stroke="#1a1a1a" strokeWidth={5} />
        <circle cx={35} cy={35} r={r} fill="none" stroke={color} strokeWidth={5}
          strokeDasharray={circ} strokeDashoffset={dash}
          style={{ transition: "stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1)" }}
          strokeLinecap="round" />
      </svg>
      <div style={{ marginTop: -8, fontSize: 13, fontWeight: 700, color, fontFamily: "monospace" }}>
        {(safeScore * 100).toFixed(0)}%
      </div>
      <div style={{ fontSize: 10, color: "#444", marginTop: 2 }}>{label}</div>
    </div>
  );
}

function Badge({ verdict }) {
  const cfg = VERDICT_CONFIG[verdict] || VERDICT_CONFIG.UNKNOWN;
  return (
    <span style={{
      background: cfg.bg, color: cfg.color,
      border: `1px solid ${cfg.border}`,
      borderRadius: 4, padding: "2px 10px",
      fontSize: 11, fontWeight: 700, fontFamily: "monospace",
      letterSpacing: "0.05em",
    }}>{cfg.label}</span>
  );
}

function ComponentCard({ title, icon, accent, children }) {
  return (
    <div style={{
      background: "#0d0d0d", border: `1px solid ${accent}22`,
      borderRadius: 8, padding: 16, flex: 1, minWidth: 0,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
        <span style={{ fontSize: 14 }}>{icon}</span>
        <span style={{ fontSize: 10, fontWeight: 700, color: accent, letterSpacing: "0.1em", textTransform: "uppercase" }}>
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}

function ResultPanel({ result }) {
  const cfg = VERDICT_CONFIG[result.verdict] || VERDICT_CONFIG.UNKNOWN;
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    setVisible(false);
    const t = setTimeout(() => setVisible(true), 60);
    return () => clearTimeout(t);
  }, [result]);

  const graphScore = result.graph?.score >= 0 ? result.graph.score : 0;
  const graphFeats = result.graph?.features || {};
  const bertProbs  = result.bert?.probabilities || {};
  const fcConf     = result.factcheck?.confidence || 0;

  return (
    <div style={{
      opacity: visible ? 1 : 0,
      transform: visible ? "translateY(0)" : "translateY(14px)",
      transition: "all 0.5s cubic-bezier(0.4,0,0.2,1)",
    }}>

      {/* Main verdict card */}
      <div style={{
        background: cfg.bg, border: `1px solid ${cfg.border}`,
        borderRadius: 12, padding: "28px 24px", marginBottom: 12,
        boxShadow: cfg.glow, textAlign: "center",
      }}>
        <div style={{ fontSize: 44, color: cfg.color, lineHeight: 1, marginBottom: 8 }}>{cfg.icon}</div>
        <div style={{ fontSize: 26, fontWeight: 800, color: cfg.color, letterSpacing: "0.18em", fontFamily: "monospace" }}>
          {result.verdict}
        </div>
        <div style={{ color: "#444", fontSize: 11, marginTop: 6 }}>
          Overall confidence:&nbsp;
          <span style={{ color: cfg.color, fontWeight: 700 }}>
            {(Math.max(0, Math.min(1, result.confidence || 0)) * 100).toFixed(0)}%
          </span>
        </div>

        {/* Three radial scores */}
        <div style={{ marginTop: 16, display: "flex", justifyContent: "center", gap: 28 }}>
          <RadialScore score={result.bert?.confidence || 0}  label="BERT"       color="#7b68ee" />
          <RadialScore score={fcConf}                        label="Fact-check" color="#00bcd4" />
          <RadialScore score={graphScore}                    label="Graph"      color="#ff7043" />
        </div>

        {/* Explanation */}
        {result.explanation && (
          <div style={{
            marginTop: 16, fontSize: 10, color: "#444", lineHeight: 1.7,
            maxWidth: 600, margin: "16px auto 0", textAlign: "left",
            background: "#0a0a0a", borderRadius: 6, padding: "10px 14px",
          }}>
            {result.explanation}
          </div>
        )}
      </div>

      {/* BERT + Fact-check row */}
      <div style={{ display: "flex", gap: 10, marginBottom: 10, flexWrap: "wrap" }}>

        <ComponentCard title="BERT Classifier" icon="🧠" accent="#7b68ee">
          <div style={{ marginBottom: 10 }}>
            <Badge verdict={result.bert?.label} />
          </div>
          {Object.entries(bertProbs).map(([lbl, prob]) => (
            <ConfidenceBar
              key={lbl} label={lbl}
              value={typeof prob === "number" ? prob : 0}
              color={VERDICT_CONFIG[lbl]?.color || "#666"}
            />
          ))}
          {Object.keys(bertProbs).length === 0 && (
            <div style={{ fontSize: 10, color: "#333" }}>No probability data</div>
          )}
          <div style={{ fontSize: 10, color: "#333", marginTop: 10 }}>
            xlm-roberta-base · multilingual
          </div>
        </ComponentCard>

        <ComponentCard title="Fact-Check API" icon="🔍" accent="#00bcd4">
          <div style={{ marginBottom: 10 }}>
            <Badge verdict={result.factcheck?.verdict} />
          </div>
          <ConfidenceBar label="Confidence" value={fcConf} color="#00bcd4" />
          <div style={{ fontSize: 10, color: "#555", marginTop: 10, lineHeight: 1.6 }}>
            {result.factcheck?.summary || "No fact-check data available. Add GOOGLE_FACTCHECK_API_KEY to enable."}
          </div>
          <div style={{ fontSize: 10, color: "#333", marginTop: 10 }}>
            Google Fact Check Tools API
          </div>
        </ComponentCard>
      </div>

      {/* Graph analysis */}
      <ComponentCard title="Propagation Graph Analysis" icon="🕸" accent="#ff7043">
        <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
          <RadialScore score={graphScore} label="Fake spread score" color="#ff7043" />
          <div style={{ flex: 1, minWidth: 180 }}>
            {Object.keys(graphFeats).length === 0 ? (
              <div style={{ fontSize: 10, color: "#333", paddingTop: 8 }}>
                No graph data provided. Pass a propagation_graph in the request to enable this signal.
              </div>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                {[
                  { label: "Nodes",       value: graphFeats.num_nodes?.toLocaleString() ?? "–" },
                  { label: "Max depth",   value: graphFeats.max_depth ?? "–" },
                  { label: "Max breadth", value: graphFeats.max_breadth?.toLocaleString() ?? "–" },
                  {
                    label: "Bot ratio",
                    value: graphFeats.bot_ratio != null ? `${(graphFeats.bot_ratio * 100).toFixed(0)}%` : "–",
                    color: graphFeats.bot_ratio > 0.4 ? "#ff3b3b" : "#00e676",
                  },
                  {
                    label: "Early spread",
                    value: graphFeats.early_spread_ratio != null ? `${(graphFeats.early_spread_ratio * 100).toFixed(0)}%` : "–",
                    color: graphFeats.early_spread_ratio > 0.6 ? "#ff3b3b" : "#00e676",
                  },
                ].map(({ label, value, color }) => (
                  <div key={label} style={{ background: "#111", borderRadius: 6, padding: "8px 10px" }}>
                    <div style={{ fontSize: 9, color: "#444", marginBottom: 3 }}>{label}</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: color || "#777", fontFamily: "monospace" }}>
                      {value}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </ComponentCard>
    </div>
  );
}

export default function App() {
  const [activeDemo, setActiveDemo]   = useState(null);
  const [customTitle, setCustomTitle] = useState("");
  const [customBody, setCustomBody]   = useState("");
  const [analyzing, setAnalyzing]     = useState(false);
  const [tab, setTab]                 = useState("demo");
  const [error, setError]             = useState("");
  const [apiStatus, setApiStatus]     = useState("unknown");

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.ok ? setApiStatus("ok") : setApiStatus("down"))
      .catch(() => setApiStatus("down"));
  }, []);

  const runDemo = useCallback((article) => {
    setError("");
    setAnalyzing(true);
    setTimeout(() => { setActiveDemo(article); setAnalyzing(false); }, 700);
  }, []);

  const runCustom = useCallback(async () => {
    if (!customTitle.trim()) return;
    setError("");
    setAnalyzing(true);
    setActiveDemo(null);

    try {
      const res = await fetch(`${API_URL}/detect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: customTitle.trim(), body: customBody.trim() }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server returned ${res.status}: ${text}`);
      }

      const data = await res.json();

      setActiveDemo({
        title:       customTitle,
        body:        customBody,
        verdict:     data.verdict     || "UNKNOWN",
        confidence:  data.confidence  ?? 0,
        explanation: data.explanation || "",
        bert: {
          label:         data.bert_label         || "UNKNOWN",
          confidence:    data.bert_confidence    ?? 0,
          probabilities: data.bert_probabilities || {},
        },
        factcheck: {
          verdict:    data.factcheck_verdict    || "NO_DATA",
          confidence: data.factcheck_confidence ?? 0,
          summary:    data.factcheck_summary    || "",
        },
        graph: {
          score:    data.graph_score ?? -1,
          features: data.graph_features || {},
        },
      });

    } catch (err) {
      console.error("Detection error:", err);
      setError(
        apiStatus === "down"
          ? "Backend is not running. Start it with: uvicorn api.server:app --port 8000 --reload"
          : `Detection failed: ${err.message}`
      );
    } finally {
      setAnalyzing(false);
    }
  }, [customTitle, customBody, apiStatus]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) runCustom();
  };

  return (
    <div style={{
      background: "#080808", minHeight: "100vh", color: "#e0e0e0",
      fontFamily: "'JetBrains Mono','Fira Code','Consolas',monospace",
      padding: "0 0 80px",
    }}>

      {/* Header */}
      <div style={{
        borderBottom: "1px solid #111", padding: "18px 32px",
        display: "flex", alignItems: "center", gap: 14,
      }}>
        {["#ff3b3b", "#ffaa00", "#00e676"].map(c => (
          <div key={c} style={{ width: 8, height: 8, borderRadius: "50%", background: c, boxShadow: `0 0 10px ${c}` }} />
        ))}
        <div style={{ flex: 1, textAlign: "center" }}>
          <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.22em", color: "#777" }}>
            FAKE NEWS DETECTOR
          </span>
          <span style={{ marginLeft: 12, fontSize: 10, color: "#2a2a2a", letterSpacing: "0.1em" }}>
            BERT · FACT-CHECK · GRAPH
          </span>
        </div>
        {/* Live API status dot */}
        <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 9, color: "#333" }}>
          <div style={{
            width: 6, height: 6, borderRadius: "50%",
            background: apiStatus === "ok" ? "#00e676" : apiStatus === "down" ? "#ff3b3b" : "#555",
          }} />
          {apiStatus === "ok" ? "API LIVE" : apiStatus === "down" ? "API DOWN" : "CHECKING"}
        </div>
      </div>

      <div style={{ maxWidth: 860, margin: "0 auto", padding: "32px 20px" }}>

        {/* Tabs */}
        <div style={{
          display: "flex", marginBottom: 28,
          border: "1px solid #161616", borderRadius: 8,
          overflow: "hidden", width: "fit-content",
        }}>
          {[
            { key: "demo",   label: "Demo Articles" },
            { key: "custom", label: "Analyze Custom" },
          ].map(({ key, label }, i) => (
            <button key={key} onClick={() => setTab(key)} style={{
              padding: "9px 22px", border: "none", cursor: "pointer",
              fontSize: 10, fontWeight: 700, letterSpacing: "0.1em",
              fontFamily: "inherit", textTransform: "uppercase",
              background: tab === key ? "#161616" : "transparent",
              color:      tab === key ? "#e0e0e0" : "#3a3a3a",
              borderRight: i === 0 ? "1px solid #161616" : "none",
              transition: "color 0.2s",
            }}>{label}</button>
          ))}
        </div>

        {/* Demo tab */}
        {tab === "demo" && (
          <div style={{ marginBottom: 28 }}>
            <div style={{ fontSize: 9, color: "#333", marginBottom: 14, letterSpacing: "0.12em" }}>
              SELECT AN ARTICLE TO ANALYZE (DEMO DATA)
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {DEMO_ARTICLES.map((art, i) => {
                const cfg = VERDICT_CONFIG[art.verdict];
                return (
                  <button key={i} onClick={() => runDemo(art)} style={{
                    background: "#0d0d0d", border: "1px solid #161616",
                    borderRadius: 8, padding: "14px 18px", cursor: "pointer",
                    textAlign: "left", display: "flex", alignItems: "center",
                    gap: 14, color: "inherit", fontFamily: "inherit",
                    transition: "border-color 0.2s, background 0.2s",
                  }}
                    onMouseEnter={e => { e.currentTarget.style.borderColor = cfg.border; e.currentTarget.style.background = "#111"; }}
                    onMouseLeave={e => { e.currentTarget.style.borderColor = "#161616"; e.currentTarget.style.background = "#0d0d0d"; }}
                  >
                    <span style={{ fontSize: 16, color: cfg.color, minWidth: 18 }}>{cfg.icon}</span>
                    <div style={{ flex: 1, fontSize: 12, color: "#bbb", lineHeight: 1.5 }}>{art.title}</div>
                    <span style={{ fontSize: 10, color: cfg.color, fontWeight: 700, letterSpacing: "0.08em", whiteSpace: "nowrap" }}>
                      {art.verdict} {(art.confidence * 100).toFixed(0)}%
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Custom tab */}
        {tab === "custom" && (
          <div style={{ marginBottom: 28 }}>
            <div style={{ fontSize: 9, color: "#333", marginBottom: 14, letterSpacing: "0.12em" }}>
              ENTER ARTICLE TO ANALYZE WITH YOUR LIVE MODEL
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <input
                value={customTitle}
                onChange={e => setCustomTitle(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Article headline or claim..."
                style={{
                  background: "#0d0d0d", border: "1px solid #1a1a1a",
                  borderRadius: 8, padding: "12px 16px",
                  color: "#e0e0e0", fontSize: 13, fontFamily: "inherit",
                  outline: "none", width: "100%", boxSizing: "border-box",
                }}
                onFocus={e => e.target.style.borderColor = "#2a2a2a"}
                onBlur={e => e.target.style.borderColor = "#1a1a1a"}
              />
              <textarea
                value={customBody}
                onChange={e => setCustomBody(e.target.value)}
                placeholder="Article body (optional — adds more context for BERT)..."
                rows={4}
                style={{
                  background: "#0d0d0d", border: "1px solid #1a1a1a",
                  borderRadius: 8, padding: "12px 16px",
                  color: "#e0e0e0", fontSize: 12, fontFamily: "inherit",
                  outline: "none", resize: "vertical", width: "100%",
                  boxSizing: "border-box",
                }}
                onFocus={e => e.target.style.borderColor = "#2a2a2a"}
                onBlur={e => e.target.style.borderColor = "#1a1a1a"}
              />

              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <button
                  onClick={runCustom}
                  disabled={!customTitle.trim() || analyzing}
                  style={{
                    background: customTitle.trim() && !analyzing ? "#161616" : "#0d0d0d",
                    border: `1px solid ${customTitle.trim() && !analyzing ? "#333" : "#1a1a1a"}`,
                    borderRadius: 8, padding: "11px 24px",
                    color: customTitle.trim() && !analyzing ? "#e0e0e0" : "#333",
                    fontSize: 10, fontWeight: 700, fontFamily: "inherit",
                    cursor: customTitle.trim() && !analyzing ? "pointer" : "not-allowed",
                    letterSpacing: "0.12em", textTransform: "uppercase",
                    transition: "all 0.2s",
                  }}
                >
                  {analyzing ? "Analyzing..." : "Run Detection"}
                </button>
                <span style={{ fontSize: 9, color: "#2a2a2a" }}>or press Enter</span>
              </div>

              {error && (
                <div style={{
                  color: "#ff3b3b", fontSize: 11, lineHeight: 1.6,
                  padding: "10px 14px", background: "#110000",
                  border: "1px solid #ff3b3b22", borderRadius: 6,
                }}>
                  ⚠ {error}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Analyzing spinner */}
        {analyzing && (
          <div style={{ textAlign: "center", padding: "52px 0", color: "#2a2a2a" }}>
            <div style={{ fontSize: 10, letterSpacing: "0.2em", marginBottom: 18 }}>ANALYZING</div>
            <div style={{ display: "flex", justifyContent: "center", gap: 8 }}>
              {["BERT", "FACT-CHECK", "GRAPH"].map((s, i) => (
                <div key={s} style={{
                  fontSize: 9, padding: "5px 12px", borderRadius: 4,
                  border: "1px solid #1a1a1a", color: "#2a2a2a",
                  animation: `pulse 1.5s ${i * 0.25}s infinite`,
                }}>{s}</div>
              ))}
            </div>
            <style>{`@keyframes pulse { 0%,100%{opacity:0.2} 50%{opacity:0.8} }`}</style>
          </div>
        )}

        {/* Results */}
        {activeDemo && !analyzing && <ResultPanel result={activeDemo} />}

        {/* Empty state */}
        {!activeDemo && !analyzing && (
          <div style={{
            border: "1px solid #111", borderRadius: 8,
            padding: "22px 20px", color: "#2a2a2a", fontSize: 11, lineHeight: 2,
          }}>
            <div style={{ marginBottom: 10, color: "#333", letterSpacing: "0.1em", fontSize: 10 }}>
              SYSTEM ARCHITECTURE
            </div>
            <div>Input → [XLM-RoBERTa]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ linguistic / semantic signal</div>
            <div style={{ color: "#1e1e1e", paddingLeft: 16 }}>└─ fine-tuned on LIAR · FakeNewsNet · CLEF</div>
            <div>Input → [Fact Check API]&nbsp;&nbsp;&nbsp;→ external publisher verdicts</div>
            <div style={{ color: "#1e1e1e", paddingLeft: 16 }}>└─ aggregated from 200+ fact-checkers</div>
            <div>Graph → [PropagationAnalyzer] → spread pattern score</div>
            <div style={{ color: "#1e1e1e", paddingLeft: 16 }}>└─ bot ratio · early spread · tree topology</div>
            <div style={{ marginTop: 8 }}>↓ Weighted fusion (BERT 45% · Fact-check 40% · Graph 15%)</div>
            <div style={{ marginTop: 2, color: "#333" }}>↓ Final verdict: FAKE / REAL / UNVERIFIED</div>
          </div>
        )}
      </div>
    </div>
  );
}