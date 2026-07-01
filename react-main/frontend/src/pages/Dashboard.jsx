import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import DashboardRoundedIcon from "@mui/icons-material/DashboardRounded";
import ScienceRoundedIcon from "@mui/icons-material/ScienceRounded";
import WaterDropRoundedIcon from "@mui/icons-material/WaterDropRounded";
import TimelineRoundedIcon from "@mui/icons-material/TimelineRounded";
import MemoryRoundedIcon from "@mui/icons-material/MemoryRounded";
import ArticleRoundedIcon from "@mui/icons-material/ArticleRounded";
import ShieldRoundedIcon from "@mui/icons-material/ShieldRounded";
import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import PendingRoundedIcon from "@mui/icons-material/PendingRounded";
import OpenInNewRoundedIcon from "@mui/icons-material/OpenInNewRounded";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import ZoomOutMapRoundedIcon from "@mui/icons-material/ZoomOutMapRounded";

const assetPath = (path) => `${import.meta.env.BASE_URL}${path}`;

const DATA = {
  benchmark: assetPath("data/benchmark_delta_srs.csv"),
  robustez: assetPath("data/resumen_robustez.csv"),
  detalle: assetPath("data/detalle_escenarios.csv"),
  saCpu: assetPath("data/qcentroid_sa_cpu_resultado.json"),
  saGpu: assetPath("data/qcentroid_sa_gpu_resultado.json"),
  qaoa: assetPath("data/qcentroid_qaoa_resultado.json"),
  simulacion: assetPath("data/simulacion_hibrida.png"),
  robustezBox: assetPath("data/boxplot_srs_robustez.png"),
  robustezHeatmap: assetPath("data/heatmap_semanas_criticas.png"),
  robustezRanking: assetPath("data/ranking_robustez.png"),
  qaoaValidation: assetPath("data/benchmark_qaoa/qaoa_validation_18q.png"),
  scalingBenchmark: assetPath("data/benchmark_qaoa/scaling_search_runtime.png"),
  costBreakdown: assetPath("data/benchmark_qaoa/cost_breakdown_srs.png"),
  reportePdf: assetPath("reports/reporte_resultados_falcon.pdf"),
  reporteTex: assetPath("reports/reporte_resultados_falcon.tex"),
};

const pendingRuns = [
  {
    name: "QCentroid SA CPU",
    status: "disponible",
    folder: "resultados/resultados_sa_cpu",
    note: "Corrida reproducible sin GPU.",
  },
  {
    name: "QCentroid SA GPU",
    status: "disponible",
    folder: "resultados/resultados_sa_gpu",
    note: "Ruta recomendada para entrega.",
  },
  {
    name: "QCentroid QAOA",
    status: "disponible",
    folder: "resultados/Resultados QAOA",
    note: "Ruta cuantica/variacional experimental.",
  },
];

const menu = [
  ["Dashboard", DashboardRoundedIcon],
  ["Benchmark", TimelineRoundedIcon],
  ["QCentroid", MemoryRoundedIcon],
  ["QAOA", ScienceRoundedIcon],
  ["Robustez", ShieldRoundedIcon],
  ["Reporte", ArticleRoundedIcon],
];

const fmt = new Intl.NumberFormat("es-MX", { maximumFractionDigits: 4 });
const pct = new Intl.NumberFormat("es-MX", { maximumFractionDigits: 1 });

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return headers.reduce((row, header, idx) => {
      const raw = values[idx] ?? "";
      const numeric = Number(raw);
      row[header] = raw !== "" && Number.isFinite(numeric) ? numeric : raw;
      return row;
    }, {});
  });
}

function splitCsvLine(line) {
  const out = [];
  let current = "";
  let quoted = false;
  for (const char of line) {
    if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      out.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  out.push(current);
  return out;
}

async function safeLoad(path, type = "csv") {
  try {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`No disponible: ${path}`);
    const text = await response.text();
    return type === "json" ? JSON.parse(text) : parseCsv(text);
  } catch {
    return type === "json" ? null : [];
  }
}

function methodShort(name = "") {
  return name
    .replace("Baseline 0 — ", "")
    .replace("Baseline 1 — ", "")
    .replace("Baseline 2 — ", "")
    .replace("Cuántico   — ", "")
    .replace("Política   — ", "")
    .replace(" (u=0)", "")
    .replace(" (DEAP)", "");
}

function Card({ title, kicker, children, className = "", action, id }) {
  return (
    <section className={`card ${className}`} id={id}>
      <div className="cardHeader">
        <div>
          {kicker && <p className="kicker">{kicker}</p>}
          <h2>{title}</h2>
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

function Stat({ icon, label, value, detail, tone = "blue", children }) {
  return (
    <div className="stat">
      <div className={`statIcon ${tone}`}>{icon}</div>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
        <span>{detail}</span>
        {children}
      </div>
    </div>
  );
}

function StatusPill({ status }) {
  const ready = status === "disponible";
  return (
    <span className={`pill ${ready ? "ready" : "pending"}`}>
      {ready ? <CheckCircleRoundedIcon /> : <PendingRoundedIcon />}
      {ready ? "Disponible" : "Pendiente"}
    </span>
  );
}

export default function Dashboard() {
  const [benchmark, setBenchmark] = useState([]);
  const [robustez, setRobustez] = useState([]);
  const [detalle, setDetalle] = useState([]);
  const [saCpu, setSaCpu] = useState(null);
  const [saGpu, setSaGpu] = useState(null);
  const [qaoa, setQaoa] = useState(null);
  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    Promise.all([
      safeLoad(DATA.benchmark),
      safeLoad(DATA.robustez),
      safeLoad(DATA.detalle),
      safeLoad(DATA.saCpu, "json"),
      safeLoad(DATA.saGpu, "json"),
      safeLoad(DATA.qaoa, "json"),
    ]).then(([bench, rob, det, cpu, gpu, q]) => {
      setBenchmark(bench);
      setRobustez(rob);
      setDetalle(det);
      setSaCpu(cpu);
      setSaGpu(gpu);
      setQaoa(q);
    });
  }, []);

  useEffect(() => {
    if (!expanded) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setExpanded(null);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [expanded]);

  const bestBenchmark = useMemo(() => {
    return benchmark.reduce((best, row) => (row.SRS > (best?.SRS ?? -Infinity) ? row : best), null);
  }, [benchmark]);

  const historic = benchmark.find((row) => String(row.Metodo).includes("Histórico"));
  const bestRobust = robustez[0];
  const feasibleCount = robustez.filter((row) => Number(row.fallos_totales) === 0).length;
  const feasibleMethods = robustez.filter((row) => Number(row.fallos_totales) === 0);
  const failedMethods = robustez.filter((row) => Number(row.fallos_totales) !== 0);

  const benchmarkChart = benchmark.map((row) => ({
    name: methodShort(row.Metodo),
    SRS: Number(row.SRS),
    delta: Number(row["ΔSRS"] ?? 0),
  }));

  const robustnessChart = robustez.map((row) => ({
    name: row.metodo,
    score: Number(row.robustez_score),
    worst: Number(row.SRS_peor),
    critical: Number(row.semanas_criticas_prom),
  }));

  const scenarioChart = useMemo(() => {
    const rows = new Map();
    detalle
      .filter((row) => row.metodo === "QUBO+SA reoptimizado" || row.metodo === "QUBO+SA nominal")
      .forEach((row) => {
        const prev = rows.get(row.escenario) || { escenario: row.escenario };
        prev[row.metodo] = Number(row.SRS);
        rows.set(row.escenario, prev);
      });
    return Array.from(rows.values());
  }, [detalle]);

  const qcentroidRuns = pendingRuns.map((run) => {
    if (run.name.includes("SA CPU") && saCpu) {
      return {
        ...run,
        srs: saCpu.SRS,
        time: saCpu.tiempo_s,
        meta: "sa_cpu",
      };
    }
    if (run.name.includes("SA GPU") && saGpu) {
      return {
        ...run,
        srs: saGpu.SRS,
        time: saGpu.tiempo_s,
        meta: "sa_gpu",
      };
    }
    if (run.name.includes("QAOA") && qaoa) {
      const fallback = qaoa.meta?.fallback_from ? `fallback desde ${qaoa.meta.fallback_from}` : "qaoa";
      return {
        ...run,
        srs: qaoa.SRS,
        time: qaoa.tiempo_s,
        meta: qaoa.meta?.modo_efectivo || qaoa.meta?.modo_solicitado || fallback,
      };
    }
    return run;
  });

  const openExpanded = (payload) => setExpanded(payload);

  const openWithKeyboard = (event, payload) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openExpanded(payload);
    }
  };

  const renderBenchmarkChart = (height = 320) => {
    const waterBaseId = `waterBase-${height}`;
    const waterMotionId = `waterMotion-${height}`;
    const waterSoftId = `waterSoft-${height}`;
    return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={benchmarkChart} margin={{ left: 8, right: 12, top: 12, bottom: 20 }}>
        <defs>
          <linearGradient id={waterBaseId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#2d7ff9" />
            <stop offset="100%" stopColor="#0f5fce" />
          </linearGradient>
          <pattern id={waterMotionId} width="48" height="16" patternUnits="userSpaceOnUse">
            <rect width="48" height="16" fill={`url(#${waterBaseId})`} />
            <path
              d="M0 9 C8 2 16 2 24 9 S40 16 48 9 V16 H0 Z"
              fill="rgba(255,255,255,0.28)"
            >
              <animateTransform
                attributeName="transform"
                type="translate"
                from="-48 0"
                to="48 0"
                dur="4s"
                repeatCount="indefinite"
              />
            </path>
            <path
              d="M0 5 C8 11 16 11 24 5 S40 -1 48 5"
              fill="none"
              stroke="rgba(255,255,255,0.32)"
              strokeWidth="1.4"
            >
              <animateTransform
                attributeName="transform"
                type="translate"
                from="48 0"
                to="-48 0"
                dur="5.5s"
                repeatCount="indefinite"
              />
            </path>
          </pattern>
          <linearGradient id={waterSoftId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#9cc5ff" />
            <stop offset="100%" stopColor="#6ea2ee" />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf6" />
        <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} />
        <YAxis
          tick={{ fontSize: 12 }}
          domain={[-0.17, -0.14]}
          tickFormatter={(value) => Number(value).toFixed(3)}
        />
        <Tooltip formatter={(value) => fmt.format(value)} />
        <Bar dataKey="SRS" radius={[8, 8, 0, 0]}>
          {benchmarkChart.map((entry) => (
            <Cell
              key={entry.name}
              fill={entry.SRS === bestBenchmark?.SRS ? `url(#${waterMotionId})` : `url(#${waterSoftId})`}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
    );
  };

  const renderRobustnessChart = (height = 310) => {
    const scoreFillId = `scoreFill-${height}`;
    return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={robustnessChart} margin={{ left: 8, right: 14, top: 12, bottom: 20 }}>
        <defs>
          <linearGradient id={scoreFillId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#1f6feb" stopOpacity={0.35} />
            <stop offset="100%" stopColor="#1f6feb" stopOpacity={0.04} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf6" />
        <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} />
        <YAxis
          tick={{ fontSize: 12 }}
          domain={[-0.50, -0.44]}
          tickFormatter={(value) => Number(value).toFixed(3)}
        />
        <Tooltip formatter={(value) => fmt.format(value)} />
        <Area dataKey="score" stroke="#1f6feb" fill={`url(#${scoreFillId})`} strokeWidth={3} />
      </AreaChart>
    </ResponsiveContainer>
    );
  };

  const renderScenarioChart = (height = 260) => (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={scenarioChart} margin={{ left: 8, right: 12, top: 12, bottom: 12 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf6" />
        <XAxis dataKey="escenario" tick={{ fontSize: 11 }} />
        <YAxis
          tick={{ fontSize: 12 }}
          domain={[-0.24, -0.13]}
          tickFormatter={(value) => Number(value).toFixed(3)}
        />
        <Tooltip formatter={(value) => fmt.format(value)} />
        <Legend />
        <Line dataKey="QUBO+SA nominal" stroke="#8ab4f8" strokeWidth={2} dot />
        <Line dataKey="QUBO+SA reoptimizado" stroke="#16a34a" strokeWidth={2} dot />
      </LineChart>
    </ResponsiveContainer>
  );

  const renderExpanded = () => {
    if (!expanded) return null;
    if (expanded.type === "benchmark") return renderBenchmarkChart(560);
    if (expanded.type === "robustez") return renderRobustnessChart(540);
    if (expanded.type === "escenarios") return renderScenarioChart(520);
    if (expanded.type === "image") {
      return <img className="modalImage" src={expanded.src} alt={expanded.title} />;
    }
    return null;
  };

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandMark">
            <WaterDropRoundedIcon />
          </div>
          <div>
            <strong>Falcon</strong>
            <span>Reservoir Lab · Equipo 3</span>
          </div>
        </div>

        <nav>
          {menu.map(([label, Icon], index) => (
            <a className={index === 0 ? "active" : ""} href={`#${label.toLowerCase()}`} key={label}>
              <Icon />
              {label}
            </a>
          ))}
        </nav>

        <div className="appearance">
          <p>Instancia</p>
          <strong>T=26 · L=5</strong>
          <span>130 variables binarias</span>
        </div>
      </aside>

      <main className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">Challenge A · International Falcon Reservoir</p>
            <h1>Dashboard de optimizacion y resiliencia hidrica</h1>
          </div>
          <div className="searchBox">Resultados completos · reporte final</div>
        </header>

        <section className="statsGrid" id="dashboard">
          <Stat
            icon={<ScienceRoundedIcon />}
            label="Mejor SRS nominal"
            value={bestBenchmark ? fmt.format(bestBenchmark.SRS) : "Pendiente"}
            detail={bestBenchmark ? methodShort(bestBenchmark.Metodo) : "Esperando resultados"}
            tone="green"
          />
          <Stat
            icon={<TimelineRoundedIcon />}
            label="Delta SRS"
            value={bestBenchmark && historic ? `+${fmt.format(bestBenchmark.SRS - historic.SRS)}` : "Pendiente"}
            detail="vs historico observado"
            tone="blue"
          />
          <Stat
            icon={<ShieldRoundedIcon />}
            label="Robustez lider"
            value={bestRobust ? "QUBO+SA" : "Pendiente"}
            detail={bestRobust ? `${bestRobust.metodo.replace("QUBO+SA ", "")} · score ${fmt.format(bestRobust.robustez_score)}` : "Sin ranking"}
            tone="amber"
          />
          <Stat
            icon={<CheckCircleRoundedIcon />}
            label="Metodos factibles"
            value={robustez.length ? `${feasibleCount}/${robustez.length}` : "Pendiente"}
            detail="en escenarios de estres"
            tone="cyan"
          >
            {robustez.length > 0 && (
              <details className="statDetails">
                <summary>Ver desglose</summary>
                <div>
                  <p>
                    El conteo compara {robustez.length} metodos evaluados en robustez, no
                    las restricciones del modelo.
                  </p>
                  <ul>
                    {feasibleMethods.map((row) => (
                      <li key={row.metodo}>{row.metodo}: sin fallos</li>
                    ))}
                    {failedMethods.map((row) => (
                      <li key={row.metodo}>
                        {row.metodo}: {row.fallos_totales} fallo en escenarios
                      </li>
                    ))}
                  </ul>
                  {failedMethods[0] && (
                    <p>
                      {failedMethods[0].metodo} fue factible en 6 de 7 escenarios
                      ({pct.format(100 * Number(failedMethods[0].factibilidad_prom))}%).
                    </p>
                  )}
                </div>
              </details>
            )}
          </Stat>
        </section>

        <section className="mainGrid">
          <Card title="Benchmark SRS" kicker="Comparativa nominal" className="wide" id="benchmark">
            <div
              className="chartArea expandable"
              role="button"
              tabIndex={0}
              onClick={() => openExpanded({ type: "benchmark", title: "Benchmark SRS" })}
              onKeyDown={(event) => openWithKeyboard(event, { type: "benchmark", title: "Benchmark SRS" })}
            >
              <span className="expandHint"><ZoomOutMapRoundedIcon /> Ampliar</span>
              {renderBenchmarkChart()}
            </div>
            <div className="table compact">
              {benchmark.map((row) => (
                <div className="tableRow" key={row.Metodo}>
                  <span>{methodShort(row.Metodo)}</span>
                  <strong>{fmt.format(row.SRS)}</strong>
                  <em>{Number(row["ΔSRS"]) > 0 ? "+" : ""}{fmt.format(row["ΔSRS"] ?? 0)}</em>
                </div>
              ))}
            </div>
          </Card>

          <Card title="QCentroid" kicker="Corridas separadas" id="qcentroid">
            <div className="runList">
              {qcentroidRuns.map((run) => (
                <div className="runItem" key={run.name}>
                  <div>
                    <strong>{run.name}</strong>
                    <span>{run.note}</span>
                    <small>{run.folder}</small>
                  </div>
                  <StatusPill status={run.status} />
                  {run.srs !== undefined && (
                    <div className="runMetric">
                      <b>{fmt.format(run.srs)}</b>
                      <span>{run.time ? `${pct.format(run.time)} s` : run.meta}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>

          <Card title="Validacion QAOA" kicker="Slice real y escalabilidad" id="qaoa">
            <div className="insightList">
              <div>
                <strong>18 qubits</strong>
                <span>QAOA statevector alcanzo el minimo QUBO exacto en una ventana real reducida.</span>
              </div>
              <div>
                <strong>130 qubits</strong>
                <span>La instancia oficial se mantiene como QUBO/SA por costo de simulacion QAOA completo.</span>
              </div>
            </div>
            <div className="stackedEvidence">
              <button
                className="imageButton"
                type="button"
                onClick={() => openExpanded({ type: "image", title: "Validacion QAOA en 18 qubits", src: DATA.qaoaValidation })}
              >
                <img src={DATA.qaoaValidation} alt="Validacion QAOA 18 qubits" />
                <span><ZoomOutMapRoundedIcon /> Ampliar</span>
              </button>
              <button
                className="imageButton"
                type="button"
                onClick={() => openExpanded({ type: "image", title: "Escalabilidad: espacio de busqueda y runtime", src: DATA.scalingBenchmark })}
              >
                <img src={DATA.scalingBenchmark} alt="Escalabilidad benchmark" />
                <span><ZoomOutMapRoundedIcon /> Ampliar</span>
              </button>
            </div>
          </Card>

          <Card title="Robustez" kicker="Ranking bajo estres" className="wide" id="robustez">
            <div
              className="chartArea expandable"
              role="button"
              tabIndex={0}
              onClick={() => openExpanded({ type: "robustez", title: "Ranking de robustez" })}
              onKeyDown={(event) => openWithKeyboard(event, { type: "robustez", title: "Ranking de robustez" })}
            >
              <span className="expandHint"><ZoomOutMapRoundedIcon /> Ampliar</span>
              {renderRobustnessChart()}
            </div>
            <div className="evidenceGrid">
              <button
                className="imageButton"
                type="button"
                onClick={() => openExpanded({ type: "image", title: "Heatmap de semanas criticas", src: DATA.robustezHeatmap })}
              >
                <img src={DATA.robustezHeatmap} alt="Heatmap de semanas criticas" />
                <span><ZoomOutMapRoundedIcon /> Ampliar</span>
              </button>
              <button
                className="imageButton"
                type="button"
                onClick={() => openExpanded({ type: "image", title: "Ranking de robustez", src: DATA.robustezRanking })}
              >
                <img src={DATA.robustezRanking} alt="Ranking de robustez" />
                <span><ZoomOutMapRoundedIcon /> Ampliar</span>
              </button>
            </div>
          </Card>

          <Card title="Trayectoria del embalse" kicker="Grafica generada por el pipeline">
            <button
              className="imageFrame imageButton tall"
              type="button"
              onClick={() => openExpanded({ type: "image", title: "Trayectoria del embalse", src: DATA.simulacion })}
            >
              <img src={DATA.simulacion} alt="Simulacion hibrida del embalse" />
              <span><ZoomOutMapRoundedIcon /> Ampliar</span>
            </button>
          </Card>

          <Card title="Escenarios" kicker="QUBO nominal vs reoptimizado">
            <div
              className="chartArea short expandable"
              role="button"
              tabIndex={0}
              onClick={() => openExpanded({ type: "escenarios", title: "Escenarios: QUBO nominal vs reoptimizado" })}
              onKeyDown={(event) => openWithKeyboard(event, { type: "escenarios", title: "Escenarios: QUBO nominal vs reoptimizado" })}
            >
              <span className="expandHint"><ZoomOutMapRoundedIcon /> Ampliar</span>
              {renderScenarioChart()}
            </div>
          </Card>

          <Card title="Costos SRS" kicker="Que impulsa la metrica">
            <button
              className="imageFrame imageButton"
              type="button"
              onClick={() => openExpanded({ type: "image", title: "Desglose de costos SRS", src: DATA.costBreakdown })}
            >
              <img src={DATA.costBreakdown} alt="Desglose de costos SRS" />
              <span><ZoomOutMapRoundedIcon /> Ampliar</span>
            </button>
          </Card>

          <Card
            title="Reporte tecnico"
            kicker="LaTeX y anexos"
            action={
              <a className="iconButton" href={DATA.reportePdf} target="_blank" rel="noreferrer">
                <OpenInNewRoundedIcon />
              </a>
            }
          >
            <div className="reportPanel" id="reporte">
              <ArticleRoundedIcon />
              <div>
                <strong>Reporte LaTeX final</strong>
                <span>Compilado con resultados nominales, QCentroid y robustez.</span>
              </div>
            </div>
            <p className="note">
              Abrir PDF final: <a href={DATA.reportePdf} target="_blank" rel="noreferrer">reporte_resultados_falcon.pdf</a>.
              Fuente LaTeX: <a href={DATA.reporteTex} target="_blank" rel="noreferrer">reporte_resultados_falcon.tex</a>.
            </p>
          </Card>
        </section>
      </main>
      {expanded && (
        <div className="modalBackdrop" role="presentation" onClick={() => setExpanded(null)}>
          <section
            className="modalPanel"
            role="dialog"
            aria-modal="true"
            aria-label={expanded.title}
            onClick={(event) => event.stopPropagation()}
          >
            <header className="modalHeader">
              <div>
                <p className="kicker">Vista ampliada</p>
                <h2>{expanded.title}</h2>
              </div>
              <button className="modalClose" type="button" onClick={() => setExpanded(null)} aria-label="Cerrar">
                <CloseRoundedIcon />
              </button>
            </header>
            <div className="modalContent">{renderExpanded()}</div>
          </section>
        </div>
      )}
    </div>
  );
}
