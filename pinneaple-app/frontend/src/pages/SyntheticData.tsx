import { useEffect, useState, useCallback } from "react";
import Plot from "react-plotly.js";
import { FlaskConical, Download, Play, RotateCcw } from "lucide-react";
import clsx from "clsx";
import api from "@/api/client";
import { Spinner } from "@/components/ui/Spinner";
import HeatmapChart from "@/components/charts/HeatmapChart";

// ── Types ─────────────────────────────────────────────────────────────────────

interface ParamDef {
  name: string;
  type: "select" | "int" | "float";
  default: string | number;
  label: string;
  options?: string[];
  min?: number;
  max?: number;
}

interface GeneratorMeta {
  label: string;
  desc: string;
  output_type: "trajectory" | "scatter";
  params: ParamDef[];
}

interface SampleData {
  fields: Record<string, number[] | number[][]>;
  coords: Record<string, number[]>;
  meta: Record<string, unknown>;
}

interface GenerateResult {
  generator: string;
  output_type: "trajectory" | "scatter";
  n_samples: number;
  samples: SampleData[];
  extras: Record<string, unknown>;
  params: Record<string, unknown>;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function defaultParams(meta: GeneratorMeta): Record<string, string | number> {
  const out: Record<string, string | number> = {};
  for (const p of meta.params) out[p.name] = p.default;
  return out;
}

function downloadBlob(content: string, filename: string, mime = "text/plain") {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function buildCSV(result: GenerateResult): string {
  if (result.output_type === "trajectory") {
    const rows: string[] = ["sample,t_idx,x_idx,u"];
    result.samples.forEach((s, si) => {
      const U = s.fields.u as number[][];
      if (!U) return;
      U.forEach((row, ti) =>
        row.forEach((val, xi) => rows.push(`${si},${ti},${xi},${val}`))
      );
    });
    return rows.join("\n");
  }
  // scatter
  const first = result.samples[0];
  if (!first) return "";
  const x = first.fields.x as number[][];
  const y = first.fields.y as number[][];
  if (!x || !y) return "";
  const dim = x[0]?.length ?? 0;
  const xCols = Array.from({ length: dim }, (_, i) => `x${i}`).join(",");
  const header = `${xCols},y0`;
  const rows = x.map((xi, i) => `${xi.join(",")},${(y[i] as unknown as number[])?.[0] ?? ""}`);
  return [header, ...rows].join("\n");
}

// ── Preview charts ────────────────────────────────────────────────────────────

function TrajectoryPreview({ samples }: { samples: SampleData[] }) {
  if (!samples.length) return null;
  const first = samples[0];
  const U = first.fields.u as number[][] | undefined;
  if (!U || !U.length) return <p className="text-muted text-sm">No field u found.</p>;

  const tArr = (first.coords.t as number[] | undefined) ??
    Array.from({ length: U.length }, (_, i) => i);
  const xArr = (first.coords.x as number[] | undefined) ??
    Array.from({ length: U[0].length }, (_, i) => i);

  // Show multiple sample curves at a single time slice for quick comparison
  const midT = Math.floor(U.length / 2);
  const traces: Plotly.Data[] = samples.slice(0, 6).map((s, i) => {
    const Ui = s.fields.u as number[][] | undefined;
    const row = Ui?.[midT] ?? [];
    return {
      x: xArr,
      y: row,
      mode: "lines" as const,
      name: `Sample ${i + 1}`,
      line: { width: 1.5 },
    };
  });

  const layout: Partial<Plotly.Layout> = {
    paper_bgcolor: "transparent",
    plot_bgcolor:  "transparent",
    font:   { color: "#8892a4", size: 11 },
    height: 220,
    xaxis:  { title: "x", gridcolor: "#2a3a5c" },
    yaxis:  { title: "u(t_mid, x)", gridcolor: "#2a3a5c" },
    legend: { bgcolor: "transparent" },
    margin: { l: 50, r: 20, t: 10, b: 40 },
  };

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="text-xs text-muted mb-1">
          Snapshot at t = {tArr[midT]?.toFixed(4)} (all samples)
        </div>
        <Plot
          data={traces}
          layout={layout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%", height: 220 }}
          useResizeHandler
        />
      </div>

      <div>
        <div className="text-xs text-muted mb-1">
          Sample 1 — space–time heatmap u(t, x)
        </div>
        <HeatmapChart
          z={U}
          x={xArr}
          y={tArr}
          xlabel="x"
          ylabel="t"
          height={240}
          colorscale="Viridis"
        />
      </div>
    </div>
  );
}

function ScatterPreview({ samples }: { samples: SampleData[] }) {
  if (!samples.length) return null;
  const first = samples[0];
  const x = first.fields.x as number[][] | undefined;
  const y = first.fields.y as number[][] | undefined;
  if (!x || !y) return <p className="text-muted text-sm">No fields x / y found.</p>;

  const x0 = x.map((r) => r[0] ?? 0);
  const x1 = x.map((r) => r[1] ?? r[0] ?? 0);
  const yFlat = y.map((r) => (Array.isArray(r) ? r[0] : r) ?? 0) as number[];

  const traces: Plotly.Data[] = [
    {
      x: x0,
      y: x1,
      mode: "markers" as const,
      marker: {
        size: 4,
        color: yFlat,
        colorscale: "Viridis",
        showscale: true,
        colorbar: { thickness: 10, tickfont: { color: "#8892a4", size: 10 } },
      },
      type: "scatter" as const,
    },
  ];

  const layout: Partial<Plotly.Layout> = {
    paper_bgcolor: "transparent",
    plot_bgcolor:  "transparent",
    font:   { color: "#8892a4", size: 11 },
    height: 340,
    xaxis:  { title: "x₀", gridcolor: "#2a3a5c" },
    yaxis:  { title: "x₁", gridcolor: "#2a3a5c" },
    margin: { l: 50, r: 60, t: 10, b: 40 },
    showlegend: false,
  };

  return (
    <div>
      <div className="text-xs text-muted mb-1">
        x₀ vs x₁ — colour = target y = Σxᵢ²
      </div>
      <Plot
        data={traces}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: 340 }}
        useResizeHandler
      />
    </div>
  );
}

// ── Parameter form ────────────────────────────────────────────────────────────

function ParamForm({
  params: paramDefs,
  values,
  onChange,
}: {
  params: ParamDef[];
  values: Record<string, string | number>;
  onChange: (name: string, value: string | number) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      {paramDefs.map((p) => (
        <div key={p.name}>
          <label className="text-xs text-muted block mb-1">{p.label}</label>
          {p.type === "select" ? (
            <select
              className="w-full bg-surface2 border border-border rounded-lg px-3 py-2
                         text-sm text-text focus:outline-none focus:ring-2 focus:ring-primary/40"
              value={String(values[p.name] ?? p.default)}
              onChange={(e) => onChange(p.name, e.target.value)}
            >
              {p.options!.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          ) : (
            <input
              type="number"
              step={p.type === "int" ? "1" : "any"}
              min={p.min}
              max={p.max}
              className="w-full bg-surface2 border border-border rounded-lg px-3 py-2
                         text-sm text-text focus:outline-none focus:ring-2 focus:ring-primary/40"
              value={values[p.name] ?? p.default}
              onChange={(e) =>
                onChange(p.name, p.type === "int" ? parseInt(e.target.value) : parseFloat(e.target.value))
              }
            />
          )}
        </div>
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function SyntheticData() {
  const [catalogue, setCatalogue] = useState<Record<string, GeneratorMeta>>({});
  const [selectedGen, setSelectedGen] = useState<string>("");
  const [paramValues, setParamValues] = useState<Record<string, string | number>>({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load catalogue on mount
  useEffect(() => {
    api.get("/synthesis/catalogue/").then((res) => {
      setCatalogue(res.data);
      const first = Object.keys(res.data)[0];
      if (first) {
        setSelectedGen(first);
        setParamValues(defaultParams(res.data[first]));
      }
    });
  }, []);

  const handleGenChange = (key: string) => {
    setSelectedGen(key);
    setResult(null);
    setError(null);
    if (catalogue[key]) setParamValues(defaultParams(catalogue[key]));
  };

  const handleParamChange = useCallback((name: string, value: string | number) => {
    setParamValues((prev) => ({ ...prev, [name]: value }));
  }, []);

  const handleReset = () => {
    if (catalogue[selectedGen]) setParamValues(defaultParams(catalogue[selectedGen]));
    setResult(null);
    setError(null);
  };

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.post("/synthesis/generate/", {
        generator: selectedGen,
        params: paramValues,
      });
      setResult(res.data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadJSON = () => {
    if (!result) return;
    downloadBlob(JSON.stringify(result, null, 2), `synth_${result.generator}.json`, "application/json");
  };

  const handleDownloadCSV = () => {
    if (!result) return;
    downloadBlob(buildCSV(result), `synth_${result.generator}.csv`, "text/csv");
  };

  const meta = catalogue[selectedGen];

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <FlaskConical size={22} className="text-primary" />
          <h1 className="text-2xl font-bold text-text">Synthetic Data Generation</h1>
        </div>
        <p className="text-muted text-sm max-w-2xl">
          Generate physics-informed synthetic datasets using PINNeAPPle's built-in generators.
          Configure a generator, preview the output, and download as JSON or CSV.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ── Left panel: config ──────────────────────────────────────────── */}
        <div className="lg:col-span-1 flex flex-col gap-4">

          {/* Generator selector */}
          <div className="card p-4">
            <div className="text-xs font-semibold text-muted uppercase tracking-wide mb-3">
              Generator
            </div>
            <div className="flex flex-col gap-2">
              {Object.entries(catalogue).map(([key, m]) => (
                <button
                  key={key}
                  onClick={() => handleGenChange(key)}
                  className={clsx(
                    "text-left p-3 rounded-lg border transition-all",
                    selectedGen === key
                      ? "border-primary/50 bg-primary/10 text-primary"
                      : "border-border hover:border-primary/30 hover:bg-surface2 text-text"
                  )}
                >
                  <div className="text-sm font-medium">{m.label}</div>
                  <div className="text-xs text-muted mt-0.5 line-clamp-2">{m.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Parameter form */}
          {meta && (
            <div className="card p-4">
              <div className="text-xs font-semibold text-muted uppercase tracking-wide mb-3">
                Parameters
              </div>
              <ParamForm
                params={meta.params}
                values={paramValues}
                onChange={handleParamChange}
              />
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2">
            <button
              className="btn-primary flex-1 flex items-center justify-center gap-2 py-2.5"
              onClick={handleGenerate}
              disabled={loading || !selectedGen}
            >
              {loading ? <Spinner size="sm" /> : <Play size={15} />}
              {loading ? "Generating…" : "Generate"}
            </button>
            <button
              className="btn-secondary px-3 py-2.5"
              onClick={handleReset}
              title="Reset parameters"
            >
              <RotateCcw size={15} />
            </button>
          </div>

          {/* Error */}
          {error && (
            <div className="card p-3 border-error/40 bg-error/5 text-error text-xs">
              {error}
            </div>
          )}
        </div>

        {/* ── Right panel: preview ────────────────────────────────────────── */}
        <div className="lg:col-span-2 flex flex-col gap-4">

          {/* Result summary bar */}
          {result && (
            <div className="card p-4 flex items-center justify-between gap-4 flex-wrap">
              <div className="flex items-center gap-4 text-sm">
                <span className="text-muted">Generator:</span>
                <span className="text-text font-medium">{result.generator}</span>
                <span className="text-muted">Samples:</span>
                <span className="text-text font-medium">{result.n_samples}</span>
                <span className="text-muted">Type:</span>
                <span className="text-text font-medium">{result.output_type}</span>
              </div>
              <div className="flex gap-2">
                <button
                  className="btn-secondary text-xs px-3 py-1.5 flex items-center gap-1.5"
                  onClick={handleDownloadJSON}
                >
                  <Download size={12} /> JSON
                </button>
                <button
                  className="btn-secondary text-xs px-3 py-1.5 flex items-center gap-1.5"
                  onClick={handleDownloadCSV}
                >
                  <Download size={12} /> CSV
                </button>
              </div>
            </div>
          )}

          {/* Preview area */}
          <div className="card p-5">
            {!result && !loading && (
              <div className="flex flex-col items-center justify-center py-16 text-center gap-3">
                <FlaskConical size={36} className="text-muted/40" />
                <div className="text-muted text-sm">
                  Configure a generator on the left and click <strong>Generate</strong>.
                </div>
              </div>
            )}

            {loading && (
              <div className="flex flex-col items-center justify-center py-16 gap-3">
                <Spinner size="lg" />
                <span className="text-muted text-sm">Running generator…</span>
              </div>
            )}

            {result && !loading && (
              <div>
                <div className="text-xs font-semibold text-muted uppercase tracking-wide mb-4">
                  Preview — {result.samples.length} sample{result.samples.length !== 1 ? "s" : ""} shown
                </div>

                {result.output_type === "trajectory" && (
                  <TrajectoryPreview samples={result.samples} />
                )}
                {result.output_type === "scatter" && (
                  <ScatterPreview samples={result.samples} />
                )}

                {/* Metadata */}
                <div className="mt-5 border-t border-border pt-4">
                  <div className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">
                    Generation metadata
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    {Object.entries(result.params).map(([k, v]) => (
                      <div key={k} className="bg-surface2 rounded-lg px-3 py-2">
                        <div className="text-xs text-muted">{k}</div>
                        <div className="text-sm text-text font-medium truncate">{String(v)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Stats card */}
          {result && (
            <div className="card p-4">
              <div className="text-xs font-semibold text-muted uppercase tracking-wide mb-3">
                Output stats
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <StatCell label="Generator" value={result.generator} />
                <StatCell label="Total samples" value={String(result.n_samples)} />
                <StatCell
                  label="Fields"
                  value={Object.keys(result.samples[0]?.fields ?? {}).join(", ") || "—"}
                />
                <StatCell
                  label="Coords"
                  value={Object.keys(result.samples[0]?.coords ?? {}).join(", ") || "—"}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface2 rounded-lg px-3 py-2">
      <div className="text-xs text-muted mb-0.5">{label}</div>
      <div className="text-sm text-text font-medium truncate">{value}</div>
    </div>
  );
}
