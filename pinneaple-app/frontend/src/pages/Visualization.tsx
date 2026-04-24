import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useActiveProject } from "@/hooks/useProject";
import { useAppStore } from "@/store";
import { getAllRuns } from "@/api/training";
import { runInference } from "@/api/inference";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/Badge";
import { Slider } from "@/components/ui/Slider";
import { TabsUnderline } from "@/components/ui/Tabs";
import { Spinner } from "@/components/ui/Spinner";
import HeatmapChart from "@/components/charts/HeatmapChart";
import AnimationChart from "@/components/charts/AnimationChart";
import ForecastChart from "@/components/charts/ForecastChart";
import type { InferenceResult, TrainingRun } from "@/types";
import { Zap, AlertCircle, Wind } from "lucide-react";
import Plot from "react-plotly.js";
import toast from "react-hot-toast";
import clsx from "clsx";

type FieldKey = "vel_mag" | "ux" | "uy" | "vorticity" | "Q" | "rho";

const FIELD_OPTIONS: { key: FieldKey; label: string; colorscale: string; reverse?: boolean }[] = [
  { key: "vel_mag",   label: "Velocity |u|",  colorscale: "Viridis"  },
  { key: "ux",        label: "ux",            colorscale: "RdBu",    reverse: true },
  { key: "uy",        label: "uy",            colorscale: "RdBu",    reverse: true },
  { key: "vorticity", label: "Vorticity ω",   colorscale: "RdBu",    reverse: true },
  { key: "Q",         label: "Q-Criterion",   colorscale: "Plasma"   },
  { key: "rho",       label: "Density ρ",     colorscale: "Cividis"  },
];

export default function Visualization() {
  const { data: project }      = useActiveProject();
  const { activeRunId, tsData, setActiveRun } = useAppStore();

  const [result, setResult]    = useState<InferenceResult | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(activeRunId);
  const [tab, setTab]          = useState("fields");
  const [field, setField]      = useState<FieldKey>("vel_mag");
  const [qThreshold, setQThreshold] = useState<number>(0);
  const [animField, setAnimField] = useState<"velocity" | "ux" | "uy">("velocity");

  const { data: allRuns } = useQuery({ queryKey: ["runs_all"], queryFn: getAllRuns });

  const doneRuns = (allRuns ?? []).filter((r) => r.status === "done")
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

  const selectedRun = doneRuns.find((r) => r.id === selectedRunId) ?? doneRuns[0] ?? null;
  const modelType   = selectedRun?.model_type ?? "";
  const isLBM       = modelType === "lbm";
  const isPINN      = modelType === "pinn_mlp";
  const isTS        = ["tcn", "lstm", "tft", "fft"].includes(modelType);
  const isFDM       = modelType === "fdm";
  const isFEM       = modelType === "fem";

  const inferMut = useMutation({
    mutationFn: () => runInference(selectedRun!.id, tsData ?? undefined),
    onSuccess:  (d) => { setResult(d); toast.success("Inference loaded"); },
    onError:    (e: Error) => toast.error(e.message),
  });

  const handleSelectRun = (run: TrainingRun) => {
    setSelectedRunId(run.id);
    setActiveRun(run.id, run.ws_run_id);
    setResult(null);
  };

  // Q-criterion thresholded field
  const qMasked: number[][] | null =
    result?.Q && qThreshold > 0
      ? result.Q.map((row) => row.map((v) => (v >= qThreshold ? v : null) as number))
      : (result?.Q ?? null);

  const TABS = [
    { id: "fields",    label: "Flow Fields"  },
    { id: "animation", label: "Animation"    },
    { id: "vortex",    label: "Vortex ID"    },
    { id: "other",     label: isTS ? "Forecast" : "Solution" },
  ];

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-text">Visualization</h1>
        <p className="text-muted text-sm mt-1">
          Interactive CFD and field visualization — animation playback, Q-criterion, and more.
        </p>
      </div>

      {/* Run selector */}
      <Card title="Run" subtitle="Select a completed run to visualize">
        <div className="flex flex-wrap gap-2 mb-3">
          {doneRuns.slice(0, 8).map((r) => (
            <button
              key={r.id}
              className={clsx(
                "px-3 py-1.5 rounded-lg border text-xs font-mono transition-all",
                r.id === selectedRun?.id
                  ? "border-primary/50 bg-primary/10 text-primary"
                  : "border-border text-muted hover:border-border/70 hover:text-text"
              )}
              onClick={() => handleSelectRun(r)}
            >
              {r.model_type.toUpperCase()} · {r.created_at.slice(5, 16)}
            </button>
          ))}
        </div>
        {selectedRun && (
          <div className="flex items-center gap-3">
            <StatusBadge status={selectedRun.status} />
            <span className="text-xs text-muted font-mono">
              {selectedRun.model_type.toUpperCase()}
            </span>
            {selectedRun.final_loss != null && (
              <span className="text-xs text-muted font-mono">
                loss: {selectedRun.final_loss.toExponential(3)}
              </span>
            )}
            <button
              className="btn-primary ml-auto"
              onClick={() => inferMut.mutate()}
              disabled={inferMut.isPending || !selectedRun}
            >
              {inferMut.isPending ? <Spinner size="sm" /> : <Zap size={14} />}
              Load Visualization
            </button>
          </div>
        )}
        {doneRuns.length === 0 && (
          <div className="flex items-center gap-2 text-muted text-sm py-4">
            <AlertCircle size={16} />
            No completed runs yet.
          </div>
        )}
      </Card>

      {/* Results */}
      {result && (
        <>
          <TabsUnderline tabs={TABS} active={tab} onChange={setTab} />

          {/* ── Flow Fields ─────────────────────────────────────────── */}
          {tab === "fields" && isLBM && (
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {FIELD_OPTIONS.filter((f) => !!result[f.key]).map((f) => (
                  <button
                    key={f.key}
                    className={clsx(
                      "px-3 py-1.5 rounded-full text-xs font-medium border transition-all",
                      field === f.key
                        ? "bg-primary/15 text-primary border-primary/40"
                        : "border-border text-muted hover:text-text"
                    )}
                    onClick={() => setField(f.key)}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
              {result[field] && (
                <Card title={FIELD_OPTIONS.find((f) => f.key === field)?.label ?? field}>
                  <HeatmapChart
                    z={result[field] as number[][]}
                    colorscale={FIELD_OPTIONS.find((f) => f.key === field)?.colorscale ?? "Viridis"}
                    reversescale={FIELD_OPTIONS.find((f) => f.key === field)?.reverse}
                    maskWith={result.obstacle}
                    height={360}
                  />
                </Card>
              )}
            </div>
          )}

          {tab === "fields" && !isLBM && (
            <Card title="Field">
              {isPINN && result.u && (
                <HeatmapChart
                  z={result.u}
                  x={result.x}
                  y={result.y}
                  title="u(x, y)"
                  colorscale="Viridis"
                  height={360}
                />
              )}
              {(isFDM || isFEM) && result.field && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <HeatmapChart z={result.field}  title="Numerical" colorscale="Viridis" height={280} />
                  {result.exact && (
                    <HeatmapChart z={result.exact} title="Exact"     colorscale="Viridis" height={280} />
                  )}
                </div>
              )}
              {isTS && (
                <div className="text-muted text-sm text-center py-8">
                  Switch to the "Forecast" tab for timeseries visualization.
                </div>
              )}
            </Card>
          )}

          {/* ── Animation ───────────────────────────────────────────── */}
          {tab === "animation" && (
            <Card title="LBM Trajectory Animation" subtitle="Frame-by-frame playback">
              {isLBM && result.trajectory_ux && result.trajectory_ux.length > 0 ? (
                <>
                  <div className="flex gap-2 mb-3">
                    {(["velocity", "ux", "uy"] as const).map((f) => (
                      <button
                        key={f}
                        className={clsx(
                          "px-3 py-1 text-xs rounded-lg border transition-all",
                          animField === f
                            ? "border-primary/50 bg-primary/10 text-primary"
                            : "border-border text-muted hover:text-text"
                        )}
                        onClick={() => setAnimField(f)}
                      >
                        {f === "velocity" ? "|u|" : f}
                      </button>
                    ))}
                  </div>
                  <AnimationChart
                    frames={result.trajectory_ux}
                    framesUy={result.trajectory_uy}
                    obstacle={result.obstacle}
                    field={animField}
                    title="LBM"
                    height={360}
                  />
                </>
              ) : (
                <div className="flex items-center justify-center py-16 text-muted text-sm">
                  {isLBM
                    ? "No trajectory frames available. Increase 'save_every' and re-run."
                    : "Animation is only available for LBM runs."}
                </div>
              )}
            </Card>
          )}

          {/* ── Vortex ID ────────────────────────────────────────────── */}
          {tab === "vortex" && (
            <div className="space-y-4">
              {isLBM && result.Q ? (
                <>
                  <Card title="Q-Criterion Threshold" subtitle="Show only Q > threshold (vortex cores)">
                    <Slider
                      label={`Threshold: ${qThreshold.toFixed(4)}`}
                      value={qThreshold}
                      onChange={setQThreshold}
                      min={0}
                      max={Math.max(...result.Q.flat().filter((v) => isFinite(v))) * 0.5}
                      step={0.0001}
                    />
                    <div className="mt-4">
                      <HeatmapChart
                        z={qMasked ?? result.Q}
                        colorscale="Plasma"
                        maskWith={result.obstacle}
                        height={360}
                      />
                    </div>
                  </Card>

                  {result.vorticity && (
                    <Card title="Vorticity + Velocity Overlay" subtitle="Vorticity ω with vel_mag contour">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <HeatmapChart
                          z={result.vorticity}
                          title="Vorticity ω"
                          colorscale="RdBu"
                          reversescale
                          maskWith={result.obstacle}
                          height={280}
                        />
                        {result.vel_mag && (
                          <HeatmapChart
                            z={result.vel_mag}
                            title="Velocity |u|"
                            colorscale="Viridis"
                            maskWith={result.obstacle}
                            height={280}
                          />
                        )}
                      </div>
                    </Card>
                  )}

                  {/* Vorticity line-out along centreline */}
                  {result.vorticity && (
                    <Card
                      title="Centreline Vorticity Profile"
                      subtitle="ω(x) at y = ny/2"
                    >
                      <CentrelineChart
                        field={result.vorticity}
                        label="ω"
                        color="#4ECDC4"
                      />
                    </Card>
                  )}
                </>
              ) : (
                <div className="card p-10 flex flex-col items-center gap-3 text-center">
                  <Wind size={36} className="text-muted" />
                  <p className="text-muted text-sm">
                    Q-criterion is only available for LBM runs.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* ── Forecast / Solution ─────────────────────────────────── */}
          {tab === "other" && (
            <Card title={isTS ? "Timeseries Forecast" : "Solution Summary"}>
              {isTS && result.forecast && (
                <ForecastChart
                  history={tsData ?? []}
                  forecast={result.forecast}
                  inputLen={result.input_len}
                  height={360}
                />
              )}
              {(isFDM || isFEM) && (
                <div className="grid grid-cols-2 gap-3">
                  {result.l2_error != null && (
                    <MetricBox label="L2 error"  value={result.l2_error.toExponential(4)} />
                  )}
                  {result.linf_error != null && (
                    <MetricBox label="L∞ error"  value={result.linf_error.toExponential(4)} />
                  )}
                  <MetricBox label="Model"  value={(modelType).toUpperCase()} />
                  {selectedRun?.final_loss != null && (
                    <MetricBox
                      label="Final loss"
                      value={selectedRun.final_loss.toExponential(4)}
                    />
                  )}
                </div>
              )}
              {isPINN && (
                <div className="grid grid-cols-2 gap-3">
                  <MetricBox label="Model"      value="PINN (MLP)"  />
                  {selectedRun?.final_loss != null && (
                    <MetricBox
                      label="Final loss"
                      value={selectedRun.final_loss.toExponential(4)}
                    />
                  )}
                </div>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function CentrelineChart({
  field,
  label,
  color,
}: {
  field: number[][];
  label: string;
  color: string;
}) {
  const ny  = field[0].length;
  const mid = Math.floor(ny / 2);
  const y   = field.map((col) => col[mid] ?? 0);
  const x   = Array.from({ length: y.length }, (_, i) => i);

  return (
    <Plot
      data={[{
        x, y, mode: "lines", name: label,
        line: { color, width: 1.5 },
      }]}
      layout={{
        paper_bgcolor: "transparent", plot_bgcolor: "transparent",
        font:  { color: "#8892a4", size: 11 },
        height: 220,
        xaxis: { title: "x (grid)", gridcolor: "#2a3a5c" },
        yaxis: { title: label,      gridcolor: "#2a3a5c" },
        margin: { l: 50, r: 20, t: 10, b: 40 },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="card-inner p-3">
      <div className="text-xs text-muted mb-0.5">{label}</div>
      <div className="text-sm font-mono text-text">{value}</div>
    </div>
  );
}
