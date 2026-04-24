import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useActiveProject } from "@/hooks/useProject";
import { useAppStore } from "@/store";
import { getAllRuns } from "@/api/training";
import { runInference } from "@/api/inference";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Spinner";
import HeatmapChart from "@/components/charts/HeatmapChart";
import ForecastChart from "@/components/charts/ForecastChart";
import LossChart from "@/components/charts/LossChart";
import type { InferenceResult, TrainingRun } from "@/types";
import {
  Zap, Eye, AlertCircle, ChevronDown, ChevronUp,
  Download, ArrowRight,
} from "lucide-react";
import toast from "react-hot-toast";
import { format } from "date-fns";
import clsx from "clsx";

export default function Inference() {
  const navigate               = useNavigate();
  const { data: project }      = useActiveProject();
  const { activeRunId, tsData, setActiveRun } = useAppStore();

  const [result, setResult]    = useState<InferenceResult | null>(null);
  const [selectedRun, setSelectedRun] = useState<TrainingRun | null>(null);
  const [expandRuns, setExpandRuns] = useState(false);

  const { data: allRuns } = useQuery({
    queryKey: ["runs_all"],
    queryFn:  getAllRuns,
  });

  const projectRuns = (allRuns ?? [])
    .filter((r) => r.project === project?.id && r.status === "done")
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

  // Auto-select the active run
  const activeRun = selectedRun
    ?? projectRuns.find((r) => r.id === activeRunId)
    ?? projectRuns[0]
    ?? null;

  const inferMut = useMutation({
    mutationFn: () =>
      runInference(activeRun!.id, tsData ?? undefined),
    onSuccess: (data) => {
      setResult(data);
      toast.success("Inference complete");
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const handleSelectRun = (run: TrainingRun) => {
    setSelectedRun(run);
    setActiveRun(run.id, run.ws_run_id);
    setResult(null);
    setExpandRuns(false);
  };

  const downloadJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `inference_${activeRun?.id?.slice(0, 8) ?? "result"}.json`;
    a.click();
  };

  const modelType = activeRun?.model_type ?? "";
  const isTS      = ["tcn", "lstm", "tft", "fft"].includes(modelType);
  const isLBM     = modelType === "lbm";
  const isPINN    = modelType === "pinn_mlp";
  const isFDM     = modelType === "fdm";
  const isFEM     = modelType === "fem";

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text">Inference</h1>
          <p className="text-muted text-sm mt-1">
            Run a trained model forward and inspect the solution fields.
          </p>
        </div>
        {result && (
          <div className="flex gap-2">
            <button className="btn-secondary" onClick={downloadJson}>
              <Download size={14} /> Export JSON
            </button>
            {isLBM && (
              <button className="btn-secondary" onClick={() => navigate("/visualization")}>
                <Eye size={14} /> Visualize
              </button>
            )}
          </div>
        )}
      </div>

      {/* Run selector */}
      {projectRuns.length > 0 ? (
        <Card
          title="Select Run"
          subtitle={`${projectRuns.length} completed run${projectRuns.length !== 1 ? "s" : ""}`}
          action={
            <button
              className="btn-ghost text-xs flex items-center gap-1"
              onClick={() => setExpandRuns((v) => !v)}
            >
              {expandRuns ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              {expandRuns ? "Collapse" : "Expand"}
            </button>
          }
        >
          {/* Always show active run row */}
          {activeRun && (
            <div className="flex items-center justify-between p-3 bg-primary/5 border border-primary/30 rounded-lg mb-2">
              <div className="flex items-center gap-3">
                <StatusBadge status={activeRun.status} />
                <div>
                  <span className="text-sm font-mono text-secondary">
                    {activeRun.model_type.toUpperCase()}
                  </span>
                  <span className="text-xs text-muted ml-2">
                    {format(new Date(activeRun.created_at), "MMM d HH:mm")}
                  </span>
                </div>
                {activeRun.final_loss != null && (
                  <span className="text-xs font-mono text-muted">
                    loss: {activeRun.final_loss.toExponential(3)}
                  </span>
                )}
              </div>
              <button
                className="btn-primary"
                onClick={() => inferMut.mutate()}
                disabled={inferMut.isPending}
              >
                {inferMut.isPending ? <Spinner size="sm" /> : <Zap size={14} />}
                Run Inference
              </button>
            </div>
          )}

          {/* Expanded run list */}
          {expandRuns && (
            <div className="space-y-1.5 mt-2">
              {projectRuns.map((r) => (
                <div
                  key={r.id}
                  className={clsx(
                    "flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-all",
                    r.id === activeRun?.id
                      ? "border-primary/40 bg-primary/5"
                      : "border-border hover:border-border/70 hover:bg-surface2"
                  )}
                  onClick={() => handleSelectRun(r)}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono text-secondary w-20">
                      {r.model_type.toUpperCase()}
                    </span>
                    <span className="text-xs text-muted">
                      {format(new Date(r.created_at), "MMM d HH:mm")}
                    </span>
                    {r.final_loss != null && (
                      <span className="text-xs font-mono text-muted">
                        loss: {r.final_loss.toExponential(3)}
                      </span>
                    )}
                  </div>
                  <StatusBadge status={r.status} />
                </div>
              ))}
            </div>
          )}
        </Card>
      ) : (
        <div className="card p-8 flex flex-col items-center gap-3 text-center">
          <AlertCircle size={32} className="text-muted" />
          <p className="text-muted text-sm">
            No completed runs yet.{" "}
            <button className="text-primary hover:underline" onClick={() => navigate("/training")}>
              Train a model first.
            </button>
          </p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Timeseries models */}
          {isTS && result.forecast && (
            <Card title="Forecast" subtitle={`${modelType.toUpperCase()} · horizon ${result.horizon}`}>
              <ForecastChart
                history={tsData ?? []}
                forecast={result.forecast}
                inputLen={result.input_len}
                title=""
                height={320}
              />
              <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3">
                <Metric label="Horizon" value={String(result.horizon ?? "—")} />
                <Metric label="Input window" value={String(result.input_len ?? "—")} />
                <Metric label="Model" value={modelType.toUpperCase()} />
                <Metric
                  label="Final loss"
                  value={activeRun?.final_loss != null ? activeRun.final_loss.toExponential(3) : "—"}
                />
              </div>
            </Card>
          )}

          {/* LBM */}
          {isLBM && (
            <>
              <Card title="LBM Flow Field" subtitle="Steady-state velocity magnitude">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {result.vel_mag && (
                    <HeatmapChart
                      z={result.vel_mag}
                      title="Velocity |u|"
                      colorscale="Viridis"
                      maskWith={result.obstacle}
                      height={260}
                    />
                  )}
                  {result.vorticity && (
                    <HeatmapChart
                      z={result.vorticity}
                      title="Vorticity ω"
                      colorscale="RdBu"
                      reversescale
                      maskWith={result.obstacle}
                      height={260}
                    />
                  )}
                </div>
              </Card>

              {result.Q && (
                <Card title="Q-Criterion" subtitle="Vortex identification (Q > 0 marks vortex cores)">
                  <HeatmapChart
                    z={result.Q}
                    title="Q-criterion"
                    colorscale="Plasma"
                    maskWith={result.obstacle}
                    height={280}
                  />
                  <div className="mt-2 text-xs text-muted">
                    Q = ½(‖Ω‖² − ‖S‖²) &gt; 0 indicates rotation-dominated regions (vortex cores).
                    Use the Visualization page for interactive thresholding and LBM animation.
                  </div>
                  <button
                    className="mt-3 btn-secondary text-xs"
                    onClick={() => navigate("/visualization")}
                  >
                    <Eye size={12} /> Open CFD Visualization
                  </button>
                </Card>
              )}

              {result.rho && (
                <Card title="Density Field">
                  <HeatmapChart
                    z={result.rho}
                    title="Density ρ"
                    colorscale="Cividis"
                    maskWith={result.obstacle}
                    height={260}
                  />
                </Card>
              )}

              <Card title="Flow Components">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {result.ux && (
                    <HeatmapChart
                      z={result.ux}
                      title="ux (horizontal)"
                      colorscale="RdBu"
                      reversescale
                      maskWith={result.obstacle}
                      height={240}
                    />
                  )}
                  {result.uy && (
                    <HeatmapChart
                      z={result.uy}
                      title="uy (vertical)"
                      colorscale="RdBu"
                      reversescale
                      maskWith={result.obstacle}
                      height={240}
                    />
                  )}
                </div>
              </Card>
            </>
          )}

          {/* PINN */}
          {isPINN && result.u && (
            <Card title="PINN Solution" subtitle="Predicted field u(x,y)">
              <HeatmapChart
                z={result.u}
                x={result.x}
                y={result.y}
                title="u(x, y)"
                colorscale="Viridis"
                height={340}
              />
              {result.history && result.history.length > 0 && (
                <div className="mt-4">
                  <div className="text-xs text-muted mb-2">Training loss curve</div>
                  <LossChart history={result.history} height={200} showComponents />
                </div>
              )}
            </Card>
          )}

          {/* FDM / FEM */}
          {(isFDM || isFEM) && result.field && (
            <>
              <Card title={`${modelType.toUpperCase()} Solution`} subtitle={result.label ?? "Numerical field"}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <HeatmapChart
                    z={result.field}
                    title="Numerical solution"
                    colorscale="Viridis"
                    height={280}
                  />
                  {result.exact && (
                    <HeatmapChart
                      z={result.exact}
                      title="Exact solution"
                      colorscale="Viridis"
                      height={280}
                    />
                  )}
                </div>
                {(result.l2_error != null || result.linf_error != null) && (
                  <div className="mt-4 grid grid-cols-2 gap-3">
                    {result.l2_error != null && (
                      <Metric label="L2 error" value={result.l2_error.toExponential(4)} />
                    )}
                    {result.linf_error != null && (
                      <Metric label="L∞ error" value={result.linf_error.toExponential(4)} />
                    )}
                  </div>
                )}
              </Card>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="card-inner p-3">
      <div className="text-xs text-muted mb-0.5">{label}</div>
      <div className="text-sm font-mono text-text">{value}</div>
    </div>
  );
}
