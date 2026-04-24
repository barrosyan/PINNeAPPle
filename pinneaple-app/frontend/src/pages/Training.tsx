import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useActiveProject } from "@/hooks/useProject";
import { useAppStore } from "@/store";
import { startTraining, getAllRuns } from "@/api/training";
import { useWebSocket } from "@/hooks/useWebSocket";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Spinner";
import LossChart from "@/components/charts/LossChart";
import type { TrainingEntry, WSMessage, TrainingRun } from "@/types";
import { Zap, Square, ArrowRight, AlertCircle, CheckCircle2, RefreshCw } from "lucide-react";
import toast from "react-hot-toast";
import { format } from "date-fns";
import clsx from "clsx";

export default function Training() {
  const navigate                  = useNavigate();
  const qc                        = useQueryClient();
  const { data: project }         = useActiveProject();
  const { activeRunId, activeWsRunId, setActiveRun, tsData } = useAppStore();

  const [history, setHistory]     = useState<TrainingEntry[]>([]);
  const [status,  setStatus]      = useState<string>("idle");
  const [total,   setTotal]       = useState<number>(0);
  const [currentEpoch, setCurrentEpoch] = useState<number>(0);
  const [finalLoss, setFinalLoss] = useState<number | null>(null);
  const [errMsg,  setErrMsg]      = useState<string | null>(null);
  const [showComponents, setShowComponents] = useState(false);
  const { data: allRuns } = useQuery({
    queryKey: ["runs_all"],
    queryFn: getAllRuns,
    refetchInterval: (q) => {
      const r = q.state.data as TrainingRun[] | undefined;
      return r?.some((x) => x.status === "running") ? 3000 : false;
    },
  });

  const projectRuns = allRuns?.filter((r) => r.project === project?.id) ?? [];

  // Init from active run on first mount
  useEffect(() => {
    if (!activeRunId || history.length > 0) return;
    const run = allRuns?.find((r) => r.id === activeRunId);
    if (run) {
      setHistory(run.history ?? []);
      setStatus(run.status);
      setFinalLoss(run.final_loss);
      if (run.history?.length) {
        setCurrentEpoch(run.history[run.history.length - 1].epoch);
      }
    }
  }, [activeRunId, allRuns]); // eslint-disable-line react-hooks/exhaustive-deps

  const startMut = useMutation({
    mutationFn: () =>
      startTraining(project!.id, project!.model_config!, tsData ?? undefined),
    onSuccess: (data) => {
      setActiveRun(data.db_run_id, data.ws_run_id);
      setHistory([]);
      setStatus("pending");
      setFinalLoss(null);
      setErrMsg(null);
      setCurrentEpoch(0);
      setTotal(0);
      toast.success("Training started!");
      qc.invalidateQueries({ queryKey: ["runs_all"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const { connected, send } = useWebSocket(activeWsRunId, {
    onMessage: (msg: WSMessage) => {
      setStatus(msg.status);

      if (msg.type === "init") {
        if (msg.history && msg.history.length > 0) {
          setHistory(msg.history);
          const last = msg.history[msg.history.length - 1];
          setCurrentEpoch(last.epoch);
        }
        if (msg.total)       setTotal(msg.total);
        if (msg.final_loss != null) setFinalLoss(msg.final_loss);
      } else if (msg.type === "progress") {
        if (msg.total != null) setTotal(msg.total);
        if (msg.epoch != null) {
          setCurrentEpoch(msg.epoch);
          setHistory((h) => [
            ...h,
            { epoch: msg.epoch!, loss: msg.loss ?? 0, pde: msg.pde, bc: msg.bc },
          ]);
        }
      } else if (msg.type === "done") {
        if (msg.final_loss != null) setFinalLoss(msg.final_loss);
        if (msg.history)            setHistory(msg.history);
        if (msg.total != null)      setCurrentEpoch(msg.total);
        qc.invalidateQueries({ queryKey: ["runs_all"] });
        toast.success("Training complete!");
      } else if (msg.type === "error") {
        setErrMsg(msg.msg ?? "Training failed");
        toast.error(msg.msg ?? "Training failed");
      } else if (msg.type === "stopped") {
        toast("Training stopped", { icon: "⏹" });
        qc.invalidateQueries({ queryKey: ["runs_all"] });
      }
    },
  });

  const handleStop = () => {
    send({ type: "stop" });
  };

  const handleSelectRun = (run: TrainingRun) => {
    setActiveRun(run.id, run.ws_run_id);
    setHistory(run.history ?? []);
    setStatus(run.status);
    setFinalLoss(run.final_loss);
    setErrMsg(null);
    if (run.history?.length) {
      setCurrentEpoch(run.history[run.history.length - 1].epoch);
    }
  };

  const isRunning  = status === "running" || status === "pending";
  const isDone     = status === "done";
  const isError    = status === "error";
  const isStopped  = status === "stopped";
  const progress   = total > 0 ? Math.min(100, Math.round((currentEpoch / total) * 100)) : 0;
  const canStart   = !!project?.model_config && !isRunning;

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text">Training</h1>
          <p className="text-muted text-sm mt-1">
            Launch and monitor solver/model training in real time.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {connected && (
            <span className="flex items-center gap-1.5 text-xs text-success">
              <span className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
              Live
            </span>
          )}
          {activeRunId && isDone && (
            <button className="btn-secondary" onClick={() => navigate("/inference")}>
              <ArrowRight size={14} /> Inference
            </button>
          )}
        </div>
      </div>

      {/* Project + model config summary */}
      {project ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="card p-4">
            <div className="text-xs text-muted mb-1">Project</div>
            <div className="text-sm font-semibold text-text truncate">{project.name}</div>
            <div className="text-xs text-muted mt-0.5 truncate">
              {project.problem_spec?.category ?? "—"}
            </div>
          </div>
          <div className="card p-4">
            <div className="text-xs text-muted mb-1">Model</div>
            {project.model_config ? (
              <>
                <div className="text-sm font-semibold text-text">
                  {project.model_config.name ?? project.model_config.type.toUpperCase()}
                </div>
                <div className="text-xs text-muted font-mono mt-0.5">
                  {project.model_config.type}
                </div>
              </>
            ) : (
              <div className="flex items-center gap-2">
                <AlertCircle size={14} className="text-warning" />
                <span className="text-xs text-warning">No model selected</span>
                <button
                  className="btn-ghost text-xs text-primary ml-1"
                  onClick={() => navigate("/models")}
                >
                  Select →
                </button>
              </div>
            )}
          </div>
          <div className="card p-4">
            <div className="text-xs text-muted mb-1">Status</div>
            <div className="flex items-center gap-2">
              <StatusBadge status={isRunning ? "running" : isDone ? "done" : isError ? "error" : isStopped ? "stopped" : "created"} />
              {isRunning && total > 0 && (
                <span className="text-xs text-muted font-mono">
                  {currentEpoch}/{total}
                </span>
              )}
            </div>
            {finalLoss != null && (
              <div className="text-xs font-mono text-secondary mt-1">
                loss: {finalLoss.toExponential(4)}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="card p-8 flex flex-col items-center gap-3 text-center">
          <AlertCircle size={32} className="text-muted" />
          <p className="text-muted text-sm">No active project. Create one first.</p>
          <button className="btn-primary" onClick={() => navigate("/problem")}>
            Create project
          </button>
        </div>
      )}

      {/* Start / Stop controls */}
      {project?.model_config && (
        <div className="flex items-center gap-3">
          <button
            className="btn-primary"
            onClick={() => startMut.mutate()}
            disabled={!canStart || startMut.isPending}
          >
            {startMut.isPending ? <Spinner size="sm" /> : <Zap size={14} />}
            {isRunning ? "Training…" : "Start Training"}
          </button>
          {isRunning && (
            <button className="btn-secondary" onClick={handleStop}>
              <Square size={14} /> Stop
            </button>
          )}
          <label className="flex items-center gap-2 text-xs text-muted cursor-pointer ml-auto">
            <input
              type="checkbox"
              className="accent-primary"
              checked={showComponents}
              onChange={(e) => setShowComponents(e.target.checked)}
            />
            Show PDE/BC components
          </label>
        </div>
      )}

      {/* Progress bar */}
      {isRunning && total > 0 && (
        <div>
          <div className="flex justify-between text-xs text-muted mb-1">
            <span>Epoch {currentEpoch} / {total}</span>
            <span>{progress}%</span>
          </div>
          <div className="h-2 bg-surface2 rounded-full overflow-hidden border border-border">
            <div
              className="h-full bg-primary rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error banner */}
      {isError && errMsg && (
        <div className="flex items-center gap-3 p-4 rounded-xl border border-error/40 bg-error/5">
          <AlertCircle size={18} className="text-error shrink-0" />
          <div>
            <div className="text-sm font-medium text-error">Training Error</div>
            <div className="text-xs text-muted mt-0.5 font-mono">{errMsg}</div>
          </div>
        </div>
      )}

      {/* Done banner */}
      {isDone && finalLoss != null && (
        <div className="flex items-center justify-between p-4 rounded-xl border border-success/40 bg-success/5">
          <div className="flex items-center gap-3">
            <CheckCircle2 size={18} className="text-success shrink-0" />
            <div>
              <div className="text-sm font-medium text-success">Training Complete</div>
              <div className="text-xs text-muted font-mono mt-0.5">
                Final loss: {finalLoss.toExponential(4)}
              </div>
            </div>
          </div>
          <button className="btn-primary" onClick={() => navigate("/inference")}>
            <ArrowRight size={14} /> Run Inference
          </button>
        </div>
      )}

      {/* Loss chart */}
      {history.length > 0 && (
        <Card
          title="Training Loss"
          subtitle={`${history.length} data points`}
          action={
            <button
              className="btn-ghost text-xs"
              onClick={() => setHistory([])}
            >
              <RefreshCw size={12} /> Clear
            </button>
          }
        >
          <LossChart history={history} height={320} showComponents={showComponents} />
          {history.length > 0 && (
            <div className="mt-3 grid grid-cols-3 gap-3">
              <div className="card-inner p-3">
                <div className="text-xs text-muted">Initial loss</div>
                <div className="text-sm font-mono text-text mt-0.5">
                  {history[0].loss.toExponential(3)}
                </div>
              </div>
              <div className="card-inner p-3">
                <div className="text-xs text-muted">Current loss</div>
                <div className="text-sm font-mono text-secondary mt-0.5">
                  {history[history.length - 1].loss.toExponential(3)}
                </div>
              </div>
              <div className="card-inner p-3">
                <div className="text-xs text-muted">Reduction</div>
                <div className="text-sm font-mono text-success mt-0.5">
                  {history.length > 1
                    ? `${((1 - history[history.length - 1].loss / history[0].loss) * 100).toFixed(1)}%`
                    : "—"}
                </div>
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Run history */}
      {projectRuns.length > 0 && (
        <Card title="Run History" subtitle={`${projectRuns.length} runs for this project`}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left border-b border-border">
                  <th className="pb-2 text-xs text-muted font-medium">Model</th>
                  <th className="pb-2 text-xs text-muted font-medium">Status</th>
                  <th className="pb-2 text-xs text-muted font-medium">Final Loss</th>
                  <th className="pb-2 text-xs text-muted font-medium">Steps</th>
                  <th className="pb-2 text-xs text-muted font-medium">Started</th>
                  <th className="pb-2 text-xs text-muted font-medium"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {projectRuns.map((r) => (
                  <tr
                    key={r.id}
                    className={clsx(
                      "hover:bg-surface2/50 transition-colors cursor-pointer",
                      r.id === activeRunId && "bg-primary/5"
                    )}
                    onClick={() => handleSelectRun(r)}
                  >
                    <td className="py-2.5 font-mono text-xs text-secondary">
                      {r.model_type.toUpperCase()}
                    </td>
                    <td className="py-2.5">
                      <StatusBadge status={r.status} />
                    </td>
                    <td className="py-2.5 font-mono text-xs text-muted">
                      {r.final_loss != null ? r.final_loss.toExponential(3) : "—"}
                    </td>
                    <td className="py-2.5 text-xs text-muted">
                      {r.history?.length ?? 0}
                    </td>
                    <td className="py-2.5 text-xs text-muted">
                      {format(new Date(r.created_at), "MMM d HH:mm")}
                    </td>
                    <td className="py-2.5 text-right">
                      {r.status === "done" && (
                        <button
                          className="btn-ghost text-xs py-1 text-primary"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSelectRun(r);
                            navigate("/inference");
                          }}
                        >
                          Infer →
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
