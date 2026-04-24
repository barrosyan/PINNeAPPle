import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { useActiveProject, useUpdateProject } from "@/hooks/useProject";
import { suggestModels } from "@/api/problems";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { NumberInput } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Slider } from "@/components/ui/Slider";
import { Spinner } from "@/components/ui/Spinner";
import type { ModelConfig, ModelCatalogEntry } from "@/types";
import { Cpu, Zap, Check, ChevronDown, ChevronUp } from "lucide-react";
import toast from "react-hot-toast";
import clsx from "clsx";

const CATALOGUE: ModelCatalogEntry[] = [
  {
    type: "pinn_mlp", name: "PINN (MLP)", category: "PINN",
    description: "Physics-Informed Neural Network. Minimises PDE residual + boundary condition loss via automatic differentiation.",
    params: {
      n_epochs:   { label: "Epochs",            type: "int",   default: 500,   min: 50,   max: 10000 },
      lr:         { label: "Learning rate",      type: "float", default: 1e-3,  min: 1e-6, max: 0.1,  step: 0.0001 },
      hidden:     { label: "Hidden width",       type: "int",   default: 64,    min: 8,    max: 512   },
      n_layers:   { label: "Depth (layers)",     type: "int",   default: 4,     min: 2,    max: 12    },
      n_interior: { label: "Colloc. points",     type: "int",   default: 2000,  min: 200,  max: 20000 },
    },
  },
  {
    type: "lbm", name: "LBM Solver", category: "Classical CFD",
    description: "Lattice Boltzmann Method — D2Q9 BGK with Zou-He velocity inlet / pressure outlet BCs, bounce-back obstacles, and optional Smagorinsky LES.",
    params: {
      steps:      { label: "Timesteps",       type: "int",   default: 4000,  min: 100,  max: 50000 },
      save_every: { label: "Save every N",    type: "int",   default: 500,   min: 10,   max: 5000  },
      nx:         { label: "Grid nx",         type: "int",   default: 160,   min: 32,   max: 1024  },
      ny:         { label: "Grid ny",         type: "int",   default: 64,    min: 16,   max: 512   },
      Re:         { label: "Reynolds number", type: "float", default: 200,   min: 1,    max: 5000, step: 10 },
      u_in:       { label: "Inlet velocity",  type: "float", default: 0.05,  min: 0.001,max: 0.3,  step: 0.005 },
    },
  },
  {
    type: "fdm", name: "FDM Solver", category: "Classical PDE",
    description: "Finite Difference Method — 5-point Gauss-Seidel solver for the Poisson equation on a structured grid.",
    params: {
      nx:    { label: "Grid nx",    type: "int", default: 64, min: 8,  max: 512 },
      ny:    { label: "Grid ny",    type: "int", default: 64, min: 8,  max: 512 },
      iters: { label: "Iterations", type: "int", default: 5000, min: 100, max: 50000 },
    },
  },
  {
    type: "fem", name: "FEM Solver", category: "Classical PDE",
    description: "Finite Element Method — 2D linear triangular elements for elliptic PDEs.",
    params: {
      nx: { label: "Elements x", type: "int", default: 20, min: 5, max: 200 },
      ny: { label: "Elements y", type: "int", default: 20, min: 5, max: 200 },
    },
  },
  {
    type: "tcn", name: "TCN Forecaster", category: "Timeseries",
    description: "Temporal Convolutional Network — dilated causal convolutions for efficient sequence modelling.",
    params: {
      input_len: { label: "Input window",  type: "int",   default: 32,   min: 8,  max: 512 },
      horizon:   { label: "Forecast horizon", type: "int", default: 16,  min: 1,  max: 256 },
      epochs:    { label: "Epochs",         type: "int",   default: 50,   min: 5,  max: 500 },
      lr:        { label: "Learning rate",  type: "float", default: 1e-3, min: 1e-6, max: 0.1, step: 0.0001 },
    },
  },
  {
    type: "lstm", name: "LSTM Forecaster", category: "Timeseries",
    description: "Long Short-Term Memory recurrent network with 2 stacked layers.",
    params: {
      input_len: { label: "Input window",  type: "int",   default: 32,  min: 8,  max: 512 },
      horizon:   { label: "Horizon",       type: "int",   default: 16,  min: 1,  max: 256 },
      epochs:    { label: "Epochs",        type: "int",   default: 50,  min: 5,  max: 500 },
      lr:        { label: "Learning rate", type: "float", default: 1e-3, min: 1e-6, max: 0.1, step: 0.0001 },
    },
  },
  {
    type: "tft", name: "TFT Forecaster", category: "Timeseries",
    description: "Temporal Fusion Transformer — interpretable attention-based model with variable selection networks.",
    params: {
      input_len: { label: "Input window",  type: "int",   default: 32,  min: 8,  max: 512 },
      horizon:   { label: "Horizon",       type: "int",   default: 16,  min: 1,  max: 256 },
      epochs:    { label: "Epochs",        type: "int",   default: 30,  min: 5,  max: 200 },
      lr:        { label: "Learning rate", type: "float", default: 5e-4, min: 1e-6, max: 0.05, step: 0.0001 },
    },
  },
  {
    type: "fft", name: "FFT Forecaster", category: "Decomposition",
    description: "Harmonic decomposition via FFT — no training required. Fits top-K harmonics and extrapolates forward.",
    params: {
      n_harmonics: { label: "Harmonics",    type: "int",  default: 5,    min: 1, max: 50 },
      horizon:     { label: "Horizon",      type: "int",  default: 50,   min: 1, max: 500 },
      detrend:     { label: "Linear detrend", type: "bool", default: true },
    },
  },
];

const CATEGORIES = ["All", "PINN", "Classical CFD", "Classical PDE", "Timeseries", "Decomposition"];

export default function Models() {
  const navigate               = useNavigate();
  const { data: project }      = useActiveProject();
  const updateProject          = useUpdateProject(project?.id ?? "");
  const [catFilter, setCatFilter] = useState("All");
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [config, setConfig]    = useState<Record<string, number | boolean>>({});
  const [obsType, setObsType]  = useState("none");
  const [obsParams, setObsParams] = useState({ cx: 40, cy: 32, r: 8, x0: 20, x1: 30, y0: 20, y1: 44 });
  const [expanded, setExpanded] = useState<string | null>(null);

  const suggestMut = useMutation({
    mutationFn: () => suggestModels(project!.problem_spec),
  });

  const selectedEntry = CATALOGUE.find((m) => m.type === selectedType);

  const handleSelect = (type: string) => {
    setSelectedType(type);
    const entry = CATALOGUE.find((m) => m.type === type);
    if (entry) {
      const defaults: Record<string, number | boolean> = {};
      Object.entries(entry.params).forEach(([k, v]) => {
        defaults[k] = v.default as number | boolean;
      });
      setConfig(defaults);
    }
  };

  const handleSave = () => {
    if (!project || !selectedType) { toast.error("No model selected"); return; }
    const entry = CATALOGUE.find((m) => m.type === selectedType);
    const cfg: ModelConfig = { type: selectedType, name: entry?.name, ...config };
    if (selectedType === "lbm" && obsType !== "none") {
      cfg.obstacle = obsType === "cylinder"
        ? { type: "cylinder", cx: obsParams.cx, cy: obsParams.cy, r: obsParams.r }
        : { type: "rectangle", x0: obsParams.x0, x1: obsParams.x1, y0: obsParams.y0, y1: obsParams.y1 };
    }
    updateProject.mutate(
      { model_config: cfg },
      {
        onSuccess: () => { toast.success("Model configuration saved"); navigate("/training"); },
        onError:   (e: Error) => toast.error(e.message),
      }
    );
  };

  const filtered = catFilter === "All"
    ? CATALOGUE
    : CATALOGUE.filter((m) => m.category === catFilter);

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text">Models</h1>
          <p className="text-muted text-sm mt-1">Select a solver or ML model and configure hyperparameters.</p>
        </div>
        {project && (
          <button
            className="btn-secondary"
            onClick={() => suggestMut.mutate()}
            disabled={suggestMut.isPending}
          >
            {suggestMut.isPending ? <Spinner size="sm" /> : <Cpu size={14} />}
            AI Suggestions
          </button>
        )}
      </div>

      {/* AI suggestions */}
      {suggestMut.data && (
        <Card title="Recommended for this problem">
          <div className="flex flex-wrap gap-3">
            {suggestMut.data.map((s) => (
              <button
                key={s.type}
                className={clsx(
                  "flex-1 min-w-[160px] p-3 rounded-lg border text-left transition-all",
                  selectedType === s.type
                    ? "border-primary/50 bg-primary/5"
                    : "border-border hover:border-primary/30"
                )}
                onClick={() => handleSelect(s.type)}
              >
                <div className="text-sm font-semibold text-text">{s.model}</div>
                <div className="text-xs text-muted mt-0.5">{s.reason}</div>
                <div className="mt-1">
                  <Badge variant={s.score >= 85 ? "success" : "warning"}>
                    score {s.score}
                  </Badge>
                </div>
              </button>
            ))}
          </div>
        </Card>
      )}

      {/* Category filter */}
      <div className="flex flex-wrap gap-2">
        {CATEGORIES.map((c) => (
          <button
            key={c}
            onClick={() => setCatFilter(c)}
            className={clsx(
              "px-3 py-1.5 rounded-full text-xs font-medium border transition-all",
              catFilter === c
                ? "bg-primary/15 text-primary border-primary/40"
                : "border-border text-muted hover:text-text hover:border-border/70"
            )}
          >
            {c}
          </button>
        ))}
      </div>

      {/* Catalogue grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {filtered.map((m) => {
          const isSelected = selectedType === m.type;
          const isSaved    = project?.model_config?.type === m.type;
          return (
            <div
              key={m.type}
              className={clsx(
                "card p-4 cursor-pointer transition-all",
                isSelected ? "border-primary/50 shadow-glow" : "hover:border-border/70",
                isSaved    && !isSelected && "border-success/30"
              )}
              onClick={() => handleSelect(m.type)}
            >
              <div className="flex items-start justify-between gap-2 mb-2">
                <span className="text-sm font-semibold text-text">{m.name}</span>
                {isSaved && <Check size={14} className="text-success shrink-0" />}
              </div>
              <Badge variant="secondary" className="mb-2 text-xs">{m.category}</Badge>
              <p className="text-xs text-muted line-clamp-3">{m.description}</p>
              <button
                className={clsx(
                  "mt-3 w-full text-xs py-1.5 rounded-lg border transition-all",
                  isSelected
                    ? "bg-primary text-white border-primary"
                    : "border-border text-muted hover:border-primary/50 hover:text-text"
                )}
              >
                {isSelected ? "Selected" : "Configure"}
              </button>
            </div>
          );
        })}
      </div>

      {/* Config panel */}
      {selectedEntry && (
        <Card
          title={`Configure: ${selectedEntry.name}`}
          subtitle={selectedEntry.category}
        >
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {Object.entries(selectedEntry.params).map(([k, p]) => {
              const val = config[k] ?? p.default;
              if (p.type === "bool") {
                return (
                  <label key={k} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={!!val}
                      onChange={(e) => setConfig((c) => ({ ...c, [k]: e.target.checked }))}
                      className="accent-primary"
                    />
                    <span className="text-sm text-muted">{p.label}</span>
                  </label>
                );
              }
              return (
                <NumberInput
                  key={k}
                  label={p.label}
                  value={val as number}
                  onChange={(v) => setConfig((c) => ({ ...c, [k]: v }))}
                  min={p.min}
                  max={p.max}
                  step={p.step ?? (p.type === "float" ? 0.0001 : 1)}
                  format={p.type as "int" | "float"}
                />
              );
            })}
          </div>

          {/* LBM obstacle config */}
          {selectedType === "lbm" && (
            <div className="mt-6 border-t border-border pt-4">
              <div className="text-sm font-medium text-text mb-3">Obstacle (optional)</div>
              <div className="flex items-end gap-3 flex-wrap">
                <Select
                  label="Type"
                  value={obsType}
                  onChange={(e) => setObsType(e.target.value)}
                  options={[
                    { value: "none",      label: "None"      },
                    { value: "cylinder",  label: "Cylinder"  },
                    { value: "rectangle", label: "Rectangle" },
                  ]}
                  className="w-36"
                />
                {obsType === "cylinder" && (
                  <>
                    <NumberInput label="cx" value={obsParams.cx} onChange={(v) => setObsParams((p) => ({ ...p, cx: v }))} format="int" />
                    <NumberInput label="cy" value={obsParams.cy} onChange={(v) => setObsParams((p) => ({ ...p, cy: v }))} format="int" />
                    <NumberInput label="r"  value={obsParams.r}  onChange={(v) => setObsParams((p) => ({ ...p, r: v  }))} format="int" />
                  </>
                )}
                {obsType === "rectangle" && (
                  <>
                    <NumberInput label="x0" value={obsParams.x0} onChange={(v) => setObsParams((p) => ({ ...p, x0: v }))} format="int" />
                    <NumberInput label="x1" value={obsParams.x1} onChange={(v) => setObsParams((p) => ({ ...p, x1: v }))} format="int" />
                    <NumberInput label="y0" value={obsParams.y0} onChange={(v) => setObsParams((p) => ({ ...p, y0: v }))} format="int" />
                    <NumberInput label="y1" value={obsParams.y1} onChange={(v) => setObsParams((p) => ({ ...p, y1: v }))} format="int" />
                  </>
                )}
              </div>
            </div>
          )}

          <div className="mt-6 flex gap-2">
            <button
              className="btn-primary"
              onClick={handleSave}
              disabled={updateProject.isPending}
            >
              {updateProject.isPending ? <Spinner size="sm" /> : <Zap size={14} />}
              Save & go to Training
            </button>
          </div>
        </Card>
      )}
    </div>
  );
}
