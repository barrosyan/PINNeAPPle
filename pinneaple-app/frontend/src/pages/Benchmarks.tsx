import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  getBenchmarks, runBenchmark, getBenchmarkResults, clearBenchmarkResults,
} from "@/api/benchmarks";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { NumberInput } from "@/components/ui/Input";
import { Spinner } from "@/components/ui/Spinner";
import HeatmapChart from "@/components/charts/HeatmapChart";
import LossChart from "@/components/charts/LossChart";
import type { BenchmarkDef, BenchmarkResult } from "@/types";
import { Trophy, Play, Trash2, Download, CheckCircle2, AlertCircle } from "lucide-react";
import toast from "react-hot-toast";
import { format } from "date-fns";
import clsx from "clsx";

// Built-in benchmark catalogue (mirrors backend definitions)
const BENCHMARK_CATALOGUE: BenchmarkDef[] = [
  {
    key: "poisson_fdm",
    name: "Poisson FDM",
    category: "Classical PDE",
    description: "5-point Gauss-Seidel on −∇²u = f with Dirichlet BCs. Compares numerical solution to analytical sin²(πx)sin²(πy).",
    params: { nx: 64, ny: 64, iters: 5000 },
  },
  {
    key: "burgers_pinn",
    name: "Burgers PINN",
    category: "PINN",
    description: "PINN for 1D viscous Burgers equation: uₜ + uuₓ = νuₓₓ. Trains for 2000 epochs and reports PDE + IC residuals.",
    params: { n_epochs: 2000, lr: 1e-3, hidden: 32, n_layers: 4 },
  },
  {
    key: "cylinder_lbm",
    name: "Cylinder LBM",
    category: "CFD",
    description: "D2Q9 LBM flow past a cylinder at Re=200. Reports Strouhal number, drag coefficient, and Q-criterion vortex count.",
    params: { nx: 160, ny: 64, Re: 200, steps: 8000 },
  },
  {
    key: "cavity_lbm",
    name: "Lid-Driven Cavity",
    category: "CFD",
    description: "LBM lid-driven cavity at Re=400. Reports vorticity extrema and velocity profile error vs. Ghia 1982.",
    params: { nx: 64, ny: 64, Re: 400, steps: 20000 },
  },
];

type ParamValue = number | boolean | string;

export default function Benchmarks() {
  const [selectedKey, setSelectedKey]   = useState<string | null>(null);
  const [params, setParams]             = useState<Record<string, ParamValue>>({});
  const [lastResult, setLastResult]     = useState<(BenchmarkResult & { metrics: Record<string, unknown> }) | null>(null);

  const { data: results, refetch: refetchResults } = useQuery({
    queryKey: ["benchmark_results"],
    queryFn:  getBenchmarkResults,
  });

  const runMut = useMutation({
    mutationFn: () => runBenchmark(selectedKey!, params),
    onSuccess:  (r) => {
      setLastResult(r);
      refetchResults();
      toast.success(`Benchmark "${selectedKey}" complete`);
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const clearMut = useMutation({
    mutationFn: clearBenchmarkResults,
    onSuccess:  () => { refetchResults(); toast.success("Results cleared"); },
    onError:    (e: Error) => toast.error(e.message),
  });

  const selectedDef = BENCHMARK_CATALOGUE.find((b) => b.key === selectedKey);

  const handleSelect = (def: BenchmarkDef) => {
    setSelectedKey(def.key);
    setParams({ ...def.params } as Record<string, ParamValue>);
    setLastResult(null);
  };

  const exportCsv = () => {
    if (!results?.length) return;
    const rows = results.map((r) => ({
      benchmark_key: r.benchmark_key,
      created_at: r.created_at,
      ...r.metrics,
    }));
    const headers = Array.from(new Set(rows.flatMap((r) => Object.keys(r))));
    const csv = [
      headers.join(","),
      ...rows.map((r) => headers.map((h) => JSON.stringify((r as Record<string, unknown>)[h] ?? "")).join(",")),
    ].join("\n");
    const a = document.createElement("a");
    a.href  = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "benchmark_results.csv";
    a.click();
  };

  const CATEGORIES = Array.from(new Set(BENCHMARK_CATALOGUE.map((b) => b.category)));
  const [catFilter, setCatFilter] = useState("All");
  const filtered = catFilter === "All"
    ? BENCHMARK_CATALOGUE
    : BENCHMARK_CATALOGUE.filter((b) => b.category === catFilter);

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text">Benchmarks</h1>
          <p className="text-muted text-sm mt-1">
            Validate solver accuracy with reference problems and compare against analytical solutions.
          </p>
        </div>
        <div className="flex gap-2">
          {results && results.length > 0 && (
            <>
              <button className="btn-secondary" onClick={exportCsv}>
                <Download size={14} /> Export CSV
              </button>
              <button
                className="btn-ghost text-error"
                onClick={() => { if (confirm("Clear all benchmark results?")) clearMut.mutate(); }}
                disabled={clearMut.isPending}
              >
                <Trash2 size={14} />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Category filter */}
      <div className="flex flex-wrap gap-2">
        {["All", ...CATEGORIES].map((c) => (
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

      {/* Benchmark cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filtered.map((b) => (
          <div
            key={b.key}
            className={clsx(
              "card p-4 cursor-pointer transition-all",
              selectedKey === b.key
                ? "border-primary/50 shadow-glow"
                : "hover:border-border/70"
            )}
            onClick={() => handleSelect(b)}
          >
            <div className="flex items-start justify-between gap-2 mb-2">
              <div className="flex items-center gap-2">
                <Trophy size={15} className="text-success shrink-0" />
                <span className="text-sm font-semibold text-text">{b.name}</span>
              </div>
              <Badge variant="secondary" className="text-xs">{b.category}</Badge>
            </div>
            <p className="text-xs text-muted line-clamp-2 mb-3">{b.description}</p>
            <div className="flex flex-wrap gap-1">
              {Object.entries(b.params).map(([k, v]) => (
                <span
                  key={k}
                  className="text-xs font-mono px-2 py-0.5 rounded bg-surface2 text-muted border border-border"
                >
                  {k}={String(v)}
                </span>
              ))}
            </div>
            <button
              className={clsx(
                "mt-3 w-full text-xs py-1.5 rounded-lg border transition-all",
                selectedKey === b.key
                  ? "bg-primary text-white border-primary"
                  : "border-border text-muted hover:border-primary/50 hover:text-text"
              )}
            >
              {selectedKey === b.key ? "Selected" : "Configure & Run"}
            </button>
          </div>
        ))}
      </div>

      {/* Config + Run */}
      {selectedDef && (
        <Card
          title={`Run: ${selectedDef.name}`}
          subtitle={selectedDef.category}
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            {Object.entries(params).map(([k, v]) => {
              if (typeof v === "boolean") {
                return (
                  <label key={k} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={v}
                      onChange={(e) => setParams((p) => ({ ...p, [k]: e.target.checked }))}
                      className="accent-primary"
                    />
                    <span className="text-sm text-muted">{k}</span>
                  </label>
                );
              }
              return (
                <NumberInput
                  key={k}
                  label={k}
                  value={v as number}
                  onChange={(val) => setParams((p) => ({ ...p, [k]: val }))}
                  format={Number.isInteger(v) ? "int" : "float"}
                  step={Number.isInteger(v) ? 1 : 0.0001}
                />
              );
            })}
          </div>

          <button
            className="btn-primary"
            onClick={() => runMut.mutate()}
            disabled={runMut.isPending}
          >
            {runMut.isPending ? <Spinner size="sm" /> : <Play size={14} />}
            {runMut.isPending ? "Running…" : "Run Benchmark"}
          </button>
          {runMut.isPending && (
            <p className="text-xs text-muted mt-2 animate-pulse">
              This may take a few seconds to several minutes depending on the solver…
            </p>
          )}
        </Card>
      )}

      {/* Last result */}
      {lastResult && (
        <BenchmarkResultCard result={lastResult} />
      )}

      {/* History table */}
      {results && results.length > 0 && (
        <Card
          title="Result History"
          subtitle={`${results.length} benchmark${results.length !== 1 ? "s" : ""} run`}
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left border-b border-border">
                  <th className="pb-2 text-xs text-muted font-medium">Benchmark</th>
                  <th className="pb-2 text-xs text-muted font-medium">Metrics</th>
                  <th className="pb-2 text-xs text-muted font-medium">When</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {results.slice().reverse().map((r) => (
                  <tr key={r.id} className="hover:bg-surface2/50 transition-colors">
                    <td className="py-2.5 font-mono text-xs text-secondary">
                      {r.benchmark_key}
                    </td>
                    <td className="py-2.5">
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(r.metrics).slice(0, 4).map(([k, v]) =>
                          typeof v === "number" ? (
                            <span
                              key={k}
                              className="text-xs font-mono px-1.5 py-0.5 rounded bg-surface2 border border-border text-muted"
                            >
                              {k}={v < 1 ? v.toExponential(2) : Number(v).toFixed(3)}
                            </span>
                          ) : null
                        )}
                      </div>
                    </td>
                    <td className="py-2.5 text-xs text-muted">
                      {format(new Date(r.created_at), "MMM d HH:mm")}
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

function BenchmarkResultCard({
  result,
}: {
  result: BenchmarkResult & { metrics: Record<string, unknown> };
}) {
  const m = result.metrics;
  const isSuccess = !m.error;

  return (
    <Card
      title={`Result: ${result.benchmark_key}`}
      subtitle={format(new Date(result.created_at), "PPpp")}
      action={
        isSuccess ? (
          <CheckCircle2 size={16} className="text-success" />
        ) : (
          <AlertCircle size={16} className="text-error" />
        )
      }
    >
      {m.error ? (
        <div className="flex items-center gap-2 text-error text-sm">
          <AlertCircle size={16} />
          {String(m.error)}
        </div>
      ) : (
        <div className="space-y-4">
          {/* Numeric metrics grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(m)
              .filter(([, v]) => typeof v === "number")
              .map(([k, v]) => (
                <div key={k} className="card-inner p-3">
                  <div className="text-xs text-muted mb-0.5">{k}</div>
                  <div className="text-sm font-mono text-text">
                    {(v as number) < 1 && (v as number) !== 0
                      ? (v as number).toExponential(4)
                      : Number(v).toFixed(4)}
                  </div>
                </div>
              ))}
          </div>

          {/* Heatmap fields */}
          {m.field && Array.isArray(m.field) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <HeatmapChart
                z={m.field as number[][]}
                title="Numerical"
                colorscale="Viridis"
                height={260}
              />
              {m.exact && Array.isArray(m.exact) && (
                <HeatmapChart
                  z={m.exact as number[][]}
                  title="Analytical"
                  colorscale="Viridis"
                  height={260}
                />
              )}
            </div>
          )}

          {/* Velocity/vorticity for CFD benchmarks */}
          {m.vel_mag && Array.isArray(m.vel_mag) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <HeatmapChart
                z={m.vel_mag as number[][]}
                title="Velocity |u|"
                colorscale="Viridis"
                height={240}
              />
              {m.Q && Array.isArray(m.Q) && (
                <HeatmapChart
                  z={m.Q as number[][]}
                  title="Q-Criterion"
                  colorscale="Plasma"
                  height={240}
                />
              )}
            </div>
          )}

          {/* Training loss for PINN benchmarks */}
          {m.history && Array.isArray(m.history) && (m.history as unknown[]).length > 0 && (
            <div>
              <div className="text-xs text-muted mb-2">Training loss</div>
              <LossChart
                history={m.history as { epoch: number; loss: number }[]}
                height={220}
                showComponents
              />
            </div>
          )}

          {/* String / info metrics */}
          {Object.entries(m)
            .filter(
              ([, v]) =>
                typeof v === "string" ||
                (typeof v === "number" && !isFinite(v as number))
            )
            .map(([k, v]) => (
              <div key={k} className="text-xs text-muted">
                <span className="text-secondary font-mono">{k}</span>: {String(v)}
              </div>
            ))}
        </div>
      )}
    </Card>
  );
}
