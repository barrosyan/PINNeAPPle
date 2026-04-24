import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useAppStore } from "@/store";
import { getProblems, getCategories } from "@/api/problems";
import { Badge } from "@/components/ui/Badge";
import {
  Zap, FlaskConical, Cpu, BarChart2, Trophy, ArrowRight,
  BookOpen, Activity, Layers, Brain, Globe, Lock,
} from "lucide-react";
import clsx from "clsx";

const FEATURES = [
  {
    icon: Brain,
    color: "text-primary",
    bg: "bg-primary/10 border-primary/20",
    title: "Physics-Informed Neural Networks",
    desc: "PINN (MLP) minimises PDE residuals + boundary-condition loss via automatic differentiation. Works for any second-order PDE.",
  },
  {
    icon: Activity,
    color: "text-secondary",
    bg: "bg-secondary/10 border-secondary/20",
    title: "Lattice Boltzmann Method",
    desc: "D2Q9 BGK solver with Zou-He BCs, bounce-back obstacles, and optional Smagorinsky LES turbulence. Full trajectory animation.",
  },
  {
    icon: Layers,
    color: "text-accent",
    bg: "bg-accent/10 border-accent/20",
    title: "Classical PDE Solvers",
    desc: "FDM (5-point Gauss-Seidel Poisson) and FEM (2D linear triangular) with comparison against exact solutions.",
  },
  {
    icon: BarChart2,
    color: "text-success",
    bg: "bg-success/10 border-success/20",
    title: "Timeseries Forecasting",
    desc: "TCN, LSTM, and Temporal Fusion Transformer for sequence modelling. FFT harmonic decomposition with no training required.",
  },
];

const WORKFLOW = [
  { n: "01", label: "Define the problem",   desc: "Use AI-assisted formulation or pick from the library of 9 reference problems." },
  { n: "02", label: "Prepare geometry",     desc: "Upload STL meshes, CSV data or NumPy arrays. Generate collocation points visually." },
  { n: "03", label: "Configure the model",  desc: "Pick a solver or ML model and tune hyperparameters via sliders and inputs." },
  { n: "04", label: "Train in real time",   desc: "WebSocket live loss curves, epoch progress bar, stop button — always in control." },
  { n: "05", label: "Visualize & export",   desc: "Heatmaps, vorticity, Q-criterion with threshold slider, LBM animation playback." },
  { n: "06", label: "Benchmark",            desc: "Validate accuracy against Poisson, Burgers, cylinder flow, and cavity benchmarks." },
];

export default function Landing() {
  const navigate    = useNavigate();
  const isAuth      = useAppStore((s) => !!s.accessToken);

  const { data: problems }   = useQuery({ queryKey: ["problems"],    queryFn: getProblems });
  const { data: categories } = useQuery({ queryKey: ["categories"],  queryFn: getCategories });

  const [catFilter, setCatFilter] = useState<string>("All");
  const filtered = catFilter === "All"
    ? (problems ?? [])
    : (problems ?? []).filter((p) => p.category === catFilter);

  return (
    <div className="min-h-screen bg-bg text-text">
      {/* ── Nav ──────────────────────────────────────────────────────────── */}
      <nav className="sticky top-0 z-50 bg-surface/80 backdrop-blur border-b border-border">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🍍</span>
            <div>
              <div className="font-bold text-sm text-text leading-tight">PINNeAPPle</div>
              <div className="text-xs text-muted">Physics AI Platform</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {isAuth ? (
              <button className="btn-primary" onClick={() => navigate("/dashboard")}>
                Go to app <ArrowRight size={14} />
              </button>
            ) : (
              <>
                <button
                  className="text-sm text-muted hover:text-text transition-colors px-3 py-1.5"
                  onClick={() => navigate("/login")}
                >
                  Sign in
                </button>
                <button className="btn-primary" onClick={() => navigate("/register")}>
                  Get started free
                </button>
              </>
            )}
          </div>
        </div>
      </nav>

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-6 pt-24 pb-20 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full
                        bg-primary/10 border border-primary/20 text-primary text-xs font-medium mb-8">
          <Zap size={12} /> Physics-Informed ML Platform
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold text-text leading-tight mb-6">
          Solve PDEs with<br />
          <span className="text-primary">AI & Classical Methods</span>
        </h1>
        <p className="text-lg text-muted max-w-2xl mx-auto mb-10">
          PINNeAPPle combines Physics-Informed Neural Networks, Lattice Boltzmann CFD,
          finite-difference and finite-element solvers, and deep learning forecasters
          — in one unified, real-time platform.
        </p>
        <div className="flex items-center justify-center gap-4 flex-wrap">
          {isAuth ? (
            <button
              className="btn-primary text-base px-8 py-3"
              onClick={() => navigate("/dashboard")}
            >
              <Zap size={16} /> Open Dashboard
            </button>
          ) : (
            <>
              <button
                className="btn-primary text-base px-8 py-3"
                onClick={() => navigate("/register")}
              >
                <Zap size={16} /> Start for free
              </button>
              <button
                className="btn-secondary text-base px-8 py-3"
                onClick={() => navigate("/login")}
              >
                Sign in
              </button>
            </>
          )}
        </div>

        {/* Feature pills */}
        <div className="flex flex-wrap justify-center gap-2 mt-10">
          {["PINN", "LBM", "FDM", "FEM", "TCN", "LSTM", "TFT", "FFT"].map((m) => (
            <span
              key={m}
              className="px-3 py-1 text-xs font-mono rounded-full
                         bg-surface2 border border-border text-muted"
            >
              {m}
            </span>
          ))}
        </div>
      </section>

      {/* ── Features ─────────────────────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-6 py-20">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-text mb-3">Everything in one place</h2>
          <p className="text-muted">
            Eight solver types, live training, CFD animation, and rigorous benchmarks.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className={clsx(
                "rounded-2xl border p-6 transition-all hover:shadow-glow",
                f.bg
              )}
            >
              <div className="flex items-center gap-3 mb-3">
                <f.icon size={22} className={f.color} />
                <h3 className="font-semibold text-text">{f.title}</h3>
              </div>
              <p className="text-sm text-muted leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Workflow ─────────────────────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-6 py-20 border-t border-border">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-text mb-3">Six-step workflow</h2>
          <p className="text-muted">From problem description to validated results.</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {WORKFLOW.map((s) => (
            <div key={s.n} className="card p-5">
              <div className="text-3xl font-mono font-black text-primary/30 mb-2">{s.n}</div>
              <h3 className="font-semibold text-text mb-1">{s.label}</h3>
              <p className="text-sm text-muted leading-relaxed">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Problem Library (public preview) ─────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-6 py-20 border-t border-border">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-text mb-1">Problem Library</h2>
            <p className="text-muted text-sm">
              Browse {problems?.length ?? "…"} reference problems — no account needed.
            </p>
          </div>
          <BookOpen size={28} className="text-muted" />
        </div>

        {/* Category filter */}
        <div className="flex flex-wrap gap-2 mb-6">
          {["All", ...(categories ?? [])].map((c) => (
            <button
              key={c}
              onClick={() => setCatFilter(c)}
              className={clsx(
                "px-3 py-1.5 rounded-full text-xs font-medium border transition-all",
                catFilter === c
                  ? "bg-primary/15 text-primary border-primary/40"
                  : "border-border text-muted hover:text-text"
              )}
            >
              {c}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.slice(0, 9).map((p) => (
            <div key={p.key} className="card p-4">
              <div className="flex items-start justify-between gap-2 mb-2">
                <h3 className="text-sm font-semibold text-text">{p.name}</h3>
                <Badge variant="secondary" className="shrink-0 text-xs">
                  {p.category?.split("/")[0].trim()}
                </Badge>
              </div>
              <p className="text-xs text-muted mb-3 line-clamp-2">{p.description}</p>
              <div className="flex flex-wrap gap-1">
                {p.solvers?.map((s: string) => (
                  <Badge key={s} variant="primary" className="text-xs">{s.toUpperCase()}</Badge>
                ))}
              </div>
            </div>
          ))}
        </div>

        {!isAuth && (
          <div className="mt-10 flex flex-col items-center gap-4 p-8
                          rounded-2xl border border-primary/20 bg-primary/5 text-center">
            <Lock size={28} className="text-primary" />
            <div>
              <div className="font-semibold text-text mb-1">
                Create a free account to start solving
              </div>
              <div className="text-sm text-muted">
                Access AI-assisted problem formulation, live training, CFD visualization, and benchmarks.
              </div>
            </div>
            <button
              className="btn-primary px-8 py-2.5"
              onClick={() => navigate("/register")}
            >
              Sign up free
            </button>
          </div>
        )}
      </section>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="border-t border-border py-10 text-center text-xs text-muted">
        <div className="flex items-center justify-center gap-2 mb-2">
          <span className="text-lg">🍍</span>
          <span className="font-semibold text-text">PINNeAPPle</span>
        </div>
        <p>Physics AI Platform — PINN · LBM · FDM · FEM · Timeseries</p>
      </footer>
    </div>
  );
}

