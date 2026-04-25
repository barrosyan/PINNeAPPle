import { useNavigate } from "react-router-dom";
import { useAppStore } from "@/store";
import { Zap, ArrowRight, Lock } from "lucide-react";
import clsx from "clsx";

// ── Stats (from presentation slide 1) ────────────────────────────────────────
const STATS = [
  { num: "26",  label: "Python Packages" },
  { num: "76",  label: "Model Architectures" },
  { num: "40+", label: "Problem Presets" },
  { num: "15+", label: "PDE Types" },
];

// ── Full lifecycle pipeline (slide 3) ────────────────────────────────────────
const PIPELINE = [
  { icon: "📐", label: "Symbolic problem specification" },
  { icon: "🗂️", label: "Geometry, voxelization & data preparation" },
  { icon: "⚙️", label: "Problem-agnostic numerical solvers (FDM · FEM · FVM · meshfree)" },
  { icon: "🧠", label: "Physics-constrained model training (76 architectures)" },
  { icon: "📊", label: "CFD-style visualization & validation" },
  { icon: "🚀", label: "Real-time digital twin deployment" },
];

// ── 8 key differentiators (slide 23) ─────────────────────────────────────────
const DIFFERENTIATORS = [
  { title: "Unified ProblemSpec",            desc: "One object consumed identically by every module — solvers, loss compilers, visualization and digital twin." },
  { title: "Problem-agnostic FDM/FEM/FVM",  desc: "Numerical solvers auto-read ProblemSpec. No boilerplate — boundary conditions applied automatically." },
  { title: "Meshfree methods",               desc: "RBF collocation (Kansa) and Moving Least Squares for scattered nodes and complex geometries." },
  { title: "Voxelization pipeline",          desc: "SDF / point cloud / domain bounds → collocation points in one call. Works directly with PINN training." },
  { title: "CFD-style visualization",        desc: "Scalar fields, streamlines, FEM/FVM plots, Q-criterion iso-surfaces, GIF animations out of the box." },
  { title: "Zarr-backed streaming",          desc: "Training on out-of-GPU-memory datasets with sharded, atomic reads and 4-worker DataLoader." },
  { title: "Live digital twin runtime",      desc: "Multi-protocol sensor fusion (MQTT · Kafka · HTTP), Kalman/EnKF state assimilation, anomaly detection." },
  { title: "Distribution-free conformal UQ", desc: "Coverage guarantees at level 1−α with zero distributional assumptions — MC dropout and ensemble UQ also available." },
];

// ── Comparison (slide 22) ─────────────────────────────────────────────────────
const COMPARE = [
  { feature: "PINN training",                   us: true,  dxde: true,  sciann: true,  physnemo: true  },
  { feature: "Neural operators",                us: true,  dxde: "○",   sciann: false, physnemo: true  },
  { feature: "Problem-agnostic FDM/FEM/FVM",    us: true,  dxde: "○",   sciann: false, physnemo: "○"   },
  { feature: "Meshfree (RBF/MLS)",              us: true,  dxde: false, sciann: false, physnemo: false },
  { feature: "Voxelization pipeline",           us: true,  dxde: false, sciann: false, physnemo: "○"   },
  { feature: "Uncertainty quantification",      us: true,  dxde: false, sciann: false, physnemo: "○"   },
  { feature: "Digital twin runtime",            us: true,  dxde: false, sciann: false, physnemo: false },
  { feature: "Meta-learning",                   us: true,  dxde: false, sciann: false, physnemo: false },
  { feature: "REST serving + ONNX export",      us: true,  dxde: false, sciann: false, physnemo: true  },
  { feature: "Unified ProblemSpec",             us: true,  dxde: false, sciann: false, physnemo: false },
];

function Cell({ v }: { v: boolean | "○" }) {
  if (v === true)  return <td className="text-center text-success font-bold">✓</td>;
  if (v === "○")   return <td className="text-center text-warning font-bold text-sm">○</td>;
  return <td className="text-center text-muted text-sm">—</td>;
}

export default function Landing() {
  const navigate = useNavigate();
  const isAuth   = useAppStore((s) => !!s.accessToken);

  return (
    <div className="min-h-screen bg-bg text-text">

      {/* ── Nav ─────────────────────────────────────────────────────────── */}
      <nav className="sticky top-0 z-50 bg-surface/80 backdrop-blur border-b border-border">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🍍</span>
            <div>
              <div className="font-bold text-sm text-text leading-tight">PINNeAPPle</div>
              <div className="text-xs text-muted">Physics-Informed ML Platform</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {isAuth ? (
              <button className="btn-primary" onClick={() => navigate("/dashboard")}>
                Open app <ArrowRight size={14} />
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

      {/* ── Hero ────────────────────────────────────────────────────────── */}
      <section className="max-w-5xl mx-auto px-6 pt-24 pb-16 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full
                        bg-primary/10 border border-primary/20 text-primary text-xs font-medium mb-8">
          <Zap size={12} /> v0.4 · Open Source · MIT License
        </div>

        <h1 className="text-5xl md:text-6xl font-extrabold text-text leading-tight mb-6">
          A Comprehensive Framework for<br />
          <span className="text-primary">Physics-Informed ML at Scale</span>
        </h1>

        <p className="text-lg text-muted max-w-2xl mx-auto mb-10">
          PINNeAPPle covers the <strong className="text-text">complete lifecycle</strong> —
          from symbolic problem specification and geometry prep, through numerical solvers
          and 76-model training, to CFD-quality visualization and real-time digital twins.
        </p>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto mb-10">
          {STATS.map((s) => (
            <div key={s.label} className="card p-4">
              <div className="text-3xl font-extrabold text-primary tracking-tight">{s.num}</div>
              <div className="text-xs text-muted mt-1">{s.label}</div>
            </div>
          ))}
        </div>

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
      </section>

      {/* ── Full lifecycle pipeline ──────────────────────────────────────── */}
      <section className="max-w-5xl mx-auto px-6 py-20 border-t border-border">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-text mb-3">End-to-end. No gaps.</h2>
          <p className="text-muted">Every stage from problem description to live deployment — connected.</p>
        </div>

        <div className="max-w-xl mx-auto flex flex-col gap-1">
          {PIPELINE.map((step, i) => (
            <div key={i} className="flex flex-col items-start">
              <div className="w-full card p-4 flex items-center gap-4">
                <span className="text-xl shrink-0">{step.icon}</span>
                <span className="text-sm text-text font-medium">{step.label}</span>
              </div>
              {i < PIPELINE.length - 1 && (
                <div className="self-center text-primary text-lg leading-none py-0.5">↓</div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* ── 8 Key Differentiators ─────────────────────────────────────────── */}
      <section className="max-w-5xl mx-auto px-6 py-20 border-t border-border">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-text mb-3">8 key differentiators</h2>
          <p className="text-muted">What PINNeAPPle does that no single tool does today.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {DIFFERENTIATORS.map((d) => (
            <div key={d.title} className="card p-5 border-l-2 border-primary/40">
              <div className="font-semibold text-sm text-primary mb-1">{d.title}</div>
              <div className="text-xs text-muted leading-relaxed">{d.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Comparison table ─────────────────────────────────────────────── */}
      <section className="max-w-5xl mx-auto px-6 py-20 border-t border-border">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-text mb-3">vs. Related tools</h2>
          <p className="text-muted text-sm">✓ full &nbsp;·&nbsp; ○ partial &nbsp;·&nbsp; — absent</p>
        </div>

        <div className="card overflow-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left p-3 text-muted font-semibold text-xs uppercase tracking-wide">Feature</th>
                <th className="text-center p-3 text-primary font-bold text-xs uppercase tracking-wide">PINNeAPPle</th>
                <th className="text-center p-3 text-muted font-semibold text-xs uppercase tracking-wide">DeepXDE</th>
                <th className="text-center p-3 text-muted font-semibold text-xs uppercase tracking-wide">SciANN</th>
                <th className="text-center p-3 text-muted font-semibold text-xs uppercase tracking-wide">PhysNeMo</th>
              </tr>
            </thead>
            <tbody>
              {COMPARE.map((row, i) => (
                <tr key={row.feature} className={clsx("border-b border-border/50", i % 2 === 0 && "bg-surface/30")}>
                  <td className="p-3 text-text text-xs">{row.feature}</td>
                  <Cell v={row.us} />
                  <Cell v={row.dxde} />
                  <Cell v={row.sciann} />
                  <Cell v={row.physnemo} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── CTA ─────────────────────────────────────────────────────────── */}
      {!isAuth && (
        <section className="max-w-5xl mx-auto px-6 py-20 border-t border-border">
          <div className="flex flex-col items-center gap-5 p-10 rounded-2xl
                          border border-primary/20 bg-primary/5 text-center">
            <Lock size={28} className="text-primary" />
            <div>
              <div className="text-xl font-bold text-text mb-2">
                Use the full pipeline — free account
              </div>
              <div className="text-sm text-muted max-w-md mx-auto">
                AI-assisted formulation, live training, CFD visualization, benchmarking
                and result export — all in one place.
              </div>
            </div>
            <div className="flex gap-3 flex-wrap justify-center">
              <button
                className="btn-primary px-8 py-2.5"
                onClick={() => navigate("/register")}
              >
                Sign up free
              </button>
              <button
                className="btn-secondary px-8 py-2.5"
                onClick={() => navigate("/login")}
              >
                Sign in
              </button>
            </div>
          </div>
        </section>
      )}

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="border-t border-border py-8 text-center text-xs text-muted">
        <div className="flex items-center justify-center gap-2 mb-2">
          <span className="text-xl">🍍</span>
          <span className="font-bold text-text text-sm">PINNeAPPle</span>
          <span className="text-muted">v0.4</span>
        </div>
        <p>A Comprehensive Python Framework for Physics-Informed ML at Scale</p>
        <p className="mt-1 opacity-60">26 packages · 76 models · 40+ presets · MIT License</p>
      </footer>
    </div>
  );
}
