import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { formulateProblem, getProblems, getCategories } from "@/api/problems";
import { useCreateProject } from "@/hooks/useProject";
import { Card } from "@/components/ui/Card";
import { Input, Textarea } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Badge } from "@/components/ui/Badge";
import { Tabs } from "@/components/ui/Tabs";
import { Spinner } from "@/components/ui/Spinner";
import { Modal } from "@/components/ui/Modal";
import type { ProblemSpec } from "@/types";
import { Sparkles, BookOpen, Code2, ArrowRight, Plus, Check } from "lucide-react";
import toast from "react-hot-toast";
import "katex/dist/katex.min.css";
// @ts-ignore
import { BlockMath, InlineMath } from "react-katex";

const TABS = [
  { id: "ai",      label: "AI Formulator", icon: <Sparkles size={14} /> },
  { id: "library", label: "Problem Library", icon: <BookOpen size={14} /> },
  { id: "expert",  label: "Expert Mode",   icon: <Code2 size={14} /> },
];

export default function ProblemSetup() {
  const navigate   = useNavigate();
  const [tab, setTab] = useState("ai");
  const [projectName, setProjectName] = useState("New Physics Project");
  const [description, setDescription] = useState("");
  const [pendingSpec, setPendingSpec]  = useState<ProblemSpec | null>(null);
  const [expertJson, setExpertJson]    = useState(JSON.stringify(EXPERT_DEFAULT, null, 2));
  const [catFilter,  setCatFilter]     = useState("");
  const [confirmModal, setConfirmModal] = useState(false);

  const { data: problems }    = useQuery({ queryKey: ["problems"], queryFn: () => getProblems() });
  const { data: categories }  = useQuery({ queryKey: ["categories"], queryFn: getCategories });
  const createProject         = useCreateProject();

  const formulateMut = useMutation({
    mutationFn: () => formulateProblem(description),
    onSuccess:  (spec) => { setPendingSpec(spec); },
    onError:    (e: Error) => toast.error(e.message),
  });

  const handleCreate = (spec: ProblemSpec) => {
    if (!projectName.trim()) { toast.error("Enter a project name"); return; }
    createProject.mutate(
      { name: projectName, problem_spec: spec },
      {
        onSuccess: () => {
          toast.success("Project created!");
          navigate("/models");
        },
        onError: (e: Error) => toast.error(e.message),
      }
    );
  };

  const handleExpertCreate = () => {
    try {
      const spec = JSON.parse(expertJson) as ProblemSpec;
      spec._source = "expert";
      handleCreate(spec);
    } catch {
      toast.error("Invalid JSON");
    }
  };

  const filtered = problems?.filter(
    (p) => !catFilter || p.category === catFilter
  ) ?? [];

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-text">Problem Setup</h1>
        <p className="text-muted text-sm mt-1">
          Define the physics problem — AI-assisted, from library, or manually.
        </p>
      </div>

      <div className="flex gap-3 items-end">
        <Input
          label="Project Name"
          value={projectName}
          onChange={(e) => setProjectName(e.target.value)}
          className="max-w-xs"
        />
      </div>

      <Tabs tabs={TABS} active={tab} onChange={setTab} />

      {/* ── AI Formulator ───────────────────────────────────────────── */}
      {tab === "ai" && (
        <div className="space-y-4">
          <Card title="Describe your problem" subtitle="Claude will extract equations, domain, BCs, and suggest solvers">
            <Textarea
              label="Problem description"
              placeholder="e.g. Simulate air flow past a circular cylinder at Reynolds number 200 using a 160×64 grid. I want to observe the von Kármán vortex street forming in the wake."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={5}
            />
            <div className="mt-4">
              <button
                className="btn-primary"
                disabled={!description.trim() || formulateMut.isPending}
                onClick={() => formulateMut.mutate()}
              >
                {formulateMut.isPending ? <Spinner size="sm" /> : <Sparkles size={14} />}
                {formulateMut.isPending ? "Analysing…" : "Formulate with AI"}
              </button>
            </div>
          </Card>

          {pendingSpec && (
            <Card title={pendingSpec.name}>
              <ProblemPreview spec={pendingSpec} />
              <div className="mt-4 flex gap-2">
                <button
                  className="btn-primary"
                  onClick={() => handleCreate(pendingSpec)}
                  disabled={createProject.isPending}
                >
                  {createProject.isPending ? <Spinner size="sm" /> : <Plus size={14} />}
                  Create project
                </button>
                <button className="btn-secondary" onClick={() => setPendingSpec(null)}>
                  Clear
                </button>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ── Problem Library ──────────────────────────────────────────── */}
      {tab === "library" && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <Select
              label="Filter by category"
              value={catFilter}
              onChange={(e) => setCatFilter(e.target.value)}
              options={[
                { value: "", label: "All categories" },
                ...(categories ?? []).map((c) => ({ value: c, label: c })),
              ]}
              className="max-w-xs"
            />
            <div className="text-muted text-sm mt-5">
              {filtered.length} problem{filtered.length !== 1 ? "s" : ""}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filtered.map((prob) => (
              <div
                key={prob.key}
                className="card p-4 hover:border-primary/40 hover:shadow-glow transition-all
                           cursor-pointer group"
                onClick={() => { setPendingSpec(prob); setTab("confirm"); }}
              >
                <div className="flex items-start justify-between gap-2 mb-2">
                  <h3 className="text-sm font-semibold text-text group-hover:text-primary transition-colors">
                    {prob.name}
                  </h3>
                  <Badge variant="secondary" className="shrink-0 text-xs">
                    {prob.category?.split("/")[0].trim()}
                  </Badge>
                </div>
                <p className="text-xs text-muted mb-3 line-clamp-2">{prob.description}</p>
                <div className="flex flex-wrap gap-1 mb-3">
                  {prob.solvers?.map((s) => (
                    <Badge key={s} variant="primary" className="text-xs">{s.toUpperCase()}</Badge>
                  ))}
                </div>
                <div className="text-xs text-muted font-mono">{prob.domain ? Object.entries(prob.domain).map(([k, v]) => `${k}∈[${v[0]},${v[1]}]`).join(", ") : ""}</div>
                <button
                  className="mt-3 btn-primary w-full text-xs py-1.5"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleCreate(prob);
                  }}
                  disabled={createProject.isPending}
                >
                  <Plus size={12} /> Use this problem
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Expert mode ──────────────────────────────────────────────── */}
      {tab === "expert" && (
        <Card title="Expert Mode" subtitle="Define the problem spec as JSON">
          <Textarea
            label="Problem spec (JSON)"
            value={expertJson}
            onChange={(e) => setExpertJson(e.target.value)}
            rows={20}
            className="font-mono text-xs"
          />
          <div className="mt-4 flex gap-2">
            <button
              className="btn-primary"
              onClick={handleExpertCreate}
              disabled={createProject.isPending}
            >
              {createProject.isPending ? <Spinner size="sm" /> : <Check size={14} />}
              Create project from JSON
            </button>
            <button
              className="btn-secondary"
              onClick={() => {
                try {
                  const parsed = JSON.parse(expertJson);
                  setExpertJson(JSON.stringify(parsed, null, 2));
                } catch {
                  toast.error("Invalid JSON");
                }
              }}
            >
              Format JSON
            </button>
          </div>
        </Card>
      )}

      {/* ── Confirm tab (from library click) ─────────────────────────── */}
      {tab === "confirm" && pendingSpec && (
        <Card title={pendingSpec.name}>
          <ProblemPreview spec={pendingSpec} />
          <div className="mt-4 flex gap-2">
            <button className="btn-primary" onClick={() => handleCreate(pendingSpec)} disabled={createProject.isPending}>
              {createProject.isPending ? <Spinner size="sm" /> : <ArrowRight size={14} />}
              Create project
            </button>
            <button className="btn-secondary" onClick={() => setTab("library")}>← Back</button>
          </div>
        </Card>
      )}
    </div>
  );
}

function ProblemPreview({ spec }: { spec: ProblemSpec }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="space-y-3">
        <div>
          <div className="text-xs text-muted uppercase tracking-wide mb-1">Category</div>
          <Badge variant="secondary">{spec.category}</Badge>
        </div>
        <div>
          <div className="text-xs text-muted uppercase tracking-wide mb-1">Description</div>
          <p className="text-sm text-text">{spec.description}</p>
        </div>
        <div>
          <div className="text-xs text-muted uppercase tracking-wide mb-1">Governing Equations</div>
          <div className="space-y-1">
            {spec.equations?.map((eq, i) => (
              <div key={i} className="bg-surface2 rounded-lg p-2 overflow-x-auto">
                <BlockMath math={eq} />
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="space-y-3">
        <div>
          <div className="text-xs text-muted uppercase tracking-wide mb-1">Domain</div>
          <div className="space-y-1">
            {Object.entries(spec.domain ?? {}).map(([k, v]) => (
              <div key={k} className="flex justify-between text-xs font-mono">
                <span className="text-secondary">{k}</span>
                <span className="text-muted">[{v[0]}, {v[1]}]</span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="text-xs text-muted uppercase tracking-wide mb-1">Boundary Conditions</div>
          {spec.bcs?.map((bc, i) => (
            <div key={i} className="text-xs text-muted">• {bc}</div>
          ))}
        </div>
        <div>
          <div className="text-xs text-muted uppercase tracking-wide mb-1">Suggested Solvers</div>
          <div className="flex flex-wrap gap-1">
            {spec.solvers?.map((s) => (
              <Badge key={s} variant="primary">{s.toUpperCase()}</Badge>
            ))}
          </div>
        </div>
        {spec.params && Object.keys(spec.params).length > 0 && (
          <div>
            <div className="text-xs text-muted uppercase tracking-wide mb-1">Parameters</div>
            <div className="font-mono text-xs space-y-0.5">
              {Object.entries(spec.params).map(([k, v]) =>
                typeof v !== "object" ? (
                  <div key={k} className="flex justify-between">
                    <span className="text-secondary">{k}</span>
                    <span className="text-muted">{String(v)}</span>
                  </div>
                ) : null
              )}
            </div>
          </div>
        )}
        {spec.ref && (
          <div className="text-xs text-muted">
            <span className="text-muted">Ref: </span>{spec.ref}
          </div>
        )}
      </div>
    </div>
  );
}

const EXPERT_DEFAULT: ProblemSpec = {
  name:        "Custom PDE",
  category:    "Other",
  description: "My custom PDE",
  equations:   ["\\partial u/\\partial t = D \\nabla^2 u"],
  domain:      { x: [0.0, 1.0], y: [0.0, 1.0], t: [0.0, 1.0] },
  params:      { D: 0.01 },
  bcs:         ["u=0 on ∂Ω"],
  ics:         ["u(x,y,0) = sin(\\pi x)sin(\\pi y)"],
  dim:         3,
  tags:        ["parabolic", "diffusion"],
  solvers:     ["fdm", "pinn"],
  ref:         "",
};
