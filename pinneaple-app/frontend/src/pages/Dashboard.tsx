import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useAppStore } from "@/store";
import { useActiveProject } from "@/hooks/useProject";
import { useProjects, useDeleteProject } from "@/hooks/useProject";
import { getAllRuns } from "@/api/training";
import { Card, MetricCard } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/Badge";
import { FullPageSpinner } from "@/components/ui/Spinner";
import LossChart from "@/components/charts/LossChart";
import { Plus, Trash2, ArrowRight, Zap, FlaskConical, Trophy, Eye } from "lucide-react";
import { format } from "date-fns";
import toast from "react-hot-toast";

export default function Dashboard() {
  const navigate        = useNavigate();
  const { activeProjectId, setActiveProjectId } = useAppStore();
  const { data: project, isLoading } = useActiveProject();
  const { data: projects }           = useProjects();
  const { data: runs }               = useQuery({
    queryKey: ["runs_all"],
    queryFn:  getAllRuns,
    refetchInterval: (q) => {
      const r = q.state.data;
      return r?.some((x: { status: string }) => x.status === "running") ? 3000 : false;
    },
  });
  const deleteMut = useDeleteProject();

  const activeRuns   = runs?.filter((r) => r.status === "running").length ?? 0;
  const completedRuns = runs?.filter((r) => r.status === "done").length ?? 0;
  const latestRun    = project
    ? runs?.filter((r) => r.project === project.id).sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      )[0]
    : null;

  if (isLoading) return <FullPageSpinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text">Dashboard</h1>
          <p className="text-muted text-sm mt-1">
            Physics AI Platform — {projects?.length ?? 0} project{projects?.length !== 1 ? "s" : ""}
          </p>
        </div>
        <button className="btn-primary" onClick={() => navigate("/problem")}>
          <Plus size={16} /> New Project
        </button>
      </div>

      {/* Global stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Projects"       value={projects?.length ?? 0} />
        <MetricCard label="Training Runs"  value={runs?.length ?? 0} />
        <MetricCard label="Running Now"    value={activeRuns} accent={activeRuns > 0} />
        <MetricCard label="Completed"      value={completedRuns} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active project */}
        <div className="lg:col-span-2 space-y-4">
          {project ? (
            <>
              <Card
                title={project.name}
                subtitle={`${project.problem_spec?.category ?? "—"} · Created ${format(new Date(project.created_at), "PPp")}`}
                action={
                  <StatusBadge status={project.latest_run_status ?? "created"} />
                }
              >
                <div className="grid grid-cols-3 gap-3 mb-4">
                  <div className="card-inner p-3">
                    <div className="text-xs text-muted mb-1">Problem</div>
                    <div className="text-sm font-medium text-text truncate">
                      {project.problem_spec?.name ?? "—"}
                    </div>
                  </div>
                  <div className="card-inner p-3">
                    <div className="text-xs text-muted mb-1">Model</div>
                    <div className="text-sm font-medium text-text">
                      {project.model_config?.type?.toUpperCase() ?? "—"}
                    </div>
                  </div>
                  <div className="card-inner p-3">
                    <div className="text-xs text-muted mb-1">Runs</div>
                    <div className="text-sm font-medium text-text">{project.run_count}</div>
                  </div>
                </div>

                {/* Latest run loss curve */}
                {latestRun?.history?.length > 0 && (
                  <div className="mt-2">
                    <div className="text-xs text-muted mb-2">Latest training loss</div>
                    <LossChart history={latestRun.history} height={200} />
                  </div>
                )}

                {/* Quick actions */}
                <div className="flex gap-2 mt-4">
                  <button className="btn-primary" onClick={() => navigate("/training")}>
                    <Zap size={14} /> Train
                  </button>
                  <button className="btn-secondary" onClick={() => navigate("/inference")}>
                    <ArrowRight size={14} /> Inference
                  </button>
                  <button className="btn-secondary" onClick={() => navigate("/visualization")}>
                    <Eye size={14} /> Visualize
                  </button>
                </div>
              </Card>
            </>
          ) : (
            <Card>
              <div className="flex flex-col items-center py-12 text-center">
                <FlaskConical size={48} className="text-muted mb-4" />
                <h3 className="text-text font-semibold mb-2">No active project</h3>
                <p className="text-muted text-sm mb-6 max-w-xs">
                  Start by setting up a physics problem — use AI formulation or pick from the library.
                </p>
                <button className="btn-primary" onClick={() => navigate("/problem")}>
                  <Plus size={16} /> Create first project
                </button>
              </div>
            </Card>
          )}
        </div>

        {/* All projects list */}
        <div>
          <Card title="All Projects">
            {!projects?.length ? (
              <p className="text-muted text-sm text-center py-6">No projects yet</p>
            ) : (
              <div className="space-y-2">
                {projects.map((p) => (
                  <div
                    key={p.id}
                    className={`flex items-center justify-between p-3 rounded-lg cursor-pointer
                                border transition-all
                                ${p.id === activeProjectId
                                  ? "border-primary/40 bg-primary/5"
                                  : "border-border hover:border-border/70 hover:bg-surface2"
                                }`}
                    onClick={() => setActiveProjectId(p.id)}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-medium text-text truncate">{p.name}</div>
                      <div className="text-xs text-muted truncate">
                        {p.problem_spec?.category ?? "—"} · {p.run_count} run{p.run_count !== 1 ? "s" : ""}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 ml-2">
                      <StatusBadge status={p.latest_run_status ?? "created"} />
                      <button
                        className="btn-ghost p-1 text-error hover:text-error"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm(`Delete "${p.name}"?`)) {
                            deleteMut.mutate(p.id, {
                              onSuccess: () => toast.success("Project deleted"),
                              onError:   (e) => toast.error(e.message),
                            });
                          }
                        }}
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Recent runs */}
      {runs && runs.length > 0 && (
        <Card title="Recent Training Runs">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left border-b border-border">
                  <th className="pb-2 text-xs text-muted font-medium">Model</th>
                  <th className="pb-2 text-xs text-muted font-medium">Status</th>
                  <th className="pb-2 text-xs text-muted font-medium">Loss</th>
                  <th className="pb-2 text-xs text-muted font-medium">Started</th>
                  <th className="pb-2 text-xs text-muted font-medium"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {runs.slice(0, 10).map((r) => (
                  <tr key={r.id} className="hover:bg-surface2/50 transition-colors">
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
                      {format(new Date(r.created_at), "MMM d HH:mm")}
                    </td>
                    <td className="py-2.5 text-right">
                      {r.status === "done" && (
                        <button
                          className="btn-ghost text-xs py-1"
                          onClick={() => {
                            useAppStore.getState().setActiveRun(r.id, r.ws_run_id);
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

      {/* Quick links */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { icon: FlaskConical, label: "Set up a problem",   path: "/problem",       color: "text-secondary" },
          { icon: Zap,          label: "Run training",        path: "/training",      color: "text-primary"   },
          { icon: Eye,          label: "Visualize results",   path: "/visualization", color: "text-accent"    },
          { icon: Trophy,       label: "Run benchmarks",      path: "/benchmarks",    color: "text-success"   },
        ].map(({ icon: Icon, label, path, color }) => (
          <button
            key={path}
            className="card p-4 flex flex-col items-center gap-2 hover:border-border/70 transition-all
                       hover:shadow-glow text-center"
            onClick={() => navigate(path)}
          >
            <Icon size={28} className={color} />
            <span className="text-sm text-muted">{label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
