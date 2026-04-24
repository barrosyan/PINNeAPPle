import { useState, useRef, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useActiveProject } from "@/hooks/useProject";
import { useAppStore } from "@/store";
import { useAuth } from "@/hooks/useAuth";
import { ChevronRight, FolderOpen, LogOut, User, ChevronDown } from "lucide-react";
import clsx from "clsx";

const PAGE_NAMES: Record<string, string> = {
  dashboard:     "Dashboard",
  problem:       "Problem Setup",
  data:          "Data & Geometry",
  models:        "Models",
  training:      "Training",
  inference:     "Inference",
  visualization: "Visualization",
  benchmarks:    "Benchmarks",
  profile:       "Profile",
};

export default function TopBar() {
  const location              = useLocation();
  const navigate              = useNavigate();
  const { data: project }     = useActiveProject();
  const activeId              = useAppStore((s) => s.activeProjectId);
  const storeUser             = useAppStore((s) => s.user);
  const { logout }            = useAuth();
  const [dropOpen, setDropOpen] = useState(false);
  const dropRef               = useRef<HTMLDivElement>(null);

  const page     = location.pathname.split("/")[1] || "dashboard";
  const pageName = PAGE_NAMES[page] || page;

  const displayName = storeUser
    ? [storeUser.first_name, storeUser.last_name].filter(Boolean).join(" ") || storeUser.username
    : "Account";
  const initials = displayName.charAt(0).toUpperCase();

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropRef.current && !dropRef.current.contains(e.target as Node)) {
        setDropOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <header className="h-16 flex items-center justify-between px-6
                        bg-surface border-b border-border shrink-0">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm">
        <span className="text-muted">PINNeAPPle</span>
        <ChevronRight size={14} className="text-border" />
        <span className="text-text font-medium">{pageName}</span>
      </div>

      <div className="flex items-center gap-3">
        {/* Active project pill */}
        {activeId ? (
          <button
            onClick={() => navigate("/dashboard")}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface2
                       border border-border text-sm text-text hover:border-primary/50
                       transition-colors"
          >
            <FolderOpen size={14} className="text-primary" />
            <span className="max-w-[160px] truncate">
              {project?.name ?? "Loading…"}
            </span>
            {project?.latest_run_status && (
              <StatusDot status={project.latest_run_status} />
            )}
          </button>
        ) : (
          <button
            onClick={() => navigate("/problem")}
            className="text-xs text-muted hover:text-primary transition-colors"
          >
            No project — create one →
          </button>
        )}

        {/* User dropdown */}
        <div ref={dropRef} className="relative">
          <button
            onClick={() => setDropOpen((v) => !v)}
            className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg
                       border border-border hover:border-border/70 hover:bg-surface2
                       transition-all"
          >
            <div className="w-6 h-6 rounded-full bg-primary/20 border border-primary/30
                            flex items-center justify-center text-xs font-bold text-primary">
              {initials}
            </div>
            <span className="text-sm text-text max-w-[120px] truncate hidden sm:block">
              {displayName}
            </span>
            <ChevronDown size={13} className={clsx("text-muted transition-transform", dropOpen && "rotate-180")} />
          </button>

          {dropOpen && (
            <div className="absolute right-0 top-full mt-1.5 w-48 bg-surface border border-border
                            rounded-xl shadow-xl z-50 overflow-hidden">
              <div className="px-3 py-2.5 border-b border-border">
                <div className="text-xs font-medium text-text truncate">{displayName}</div>
                <div className="text-xs text-muted truncate">{storeUser?.email || storeUser?.username}</div>
              </div>
              <div className="p-1">
                <button
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm
                             text-muted hover:text-text hover:bg-surface2 transition-colors"
                  onClick={() => { setDropOpen(false); navigate("/profile"); }}
                >
                  <User size={14} />
                  Profile
                </button>
                <button
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm
                             text-error hover:bg-error/10 transition-colors"
                  onClick={() => { setDropOpen(false); logout(); }}
                >
                  <LogOut size={14} />
                  Sign out
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

function StatusDot({ status }: { status: string }) {
  const colors: Record<string, string> = {
    running: "bg-warning animate-pulse",
    done:    "bg-success",
    error:   "bg-error",
    stopped: "bg-muted",
    pending: "bg-secondary animate-pulse",
  };
  return (
    <span className={`inline-block w-2 h-2 rounded-full ${colors[status] ?? "bg-muted"}`} />
  );
}
