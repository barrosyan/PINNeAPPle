import { NavLink, useNavigate } from "react-router-dom";
import { useAppStore } from "@/store";
import { useAuth } from "@/hooks/useAuth";
import {
  LayoutDashboard, FlaskConical, Database, Cpu, Zap,
  Target, Eye, Trophy, ChevronLeft, ChevronRight,
  User, LogOut,
} from "lucide-react";
import clsx from "clsx";

const NAV = [
  { to: "/dashboard",     icon: LayoutDashboard, label: "Dashboard"      },
  { to: "/problem",       icon: FlaskConical,    label: "Problem Setup"  },
  { to: "/data",          icon: Database,        label: "Data & Geometry" },
  { to: "/models",        icon: Cpu,             label: "Models"         },
  { to: "/training",      icon: Zap,             label: "Training"       },
  { to: "/inference",     icon: Target,          label: "Inference"      },
  { to: "/visualization", icon: Eye,             label: "Visualization"  },
  { to: "/benchmarks",    icon: Trophy,          label: "Benchmarks"     },
];

export default function Sidebar() {
  const { sidebarCollapsed: collapsed, toggleSidebar, user } = useAppStore();
  const { logout } = useAuth();
  const navigate   = useNavigate();

  const displayName = user
    ? [user.first_name, user.last_name].filter(Boolean).join(" ") || user.username
    : "Account";

  return (
    <aside
      className={clsx(
        "fixed left-0 top-0 h-full bg-surface border-r border-border flex flex-col z-30",
        "transition-all duration-300",
        collapsed ? "w-16" : "w-60"
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-border min-h-[64px]">
        <span className="text-2xl">🍍</span>
        {!collapsed && (
          <div className="animate-fade-in">
            <div className="text-text font-bold text-sm leading-tight">PINNeAPPle</div>
            <div className="text-muted text-xs">Physics AI Platform</div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all",
                "group relative",
                isActive
                  ? "bg-primary/15 text-primary border border-primary/20"
                  : "text-muted hover:text-text hover:bg-surface2"
              )
            }
          >
            <Icon size={18} className="shrink-0" />
            {!collapsed && (
              <span className="animate-fade-in truncate">{label}</span>
            )}
            {collapsed && (
              <div className="absolute left-full ml-2 px-2 py-1 bg-surface3 border border-border
                              rounded text-xs text-text opacity-0 group-hover:opacity-100
                              pointer-events-none whitespace-nowrap z-50 transition-opacity">
                {label}
              </div>
            )}
          </NavLink>
        ))}
      </nav>

      {/* User section */}
      <div className="border-t border-border">
        {/* Profile link */}
        <NavLink
          to="/profile"
          className={({ isActive }) =>
            clsx(
              "flex items-center gap-3 px-3 py-2.5 mx-2 my-1 rounded-lg text-sm transition-all group relative",
              isActive
                ? "bg-primary/15 text-primary border border-primary/20"
                : "text-muted hover:text-text hover:bg-surface2"
            )
          }
        >
          <div className="w-[18px] h-[18px] rounded-full bg-primary/20 border border-primary/30
                          flex items-center justify-center text-xs font-bold text-primary shrink-0">
            {displayName.charAt(0).toUpperCase()}
          </div>
          {!collapsed && (
            <span className="animate-fade-in truncate text-xs">{displayName}</span>
          )}
          {collapsed && (
            <div className="absolute left-full ml-2 px-2 py-1 bg-surface3 border border-border
                            rounded text-xs text-text opacity-0 group-hover:opacity-100
                            pointer-events-none whitespace-nowrap z-50 transition-opacity">
              Profile
            </div>
          )}
        </NavLink>

        {/* Logout */}
        <button
          onClick={logout}
          className={clsx(
            "flex items-center gap-3 px-3 py-2 mx-2 mb-1 w-[calc(100%-16px)] rounded-lg",
            "text-sm text-muted hover:text-error hover:bg-error/10 transition-all group relative"
          )}
        >
          <LogOut size={16} className="shrink-0" />
          {!collapsed && <span className="text-xs">Sign out</span>}
          {collapsed && (
            <div className="absolute left-full ml-2 px-2 py-1 bg-surface3 border border-border
                            rounded text-xs text-text opacity-0 group-hover:opacity-100
                            pointer-events-none whitespace-nowrap z-50 transition-opacity">
              Sign out
            </div>
          )}
        </button>

        {/* Collapse toggle */}
        <button
          onClick={toggleSidebar}
          className="flex items-center justify-center w-full py-3 border-t border-border
                     text-muted hover:text-text hover:bg-surface2 transition-colors"
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>
    </aside>
  );
}
