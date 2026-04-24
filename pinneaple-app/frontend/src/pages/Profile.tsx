import { useState, useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAppStore } from "@/store";
import { getMe, updateMe, changePassword } from "@/api/auth";
import { Card } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { Spinner } from "@/components/ui/Spinner";
import { getAllRuns } from "@/api/training";
import { useProjects } from "@/hooks/useProject";
import { User, Lock, BarChart2, Check, AlertCircle, Eye, EyeOff } from "lucide-react";
import toast from "react-hot-toast";
import { format } from "date-fns";
import clsx from "clsx";

export default function Profile() {
  const { user, setAuth, accessToken, refreshToken } = useAppStore();
  const qc = useQueryClient();

  // Sync user from API on mount
  const { data: me } = useQuery({
    queryKey: ["me"],
    queryFn:  getMe,
  });

  // Profile form
  const [form, setForm] = useState({
    first_name: user?.first_name ?? "",
    last_name:  user?.last_name  ?? "",
    email:      user?.email      ?? "",
  });

  useEffect(() => {
    if (me) {
      setForm({ first_name: me.first_name, last_name: me.last_name, email: me.email });
      setAuth(me, accessToken!, refreshToken!);
    }
  }, [me]); // eslint-disable-line react-hooks/exhaustive-deps

  const profileMut = useMutation({
    mutationFn: () => updateMe(form),
    onSuccess: (updated) => {
      setAuth(updated, accessToken!, refreshToken!);
      qc.invalidateQueries({ queryKey: ["me"] });
      toast.success("Profile updated");
    },
    onError: (e: Error) => toast.error(e.message),
  });

  // Password form
  const [pwd, setPwd] = useState({ current: "", next: "", confirm: "" });
  const [showPwd, setShowPwd] = useState(false);

  const pwdMut = useMutation({
    mutationFn: () => changePassword(pwd.current, pwd.next),
    onSuccess: (data) => {
      setAuth(user!, data.access, data.refresh);
      setPwd({ current: "", next: "", confirm: "" });
      toast.success("Password changed — you have been re-authenticated");
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const handlePwd = (e: React.FormEvent) => {
    e.preventDefault();
    if (pwd.next.length < 8)           { toast.error("New password must be ≥ 8 characters"); return; }
    if (pwd.next !== pwd.confirm)      { toast.error("Passwords don't match"); return; }
    pwdMut.mutate();
  };

  // Stats
  const { data: projects } = useProjects();
  const { data: runs }     = useQuery({ queryKey: ["runs_all"], queryFn: getAllRuns });

  const stats = [
    { label: "Projects",      value: projects?.length ?? 0 },
    { label: "Training Runs", value: runs?.length ?? 0 },
    { label: "Completed",     value: runs?.filter((r) => r.status === "done").length ?? 0 },
    { label: "Member since",  value: me ? format(new Date(me.date_joined), "MMM yyyy") : "—" },
  ];

  const displayName = me
    ? [me.first_name, me.last_name].filter(Boolean).join(" ") || me.username
    : user?.username ?? "…";

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-text">Profile</h1>
        <p className="text-muted text-sm mt-1">Manage your account and preferences.</p>
      </div>

      {/* Avatar + stats */}
      <div className="card p-6 flex flex-col sm:flex-row items-start sm:items-center gap-5">
        <div className="w-16 h-16 rounded-full bg-primary/20 border border-primary/30
                        flex items-center justify-center shrink-0">
          <span className="text-2xl font-bold text-primary">
            {displayName.charAt(0).toUpperCase()}
          </span>
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-lg font-semibold text-text truncate">{displayName}</div>
          <div className="text-sm text-muted">{me?.email || me?.username || "—"}</div>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {stats.map((s) => (
            <div key={s.label} className="text-center">
              <div className="text-xl font-bold text-text">{s.value}</div>
              <div className="text-xs text-muted">{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Edit profile */}
      <Card title="Personal Information" action={<User size={16} className="text-muted" />}>
        <form
          onSubmit={(e) => { e.preventDefault(); profileMut.mutate(); }}
          className="space-y-4"
        >
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Input
              label="First name"
              value={form.first_name}
              onChange={(e) => setForm((f) => ({ ...f, first_name: e.target.value }))}
              placeholder="Jane"
            />
            <Input
              label="Last name"
              value={form.last_name}
              onChange={(e) => setForm((f) => ({ ...f, last_name: e.target.value }))}
              placeholder="Doe"
            />
          </div>
          <Input
            label="Email"
            type="email"
            value={form.email}
            onChange={(e) => setForm((f) => ({ ...f, email: e.target.value }))}
            placeholder="jane@example.com"
          />
          <div className="flex items-center gap-3">
            <button type="submit" className="btn-primary" disabled={profileMut.isPending}>
              {profileMut.isPending ? <Spinner size="sm" /> : <Check size={14} />}
              Save changes
            </button>
          </div>
        </form>
      </Card>

      {/* Change password */}
      <Card title="Change Password" action={<Lock size={16} className="text-muted" />}>
        <form onSubmit={handlePwd} className="space-y-4">
          <div className="relative">
            <Input
              label="Current password"
              type={showPwd ? "text" : "password"}
              value={pwd.current}
              onChange={(e) => setPwd((p) => ({ ...p, current: e.target.value }))}
              placeholder="••••••••"
              autoComplete="current-password"
            />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="relative">
              <Input
                label="New password"
                type={showPwd ? "text" : "password"}
                value={pwd.next}
                onChange={(e) => setPwd((p) => ({ ...p, next: e.target.value }))}
                placeholder="≥ 8 characters"
                autoComplete="new-password"
              />
            </div>
            <Input
              label="Confirm new password"
              type={showPwd ? "text" : "password"}
              value={pwd.confirm}
              onChange={(e) => setPwd((p) => ({ ...p, confirm: e.target.value }))}
              placeholder="••••••••"
              autoComplete="new-password"
            />
          </div>
          <div className="flex items-center gap-3">
            <button type="submit" className="btn-primary" disabled={pwdMut.isPending}>
              {pwdMut.isPending ? <Spinner size="sm" /> : <Lock size={14} />}
              Change password
            </button>
            <button
              type="button"
              className="btn-ghost text-xs text-muted"
              onClick={() => setShowPwd((v) => !v)}
            >
              {showPwd ? <EyeOff size={14} /> : <Eye size={14} />}
              {showPwd ? "Hide" : "Show"}
            </button>
          </div>
        </form>
      </Card>

      {/* Account info */}
      <Card title="Account Info">
        <div className="space-y-3 text-sm">
          <Row label="Username"     value={me?.username ?? user?.username ?? "—"} />
          <Row label="User ID"      value={String(me?.id ?? "—")} mono />
          <Row label="Member since" value={me ? format(new Date(me.date_joined), "PPP") : "—"} />
        </div>
      </Card>
    </div>
  );
}

function Row({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border last:border-0">
      <span className="text-muted">{label}</span>
      <span className={clsx("text-text", mono && "font-mono text-xs")}>{value}</span>
    </div>
  );
}
