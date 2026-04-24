import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { useAppStore } from "@/store";
import { register } from "@/api/auth";
import { Spinner } from "@/components/ui/Spinner";
import { Input } from "@/components/ui/Input";
import { CheckCircle2, Zap, Eye, EyeOff } from "lucide-react";
import toast from "react-hot-toast";
import clsx from "clsx";

const PERKS = [
  "AI-assisted PDE formulation",
  "Live training with WebSocket updates",
  "LBM CFD animation & vortex visualization",
  "PINN, FDM, FEM, and 4 timeseries models",
  "Rigorous benchmark suite",
];

export default function Register() {
  const navigate  = useNavigate();
  const { setAuth } = useAppStore();

  const [form, setForm] = useState({
    username:    "",
    email:       "",
    password:    "",
    confirm:     "",
    first_name:  "",
    last_name:   "",
  });
  const [showPwd, setShowPwd] = useState(false);

  const update = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  const registerMut = useMutation({
    mutationFn: () =>
      register({
        username:   form.username.trim(),
        password:   form.password,
        email:      form.email.trim(),
        first_name: form.first_name.trim(),
        last_name:  form.last_name.trim(),
      }),
    onSuccess: (data) => {
      setAuth(data.user, data.access, data.refresh);
      toast.success(`Welcome to PINNeAPPle, ${data.user.first_name || data.user.username}!`);
      navigate("/dashboard", { replace: true });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.username.trim()) { toast.error("Username is required");          return; }
    if (form.username.length < 3) { toast.error("Username must be ≥ 3 chars"); return; }
    if (form.password.length < 8) { toast.error("Password must be ≥ 8 chars"); return; }
    if (form.password !== form.confirm) { toast.error("Passwords don't match"); return; }
    registerMut.mutate();
  };

  const pwdStrength = (() => {
    const p = form.password;
    if (!p) return 0;
    let s = 0;
    if (p.length >= 8)    s++;
    if (p.length >= 12)   s++;
    if (/[A-Z]/.test(p))  s++;
    if (/[0-9]/.test(p))  s++;
    if (/[^a-zA-Z0-9]/.test(p)) s++;
    return s;
  })();

  const strengthLabel = ["", "Weak", "Fair", "Good", "Strong", "Very strong"][pwdStrength];
  const strengthColor = ["", "bg-error", "bg-warning", "bg-warning", "bg-success", "bg-success"][pwdStrength];

  return (
    <div className="min-h-screen bg-bg flex items-center justify-center p-4">
      <div className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-8 items-start">

        {/* Left — perks */}
        <div className="hidden md:block pt-8">
          <div className="text-4xl mb-3">🍍</div>
          <h1 className="text-3xl font-extrabold text-text mb-2">PINNeAPPle</h1>
          <p className="text-muted mb-8">
            The unified platform for physics simulation and machine learning.
            Create a free account and start solving PDEs today.
          </p>
          <div className="space-y-3">
            {PERKS.map((perk) => (
              <div key={perk} className="flex items-center gap-3">
                <CheckCircle2 size={16} className="text-success shrink-0" />
                <span className="text-sm text-text">{perk}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right — form */}
        <div className="card p-8">
          <h2 className="text-lg font-semibold text-text mb-6">Create your account</h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <Input
                label="First name"
                value={form.first_name}
                onChange={update("first_name")}
                placeholder="Jane"
                autoComplete="given-name"
                autoFocus
              />
              <Input
                label="Last name"
                value={form.last_name}
                onChange={update("last_name")}
                placeholder="Doe"
                autoComplete="family-name"
              />
            </div>

            <Input
              label="Username *"
              value={form.username}
              onChange={update("username")}
              placeholder="jane_doe"
              autoComplete="username"
            />

            <Input
              label="Email"
              type="email"
              value={form.email}
              onChange={update("email")}
              placeholder="jane@example.com"
              autoComplete="email"
            />

            <div>
              <div className="relative">
                <Input
                  label="Password *"
                  type={showPwd ? "text" : "password"}
                  value={form.password}
                  onChange={update("password")}
                  placeholder="≥ 8 characters"
                  autoComplete="new-password"
                />
                <button
                  type="button"
                  className="absolute right-3 top-8 text-muted hover:text-text"
                  onClick={() => setShowPwd((v) => !v)}
                  tabIndex={-1}
                >
                  {showPwd ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
              {form.password && (
                <div className="mt-1.5 flex items-center gap-2">
                  <div className="flex gap-1 flex-1">
                    {[1,2,3,4,5].map((i) => (
                      <div
                        key={i}
                        className={clsx(
                          "h-1 flex-1 rounded-full transition-all",
                          i <= pwdStrength ? strengthColor : "bg-surface2"
                        )}
                      />
                    ))}
                  </div>
                  <span className="text-xs text-muted w-20">{strengthLabel}</span>
                </div>
              )}
            </div>

            <Input
              label="Confirm password *"
              type={showPwd ? "text" : "password"}
              value={form.confirm}
              onChange={update("confirm")}
              placeholder="••••••••"
              autoComplete="new-password"
            />

            <button
              type="submit"
              className="btn-primary w-full justify-center py-2.5 mt-2"
              disabled={registerMut.isPending}
            >
              {registerMut.isPending ? <Spinner size="sm" /> : <Zap size={15} />}
              {registerMut.isPending ? "Creating account…" : "Create account"}
            </button>
          </form>

          <p className="text-center text-sm text-muted mt-6">
            Already have an account?{" "}
            <Link to="/login" className="text-primary hover:underline font-medium">
              Sign in
            </Link>
          </p>
        </div>
      </div>

      <div className="fixed bottom-4 left-1/2 -translate-x-1/2">
        <Link to="/" className="text-xs text-muted hover:text-text transition-colors">
          ← Back to home
        </Link>
      </div>
    </div>
  );
}
