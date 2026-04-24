import { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { useAppStore } from "@/store";
import { login, getMe } from "@/api/auth";
import { Spinner } from "@/components/ui/Spinner";
import { Input } from "@/components/ui/Input";
import { Zap, Eye, EyeOff } from "lucide-react";
import toast from "react-hot-toast";

export default function Login() {
  const navigate              = useNavigate();
  const location              = useLocation();
  const { setTokens, setAuth } = useAppStore();
  const from = (location.state as { from?: { pathname: string } } | null)
    ?.from?.pathname ?? "/dashboard";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPwd,  setShowPwd]  = useState(false);

  const loginMut = useMutation({
    mutationFn: () => login(username.trim(), password),
    onSuccess: async (tokens) => {
      // Set tokens first so the getMe() call includes Authorization header
      setTokens(tokens.access, tokens.refresh);
      try {
        const user = await getMe();
        setAuth(user, tokens.access, tokens.refresh);
      } catch {
        // If /me fails for any reason, still proceed with a minimal user object
        setAuth(
          { id: 0, username: username.trim(), email: "", first_name: "", last_name: "", date_joined: "" },
          tokens.access,
          tokens.refresh,
        );
      }
      toast.success("Welcome back!");
      navigate(from, { replace: true });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim()) { toast.error("Enter your username"); return; }
    if (!password)         { toast.error("Enter your password"); return; }
    loginMut.mutate();
  };

  return (
    <div className="min-h-screen bg-bg flex items-center justify-center p-4">
      <div className="w-full max-w-sm">
        {/* Brand */}
        <div className="text-center mb-8">
          <div className="text-4xl mb-2">🍍</div>
          <h1 className="text-2xl font-bold text-text">PINNeAPPle</h1>
          <p className="text-muted text-sm mt-1">Physics AI Platform</p>
        </div>

        {/* Card */}
        <div className="card p-8">
          <h2 className="text-lg font-semibold text-text mb-6">Sign in to your account</h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <Input
              label="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
              placeholder="your_username"
            />

            <div className="relative">
              <Input
                label="Password"
                type={showPwd ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
                placeholder="••••••••"
              />
              <button
                type="button"
                className="absolute right-3 top-8 text-muted hover:text-text transition-colors"
                onClick={() => setShowPwd((v) => !v)}
                tabIndex={-1}
              >
                {showPwd ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>

            <button
              type="submit"
              className="btn-primary w-full justify-center py-2.5 mt-2"
              disabled={loginMut.isPending}
            >
              {loginMut.isPending ? <Spinner size="sm" /> : <Zap size={15} />}
              {loginMut.isPending ? "Signing in…" : "Sign in"}
            </button>
          </form>

          <p className="text-center text-sm text-muted mt-6">
            Don&apos;t have an account?{" "}
            <Link to="/register" className="text-primary hover:underline font-medium">
              Sign up free
            </Link>
          </p>
        </div>

        <p className="text-center mt-6">
          <Link to="/" className="text-xs text-muted hover:text-text transition-colors">
            ← Back to home
          </Link>
        </p>
      </div>
    </div>
  );
}
