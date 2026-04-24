import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAppStore } from "@/store";
import { logout as apiLogout } from "@/api/auth";

export function useAuth() {
  const {
    user, accessToken, refreshToken, authLogout,
  } = useAppStore();

  const isAuthenticated = !!accessToken;
  const navigate        = useNavigate();

  const logout = useCallback(async () => {
    if (refreshToken) {
      try { await apiLogout(refreshToken); } catch { /* ignore — blacklist is best-effort */ }
    }
    authLogout();
    navigate("/", { replace: true });
  }, [refreshToken, authLogout, navigate]);

  return { user, isAuthenticated, logout };
}
