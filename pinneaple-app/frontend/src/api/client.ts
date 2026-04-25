import axios from "axios";
import { useAppStore } from "@/store";

const api = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
  timeout: 120_000,
});

// ── Request: attach JWT access token ─────────────────────────────────────────
api.interceptors.request.use((config) => {
  const token = useAppStore.getState().accessToken;
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// ── Response: auto-refresh on 401, normalise errors ──────────────────────────
let isRefreshing = false;
let queue: Array<{ resolve: (t: string) => void; reject: (e: Error) => void }> = [];

const drainQueue = (error: Error | null, token: string | null) => {
  queue.forEach((p) => (error ? p.reject(error) : p.resolve(token!)));
  queue = [];
};

api.interceptors.response.use(
  (res) => res,
  async (err) => {
    const original = err.config as typeof err.config & { _retry?: boolean };

    if (err.response?.status === 401 && !original._retry) {
      const refresh = useAppStore.getState().refreshToken;

      if (!refresh) {
        useAppStore.getState().authLogout();
        return Promise.reject(new Error("Session expired — please log in again"));
      }

      if (isRefreshing) {
        return new Promise<string>((resolve, reject) => {
          queue.push({ resolve, reject });
        }).then((newToken) => {
          original.headers.Authorization = `Bearer ${newToken}`;
          return api(original);
        });
      }

      original._retry = true;
      isRefreshing    = true;

      try {
        const { data } = await axios.post("/api/auth/token/refresh/", { refresh });
        const newAccess: string = data.access;
        useAppStore.getState().setTokens(newAccess, refresh);
        drainQueue(null, newAccess);
        original.headers.Authorization = `Bearer ${newAccess}`;
        return api(original);
      } catch (refreshErr) {
        drainQueue(refreshErr as Error, null);
        useAppStore.getState().authLogout();
        window.location.href = "/login";
        return Promise.reject(refreshErr);
      } finally {
        isRefreshing = false;
      }
    }

    const msg =
      err.response?.data?.error   ||
      err.response?.data?.detail  ||
      err.response?.data?.non_field_errors?.[0] ||
      err.message                  ||
      "Unknown error";
    return Promise.reject(new Error(msg));
  }
);

export default api;
