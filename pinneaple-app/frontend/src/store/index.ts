import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { InferenceResult } from "@/types";
import type { AuthUser } from "@/api/auth";

interface AppState {
  // ── Auth ──────────────────────────────────────────────────────────────────
  user:         AuthUser | null;
  accessToken:  string | null;
  refreshToken: string | null;
  setAuth:      (user: AuthUser, access: string, refresh: string) => void;
  setTokens:    (access: string, refresh: string) => void;
  authLogout:   () => void;

  // ── Active project ────────────────────────────────────────────────────────
  activeProjectId: string | null;
  setActiveProjectId: (id: string | null) => void;

  // ── Active training run ───────────────────────────────────────────────────
  activeRunId:    string | null;   // DB run id
  activeWsRunId:  string | null;   // WebSocket run id
  setActiveRun:   (dbId: string, wsId: string) => void;
  clearActiveRun: () => void;

  // ── Inference result cache ────────────────────────────────────────────────
  inferenceResult: InferenceResult | null;
  setInferenceResult: (r: InferenceResult | null) => void;

  // ── Timeseries data ───────────────────────────────────────────────────────
  tsData:    number[] | null;
  tsCol:     string  | null;
  setTsData: (data: number[], col: string) => void;
  clearTsData: () => void;

  // ── UI ────────────────────────────────────────────────────────────────────
  sidebarCollapsed: boolean;
  toggleSidebar:    () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Auth
      user:         null,
      accessToken:  null,
      refreshToken: null,
      setAuth: (user, access, refresh) =>
        set({ user, accessToken: access, refreshToken: refresh }),
      setTokens: (access, refresh) =>
        set({ accessToken: access, refreshToken: refresh }),
      authLogout: () =>
        set({
          user: null, accessToken: null, refreshToken: null,
          activeProjectId: null, activeRunId: null, activeWsRunId: null,
          inferenceResult: null, tsData: null, tsCol: null,
        }),

      // Project
      activeProjectId: null,
      setActiveProjectId: (id) => set({ activeProjectId: id }),

      // Run
      activeRunId:    null,
      activeWsRunId:  null,
      setActiveRun:   (dbId, wsId) => set({ activeRunId: dbId, activeWsRunId: wsId }),
      clearActiveRun: () => set({ activeRunId: null, activeWsRunId: null }),

      // Inference
      inferenceResult: null,
      setInferenceResult: (r) => set({ inferenceResult: r }),

      // Timeseries
      tsData:      null,
      tsCol:       null,
      setTsData:   (data, col) => set({ tsData: data, tsCol: col }),
      clearTsData: () => set({ tsData: null, tsCol: null }),

      // UI
      sidebarCollapsed: false,
      toggleSidebar: () =>
        set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
    }),
    {
      name: "pinneaple-store",
      // Persist auth tokens and lightweight UI prefs only
      partialize: (s) => ({
        accessToken:      s.accessToken,
        refreshToken:     s.refreshToken,
        user:             s.user,
        activeProjectId:  s.activeProjectId,
        sidebarCollapsed: s.sidebarCollapsed,
      }),
    }
  )
);
