import axios from 'axios'

const BASE = '/api'
const http = axios.create({ baseURL: BASE, timeout: 60_000 })

export interface ProblemItem {
  name: string
  family: string
  description: string
  tags: string[]
  time_dependent: boolean
  recommended_models: string[]
  recommended_solver: string
}

export interface ModelItem {
  name: string
  family: string
  description: string
  supports_physics_loss: boolean
  tags: string[]
  recommended_for: string[]
}

export interface MetricItem { key: string; label: string }

export interface ExperimentRequest {
  problem_name: string
  custom_problem?: Record<string, unknown>
  collocation: {
    strategy: string
    n_interior: number
    n_boundary: number
    n_initial: number
    seed: number
    use_geometry: boolean
    domain_name?: string
  }
  data: {
    solver_key?: string
    n_snapshots: number
    grid_resolution: number
    t_end: number
    use_solver: boolean
  }
  models: { name: string; extra_kwargs: Record<string, unknown> }[]
  metrics: string[]
  epochs: number
  lr: number
  device: string
  seed: number
}

export interface LeaderboardRow {
  model: string
  l2_relative?: number
  mse?: number
  pde_residual?: number
  bc_residual?: number
  train_time_s?: number
  n_params?: number
  [key: string]: unknown
}

export interface BenchmarkPayload {
  experiment_id: string
  problem_name: string
  completed_at: string
  leaderboard: LeaderboardRow[]
  charts: Record<string, string>   // base64 PNGs
  summary: string
  errors: Record<string, string>
}

// ── API calls ─────────────────────────────────────────────────────────────

export const api = {
  health:       () => http.get<{ status: string }>('/health'),
  info:         () => http.get<Record<string, unknown>>('/info'),

  problems: {
    list:       () => http.get<ProblemItem[]>('/problems'),
    get:        (name: string) => http.get<ProblemItem>(`/problems/${name}`),
    validate:   (body: unknown) => http.post('/problems/custom/validate', body),
  },

  models: {
    list:       (family?: string) => http.get<ModelItem[]>('/models', { params: family ? { family } : {} }),
    families:   () => http.get<{ families: string[] }>('/models/families'),
    recommend:  (family: string, n = 5) => http.get<{ recommendations: string[] }>(`/models/recommend/${family}`, { params: { n } }),
    metrics:    () => http.get<{ available: MetricItem[]; defaults: string[] }>('/models/metrics'),
    get:        (name: string) => http.get<ModelItem>(`/models/${name}`),
  },

  experiments: {
    launch:     (body: ExperimentRequest) => http.post<{ experiment_id: string; status: string; progress: number; message: string }>('/experiments/launch', body),
    status:     (id: string) => http.get<{ experiment_id: string; status: string; progress: number; message: string }>(`/experiments/${id}/status`),
    results:    (id: string) => http.get<BenchmarkPayload>(`/experiments/${id}/results`),
    list:       () => http.get<{ experiment_id: string; status: string }[]>('/experiments'),
  },
}

// ── WebSocket helper ──────────────────────────────────────────────────────
export function connectTrainingWs(
  experimentId: string,
  onEvent: (ev: Record<string, unknown>) => void,
  onDone: () => void,
): () => void {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = window.location.host
  const ws = new WebSocket(`${protocol}://${host}/api/experiments/${experimentId}/ws`)

  ws.onmessage = (e) => {
    try {
      const ev = JSON.parse(e.data)
      if (ev.type === 'done') { onDone(); ws.close() }
      else onEvent(ev)
    } catch (_) {}
  }
  ws.onerror = () => onDone()
  ws.onclose = () => onDone()

  return () => ws.close()
}
