export interface ProblemSpec {
  key?: string;
  name: string;
  category: string;
  description: string;
  equations: string[];
  domain: Record<string, [number, number]>;
  params: Record<string, number | string | Record<string, unknown>>;
  bcs: string[];
  ics: string[];
  dim: number;
  tags: string[];
  solvers: string[];
  ref?: string;
  _source?: string;
  _preset_key?: string;
}

export interface ObstacleConfig {
  type: "cylinder" | "rectangle";
  cx?: number; cy?: number; r?: number;
  x0?: number; x1?: number; y0?: number; y1?: number;
}

export interface ModelConfig {
  type: string;
  name?: string;
  // PINN
  n_epochs?: number;
  lr?: number;
  hidden?: number;
  n_layers?: number;
  n_interior?: number;
  // LBM
  steps?: number;
  save_every?: number;
  nx?: number;
  ny?: number;
  Re?: number;
  u_in?: number;
  obstacle?: ObstacleConfig;
  // FDM/FEM
  iters?: number;
  // TS
  input_len?: number;
  horizon?: number;
  epochs?: number;
  // FFT
  n_harmonics?: number;
  detrend?: boolean;
}

export interface Project {
  id: string;
  name: string;
  problem_spec: ProblemSpec;
  model_config: ModelConfig | null;
  status: string;
  created_at: string;
  updated_at: string;
  run_count: number;
  latest_run_status: string | null;
}

export interface TrainingEntry {
  epoch: number;
  loss: number;
  pde?: number;
  bc?: number;
}

export interface TrainingRun {
  id: string;
  project: string;
  model_type: string;
  config: ModelConfig;
  history: TrainingEntry[];
  final_loss: number | null;
  status: "pending" | "running" | "done" | "error" | "stopped";
  error_msg: string;
  result_data: Record<string, unknown>;
  ws_run_id: string;
  created_at: string;
  completed_at: string | null;
}

export interface WSMessage {
  type: "init" | "progress" | "done" | "error" | "stopped";
  status: string;
  epoch?: number;
  loss?: number;
  pde?: number;
  bc?: number;
  total?: number;
  msg?: string;
  result?: Record<string, unknown>;
  final_loss?: number;
  history?: TrainingEntry[];
}

export interface BenchmarkDef {
  key: string;
  name: string;
  category: string;
  description: string;
  params: Record<string, unknown>;
}

export interface BenchmarkResult {
  id: string;
  benchmark_key: string;
  config: Record<string, unknown>;
  metrics: Record<string, unknown>;
  created_at: string;
}

export interface UploadedFile {
  id: string;
  project: string | null;
  name: string;
  file: string;
  file_type: string;
  meta: Record<string, unknown>;
  created_at: string;
}

export interface InferenceResult {
  // PINN
  x?: number[];
  y?: number[];
  u?: number[][];
  coord_keys?: string[];
  // LBM
  vel_mag?: number[][];
  ux?: number[][];
  uy?: number[][];
  rho?: number[][];
  vorticity?: number[][];
  Q?: number[][];
  obstacle?: boolean[][] | null;
  nx?: number;
  ny?: number;
  trajectory_ux?: number[][][];
  trajectory_uy?: number[][][];
  // TS forecast
  forecast?: number[];
  horizon?: number;
  input_len?: number;
  type?: string;
  history?: TrainingEntry[];
  // FDM/FEM
  field?: number[][];
  nodes?: number[][];
  exact?: number[][];
  coord_keys_fdm?: string[];
  label?: string;
  l2_error?: number;
  linf_error?: number;
}

export interface ModelCatalogEntry {
  type: string;
  name: string;
  category: string;
  description: string;
  params: Record<string, ParamDef>;
}

export interface ParamDef {
  label: string;
  type: "int" | "float" | "bool" | "select";
  default: number | boolean | string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
}
