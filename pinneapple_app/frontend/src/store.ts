import { create } from 'zustand'
import type { ProblemItem, ModelItem, BenchmarkPayload, ExperimentRequest } from './api'

export type Step = 'problem' | 'geometry' | 'models' | 'run' | 'results'

export interface GeometryConfig {
  useGeometry: boolean
  domainName: string
  collocationStrategy: string
  nInterior: number
  nBoundary: number
  nInitial: number
  seed: number
}

export interface DataConfig {
  useSolver: boolean
  solverKey: string         // '' = auto
  nSnapshots: number
  gridResolution: number
  tEnd: number
}

export interface TrainConfig {
  epochs: number
  lr: number
  device: string
  seed: number
}

export interface TrainingEvent {
  model: string
  epoch: number
  totalEpochs: number
  loss: number
  physLoss: number
  bcLoss: number
  overallProgress: number
}

interface AppState {
  // wizard
  step: Step
  setStep: (s: Step) => void

  // problem
  selectedProblem: ProblemItem | null
  isCustomProblem: boolean
  customProblemDef: Record<string, unknown> | null
  setSelectedProblem: (p: ProblemItem | null) => void
  setCustomProblem: (def: Record<string, unknown> | null) => void

  // geometry & collocation
  geometry: GeometryConfig
  setGeometry: (g: Partial<GeometryConfig>) => void

  // data
  dataConfig: DataConfig
  setDataConfig: (d: Partial<DataConfig>) => void

  // models
  selectedModels: ModelItem[]
  selectedMetrics: string[]
  setSelectedModels: (m: ModelItem[]) => void
  setSelectedMetrics: (m: string[]) => void

  // training
  trainConfig: TrainConfig
  setTrainConfig: (t: Partial<TrainConfig>) => void

  // run state
  experimentId: string | null
  experimentStatus: string
  experimentProgress: number
  trainingEvents: TrainingEvent[]
  setExperimentId: (id: string | null) => void
  setExperimentStatus: (s: string) => void
  setExperimentProgress: (p: number) => void
  pushTrainingEvent: (ev: TrainingEvent) => void
  resetRun: () => void

  // results
  benchmarkPayload: BenchmarkPayload | null
  setBenchmarkPayload: (p: BenchmarkPayload | null) => void
}

export const useStore = create<AppState>((set) => ({
  // wizard
  step: 'problem',
  setStep: (step) => set({ step }),

  // problem
  selectedProblem: null,
  isCustomProblem: false,
  customProblemDef: null,
  setSelectedProblem: (p) => set({ selectedProblem: p, isCustomProblem: false }),
  setCustomProblem: (def) => set({ customProblemDef: def, isCustomProblem: true, selectedProblem: null }),

  // geometry
  geometry: {
    useGeometry: false,
    domainName: '',
    collocationStrategy: 'lhs',
    nInterior: 4096,
    nBoundary: 512,
    nInitial: 256,
    seed: 42,
  },
  setGeometry: (g) => set((s) => ({ geometry: { ...s.geometry, ...g } })),

  // data
  dataConfig: {
    useSolver: true,
    solverKey: '',
    nSnapshots: 5,
    gridResolution: 32,
    tEnd: 1.0,
  },
  setDataConfig: (d) => set((s) => ({ dataConfig: { ...s.dataConfig, ...d } })),

  // models
  selectedModels: [],
  selectedMetrics: ['l2_relative', 'mse', 'pde_residual', 'bc_residual', 'train_time_s', 'n_params'],
  setSelectedModels: (m) => set({ selectedModels: m }),
  setSelectedMetrics: (m) => set({ selectedMetrics: m }),

  // training
  trainConfig: { epochs: 2000, lr: 1e-3, device: 'cpu', seed: 42 },
  setTrainConfig: (t) => set((s) => ({ trainConfig: { ...s.trainConfig, ...t } })),

  // run
  experimentId: null,
  experimentStatus: '',
  experimentProgress: 0,
  trainingEvents: [],
  setExperimentId: (id) => set({ experimentId: id }),
  setExperimentStatus: (s) => set({ experimentStatus: s }),
  setExperimentProgress: (p) => set({ experimentProgress: p }),
  pushTrainingEvent: (ev) => set((s) => ({ trainingEvents: [...s.trainingEvents.slice(-200), ev] })),
  resetRun: () => set({ experimentId: null, experimentStatus: '', experimentProgress: 0, trainingEvents: [] }),

  // results
  benchmarkPayload: null,
  setBenchmarkPayload: (p) => set({ benchmarkPayload: p }),
}))
