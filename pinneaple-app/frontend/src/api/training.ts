import api from "./client";
import type { ModelConfig, TrainingRun } from "@/types";

export const startTraining = (
  projectId: string,
  model_config: ModelConfig,
  ts_data?: number[]
) =>
  api.post<{ ws_run_id: string; db_run_id: string; ws_url: string }>(
    `/projects/${projectId}/train/`,
    { model_config, ts_data }
  ).then((r) => r.data);

export const getTrainingRun = (runId: string) =>
  api.get<TrainingRun>(`/training/${runId}/`).then((r) => r.data);

export const getAllRuns = () =>
  api.get<TrainingRun[]>("/training/").then((r) => r.data);

export const stopTraining = (wsRunId: string) =>
  api.post(`/training/ws/${wsRunId}/stop/`).then((r) => r.data);

export const getTrainingStatus = (wsRunId: string) =>
  api.get(`/training/ws/${wsRunId}/status/`).then((r) => r.data);
