import api from "./client";
import type { InferenceResult } from "@/types";

export const runInference = (runId: string, ts_data?: number[]) =>
  api.post<InferenceResult>(`/inference/${runId}/`, { ts_data }).then((r) => r.data);
