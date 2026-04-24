import api from "./client";
import type { Project, ModelConfig, ProblemSpec } from "@/types";

export const getProjects = () =>
  api.get<Project[]>("/projects/").then((r) => r.data);

export const getProject = (id: string) =>
  api.get<Project>(`/projects/${id}/`).then((r) => r.data);

export const createProject = (name: string, problem_spec: ProblemSpec) =>
  api.post<Project>("/projects/", { name, problem_spec }).then((r) => r.data);

export const updateProject = (
  id: string,
  data: Partial<{ name: string; problem_spec: ProblemSpec; model_config: ModelConfig; status: string }>
) => api.patch<Project>(`/projects/${id}/`, data).then((r) => r.data);

export const deleteProject = (id: string) =>
  api.delete(`/projects/${id}/`);

export const getProjectRuns = (id: string) =>
  api.get(`/projects/${id}/runs/`).then((r) => r.data);

export const getLatestRun = (id: string) =>
  api.get(`/projects/${id}/latest_run/`).then((r) => r.data);
