import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAppStore } from "@/store";
import {
  getProjects, getProject, createProject, updateProject,
  deleteProject, getProjectRuns, getLatestRun,
} from "@/api/projects";
import type { ProblemSpec, ModelConfig } from "@/types";

export function useProjects() {
  return useQuery({ queryKey: ["projects"], queryFn: getProjects });
}

export function useProject(id: string | null) {
  return useQuery({
    queryKey: ["project", id],
    queryFn:  () => getProject(id!),
    enabled:  !!id,
  });
}

export function useActiveProject() {
  const id = useAppStore((s) => s.activeProjectId);
  return useProject(id);
}

export function useCreateProject() {
  const qc  = useQueryClient();
  const set  = useAppStore((s) => s.setActiveProjectId);
  return useMutation({
    mutationFn: ({ name, problem_spec }: { name: string; problem_spec: ProblemSpec }) =>
      createProject(name, problem_spec),
    onSuccess: (p) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      set(p.id);
    },
  });
}

export function useUpdateProject(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<{ name: string; problem_spec: ProblemSpec; model_config: ModelConfig }>) =>
      updateProject(id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["project", id] });
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });
}

export function useDeleteProject() {
  const qc  = useQueryClient();
  const clear = useAppStore((s) => s.setActiveProjectId);
  return useMutation({
    mutationFn: (id: string) => deleteProject(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      clear(null);
    },
  });
}

export function useProjectRuns(id: string | null) {
  return useQuery({
    queryKey: ["runs", id],
    queryFn:  () => getProjectRuns(id!),
    enabled:  !!id,
  });
}

export function useLatestRun(id: string | null) {
  return useQuery({
    queryKey: ["latest_run", id],
    queryFn:  () => getLatestRun(id!),
    enabled:  !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "running" || status === "pending" ? 3000 : false;
    },
  });
}
