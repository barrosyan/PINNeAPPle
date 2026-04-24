import api from "./client";
import type { ProblemSpec } from "@/types";

export const getProblems = (category?: string) =>
  api.get<ProblemSpec[]>("/problems/", { params: category ? { category } : {} })
     .then((r) => r.data);

export const getProblem = (key: string) =>
  api.get<ProblemSpec>(`/problems/${key}/`).then((r) => r.data);

export const getCategories = () =>
  api.get<string[]>("/problems/categories/").then((r) => r.data);

export const formulateProblem = (description: string) =>
  api.post<ProblemSpec>("/problems/formulate/", { description }).then((r) => r.data);

export const suggestModels = (problem: ProblemSpec) =>
  api.post<Array<{ model: string; type: string; reason: string; score: number }>>(
    "/problems/suggest/", problem
  ).then((r) => r.data);
