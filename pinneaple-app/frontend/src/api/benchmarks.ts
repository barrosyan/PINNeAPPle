import api from "./client";
import type { BenchmarkDef, BenchmarkResult } from "@/types";

export const getBenchmarks = () =>
  api.get<BenchmarkDef[]>("/benchmarks/").then((r) => r.data);

export const runBenchmark = (benchmark_key: string, config: Record<string, unknown>) =>
  api.post<BenchmarkResult & { metrics: Record<string, unknown> }>(
    "/benchmarks/run/",
    { benchmark_key, config }
  ).then((r) => r.data);

export const getBenchmarkResults = () =>
  api.get<BenchmarkResult[]>("/benchmarks/results/").then((r) => r.data);

export const clearBenchmarkResults = () =>
  api.delete("/benchmarks/results/").then((r) => r.data);
