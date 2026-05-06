from pinneaple_tools.benchmark_suite.benchmark import PINNArenaBenchmark, all_model_specs

specs = all_model_specs()
print(f"Total models: {len(specs)}")
print(f"Total problems: {len(PINNArenaBenchmark.list_problems())}")

bench = PINNArenaBenchmark.default(
    problems=["burgers_1d"],
    models=["siren", "vanilla_pinn", "bench_fourier_mlp", "hamiltonian_nn"],
    epochs=30,
)
results = bench.run(verbose=False)
print(f"\nResults: {len(results)} runs")
for r in results:
    status = f"rel_l2={r.metrics.get('rel_l2', float('nan')):.2e}" if r.rank > 0 else "ERROR"
    print(f"  {r.model_id:<25s} {status}  {r.elapsed_s:.1f}s  params={r.n_params:,}")
