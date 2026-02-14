# scripts/ — Runbook do MVP (Flow Obstacle 2D)

Este diretório contém scripts executáveis para rodar o MVP de escoamento 2D com obstáculo
usando o pipeline:

1) (Opcional) Cosmos → gerar spec e variações de cenário
2) (Opcional) Omniverse → gerar USD / amostrar pontos / exportar bundle
3) Local → validar bundle + rodar PINNArena (native e/ou PhysicsNeMo)
4) Local → gerar leaderboard e sumarizar runs

## Pré-requisitos (local)
- Python 3.10+ (recomendado)
- Pacotes do seu repo instalados (venv + `pip install -e .`)
- `pyarrow` e `pandas` (para ler/escrever parquet)
- `torch` (para backend nativo)
- (Opcional) `nvidia-physicsnemo-sym` para backend PhysicsNeMo

## Pré-requisitos (externo)
### Omniverse
Scripts em `scripts/omniverse/` rodam DENTRO do Omniverse Kit (Omniverse Code / Isaac Sim).
Eles dependem do runtime do Kit (USD/omni.*).

### Cosmos
Scripts em `scripts/cosmos/` assumem que você tem um endpoint HTTP para um modelo Cosmos
ou um gateway interno. O script funciona com um endpoint JSON (ver `scripts/cosmos/README.md`).

> Importante: se você não tiver Cosmos disponível agora, use o spec estático em
`configs/problemdesign/flow_obstacle_2d_spec_example.yaml` e pule a pasta cosmos.

## Caminho rápido (sem Cosmos e sem Omniverse)
1) Garanta que existem:
   - data/bundles/flow_obstacle_2d/v0/bundle/conditions.json
   - data/bundles/flow_obstacle_2d/v0/bundle/manifest.json
   - data/bundles/flow_obstacle_2d/v0/derived/points_boundary.parquet
   - data/bundles/flow_obstacle_2d/v0/derived/points_collocation.parquet

2) Valide o bundle:
   python scripts/mvp/validate_bundle.py

3) Rode Arena (native):
   python scripts/arena/run_benchmark.py --run configs/arena/runs/vanilla_pinn_native.yaml

4) Veja leaderboard:
   python scripts/arena/leaderboard.py

## Caminho completo (Cosmos + Omniverse + Arena)
A) Cosmos:
   python scripts/cosmos/scene_to_spec.py --scene configs/problemdesign/flow_obstacle_2d_prompt.yaml --out data/cache/cosmos_outputs/spec.yaml
   python scripts/cosmos/generate_variations.py --spec data/cache/cosmos_outputs/spec.yaml --out data/cache/cosmos_outputs/variations.json

B) Omniverse (rodar dentro do Kit):
   - Veja `scripts/omniverse/README.md`
   - Rode `export_bundle_from_stage.py` para produzir o bundle

C) Arena:
   python scripts/mvp/run_arena_matrix.py

## Saídas
- Runs: data/artifacts/runs/<run_id>/*
- Leaderboard: data/artifacts/leaderboard.json
