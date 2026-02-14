# scripts/omniverse — executar dentro do Omniverse Kit

Esses scripts rodam DENTRO do Omniverse Code / Isaac Sim.

## Objetivo
Gerar o bundle do MVP:
data/bundles/flow_obstacle_2d/v0/
  bundle/geometry.usd
  bundle/conditions.json
  bundle/manifest.json
  derived/points_boundary.parquet
  derived/points_collocation.parquet

Os JSONs (conditions/manifest) você já pode criar localmente (ver scripts/mvp/build_bundle_jsons.py)
e depois copiar para a pasta bundle/ se preferir.

## Passo a passo (Kit)
1) Abra Omniverse Code (ou Isaac Sim).
2) Abra o Stage USD do cenário (canal + obstáculo).
3) Garanta que existem prims com labels semânticos:
   - inlet / outlet / walls / obstacle
   Use Semantics Schema Editor para aplicar as labels.
4) No Script Editor do Kit:
   - aponte o sys.path para o seu repo (para importar pinneaple_integrations)
   - rode `export_bundle_from_stage.py`

## Como importar o repo dentro do Kit
No Script Editor, antes de rodar:
```python
import sys
sys.path.append("C:/PINNeAPPle")  # ajuste para o caminho do seu repo

Execução

generate_scenarios.py: duplica uma cena base para criar variações (USDs).

export_bundle_from_stage.py: exporta bundle + derived points.

Observações importantes

Este script assume que pinneaple_integrations.omniverse.export_flow_bundle_from_usd
existe no seu repo (na parte de integrações).

Se você ainda não tiver esse export no repo, você pode:
(A) exportar manualmente pontos e salvar parquets, ou
(B) implementar o export dentro de pinneaple_integrations/omniverse.