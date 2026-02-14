# configs/ — MVP Flow Obstacle 2D

Este diretório contém as configurações do MVP para o problema:
**Navier–Stokes 2D incompressível (steady) em canal com obstáculo**.

## Como essas configs são usadas

### 1) Bundle (dados do problema)
Você precisa de um bundle em `data/bundles/flow_obstacle_2d/v0/` com:

- bundle/geometry.(usd|stl)
- bundle/conditions.json
- bundle/manifest.json
- derived/points_boundary.parquet
- derived/points_collocation.parquet
- opcional: bundle/sensors.parquet

A validação do bundle segue:
- `configs/data/bundle_schema.yaml`

> Geração do bundle: via Omniverse export (se usar USD) ou via pipeline STL/mesh no seu `pinneaple_geom`.
> Isso não é gerado por configs; é gerado por scripts/integrações.

### 2) Arena (benchmark)
- `configs/arena/tasks/flow_obstacle_2d.yaml` aponta para o bundle_root.
- `configs/arena/runs/*.yaml` define backend + hiperparâmetros.
- `configs/arena/backends/*.yaml` são presets (opcionais).

Rodando:
```python
from pinneaple_arena.runner import run_benchmark

res = run_benchmark(
  artifacts_dir="data/artifacts",
  task_cfg_path="configs/arena/tasks/flow_obstacle_2d.yaml",
  run_cfg_path="configs/arena/runs/vanilla_pinn_native.yaml",
  bundle_schema_path="configs/data/bundle_schema.yaml",
)
print(res["metrics"])
3) Problem Design (Cosmos-in-the-loop)

configs/problemdesign/flow_obstacle_2d_prompt.yaml define o prompt e regras.

configs/problemdesign/flow_obstacle_2d_spec_schema.yaml define o schema do spec.

configs/problemdesign/flow_obstacle_2d_spec_example.yaml é um exemplo válido do spec.

Mesmo sem Cosmos, você pode usar o exemplo para gerar manifest/conditions.

Nota importante sobre inlet parabólico

O MVP aqui usa inlet constante (u=1, v=0).
Se você quiser inlet parabólico, faça de modo determinístico:

inclua no manifest.json os parâmetros do perfil (u_max, y0, y1),
OU

exporte u_target/v_target por ponto do inlet em points_boundary.parquet.

Sem isso, o solver não deve “inventar” o perfil.

--------------

passo a passo completo para você gerar a geometria e os derived points do MVP (canal 2×1 com obstáculo circular em (0.7,0.5), r=0.15), nos dois caminhos:

A) Omniverse (USD) — recomendado se você quer “simulação/visual” e pipeline NVIDIA

B) CadQuery/STL + Python — recomendado se você quer independência e rapidez

No fim, você terá exatamente:

data/bundles/flow_obstacle_2d/v0/
  bundle/
    geometry.usd  (ou geometry.stl)
    conditions.json
    manifest.json
  derived/
    points_boundary.parquet
    points_collocation.parquet

A) Omniverse (USD) — passo a passo completo
A1) Criar a cena (canal + obstáculo)

Abra Omniverse Code (ou Isaac Sim).

Crie um novo stage:

File → New

Garanta que você está trabalhando em metros (não precisa ser perfeito no MVP, mas mantenha consistente).

Crie o canal como um plano/retângulo (no plano XY):

Você pode criar um Plane e ajustar escala para representar o retângulo do domínio.

Domínio do MVP:

x ∈ [0, 2]

y ∈ [0, 1]

Dica prática (simples para MVP):
Use um plano no centro e trate o “domínio” como bounding box lógico. O que importa pro PINN é o domain no manifest.json e a amostragem de pontos.

Crie o obstáculo:

Create → Mesh → Cylinder

Ajuste:

raio = 0.15

altura pequena (ex: 0.01) só pra existir como mesh

Posicione o cilindro no plano XY em:

(x=0.7, y=0.5)

Salve o arquivo:

flow_obstacle_2d.usd

A2) Marcar as regiões (semantics)

Você precisa marcar 4 regiões: inlet, outlet, walls, obstacle.

A2.1 Habilitar ferramentas de Semantics

Window → Extensions

Habilite:

Semantics Schema Editor

omni.replicator.core (útil para tooling)

A2.2 Criar prims para as bordas do canal

Você precisa de prims “representando” cada fronteira:

inlet: linha/face em x=0

outlet: linha/face em x=2

walls: y=0 e y=1

obstacle: o cilindro

Jeito MVP (funciona e é fácil):

Crie 4 meshes “fininhas” (thin planes/boxes) nas bordas:

inlet: um plano fino em x=0

outlet: um plano fino em x=2

wall_bottom: plano fino em y=0

wall_top: plano fino em y=1

Agrupe wall_bottom + wall_top dentro de um Xform “walls”.

Isso facilita o export, porque cada região fica um prim claro.

A2.3 Aplicar semantics labels

Abra o Semantics Schema Editor e aplique:

No prim inlet: label inlet

No prim outlet: label outlet

No prim walls (grupo ou cada um): label walls

No obstáculo (cilindro): label obstacle

Salve o stage.

A3) Exportar bundle + derived points dentro do Omniverse

No Script Editor do Omniverse, rode o export (o seu pipeline):

Copie seu script export (o que você já tem no repo pinneaple_integrations/omniverse/export_bundle_from_usd.py).

Rode com:

from pinneaple_integrations.omniverse import export_flow_bundle_from_usd

export_flow_bundle_from_usd(
    usd_path="C:/cenas/flow_obstacle_2d.usd",
    out_bundle_dir="C:/PINNeAPPle/data/bundles/flow_obstacle_2d/v0",
    domain_xy=((0.0, 2.0), (0.0, 1.0)),
    obstacle_circle=(0.7, 0.5, 0.15),
    nu=0.01,
    n_boundary=20000,
    n_collocation=200000,
    seed=0
)

Resultado esperado

bundle/geometry.usd (cópia/export do stage)

derived/points_boundary.parquet com colunas x,y,region

derived/points_collocation.parquet com colunas x,y

A4) Copiar os JSONs do bundle

Agora coloque os arquivos que eu te passei em:

data/bundles/flow_obstacle_2d/v0/bundle/conditions.json

data/bundles/flow_obstacle_2d/v0/bundle/manifest.json

Pronto: você tem um bundle completo.

B) CadQuery/STL + Python (sem Omniverse) — passo a passo completo

Esse caminho é “direto ao ponto” e te deixa independente.

B1) Gerar a geometria em STL com CadQuery

Instale cadquery:

pip install cadquery


Gere um STL do obstáculo (e opcionalmente do domínio).
Como seu domínio é um retângulo simples e o obstáculo é um círculo, no MVP você pode salvar só o obstáculo como STL e usar manifest.domain como domínio lógico.

Opção simples recomendada: obstáculo STL

Crie um cilindro com raio 0.15 e altura 0.01

Salve como geometry.stl

Se você quiser também o canal como mesh, dá pra criar uma placa (2×1×0.01). Mas o PINN não precisa disso; ele só precisa do domínio e do círculo.

Salve STL em:

data/bundles/flow_obstacle_2d/v0/bundle/geometry.stl

Ajuste o schema se você quiser validar STL (em bundle_schema.yaml):

troque bundle/geometry.usd por bundle/geometry.stl

B2) Gerar points_collocation.parquet (interior)

Você vai amostrar uniformemente no retângulo e rejeitar os pontos dentro do círculo.

Regras:

domínio: x ∈ [0,2], y ∈ [0,1]

obstáculo: (0.7,0.5), r=0.15

n_collocation: 200000

Saída:

colunas x, y

B3) Gerar points_boundary.parquet (fronteiras rotuladas)

Você vai gerar pontos nas 4 fronteiras:

inlet: x=0, y ~ Uniform(0,1)

outlet: x=2, y ~ Uniform(0,1)

walls:

y=0, x ~ Uniform(0,2)

y=1, x ~ Uniform(0,2)

obstacle:

(x,y) no círculo: x=cx+r cosθ, y=cy+r sinθ, θ~Uniform(0,2π)

Saída:

colunas x, y, region

Com region exatamente em:

"inlet", "outlet", "walls", "obstacle"

Distribuição recomendada (total 20000):

inlet: 5000

outlet: 5000

walls: 5000 (2500 em y=0 e 2500 em y=1)

obstacle: 5000

B4) Salvar em parquet

Instale pyarrow:

pip install pyarrow pandas


Salve:

data/bundles/flow_obstacle_2d/v0/derived/points_collocation.parquet

data/bundles/flow_obstacle_2d/v0/derived/points_boundary.parquet

B5) Adicionar os JSONs do bundle

Coloque os JSONs que eu escrevi em:

data/bundles/flow_obstacle_2d/v0/bundle/conditions.json

data/bundles/flow_obstacle_2d/v0/bundle/manifest.json

Pronto: bundle completo sem Omniverse.

Checklist final (vale pros dois caminhos)

Arquivos existem?

bundle/conditions.json

bundle/manifest.json

derived/points_collocation.parquet

derived/points_boundary.parquet

points_boundary.parquet tem:

region com valores: inlet/outlet/walls/obstacle

manifest.json tem:

nu

domain.x e domain.y

weights

conditions.json bate com o MVP:

inlet u=1 v=0

outlet p=0

walls/obstacle no-slip