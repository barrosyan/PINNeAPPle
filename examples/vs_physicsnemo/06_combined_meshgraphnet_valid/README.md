# Exemplo 06 — Pipeline Combinado: PhysicsNeMo MeshGraphNet + PINNeAPPle Validação + Export

## O que este exemplo demonstra

O pipeline mais próximo de um sistema CFD de produção real:

```
Malha CFD não-estruturada (aerofólio NACA 0012)
         │
         ▼
┌─────────────────────────────────────────┐
│  PhysicsNeMo MeshGraphNet              │  ← PhysicsNeMo strength
│  • Message Passing GNN em malha real   │
│  • cuDNN graph kernels, DDP multi-GPU  │
│  • Suporte a 1M+ nós via mini-batch    │
│  • Prediz (u, v, p) em cada nó         │
└──────────────────┬──────────────────────┘
                   │  pred = nn.Module padrão
                   ▼
┌─────────────────────────────────────────┐
│  PINNeAPPle PhysicsValidator           │  ← PINNeAPPle exclusive
│  • ∇·u ≈ 0  (conservação de massa)    │
│  • No-slip BC: u=v=0 no aerofólio      │
│  • Inlet BC verificado                 │
│  • Simetria de pressão (AoA=0°)        │
│  • Pressão de stagnação validada       │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  PINNeAPPle pinneaple_export           │  ← PINNeAPPle exclusive
│  • export_torchscript() → .pt          │
│  • export_onnx()        → .onnx        │
│  • Verifica fidelidade orig vs exported│
│  • Dynamic axes: qualquer malha        │
└─────────────────────────────────────────┘
```

## Por que MeshGraphNet para CFD?

### Quando usar PhysicsNeMo MeshGraphNet

| Situação | MeshGraphNet | FNO |
|---|---|---|
| Geometria complexa (aerofólio, turbina) | ✅ Ideal | ❌ Só grades regulares |
| Malha não-estruturada com 100k+ nós | ✅ Nativo | ❌ Não suporta |
| Refinamento local (boundary layer) | ✅ kNN preservado | ❌ Perde refinamento |
| Mudança de topologia entre samples | ✅ Qualquer grafo | ❌ Mesma grade |
| Throughput máximo em grade regular | ❌ GNN overhead | ✅ FFT batch rápido |

### Diferença de arquitetura

```
FNO:  grade regular  →  FFT  →  spectral conv  →  campo
MGN:  grafo qualquer →  edge MLP  →  node MLP  →  campo por nó
```

O MeshGraphNet aprende a **propagar informação pelo grafo** da malha. Nós próximos à fronteira "comunicam" o valor da BC para nós interiores via message passing — exatamente como acontece no solver numérico.

## Por que validar o GNN?

GNNs treinados em dados CFD podem:
- **Violar conservação de massa** (∇·u ≠ 0) — campo fisicamente impossível
- **Ignorar BCs** — velocidade não-nula na parede em casos de extrapolação
- **Distribuição de pressão incorreta** — Bernoulli violado

O `PhysicsValidator` do PINNeAPPle detecta esses problemas **antes do deploy**:

```python
from pinneaple_validate import PhysicsValidator, ConservationCheck, BoundaryCheck

validator = PhysicsValidator([
    ConservationCheck("divergence", threshold=0.05),
    BoundaryCheck("noslip", boundary_mask=airfoil_mask),
    BoundaryCheck("inlet", expected={"u": 1.0, "v": 0.0}),
])
report = validator.validate(model, mesh_data)
# → ValidationReport com score, detalhes por check, sugestões de retreino
```

## Por que exportar para TorchScript/ONNX?

| Formato | Caso de uso |
|---|---|
| TorchScript | Inferência em C++ (libtorch), sem Python |
| ONNX | ONNX Runtime, TensorRT, deploy multiplataforma |
| Original `.pt` | Só Python, com PyTorch instalado |

**Exemplo de uso do TorchScript em C++:**
```cpp
#include <torch/script.h>

auto model = torch::jit::load("mgn_airfoil.torchscript.pt");
auto output = model.forward({node_feats, edge_index, edge_feats}).toTensor();
// Retorna campos (u, v, p) para todos os nós — sem Python, em microsegundos
```

O PINNeAPPle verifica que `max(|output_orig − output_exported|) < 1e-5` antes de considerar o export válido.

## A chave: PhysicsNeMo treina, PINNeAPPle valida e exporta

```python
# PhysicsNeMo: treina em malhas reais com performance máxima
from physicsnemo.models.meshgraphnet import MeshGraphNet
model = MeshGraphNet(node_in=7, edge_in=4, hidden=128, out_fields=3)
train_distributed(model, cfd_dataset)          # DDP, cuDNN, fp16

# PINNeAPPle: valida física — sem alterar o modelo
from pinneaple_validate import PhysicsValidator
report = PhysicsValidator.from_config("incompressible_ns").validate(model, mesh)
if report.score < 0.8:
    raise ValueError(f"Modelo não passou validação: {report.failed_checks}")

# PINNeAPPle: exporta para produção
from pinneaple_export import export_torchscript, export_onnx
export_torchscript(model, example_inputs, "mgn.pt", verify=True)
export_onnx(model, example_inputs, "mgn.onnx", dynamic_axes={"N": 0, "E": 0})
```

## Decisão de split

| Tarefa | Ferramenta ideal | Motivo |
|---|---|---|
| GNN em malha com 500k nós | PhysicsNeMo | CUDA graph kernels, DDP automático |
| Training com data from OpenFOAM | PhysicsNeMo | Integração nativa com solvers NVIDIA |
| Verificar ∇·u = 0 na predição | PINNeAPPle | `ConservationCheck` pronto |
| Detectar violação de BC no-slip | PINNeAPPle | `BoundaryCheck` pronto |
| Export TorchScript com verificação | PINNeAPPle | `export_torchscript(verify=True)` |
| Deploy C++ sem Python | PINNeAPPle | ONNX + TorchScript gerados |

## Como executar

```bash
cd examples/vs_physicsnemo/06_combined_meshgraphnet_valid
python example.py
```

**Output esperado:**
```
[INFO] PhysicsNeMo não instalado — usando implementação de referência

[1/4] Gerando malha não-estruturada (aerofólio NACA 0012)...
  Nós: 734  |  Arestas: 5,630
  Tipos: 441 interior  |  80 aerofólio  |  153 farfield  |  60 outlet

[2/4] Treinando MeshGraphNet...
  Nós: 734  |  Arestas: 5,630  |  Parâmetros: 214,275
    epoch  100  loss=2.3e-02  12.4s
    epoch  200  loss=8.1e-03  24.1s
    epoch  300  loss=3.2e-03  36.2s

  RMSE — u: 0.0421  |  v: 0.0318  |  p: 0.0284

[3/4] Validação física (PINNeAPPle PhysicsValidator)...

═══════════════════════════════════════════════════════
  RELATÓRIO DE VALIDAÇÃO FÍSICA  (pinneaple_validate)
═══════════════════════════════════════════════════════
  [✓] PASS  Conservação de massa (∇·u≈0)
      div médio = 0.0312  (87% nós OK)

  [✓] PASS  No-slip BC (u=v=0 no aerofólio)
      erro médio = 0.0180 m/s

  [✓] PASS  Inlet BC (u=1, v=0)
      Δu = 0.0421  Δv = 0.0198

  [✓] PASS  Simetria de pressão (AoA=0°)
      erro simetria = 0.0312 Pa

  [✓] PASS  Pressão de stagnação
      p_stag = 0.4821 Pa

  Score: 5/5  [████████████████████]  100%

[4/4] Exportando modelo (PINNeAPPle pinneaple_export)...
  [✓] TorchScript salvo: mgn_airfoil.torchscript.pt  (831.2 KB)
       Diferença máxima orig vs scripted: 2.38e-07
  [!] ONNX não disponível (pip install onnx). Pulando.
```

Output: `meshgraphnet_valid_results.png` com:
- Campos u, v, p analíticos e preditos pelo MeshGraphNet
- Mapa de erro por campo
- Tipos de nós da malha não-estruturada
- Barras de resultado da validação física
- Status de export com tamanho e fidelidade
- Diagrama ASCII do pipeline completo

## Instalação opcional

```bash
# PhysicsNeMo (para MeshGraphNet real com cuDNN)
pip install nvidia-physicsnemo

# ONNX export
pip install onnx onnxruntime

# SciPy (já usado para kNN da malha)
pip install scipy
```
