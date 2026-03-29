# Exemplo 04 — FNO Operator Learning (Forças do PhysicsNeMo)

## O que este exemplo demonstra

O PhysicsNeMo tem vantagem real quando o objetivo é **throughput máximo em hardware NVIDIA**:

| Feature | PhysicsNeMo FNO | Implementação de referência |
|---|---|---|
| Arquitetura FNO | Idêntica | Idêntica |
| cuDNN kernels otimizados | ✅ | ❌ |
| NVIDIA Transformer Engine | ✅ | ❌ |
| fp16/bf16 automático | ✅ | Manual |
| Multi-GPU com DDP nativo | ✅ | DataParallel básico |
| Multi-nó (NCCL) | ✅ | Manual |
| TensorRT export | ✅ | ❌ |
| Triton inference server | ✅ | ❌ |
| Speedup típico | **2–5×** | baseline |
| UQ integrado | ❌ → usar PINNeAPPle | — |
| Digital Twin | ❌ → usar PINNeAPPle | — |

## O que é um operador neural (FNO)?

Um PINN aprende **uma solução específica** u(x,t) para parâmetros fixos.

Um FNO aprende **o operador** que mapeia qualquer condição inicial/fronteira para a solução:

```
PINN:  fixa nu=0.01  →  aprende u(x,t) para esses parâmetros específicos
FNO:   aprende  G: u₀(x) → u(x,T)  para QUALQUER u₀

Uma vez treinado, o FNO responde em < 1ms vs horas de simulação numérica.
```

## Modelos disponíveis no PhysicsNeMo

```python
from physicsnemo.models.fno import FNO          # Fourier Neural Operator
from physicsnemo.models.afno import AFNO        # Adaptive FNO (mais eficiente)
from physicsnemo.models.sfno import SFNO        # Spherical FNO (clima global)
from physicsnemo.models.meshgraphnet import MeshGraphNet  # GNN em malhas
from physicsnemo.models.graphcastnet import GraphCastNet  # Previsão climática
from physicsnemo.models.dlwp import DLWP        # Deep Learning Weather Prediction
```

## Multi-GPU com PhysicsNeMo

```python
# PhysicsNeMo abstrai o DDP para você:
from physicsnemo.distributed import DistributedManager

dist = DistributedManager()
model = FNO(...)
if dist.distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model.to(dist.device),
        device_ids=[dist.local_rank],
    )
```

## Instalação

```bash
# PhysicsNeMo
pip install nvidia-physicsnemo

# Com suporte GPU completo (recomendado)
pip install nvidia-physicsnemo[all]
```

O exemplo detecta automaticamente se PhysicsNeMo está instalado e usa a implementação de referência como fallback.

## Como executar

```bash
cd examples/vs_physicsnemo/04_physicsnemo_fno_operator
python example.py
```

**Output esperado**:
```
[INFO] PhysicsNeMo não instalado — usando implementação de referência
       Para usar PhysicsNeMo: pip install nvidia-physicsnemo

[1/3] Gerando dataset Burgers (500 pares IC→solução)...
[2/3] Construindo e treinando FNO operador...
  epoch  100  train=8.4e-03  test=9.1e-03  12.4s
  epoch  200  train=3.2e-03  test=3.8e-03  24.1s
  ...
  Throughput: 4320 samples/s
  Test MSE:   1.8e-04

FNO aprendeu o operador: u₀ → u(·,T=1)
Uma vez treinado, mapeia qualquer IC para a solução em <1ms
```

Output: `fno_operator_results.png`

## Próximo passo

Ver **Exemplo 05** — como pegar este modelo FNO e adicionar UQ + Digital Twin via PINNeAPPle (pipeline combinado).
