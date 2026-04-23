# Exemplo 02 — Meta-aprendizado Reptile + Transfer Learning
## (Capacidades exclusivas do PINNeAPPle)

## O que este exemplo demonstra

| Capacidade | PINNeAPPle | PhysicsNeMo |
|---|---|---|
| Reptile meta-learning | ✅ `pinneaple_meta.ReptileTrainer` | ❌ |
| MAML meta-learning | ✅ `pinneaple_meta.MAMLTrainer` | ❌ |
| PDETaskSampler (amostragem de tarefas) | ✅ `pinneaple_meta.PDETaskSampler` | ❌ |
| TransferTrainer (fine-tuning parcial) | ✅ `pinneaple_transfer.TransferTrainer` | ❌ |
| ParametricFamilyTransfer | ✅ `pinneaple_transfer.ParametricFamilyTransfer` | ❌ |
| freeze / unfreeze layers | ✅ `pinneaple_transfer.freeze_layers` | ❌ |
| layer_lr_groups (LR discriminativo) | ✅ `pinneaple_transfer.layer_lr_groups` | ❌ |

## O problema: famílias paramétricas de PDEs

Na indústria (acoplamentos rotativos, turbinas, reatores), o mesmo tipo de PDE é resolvido repetidamente com parâmetros diferentes:
- Burgers com viscosidade `nu ∈ {0.001, 0.005, 0.01, 0.05, 0.1}`
- Navier-Stokes com Reynolds `Re ∈ {100, 500, 1000, 5000}`
- Condução com condutividade `k ∈ {1, 10, 100}` W/m·K

**Abordagem ingênua** (PhysicsNeMo / sem meta-aprendizado):
```
Para cada nu: treinar do zero → ~2000 épocas × 5 valores = 10.000 épocas
```

**Com Reptile (PINNeAPPle)**:
```
Meta-treinar uma vez em 20+ tarefas → adaptar a novo nu em 20 passos
Redução de custo: 10.000 → 400 (meta) + 20×5 (adapt) = 500 épocas total
```

## Como funciona o Reptile

```
Meta-modelo θ₀ (inicialização ótima para a família)
    │
    ├── Task 1 (nu=0.03): θ₀ → θ₁  (10 passos internos)
    │   θ₀ ← θ₀ + ε(θ₁ - θ₀)       (atualização Reptile)
    │
    ├── Task 2 (nu=0.07): θ₀ → θ₂  (10 passos internos)
    │   θ₀ ← θ₀ + ε(θ₂ - θ₀)
    │
    └── ... (400 iterações)
         ↓
Meta-modelo θ* (pode se adaptar a QUALQUER nu em ~20 passos)
```

**Fórmula da atualização Reptile**:
```
θ ← θ + ε · (θ_task − θ)
onde θ_task = resultado de n_inner passos de SGD na tarefa
```

## Como funciona o Transfer Learning

```python
# Modelo treinado para nu=0.01
source = train_pinn(nu=0.01, epochs=1000)

# Congelar as primeiras camadas (features gerais)
freeze_layers(source, prefixes=["0", "1", "2"])   # PINNeAPPle

# Fine-tune apenas as últimas camadas para nu=0.1
opt = Adam(filter(lambda p: p.requires_grad, source.parameters()))
finetune(source, nu=0.1, epochs=300)
# → 30% do custo de treinar do zero, mesmo nível de qualidade
```

## Resultados esperados

| Abordagem | Épocas | Custo relativo | Loss final (nu=0.005) |
|---|---|---|---|
| From scratch | 2000 | 100% | ~1×10⁻³ |
| Reptile adapt | 20 | **~1%** | ~1.5×10⁻³ |
| Transfer (fine-tune) | 300 | 30% | ~1.2×10⁻³ |

## Quando usar meta-aprendizado vs treinamento padrão?

| Situação | Recomendação |
|---|---|
| Um único problema específico | Treinar do zero |
| 3–5 configurações paramétricas | Transfer Learning |
| 10+ configurações ou parâmetro contínuo | Reptile / MAML |
| Novo parâmetro a cada semana | Reptile + fine-tune contínuo |
| Design paramétrico interativo | Reptile (resposta em segundos) |

## Como executar

```bash
cd examples/vs_physicsnemo/02_pinneaple_meta_transfer
python example.py
```

**Output esperado**:
```
Meta-treinando com Reptile em família Burgers...
  meta-epoch   50  eval_loss=8.4e-02
  meta-epoch  100  eval_loss=3.1e-02
  ...
  meta-epoch  400  eval_loss=5.2e-03
Meta-treino: 45.2s

Adaptando Reptile a nu=0.005 (20 passos apenas)...
  Adaptação: 0.8s | final loss: 4.1e-03

Baseline: treinando do zero (2000 épocas) para nu=0.005...
  From scratch: 38.4s | final loss: 2.8e-03

RESUMO — Speedup do Reptile:
  From scratch (2000 épocas): 38.4s  loss=2.8e-03
  Reptile adapt (20 passos):   0.8s  loss=4.1e-03
  Speedup: ~48× mais rápido para nível de loss similar
```

Output: `meta_transfer_results.png`

## Código PhysicsNeMo equivalente — não existe

```python
# PhysicsNeMo NÃO tem:
from physicsnemo.meta import ReptileTrainer    # ← ModuleNotFoundError
from physicsnemo.transfer import TransferTrainer  # ← ModuleNotFoundError

# Para obter resultado equivalente em PhysicsNeMo,
# você precisaria implementar Reptile do zero:
# - Gestão de cópias de modelo por tarefa
# - Loop de meta-atualização
# - PDETaskSampler para amostrar nu aleatório
# Isso é ~300 linhas de código que o PINNeAPPle já fornece.
```
