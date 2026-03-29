# Exemplo 03 — Active Collocation (RAD) + SA-PINN Weight Scheduling
## (Capacidades exclusivas do PINNeAPPle)

## O que este exemplo demonstra

| Capacidade | PINNeAPPle | PhysicsNeMo |
|---|---|---|
| Residual-based Adaptive Collocation (RAD) | ✅ `pinneaple_data.ResidualBasedAL` | ❌ |
| Residual-based Adaptive Refinement (RAR) | ✅ `pinneaple_data.ResidualBasedAL` | ❌ |
| Variance-based Active Learning | ✅ `pinneaple_data.VarianceBasedAL` | ❌ |
| AdaptiveCollocationTrainer | ✅ | ❌ |
| SA-PINN (self-adaptive weights) | ✅ `WeightScheduler(method="self_adaptive")` | ❌ |
| GradNorm balancing | ✅ `WeightScheduler(method="gradnorm")` | ❌ |
| NTK-based weight balancing | ✅ `WeightScheduler(method="ntk")` | ❌ |
| LossRatio balancing | ✅ `WeightScheduler(method="loss_ratio")` | ❌ |

## Problema

**Equação de Laplace 2D**: `u_xx + u_yy = 0` no quadrado unitário com BC não-trivial:
```
u(x,0) = sin(πx)   ← interessante: cria gradiente alto perto de y=0
u(x,1) = 0
u(0,y) = u(1,y) = 0
Solução: u(x,y) = sin(πx) · sinh(π(1-y)) / sinh(π)
```

## Por que collocação uniforme falha aqui?

A solução tem gradientes muito maiores perto de `y=0` do que em `y=1`. Com pontos uniformes, a maioria dos pontos cai onde o resíduo é baixo — desperdiçando capacidade computacional.

```
Uniforme:  ░░░░░░░░░░░░  (pontos espalhados uniformemente)
           ░░░░░░░░░░░░
           ░░░░░░░░░░░░  ← y≈0: alto gradiente, poucos pontos aqui
           ▓▓▓▓▓▓▓▓▓▓▓▓  BC: sin(πx)

RAD:       ░░░░░░░░░░░░  (pontos onde resíduo é baixo)
           ░░░░░░░░░░░░
           ▓▓▓▓▓▓▓▓▓▓▓▓  ← RAD concentra aqui (alto resíduo)
           ▓▓▓▓▓▓▓▓▓▓▓▓  BC: sin(πx)
```

## Como o RAD funciona

```
1. Amostrar N_candidates = 10 × N pontos uniformemente (pool de candidatos)
2. Calcular |resíduo de PDE| em cada candidato via autograd
3. Probabilidade de seleção ∝ |resíduo|   ← ponto difícil → mais chance
4. Amostrar N pontos do pool com essa distribuição
5. Repetir a cada 500 épocas
```

**Fórmula RAD**:
```
p_i = |R(x_i)| / Σ|R(x_j)|
```

## Como o SA-PINN funciona

```python
# Pesos λ_pde, λ_bc são parâmetros da rede (não hiperparâmetros fixos)
λ_pde = exp(log_λ_pde)   # sempre positivo
λ_bc  = exp(log_λ_bc)

# Loss total
L = λ_pde · L_pde + λ_bc · L_bc

# Atualização DUPLA:
# 1. Modelo: descent em L (quer minimizar)
model_opt.step()

# 2. Pesos: ASCENT em L (quer maximizar — os pesos crescem onde é difícil)
for p in sa_weights.parameters():
    p.grad.neg_()  # ← inverte gradiente = ascent
weight_opt.step()
```

**Resultado**: λ_bc cresce quando L_bc > L_pde → modelo foca mais nas BCs.

## Resultados esperados

```
Baseline (uniforme + fixos):   RMSE ≈ 0.015
PINNeAPPle (RAD + SA-PINN):    RMSE ≈ 0.006
Redução de erro: ~60% com o MESMO número de pontos
```

## Quando usar cada método de pesos?

| Método | Quando usar | Custo extra |
|---|---|---|
| `fixed` | Você já sabe os pesos certos | Nenhum |
| `self_adaptive` | Padrão — funciona bem sempre | Baixo (+1 optimizer step) |
| `loss_ratio` | Você quer controlar ratios específicas | Muito baixo |
| `gradnorm` | Múltiplas tarefas com magnitudes díspares | Moderado |
| `ntk` | Máxima fundamentação teórica | Alto (backward por perda) |

## Como executar

```bash
cd examples/vs_physicsnemo/03_pinneaple_active_weight_sched
python example.py
```

**Output esperado**:
```
[1/2] Baseline — collocação UNIFORME + pesos FIXOS (2000 épocas)...
      RMSE final: 0.0147
[2/2] PINNeAPPle — collocação RAD + SA-PINN (2000 épocas)...
      RMSE final: 0.0058
REDUÇÃO DE ERRO: 60.5%  (RAD + SA-PINN vs uniforme + fixo)
```

Output: `active_weight_results.png` com:
- Curvas de convergência comparadas
- RMSE vs analítica ao longo do treino
- Distribuição de pontos antes e depois do RAD
- Evolução dos pesos SA-PINN λ_pde e λ_bc
- Campo de erro comparativo
