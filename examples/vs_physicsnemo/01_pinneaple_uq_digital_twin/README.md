# Exemplo 01 — UQ + Digital Twin em Tempo Real
## (Capacidades exclusivas do PINNeAPPle)

## O que este exemplo demonstra

| Capacidade | PINNeAPPle | PhysicsNeMo |
|---|---|---|
| Monte Carlo Dropout UQ | ✅ `pinneaple_uq.MCDropoutWrapper` | ❌ |
| Conformal Prediction (intervalos calibrados) | ✅ `pinneaple_uq.ConformalPredictor` | ❌ |
| Expected Calibration Error (ECE) | ✅ `pinneaple_uq.CalibrationMetrics` | ❌ |
| Digital twin com streams de sensores | ✅ `pinneaple_digital_twin.MockStream` | ❌ |
| Integração MQTT / Kafka / REST | ✅ `pinneaple_digital_twin.io.stream` | ❌ |
| Assimilação EKF / EnKF | ✅ `pinneaple_digital_twin.assimilation` | ❌ |
| Detecção de anomalias em tempo real | ✅ `pinneaple_digital_twin.monitoring` | ❌ |

## Problema físico

**Equação do Calor 1D**:

```
u_t = α · u_xx,   x ∈ [0,1], t ∈ [0,1]
u(0,t) = u(1,t) = 0     (Dirichlet)
u(x,0) = sin(πx)        (condição inicial)

Solução analítica: u(x,t) = exp(-π²αt) · sin(πx),  α = 0.1
```

## Por que UQ é crítico em engenharia industrial?

Sem UQ, uma predição de `u = 0.42` não diz **nada** sobre confiança.

Com UQ:
```
u = 0.42 ± 0.02   → sensor pode ser usado para decisão segura
u = 0.42 ± 0.18   → incerteza alta → acionar simulação de alta fidelidade
```

Exemplos de uso industrial onde UQ é obrigatório:
- Diagnóstico estrutural em turbinas (limite de fadiga com incerteza)
- Controle de reatores nucleares (temperatura com intervalo de confiança)
- Digital twin de plataformas offshore (fator de segurança = f(incerteza))

## Como o MC Dropout funciona

```python
# 1. Treinar PINN normalmente
model = HeatPINN()
train(model, n_epochs=3000)

# 2. Wrapping com MC Dropout (PINNeAPPle)
from pinneaple_uq import MCDropoutWrapper, MCDropoutConfig

uq_model = MCDropoutWrapper(model, MCDropoutConfig(n_samples=100, dropout_p=0.05))

# 3. Inferência com incerteza
result = uq_model.predict_with_uncertainty(x_test, n_samples=100)
# result.mean  → predição média
# result.std   → desvio padrão (incerteza)
lower, upper = result.confidence_interval(alpha=0.05)  # IC 95%
```

**Internamente**: roda 100 passes forward com dropout ativo → calcula média e std dos 100 outputs.

## Como o Digital Twin funciona

```
MockStream (simula sensor PIV/termopar)
     │
     ▼  observação: {x: 0.5, t: 0.3, T: 0.284}
DigitalTwin.update_loop()
     │
     ├── _torch_predict(domain_pts)    → campo de temperatura completo
     ├── _check_anomalies(obs)         → ThresholdDetector + ZScoreDetector
     └── _assimilation_update(obs)     → EKF atualiza estado interno
```

## Como executar

```bash
cd examples/vs_physicsnemo/01_pinneaple_uq_digital_twin
python example.py
```

**Output esperado:**
```
Device: cpu
Training Heat-equation PINN...
  epoch  200  loss=8.32e-03  pde=4.1e-04  bc=3.9e-04  ic=7.5e-04
  epoch  400  loss=2.14e-03  ...
  ...
[UQ] ECE: 0.0312
[UQ] Empirical coverage at 95% CI: 94.5%
[DT] Observations processed: 8
[DT] Anomalies detected: 3
[PLOT] Saved → heat_pinn_results.png
```

O arquivo `heat_pinn_results.png` contém:
- **A**: Curva de convergência do treinamento
- **B**: Predição média ± 2σ vs solução analítica em t=0.5
- **C**: |Erro| vs σ — bem calibrado quando error ≤ σ
- **D**: Campo completo u(x,t) predito
- **E**: Mapa de erro absoluto |predito − analítico|

## Interpretando os resultados de UQ

| Métrica | Significado | Valor típico bom |
|---|---|---|
| ECE | Erro de calibração esperado (0=perfeito) | < 0.05 |
| Cobertura 95% | Fração de pontos dentro do IC 95% | 93–97% |
| Sharpness | Média do std (menor = mais preciso) | < 0.02 para este problema |

## Código equivalente em PhysicsNeMo — não existe

```python
# PhysicsNeMo NÃO tem:
from modulus.uq import MCDropout       # ← ModuleNotFoundError
from modulus.digital_twin import ...   # ← ModuleNotFoundError
from modulus.monitoring import ...     # ← ModuleNotFoundError
```

Para ter UQ com PhysicsNeMo você precisaria implementar do zero ou usar uma biblioteca externa separada. No PINNeAPPle, tudo já está integrado.
