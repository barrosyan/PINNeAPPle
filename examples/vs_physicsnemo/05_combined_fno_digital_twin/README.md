# Exemplo 05 — Pipeline Combinado: PhysicsNeMo FNO + PINNeAPPle UQ + Digital Twin

## Este é o exemplo mais importante

Mostra como as duas ferramentas se complementam para criar um sistema de produção completo que **nenhuma das duas conseguiria sozinha**.

## Arquitetura do pipeline

```
Dados CFD (simulações, experimentos)
         │
         ▼
┌─────────────────────────────────────────┐
│  FASE 1: PhysicsNeMo FNO               │  ← PhysicsNeMo strength
│  • cuDNN kernels, fp16, multi-GPU       │
│  • Aprende (Re, x, y) → (u, v, p)      │
│  • Checkpoint salvo como nn.Module      │
└──────────────────┬──────────────────────┘
                   │  modelo = qualquer nn.Module
                   ▼
┌─────────────────────────────────────────┐
│  FASE 2: PINNeAPPle UQ                 │  ← PINNeAPPle exclusive
│  • MCDropoutWrapper (2 linhas)          │
│  • Predição com σ por ponto do domínio  │
│  • Calibração ECE verificada            │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  FASE 3: PINNeAPPle Digital Twin       │  ← PINNeAPPle exclusive
│  • Sensores PIV/manômetro via MockStream│
│  • Detecção de anomalias em tempo real  │
│  • EKF assimilation disponível          │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  FASE 4: PINNeAPPle Validação Física   │  ← PINNeAPPle exclusive
│  • Verifica ∇·u ≈ 0 (conservação massa)│
│  • Reporta se FNO prediz campo físico   │
└─────────────────────────────────────────┘
```

## Por que este split faz sentido?

| Tarefa | Ferramenta ideal | Motivo |
|---|---|---|
| Treinar surrogate em 100k samples | PhysicsNeMo | cuDNN, fp16, DDP |
| Inferir campo completo em <1ms | PhysicsNeMo | TensorRT, Triton |
| Quantificar incerteza da predição | PINNeAPPle | MCDropout, Conformal |
| Monitorar sensores de campo | PINNeAPPle | DigitalTwin, streams |
| Detectar comportamento anômalo | PINNeAPPle | AnomalyMonitor |
| Validar conservação de massa | PINNeAPPle | PhysicsValidator |
| Deploy REST API | PINNeAPPle | pinneaple_serve (FastAPI) |

## A chave: ambos usam `nn.Module`

O modelo PhysicsNeMo é um `nn.Module` padrão do PyTorch. O PINNeAPPle trabalha com qualquer `nn.Module`:

```python
# Fase 1: PhysicsNeMo treina
from physicsnemo.models.fno import FNO
model = FNO(in_channels=3, out_channels=3, ...)
train(model, cfd_data)                    # PhysicsNeMo faz isso bem
torch.save(model.state_dict(), "fno.pt")

# Fase 2: PINNeAPPle adiciona UQ
from pinneaple_uq import MCDropoutWrapper
uq_model = MCDropoutWrapper(model)        # 1 linha
result = uq_model.predict_with_uncertainty(x_test)
lower, upper = result.confidence_interval(alpha=0.05)

# Fase 3: PINNeAPPle digital twin
from pinneaple_digital_twin import DigitalTwin
dt = DigitalTwin(model=model, field_names=["u","v","p"])
dt.add_stream(mqtt_stream)
with dt:
    time.sleep(60)  # monitorando em tempo real
```

## Caso de uso industrial típico

**Digital twin de turbina eólica**:
1. PhysicsNeMo FNO: treina em 50.000 simulações CFD de escoamento ao redor de pá
2. PINNeAPPle UQ: quantifica incerteza em cada ponto do campo de pressão
3. PINNeAPPle DT: consome dados de 200 anemômetros em tempo real
4. PINNeAPPle Validate: verifica conservação de energia a cada atualização
5. PINNeAPPle Serve: expõe endpoint `/predict` para SCADA system

## Como executar

```bash
cd examples/vs_physicsnemo/05_combined_fno_digital_twin
python example.py
```

Output: `combined_pipeline_results.png` com:
- Campos u, v, p preditos pelo FNO
- Mapa de incerteza σ(u) por MC Dropout
- Timeline de anomalias detectadas
- Mapa de |∇·u| (validação de incompressibilidade)
- Diagrama ASCII do pipeline
- Resumo de todas as 4 fases
