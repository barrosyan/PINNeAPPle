# PINNeAPPle — Visão Geral da Biblioteca

> **P**hysics-**I**nformed **N**eural **Ne**tworks **A**pplication & **P**hysics **P**roblem **le**arning
>
> Framework end-to-end para redes neurais informadas por física (PINNs), operadores neurais, gêmeos digitais e otimização de design em engenharia.

---

## Índice

1. [Arquitetura Geral](#1-arquitetura-geral)
2. [pinneapple\_physics — Definição de Problemas Físicos](#2-pinneapple_physics--definição-de-problemas-físicos)
3. [pinneapple\_neural — Arquiteturas e Treinamento](#3-pinneapple_neural--arquiteturas-e-treinamento)
4. [pinneapple\_design — Geometria e Otimização de Design](#4-pinneapple_design--geometria-e-otimização-de-design)
5. [pinneapple\_simulation — Simulação Numérica e Partículas](#5-pinneapple_simulation--simulação-numérica-e-partículas)
6. [pinneapple\_analysis — Validação, Inversão e Incerteza](#6-pinneapple_analysis--validação-inversão-e-incerteza)
7. [pinneapple\_adaptation — Transfer Learning e Meta-Learning](#7-pinneapple_adaptation--transfer-learning-e-meta-learning)
8. [pinneapple\_systems — Sistemas Acoplados e Gêmeos Digitais](#8-pinneapple_systems--sistemas-acoplados-e-gêmeos-digitais)
9. [pinneapple\_tools — Visualização, Export e Benchmarking](#9-pinneapple_tools--visualização-export-e-benchmarking)
10. [pinneapple\_data — Colação e Active Learning](#10-pinneapple_data--colação-e-active-learning)
11. [pinneapple\_pdb — Base de Dados de Física](#11-pinneapple_pdb--base-de-dados-de-física)
12. [pinneapple\_problemdesign — Agente NLP → PDE](#12-pinneapple_problemdesign--agente-nlp--pde)
13. [Fluxos de Uso Típicos](#13-fluxos-de-uso-típicos)
14. [Problemas Físicos Suportados](#14-problemas-físicos-suportados)
15. [Exemplos Disponíveis](#15-exemplos-disponíveis)

---

## 1. Arquitetura Geral

O PINNeAPPle é organizado em **módulos independentes** que se integram por interfaces bem definidas. Cada módulo pode ser usado isoladamente ou em conjunto.

```
pinneapple_physics      ← Define O QUE resolver (PDEs, BCs, domínio)
       ↓
pinneapple_data         ← Define ONDE amostrar (colação, active learning)
       ↓
pinneapple_simulation   ← Gera DADOS de referência (solvers numéricos, ext. tools)
       ↓
pinneapple_neural       ← Define COMO resolver (arquitetura, treinamento)
       ↓
pinneapple_analysis     ← Valida e analisa (UQ, inversão, validação)
       ↓
pinneapple_design       ← Otimiza geometria e parâmetros
pinneapple_adaptation   ← Transfer learning e meta-learning
pinneapple_systems      ← Sistemas acoplados, séries temporais, gêmeos digitais
pinneapple_tools        ← Visualização, export, benchmarking
```

**Quickstart de 4 linhas:**

```python
import pinneapple as pp

spec  = pp.get_preset("burgers_1d", nu=0.01)
model = pp.build_model("SIREN", in_dim=2, out_dim=1, hidden_dim=64, n_layers=4)
result = pp.train_model(model, spec.compile_losses(), epochs=5000)
```

---

## 2. `pinneapple_physics` — Definição de Problemas Físicos

**Propósito:** Especificar problemas físicos como objetos Python estruturados — sem escrever equações na mão. É o ponto de entrada de toda a pipeline.

### 2.1 Especificação de Problemas (`pde_environment`)

#### Classes Principais

| Classe | Descrição |
|--------|-----------|
| `ProblemSpec` | Especificação completa: dimensão, coordenadas, campos, PDE, condições de contorno, escalas, domínio |
| `PDETermSpec` | Descritor da equação: `kind` (tipo de PDE), campos envolvidos, parâmetros numéricos |
| `ScaleSpec` | Normalização por escala de comprimento `L`, velocidade `U` e difusividade `alpha` |
| `ConditionSpec` | Restrição genérica (Dirichlet, Neumann, Robin, IC, dados supervisionados) |

#### Construtores de Condições de Contorno

```python
# Forma simples (dict de campo → valor)
bc = DirichletBC({"u": 0.0, "v": 0.0})

# Forma completa (com seletor espacial)
bc = DirichletBC(
    "inlet",
    fields=("u", "v"),
    selector_type="callable",
    selector=lambda X, ctx: X[:, 0] < 1e-6,
    value_fn=lambda X, ctx: np.column_stack([U_inf * np.ones(X.shape[0]),
                                              np.zeros(X.shape[0])]),
    weight=10.0,
)
NeumannBC(...)    # fluxo normal prescrito
RobinBC(...)      # combinação linear u + ∂u/∂n = g
InitialCondition(...)  # u(x, t=0) = g(x)
DataConstraint(...)    # perda supervisionada em pontos medidos
```

#### Presets Registrados (41+)

Os presets encapsulam todo o conhecimento do problema (PDEs, BCs, escalas, limites de domínio) em uma chamada:

```python
from pinneapple_physics import get_preset, list_presets

spec = get_preset("ns_incompressible_2d", Re=200.0)
spec = get_preset("axial_compressor_meanline", num_stages=5, pressure_ratio=3.0)
```

**Acadêmicos**

| Preset | Domínio | Campos |
|--------|---------|--------|
| `burgers_1d` | (x, t) | u |
| `laplace_2d` | (x, y) | u |
| `poisson_2d` | (x, y) | u |

**CFD / Navier-Stokes**

| Preset | Domínio | Campos |
|--------|---------|--------|
| `ns_incompressible_2d` | (x, y) | u, v, p |
| `ns_incompressible_3d` | (x, y, z) | u, v, w, p |
| `lid_driven_cavity_3d` | (x, y, z) | u, v, w, p |
| `channel_flow_3d` | (x, y, z) | u, v, w, p |
| `pipe_flow_3d` | (r, z) | u_r, u_z, p |

**Aeroespacial**

| Preset | Descrição |
|--------|-----------|
| `rocket_nozzle_cfd` | Euler axissimétrico em bocal convergente-divergente |
| `rocket_structural` | Casing sob pressão interna + gradiente térmico |
| `aircraft_wing_aerodynamics` | RANS 2D simplificado em aerofólio |
| `aircraft_wing_structural` | Longarina composta (plane stress) |

**Turbomachinery** *(integração TurboDesigner)*

| Preset | Dimensão | Campos |
|--------|----------|--------|
| `axial_compressor_meanline` | 1D (s) | T_t, p_t, rho, u, c_theta |
| `axial_compressor_cascade_2d` | 2D (x,y) | rho, u, v, p, T |
| `axial_compressor_stage_3d` | 3D (r,θ,z) | rho, u_r, u_θ, u_z, p, T |

**Automotivo, Industrial, Datacenter, Estrutural, Multidisciplinar**
(automotive thermal/fatigue/aero, furnace, datacenter airflow, elasticity, thermoelasticity, terramechanics, finanças, epidemiologia, farmacocinética — ver `list_presets()`)

#### Registro de Presets Customizados

```python
from pinneapple_physics.pde_environment.presets.registry import register_preset
from pinneapple_physics import ProblemSpec, PDETermSpec, DirichletBC

@register_preset("meu_problema")
def meu_problema(nu: float = 0.01) -> ProblemSpec:
    coords = ("x", "t")
    fields = ("u",)
    return ProblemSpec(
        name="meu_problema",
        dim=2,
        coords=coords,
        fields=fields,
        pde=PDETermSpec(kind="burgers", fields=fields, coords=coords, params={"nu": nu}),
        conditions=(DirichletBC({"u": 0.0}),),
        domain_bounds={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
    )
```

### 2.2 Compilador PINN (`pinn_solver`)

Converte um `ProblemSpec` em funções de loss prontas para treinamento.

| Função/Classe | Descrição |
|---------------|-----------|
| `compile_problem(spec)` | Compila ProblemSpec → função de loss |
| `LossWeights` | Pesos para componentes: `w_pde`, `w_bc`, `w_ic`, `w_data` |
| `grad()`, `jacobian()`, `divergence()`, `laplacian()` | Operadores autograd |
| `Subdomain`, `SubdomainPINN` | Decomposição de domínio |
| `DoMINO` | Domain Decomposition PINN (DAS + time marching) |

**Tipos de PDE suportados pelo compilador:** `laplace`, `poisson`, `burgers`, `navier_stokes_incompressible`, `heat`, `wave`, `elasticity`, `darcy`, `helmholtz`, `advection`, `reaction_diffusion` — e tipos customizados via `SymbolicPDE`.

### 2.3 PDEs Simbólicas (`symbolic_pde`)

Permite definir equações com SymPy e compilá-las automaticamente para residuals PyTorch.

```python
from pinneapple_physics.symbolic_pde import SymbolicPDE, pde_from_sympy
import sympy as sp

x, t, u = sp.symbols("x t u")
pde = pde_from_sympy(sp.diff(u, t) + u * sp.diff(u, x), fields=["u"], coords=["x", "t"])
```

| Classe | Descrição |
|--------|-----------|
| `SymbolicPDE` | Compila expressão SymPy → residual autograd |
| `HardBC` | Satisfaz BCs exatamente via ansatz de distância |
| `PeriodicBC` | Condições periódicas |
| `auto_residual()` | Auto-derivação de residuals |

### 2.4 Turbulência (RANS)

| Classe | Descrição |
|--------|-----------|
| `KOmegaSSTResiduals` | Modelo k-ω SST completo |
| `SpalartAllmarasResiduals` | Modelo de uma equação S-A |
| `get_rans_preset()` | Preset rápido para RANS |

### 2.5 Identificação de PDEs

```python
from pinneapple_physics import identify, define_problem

spec = identify("escoamento incompressível em canal com Re=500")
spec = define_problem("condução de calor em placa com fonte")
```

---

## 3. `pinneapple_neural` — Arquiteturas e Treinamento

**Propósito:** Instanciar, treinar e fazer inferência com 100+ arquiteturas de redes neurais para física computacional.

### 3.1 Registro de Modelos (`architectures`)

Todos os modelos são acessíveis por nome via `ModelRegistry`:

```python
from pinneapple_neural import build_model

model = build_model("SIREN", in_dim=3, out_dim=5, hidden_dim=256, n_layers=6)
model = build_model("FNO", in_channels=1, out_channels=1, modes=16, width=64)
model = build_model("DeepONet", branch_dim=100, trunk_dim=2, hidden=128, layers=4)
```

#### Família PINNs

| Modelo | Descrição |
|--------|-----------|
| `VanillaPINN` | MLP com ativações Tanh/ReLU/GELU padrão |
| `SIREN` | Redes com ativação seno (captura alta frequência) |
| `ModifiedMLP` | MLP com Fourier feature embedding para PINNs |
| `HashGridMLP` | MLP com hash grid encoding (aceleração) |
| `InversePINN` | PINN com parâmetros de PDE treináveis (problema inverso) |
| `VPINN` | Variational PINN |
| `XPINN` | Extended PINN com decomposição de domínio |
| `PINNsFormer` | PINN baseado em Transformer |
| `PIELM` | Physics-Informed Extreme Learning Machine |

#### Operadores Neurais

| Modelo | Descrição |
|--------|-----------|
| `FourierNeuralOperator` (FNO) | Aprendizado espectral via FFT |
| `DeepONet` | Operador universal branch-trunk |
| `PINO` | Physics-Informed Neural Operator |
| `AFNO` | Adaptive Fourier Neural Operator |
| `GraphNeuralOperator` (GNO) | Operador em grafos de malha |

#### Modelos Contínuos

| Modelo | Descrição |
|--------|-----------|
| `NeuralODE` | ODE com integração Runge-Kutta 4 |
| `NeuralSDE` | Equações diferenciais estocásticas |
| `NeuralCDE` | Equações diferenciais controladas |
| `HamiltonianNN` | Preserva estrutura hamiltoniana |
| `SymplecticODE` | Preserva estrutura simplética |
| `LatentODE` | ODE em espaço latente variacional |
| `BayesianRNN` | RNN Bayesiana |

#### Autoencoders / ROM

| Modelo | Descrição |
|--------|-----------|
| `VAE` | Variational Autoencoder |
| `KoopmanAE` | Autoencoder com operador de Koopman |
| `AE_ROM_Hybrid` | ROM + correção neural |
| `DeepUQROM` | ROM com quantificação de incerteza |

#### Graph Neural Networks

| Modelo | Descrição |
|--------|-----------|
| `MeshGraphNet` | Message passing em malhas não-estruturadas |
| `EquivariantGNN` | Equivariante a rotações e translações |
| `GNN_ODE` | GNN + integração de ODE |

#### Transformers para Séries Temporais

`Transformer`, `Informer`, `Autoformer`, `FedFormer`, `TimesNet`, `TFT`

### 3.2 Treinamento (`trainer`)

#### Trainers Disponíveis

| Classe | Descrição |
|--------|-----------|
| `Trainer` | Treinador unificado: métricas, callbacks, logging, AMP |
| `TwoPhaseTrainer` | Fase 1: física; Fase 2: supervisionado |
| `CausalPINNTrainer` | PINN causal com curriculum temporal |
| `TimeMarchingTrainer` | Marching temporal estágio a estágio |
| `DDPPINNTrainer` | Distributed Data Parallel (multi-GPU) |
| `GradAccumTrainer` | Gradient accumulation para batches grandes |

#### Balanceamento de Perdas

| Classe | Estratégia |
|--------|------------|
| `SelfAdaptiveWeights` | Aprende pesos automaticamente |
| `GradNormBalancer` | Balanceia por norma de gradiente |
| `NTKWeightBalancer` | Baseado em Neural Tangent Kernel |
| `LossRatioBalancer` | Mantém proporção entre componentes |
| `WeightScheduler` | Agenda dinâmica de pesos |

#### Utilitários de Treinamento

```python
from pinneapple_neural.trainer import Trainer, TrainConfig

cfg = TrainConfig(
    epochs=10_000,
    lr=1e-3,
    device="cuda",
    amp=True,          # mixed precision
    log_every=500,
)
trainer = Trainer(model, loss_fn, cfg)
trainer.fit(train_loader)
```

**Infraestrutura HPC:** suporte a `FSDP`, `DeepSpeed ZeRO`, `CUDA Graphs`, compressão de gradientes (`PowerSGD`, `TopK`), scripts SLURM, profiling integrado.

### 3.3 Inferência (`predictor`)

```python
from pinneapple_neural import predict, build_model

result = predict(model, x_test, device="cuda", batch_size=10_000)

# Avaliação em grid estruturado
from pinneapple_neural.predictor import infer_on_grid_2d
field = infer_on_grid_2d(model, x_range=(0,1), y_range=(0,1), nx=256, ny=256)
```

---

## 4. `pinneapple_design` — Geometria e Otimização de Design

**Propósito:** Definir domínios geométricos (via SDF ou malhas), amostrar pontos de colocação e executar otimização de design orientada por física.

### 4.1 Geometria (`geometry`)

#### Primitivas SDF 2D

```python
from pinneapple_design.geometry import circle, rectangle, ellipse, annulus

d = circle(center=(0.5, 0.5), radius=0.3)
d = rectangle(center=(0.5, 0.5), half_extents=(0.4, 0.2))
d = annulus(center=(0,0), inner_r=0.2, outer_r=0.5)
```

#### Operações CSG (Boolean)

```python
from pinneapple_design.geometry import sdf_union, sdf_difference, sdf_intersection

shape = sdf_difference(rectangle(...), circle(...))    # subtração
shape = sdf_smooth_union(circle1, circle2, k=0.05)    # união suave
```

#### Primitivas SDF 3D

`sdf3d_sphere`, `sdf3d_box`, `sdf3d_cylinder`, `sdf3d_torus`, `sdf3d_capsule`

#### Domínios Físicos 2D

| Domínio | Descrição |
|---------|-----------|
| `ChannelDomain2D` | Canal retangular com inlet/outlet/walls |
| `ChannelWithObstacleDomain2D` | Canal com obstáculo cilíndrico |
| `LidDrivenCavityDomain2D` | Cavidade com tampa deslizante |
| `LShapeDomain2D` | Domínio em L (concentrador de stress) |
| `AnnularDomain2D` | Domínio anular |
| `TJunctionDomain2D` | Junção em T |
| `SDFDomain2D` | Domínio genérico a partir de SDF |

```python
from pinneapple_design.geometry import get_domain

domain = get_domain("channel_2d", length=4.0, height=1.0, obstacle_radius=0.1)
pts_interior = domain.sample_interior(n=50_000)
pts_boundary = domain.sample_boundary(n=10_000)
```

#### Domínios Físicos 3D

`LidDrivenCavityDomain3D`, `ChannelDomain3D`, `PipeFlowDomain3D`

#### Malha e Colação 3D

| Classe | Descrição |
|--------|-----------|
| `MeshCollocator` | Amostra no interior/boundary de malha 3D |
| `STLDomainBatchBuilder` | Pipeline STL → batch de colação |
| `mesh_rectangle_structured()` | Malha estruturada retangular |
| `mesh_sdf_2d()` | Malha de SDF via marching squares |
| `RBFInterpolator` | Interpolação em nuvem de pontos |
| `naca_parametric()` | Geração de aerofólios NACA 4-dígitos |

### 4.2 Otimização de Design (`design_optimizer`)

Pipeline completo para otimização de forma e parâmetros orientada por física.

#### Objetivos

| Classe | Objetivo |
|--------|----------|
| `DragObjective` | Minimizar arrasto (CFD) |
| `ThermalEfficiencyObjective` | Maximizar eficiência térmica |
| `StructuralObjective` | Minimizar tensão/deformação |
| `WeightMinimizationObjective` | Minimizar massa |
| `CompositeObjective` | Combinação ponderada de objetivos |

#### Otimizadores

| Classe | Método |
|--------|--------|
| `GradientDesignOptimizer` | Gradiente via adjunto contínuo |
| `BayesianDesignOptimizer` | Gaussian Process + Expected Improvement |
| `EvolutionaryDesignOptimizer` | Algoritmo genético |

#### Pareto e Multi-objetivo

```python
from pinneapple_design.design_optimizer import compute_pareto_front, ParetoFront

front = compute_pareto_front(objectives_matrix)  # shape (N, n_obj)
front.plot_2d(labels=["Drag", "Weight"])
```

#### Loop de Otimização

```python
from pinneapple_design.design_optimizer import DesignOptLoop, DesignOptConfig

loop = DesignOptLoop(
    surrogate=trained_pinn,
    objective=DragObjective(),
    param_space=ParamSpace(bounds={"thickness": (0.05, 0.3)}),
    config=DesignOptConfig(method="bayesian", n_trials=200),
)
best = loop.run()
```

---

## 5. `pinneapple_simulation` — Simulação Numérica e Partículas

**Propósito:** Gerar dados de referência para treinamento de PINNs — usando solvers próprios (FDM/FEM) ou bridges para ferramentas externas (OpenFOAM, FEniCS, MATLAB, etc.).

### 5.1 Solvers Numéricos (`numerical_solvers`)

#### Solvers 3D Integrados (FDM)

| Classe | Equação |
|--------|---------|
| `HeatConduction3D` | Difusão térmica (steady/transient) |
| `NavierStokes3D` | Navier-Stokes 3D (SIMPLE/SIMPLER) |
| `ElasticWave3D` | Equação de onda elástica |
| `LidDrivenCavitySolver3D` | Cavidade com tampa — NS 3D |
| `ChannelFlowSolver3D` | Escoamento em canal 3D |

```python
from pinneapple_simulation import simulate, generate_pinn_dataset

output = simulate("heat_3d", nx=64, ny=64, nz=64, kappa=0.1, t_end=1.0)
dataset = generate_pinn_dataset("ns_3d", n_samples=100)
```

#### Geração de Dataset para PINN

```python
dataset = generate_pinn_dataset(
    scenario="heat_3d",
    n_samples=200,
    param_ranges={"kappa": (0.01, 1.0)},
)
```

### 5.2 Dinâmica de Partículas (`particle_dynamics`)

Todos os simuladores são implementados em PyTorch puro (diferenciáveis, compatíveis com autograd).

| Classe | Método | Aplicação |
|--------|--------|-----------|
| `RigidBodySystem` | Euler simpléctica | Multi-body 2D/3D |
| `MPMSimulator` | MLS-MPM | Sólidos elásticos, fluidos viscosos, neve, areia |
| `SPHParticles` | SPH | Escoamentos de superfície livre |
| `ParticleSystem` | Genérico | Sistema customizável |

**Plasticidade no MPM:** Drucker-Prager (granular/neve).

### 5.3 Bridges Externas (`external_solvers`)

Todas as bridges são opcionais — a lib importa mesmo sem a ferramenta instalada.

#### OpenFOAM

```python
from pinneapple_simulation.external_solvers import (
    OpenFOAMCaseTemplate, run_openfoam_case, openfoam_case_to_upd
)
template = OpenFOAMCaseTemplate.from_scenario("channel_flow")
run_openfoam_case(case_dir, cfg)
sample = openfoam_case_to_upd(case_dir)
```

#### FEniCS

```python
from pinneapple_simulation.external_solvers import FEniCSConfig, FEniCSWorkflow

wf = FEniCSWorkflow(FEniCSConfig(pde="heat_equation_steady", domain={...}, bcs=[...]))
samples = wf.sweep({"k": [0.5, 1.0, 2.0, 5.0]})
```

#### TurboDesigner *(bridge integrado)*

```python
from pinneapple_simulation.external_solvers.turbodesigner import (
    TurboDesignerConfig, TurboDesignerWorkflow
)
cfg = TurboDesignerConfig(pressure_ratio=3.0, num_stages=5, rpm=10_000)
wf  = TurboDesignerWorkflow(cfg)
data = wf.solve()                                        # ponto único
samples = wf.sweep({"pressure_ratio": [2.0, 3.0, 4.0]}, as_upd=True)
```

> Sem `turbodesigner` instalado, usa solver analítico embutido automaticamente.

#### Outras Bridges

| Bridge | Pacote necessário |
|--------|-------------------|
| MATLAB | `matlab.engine` |
| OpenModelica / FMU | `fmpy`, `OMPython` |
| MuJoCo | `mujoco >= 3.0` |
| Genesis AI | `genesis-world` |

---

## 6. `pinneapple_analysis` — Validação, Inversão e Incerteza

**Propósito:** Avaliar confiabilidade dos modelos treinados: incerteza, validação física e extração de parâmetros por inversão.

### 6.1 Quantificação de Incerteza (`uncertainty`)

| Método | Classe | Tipo |
|--------|--------|------|
| MC Dropout | `MCDropoutWrapper` | Epistêmica |
| Ensemble | `EnsembleUQ` | Epistêmica |
| Variância Aleatória | `AleatoricHead` | Aleatória |
| Predição Conformal | `ConformalPredictor` | Cobertura garantida |
| Regressão Quantil | `QuantileHead` | Intervalos de predição |

```python
from pinneapple_analysis import analyze_model

report = analyze_model(
    model, spec, x_test,
    uq_method="ensemble",    # "mc_dropout" | "ensemble" | "conformal"
    run_validate=True,
    run_uq=True,
)
print(report.uncertainty.coverage_90)
```

### 6.2 Validação de Física (`validation`)

Verifica automaticamente se o modelo respeita leis de conservação, BCs e simetrias:

```python
from pinneapple_analysis.validation import PhysicsValidator

validator = PhysicsValidator(model, spec)
report = validator.run_all()
# Checks: ConservationCheck, BoundaryCheck, SymmetryCheck
print(report.summary())
```

### 6.3 Problemas Inversos (`inverse_problems`)

#### Identificação de Parâmetros (PDE Parameter Estimation)

```python
from pinneapple_analysis import invert

result = invert(
    model=pinn,
    y_obs=sensor_readings,
    sensor_locs=sensor_positions,
    noise_std=0.01,
    lambda_reg=1e-4,
    method="eki",        # "gradient" | "eki" | "sindy"
    n_iters=100,
)
print(result.params_estimated)
```

#### Modelos de Ruído

| Classe | Quando usar |
|--------|-------------|
| `GaussianMisfit` | Ruído gaussiano padrão |
| `HuberMisfit` | Robusto a outliers |
| `HeteroscedasticMisfit` | Variância não-homogênea |

#### Regularização

| Classe | Penalidade |
|--------|------------|
| `TikhonovRegularizer` | L2 — suavidade |
| `SparsityRegularizer` | L1 — esparsidade |
| `TotalVariationRegularizer` | TV — descontinuidades |
| `LCurveSelector` | Seleciona λ automaticamente |

#### Ensemble Kalman Inversion (EKI)

```python
from pinneapple_analysis.inverse_problems import EnsembleKalmanInversion, EKIConfig

eki = EnsembleKalmanInversion(
    forward_model=pinn,
    obs_operator=PointObsOperator(sensor_locs),
    config=EKIConfig(n_ensemble=200, n_iters=50),
)
result = eki.run(y_obs=observations)
```

#### Descoberta de Equações (SINDy)

```python
from pinneapple_analysis.inverse_problems import SINDyIdentifier

sindy = SINDyIdentifier(library=CandidateLibrary(poly_order=3))
result = sindy.fit(X_data, dX_data)
print(result.equations)
```

---

## 7. `pinneapple_adaptation` — Transfer Learning e Meta-Learning

**Propósito:** Reutilizar modelos treinados para novos domínios físicos com pouco dado adicional.

### 7.1 Transfer Learning (`transfer_learning`)

```python
from pinneapple_adaptation.transfer_learning import TransferTrainer, TransferConfig

# Fine-tune apenas as últimas 2 camadas
cfg = TransferConfig(
    strategy="last_layers",
    epochs=500,
    finetune_lr=1e-4,
    layer_freezing={"freeze_prefix": "encoder"},
)
new_model = TransferTrainer(pretrained, new_spec, cfg).finetune(new_data)
```

| Função | Descrição |
|--------|-----------|
| `freeze_layers()` | Congela camadas por nome/prefixo |
| `layer_lr_groups()` | LRs discriminativos por camada |
| `PhysicsTransferAdapter` | Adaptação com MMD loss entre domínios |
| `ParametricFamilyTransfer` | Interpolação entre variantes paramétricas |

### 7.2 Meta-Learning (`meta_learning`)

Treina um modelo com capacidade de adaptação rápida (few-shot) para famílias de PDEs.

```python
from pinneapple_adaptation.meta_learning import MAMLTrainer, MAMLConfig, PDETaskSampler

sampler = PDETaskSampler(family="navier_stokes", param_ranges={"Re": (50, 2000)})
meta_model = MAMLTrainer(model, sampler, MAMLConfig(inner_steps=5, meta_lr=1e-3)).train()

# Adaptação rápida a novo Re em 5 gradientes
adapted = meta_adapt(meta_model, new_task_data, n_steps=5)
```

| Algoritmo | Classe |
|-----------|--------|
| MAML | `MAMLTrainer` |
| Reptile | `ReptileTrainer` |

---

## 8. `pinneapple_systems` — Sistemas Acoplados e Gêmeos Digitais

**Propósito:** Modelar sistemas multi-componentes: séries temporais de sinais físicos, co-simulação de múltiplos modelos e gêmeos digitais ao vivo.

### 8.1 Séries Temporais (`time_series`)

#### Modelos Disponíveis

**Baseline:** `NaiveForecaster`, `SeasonalNaiveForecaster`, `DriftForecaster`

**Machine Learning:** `XGBoostForecaster`, `LightGBMForecaster`, `RandomForestForecaster`, `GPRForecaster`

**Deep Learning:**

| Modelo | Arquitetura |
|--------|-------------|
| `LSTMForecaster` | LSTM encoder-decoder |
| `GRUForecaster` | GRU |
| `NBeats` | N-BEATS (interpretable) |
| `TCNForecaster` | Temporal Convolutional Network |
| `TFTForecaster` | Temporal Fusion Transformer |
| `FFTForecaster` | FFT + NN |
| `HHTNNForecaster` | Hilbert-Huang + NN |

```python
from pinneapple_systems import forecast

predictions = forecast(
    data=sensor_df,
    horizon=24,
    model="TFT",
    input_len=168,
)
```

#### Análise e Visualização

`plot_trend`, `power_spectrum`, `plot_acf_pacf`, `stationarity_report`, `animate_rolling_forecast`

#### Backtesting

```python
from pinneapple_systems.time_series import BacktestRunner, BacktestConfig

runner = BacktestRunner(
    model=forecaster,
    config=BacktestConfig(n_splits=5, strategy="expanding"),
)
result = runner.run(series)
```

### 8.2 Co-Simulação (`cosimulation`)

Motor de co-simulação baseado em grafos — conecta PINNs, modelos analíticos, solvers e séries temporais.

#### Tipos de Nó

| Nó | Descrição |
|----|-----------|
| `PINNNode` | Nó PINN treinado |
| `AnalyticalNode` | Função Python analítica |
| `TimeSeriesCoSimNode` | Forecaster de série temporal |
| `SymbolicPDENode` | PDE simbólica |
| `BlackBoxNode` | Caixa preta genérica |

```python
from pinneapple_systems.cosimulation import CoSimGraph, CoSimEngine

graph = CoSimGraph()
graph.add_node("cfd", PINNNode(ns_pinn))
graph.add_node("thermal", AnalyticalNode(heat_fn))
graph.add_connection(Connection("cfd.velocity", "thermal.u"))

engine = CoSimEngine(graph)
trajectory = engine.run(t_span=(0, 10), dt=0.01)
```

### 8.3 Gêmeo Digital (`digital_twin`)

Fusão de dados de sensor em tempo real com modelo PINN + filtros de Kalman.

```python
from pinneapple_systems.digital_twin import build_digital_twin, DigitalTwinConfig

twin = build_digital_twin(
    model=pinn,
    spec=problem_spec,
    config=DigitalTwinConfig(
        assimilation="enkf",       # "ekf" | "enkf"
        anomaly_detector="zscore", # "threshold" | "zscore" | "mahalanobis"
        stream="mqtt",             # "mqtt" | "kafka" | "http" | "file"
    ),
)
twin.start()   # inicia loop de assimilação
```

#### Fontes de Dados (Streams)

| Stream | Protocolo |
|--------|-----------|
| `MQTTStream` | IoT / MQTT broker |
| `KafkaStream` | Apache Kafka |
| `HTTPPollStream` | REST API polling |
| `FileWatchStream` | Arquivo em disco |
| `MockStream` | Dados sintéticos (desenvolvimento) |

---

## 9. `pinneapple_tools` — Visualização, Export e Benchmarking

**Propósito:** Visualizar campos físicos, exportar modelos para produção e comparar modelos sistematicamente.

### 9.1 Visualização (`visualization`)

```python
from pinneapple_tools.visualization import plot_scalar, plot_streamlines, animate_scalar_field

plot_scalar(x, y, field, cmap="coolwarm", title="Pressão")
plot_streamlines(x, y, u, v, density=2.0)
animate_scalar_field(field_sequence, dt=0.1, save_as="evolution.gif")
```

#### Visualizações CFD Específicas

| Função | Saída |
|--------|-------|
| `plot_vorticity()` | Campo de vorticidade 2D |
| `plot_q_criterion_2d/3d()` | Q-criterion para identificação de vórtices |
| `plot_lambda2_3d()` | Lambda-2 criterion |
| `plot_pde_residual()` | Resíduo da PDE por ponto |
| `plot_collocation()` | Pontos de colocação coloridos por loss |
| `plot_loss_history()` | Curva de convergência multi-componente |

### 9.2 Export de Modelos (`model_export`)

```python
from pinneapple_tools import export_model

export_model(model, "solver.onnx",  fmt="onnx",        input_shape=(1, 3))
export_model(model, "solver.pt",    fmt="torchscript")
export_model(model, "outputs.csv",  fmt="csv",          x_test=x)
export_model(model, "outputs.npz",  fmt="npz",          x_test=x)
```

### 9.3 Benchmarking — Arena (`benchmark_suite`)

Sistema YAML-driven para comparar múltiplos modelos em múltiplas tarefas:

```yaml
# configs/arena/burgers_benchmark.yaml
tasks:
  - burgers_1d
  - ns_incompressible_2d
models:
  - SIREN
  - FNO
  - DeepONet
metrics: [l2_error, max_error, training_time]
```

```python
from pinneapple_tools.benchmark_suite import Arena

runner = Arena.from_yaml("configs/arena/burgers_benchmark.yaml")
results = runner.run_all()
results.leaderboard()
results.plot_comparison()
```

---

## 10. `pinneapple_data` — Colação e Active Learning

**Propósito:** Gerenciar pontos de colocação, datasets físicos e amostragem adaptativa.

### 10.1 Colação

```python
from pinneapple_data import CollocationSampler, CollocationConfig

sampler = CollocationSampler(
    domain=my_domain,
    config=CollocationConfig(
        n_interior=50_000,
        n_boundary=10_000,
        method="latin_hypercube",   # "random" | "sobol" | "latin_hypercube"
    )
)
batch = sampler.sample()
```

### 10.2 Active Learning

Concentra os pontos de colocação onde a rede tem maior resíduo ou incerteza:

| Estratégia | Classe | Critério |
|------------|--------|----------|
| Residual-based | `ResidualBasedAL` | Pontos com maior resíduo PDE |
| Variance-based | `VarianceBasedAL` | Pontos com maior variância (MC Dropout) |
| Combinada | `CombinedAL` | Combinação ponderada |

```python
from pinneapple_data import AdaptiveCollocationTrainer

trainer = AdaptiveCollocationTrainer(
    model=pinn,
    loss_fn=losses,
    al_strategy="residual",
    refine_every=500,    # épocas entre refinamentos
    n_add=1000,          # pontos adicionados por refinamento
)
trainer.fit(epochs=10_000)
```

### 10.3 Formato UPD (Universal Physical Data)

Formato interno para amostras físicas com metadados:

```python
from pinneapple_data.physical_sample import PhysicalSample
import torch

sample = PhysicalSample(
    fields={"u": torch.tensor([...]), "p": torch.tensor([...])},
    coords={"x": x_array, "y": y_array},
    meta={"units": {"u": "m/s", "p": "Pa"}, "source": "openfoam"},
)
```

### 10.4 Registro de Datasets

```python
from pinneapple_data import load_dataset, list_datasets

print(list_datasets())
ds = load_dataset("cylinder_wake_re100")
```

---

## 11. `pinneapple_pdb` — Base de Dados de Física

**Propósito:** Construir e consultar datasets físicos de fontes externas (NASA, ECMWF, satélites, etc.) com esquema padronizado.

```python
from pinneapple_pdb import PhysicalDatasetBuilder, HubQuery

builder = PhysicalDatasetBuilder()
query = HubQuery(
    variable=VariableSelection(names=["temperature", "pressure"]),
    space_time=SpaceTime(lat=(-90, 90), lon=(-180, 180), time=("2020-01", "2020-12")),
)
dataset = builder.build(query)
```

---

## 12. `pinneapple_problemdesign` — Agente NLP → PDE

**Propósito:** Converter descrições em linguagem natural de problemas físicos em `ProblemSpec` completos, com identificação automática de PDEs relevantes, plano de solução e código gerado.

```python
from pinneapple_problemdesign import DesignAgent

agent = DesignAgent(provider="gemini")   # ou provider customizado

report = agent.design(
    "Quero simular escoamento laminar em torno de um cilindro com Re=100, "
    "e extrair o coeficiente de arrasto."
)

print(report.problem_spec)     # ProblemSpec gerado
print(report.plan)             # plano de ação passo a passo
print(report.gaps)             # lacunas identificadas
print(report.pinneapple_code)  # código Python pronto para executar
```

---

## 13. Fluxos de Uso Típicos

### A. PINN simples (resolver uma PDE)

```python
import pinneapple as pp

spec   = pp.get_preset("burgers_1d", nu=0.01)
model  = pp.build_model("SIREN", in_dim=2, out_dim=1, hidden_dim=128, n_layers=6)
result = pp.train_model(model, spec.compile_losses(), epochs=10_000)
pp.plot(model, x_test, field_name="u", dim=1)
```

### B. Geração de dados + treinamento supervisionado

```python
from pinneapple_simulation import generate_pinn_dataset
from pinneapple_neural import build_model, train_model

dataset = generate_pinn_dataset("heat_3d", n_samples=200)
model   = build_model("FNO", in_channels=3, out_channels=1)
result  = train_model(model, dataset, epochs=5_000, supervised=True)
```

### C. Problema inverso (identificação de parâmetros)

```python
from pinneapple_neural import build_model
from pinneapple_analysis import invert

pinn   = build_model("InversePINN", in_dim=2, out_dim=1, n_params=1)
result = invert(pinn, y_obs=measurements, sensor_locs=coords,
                noise_std=0.01, method="eki", n_iters=100)
print(f"ν estimado: {result.params_estimated['nu']:.4f}")
```

### D. Otimização de design

```python
from pinneapple_design.design_optimizer import DesignOptLoop, BayesianDesignOptimizer
from pinneapple_analysis.uncertainty import EnsembleUQ

surrogate = train_surrogate_pinn(ns_spec, domain)
loop = DesignOptLoop(
    surrogate=EnsembleUQ(surrogate, n_models=5),
    objective=DragObjective(),
    optimizer=BayesianDesignOptimizer(n_trials=100),
)
best_design = loop.run()
```

### E. Gêmeo digital ao vivo

```python
from pinneapple_systems.digital_twin import build_digital_twin

twin = build_digital_twin(model=pinn, spec=spec,
                           config=DigitalTwinConfig(stream="mqtt", assimilation="enkf"))
twin.start()
# loop de assimilação em background — anomalias emitem callbacks
```

### F. Turbomachinery (TurboDesigner + PINN)

```python
import pinneapple as pp
from pinneapple_simulation.external_solvers.turbodesigner import (
    TurboDesignerConfig, TurboDesignerWorkflow
)

# 1. Gerar dados analíticos do TurboDesigner
cfg  = TurboDesignerConfig(pressure_ratio=3.0, num_stages=5, rpm=10_000)
data = TurboDesignerWorkflow(cfg).sweep({"pressure_ratio": [2.0, 3.0, 4.0]}, as_upd=True)

# 2. Treinar PINN com constraints físicas + dados analíticos como âncora
spec  = pp.get_preset("axial_compressor_meanline", pressure_ratio=3.0)
model = pp.build_model("SIREN", in_dim=1, out_dim=5, hidden_dim=128, n_layers=6)
result = pp.train_model(model, spec.compile_losses(), epochs=15_000,
                         data_samples=data)
```

---

## 14. Problemas Físicos Suportados

| Domínio | Tipos |
|---------|-------|
| **Fluxo de fluidos** | NS 2D/3D incompressível, Stokes, Darcy, Burgers, advecção, escoamento em canal/tubo |
| **Compressível** | Euler axissimétrico, bocal convergente-divergente, cascata 2D de pás |
| **Turbulência** | RANS k-ω SST, Spalart-Allmaras |
| **Térmica** | Condução estacionária/transiente 2D/3D, termóelasticidade, PCB cooling, dissipadores |
| **Estrutural** | Elasticidade linear 2D/3D, plane stress/strain, Von Mises, torção, fadiga |
| **Onda** | Equação de onda 1D/2D, ultrassom, Helmholtz acústico |
| **Turbomachinery** | Linha média axial, cascata 2D, estágio 3D em frame rotativo |
| **Terramecânica** | Bekker-Wong solo-roda, rover mobility |
| **Eletromagnético** | Corrente de Foucault, Maxwell, onda EM, guia de onda TM |
| **Reação-difusão** | Sistemas genéricos, combustão simplificada |
| **Partículas** | MPM (sólido/fluido/neve/areia), SPH, rigid body |
| **Multi-física** | Fluid-structure, thermoelastic, magneto-elastic |
| **Finanças** | Black-Scholes, Heston PDE |
| **Biologia** | SIR epidemiológico, difusão de fármaco, compartimental PK |

---

## 15. Exemplos Disponíveis

O projeto conta com **176 exemplos** organizados em categorias:

| Categoria | Exemplos | Destaques |
|-----------|----------|-----------|
| `getting_started/` | 10 | Oscilador harmônico, Lotka-Volterra, van der Pol, Lorenz |
| `pde_environment/` | 4 | Laplace 2D, Burgers 1D, NS 2D canal, Heat 3D STL |
| `pinn_solver/` | 6 | Perda simbólica, DoMINO, inversão de parâmetro |
| `architectures/` | 14 | Tour pelo registry, FNO, DeepONet, MeshGraphNet |
| `data_pipeline/` | 8 | UPD dataloaders, synthetic PDE, zarr sharded |
| `numerical_solvers/` | 11 | FEM, FVM, SPH, LBM, espectral, CAD-CFD |
| `geometry/` | 7 | STL batch, malhas, CSG, curvatura |
| `benchmark_suite/` | 17 | Arena YAML, NACA aerodinâmico, digital twin |
| `time_series/` | 8 | FNO temporal, backtest, FFT-LSTM, HHT-LSTM |
| `arena_pipelines/` | 9 | Kovasznay NS, benchmarks multi-modelo |
| `trainer/` | 5 | DDP torchrun, audited training, DataModule |
| `cosimulation/` | 1 | Spring-mass PINN acoplado |
| `electrodynamics/` | 6 | Capacitor, dipolo, magnetostática, guia TM |
| `visualizations/` | 6 | Cilindro, calor 2D, vórtices, estrutural |
| `hpo_experiments/` | 7 | Discover, build KB, reproduce |
| `problem_designer/` | 5 | API NLP→PDE, batch generate |
| `use_cases/` | 2 | Drill pipe torsion, terramechanics rover |

**Executar um exemplo:**

```bash
cd examples/getting_started
python harmonic_oscillator.py
```

**Executar benchmark Arena:**

```bash
cd examples/arena_pipelines
python run_arena_yaml.py --config ../../configs/arena/burgers_benchmark.yaml
```

---

*Documento gerado a partir do código-fonte em `main` — maio 2026.*