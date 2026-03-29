# PINNeAPPle vs PhysicsNeMo — Comparison & Integration Guide

## Resumo executivo

| Pergunta | Ferramenta |
|---|---|
| Preciso de UQ / intervalos de confiança nas predições? | **PINNeAPPle** |
| Preciso treinar em multi-GPU / cluster com NCCL? | **PhysicsNeMo** |
| Preciso de digital twin com streams de sensores reais? | **PINNeAPPle** |
| Preciso de FNO/AFNO para operator learning em larga escala? | **PhysicsNeMo** |
| Preciso de meta-aprendizado para famílias paramétricas de PDEs? | **PINNeAPPle** |
| Preciso de MeshGraphNet para malhas não-estruturadas >200k nós? | **PhysicsNeMo** |
| Preciso de collocação adaptativa (RAR/RAD)? | **PINNeAPPle** |
| Preciso de FourCastNet / SFNO para previsão climática global? | **PhysicsNeMo** |
| Preciso de XPINN com decomposição de domínio? | **PINNeAPPle** |
| Preciso de assimilação de dados (EKF/EnKF)? | **PINNeAPPle** |
| Preciso de TensorRT / Triton para inferência em produção NVIDIA? | **PhysicsNeMo** |
| Preciso de presets para finanças, fármacos, sistemas sociais? | **PINNeAPPle** |
| Preciso de validação de consistência física (conservação, BCs)? | **PINNeAPPle** |
| Quero o melhor throughput bruto em CFD em hardware NVIDIA? | **PhysicsNeMo** |

---

## Filosofias diferentes

### PhysicsNeMo (ex-NVIDIA Modulus)
- **GPU-first**: built para tirar o máximo de hardware NVIDIA (cuDNN, Transformer Engine, TensorRT)
- **Escala**: de um GPU a clusters multi-nó via `torch.distributed` + NCCL
- **Foco**: surrogate de alto throughput para CFD e previsão do tempo
- **Deploy**: integrado com Omniverse / NVIDIA digital twin platform
- **Modelos pré-treinados**: FourCastNet, SFNO, GraphCastNet disponíveis como checkpoints

### PINNeAPPle
- **Pesquisa → Produção**: pipeline completo de geometria até digital twin deployed
- **Domínios amplos**: não só CFD — finanças, farmacologia, materiais, sistemas sociais
- **Inteligência sobre velocidade**: UQ, assimilação, meta-aprendizado, validação física
- **Facilidade de uso**: `pp.quickstart("burgers_1d")`, presets prontos, YAML pipeline
- **Open research**: implementações de referência limpas de SA-PINN, XPINN, XTFC, VPINN

---

## Tabela de features completa

| Feature | PINNeAPPle | PhysicsNeMo | Notas |
|---|---|---|---|
| **Arquiteturas PINN** | 8 variantes (ver catálogo abaixo) | FullyConnected (via Sym) | PINNeAPPle mais variado |
| **Operadores neurais** | FNO, DeepONet, GNO, PINO, UNO, MultiScale-DeepONet | FNO, AFNO, SFNO, FNO-2D/3D | PhysicsNeMo mais otimizado |
| **GNN** | GNN, EGNN, GNN-ODE, STGNN, GraphCast | MeshGraphNet, AeroGraphNet, X-MeshGraphNet, DoMINO | PhysicsNeMo especializado para CFD |
| **ROMs** | POD, DMD, HAVOK, OpInf, DeepUQROM | ❌ | PINNeAPPle exclusivo |
| **Neural ODEs** | NeuralODE, LatentODE, HNN, SymplecticODE, NeuralSDE | ❌ | PINNeAPPle exclusivo |
| **Autoencoders** | DenseAE, VAE, KAE, PI-Koopman-AE, AEROMHybrid | ❌ | PINNeAPPle exclusivo |
| **Reservoir Computing** | ELM, RBF, ESN, Koopman, HybridRBF | ❌ | PINNeAPPle exclusivo |
| **Séries Temporais** | LSTM, GRU, Seq2Seq, Informer, TFT, Autoformer, TimesNet | ❌ | PINNeAPPle exclusivo |
| **Clima / Tempo** | — | FourCastNet, SFNO, DLWP, GraphCastNet | PhysicsNeMo exclusivo |
| **Multi-GPU** | `torch.DataParallel` + `maybe_compile` | DDP completo + NCCL + multi-nó | PhysicsNeMo mais robusto |
| **UQ** | MC Dropout, Ensemble, Conformal | ❌ | PINNeAPPle exclusivo |
| **Digital Twin** | Streams MQTT/Kafka/HTTP/Mock, EKF/EnKF, anomalia | ❌ | PINNeAPPle exclusivo |
| **Meta-aprendizado** | MAML, Reptile, PDETaskSampler | ❌ | PINNeAPPle exclusivo |
| **Transfer Learning** | TransferTrainer, ParametricFamilyTransfer | ❌ | PINNeAPPle exclusivo |
| **Active Learning** | RAR, RAD, CombinedAL, AdaptiveCollocationTrainer | ❌ | PINNeAPPle exclusivo |
| **Ajuste de pesos** | SA-PINN, GradNorm, NTK, LossRatio | ❌ | PINNeAPPle exclusivo |
| **Validação física** | ConservationCheck, BoundaryCheck, SymmetryCheck | ❌ | PINNeAPPle exclusivo |
| **Problem design** | ProblemBuilder → ProblemSpec → Arena pipeline | Scripts Python | PINNeAPPle exclusivo |
| **Presets** | 35+ (CFD, estrutural, térmico, finanças, fármacos…) | CFD focado | PINNeAPPle mais amplo |
| **YAML pipeline** | `run_full_pipeline(config.yaml)` | Scripts Python | PINNeAPPle mais simples |
| **Export** | ONNX, TorchScript, checkpoint | ONNX, TensorRT, Triton | PhysicsNeMo melhor deploy NVIDIA |
| **Solvers** | FEniCS, OpenFOAM, FDM, FEM, SPH, LBM | OpenFOAM (básico) | PINNeAPPle mais completo |
| **Licença** | Open source (MIT) | Open source (Apache 2.0) | Ambos abertos |

---

## Catálogo completo de modelos — PINNeAPPle (92 modelos)

### PINNs — Physics-Informed Neural Networks (8)

| Classe | Arquivo | Descrição |
|---|---|---|
| `VanillaPINN` | `pinns/vanilla.py` | PINN base — MLP com physics residual |
| `InversePINN` | `pinns/inverse.py` | Identifica parâmetros desconhecidos da PDE |
| `PIELM` | `pinns/pielm.py` | Physics-Informed ELM — treinamento linear rápido |
| `PINNLSTM` | `pinns/pinn_lstm.py` | PINN com backbone LSTM para séries temporais |
| `PINNsFormer` | `pinns/pinnsformer.py` | PINN com Transformer encoder |
| `VPINN` | `pinns/vpinn.py` | Variational PINN — formulação fraca da PDE |
| `XPINN` | `pinns/xpinn.py` | Extended PINN — decomposição de domínio em subdomínios |
| `XTFC` | `pinns/xtfc.py` | eXtreme Theory of Functional Connections — satisfação exata de BCs |

### Operadores Neurais (6)

| Classe | Arquivo | Descrição |
|---|---|---|
| `FourierNeuralOperator` / `FNO` | `neural_operators/fno.py` | Aprende operadores via camadas espectrais |
| `DeepONet` | `neural_operators/deeponet.py` | Branch + trunk network para operadores |
| `GalerkinNeuralOperator` / `GNO` | `neural_operators/gno.py` | Operador via atenção de Galerkin |
| `PhysicsInformedNeuralOperator` / `PINO` | `neural_operators/pino.py` | FNO com restrições de PDE |
| `UniversalUNO` | `neural_operators/uno.py` | U-Net + operador neural |
| `MultiScaleDeepONet` | `neural_operators/ms_deeponet.py` | DeepONet multi-escala |

### Graph Neural Networks (5)

| Classe | Arquivo | Descrição |
|---|---|---|
| `GraphNeuralNetwork` / `GNN` | `graphnn/gnn.py` | Message passing em malhas não-estruturadas |
| `EquivariantGNN` / `EGNN` | `graphnn/equivariant_gnn.py` | GNN com equivariância SO(3) |
| `GraphNeuralODE` | `graphnn/gnn_ode.py` | GNN com integrador de ODE contínuo |
| `SpatiotemporalGNN` / `STGNN` | `graphnn/spatiotemporal_gnn.py` | GNN para dados espaço-temporais |
| `GraphCast` | `graphnn/graphcast.py` | Previsão meteorológica estilo DeepMind |

### Modelos de Tempo Contínuo / Neural ODEs (11)

| Classe | Arquivo | Descrição |
|---|---|---|
| `NeuralODE` | `continuous/neural_ode.py` | Rede neural como campo de vetores |
| `LatentODE` | `continuous/latent_ode.py` | VAE + ODE no espaço latente |
| `ODERNN` | `continuous/ode_rnn.py` | RNN com transições ODE entre observações |
| `NeuralCDE` | `continuous/neural_cde.py` | Controlled differential equation para dados irregulares |
| `NeuralSDE` | `continuous/neural_sde.py` | SDE estocástica como modelo generativo |
| `HamiltonianNeuralNetwork` / `HNN` | `continuous/hamiltonian.py` | Aprende Hamiltoniano — conserva energia |
| `SymplecticODENet` | `continuous/symplectic_ode.py` | ODE com integradores simpléticos |
| `SymplecticRNN` | `continuous/symplectic_rnn.py` | RNN simpléctica para dinâmicas conservativas |
| `BayesianRNN` | `continuous/bayesian_rnn.py` | RNN com pesos bayesianos |
| `DeepStateSpaceModel` / `DSSM` | `continuous/deep_state_space.py` | SSM com encoder/decoder profundo |
| `NeuralGaussianProcess` / `NGP` | `continuous/neural_gp.py` | Processo Gaussiano com kernel neural |

### Recurrent / Sequence Models (5)

| Classe | Arquivo | Descrição |
|---|---|---|
| `LSTMModel` | `recurrent/lstm.py` | LSTM padrão para séries temporais |
| `BiLSTMModel` | `recurrent/lstm.py` | LSTM bidirecional |
| `GRUModel` | `recurrent/gru.py` | GRU — alternativa leve ao LSTM |
| `BiGRUModel` | `recurrent/gru.py` | GRU bidirecional |
| `Seq2SeqRNN` | `recurrent/seq2seq.py` | Encoder-decoder para previsão de sequências |

### Transformers para Séries Temporais (6)

| Classe | Arquivo | Descrição |
|---|---|---|
| `VanillaTransformer` | `transformers/transformer.py` | Transformer padrão para TS |
| `Informer` | `transformers/informer.py` | Atenção esparsa para sequências longas |
| `TemporalFusionTransformer` / `TFT` | `transformers/tft.py` | Multi-horizon forecasting com variáveis externas |
| `Autoformer` | `transformers/autoformer.py` | Decomposição sazonal + atenção autocorrelação |
| `FEDformer` | `transformers/fedformer.py` | Frequência + Transformer |
| `TimesNet` | `transformers/timesnet.py` | TS como imagem 2D |

### Autoencoders (6)

| Classe | Arquivo | Descrição |
|---|---|---|
| `DenseAutoencoder` | `autoencoders/dense_ae.py` | AE denso padrão |
| `Autoencoder2D` | `autoencoders/ae_2d.py` | AE convolucional 2D |
| `VariationalAutoencoder` / `VAE` | `autoencoders/vae.py` | VAE com reparametrização |
| `KAEAutoencoder` / `KAE` | `autoencoders/kae.py` | Koopman AE — dinâmicas lineares no espaço latente |
| `PhysicsInformedKoopmanAE` | `autoencoders/koopman_pi_ae.py` | Koopman AE com restrições físicas |
| `AEROMHybrid` | `autoencoders/ae_rom_hybrid.py` | AE + ROM para redução de ordem |

### Modelos de Ordem Reduzida — ROM (6)

| Classe | Arquivo | Descrição |
|---|---|---|
| `POD` | `rom/pod.py` | Proper Orthogonal Decomposition — SVD de snapshots |
| `DynamicModeDecomposition` / `DMD` | `rom/dmd.py` | DMD — modos e valores próprios de Koopman |
| `HAVOK` | `rom/havok.py` | Hankel Alternative View of Koopman |
| `OperatorInference` / `OpInf` | `rom/opinf.py` | ROM intrusivo por inferência de operadores |
| `ROMHybrid` | `rom/rom_hybrid.py` | ROM + rede neural para closure |
| `DeepUQROM` | `rom/deep_uq_rom.py` | ROM com quantificação de incerteza |

### Reservoir Computing (6)

| Classe | Arquivo | Descrição |
|---|---|---|
| `ExtremeLearningMachine` / `ELM` | `reservoir_computing/elm.py` | Rede com pesos fixos + regressão linear |
| `RBFNetwork` | `reservoir_computing/rbf.py` | Rede de funções de base radial |
| `HybridRBFNetwork` | `reservoir_computing/hybrid_rbf.py` | RBF + componente neural treinável |
| `EchoStateNetwork` / `ESN` | `reservoir_computing/esn.py` | Reservoir computing com reservatório fixo |
| `ESNRC` | `reservoir_computing/esn_rc.py` | ESN com readout classificador |
| `KoopmanOperator` | `reservoir_computing/koopman.py` | Operador de Koopman para dinâmicas não-lineares |

### Modelos Clássicos de Séries Temporais (7)

| Classe | Arquivo | Descrição |
|---|---|---|
| `VAR` | `classical_ts/var.py` | Vector Autoregression |
| `ARIMA` | `classical_ts/arima.py` | ARIMA com diferenciação |
| `KalmanFilter` | `classical_ts/kalman.py` | Filtro de Kalman linear |
| `ExtendedKalmanFilter` / `EKF` | `classical_ts/ekf.py` | Kalman não-linear (linearização local) |
| `UnscentedKalmanFilter` / `UKF` | `classical_ts/ukf.py` | Kalman não-linear (sigma points) |
| `EnsembleKalmanFilter` / `EnKF` | `classical_ts/enkf.py` | Kalman por ensemble Monte Carlo |
| `TCN` | `classical_ts/tcn.py` | Temporal Convolutional Network — receptive field explícito |

### Convolucionais (3)

| Classe | Arquivo | Descrição |
|---|---|---|
| `Conv1DModel` | `convolutions/conv1d.py` | CNN 1D para séries e campos 1D |
| `Conv2DModel` | `convolutions/conv2d.py` | CNN 2D para campos espaciais |
| `Conv3DModel` | `convolutions/conv3d.py` | CNN 3D para campos volumétricos |

### Physics-Aware (2)

| Classe | Arquivo | Descrição |
|---|---|---|
| `PhysicsAwareNeuralNetwork` / `PANN` | `physics_aware/pann.py` | Rede com termos físicos explícitos embutidos |
| `StructurePreservingNetwork` / `SPN` | `physics_aware/spn.py` | Preserva estrutura geométrica (Lie groups, etc.) |

### Baselines / Benchmarks (5)

| Classe | Arquivo | Descrição |
|---|---|---|
| `GenericMLP` | `benchmarks/generic_pinn_models.py` | MLP baseline |
| `GenericFourierMLP` | `benchmarks/generic_pinn_models.py` | MLP com encoding de Fourier |
| `GenericSIREN` | `benchmarks/generic_pinn_models.py` | SIREN — ativações sinusoidais |
| `GenericResMLP` | `benchmarks/generic_pinn_models.py` | MLP com skip connections |
| `GenericLinear` | `benchmarks/generic_pinn_models.py` | Baseline linear |

---

## Presets de problemas (35+)

### Acadêmicos / Didáticos
`laplace_2d` · `poisson_2d` · `burgers_1d`

### CFD / Navier-Stokes
`ns_incompressible_2d` · `ns_incompressible_3d`

### Industrial
`steady_heat_conduction_3d` · `transient_heat_3d` · `linear_elasticity_3d` · `darcy_pressure_only_3d` · `helmholtz_acoustics_3d` · `wave_ultrasound_3d` · `reaction_diffusion_2d`

### Estrutural
`plane_stress_2d` · `plane_strain_2d` · `von_mises_2d` · `linear_elasticity_3d` · `drill_pipe_torsion` · `thermoelasticity_2d`

### Engenharia (18)
`rocket_nozzle_cfd` · `rocket_structural` · `aircraft_wing_aerodynamics` · `aircraft_wing_structural` · `car_external_aero` · `car_brake_thermal` · `car_suspension_fatigue` · `cpu_heatsink_thermal` · `pcb_thermal` · `fan_cooler_cfd` · `industrial_furnace_thermal` · `refractory_lining` · `furnace_combustion_zone` · `datacenter_airflow_2d` · `datacenter_server_thermal` · `datacenter_cfd_3d` ·…

### Multidisciplinar
`climate_atmosphere_2d` · `climate_ocean_gyre` · `crystal_phonon` · `material_fracture_2d` · `black_scholes_1d` · `heston_pde_2d` · `pk_two_compartment` · `drug_diffusion_tissue` · `sir_epidemic` · `opinion_dynamics_2d`

```python
from pinneaple_environment import list_presets
print(list_presets())  # lista todos os presets disponíveis
```

---

## Padrões de integração

### Padrão 1 — PhysicsNeMo treina, PINNeAPPle opera
```
Dados CFD → PhysicsNeMo FNO (throughput, GPU) → checkpoint
         → PINNeAPPle carrega → MCDropoutWrapper (UQ)
         → DigitalTwin + MockStream (sensores)
         → Anomaly detection + EKF assimilation
         → REST API via pinneaple_serve
```
**Melhor para**: surrogate de alta fidelidade + digital twin em produção.

### Padrão 2 — PINNeAPPle seleciona pontos, PhysicsNeMo retreina
```
PINNeAPPle ResidualBasedAL → pontos hard de collocation
PhysicsNeMo retreina FNO nesses pontos (GPU rápido)
→ loop: AL seleciona → PhysicsNeMo retreina → AL seleciona...
```
**Melhor para**: reduzir número de simulações necessárias.

### Padrão 3 — PhysicsNeMo para geometria complexa, PINNeAPPle valida
```
Malha STL → PhysicsNeMo MeshGraphNet (inferência em malha não-estruturada)
          → PINNeAPPle PhysicsValidator (conservação de massa, BCs, simetria)
          → PINNeAPPle export_onnx() para deploy C++
```
**Melhor para**: validar que GNNs não predizem soluções não-físicas.

---

## Exemplos neste diretório

| # | Diretório | O que mostra | Ferramentas |
|---|---|---|---|
| 01 | `01_pinneaple_uq_digital_twin/` | UQ + Digital Twin em equação do calor | PINNeAPPle **only** |
| 02 | `02_pinneaple_meta_transfer/` | Meta-aprendizado Reptile + Transfer Learning em Burgers | PINNeAPPle **only** |
| 03 | `03_pinneaple_active_weight_sched/` | Active collocation RAD + SA-PINN em Laplace 2D | PINNeAPPle **only** |
| 04 | `04_physicsnemo_fno_operator/` | FNO operator learning + multi-GPU | PhysicsNeMo (+ fallback) |
| 05 | `05_combined_fno_digital_twin/` | PhysicsNeMo FNO surrogate → PINNeAPPle UQ + DT | **Combined** |
| 06 | `06_combined_meshgraphnet_valid/` | PhysicsNeMo GNN → PINNeAPPle validação + export | **Combined** |

---

## Instalação

```bash
# PINNeAPPle (este repositório)
pip install -e .

# PhysicsNeMo (opcional — exemplos têm fallback)
pip install nvidia-physicsnemo

# Dependências opcionais para os exemplos
pip install matplotlib scipy
```

## Como responder à pergunta "qual a diferença?"

> **PhysicsNeMo** é uma fábrica de surrogates de alto desempenho otimizada para hardware NVIDIA.
> **PINNeAPPle** é um pipeline completo de pesquisa à produção, com UQ, digital twins, meta-aprendizado
> e validação física — cobrindo domínios muito além de CFD.
>
> **Juntos são mais fortes**: PhysicsNeMo treina rápido e barato; PINNeAPPle transforma o modelo
> em um digital twin inteligente com incerteza quantificada e validação física.
