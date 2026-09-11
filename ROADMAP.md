# Roadmap

Catálogo único de iniciativas propostas para o PINNeAPPle e seu ecossistema
(`PINNeAPPle-arena`, `pinneapple-os`, `veriphysics`, `pinneapple-apps`,
`pinneapple_splash`, `reality2physics`, `physcurator`, `ge3`, `portfolio`).
Consolida duas fontes: o brainstorm de referências externas trazido nesta
sessão e o catálogo de 33 ideias já mantido em CoupleTasks (com Laís) —
sem repetir item já coberto em nenhuma das duas.

Cada entrada usa um destes status, herdados do vocabulário já usado no
catálogo do CoupleTasks e nos `ROADMAP.md` de `veriphysics`/`pinneapple-os`:

- **Já existe** — real, rodando, em algum repo do ecossistema.
- **Sobreposição parcial** — existe uma peça real que cobre parte do escopo.
- **Projeto novo** — ideia, nada construído ainda.
- **Peça em aberto** — a peça está faltando por definição (ex.: GenAItor).

> **Recomendação estratégica (herdada do CoupleTasks):** priorizar o
> **GenAItor** (camada fina de orquestração) sobre os repositórios já
> existentes (`reality2physics`, `PINNeAPPle-arena`, `ge3`, `portfolio`)
> antes de iniciar os domínios novos listados abaixo — ver [[chordiq_portfolio]]
> na memória: a maior parte da "Scientific Intelligence Platform" já existe
> espalhada, o que falta é o roteador que amarra tudo.

---

## 1. Scientific Intelligence Platform — GenAItor + PINNeAPPle + Blender

Núcleo estratégico: um roteador que interpreta um pedido em linguagem
natural, classifica domínio físico + capacidade desejada (simular,
otimizar, prever, descobrir, comparar, decidir) e despacha para o
repositório/motor certo, narrando o resultado.

### GenAItor — Camada de Orquestração
**Peça em aberto.** Agente/roteador central. É a única peça da visão que
ainda não existe de fato — o protótipo de 2024 encontrado em
`barrosyan/pinneaple/genaitor/` é só um gerador de system-prompts via
Flask/llama.cpp, sem lógica de roteamento real (ver memória
`chordiq_portfolio`, correção de 2026-09-11).

### PINNeAPPle Physics AI Arena / Model Benchmarking
**Já existe**, em expansão. Para um mesmo problema físico, treina/avalia
múltiplos modelos (PINN, FNO, DeepONet, GNN) e compara acurácia, resíduo
de PDE, tempo de treino/inferência, memória e generalização OOD — usado
pelo GenAItor para responder "qual modelo devo usar?". Vive em
`PINNeAPPle-arena` (backend + frontend + leaderboard); está recebendo o
catálogo externo de ~94 bibliotecas de terceiros (ver §4, "Noether
Surrogate Benchmark" e a integração `physics-based-references` em
andamento).

### Reality-to-Simulation Agent (Reality2Physics)
**Já existe** (MVP validado, repo `reality2physics`). Converte vídeo/
imagem/sensores em representações físicas estruturadas (campo físico,
parâmetro escalar, PDE) — validado em Navier-Stokes, calor e onda com uma
única arquitetura compartilhada.

### Physics Discovery / Equation Discovery
**Já existe**: `portfolio/pinneapple/inverse_sindy` — EKI + SINDy
redescobre o sistema de Lorenz (100% de acerto na estrutura). Dado dados
observacionais sem a equação explícita, descobre a PDE governante via
regressão simbólica/esparsa restrita por física.

### Uncertainty Quantification Lab
**Sobreposição parcial.** Padronizar a saída de todo modelo Physics AI do
portfólio com intervalo de confiança, comparando calibração, detecção de
OOD e robustez entre PINN/FNO com e sem UQ. Backlog identificado em
`portfolio/pinneapple/missile_aero` ("UQ deferred"). Módulo relacionado:
`pinneapple_analysis.uncertainty`.

### Autonomous Scientific Experimentation
**Sobreposição parcial.** Agente de loop fechado que formula hipótese,
gera candidatos de design, simula, analisa e decide o próximo experimento
sem intervenção humana a cada iteração. Precedente parcial:
`portfolio/pinneapple/self_healing` (`TrainingAdvisor`, auto-retrain,
~20x redução de erro).

### PINNeAPPle Auto-PINN
**Projeto novo.** Dado apenas a formulação matemática de uma PDE (domínio,
condições de contorno/iniciais), gerar automaticamente arquitetura, função
de perda e treinamento do PINN correspondente. Módulo relacionado:
`pinneapple_problemdesign` (já existe um agente NLP→PDE parcial —
`DesignAgent`/`UnifiedPhysicsAgent` — este projeto fecha o ciclo até o
treino real).

### PINNeAPPle Surrogate Factory
**Projeto novo.** Pipeline industrializado: CAD paramétrico → DOE →
execução em lote do solver → dataset → treino de surrogate
(FNO/DeepONet) → validação → registro/deploy, sem intervenção manual.
Ver §3 "Autonomous DOE-CFD" para o blueprint técnico validado em escala
industrial (nTop/CoreWeave) que valida a viabilidade desta ideia.

### PINNeAPPle Model Zoo
**Sobreposição parcial** com `PINNeAPPle-arena` (falta a biblioteca
padronizada por PDE). Coleção padronizada de PDEs e operadores de
referência (Burgers, Navier-Stokes, calor, onda, Poisson, Allen-Cahn,
reação-difusão, Darcy, elasticidade), cada um com dataset, arquitetura,
benchmark e pesos publicados.

---

## 2. CAD generativo

### Spec-to-Solid (LLM → CAD paramétrico)
**Projeto novo.** Agente que recebe especificação em linguagem natural
(ou perfis/pontos, como aerofólios) e gera script CADQuery/Build123D;
quando a topologia falha, recorre à API do OnShape/SpaceClaim para
reparar a geometria. Saída validada automaticamente (sólido fechado,
pronto para malhar em CFD/FEM). Fonte: LinkedIn — Jaydeep Singh
(SpaceClaim scripting + CADQuery/Build123D/NumPy + Gemini). Módulo
relacionado: `pinneapple_design.geometry` (já tem SDF/CSG/mesh/NACA
airfoil — falta o passo LLM→script e o reparo de topologia).
Ver também §7, `pinneapple_llm.cad_draft` (usado pelo produto `Text2Part`
em `pinneapple-apps`) como ponto de partida real já validado.

---

## 3. Infraestrutura de Physics AI / Scientific ML

### PhysicsNeMo Backbone Swap
**Projeto novo.** Avaliar/trocar o NVIDIA PhysicsNeMo como backbone dos
surrogates CFD do ChordIQ (mixing-tank, cloramina), comparando contra a
abordagem PINNeAPPle atual. Já existe um precedente real de comparação
lado a lado: `portfolio/pinneapple/vs_physicsnemo` (PINNeAPPle 4.4%
rel-L2/70k params vs. PhysicsNeMo 1.0%/137k params, ver memória
`chordiq_portfolio`) — este item estende essa comparação para os
surrogates de produção, não só para o card de demonstração.

### Newton Multibody Data Forge
**Projeto novo.** Usar o NVIDIA Newton para acelerar simulações
multibody em GPU e gerar dados sintéticos de treino em escala para os
surrogates físicos existentes.

### Multi-Physics Sandbox
**Projeto novo.** Usar o simulador unificado do Genesis (rigid body + FEM
+ MPM + SPH/PBD, compilador cross-platform Quadrants para
CUDA/ROCm/Metal/**Vulkan**/x86/ARM64) para gerar dados sintéticos de
escoamento/partículas, complementando os dados OpenFOAM que já alimentam
os twins do ChordIQ. Fonte:
[Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world)
(29.9k stars, engine de física unificado para robótica/embodied AI —
verificado nesta sessão via `gh api`). Módulo relacionado:
`pinneapple_simulation.particle_dynamics` (MPM/SPH/rigid-body hoje em
PyTorch puro — Genesis é uma alternativa muito mais madura como backend
plugável) e `pinneapple_worldmodel` (ambientes para agentes). Conecta
também com o backend Vulkan do item "LatticePT" (§7) — mesma direção de
runtime GPU-nativo/multiplataforma.

### Noether Surrogate Benchmark
**Projeto novo.** Framework de transformers pré-construídos para
CFD/aerodinâmica (ex.: AB-UPT no DrivAerML); usar como benchmark antes de
investir em arquitetura própria. Fonte:
[Emmi-AI/noether](https://github.com/Emmi-AI/noether). Alimenta
diretamente o catálogo externo do `PINNeAPPle-arena` (§1).

### GPU Signal-Prep Kit
**Projeto novo.** Reaproveitar o padrão de pipeline GPU-first do
cuPhoton (CuPy/Numba-CUDA) como pré-processamento de sensores de planta
industrial antes de virar input de um surrogate. Fonte:
[nvidia/cuPhoton](https://github.com/nvidia/cuPhoton). Módulo relacionado:
`pinneapple_systems.digital_twin.io` (streams MQTT/Kafka/OPC-UA/Modbus).

### Scientific ML Curriculum Pipeline
**Projeto novo.** Usar a estrutura de 20 capítulos do livro "Data-Driven
Modeling and Scientific Computation" como esqueleto para pipelines de
Scientific ML de referência antes de treinar modelos próprios. Fonte:
[nathankutz/ScientificComputing](https://github.com/nathankutz/ScientificComputing).

### Agentic PDE Debugger
**Projeto novo.** Time multiagente (inspirado no paper ATHENA) que
escolhe o método numérico, implementa, detecta falhas de física
silenciosas (ex.: instabilidade Kelvin-Helmholtz/Rayleigh-Taylor) e
corrige sozinho; aplicável ao auto-tuning dos solvers CFD do roadmap de
cloramina. Fonte: arXiv — ATHENA. Módulo relacionado:
`pinneapple_analysis.verification` (guardrails, conservação) +
`pinneapple_llm.guardrail.PhysicsGuardrail` já fazem parte disso — este
projeto fecha o ciclo com correção automática, não só detecção.

### Paper-to-Repro Benchmark Suite
**Projeto novo.** Pipeline que extrai método e resultados numéricos de
papers do arXiv e usa um agente tipo ATHENA para reproduzi-los, virando um
benchmark contínuo de "quão bem uma IA reproduz física publicada". Fonte:
arXiv — ATHENA + ideia própria. Precedente real direto:
`pinn_reproduction_results` (reprodução de Raissi et al. 2017 e Lu et al.
2019, incluindo bugs reais encontrados e corrigidos) — este projeto
generaliza esse esforço manual em um pipeline automatizado e contínuo.

### CrunchOptimizer/PINNs — SS-Quasi-Newton (SSBFGS/SSBroyden)
**Projeto novo.** Otimizadores quasi-Newton curvature-aware
(SSBFGS/SSBroyden), construídos sobre Optimistix (JAX), para treino de
PINN de alta precisão — vão além do que Adam/L-BFGS entregam. Fonte:
[CrunchOptimizer/PINNs](https://github.com/CrunchOptimizer/PINNs)
(verificado nesta sessão: "Curvature-Aware Optimization for
High-Precision Physics-Informed Neural Networks"). Conexão direta com
achado real já documentado em `pinn_reproduction_results/INSIGHTS.md`:
trocar Adam por L-BFGS já cortou o erro em 100-1000x nos problemas de
Burgers/Allen-Cahn/Schrödinger — SSBFGS/SSBroyden são o próximo passo
natural dessa mesma descoberta. Módulo relacionado:
`pinneapple_neural.trainer` + `pinneapple_tools.compute_backends` (já tem
backend JAX, onde o Optimistix vive nativamente).

---

## 4. Simulação em escala / CFD industrial

### Autonomous DOE-CFD
**Projeto novo**, com blueprint técnico validado em escala industrial.
Geometria paramétrica via signed distance fields (sem falha topológica ao
variar parâmetros) + burst de GPU em nuvem, para rodar milhares de
variantes de reator/tanque automaticamente. Aplicável à otimização do
misturador no roadmap de cloramina. Fonte: nTop / CoreWeave — NASA 2030
Grand Challenge in CFD (verificado nesta sessão: 280 GPUs NVIDIA RTX Pro
Blackwell, 2.400 geometrias × 5 ângulos de ataque = 12.000 simulações em
menos de 24h, zero falhas de geometria, usando LBM sobre grid cartesiano
fixo — meta que a NASA tinha projetado só para 2030, alcançada 4 anos
antes). É o blueprint concreto de como construir a "PINNeAPPle Surrogate
Factory" (§1) em escala real: geometria implícita (SDF) em vez de
b-rep é o que evita a quebra topológica que travaria um DOE massivo, e
LBM é o solver certo para isso porque opera sobre grid fixo sem malha.

### The Well Benchmark Adapter
**Projeto novo.** Usar o dataset físico de 15 TB da Polymathic AI ("The
Well") como pretraining/benchmark externo para testar se os surrogates do
ChordIQ generalizam fora do domínio de mistura/cloramina. Fonte:
[PolymathicAI/the_well](https://github.com/PolymathicAI/the_well).

---

## 5. World models / robótica

### Cosmos Synthetic Perception Pretrainer
**Projeto novo.** Usar os world models do NVIDIA Cosmos (modo Generator)
para gerar vídeo+ação sintéticos de processos industriais, pré-treinando
visão para monitoramento de planta/robótica. Fonte: NVIDIA/Cosmos. Nota:
o modelo de visão da família `nvidia/Cosmos-Reason2-2B` já está em uso
real no `physcurator` (curadoria de dados sintéticos antes do treino de
PINN) — este projeto é o mesmo ecossistema Cosmos aplicado à geração
(em vez de à curadoria).

### Low-Cost Autonomous Robot
**Projeto novo.** Plataforma robótica autônoma de baixo custo (Raspberry
Pi/ESP32 + sensores low-cost) para testes de campo e monitoramento.
Fonte: brainstorm interno — "Criar robô autônomo barato".

---

## 6. Novos domínios físicos (flagship demos)

### Space Debris Tracking & Collision Risk
**Sobreposição parcial**: `ge3` (RK4 3DOF/ISA-76/Barrowman). Trajetória
orbital com perturbações (drag atmosférico, J2, pressão de radiação
solar) + surrogate PINNeAPPle + estimativa de risco de colisão,
visualizado em 3D. Nota: `pinneapple-apps/satellite_conjunction_
screening` (SatScreen) já cobre boa parte disso como produto comercial
(CW relative motion, J2, Kepler, CR3BP + fórmula própria de Pc) — este
item é o flagship demo público, SatScreen é a versão paga.

### Airfoil / Engineering Design Optimization
**Já existe**: `portfolio/pinneapple/missile_aero` (Cp R² 0.99) +
`portfolio/01_automotive_aero`. Geometria → CFD dataset → surrogate
(FNO/DeepONet) → otimização de milhares de designs → melhor geometria
(lift/drag).

### Astrophysical Parameter Discovery
**Sobreposição parcial**: `ge3` (lightkurve/BLS) — falta a camada de UQ.
Inferir massa, raio, temperatura, metalicidade e idade estelar a partir
de curva de luz observada, com quantificação de incerteza.

### Satellite Thermal Digital Twin
**Projeto novo.** Modelo térmico de satélite (radiação solar/terrestre,
orientação, propriedades de material) aprendido por PINNeAPPle, com
recomendação de orientação que minimiza estresse térmico.

### Drilling Hydraulics Digital Twin / Smart Mud Pump
**Sobreposição parcial**: base já existe em
`~/Documents/GitHub/biaml/database/geometries` (BOP, desanders, chokes).
Estado de hidráulica de perfuração (ECD, pressão, vazão, temperatura) em
tempo real + otimização da bomba de lama para minimizar energia mantendo
ECD dentro do limite de segurança.

### Digital Twin Setorial (padrão replicável)
**Sobreposição parcial**: `portfolio/03_mixing_tank` e
`portfolio/02_wind_turbine`. Pipeline único reaproveitável —
sensores/CAD → PINNeAPPle → surrogate → predição → otimização — aplicado
a vários setores: bateria/EV (gestão térmica), wind farm (yaw, +8,7%
potência), HVAC predial (-21% energia), trocador de calor (+8%
recuperação), estrutural (manutenção preditiva), power grid, braço
robótico (MPC), fábrica (anomalia) e bombeamento industrial.

### Flood Prediction / Urban Hydrology
**Projeto novo.** Modelo físico de chuva + terreno (DEM) + hidrologia →
mapa previsto de profundidade de inundação ao longo do tempo, comparado a
interpolação tradicional.

### Climate Downscaling
**Projeto novo.** Modelo Physics AI que refina resolução de simulação
climática (25 km → 1 km) para temperatura, precipitação, vento e
umidade, comparado contra interpolação e ML puro.

### Materials Inverse Design
**Projeto novo.** Dado um requisito de propriedade alvo (ex.:
condutividade térmica), buscar composição/microestrutura candidata via
formulação física + PINNeAPPle + busca.

### Biomechanics AI
**Projeto novo.** Prever distribuição de pressão/deformação em uma
articulação a partir de geometria, propriedades de material e
carregamento, com Neural Operator substituindo FEM tradicional (17 min →
0,4 s por caso).

### Solar Farm / Solar Panel Optimization
**Projeto novo.** Prever irradiância, temperatura e cobertura de nuvens,
e otimizar orientação dos painéis, cronograma de limpeza e resfriamento
para maximizar energia e minimizar custo operacional.

---

## 7. GPU-native / verificação de hardware / plataformas comerciais (referências externas, fora do catálogo CoupleTasks)

### LatticePT Boltzmann Reactor: Process Studio
**Projeto novo.** App CFD+CHT (conjugate heat transfer) 100%
GPU-nativo em navegador: geometria paramétrica em OpenCascade → malha de
casca QUAD8 → volume fluido voxelizado por marching cubes → LBM+LES com
immersed boundary method para agitadores móveis → coeficientes de
convecção nas paredes molhadas pelo processo (correlações Kawase-Moo &
John Thomas) → lado da jaqueta via correlação de Gnielinski → acoplamento
explícito de condução através dos elementos de casca. Roda localmente no
Chrome, sem servidor, e é deployável via **Vulkan** em qualquer GPU
(AMD/Intel/NVIDIA/Apple). Fonte: post do LinkedIn (LATTICEPT, "Reactor
Lab"), construído com Anthropic Fable 5.1 em 4 semanas. Módulos
relacionados: `pinneapple_simulation.numerical_solvers` (já tem LBM),
`pinneapple_design.geometry` (SDF/CSG — falta OpenCascade paramétrico),
`pinneapple_tools.compute_backends` (PyTorch+JAX hoje — falta um backend
WebGPU/Vulkan para rodar no browser), `pinneapple_systems.digital_twin`
(agitador móvel + jaqueta térmica = twin clássico). Mesma direção do
"Multi-Physics Sandbox" (§3, Genesis) no eixo de runtime GPU-nativo
multiplataforma.

### OpenV — Verification-first hardware engineering
**Projeto novo.** Pipeline open-source onde um agente (Astra) propõe
requisitos/design de hardware, ferramentas externas de CAD/cálculo/solver
produzem evidência, e uma comparação determinística decide
PASS/FAIL/UNKNOWN — o LLM nunca avalia seu próprio trabalho. Inclui
invalidação automática de evidência dependente quando o design muda, e
histórico rastreável de cada experimento (hipótese → mudança → efeito
esperado → efeito real) via um modelo de engenharia persistente (Dalus
via MCP). Fonte: [sebastianvkl/OpenV](https://github.com/sebastianvkl/OpenV)
(verificado nesta sessão). Conexão direta: é o mesmo princípio do
`veriphysics` (execução + verificação + evidência, Decision Record,
trust score) aplicado a hardware/CAD em vez de PDEs — o mecanismo de
"invalidação automática de evidência dependente ao mudar o design" é
genuinamente novo e vale portar para `pinneapple_analysis.verification
.evidence_graph`/`provenance`.

### Luminary Cloud — Physics AI Stack (referência competitiva)
**Projeto novo** (pesquisa de benchmark, não integração de código).
Todo o conteúdo relevante em luminary.ai/resources: modelos "Physics AI"
treinados sobre simulação para acelerar design de engenharia
(aeroespacial, automotivo, defesa), arquitetura mesh-independent
("Luminary-SMART"), UQ/validação, cases como "Physics AI Cuts Aircraft
Design Costs By 80%" (verificado nesta sessão via fetch da página).
Uso recomendado: benchmark de posicionamento para `pinneapple-apps`
(especialmente `VerifiedPhysics`/`physics_verification_engine`) e para o
`PINNeAPPle-arena` — no espírito anti-fabricação já documentado no
`tool_recommendation.py`, qualquer claim tipo "-80% custo" deveria ser
tratada como claim de terceiro a verificar independentemente, não
absorvida como fato.

### Reality2Physics estendido — Digital Complex-Systems Benchmark
**Projeto novo** (o mais especulativo e mais amplo desta lista).
Generaliza o `reality2physics` (hoje: vídeo/sensor → campo físico +
parâmetro escalar, validado em 3 PDEs) para sistemas complexos onde a
"física" é a dinâmica interna de um organismo/sistema, seguindo sempre o
mesmo padrão: **Observar → Reconstruir → Inferir → Simular → Perturbar →
Validar**. Ponto de partida recomendado (mais tratável, conectoma
completo + músculos + circuitos já públicos via OpenWorm):
**C. elegans Digital Twin** — connectome (~300 neurônios) → grafo neural
→ modelo de neurônio → modelo de músculo → biomecânica do corpo →
ambiente → comportamento; validação = "o modelo reconstruído reproduz
locomoção real?". Extensões mapeadas na mesma sessão de brainstorm
(cada uma como capítulo futuro do mesmo benchmark, não itens separados):
- **Drosophila** via FlyWire/BANC/hemibrain (~140k neurônios, 50M+
  sinapses) — conectoma maduro, mas só estrutura; pesos sinápticos,
  neurotransmissores e corpo ainda precisam ser inferidos/calibrados.
- **MICrONS** (córtex visual de camundongo, Allen Institute) — o dataset
  mais forte por combinar estrutura *e* função nos mesmos neurônios
  (~1mm³, centenas de milhões de sinapses, dezenas de milhares de
  neurônios com atividade registrada).
- **Mind2Physics** — tratar estado psicológico como sistema dinâmico
  parcialmente observável (θ\* = argmin L(comportamento_sim,
  comportamento_real)) em vez de classificador — mesmo paradigma
  PINN/inverse-problem aplicado a séries temporais comportamentais.
- **PINNeAPPle Anomaly Lab** — mesmo pipeline aplicado a alegações
  paranormais/anômalas (casas "assombradas", UAP, EVP) como
  reconstrução física multi-sensor + teste cego + classificação em
  Explained/Unresolved/Reproducibly anomalous — valor científico está no
  rigor do protocolo, não na conclusão.
- **PINNeAPPle Football Dynamics** — tracking de 22 jogadores + bola como
  sistema dinâmico multiagente; simulação contrafactual ("e se o
  jogador X estivesse 3m à esquerda?"), papel emergente de jogador,
  time como rede/campo de influência.
Todas compartilham a mesma pergunta central: dado um sistema
parcialmente observável, consigo reconstruir a dinâmica que produz o
comportamento observado, e depois perturbar essa reconstrução para
prever algo que ainda não vi? Módulo relacionado: `pinneapple_worldmodel`
(agentes/ambientes) + `reality2physics` como base de pipeline.

---

## Como este roadmap se conecta ao resto do ecossistema

- §1 (Scientific Intelligence Platform) é onde a maior parte do valor já
  construído (`reality2physics`, `PINNeAPPle-arena`, `ge3`, `portfolio`)
  já vive — a recomendação estratégica do CoupleTasks é amarrar isso via
  GenAItor antes de abrir os domínios novos de §6.
- §3 e §4 (infraestrutura/CFD em escala) alimentam diretamente a
  "PINNeAPPle Surrogate Factory" (§1) com peças reais (Genesis, nTop/
  CoreWeave, Noether) em vez de reinventar cada uma do zero.
- §7 é o bucket de referências externas trazidas fora do catálogo
  CoupleTasks nesta sessão — cada uma tem uma conexão explícita com um
  módulo real do PINNeAPPle ou com `veriphysics`/`pinneapple-apps`,
  nunca uma integração "porque é legal".
