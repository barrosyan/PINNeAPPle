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
regressão simbólica/esparsa restrita por física. Este item é a semente de
uma vertente muito maior — geometria/manifold escondido, transições
ordem↔caos, invariantes desconhecidos, causalidade estrutural, eventos
extremos — consolidada em **§8, "Structure Discovery / Chaos-to-Law"**.

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

### Corpus de referência PIELM/XTFC/TFC/OpInf (material recebido, não integrado)
**Peça em aberto.** `PINNeAPPle-Talk/resources/` (ver `MANIFEST.md`)
recebeu nesta sessão implementações reais em MATLAB de PIELM e X-TFC para
PDEs de advecção-difusão (`Advection-Diffusion_PIELM-&-XTFC/`,
`PDE_matlab/`, `Laura/`, `codes/`, `common/`), mais uma pasta de
referências dedicada a TFC (`References/TFC/`) e papers de Operator
Inference/POD (`OpInf_summary_2022{a,b}.pdf`, `POD-ROM.pdf`, em
`deeponet_papers_and_notebooks/`). Conecta diretamente com módulos já
reais: `pinneapple_neural/architectures/rom/{opinf,pod}.py` (existentes)
e o `TFC/ELM` já usado em produção pelo `HelioTFC` (`pinneapple-apps`).
Vale avaliar se o código MATLAB tem algo que valide/estenda o `OpInf`/
`POD` já implementados, antes de tratar como só leitura de referência.

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

### Fire Spread / Wildfire PINN
**Projeto novo**, com uma semente de código real. Existe um script
standalone `Fire_PINN_PlusLowDifusion.py` em
`PINNeAPPle-Talk/resources/deeponet_papers_and_notebooks/` (ver
`PINNeAPPle-Talk/resources/MANIFEST.md`) — um PINN para dinâmica de
propagação de fogo/baixa difusão, nunca integrado a este portfólio.
Avaliar se vale a pena portar como ponto de partida em vez de começar do
zero.

### Optimal Control / Nuclear & Radiative Transport (domínios não cobertos)
**Peça em aberto — nenhum repo do ecossistema cobre isso hoje.** Um
levantamento de referências recebido nesta sessão (`PINNeAPPle-Talk/
resources/References/`, ver MANIFEST) tem pastas inteiras dedicadas a
controle ótimo/HJB/GNC/controle adaptativo (`Optimal Control /`,
`Roberto-Suggestions/`) e a transporte nuclear/radiativo/equações de
cinética pontual (`Transport/{Radiative,Neutron,PKE}`) — nenhum dos dois
domínios existe em nenhum repo do PINNeAPPle-Labs hoje. Não é
necessariamente um novo produto, mas vale decidir deliberadamente se
algum dos dois merece entrar no catálogo de domínios físicos antes de
continuar acumulando referência sem repo correspondente.

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

## 8. Structure Discovery / Chaos-to-Law — nova vertente

Trazido por Yan nesta sessão como uma pergunta diferente da que o
portfólio normalmente faz. Em vez de "como eu preveja este sistema",
perguntar **"que estrutura mais simples explica este sistema"** —
geometria intrínseca, dinâmica reduzida, causalidade, invariantes,
regimes de caos/ordem, eventos extremos. O item "Physics Discovery /
Equation Discovery" (§1) já é uma instância real e validada disso; esta
seção generaliza a mesma pergunta para eixos que o portfólio ainda não
cobre, verificado diretamente no código desta sessão (não assumido).

Pipeline-alvo da vertente inteira:

```
sistema observado
   ├─ geometria  (manifold/TDA)
   ├─ dinâmica   (Koopman/DMD/SINDy)  ── já existe, ver abaixo
   └─ causalidade (GNN/NOTEARS/PCMCI)
        │
   estrutura latente
        │
   invariantes / leis / regimes
        │
   modelo físico → PINN / FNO / GNN (motor já existente do PINNeAPPle)
```

### Dynamics-to-Law: SINDy / Koopman / DMD / HAVOK / POD / OpInf
**Já existe**, o pilar mais maduro desta lista de longe.
`pinneapple_neural/architectures/rom/` já implementa `SINDy`,
`DynamicModeDecomposition`, `HAVOK` (Hankel-DMD via delay embedding),
`KoopmanAutoencoder` (uma segunda implementação vive em
`reservoir_computing/koopman.py`), `OperatorInference`, `POD`,
`NeuralROM`, `ROMHybrid` e `DeepUQROM`, todos catalogados via
`ROMCatalog` (`rom/registry.py`). O precedente citado em §1
(`inverse_sindy` redescobrindo Lorenz via EKI+SINDy, 100% de acerto
estrutural) é a validação ponta a ponta deste pilar — é literalmente o
"problema concreto 1" do brainstorm desta sessão (Lorenz → descoberta
automática da estrutura), só que já feito. **Falta**: um benchmark
dedicado comparando SINDy vs. Koopman vs. DMD/HAVOK na mesma bateria de
sistemas (ver "Bateria de validação" abaixo) com métrica de *acerto
estrutural*, não só erro numérico — encaixe natural em
`PINNeAPPle-arena` (§1), no mesmo espírito do benchmark PINN/FNO/DeepONet
que já existe lá.

### Symbolic regression livre (estilo PySR / AI Feynman)
**Peça em aberto.** `pinneapple_neural/trainer/graybox.py` já antecipa a
ideia no próprio docstring do `GrayBoxNet` ("if the term is later
distilled into a closed-form expression, e.g. via symbolic regression")
mas não implementa busca simbólica livre — hoje toda "descoberta de
equação" do portfólio é restrita a uma base de termos conhecida (SINDy)
ou a uma rede substituta (gray-box), nunca uma busca evolutiva por
expressão fechada como PySR/AI Feynman. Projeto novo:
`pinneapple_neural.architectures.symbolic` — wrapper sobre PySR com
verificação determinística de erro de ajuste antes de aceitar qualquer
expressão (mesmo princípio anti-fabricação do resto do portfólio), com
uma rota explícita para "distilar" um `GrayBoxNet` já treinado numa
expressão fechada.

### Geometria / manifold escondido
**Projeto novo — gap confirmado no código.** Nenhuma implementação de
UMAP, Isomap, Diffusion Maps ou autoencoder-como-manifold-discovery
encontrada no repo (`POD` é hoje a única redução de dimensionalidade, e é
linear). Diffusion Maps é o candidato mais interessante trazido nesta
sessão: descobre a geometria intrínseca de dados de alta dimensão sem
assumir linearidade — complementa, não substitui, o `POD`/`DMD` lineares
já existentes. Composição proposta: dados observados → alta dimensão →
Diffusion Maps/UMAP → coordenadas intrínsecas → alimentar como input do
`SINDy`/`KoopmanAutoencoder` já existentes (manifold discovery + SINDy +
PINN). Módulo relacionado: novo `pinneapple_neural.architectures.manifold`
ao lado de `rom/`. Referência já disponível: "Elements of Dimensionality
Reduction and Manifold Learning" (Ghojogh, Crowley, Karray, Ghodsi,
Springer 2023) está duplicado em `PINNeAPPle-Talk/resources/
deeponet_papers_and_notebooks/Papers e Documentos/` e em
`PINNeAPPle-Talk/resources/cfd_pde_neuralnets_notebooks/` (ver MANIFEST).

### Invariant Discovery Engine
**Projeto novo.** Hoje o portfólio só *impõe* invariantes já conhecidos
(ex.: divergente nulo em `reality2physics`, leis de conservação como
termo de perda nos PINNs) — nada *descobre* uma quantidade conservada
desconhecida a partir de trajetórias observadas. Ideia concreta desta
sessão: treinar uma rede pequena `I_θ(x)` penalizando `dI_θ/dt` ao longo
de trajetórias reais e, depois, tentar destilar `I_θ` numa expressão
simbólica via o item de symbolic regression acima — a mesma composição
"descobrir → destilar" do resto desta seção. Conecta com `veriphysics`:
um invariante descoberto e destilado é exatamente o tipo de claim que o
Decision Record/trust score do `veriphysics` deveria verificar
independentemente antes de ser aceito como real, não só o PINNeAPPle
produzindo o número.

### Transição ordem↔caos (Lyapunov, RQA, bifurcação, entropia)
**Projeto novo — gap confirmado no código.** Nenhuma métrica de caos
(maior expoente de Lyapunov, entropia de Kolmogorov-Sinai/permutação,
dimensão de correlação, recurrence plots/RQA, seções de Poincaré,
análise de bifurcação) encontrada no repo. É infraestrutura de
diagnóstico, não um modelo — barata de construir e reutilizável por
qualquer item acima (ex.: usar RQA para decidir automaticamente se um
sistema está no regime em que SINDy/Koopman conseguem generalizar, antes
de gastar treino de verdade). Módulo relacionado: novo
`pinneapple_analysis.chaos_metrics`.

### Causal discovery estrutural (GNN / NOTEARS / PCMCI)
**Projeto novo — gap confirmado no código.** Nenhum algoritmo de
descoberta causal (NOTEARS, PCMCI+, LiNGAM, Neural Relational Inference,
GNN causal) encontrado no repo. Pergunta central: dado `x_1(t), ...,
x_n(t)` de um sistema de alta dimensão, existe uma estrutura causal
esparsa por trás da bagunça estatística? Conecta com `pinneapple_worldmodel`
(agentes/ambientes, já citado em §7 para os digital twins biológicos) —
descoberta causal seria o passo que precede a construção de qualquer um
daqueles twins a partir de dados observacionais puros, em vez de assumir
a topologia do grafo a priori.

### Extreme events / rare-event discovery
**Projeto novo.** Nenhuma infraestrutura de Extreme Value Theory,
rare-event/importance sampling ou large deviation theory encontrada.
Em vez de estudar o comportamento médio, estudar o que produz os eventos
raros (turbulência extrema, falha industrial, crash) — pergunta natural
para o mesmo domínio industrial que já motiva `pinneapple_systems.digital_twin`
e os produtos de monitoramento do `PINNeAPPle-apps`. Ponto de entrada
mais barato: aplicar EVT sobre os mesmos dados de sensor/SCADA que já
alimentam twins industriais existentes (ex. `shinagawa-ai-platform`,
hoje em `ChordIQ-tech` — o método é agnóstico a onde os dados moram)
antes de qualquer simulação nova de rare-event.

### Bateria de validação (extensão do PINNeAPPle-arena)
**Projeto novo**, extensão natural do `PINNeAPPle-arena` (§1) — antes de
tratar "Structure Discovery" como produto, validar contra ground truth
conhecida, no mesmo espírito anti-fabricação do `PINNeAPPle-Research`
(que só pontua contra problemas já resolvidos). Bateria mínima trazida
nesta sessão:
1. **Lorenz → recuperação da equação** — já validado via `inverse_sindy`
   (ver acima); vira o baseline "fácil" da bateria.
2. **Navier-Stokes turbulento → detecção automática de troca de regime**
   (POD + Koopman + Lyapunov + clustering, todos já existentes ou
   listados acima).
3. **Sistemas multiestáveis** — aprender bacias de atração automaticamente.
4. **Rare events** — caminho mais provável de um estado normal a um
   evento extremo.
5. **Invariant discovery cego** — dado só `x(t), y(t), z(t)`, redescobrir
   o que é conservado.
6. **Generalização entre regimes nunca vistos** (treinar em `Re_1, Re_2`,
   testar em `Re_3`) — testa se o método descobre a lei, não memoriza o
   regime.
7. **Universal Structure Discovery** (o item mais ambicioso): dado
   qualquer sinal (série temporal, imagem, grafo, simulação) sem dizer a
   matemática a priori, decidir sozinho qual das perguntas acima se
   aplica — e produzir uma explicação, não só uma previsão.

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
- §8 (Structure Discovery / Chaos-to-Law) generaliza o item "Physics
  Discovery / Equation Discovery" de §1: a metade "dinâmica" já existe de
  verdade (`rom/` — SINDy, Koopman, DMD, HAVOK, POD, OpInf), a metade
  "geometria/causalidade/regime/evento extremo" é gap confirmado no
  código, não suposição. A bateria de validação proposta em §8 é o
  candidato mais natural para o próximo ciclo de expansão do
  `PINNeAPPle-arena` (§1), e `reality2physics` (§1, §7) é a superfície de
  aplicação onde a "camada de descoberta de PDE" do seu próprio roadmap
  (`README.md`, seção Roadmap) deveria consumir este pilar em vez de
  reimplementá-lo.
