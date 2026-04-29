# pinneaple_learning — Your Physics AI Learning Path

This module is your guide through Physics AI: from your very first physics-informed neural network all the way to production-scale deployments with NVIDIA PhysicsNeMo.

---

## Quick start

```python
import pinneaple_learning as pl

# Find out where you are
pl.where_am_i()

# Get your personalised learning path
pl.learning_path(tier=1)   # Explorer
pl.learning_path(tier=2)   # Experimenter
pl.learning_path(tier=3)   # Builder

# Explore Physics AI concepts
pl.list_topics()
pl.explain("pinn")
pl.explain("neural_operator")
pl.explain("inverse_problem")
pl.explain("uq")
pl.explain("physicsnemo")
```

---

## The three tiers

### Tier 1 — Explorer
> *"I understand the physics. I want to see what AI can do with it."*

- **Prerequisites:** Basic Python · NumPy · some calculus and ODEs/PDEs
- **Start here:** [`tier1_explorer/README.md`](tier1_explorer/README.md)
- **First step:** `python examples/getting_started/01_harmonic_oscillator.py`

```python
import pinneaple_learning.tier1_explorer as t1
t1.quickstart()       # orientation guide
t1.what_is_pinn()    # core concept
t1.what_is_loss()    # loss function breakdown
```

### Tier 2 — Experimenter
> *"I have run PINNs before. I want to compare approaches and build experiments."*

- **Prerequisites:** Tier 1 complete · PyTorch basics · understanding of at least one PDE
- **Start here:** [`tier2_experimenter/README.md`](tier2_experimenter/README.md)

```python
import pinneaple_learning.tier2_experimenter as t2
t2.architecture_guide()   # PINN vs FNO vs DeepONet vs GNN
t2.pipeline_anatomy()     # full 6-stage experiment pipeline
```

### Tier 3 — Builder
> *"I have a validated approach. I want to scale it and prepare for production."*

- **Prerequisites:** Tier 2 complete · understanding of distributed training · deployment basics
- **Start here:** [`tier3_builder/README.md`](tier3_builder/README.md)

```python
import pinneaple_learning.tier3_builder as t3
t3.production_checklist()    # what to verify before shipping
t3.physicsnemo_readiness()   # are you ready to migrate?
```

---

## The destination: NVIDIA PhysicsNeMo

When you complete Tier 3, you are ready for **NVIDIA PhysicsNeMo** — the enterprise platform for Physics AI:

```python
import pinneaple_learning.physicsnemo_roadmap as nm
nm.migration_overview()   # full migration path
```

See also:
- [`physicsnemo_roadmap/README.md`](physicsnemo_roadmap/README.md) — why PhysicsNeMo for production
- [`physicsnemo_roadmap/migration_guide.md`](physicsnemo_roadmap/migration_guide.md) — step-by-step migration

---

## Learning path diagram

```
Your physics problem
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                       PINNeAPPle                              │
│                                                               │
│  Tier 1: Explorer        Tier 2: Experimenter                 │
│  ─────────────────       ──────────────────────────           │
│  Run first PINN          Compare architectures                │
│  Understand losses       Solve inverse problems               │
│  Explore parameters      Quantify uncertainty                 │
│  10 getting_started/     Active learning                      │
│  examples                templates/16–29                      │
│                                                               │
│  Tier 3: Builder                                              │
│  ──────────────────────────────────────────────               │
│  Scale with DDP                                               │
│  Export to ONNX                                               │
│  REST serving                                                 │
│  Digital twin                                                 │
│  templates/07,22,23,24,33,34,35                               │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                  NVIDIA PhysicsNeMo                           │
│                                                               │
│  Multi-GPU training (512+ GPUs)                               │
│  Triton Inference Server                                      │
│  NVIDIA Omniverse integration                                 │
│  Enterprise support + SLA                                     │
│  Industry-validated models                                    │
└───────────────────────────────────────────────────────────────┘
```

---

## Available concepts

```python
import pinneaple_learning as pl
pl.list_topics()
```

```
Tier 1 — Explorer
  • pinn              — Physics-Informed Neural Networks
  • physics_ai_pipeline — The end-to-end Physics AI workflow

Tier 2 — Experimenter
  • neural_operator   — FNO and DeepONet
  • uq                — Uncertainty Quantification
  • inverse_problem   — Recovering unknown PDE parameters
  • rom               — Reduced Order Models

Tier 3 — Builder / Production
  • physicsnemo       — NVIDIA PhysicsNeMo
  • digital_twin      — Digital twins and sensor assimilation
```

---

## Module structure

```
pinneaple_learning/
├── __init__.py              ← pl.explain(), pl.list_topics(), pl.learning_path()
├── learning_path.py         ← LearningPath, where_am_i()
├── concepts.py              ← ConceptGuide, explain(), list_topics()
├── README.md                ← this file
│
├── tier1_explorer/
│   ├── __init__.py
│   ├── guide.py             ← quickstart(), what_is_pinn(), what_is_loss()
│   └── README.md            ← comprehensive Explorer guide
│
├── tier2_experimenter/
│   ├── __init__.py
│   ├── guide.py             ← architecture_guide(), pipeline_anatomy()
│   └── README.md            ← comprehensive Experimenter guide
│
├── tier3_builder/
│   ├── __init__.py
│   ├── guide.py             ← production_checklist(), physicsnemo_readiness()
│   └── README.md            ← comprehensive Builder guide
│
└── physicsnemo_roadmap/
    ├── __init__.py
    ├── guide.py             ← migration_overview()
    ├── README.md            ← why PhysicsNeMo for production
    └── migration_guide.md   ← step-by-step migration walkthrough
```
