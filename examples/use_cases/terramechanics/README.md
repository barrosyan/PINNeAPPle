# Terramechanics for Rovers — Bekker-Wong PINN Surrogate

> **Use case:** Real-time wheel-soil interaction surrogate for lunar/planetary rover simulation.
> **Target integration:** LunCoSim (OmniLRS) + PINNeAPPle digital twin.

---

## Why this use case?

The Bekker-Wong terramechanics model is the industry standard for off-road vehicle
dynamics on deformable soil. It describes how a rigid wheel sinks into regolith,
builds shear stress across its contact patch, and generates traction forces.

**The problem:** The numerical Bekker-Wong solver requires `scipy.integrate.quad`
calls over the contact patch for every (slip, sinkage) evaluation — too slow for
real-time multi-wheel simulation in LunCoSim.

**The solution:** A PINN surrogate trained on Bekker-Wong reference data,
plus physics-informed constraints that embed the governing equations. Once trained,
inference is a single forward pass (~0.1 ms vs ~20 ms for numerical integration).

---

## Physical Model

```
Wheel (radius R, width b) rolling on deformable soil
       ────────────────────────────────────────
              θ_f   θ_m    θ_r
       ┌─────/────────╲─────╲
       │              soil  │
       │     contact patch   │
       └───────────────────┘

Normal stress  σ(θ):  Bekker pressure-sinkage law
                      σ_max = [(c·k_c/b) + (ρ·g·k_φ)] · (R/b)^n

Shear stress   τ(θ):  Mohr-Coulomb + exponential shear displacement (Wong 1978)
                      τ(θ) = (c + σ(θ)·tan φ) · (1 − exp(−j(θ)/K))

Shear displ.   j(θ):  j = R·[(θ_f−θ) − (1−s)·(sin θ_f − sin θ)]

Forces (integration over contact patch width b):
  F_x = R·b · ∫[τ cos θ − σ sin θ] dθ    ← drawbar pull
  F_z = R·b · ∫[σ cos θ + τ sin θ] dθ    ← normal load
  M_y = R²·b · ∫ τ dθ                     ← driving torque
```

### Key parameters (GRC-1 lunar regolith simulant, compacted)

| Symbol | Name | Value | Unit |
|--------|------|-------|------|
| c | Cohesion | 1 400 | Pa |
| φ | Internal friction angle | 30 | ° |
| K | Shear deformation modulus | 0.018 | m |
| k_c | Bekker cohesion modulus | 1 370 | N/m^(n+1) |
| k_φ | Bekker friction modulus | 814 000 | N/m^(n+2) |
| n | Sinkage exponent | 1.0 | — |
| g | Gravity (lunar) | 1.62 | m/s² |

---

## Pipeline (without code)

```
Step 1 — Parameter definition
      SoilParams (Bekker-Wong)  +  WheelParams (geometry)
      Lunar GRC-1 regolith simulant values

Step 2 — Bekker-Wong numerical solver (scipy.integrate.quad)
      For each (slip_ratio s, sinkage z):
        ├─ compute contact angles θ_f, θ_m, θ_r
        ├─ compute σ_max (Bekker pressure-sinkage law)
        ├─ integrate σ(θ) over upper and lower contact zones
        ├─ integrate τ(θ) via Mohr-Coulomb + shear displacement j(θ)
        └─ return F_x, F_z, M_y

Step 3 — Dataset generation
      Grid sweep: s ∈ [0, 0.75] × z ∈ [0.002, 0.06 m]
      + Latin Hypercube Sampling for better coverage
      ≈ 2 200 (slip, sinkage) → (F_x, F_z, M_y) pairs
      Normalize all inputs and outputs to [−1, 1]

Step 4 — PINN architecture (TerraMechanicsPINN)
      Input: (s_norm, z_norm)  — 2D
      Encoding: Random Fourier Features  B ∈ ℝ^{20×2}  →  [sin(Bx), cos(Bx)]  ← 40D
      Hidden: 5 ResNet blocks × 128 units, Tanh activation
      Output: (Fx_norm, Fz_norm, My_norm)  — 3D

Step 5 — Physics residuals (PINN constraints)
      R1  Zero-slip BC:       F_x(s=0, z) = 0          (no drawbar without slip)
      R2  Mohr-Coulomb limit: F_x ≤ c·A + F_z·tan(φ)   (shear strength cap)
      R3  Monotonicity:       ∂F_x/∂s ≥ 0  for s ∈ [0, 0.4]  (autograd constraint)
      R4  Torque coupling:    M_y ≥ R · F_x              (thermodynamic consistency)

Step 6 — Multi-loss training (Adam + CosineAnnealingLR)
      L_total = w_data·L_data + w_R1·L_R1 + w_R2·L_R2 + w_R3·L_R3 + w_R4·L_R4
      4 000 epochs, batch size 512, lr = 5×10⁻⁴
      Gradient clipping ||∇||₂ ≤ 1.0

Step 7 — Evaluation & visualization
      ├─ Traction curves: F_x, F_z, M_y vs slip at 4 sinkage levels
      │    PINN surrogate (dashed) vs Bekker-Wong numerical (solid)
      ├─ 2D force maps: F_x, F_z, η_T = F_x/F_z over (s, z) parameter space
      └─ Training loss: total, data MSE, validation, 4 physics residuals

Step 8 — Export for LunCoSim integration
      TorchScript model  →  terramechanics_surrogate.pt
      Normalization metadata + soil/wheel params  →  surrogate_metadata.json
      API: predict(slip, sinkage) → (Fx, Fz, My)  in ~0.1 ms
```

---

## Physics constraints detail

### R1 — Zero-drawbar at zero slip
At s = 0 (perfect rolling), there is no relative motion between wheel and soil.
The Mohr-Coulomb shear stress τ = (c + σ tan φ)(1 − exp(−j/K)) vanishes when
j(θ) = 0, which happens at s = 0. Therefore F_x = 0.
This is a Dirichlet BC in slip space.

### R2 — Mohr-Coulomb traction limit
The maximum traction the soil can provide is bounded by its shear strength:
F_x_max = c·A_contact + F_z·tan(φ)
This is enforced as a one-sided soft penalty (ReLU²).

### R3 — Monotonicity in slip (pre-peak regime)
For small slip ratios (s < 0.4), traction increases with slip before reaching the
peak. This is enforced via ∂F_x/∂s ≥ 0 using autograd (second-order gradients).

### R4 — Torque-force thermodynamic constraint
The driving torque M_y must be at least R times the drawbar pull F_x, because:
M_y = R·(F_x + F_rolling)  and  F_rolling ≥ 0
In normalized space: My_norm ≥ R_factor · Fx_norm

---

## LunCoSim Integration

```python
# In LunCoSim / OmniLRS:
import json
import torch

ts_model = torch.jit.load("terramechanics_surrogate.pt")
meta     = json.load(open("surrogate_metadata.json"))

lo_x, hi_x = torch.tensor(meta["norm_x_lo"]), torch.tensor(meta["norm_x_hi"])
lo_y, hi_y = torch.tensor(meta["norm_y_lo"]), torch.tensor(meta["norm_y_hi"])

def predict_forces(slip: float, sinkage_m: float):
    x = torch.tensor([[slip, sinkage_m]])
    x_n = 2.0 * (x - lo_x) / (hi_x - lo_x).clamp(min=1e-12) - 1.0
    with torch.no_grad():
        y_n = ts_model(x_n)
    y = (y_n + 1.0) * (hi_y - lo_y) / 2.0 + lo_y
    Fx, Fz, My = y[0, 0].item(), y[0, 1].item(), y[0, 2].item()
    return Fx, Fz, My  # [N], [N], [N·m]
```

---

## Prerequisites

```bash
pip install pinneapple          # or: pip install -e .  from repo root
pip install scipy              # Bekker-Wong numerical integration
pip install matplotlib numpy   # visualization
```

---

## References

- Bekker, M.G. (1969). *Introduction to Terrain-Vehicle Systems*. U Michigan Press.
- Wong, J.Y. (1978). *Theory of Ground Vehicles*. Wiley.
- OmniLRS `terramechanics_solver.py` — https://github.com/OmniLRS/OmniLRS
- Traction Performance Evaluation for Rashid-1 Rover (ResearchGate 2025)
- Peiret et al. (2018). Simulation techniques for terramechanics at the wheel-soil interface. ISTVS.
- Raissi, M. et al. (2019). Physics-informed neural networks. *J. Comp. Physics*.

---

## File structure

```
examples/use_cases/terramechanics/
├── README.md                          ← this file — pipeline description
├── terramechanics_rover_pinn.py       ← complete implementation (run this)
└── outputs/
    ├── 01_traction_curves.png         ← F_x, F_z, M_y vs slip (PINN vs Bekker-Wong)
    ├── 02_force_maps.png              ← 2D (slip, sinkage) force heatmaps
    ├── 03_training_history.png        ← loss convergence + physics residuals
    ├── terramechanics_surrogate.pt    ← TorchScript export for LunCoSim
    └── surrogate_metadata.json       ← normalization + soil/wheel params
```
