"""06 — Engineering presets showcase: aerospace, automotive, PC cooling, datacenter.

What this demonstrates
----------------------
- Loading pre-defined engineering problems from pinneaple_environment
- Inspecting problem conditions, PDE parameters, domain bounds, solver spec
- Generating collocation points from domain bounds
- Training a lightweight PINN surrogate for the CPU heatsink thermal problem
- Listing all available presets in the library

Run from repo root:
    python examples/pinneaple_arena/06_engineering_presets_showcase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pinneaple_environment import list_presets, get_preset


# ------------------------------------------------------------------
# 1. List all available presets
# ------------------------------------------------------------------
print("=" * 60)
print("Available problem presets in pinneaple_environment:")
print("=" * 60)
all_presets = list_presets()
for name in all_presets:
    print(f"  • {name}")
print(f"\nTotal: {len(all_presets)} presets\n")


# ------------------------------------------------------------------
# 2. Inspect a few engineering presets
# ------------------------------------------------------------------
showcase = [
    ("rocket_nozzle_cfd",        {}),
    ("aircraft_wing_structural", {}),
    ("car_brake_thermal",        {}),
    ("cpu_heatsink_thermal",     {}),
    ("datacenter_airflow_2d",    {}),
    ("industrial_furnace_thermal", {}),
    ("refractory_lining",        {}),
]

for name, kwargs in showcase:
    try:
        spec = get_preset(name, **kwargs)
        print(f"[{spec.problem_id}]")
        print(f"  PDE kind   : {spec.pde.kind}")
        print(f"  Fields     : {spec.fields}")
        print(f"  Coord names: {spec.coord_names}")
        print(f"  Domain     : {spec.domain_bounds}")
        print(f"  Solver     : {spec.solver_spec.get('name')} / {spec.solver_spec.get('solver','')}")
        print(f"  Description: {spec.meta.get('description','')}")
        if "digital_twin_fields" in spec.meta:
            print(f"  DT fields  : {spec.meta['digital_twin_fields']}")
        if "alert_T_max" in spec.meta:
            print(f"  Alert T_max: {spec.meta['alert_T_max']} K")
        print()
    except Exception as exc:
        print(f"[{name}] Could not load: {exc}\n")


# ------------------------------------------------------------------
# 3. Quick PINN on CPU Heatsink thermal (1D reduction for speed)
# ------------------------------------------------------------------
print("=" * 60)
print("Quick PINN: CPU Heatsink Thermal (1D steady-state)")
print("=" * 60)

try:
    spec = get_preset("cpu_heatsink_thermal", q_cpu=150.0)
    print(f"  q_cpu = 150 W | T_ambient = {spec.meta.get('T_junction_estimate', 'N/A'):.1f} K (estimate)")

    import torch
    import torch.nn as nn
    from pinneaple_train import best_device

    DEVICE = best_device()

    # 1D reduction: z ∈ [0, H] (height axis), T(0) = T_ambient, -k dT/dz|z=0 = q_flux
    k  = spec.pde.params["k"]
    q  = spec.meta["q_flux_W_m2"]
    H  = spec.domain_bounds.get("z", (0.0, 0.05))[1]
    T0 = 293.0   # T_ambient at top

    rng = np.random.default_rng(0)
    z_col = rng.uniform(0, H, 2000).astype(np.float32)
    z_bc  = np.array([H], dtype=np.float32)   # top surface: T = T0
    T_bc  = np.array([T0], dtype=np.float32)

    # Analytical solution: T(z) = T0 + q/k * (H - z)
    T_exact = T0 + q / k * (H - z_col)
    print(f"  Analytical T_base = {T0 + q/k*H:.2f} K  (z=0, CPU die)")

    # Simple linear PINN (ODE: d²T/dz² = 0 → linear T)
    class ThermalPINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 32), nn.Tanh(),
                nn.Linear(32, 32), nn.Tanh(),
                nn.Linear(32, 1),
            )
        def forward(self, z):
            return self.net(z)

    model = ThermalPINN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    z_t   = torch.from_numpy(z_col[:, None]).to(DEVICE).requires_grad_(True)
    z_bc_t = torch.tensor([[H]], dtype=torch.float32).to(DEVICE)
    T_bc_t = torch.tensor([[T0]], dtype=torch.float32).to(DEVICE)

    for epoch in range(2000):
        opt.zero_grad()
        T_pred = model(z_t)

        # PDE residual: d²T/dz² = 0
        dT_dz = torch.autograd.grad(T_pred, z_t, torch.ones_like(T_pred),
                                    create_graph=True, retain_graph=True)[0]
        d2T_dz2 = torch.autograd.grad(dT_dz, z_t, torch.ones_like(dT_dz),
                                      create_graph=True)[0]
        loss_pde = (d2T_dz2 ** 2).mean()

        # BC: T(H) = T0
        loss_bc = ((model(z_bc_t) - T_bc_t) ** 2).mean()

        # Neumann BC at z=0: -k dT/dz = q
        z0_t = torch.tensor([[0.0]], dtype=torch.float32).to(DEVICE).requires_grad_(True)
        T_z0 = model(z0_t)
        dT_z0 = torch.autograd.grad(T_z0, z0_t, torch.ones_like(T_z0), create_graph=True)[0]
        loss_neumann = ((-k * dT_z0 - q) ** 2).mean()

        loss = loss_pde + 100 * loss_bc + 100 * loss_neumann
        loss.backward()
        opt.step()

        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1} | loss={loss.item():.4e}")

    model.eval()
    with torch.no_grad():
        z_test = torch.linspace(0, H, 20, dtype=torch.float32)[:, None].to(DEVICE)
        T_pinn = model(z_test).cpu().numpy().ravel()
        T_anal = T0 + q / k * (H - z_test.cpu().numpy().ravel())

    rel_l2 = np.linalg.norm(T_pinn - T_anal) / np.linalg.norm(T_anal)
    print(f"\n  PINN L2 error vs analytic: {rel_l2:.4f}")
    print(f"  T_base (PINN) = {T_pinn[0]:.2f} K  |  (analytic) = {T_anal[0]:.2f} K")

except Exception as exc:
    print(f"  [Skipped] {exc}")


# ------------------------------------------------------------------
# 4. Datacenter digital twin preset for monitoring
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Datacenter preset for Digital Twin integration:")
print("=" * 60)
try:
    dc_spec = get_preset("datacenter_airflow_2d", n_racks=8, Q_rack=15000.0)
    print(f"  Problem: {dc_spec.problem_id}")
    print(f"  Total IT load: {dc_spec.meta['total_IT_load_kW']:.1f} kW")
    print(f"  Re: {dc_spec.meta['Re']:.0f}")
    print(f"  Alert T_max: {dc_spec.meta.get('alert_T_max', 'N/A')} K")
    print(f"  Fields for digital twin: {dc_spec.meta['digital_twin_fields']}")
    print(f"  Solver: {dc_spec.solver_spec['name']} / {dc_spec.solver_spec['solver']}")
    print()
    print("  To use with DigitalTwin:")
    print("    from pinneaple_digital_twin import build_digital_twin")
    print("    dt = build_digital_twin(model, field_names=dc_spec.meta['digital_twin_fields'])")
    print("    dt.anomaly_monitor.add_detector(")
    print(f"      ThresholdDetector({{'T': {dc_spec.meta.get('alert_T_max', 318.0)}}})")
    print("    )")
except Exception as exc:
    print(f"  [Skipped] {exc}")


print("\n=== SHOWCASE COMPLETE ===")
