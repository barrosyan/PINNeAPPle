# Electrodynamics & Electromagnetics Examples

Physics-informed neural network (PINN) solutions to classical electrostatics,
magnetostatics and electromagnetics problems.

## Examples

| File | Problem | PDE | Exact solution |
|------|---------|-----|----------------|
| `01_laplace_capacitor.py` | Parallel-plate capacitor | ∇²φ = 0 | φ = y |
| `02_poisson_charge.py` | Gaussian charge in a grounded box | ∇²φ = −ρ | FDM reference |
| `03_electric_dipole.py` | Electric dipole — field lines & equipotentials | ∇²φ = −(ρ₊ − ρ₋) | Dipole far field |
| `04_magnetostatics_wire.py` | Infinite current-carrying wire | ∇²A_z = −μ₀J_z | Analytical inside/outside |
| `05_em_wave_1d.py` | Transverse EM plane wave | E_tt = c²E_xx | sin(πx)cos(πct) |
| `06_tm_waveguide.py` | TM₁₁ resonant cavity mode | ∇²E_z + k²E_z = 0 | sin(πx)sin(πy) |

## Running

Each example is standalone:

```bash
python examples/electrodynamics/01_laplace_capacitor.py
python examples/electrodynamics/02_poisson_charge.py
python examples/electrodynamics/03_electric_dipole.py
python examples/electrodynamics/04_magnetostatics_wire.py
python examples/electrodynamics/05_em_wave_1d.py
python examples/electrodynamics/06_tm_waveguide.py
```

Or run all at once:

```bash
for f in examples/electrodynamics/*.py; do python $f; done
```

Results (PNG images) are saved to `examples/electrodynamics/results/`.

## Dependencies

- `torch` (PyTorch)
- `numpy`
- `matplotlib`
- `scipy` (optional — only for FDM reference in example 02)
