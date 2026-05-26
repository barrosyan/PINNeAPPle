# LinkedIn Post — PINNeAPPle: Physics AI for Space & Defense Applications

---

We just open-sourced **PINNeAPPle** — a modular Physics-Informed Neural Network framework built for real engineering problems.

Here's what we benchmarked this week:

---

**Lunar Rover Terramechanics — 1,420x faster than numerical integration**

We trained a PINN surrogate on the Bekker-Wong wheel-soil interaction model (GRC-1 lunar regolith, Rashid-1 class wheel geometry) and benchmarked it against two reference solvers:

| Solver | Throughput | Fx Error |
|---|---|---|
| scipy.quad (high-accuracy reference) | 349 evals/s | — |
| numpy trapz (OmniLRS-style) | 19,000 evals/s | — |
| **PINNeAPPle PINN surrogate** | **496,000 evals/s** | **< 0.7%** |

496,000 evaluations per second — on CPU — with sub-1% error on drawbar pull, normal load, and driving torque. The model is exported as TorchScript and integrates directly into real-time simulators like LunCoSim via `tch-rs`.

---

**What else does PINNeAPPle do?**

Beyond terramechanics, the framework now includes pipelines inspired by state-of-the-art physics AI products:

**Crashworthiness (SHIFT-Crash inspired):**
- Abramowicz-Jones thin-walled tube crush model as synthetic data source
- Parameter-conditioned PINN: wall thickness, section side, yield strength, impact velocity → force-displacement curve
- Transfer learning from SUV program (5,000 samples) to Sedan program (300 samples) with frozen trunk
- 5-model ensemble uncertainty quantification

**Supersonic Aerodynamics (SHIFT-Missile inspired):**
- Modified Newtonian Theory + Van Driest II boundary layer → surface Cp and Cf
- Geometry-conditioned PINN: Mach, AoA, nose bluntness, canard and fin geometry → full Cp/Cf distribution
- Integrated AeroDB generation across Mach 1.5–3.5 × AoA 0°–8°
- 6-DOF trajectory integration using surrogate aerodynamics

---

**Framework highlights:**

- Modular preset registry: 30+ physics problems from Burgers 1D to 3D axial compressor stage
- External solver bridges: OpenFOAM, MATLAB, Modelica/FMU, MuJoCo, Genesis AI, TurboDesigner
- Terramechanics library with Bekker-Wong + Janosi-Hanamoto models
- Transfer learning and ensemble UQ built-in
- All models export to TorchScript for real-time integration

---

Physics-informed AI closes the gap between fast-but-wrong empirical models and slow-but-accurate numerical solvers. The PINN sits right in the middle: physics-consistent, GPU-acceleratable, and deployable in real-time simulation loops.

We're using it at **BiaTech** for rover mobility, structural crashworthiness, and aerodynamic design space exploration.

GitHub: https://github.com/barrosyan/PINNeAPPle

---

**Tags:** #PhysicsAI #PINN #SpaceEngineering #LunarRover #DeepLearning #SimulationSoftware #MachineLearnig #BiaTech
