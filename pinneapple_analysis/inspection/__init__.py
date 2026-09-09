"""pinneapple_analysis.inspection -- Non-Destructive Evaluation (NDE) /
inspection PINN suite: synthetic signal generation + physics-informed
training for six inspection modalities.

Relationship to neighboring packages
-------------------------------------
``pinneapple_perception`` (audio_modal, video_piv, image_geometry) goes
SENSOR-MEDIA -> PHYSICS-OBSERVABLE: it extracts a physical quantity
(a frequency, a velocity field, a boundary) FROM a generic media
artifact (audio, video, an image) that was presumably captured of a
real system, for use as a downstream training/validation signal.

This package goes the other, complementary direction: SENSOR-SIGNAL ->
INTERNAL-MATERIAL-STATE. Each modality module here generates a
physically-plausible SYNTHETIC inspection signal (an eddy-current probe
scan, an MFL leakage trace, an ultrasonic B-scan, ...) from a physics
model of the test-piece, and trains a physics-informed neural network
(``InspectionPINN``) to map (probe/scan coordinates) -> (an internal
material/defect state), constrained by the same governing PDE that
generated the data. Where ``pinneapple_perception`` interprets media
that already exists, this package is training a physics-constrained
model to relate an inspection signal to what's underneath the surface
that produced it -- the inverse-flavored problem NDE inspection
actually poses.

Relationship to ``pinneapple_simulation.numerical_solvers.eddy_current_fdm``
-------------------------------------------------------------------------------
That module ALREADY implements a full axisymmetric complex-Helmholtz FDM
solver for eddy-current physics (5-point stencil sparse assembly, a
lumped-coil source term, derived eddy-current density/flux) -- this
package's ``eddy_current`` module does not reimplement any of that. It
calls the existing solver directly and adds exactly what was missing: a
synthetic PROBE-SCAN signal (the solver only produces one field solve at
a time, not a scan) and a PINN training path whose physics loss reuses
the same governing Helmholtz equation.

What's here
-----------
``base``
    ``InspectionPINN`` -- a thin MLP subclassing `PINNBase` (mirrors
    `VanillaPINN`'s constructor exactly) -- and
    ``train_generic_inspection_pinn`` -- a thin wrapper around
    ``pinneapple_neural.trainer.trainer.Trainer`` shared by all six
    modalities below.

Six modality modules, each exposing ``generate_<modality>_synthetic(...)``
(coordinates + a physically-plausible, EXPLICITLY SYNTHETIC/ILLUSTRATIVE
signal -- never measured data) and ``train_<modality>(...)``:

``eddy_current``
    Probe-scan signal built on top of ``eddy_current_fdm``'s existing
    axisymmetric Helmholtz solve; PINN physics loss reuses that same
    equation (split into real/imaginary parts).
``magnetic_flux_leakage``
    A new lightweight 2D nonlinear-permeability (Frohlich-Kennelly
    saturation, Picard-iterated) magnetostatic FDM ground truth, with a
    simple two-pole magnet Dirichlet boundary condition; PIML loss
    enforces a locally-linearized ``div(mu*grad(phi))=0``.
``acoustic``
    1D FDTD wave-equation impact-echo / resonance-spectrum synthetic
    generator (a defect locally slows the wave, shifting modal
    frequencies down); PIML loss enforces the wave equation.
``emat``
    A deliberately simplified scalar Lorentz-force-like EM source term
    driving a 1D elastic wave equation (full coupled EM+elastodynamic
    EMAT physics is out of scope -- see the module docstring for the
    explicit simplification).
``guided_wave_ultrasonic``
    1D dispersive (Klein-Gordon-type) guided-wave pulse-echo synthetic
    generator -- a simplified stand-in for the true Rayleigh-Lamb
    dispersion relation (see module docstring).
``phased_array_ultrasonic``
    Synthetic pulse-echo B-scan from an array of point elements,
    reconstructed via delay-and-sum / Total Focusing Method beamforming;
    PIML loss applies a 2D Helmholtz consistency regularizer to the
    image-intensity surrogate (an honestly-labeled approximation, not an
    exact physical law for image intensity -- see module docstring).

Deliberately NOT included: a 7th "chemical" inspection modality -- that
overlaps ``pinneapple_systems.process_components.reaction_kinetics`` and
is out of this package's scope.
"""
from __future__ import annotations

from pinneapple_analysis.inspection.base import (
    InspectionPINN,
    train_generic_inspection_pinn,
)
from pinneapple_analysis.inspection.eddy_current import (
    generate_eddy_current_synthetic,
    make_eddy_current_physics_loss,
    train_eddy_current,
)
from pinneapple_analysis.inspection.magnetic_flux_leakage import (
    generate_magnetic_flux_leakage_synthetic,
    make_mfl_physics_loss,
    solve_mfl_nonlinear,
    train_magnetic_flux_leakage,
)
from pinneapple_analysis.inspection.acoustic import (
    generate_acoustic_synthetic,
    make_acoustic_physics_loss,
    train_acoustic,
)
from pinneapple_analysis.inspection.emat import (
    generate_emat_synthetic,
    make_emat_physics_loss,
    train_emat,
)
from pinneapple_analysis.inspection.guided_wave_ultrasonic import (
    generate_guided_wave_synthetic,
    make_guided_wave_physics_loss,
    train_guided_wave_ultrasonic,
)
from pinneapple_analysis.inspection.phased_array_ultrasonic import (
    generate_phased_array_synthetic,
    make_phased_array_physics_loss,
    train_phased_array_ultrasonic,
)

__all__ = [
    # base
    "InspectionPINN",
    "train_generic_inspection_pinn",
    # eddy_current
    "generate_eddy_current_synthetic",
    "make_eddy_current_physics_loss",
    "train_eddy_current",
    # magnetic_flux_leakage
    "generate_magnetic_flux_leakage_synthetic",
    "make_mfl_physics_loss",
    "solve_mfl_nonlinear",
    "train_magnetic_flux_leakage",
    # acoustic
    "generate_acoustic_synthetic",
    "make_acoustic_physics_loss",
    "train_acoustic",
    # emat
    "generate_emat_synthetic",
    "make_emat_physics_loss",
    "train_emat",
    # guided_wave_ultrasonic
    "generate_guided_wave_synthetic",
    "make_guided_wave_physics_loss",
    "train_guided_wave_ultrasonic",
    # phased_array_ultrasonic
    "generate_phased_array_synthetic",
    "make_phased_array_physics_loss",
    "train_phased_array_ultrasonic",
]
