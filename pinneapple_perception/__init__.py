"""pinneapple_perception — extract physics observations from images,
video, and audio, for use as `DataConstraint` targets or boundary
geometry anywhere else in PINNeAPPle.

This is the inverse direction of `pinneapple_worldmodel` (which renders
physics simulations INTO photorealistic synthetic images/video for
training data generation) -- here, real-world images/video/audio go IN
and physics observations (velocity fields, boundary geometry, modal
frequencies) come OUT.

Sub-modules
-----------
video_piv
    Cross-correlation Particle Image Velocimetry: extract a velocity
    field from a video of a seeded/textured flow -- the standard
    experimental-fluid-dynamics technique, not a generic optical-flow
    method repurposed for this.
image_geometry
    Extract a boundary point cloud (and, for round parts, a fitted
    circle) from an image -- e.g. a photo or scan of a part's
    cross-section -- for use as domain geometry.
audio_modal
    Extract dominant vibration/acoustic frequencies from a recording
    (e.g. an impulse-response test) via FFT peak-picking with sub-bin
    frequency refinement -- for calibrating/validating structural-
    dynamics presets against a real measured part.

All three depend only on numpy/scipy (already core PINNeAPPle
dependencies) -- no new third-party dependency required. Every technique
here was validated against a synthetic, known-ground-truth case before
being shipped (see `tests/test_perception.py`): PIV against a known
sub-pixel image shift, boundary/circle extraction against a known
circle, and modal extraction against known sine-wave frequencies.

Quick start
-----------
>>> from pinneapple_perception import piv_velocity_field
>>> field = piv_velocity_field(frame_a, frame_b, dt=1/30, units_per_pixel=1e-3)
>>> # field["x"], field["y"], field["u"], field["v"] -- feed directly into
>>> # a DataConstraint on a navier_stokes_incompressible-family preset.
"""
from __future__ import annotations

from .video_piv import piv_velocity_field, piv_velocity_sequence
from .image_geometry import extract_boundary_points, estimate_bounding_circle
from .audio_modal import extract_dominant_frequencies

__all__ = [
    "piv_velocity_field",
    "piv_velocity_sequence",
    "extract_boundary_points",
    "estimate_bounding_circle",
    "extract_dominant_frequencies",
]
