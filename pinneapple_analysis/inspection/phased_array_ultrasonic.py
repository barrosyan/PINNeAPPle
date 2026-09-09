"""pinneapple_analysis.inspection.phased_array_ultrasonic -- synthetic
B-scan generation from an array of point-source/receiver elements via
delay-and-sum (Total Focusing Method, TFM) beamforming, plus
physics-informed training.

Physical picture
-----------------
An N-element linear phased-array probe sits on the surface (z=0) of a
test-piece; a point-like defect (scatterer) sits at depth (x_d, z_d). In
PULSE-ECHO mode, element i transmits and receives its own echo: the
round-trip time-of-flight is ``2 * dist(element_i, defect) / c`` and the
received amplitude decays geometrically (``1/dist``) -- each element's
time trace is modeled as a RICKER WAVELET (the standard synthetic
ultrasonic pulse shape used throughout seismic/NDE simulation; it is the
second derivative of a Gaussian and closely resembles a real
piezoelectric transducer's broadband pulse response) centered at that
arrival time.

The B-scan IMAGE is then reconstructed via delay-and-sum / Total
Focusing Method (TFM) beamforming (Holmes, Drinkwater & Wilcox,
"Post-processing of the full matrix of ultrasonic transmit-receive array
data for non-destructive evaluation", NDT&E International, 2005): for
every hypothesis pixel (x, z), sum each element's recorded trace at the
travel time implied by that hypothesis location -- constructive
interference focuses the image at the true scatterer location.

Physics-loss caveat (read before treating this as more than a PIML
regularizer): the reconstructed image intensity is a nonlinear function
of the underlying acoustic pressure field, not itself a solution of the
wave equation. This module nonetheless trains the PINN's image-intensity
surrogate against a 2D HELMHOLTZ consistency constraint
(``laplacian(p) + k0^2 * p = 0`` at the array's carrier wavenumber
``k0 = 2*pi*f0/c``) as a smooth, wave-consistent regularizer -- the same
spirit as a matched-field-processing image inheriting spatial-frequency
content from the governing wave equation -- NOT an exact physical law
for image intensity. This is stated explicitly so it is not mistaken for
a first-principles derivation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import InspectionPINN, PhysicsLossFn, train_generic_inspection_pinn


def _ricker_wavelet(t: np.ndarray, t0: float, f0: float) -> np.ndarray:
    """Standard Ricker ("Mexican hat") wavelet, the usual synthetic
    ultrasonic/seismic pulse shape: second derivative of a Gaussian,
    peak frequency f0."""
    tau = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * tau ** 2) * np.exp(-tau ** 2)


def generate_phased_array_synthetic(
    *,
    n_elements: int = 24,
    array_length: float = 0.02,   # m
    c: float = 5900.0,            # m/s, longitudinal-wave speed (steel-like)
    burst_freq: float = 5.0e6,    # Hz, typical phased-array UT frequency
    defect_x: float = 0.0,        # m, relative to array center
    defect_z: float = 0.015,      # m, depth
    dt: Optional[float] = None,
    n_time: int = 400,
    nx_image: int = 40,
    nz_image: int = 40,
    image_x_span: float = 0.024,  # m, centered on the array
    image_z_span: tuple = (0.002, 0.03),
    n_supervised: int = 500,
    n_collocation: int = 250,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """Generate a synthetic phased-array pulse-echo B-scan and its
    delay-and-sum (TFM) reconstruction.

    Returns a dict with:
      element_x        -- (n_elements,) element positions (m)
      t_grid            -- (n_time,) fast-time grid
      raw_traces        -- (n_elements, n_time) per-element pulse-echo A-scans
      image_x, image_z  -- 1D grids of the reconstructed B-scan image
      image             -- (nx_image, nz_image) delay-and-sum image intensity
      x, y, x_col       -- InspectionPINN training data over the image field
      k0                -- carrier wavenumber (for the physics residual)
    """
    rng = np.random.default_rng(seed)

    element_x = np.linspace(-array_length / 2, array_length / 2, n_elements)
    if dt is None:
        dt = 1.0 / (20.0 * burst_freq)  # >= 20 samples per carrier cycle
    t = np.arange(n_time) * dt

    dist_i = 2.0 * np.sqrt((element_x - defect_x) ** 2 + defect_z ** 2)
    arrival_i = dist_i / c
    amp_i = 1.0 / np.maximum(dist_i, 1e-6)

    raw_traces = amp_i[:, None] * _ricker_wavelet(t[None, :], arrival_i[:, None], burst_freq)

    image_x = np.linspace(-image_x_span / 2, image_x_span / 2, nx_image)
    image_z = np.linspace(image_z_span[0], image_z_span[1], nz_image)
    image = np.zeros((nx_image, nz_image))
    for ix, xg in enumerate(image_x):
        for iz, zg in enumerate(image_z):
            travel_t = 2.0 * np.sqrt((element_x - xg) ** 2 + zg ** 2) / c
            contrib = 0.0
            for e in range(n_elements):
                contrib += np.interp(travel_t[e], t, raw_traces[e], left=0.0, right=0.0)
            image[ix, iz] = contrib

    img_scale = float(np.max(np.abs(image)))
    if img_scale <= 0:
        img_scale = 1.0

    XX, ZZ = np.meshgrid(image_x, image_z, indexing="ij")
    n_sup = min(n_supervised, nx_image * nz_image)
    flat_idx = rng.choice(nx_image * nz_image, size=n_sup, replace=False)
    xi, zi = flat_idx // nz_image, flat_idx % nz_image
    x_sup = np.stack([XX.ravel()[flat_idx], ZZ.ravel()[flat_idx]], axis=1).astype(np.float32)
    y_sup = (image[xi, zi] / img_scale).astype(np.float32)[:, None]

    margin_x = 0.1 * image_x_span
    x_col = np.stack(
        [
            rng.uniform(-image_x_span / 2 + margin_x, image_x_span / 2 - margin_x, n_collocation),
            rng.uniform(image_z_span[0] * 1.2, image_z_span[1] * 0.9, n_collocation),
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "element_x": element_x,
        "t_grid": t,
        "raw_traces": raw_traces,
        "image_x": image_x,
        "image_z": image_z,
        "image": image,
        "x": x_sup,
        "y": y_sup,
        "x_col": x_col,
        "img_scale": img_scale,
        "k0": float(2.0 * np.pi * burst_freq / c),
        "c": float(c),
        "burst_freq": float(burst_freq),
        "defect_x": float(defect_x),
        "defect_z": float(defect_z),
    }


def make_phased_array_physics_loss(k0: float) -> PhysicsLossFn:
    """Build a `(model, batch) -> (loss, comps)` physics loss enforcing
    the 2D Helmholtz consistency constraint ``laplacian(p) + k0^2 * p =
    0`` on the PINN's image-intensity surrogate (see module docstring
    for the honest caveat about this being a regularizer, not an exact
    physical law for image intensity)."""

    def physics_loss_fn(model: torch.nn.Module, batch: Dict[str, Any]):
        x_col = batch["x_col"]
        if not x_col.requires_grad:
            x_col = x_col.clone().requires_grad_(True)

        p = model.predict(x_col) if hasattr(model, "predict") else model(x_col)
        p = p[:, 0:1]

        g = torch.autograd.grad(p, x_col, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        dpdx, dpdz = g[:, 0:1], g[:, 1:2]
        d2pdx2 = torch.autograd.grad(dpdx, x_col, grad_outputs=torch.ones_like(dpdx), create_graph=True, retain_graph=True)[0][:, 0:1]
        d2pdz2 = torch.autograd.grad(dpdz, x_col, grad_outputs=torch.ones_like(dpdz), create_graph=True, retain_graph=True)[0][:, 1:2]

        laplacian = d2pdx2 + d2pdz2
        residual = laplacian + (k0 ** 2) * p
        loss = torch.mean(residual ** 2)
        comps = {"residual_rmse": float(torch.sqrt(torch.mean(residual.detach() ** 2)).item())}
        return loss, comps

    return physics_loss_fn


def train_phased_array_ultrasonic(
    synthetic_data: Optional[Dict[str, Any]] = None,
    *,
    hidden=(64, 64, 64),
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
    seed: Optional[int] = 0,
    log_dir: str = "runs/inspection_phased_array",
    run_name: str = "phased_array_pinn",
    w_supervised: float = 1.0,
    w_physics: float = 0.1,
    **synthetic_kwargs: Any,
) -> Dict[str, Any]:
    if synthetic_data is None:
        synthetic_data = generate_phased_array_synthetic(seed=seed, **synthetic_kwargs)

    model = InspectionPINN(in_dim=2, out_dim=1, hidden=list(hidden), modality="phased_array_ultrasonic")
    physics_fn = make_phased_array_physics_loss(synthetic_data["k0"])
    train_out = train_generic_inspection_pinn(
        model, synthetic_data, physics_fn,
        epochs=epochs, lr=lr, batch_size=batch_size, device=device, seed=seed,
        log_dir=log_dir, run_name=run_name, w_supervised=w_supervised, w_physics=w_physics,
    )
    return {"model": model, "train_out": train_out, "synthetic_data": synthetic_data}
