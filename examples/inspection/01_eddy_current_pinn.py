"""pinneapple_analysis.inspection example 01: eddy-current NDE end to end.

Generates a synthetic eddy-current probe-scan dataset on top of the
existing axisymmetric complex-Helmholtz FDM solver
(`pinneapple_simulation.numerical_solvers.eddy_current_fdm`), trains an
`InspectionPINN` on it with a physics loss reusing that same Helmholtz
equation, and predicts the learned vector-potential field at a few
points.

Run (this repo isn't pip-installed in this environment, so PYTHONPATH
must include the repo root):

    PYTHONPATH=. python3 examples/inspection/01_eddy_current_pinn.py
"""
from __future__ import annotations

import numpy as np
import torch

from pinneapple_analysis.inspection import generate_eddy_current_synthetic, train_eddy_current


def main() -> None:
    print("=== Eddy-current NDE: synthetic probe scan ===")
    data = generate_eddy_current_synthetic(
        nr=24, nz=40, scan_positions=np.linspace(-0.01, 0.01, 9),
    )
    print(f"grid: r={data['r'].shape}, z={data['z'].shape}, A_baseline={data['A_baseline'].shape} (complex)")
    print(f"probe-scan positions (m): {data['scan_positions']}")
    print("differential probe signal magnitude at each scan position:")
    for pos, sig in zip(data["scan_positions"], data["scan_signal"]):
        print(f"  scan_position={pos:+.4f} m -> |signal|={abs(sig):.4e}, phase={np.angle(sig):.3f} rad")

    print("\n=== Training InspectionPINN (Helmholtz physics loss) ===")
    out = train_eddy_current(
        synthetic_data=data,
        hidden=[64, 64, 64],
        epochs=60,
        lr=1e-3,
        batch_size=32,
        log_dir="examples/_out/inspection_eddy_current",
        run_name="eddy_current_demo",
    )
    history = out["train_out"]["history"]
    print(f"epoch 0 val_total:  {history[0]['val_total']:.6e}")
    print(f"final val_total:    {history[-1]['val_total']:.6e}")
    print(f"best_val:           {out['train_out']['best_val']:.6e}")

    print("\n=== Predicting the learned vector-potential field ===")
    model = out["model"]
    model.eval()
    r_probe = np.array([0.015, 0.02, 0.025])
    z_probe = np.array([-0.005, -0.01, -0.015])
    x_query = torch.tensor(np.stack([r_probe, z_probe], axis=1), dtype=torch.float32)
    with torch.no_grad():
        y_hat = model.predict(x_query)
    a_scale = data["a_scale"]
    for (r_q, z_q), (ar, ai) in zip(x_query.numpy(), y_hat.numpy()):
        print(f"  (r={r_q:.3f}, z={z_q:.3f}) -> A ~= ({ar * a_scale:.4e} + {ai * a_scale:.4e}j)  [rescaled to SI]")

    print("\nDone.")


if __name__ == "__main__":
    main()
