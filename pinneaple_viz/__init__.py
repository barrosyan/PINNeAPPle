"""
pinneaple_viz — CFD-style visualisation for PINNeAPPle.

Quick start
-----------
from pinneaple_viz import use_cfd_style, plot_scalar, plot_vectors, plot_streamlines
from pinneaple_viz import plot_loss_history, plot_collocation, plot_pinn_prediction
from pinneaple_viz import animate_scalar_field, plot_solver_output

use_cfd_style()   # optional: enable dark CFD theme globally
"""
from .style import (
    use_cfd_style,
    get_cmap,
    make_figure,
    CMAPS,
    DEFAULT_CMAP,
)

from .fields import (
    plot_scalar,
    plot_scalar_3d,
    plot_vectors,
    plot_streamlines,
    compare_fields,
    plot_error,
)

from .pinn import (
    plot_loss_history,
    plot_multi_loss,
    plot_collocation,
    plot_pinn_prediction,
    plot_pde_residual,
    plot_gradient_magnitude,
)

from .solver import (
    plot_solver_output,
    plot_fem_result,
    plot_fvm_result,
    plot_residuals,
)

from .mesh import (
    plot_mesh,
    plot_boundary,
    plot_point_cloud,
)

from .voxel import (
    plot_voxel_slice,
    plot_voxel_3d,
    plot_voxel_histogram,
)

from .animation import (
    animate_scalar_field,
    animate_streamlines,
    make_gif,
)

from .vortex import (
    compute_vorticity_2d,
    compute_q_criterion_2d,
    compute_q_criterion_3d,
    compute_lambda2_3d,
    plot_vorticity,
    plot_q_criterion_2d,
    plot_q_criterion_3d,
    plot_vortex_identification,
    plot_lbm_flow,
    plot_flow_panel,
)

__all__ = [
    # Style
    "use_cfd_style", "get_cmap", "make_figure", "CMAPS", "DEFAULT_CMAP",
    # Scalar / vector fields
    "plot_scalar", "plot_scalar_3d", "plot_vectors", "plot_streamlines",
    "compare_fields", "plot_error",
    # PINN
    "plot_loss_history", "plot_multi_loss", "plot_collocation",
    "plot_pinn_prediction", "plot_pde_residual", "plot_gradient_magnitude",
    # Solvers
    "plot_solver_output", "plot_fem_result", "plot_fvm_result", "plot_residuals",
    # Mesh
    "plot_mesh", "plot_boundary", "plot_point_cloud",
    # Voxel
    "plot_voxel_slice", "plot_voxel_3d", "plot_voxel_histogram",
    # Animation
    "animate_scalar_field", "animate_streamlines", "make_gif",
    # Vortex / Q-criterion
    "compute_vorticity_2d", "compute_q_criterion_2d",
    "compute_q_criterion_3d", "compute_lambda2_3d",
    "plot_vorticity", "plot_q_criterion_2d", "plot_q_criterion_3d",
    "plot_vortex_identification", "plot_lbm_flow", "plot_flow_panel",
]
