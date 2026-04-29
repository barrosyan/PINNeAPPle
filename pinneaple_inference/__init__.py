from .infer import (
    infer,
    infer_on_grid_1d,
    infer_on_grid_2d,
    InferenceResult,
)
from .visualize import (
    plot_field_1d,
    plot_field_2d,
    plot_error_map_1d,
    plot_error_map_2d,
    plot_loss_curve,
    plot_model_comparison_1d,
    plot_model_comparison_2d,
    render_visualizations,
    # 3D / internal flow
    plot_velocity_slice,
    plot_velocity_magnitude_slice,
    plot_streamlines_2d,
    plot_centerline_velocity,
    plot_vorticity_slice,
    plot_internal_flow_summary,
    # Design optimization
    plot_design_opt_convergence,
    plot_pareto_front_2d,
)
from .postprocess import (
    compute_streamlines,
    compute_isosurface,
    plot_streamlines_2d as plot_streamlines_2d_model,
    plot_isosurface_3d,
    plot_volume_slice,
    FlowVisualizer,
)

__all__ = [
    "infer",
    "infer_on_grid_1d",
    "infer_on_grid_2d",
    "InferenceResult",
    "plot_field_1d",
    "plot_field_2d",
    "plot_error_map_1d",
    "plot_error_map_2d",
    "plot_loss_curve",
    "plot_model_comparison_1d",
    "plot_model_comparison_2d",
    "render_visualizations",
    # 3D / internal flow
    "plot_velocity_slice",
    "plot_velocity_magnitude_slice",
    "plot_streamlines_2d",
    "plot_centerline_velocity",
    "plot_vorticity_slice",
    "plot_internal_flow_summary",
    # Design optimization
    "plot_design_opt_convergence",
    "plot_pareto_front_2d",
    # Postprocess (Feature 17)
    "compute_streamlines",
    "compute_isosurface",
    "plot_streamlines_2d_model",
    "plot_isosurface_3d",
    "plot_volume_slice",
    "FlowVisualizer",
]
