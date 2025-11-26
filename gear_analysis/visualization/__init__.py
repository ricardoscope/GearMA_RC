"""
Visualization subpackage.

This package provides 2D and 3D visualization capabilities
for gear analysis results.
"""

from gear_analysis.visualization.plot_2d import (
    plot_2d_analysis,
    plot_tooth_details,
)
from gear_analysis.visualization.view_3d import (
    Visualizer3D,
    build_3d_geometries,
    show_3d_visualization,
    show_3d_with_custom_lighting,
)

__all__ = [
    # 2D plotting
    "plot_2d_analysis",
    "plot_tooth_details",
    # 3D visualization
    "Visualizer3D",
    "build_3d_geometries",
    "show_3d_visualization",
    "show_3d_with_custom_lighting",
]