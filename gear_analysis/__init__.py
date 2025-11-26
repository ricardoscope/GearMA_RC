"""
Crown Gear Ghost Circle Analysis Package

This package provides tools for analyzing crown gear geometry to detect
manufacturing setup errors by computing the "ghost circle" formed by
bisector intersections.

Example usage:
    from gear_analysis import AnalysisConfig, GearAnalyzer
    
    config = AnalysisConfig(
        mesh_path=Path("gear.stl"),
        slice_z=-0.2,
        r_inner=2.38,
        r_outer=2.58,
        n_teeth=38
    )
    
    analyzer = GearAnalyzer(config)
    result = analyzer.run()
    
    print(f"Ghost circle radius: {result.ghost_circle.radius}")
"""

__version__ = "1.0.0"
__author__ = "Gear Analysis Team"

# Public API
from gear_analysis.config import AnalysisConfig
from gear_analysis.models import (
    FlankLine,
    PairBisector,
    GhostCircle,
    GearCenter,
    OffsetAnalysis,
    AnalysisResult,
)
from gear_analysis.analysis.pipeline import GearAnalyzer
from gear_analysis.io.export import ResultExporter
from gear_analysis.visualization.plot_2d import plot_2d_analysis
from gear_analysis.visualization.view_3d import build_3d_geometries

__all__ = [
    # Configuration
    "AnalysisConfig",
    # Data models
    "FlankLine",
    "PairBisector",
    "GhostCircle",
    "GearCenter",
    "OffsetAnalysis",
    "AnalysisResult",
    # Analysis
    "GearAnalyzer",
    # I/O
    "ResultExporter",
    # Visualization
    "plot_2d_analysis",
    "build_3d_geometries",
]
