"""
Configuration module for gear analysis.

This module defines the AnalysisConfig dataclass that encapsulates all
configuration parameters for the gear analysis pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AnalysisConfig:
    """Configuration parameters for gear analysis.
    
    All parameters that control the analysis pipeline are encapsulated here,
    making the analysis fully configurable, reproducible, and testable.
    
    Attributes:
        mesh_path: Path to the input STL mesh file
        output_dir: Directory for output files (JSON, plots)
        target_triangles: Target triangle count for mesh simplification
        slice_z: Z-coordinate of the horizontal slicing plane
        slice_interpolation_density: Maximum distance between interpolated points
        r_inner: Inner radius bound for point filtering
        r_outer: Outer radius bound for point filtering
        n_teeth: Expected number of teeth in the gear
        min_points_per_cluster: Minimum points for a valid tooth cluster
        min_points_per_flank: Minimum points needed for flank line fitting
        flank_segment_length: Visual length of flank lines in plots
        bisector_length: Visual length of bisector lines in plots
        parallel_threshold: Threshold for detecting parallel bisectors
        intersection_r_min_factor: Factor of r_inner for minimum intersection radius
        intersection_r_max_factor: Factor of r_outer for maximum intersection radius
        ransac_min_samples: Minimum samples for RANSAC circle fitting
        ransac_residual_threshold: Maximum residual for RANSAC inliers
        ransac_iterations: Number of RANSAC iterations
        gear_center_method: Method for gear center estimation
    
    Example:
        >>> config = AnalysisConfig(
        ...     mesh_path=Path("gear.stl"),
        ...     slice_z=-0.2,
        ...     r_inner=2.38,
        ...     r_outer=2.58,
        ...     n_teeth=38
        ... )
        >>> config.intersection_r_min
        0.119
    """
    
    # File paths
    mesh_path: Path = field(default_factory=lambda: Path("gear.stl"))
    output_dir: Path = field(default_factory=lambda: Path("results"))
    
    # Mesh processing
    target_triangles: int = 1_000_000
    
    # Slice extraction
    slice_z: float = -0.2
    slice_interpolation_density: float = 0.002
    
    # Radius filtering
    r_inner: float = 2.38
    r_outer: float = 2.58
    
    # Tooth detection
    n_teeth: int = 38
    min_points_per_cluster: int = 5
    min_points_per_flank: int = 5
    
    # Visualization
    flank_segment_length: float = 0.5
    bisector_length: float = 5.0
    
    # Bisector intersection filtering
    parallel_threshold: float = 0.05
    intersection_r_min_factor: float = 0.05
    intersection_r_max_factor: float = 0.5
    
    # RANSAC parameters
    ransac_min_samples: int = 3
    ransac_residual_threshold: float = 0.05
    ransac_iterations: int = 100
    
    # Gear center estimation
    gear_center_method: str = "outer_tips"  # or "boundary_centroid"
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.r_inner >= self.r_outer:
            raise ValueError(f"r_inner ({self.r_inner}) must be less than r_outer ({self.r_outer})")
        if self.n_teeth < 1:
            raise ValueError(f"n_teeth must be positive, got {self.n_teeth}")
        if self.gear_center_method not in ("outer_tips", "boundary_centroid"):
            raise ValueError(f"Invalid gear_center_method: {self.gear_center_method}")
    
    @property
    def intersection_r_min(self) -> float:
        """Minimum radius for valid bisector intersections."""
        return self.r_inner * self.intersection_r_min_factor
    
    @property
    def intersection_r_max(self) -> float:
        """Maximum radius for valid bisector intersections."""
        return self.r_outer * self.intersection_r_max_factor
    
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "AnalysisConfig":
        """Create configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            AnalysisConfig instance
        """
        return cls(
            mesh_path=Path(args.stl),
            output_dir=Path(args.outdir),
            slice_z=args.z_tip * args.units_scale,
            r_inner=args.r_inner * args.units_scale,
            r_outer=args.r_outer * args.units_scale,
            n_teeth=args.n_teeth,
            gear_center_method=args.gear_center_method,
        )
    
    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisConfig":
        """Create configuration from a dictionary.
        
        Args:
            data: Dictionary of configuration parameters
            
        Returns:
            AnalysisConfig instance
        """
        # Convert string paths to Path objects
        if "mesh_path" in data and isinstance(data["mesh_path"], str):
            data["mesh_path"] = Path(data["mesh_path"])
        if "output_dir" in data and isinstance(data["output_dir"], str):
            data["output_dir"] = Path(data["output_dir"])
        
        return cls(**data)
    
    def to_dict(self) -> dict:
        """Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "mesh_path": str(self.mesh_path),
            "output_dir": str(self.output_dir),
            "target_triangles": self.target_triangles,
            "slice_z": self.slice_z,
            "slice_interpolation_density": self.slice_interpolation_density,
            "r_inner": self.r_inner,
            "r_outer": self.r_outer,
            "n_teeth": self.n_teeth,
            "min_points_per_cluster": self.min_points_per_cluster,
            "min_points_per_flank": self.min_points_per_flank,
            "flank_segment_length": self.flank_segment_length,
            "bisector_length": self.bisector_length,
            "parallel_threshold": self.parallel_threshold,
            "intersection_r_min_factor": self.intersection_r_min_factor,
            "intersection_r_max_factor": self.intersection_r_max_factor,
            "ransac_min_samples": self.ransac_min_samples,
            "ransac_residual_threshold": self.ransac_residual_threshold,
            "ransac_iterations": self.ransac_iterations,
            "gear_center_method": self.gear_center_method,
        }
