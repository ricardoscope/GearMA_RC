"""
Configuration module for gear analysis.

Parameters automatically scale based on gear size (r_inner, r_outer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AnalysisConfig:
    """Configuration parameters for gear flank analysis.
    
    Set r_inner and r_outer for YOUR gear - other parameters scale automatically.
    """
    
    # === INPUT/OUTPUT ===
    mesh_path: Path = field(default_factory=lambda: Path("gear.stl"))
    output_dir: Path = field(default_factory=lambda: Path("results"))
    
    # === GEAR GEOMETRY (SET THESE FOR YOUR GEAR) ===
    slice_z: float = -0.2
    r_inner: float = 2.46
    r_outer: float = 2.68
    n_teeth: int = 38
    
    # === MESH PROCESSING ===
    target_triangles: int = 1_000_000
    slice_interpolation_density: float = 0.002
    
    # === CLUSTERING ===
    min_points_per_cluster: int = 5
    min_points_per_flank: int = 5
    
    # === VISUALIZATION (auto-scaled if None) ===
    flank_segment_length: Optional[float] = None
    bisector_length: Optional[float] = None
    
    # === BISECTOR/INTERSECTION ===
    parallel_threshold: float = 0.05
    
    # === RANSAC ===
    ransac_min_samples: int = 3
    ransac_residual_threshold: float = 0.003  # Works well for most gear sizes
    ransac_iterations: int = 200
    
    # === INTERSECTION FILTERING (auto-computed from r_inner/r_outer) ===
    intersection_r_min: Optional[float] = None
    intersection_r_max: Optional[float] = None
    
    # === GEAR CENTER ===
    gear_center_method: str = "outer_tips"
    
    def __post_init__(self):
        """Compute derived/scaled values after initialization."""
        if isinstance(self.mesh_path, str):
            self.mesh_path = Path(self.mesh_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Auto-scale visualization lengths based on gear size
        gear_radius = (self.r_inner + self.r_outer) / 2
        
        if self.flank_segment_length is None:
            self.flank_segment_length = gear_radius * 0.2  # 20% of radius
        
        if self.bisector_length is None:
            self.bisector_length = gear_radius * 2.0  # 200% of radius
        
        # Intersection bounds (same formula as original code):
        # r_min = R_INNER * 0.05
        # r_max = R_OUTER * 0.5
        if self.intersection_r_min is None:
            self.intersection_r_min = self.r_inner * 0.01
        
        if self.intersection_r_max is None:
            self.intersection_r_max = self.r_outer * 0.1
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "mesh_path": str(self.mesh_path),
            "output_dir": str(self.output_dir),
            "slice_z": self.slice_z,
            "r_inner": self.r_inner,
            "r_outer": self.r_outer,
            "n_teeth": self.n_teeth,
            "target_triangles": self.target_triangles,
            "slice_interpolation_density": self.slice_interpolation_density,
            "min_points_per_cluster": self.min_points_per_cluster,
            "min_points_per_flank": self.min_points_per_flank,
            "flank_segment_length": self.flank_segment_length,
            "bisector_length": self.bisector_length,
            "parallel_threshold": self.parallel_threshold,
            "ransac_min_samples": self.ransac_min_samples,
            "ransac_residual_threshold": self.ransac_residual_threshold,
            "ransac_iterations": self.ransac_iterations,
            "intersection_r_min": self.intersection_r_min,
            "intersection_r_max": self.intersection_r_max,
            "gear_center_method": self.gear_center_method,
        }