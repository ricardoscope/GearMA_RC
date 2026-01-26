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
    r_inner: float = 2.45
    r_outer: float = 2.68
    n_teeth: int = 38
    
    # === MESH PROCESSING ===
    target_triangles: int = 10_000_000
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
    ransac_residual_threshold: float = 0.001  # Tighter threshold for better fit
    ransac_iterations: int = 200  # More iterations for better result
    
    # === GHOST CIRCLE CONSTRAINTS ===
    # Expected radius for ghost circle (None = auto-compute from data)
    expected_ghost_radius: Optional[float] = None
    # Tolerance for ghost radius constraint (0.5 = ±50%)
    ghost_radius_tolerance: float = 0.5


    # === INTERSECTION FILTERING (auto-computed from r_inner/r_outer) ===
    intersection_r_min: Optional[float] = None
    intersection_r_max: Optional[float] = None
    
    # === GEAR CENTER ===
    gear_center_method: str = "outer_tips"
    
    # === NORMAL-BASED VALIDATION (NEW) ===
    use_surface_normals: bool = True  # Extract and use surface normals for validation
    normal_direction_threshold_deg: float = 15.0  # Max acceptable SVD/normal direction difference
    
    # === ROBUST FLANK CLASSIFICATION ===
    # When True, uses surface normals as PRIMARY classifier for left/right
    # instead of angular position. More robust but slightly slower.
    use_normal_based_classification: bool = False
    
    # Threshold for normal-based classification
    # |tangential_component| must exceed this to classify as left/right
    normal_classification_threshold: float = 0.15
    
    def __post_init__(self):
        """Compute derived/scaled values after initialization."""
        if isinstance(self.mesh_path, str):
            self.mesh_path = Path(self.mesh_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Auto-scale visualization lengths based on gear size
        gear_radius = (self.r_inner + self.r_outer) / 2
        
        if self.flank_segment_length is None:
            self.flank_segment_length = gear_radius * 0.2
        
        if self.bisector_length is None:
            self.bisector_length = gear_radius * 2.0
        
        # TIGHTER intersection bounds for more accurate circle:
        # - r_min stays small to capture center intersections
        # - r_max reduced from 0.5 to 0.3 to exclude outlier intersections
        if self.intersection_r_min is None:
            self.intersection_r_min = self.r_inner * 0.02  # 2% of r_inner
        
        if self.intersection_r_max is None:
            self.intersection_r_max = self.r_outer * 0.3   # 30% of r_outer (was 50%)
    
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
            "use_surface_normals": self.use_surface_normals,
            "normal_direction_threshold_deg": self.normal_direction_threshold_deg,
        }