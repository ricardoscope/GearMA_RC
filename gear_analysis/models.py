"""
Data models for gear analysis.

This module defines the data structures used throughout the gear analysis
pipeline, including flank lines, bisectors, ghost circles, and analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import open3d as o3d
    from gear_analysis.config import AnalysisConfig

# Type aliases for clarity
FloatArray = NDArray[np.floating]
Vector2D = NDArray[np.floating]  # Shape: (2,)
Vector3D = NDArray[np.floating]  # Shape: (3,)
Points2D = NDArray[np.floating]  # Shape: (N, 2)
Points3D = NDArray[np.floating]  # Shape: (N, 3)


@dataclass(frozen=True)
class FlankLine:
    """Represents a fitted line for a tooth flank.
    
    The flank line is defined by a point (centroid of the flank points)
    and a direction vector (principal axis from SVD).
    
    Attributes:
        tooth: Tooth number (1-indexed)
        point: 2D centroid of the flank points
        direction: 2D unit vector defining flank direction
        cluster_size: Number of points in this tooth cluster
    
    Example:
        >>> flank = FlankLine(
        ...     tooth=1,
        ...     point=np.array([2.5, 0.1]),
        ...     direction=np.array([0.707, 0.707]),
        ...     cluster_size=150
        ... )
        >>> flank.to_dict()
        {'tooth': 1, 'side': 'right', 'point': [2.5, 0.1], ...}
    """
    tooth: int
    point: Vector2D
    direction: Vector2D
    cluster_size: int
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "tooth": self.tooth,
            "side": "right",
            "point": self.point.tolist(),
            "direction": self.direction.tolist(),
            "cluster_size": self.cluster_size,
        }


@dataclass(frozen=True)
class PairBisector:
    """Represents the angle bisector between two adjacent tooth flanks.
    
    The bisector is used to find the theoretical center of the gear by
    computing intersections of multiple bisectors.
    
    Attributes:
        between_teeth: Tuple of two tooth numbers (e.g., (2, 3))
        origin: 2D midpoint between the two flank centroids
        direction: 2D unit vector of the bisector direction
        length: Visual length for rendering
    
    Example:
        >>> bisector = PairBisector(
        ...     between_teeth=(2, 3),
        ...     origin=np.array([2.4, 0.2]),
        ...     direction=np.array([-0.8, -0.6]),
        ...     length=5.0
        ... )
    """
    between_teeth: tuple[int, int]
    origin: Vector2D
    direction: Vector2D
    length: float
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "between_teeth": list(self.between_teeth),
            "origin": self.origin.tolist(),
            "direction": self.direction.tolist(),
            "length": self.length,
        }


@dataclass(frozen=True)
class GhostCircle:
    """Represents the fitted ghost circle from bisector intersections.
    
    The ghost circle is fitted through the intersection points of bisector
    lines. Its center represents an estimate of the gear's true center,
    and deviations from the nominal center indicate setup errors.
    
    Attributes:
        center: 2D center coordinates (cx, cy)
        radius: Circle radius
        inliers: Array of inlier intersection points (used for fitting)
        outliers: Array of outlier intersection points (excluded from fit)
        rmse: Root mean square error of the fit
        n_intersections: Total number of intersections found
    
    Example:
        >>> gc = GhostCircle(
        ...     center=np.array([0.001, -0.002]),
        ...     radius=0.15,
        ...     inliers=np.array([[0.1, 0.1], [0.15, 0.05]]),
        ...     outliers=np.array([[0.5, 0.5]]),
        ...     rmse=0.0012,
        ...     n_intersections=50
        ... )
    """
    center: Vector2D
    radius: float
    inliers: Points2D
    outliers: Points2D
    rmse: float
    n_intersections: int
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "center": self.center.tolist(),
            "radius": float(self.radius),
            "rmse": float(self.rmse),
            "n_intersections": self.n_intersections,
            "n_inliers": len(self.inliers),
            "n_outliers": len(self.outliers),
            "inlier_points": self.inliers.tolist(),
            "outlier_points": self.outliers.tolist(),
        }


@dataclass(frozen=True)
class GearCenter:
    """Represents the estimated gear center.
    
    The gear center can be estimated using different methods:
    - 'outer_tips': Fit circle to outer boundary points
    - 'boundary_centroid': Use centroid of all boundary points
    
    Attributes:
        center: 2D center coordinates (gx, gy)
        method: Method used for estimation
        radius: Associated radius (if applicable)
    
    Example:
        >>> gc = GearCenter(
        ...     center=np.array([0.0, 0.0]),
        ...     method='outer_tips',
        ...     radius=2.58
        ... )
    """
    center: Vector2D
    method: str
    radius: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "center": self.center.tolist(),
            "method": self.method,
            "radius": float(self.radius) if self.radius is not None else None,
        }


@dataclass(frozen=True)
class OffsetAnalysis:
    """Represents the offset between ghost circle and gear center.
    
    The offset between the ghost circle center and the gear center serves
    as a proxy for detecting X-axis setup errors in the manufacturing process.
    
    Attributes:
        offset_vector: 2D vector from gear center to ghost circle center
        magnitude: Magnitude of offset (proxy for X-axis setup error)
        angle_deg: Angle of offset in degrees from +X axis
        ghost_center: Ghost circle center (cx, cy)
        gear_center: Gear center (gx, gy)
    
    Example:
        >>> offset = OffsetAnalysis(
        ...     offset_vector=np.array([0.001, -0.002]),
        ...     magnitude=0.00223,
        ...     angle_deg=-63.4,
        ...     ghost_center=np.array([0.001, -0.002]),
        ...     gear_center=np.array([0.0, 0.0])
        ... )
    """
    offset_vector: Vector2D
    magnitude: float
    angle_deg: float
    ghost_center: Vector2D
    gear_center: Vector2D
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "offset_vector": self.offset_vector.tolist(),
            "magnitude": float(self.magnitude),
            "angle_deg": float(self.angle_deg),
            "ghost_center": self.ghost_center.tolist(),
            "gear_center": self.gear_center.tolist(),
        }


@dataclass
class AnalysisResult:
    """Complete analysis results container.
    
    This dataclass contains all results from a gear analysis run,
    including the processed mesh, extracted points, fitted features,
    and computed metrics.
    
    Attributes:
        config: Configuration used for this analysis
        mesh: Original 3D mesh (cleaned and potentially simplified)
        slice_points: All points from the horizontal slice at slice_z
        filtered_points: Subset of slice_points within r_inner and r_outer
        center_shift: 2D offset applied to recenter the mesh
        flanks: List of fitted flank lines for each detected tooth
        bisectors: List of bisectors between adjacent tooth pairs
        ghost_circle: Fitted ghost circle from bisector intersections
        gear_center: Estimated gear center
        offset_analysis: Offset analysis between ghost circle and gear center
    
    Example:
        >>> result = analyzer.run()
        >>> print(f"Found {len(result.flanks)} tooth flanks")
        >>> if result.ghost_circle:
        ...     print(f"Ghost circle radius: {result.ghost_circle.radius:.4f}")
    """
    config: "AnalysisConfig"
    mesh: "o3d.geometry.TriangleMesh"
    slice_points: Points2D
    filtered_points: Points2D
    center_shift: Vector2D
    flanks: list[FlankLine]
    bisectors: list[PairBisector]
    ghost_circle: Optional[GhostCircle] = None
    gear_center: Optional[GearCenter] = None
    offset_analysis: Optional[OffsetAnalysis] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary.
        
        Returns:
            Dictionary containing all analysis results suitable for JSON export.
        """
        data = {
            "configuration": {
                "slice_z": self.config.slice_z,
                "inner_radius": self.config.r_inner,
                "outer_radius": self.config.r_outer,
                "n_teeth": self.config.n_teeth,
            },
            "slice_statistics": {
                "center_shift": self.center_shift.tolist(),
                "slice_point_count": len(self.slice_points),
                "filtered_point_count": len(self.filtered_points),
            },
            "flank_lines": [f.to_dict() for f in self.flanks],
            "pair_bisectors": [b.to_dict() for b in self.bisectors],
        }
        
        if self.ghost_circle is not None:
            data["ghost_circle"] = self.ghost_circle.to_dict()
        if self.gear_center is not None:
            data["gear_center"] = self.gear_center.to_dict()
        if self.offset_analysis is not None:
            data["offset_analysis"] = self.offset_analysis.to_dict()
        
        return data
    
    @property
    def has_ghost_circle(self) -> bool:
        """Check if ghost circle was successfully computed."""
        return self.ghost_circle is not None
    
    @property
    def has_offset_analysis(self) -> bool:
        """Check if offset analysis was successfully computed."""
        return self.offset_analysis is not None
    
    @property
    def n_flanks(self) -> int:
        """Number of fitted flank lines."""
        return len(self.flanks)
    
    @property
    def n_bisectors(self) -> int:
        """Number of computed bisectors."""
        return len(self.bisectors)
