"""
Crown Gear Ghost Circle Analysis Tool

This module analyzes crown gear geometry to detect manufacturing setup errors
by computing the "ghost circle" formed by bisector intersections.

Refactored version with:
- Configuration dataclass (no global state)
- Improved type hints
- Dependency injection
- Better separation of concerns
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
from numpy.typing import NDArray
from scipy.optimize import least_squares

# Type aliases for clarity
FloatArray = NDArray[np.floating]
Vector2D = NDArray[np.floating]  # Shape: (2,)
Vector3D = NDArray[np.floating]  # Shape: (3,)
Points2D = NDArray[np.floating]  # Shape: (N, 2)
Points3D = NDArray[np.floating]  # Shape: (N, 3)
Color = tuple[float, float, float]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ==================== Configuration ====================
@dataclass
class AnalysisConfig:
    """Configuration parameters for gear analysis.
    
    All parameters that were previously global constants are now
    encapsulated here, making the analysis fully configurable and testable.
    """
    # File paths
    mesh_path: Path = field(default_factory=lambda: Path(r"C:\Users\alibi\Documents\Gears Examples\SimRes\.stl"))
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
        """Create configuration from command line arguments."""
        return cls(
            mesh_path=Path(args.stl),
            output_dir=Path(args.outdir),
            slice_z=args.z_tip * args.units_scale,
            r_inner=args.r_inner * args.units_scale,
            r_outer=args.r_outer * args.units_scale,
            n_teeth=args.n_teeth,
            gear_center_method=args.gear_center_method,
        )


# ==================== Data Structures ====================
@dataclass(frozen=True)
class FlankLine:
    """Represents a fitted line for a tooth flank.
    
    Attributes:
        tooth: Tooth number (1-indexed)
        point: 2D centroid of the flank points
        direction: 2D unit vector defining flank direction
        cluster_size: Number of points in this tooth cluster
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
    
    Attributes:
        between_teeth: Tuple of two tooth numbers (e.g., (2, 3))
        origin: 2D midpoint between the two flank centroids
        direction: 2D unit vector of the bisector direction
        length: Visual length for rendering
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
    
    Attributes:
        center: 2D center coordinates (cx, cy)
        radius: Circle radius
        inliers: Array of inlier intersection points
        outliers: Array of outlier intersection points
        rmse: Root mean square error of the fit
        n_intersections: Total number of intersections found
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
    
    Attributes:
        center: 2D center coordinates (gx, gy)
        method: Method used ('outer_tips' or 'boundary_centroid')
        radius: Associated radius (if applicable)
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
    
    Attributes:
        offset_vector: 2D vector from gear center to ghost circle center
        magnitude: Magnitude of offset (proxy for X-axis setup error)
        angle_deg: Angle of offset in degrees from +X axis
        ghost_center: Ghost circle center (cx, cy)
        gear_center: Gear center (gx, gy)
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
    """
    config: AnalysisConfig
    mesh: o3d.geometry.TriangleMesh
    slice_points: Points2D
    filtered_points: Points2D
    center_shift: Vector2D
    flanks: list[FlankLine]
    bisectors: list[PairBisector]
    ghost_circle: Optional[GhostCircle] = None
    gear_center: Optional[GearCenter] = None
    offset_analysis: Optional[OffsetAnalysis] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
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


# ==================== Utility Functions ====================
def unit_vector(v: FloatArray) -> FloatArray:
    """Normalize vector to unit length. Returns original if zero-length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


def to_3d(point_2d: Vector2D, z: float) -> Vector3D:
    """Convert 2D point (x, y) to 3D by appending z coordinate."""
    return np.array([point_2d[0], point_2d[1], z])


# ==================== Mesh Processing ====================
class MeshLoader:
    """Handles mesh loading, cleaning, and format conversion."""
    
    @staticmethod
    def load(path: Path, target_triangles: int) -> o3d.geometry.TriangleMesh:
        """Load, clean, and optionally simplify a mesh from file.
        
        Steps:
        1. Load STL file
        2. Remove duplicate vertices and degenerate triangles
        3. Compute vertex normals
        4. Simplify to target_triangles if mesh is too large
        
        Args:
            path: Path to STL file
            target_triangles: Target triangle count for simplification
            
        Returns:
            Cleaned Open3D triangle mesh
            
        Raises:
            FileNotFoundError: If mesh file doesn't exist
            RuntimeError: If mesh has no triangles
        """
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found: {path}")
        
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty() or len(mesh.triangles) == 0:
            raise RuntimeError("Loaded mesh has no triangles.")
        
        # Clean up mesh geometry
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        
        # Simplify if mesh is too complex
        num_triangles = len(mesh.triangles)
        if num_triangles > target_triangles:
            logger.info(f"Simplifying mesh: {num_triangles} → {target_triangles} triangles")
            mesh = mesh.simplify_quadric_decimation(target_triangles)
        
        return mesh
    
    @staticmethod
    def to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
        """Convert Open3D mesh to Trimesh format for slicing operations."""
        return trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
            process=False  # Skip automatic processing for speed
        )


# ==================== Slice Processing ====================
class SliceExtractor:
    """Extracts and processes 2D slices from 3D meshes."""
    
    @staticmethod
    def interpolate_path(path: Points2D, max_step: float) -> Points2D:
        """Add interpolated points to ensure uniform point density along a path.
        
        For each segment longer than max_step, adds intermediate points.
        This ensures consistent point density for accurate tooth detection.
        
        Args:
            path: Array of 2D points forming a path
            max_step: Maximum allowed distance between consecutive points
            
        Returns:
            Densified path with interpolated points
        """
        if len(path) < 2:
            return path
        
        segments: list[Points2D] = []
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            segment_length = np.linalg.norm(p1 - p0)
            
            if segment_length > max_step:
                num_points = int(np.ceil(segment_length / max_step)) + 1
                t = np.linspace(0, 1, num_points)
                segment = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
                segments.append(segment[:-1])
            else:
                segments.append(p0[None, :])
        
        segments.append(path[-1:])
        return np.vstack(segments)
    
    @classmethod
    def extract(cls, tm: trimesh.Trimesh, z: float, max_step: float) -> Points2D:
        """Extract a horizontal slice from the mesh and interpolate points.
        
        Creates a 2D cross-section by intersecting the mesh with a horizontal plane.
        The resulting contour is then densified to ensure uniform point spacing.
        
        Args:
            tm: Trimesh object
            z: Z-coordinate of slicing plane
            max_step: Maximum spacing between interpolated points
            
        Returns:
            Array of 2D points (N, 2) representing the slice contour
            
        Raises:
            RuntimeError: If no intersection found at specified Z
        """
        section = tm.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if section is None or section.vertices.size == 0:
            raise RuntimeError(f"No intersection found at Z={z}")
        
        verts = np.asarray(section.vertices, dtype=float)
        xy = verts[:, :2] if verts.shape[1] >= 2 else verts
        
        paths: list[Points2D] = []
        if hasattr(section, "entities") and len(section.entities) > 0:
            for ent in section.entities:
                if hasattr(ent, "points") and len(ent.points) >= 2:
                    pts = xy[np.asarray(ent.points, dtype=int)]
                    paths.append(cls.interpolate_path(pts, max_step))
        
        if not paths and len(xy) >= 2:
            paths.append(cls.interpolate_path(xy, max_step))
        
        if not paths:
            raise RuntimeError(f"No valid paths extracted from slice at Z={z}")
        
        return np.vstack(paths)


# ==================== Point Filtering ====================
class PointFilter:
    """Filters points based on geometric criteria."""
    
    @staticmethod
    def by_radius(
        points: Points2D,
        r_inner: float,
        r_outer: float,
        relaxation_factor: float = 0.1
    ) -> Points2D:
        """Filter points to keep only those within an annular region.
        
        Keeps points where r_inner <= distance_from_origin <= r_outer.
        If no points found, tries relaxed bounds.
        
        Args:
            points: 2D point array (N, 2)
            r_inner: Inner radius bound
            r_outer: Outer radius bound
            relaxation_factor: Factor to relax bounds if no points found
            
        Returns:
            Filtered 2D point array
            
        Raises:
            RuntimeError: If no points found even with relaxed bounds
        """
        radii = np.linalg.norm(points, axis=1)
        mask = (radii >= r_inner) & (radii <= r_outer)
        filtered = points[mask]
        
        if len(filtered) == 0:
            relaxed_inner = r_inner * (1 - relaxation_factor)
            relaxed_outer = r_outer * (1 + relaxation_factor)
            mask = (radii >= relaxed_inner) & (radii <= relaxed_outer)
            filtered = points[mask]
            
            if len(filtered) == 0:
                r_min = float(np.min(radii)) if len(radii) > 0 else 0.0
                r_max = float(np.max(radii)) if len(radii) > 0 else 0.0
                r_p25 = float(np.percentile(radii, 25)) if len(radii) > 0 else 0.0
                r_p75 = float(np.percentile(radii, 75)) if len(radii) > 0 else 0.0
                
                raise RuntimeError(
                    f"No points found within radii {r_inner:.2f}-{r_outer:.2f}\n"
                    f"  Actual range: {r_min:.4f}-{r_max:.4f}\n"
                    f"  Suggested: R_INNER ≈ {r_p25:.4f}, R_OUTER ≈ {r_p75:.4f}"
                )
            
            logger.warning(f"Using relaxed radii: {relaxed_inner:.2f}-{relaxed_outer:.2f}")
        
        return filtered


# ==================== Tooth Clustering ====================
class ToothClusterer:
    """Partitions points into tooth clusters based on angular position."""
    
    @staticmethod
    def partition_by_angle(
        points_xy: Points2D,
        n_teeth: int,
        min_points_per_cluster: int
    ) -> list[Points2D]:
        """Partition points into exactly n_teeth angular sectors.
        
        Divides the full circle into equal angular bins, one per tooth.
        
        Args:
            points_xy: 2D point array (N, 2)
            n_teeth: Expected number of teeth
            min_points_per_cluster: Minimum points for a valid cluster
            
        Returns:
            List of n_teeth arrays (one per tooth, may be empty)
        """
        if len(points_xy) == 0 or n_teeth < 1:
            return []
        
        angles = np.arctan2(points_xy[:, 1], points_xy[:, 0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        
        edges = np.linspace(0.0, 2 * np.pi, n_teeth + 1)
        bin_idx = np.digitize(angles, edges, right=False) - 1
        bin_idx[bin_idx == n_teeth] = 0
        
        clusters: list[Points2D] = []
        for k in range(n_teeth):
            pts = points_xy[bin_idx == k]
            if len(pts) >= min_points_per_cluster:
                clusters.append(pts)
            else:
                clusters.append(np.empty((0, 2)))
        
        return clusters


# ==================== Line Fitting ====================
class LineFitter:
    """Fits lines to point clouds using various methods."""
    
    @staticmethod
    def fit_svd(points: Points2D) -> tuple[Vector2D, Vector2D]:
        """Fit a line to 2D points using SVD (Principal Component Analysis).
        
        Returns the centroid and principal direction of the point cloud.
        
        Args:
            points: 2D point array (N, 2)
            
        Returns:
            Tuple of (centroid, direction_vector)
            
        Raises:
            ValueError: If fewer than 2 points provided
        """
        if len(points) < 2:
            raise ValueError("Need at least two points to fit a line.")
        
        centroid = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
        direction = unit_vector(vt[0])
        return centroid, direction
    
    @classmethod
    def extract_right_flank(
        cls,
        points: Points2D,
        min_points: int
    ) -> tuple[Vector2D, Vector2D]:
        """Extract and fit a line to the right flank of a tooth cluster.
        
        Strategy:
        1. Calculate cluster center and local coordinate system
        2. Split points into left/right based on tangential projection
        3. Fit line to right-side points only
        
        Args:
            points: 2D points belonging to one tooth (N, 2)
            min_points: Minimum points needed for fitting
            
        Returns:
            Tuple of (flank_centroid, flank_direction)
            
        Raises:
            ValueError: If insufficient points for fitting
        """
        center = points.mean(axis=0)
        
        radial = unit_vector(center) if np.linalg.norm(center) > 1e-10 else np.array([1.0, 0.0])
        tangential = np.array([-radial[1], radial[0]])
        
        projections = (points - center) @ tangential
        median = np.median(projections)
        right = points[projections > median]
        
        if len(right) < min_points:
            order = np.argsort(projections)
            half = max(min_points, len(points) // 2)
            right = points[order[-half:]]
            if len(right) < 3:
                raise ValueError("Insufficient points for flank fitting.")
        
        return cls.fit_svd(right)


# ==================== Bisector Computation ====================
class BisectorComputer:
    """Computes bisectors between tooth flanks."""
    
    @staticmethod
    def compute_bisector(
        point_a: Vector2D, dir_a: Vector2D,
        point_b: Vector2D, dir_b: Vector2D
    ) -> tuple[Vector2D, Vector2D]:
        """Compute the angle bisector between two lines.
        
        Args:
            point_a: Center of first line
            dir_a: Direction of first line
            point_b: Center of second line
            dir_b: Direction of second line
            
        Returns:
            Tuple of (bisector_origin, bisector_direction)
        """
        dir_a = unit_vector(dir_a)
        dir_b = unit_vector(dir_b)
        
        if np.dot(dir_a, dir_b) < 0:
            dir_b = -dir_b
        
        bisector_dir = unit_vector(dir_a + dir_b)
        origin = 0.5 * (point_a + point_b)
        
        if np.linalg.norm(bisector_dir) < 1e-8:
            bisector_dir = unit_vector(origin) if np.linalg.norm(origin) > 1e-10 else dir_a
        
        return origin, bisector_dir
    
    @classmethod
    def compute_pair_bisectors(
        cls,
        flanks: list[FlankLine],
        length: float,
        n_teeth: int
    ) -> list[PairBisector]:
        """Compute bisectors for consecutive even-odd tooth pairs with wraparound.
        
        Creates bisectors between: (2-3), (4-5), (6-7), ..., (N-1)
        
        Args:
            flanks: List of fitted flank lines
            length: Visual length for bisector rendering
            n_teeth: Total number of teeth
            
        Returns:
            List of bisector objects
        """
        if len(flanks) < 2:
            return []
        
        flank_dict = {f.tooth: f for f in flanks}
        bisectors: list[PairBisector] = []
        
        for tooth_num in range(2, n_teeth + 1, 2):
            next_tooth_num = tooth_num + 1 if tooth_num < n_teeth else 1
            
            if tooth_num in flank_dict and next_tooth_num in flank_dict:
                current = flank_dict[tooth_num]
                nxt = flank_dict[next_tooth_num]
                
                origin, direction = cls.compute_bisector(
                    current.point, current.direction,
                    nxt.point, nxt.direction
                )
                bisectors.append(PairBisector(
                    between_teeth=(current.tooth, nxt.tooth),
                    origin=origin,
                    direction=direction,
                    length=length,
                ))
        
        return bisectors


# ==================== Bisector Intersection ====================
class IntersectionFinder:
    """Finds intersections between lines."""
    
    @staticmethod
    def line_intersection_2d(
        p1: Vector2D, d1: Vector2D,
        p2: Vector2D, d2: Vector2D
    ) -> Optional[Vector2D]:
        """Find intersection point of two 2D lines.
        
        Args:
            p1: Point on first line
            d1: Direction of first line (unit vector)
            p2: Point on second line
            d2: Direction of second line (unit vector)
            
        Returns:
            Intersection point or None if parallel
        """
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-10:
            return None
        
        dp = p2 - p1
        t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
        
        return p1 + t * d1
    
    @classmethod
    def compute_bisector_intersections(
        cls,
        bisectors: list[PairBisector],
        r_min: float,
        r_max: float,
        parallel_threshold: float
    ) -> list[Vector2D]:
        """Compute pairwise intersections of bisectors near the center.
        
        Args:
            bisectors: List of bisector lines
            r_min: Minimum radius for valid intersections
            r_max: Maximum radius for valid intersections
            parallel_threshold: Threshold for parallel line detection
            
        Returns:
            List of valid intersection points
        """
        if len(bisectors) < 2:
            return []
        
        intersections: list[Vector2D] = []
        
        for i in range(len(bisectors)):
            for j in range(i + 1, len(bisectors)):
                b1, b2 = bisectors[i], bisectors[j]
                
                dot_product = abs(np.dot(b1.direction, b2.direction))
                if dot_product > (1.0 - parallel_threshold):
                    continue
                
                point = cls.line_intersection_2d(
                    b1.origin, b1.direction,
                    b2.origin, b2.direction
                )
                
                if point is not None:
                    radius = np.linalg.norm(point)
                    if r_min <= radius <= r_max:
                        intersections.append(point)
        
        return intersections


# ==================== Circle Fitting ====================
class CircleFitter:
    """Fits circles to point clouds using various methods."""
    
    @staticmethod
    def fit_kasa(points: Points2D) -> tuple[Vector2D, float]:
        """Fit circle using Kåsa algebraic method (least squares).
        
        Args:
            points: 2D points (N, 2)
            
        Returns:
            Tuple of (center, radius)
            
        Raises:
            ValueError: If fewer than 3 points provided
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        
        x, y = points[:, 0], points[:, 1]
        A = np.column_stack([x, y, np.ones(len(points))])
        b = -(x**2 + y**2)
        
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        a, b_coef, c = params
        
        center = np.array([-a/2, -b_coef/2])
        radius = np.sqrt(center[0]**2 + center[1]**2 - c)
        
        return center, radius
    
    @staticmethod
    def fit_taubin(points: Points2D) -> tuple[Vector2D, float]:
        """Fit circle using Taubin algebraic method.
        
        More accurate than Kåsa for small arcs.
        
        Args:
            points: 2D points (N, 2)
            
        Returns:
            Tuple of (center, radius)
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        
        centroid = points.mean(axis=0)
        points_centered = points - centroid
        
        x, y = points_centered[:, 0], points_centered[:, 1]
        Mxx = (x**2).mean()
        Myy = (y**2).mean()
        Mxy = (x * y).mean()
        Mxz = (x * (x**2 + y**2)).mean()
        Myz = (y * (x**2 + y**2)).mean()
        Mzz = ((x**2 + y**2)**2).mean()
        
        M = np.array([[Mxx, Mxy, Mxz],
                      [Mxy, Myy, Myz],
                      [Mxz, Myz, Mzz]])
        
        N = np.array([[0, 0, -2],
                      [0, 0, -2],
                      [-2, -2, 8 * (Mxx + Myy)]])
        
        try:
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(N, M))
            idx = eigenvalues.argmin()
            A = eigenvectors[:, idx]
        except np.linalg.LinAlgError:
            return CircleFitter.fit_kasa(points)
        
        center_offset = A[:2] / (2 * A[2]) if abs(A[2]) > 1e-10 else np.zeros(2)
        center = centroid + center_offset
        radius = np.sqrt(center_offset[0]**2 + center_offset[1]**2 - A[2])
        
        return center, radius
    
    @staticmethod
    def _circle_residuals(params: FloatArray, points: Points2D) -> FloatArray:
        """Calculate residuals for circle fitting (geometric distance)."""
        cx, cy, r = params
        distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        return distances - r
    
    @classmethod
    def fit_nonlinear(
        cls,
        points: Points2D,
        initial: tuple[Vector2D, float]
    ) -> tuple[Vector2D, float]:
        """Refine circle fit using nonlinear least squares.
        
        Args:
            points: 2D points (N, 2)
            initial: Initial guess (center, radius)
            
        Returns:
            Tuple of (center, radius)
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        
        center_init, radius_init = initial
        params_init = np.array([center_init[0], center_init[1], radius_init])
        
        result = least_squares(
            cls._circle_residuals,
            params_init,
            args=(points,),
            method='lm'
        )
        
        center = result.x[:2]
        radius = result.x[2]
        
        return center, radius
    
    @classmethod
    def fit_ransac(
        cls,
        points: Points2D,
        min_samples: int,
        residual_threshold: float,
        n_iterations: int
    ) -> tuple[Vector2D, float, Points2D, Points2D, float]:
        """Fit circle using RANSAC to handle outliers robustly.
        
        Args:
            points: 2D points (N, 2)
            min_samples: Minimum samples for circle fitting
            residual_threshold: Maximum residual for inliers
            n_iterations: Number of RANSAC iterations
            
        Returns:
            Tuple of (center, radius, inliers, outliers, rmse)
        """
        if len(points) < min_samples:
            raise ValueError(f"Need at least {min_samples} points for RANSAC")
        
        # Initial fit with Taubin
        try:
            center_init, radius_init = cls.fit_taubin(points)
        except (ValueError, np.linalg.LinAlgError):
            center_init, radius_init = cls.fit_kasa(points)
        
        # Refine with nonlinear
        try:
            center_init, radius_init = cls.fit_nonlinear(points, (center_init, radius_init))
        except Exception:
            pass
        
        # RANSAC loop
        best_inliers: Points2D = np.empty((0, 2))
        best_center = center_init
        best_radius = radius_init
        
        actual_iterations = min(n_iterations, len(points) * 2)
        
        for _ in range(actual_iterations):
            sample_idx = np.random.choice(len(points), min_samples, replace=False)
            sample = points[sample_idx]
            
            try:
                center, radius = cls.fit_kasa(sample)
                distances = np.abs(cls._circle_residuals(
                    [center[0], center[1], radius], points
                ))
                inlier_mask = distances < residual_threshold
                inliers = points[inlier_mask]
                
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_center = center
                    best_radius = radius
            except Exception:
                continue
        
        # Final refit on all inliers
        if len(best_inliers) >= min_samples:
            try:
                best_center, best_radius = cls.fit_nonlinear(
                    best_inliers,
                    (best_center, best_radius)
                )
            except Exception:
                pass
        
        # Separate inliers and outliers
        distances = np.abs(cls._circle_residuals(
            [best_center[0], best_center[1], best_radius], points
        ))
        inlier_mask = distances < residual_threshold
        inliers = points[inlier_mask]
        outliers = points[~inlier_mask]
        
        # Calculate RMSE
        if len(inliers) > 0:
            rmse = np.sqrt(np.mean(cls._circle_residuals(
                [best_center[0], best_center[1], best_radius], inliers
            )**2))
        else:
            rmse = float('inf')
        
        return best_center, best_radius, inliers, outliers, rmse


# ==================== Gear Center Estimation ====================
class GearCenterEstimator:
    """Estimates the gear center using various methods."""
    
    @staticmethod
    def from_outer_tips(
        filtered_points: Points2D,
        r_outer: float
    ) -> tuple[Vector2D, float]:
        """Estimate gear center by fitting circle to outer boundary points.
        
        Args:
            filtered_points: Points in the annular region
            r_outer: Outer radius threshold
            
        Returns:
            Tuple of (center, radius)
        """
        if len(filtered_points) == 0:
            return np.zeros(2), r_outer
        
        radii = np.linalg.norm(filtered_points, axis=1)
        outer_threshold = r_outer * 0.95
        outer_points = filtered_points[radii >= outer_threshold]
        
        if len(outer_points) < 3:
            outer_points = filtered_points
        
        try:
            center, radius = CircleFitter.fit_taubin(outer_points)
            center, radius = CircleFitter.fit_nonlinear(outer_points, (center, radius))
        except Exception:
            center = outer_points.mean(axis=0)
            radius = np.linalg.norm(outer_points - center, axis=1).mean()
        
        return center, radius
    
    @staticmethod
    def from_boundary_centroid(
        slice_points: Points2D,
        r_outer: float
    ) -> tuple[Vector2D, float]:
        """Estimate gear center as centroid of boundary points projected to r_outer.
        
        Args:
            slice_points: All slice boundary points
            r_outer: Outer radius for projection
            
        Returns:
            Tuple of (center, radius)
        """
        if len(slice_points) == 0:
            return np.zeros(2), r_outer
        
        radii = np.linalg.norm(slice_points, axis=1)
        radii[radii < 1e-10] = 1.0
        
        projected = slice_points * (r_outer / radii[:, None])
        center = projected.mean(axis=0)
        
        return center, r_outer


# ==================== Visualization ====================
class Visualizer:
    """Creates visualizations for gear analysis results."""
    
    @staticmethod
    def make_lineset(
        p0: Vector3D,
        p1: Vector3D,
        color: Color
    ) -> o3d.geometry.LineSet:
        """Create a colored line segment for visualization."""
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([p0, p1]),
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        ls.colors = o3d.utility.Vector3dVector([color])
        return ls
    
    @staticmethod
    def make_circle(
        radius: float,
        z: float,
        color: Color,
        segments: int = 256
    ) -> o3d.geometry.LineSet:
        """Create a circular reference marker at given height and radius."""
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        pts = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full(segments, z)
        ])
        lines = [[i, (i + 1) % segments] for i in range(segments)]
        circle = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        circle.colors = o3d.utility.Vector3dVector([color] * len(lines))
        return circle
    
    @staticmethod
    def line_circle_intersections(
        origin: Vector2D,
        direction: Vector2D,
        radius: float
    ) -> Optional[Points2D]:
        """Return the two intersection points of a line with a centered circle."""
        if radius <= 0:
            return None
        
        d = unit_vector(direction)
        if np.linalg.norm(d) < 1e-10:
            return None
        
        a = np.dot(d, d)
        b = 2.0 * np.dot(origin, d)
        c = np.dot(origin, origin) - radius**2
        discriminant = b**2 - 4 * a * c
        
        if discriminant <= 0:
            return None
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        if t1 == t2:
            return None
        
        p1 = origin + t1 * d
        p2 = origin + t2 * d
        return np.vstack([p1, p2]) if t1 < t2 else np.vstack([p2, p1])
    
    @classmethod
    def bisector_display_segment(
        cls,
        bisector: PairBisector,
        r_outer: float
    ) -> tuple[Vector2D, Vector2D]:
        """Return start/end points for displaying a bisector line."""
        direction = unit_vector(bisector.direction)
        if np.linalg.norm(direction) < 1e-10:
            direction = np.array([1.0, 0.0])
        
        if np.dot(direction, bisector.origin) > 0:
            direction = -direction
        
        start = bisector.origin.copy()
        start_radius = np.linalg.norm(start)
        if start_radius > r_outer:
            intersections = cls.line_circle_intersections(bisector.origin, direction, r_outer)
            if intersections is not None:
                distances = np.linalg.norm(intersections - bisector.origin, axis=1)
                start = intersections[np.argmin(distances)]
        
        end = start + direction * bisector.length
        return start, end
    
    @classmethod
    def build_3d_geometries(cls, result: AnalysisResult) -> list[o3d.geometry.Geometry]:
        """Build all visualization geometries from analysis results."""
        config = result.config
        geometries: list[o3d.geometry.Geometry] = [
            result.mesh,
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]),
        ]
        
        # Slice points as gray point cloud
        slice_pc = o3d.geometry.PointCloud()
        slice_pc.points = o3d.utility.Vector3dVector(
            np.column_stack([
                result.filtered_points,
                np.full(len(result.filtered_points), config.slice_z)
            ])
        )
        slice_pc.paint_uniform_color([0.8, 0.8, 0.8])
        geometries.append(slice_pc)
        
        # Reference circles
        geometries.extend([
            cls.make_circle(config.r_inner, config.slice_z, (0.9, 0.4, 0.1)),
            cls.make_circle(config.r_outer, config.slice_z, (0.1, 0.6, 0.9)),
        ])
        
        # Flank lines
        for flank in result.flanks:
            center_3d = to_3d(flank.point, config.slice_z)
            direction_3d = np.append(flank.direction, 0.0)
            start = center_3d - config.flank_segment_length * direction_3d
            end = center_3d + config.flank_segment_length * direction_3d
            
            if flank.tooth == 1:
                geometries.append(cls.make_lineset(start, end, (1.0, 0.0, 0.0)))
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                sphere.translate(center_3d)
                sphere.paint_uniform_color([1.0, 0.0, 0.0])
                geometries.append(sphere)
            else:
                geometries.append(cls.make_lineset(start, end, (0.1, 0.8, 0.2)))
        
        # Bisector lines
        for bisector in result.bisectors:
            start_2d, end_2d = cls.bisector_display_segment(bisector, config.r_outer)
            start_3d = to_3d(start_2d, config.slice_z)
            end_3d = to_3d(end_2d, config.slice_z)
            geometries.append(cls.make_lineset(start_3d, end_3d, (0.0, 0.0, 0.0)))
        
        return geometries
    
    @classmethod
    def plot_2d_analysis(
        cls,
        result: AnalysisResult,
        output_path: Optional[Path] = None
    ) -> None:
        """Create comprehensive 2D visualization of the analysis."""
        config = result.config
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Slice points
        ax.scatter(result.slice_points[:, 0], result.slice_points[:, 1],
                   s=1, alpha=0.3, label='Slice boundary')
        ax.scatter(result.filtered_points[:, 0], result.filtered_points[:, 1],
                   s=2, alpha=0.5, label='Filtered points')
        
        # Reference circles
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(config.r_inner * np.cos(theta), config.r_inner * np.sin(theta),
                '--', linewidth=1.5, label=f'Inner radius ({config.r_inner})')
        ax.plot(config.r_outer * np.cos(theta), config.r_outer * np.sin(theta),
                '--', linewidth=1.5, label=f'Outer radius ({config.r_outer})')
        
        # Flank lines
        for flank in result.flanks:
            start = flank.point - config.flank_segment_length * flank.direction
            end = flank.point + config.flank_segment_length * flank.direction
            
            if flank.tooth == 1:
                ax.plot([start[0], end[0]], [start[1], end[1]],
                       linewidth=2, label='Tooth 1 flank')
            else:
                ax.plot([start[0], end[0]], [start[1], end[1]],
                       linewidth=1, alpha=0.7)
        
        # Bisectors
        for i, bisector in enumerate(result.bisectors):
            start, end = cls.bisector_display_segment(bisector, config.r_outer)
            ax.plot([start[0], end[0]], [start[1], end[1]],
                   'k-', linewidth=1, alpha=0.6,
                   label='Bisectors' if i == 0 else '')
        
        # Ghost circle
        if result.ghost_circle is not None:
            gc = result.ghost_circle
            
            if len(gc.inliers) > 0:
                ax.scatter(gc.inliers[:, 0], gc.inliers[:, 1],
                          s=50, marker='o', linewidths=2, facecolors='none',
                          label=f'Inlier intersections ({len(gc.inliers)})')
            
            circle_points = gc.center[:, None] + gc.radius * np.array([np.cos(theta), np.sin(theta)])
            ax.plot(circle_points[0], circle_points[1],
                   linewidth=2.5, label=f'Ghost circle (r={gc.radius:.3f}, RMSE={gc.rmse:.4f})')
            
            ax.annotate(
                f"r = {gc.radius:.3f}",
                xy=(gc.center[0] + gc.radius, gc.center[1]),
                xytext=(gc.center[0] + gc.radius + 0.3, gc.center[1]),
                fontsize=9, color="#0b5394", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", linewidth=1, color="#0b5394"),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            )
            
            ax.scatter([gc.center[0]], [gc.center[1]],
                      s=100, marker='+', linewidths=3, label='Ghost circle center')
        
        # Offset vector
        if result.gear_center is not None and result.offset_analysis is not None:
            gc_center = result.gear_center.center
            offset = result.offset_analysis
            
            ax.arrow(gc_center[0], gc_center[1],
                    offset.offset_vector[0], offset.offset_vector[1],
                    head_width=0.1, head_length=0.1, linewidth=2,
                    length_includes_head=True,
                    label=f'Offset ({offset.magnitude:.4f} @ {offset.angle_deg:.1f}°)')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Gear Flank Analysis with Ghost Circle')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved 2D plot to: {output_path}")
        else:
            plt.show()
        
        plt.close()


# ==================== Analysis Pipeline ====================
class GearAnalyzer:
    """Main analysis pipeline for gear flank analysis."""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize analyzer with configuration.
        
        Args:
            config: Analysis configuration parameters
        """
        self.config = config
    
    def run(self) -> AnalysisResult:
        """Execute complete gear flank analysis pipeline.
        
        Returns:
            AnalysisResult containing all computed geometry
        """
        config = self.config
        
        logger.info("Loading mesh...")
        mesh = MeshLoader.load(config.mesh_path, config.target_triangles)
        tm = MeshLoader.to_trimesh(mesh)
        
        logger.info(f"Extracting slice at Z={config.slice_z}...")
        slice_points = SliceExtractor.extract(
            tm, config.slice_z, config.slice_interpolation_density
        )
        
        logger.info(f"Filtering points between radii {config.r_inner:.2f} and {config.r_outer:.2f}...")
        filtered_points = PointFilter.by_radius(
            slice_points, config.r_inner, config.r_outer
        )
        logger.info(f"  Kept {len(filtered_points)} / {len(slice_points)} points")
        
        # Recenter geometry
        center_xy = slice_points[:, :2].mean(axis=0)
        logger.info(f"  Recentering by offset: [{center_xy[0]:.3f}, {center_xy[1]:.3f}]")
        slice_points[:, :2] -= center_xy
        filtered_points -= center_xy
        mesh.translate(np.array([-center_xy[0], -center_xy[1], 0.0]))
        
        # Partition into tooth sectors
        logger.info(f"Partitioning points into {config.n_teeth} tooth clusters...")
        tooth_clusters = ToothClusterer.partition_by_angle(
            filtered_points, config.n_teeth, config.min_points_per_cluster
        )
        non_empty = sum(1 for c in tooth_clusters if len(c) > 0)
        logger.info(f"  Found {non_empty} / {config.n_teeth} non-empty clusters")
        
        # Fit flank lines
        logger.info("Fitting flank lines...")
        flanks: list[FlankLine] = []
        for idx, cluster in enumerate(tooth_clusters, start=1):
            if len(cluster) == 0:
                continue
            try:
                point, direction = LineFitter.extract_right_flank(
                    cluster, config.min_points_per_flank
                )
                flanks.append(FlankLine(
                    tooth=idx,
                    point=point,
                    direction=unit_vector(direction),
                    cluster_size=len(cluster),
                ))
            except ValueError as e:
                logger.warning(f"Skipping tooth {idx}: {e}")
        
        if not flanks:
            raise RuntimeError("Unable to fit flanks for any tooth.")
        logger.info(f"  Successfully fitted {len(flanks)} flank lines")
        
        # Compute bisectors
        bisectors = BisectorComputer.compute_pair_bisectors(
            flanks, config.bisector_length, config.n_teeth
        )
        logger.info(f"  Computed {len(bisectors)} pair bisectors")
        
        # Compute bisector intersections and fit ghost circle
        logger.info("\n--- Ghost Circle Analysis ---")
        logger.info(f"Computing bisector intersections in radius range "
                   f"[{config.intersection_r_min:.3f}, {config.intersection_r_max:.3f}]...")
        
        intersections = IntersectionFinder.compute_bisector_intersections(
            bisectors,
            config.intersection_r_min,
            config.intersection_r_max,
            config.parallel_threshold
        )
        logger.info(f"  Found {len(intersections)} valid intersections")
        
        ghost_circle: Optional[GhostCircle] = None
        if len(intersections) >= config.ransac_min_samples:
            logger.info("Fitting ghost circle with RANSAC...")
            intersections_array = np.array(intersections)
            center, radius, inliers, outliers, rmse = CircleFitter.fit_ransac(
                intersections_array,
                config.ransac_min_samples,
                config.ransac_residual_threshold,
                config.ransac_iterations
            )
            
            ghost_circle = GhostCircle(
                center=center,
                radius=radius,
                inliers=inliers,
                outliers=outliers,
                rmse=rmse,
                n_intersections=len(intersections)
            )
            
            logger.info(f"  Ghost circle center: ({center[0]:.4f}, {center[1]:.4f})")
            logger.info(f"  Ghost circle radius: {radius:.4f}")
            logger.info(f"  Inliers: {len(inliers)} / {len(intersections)}")
            logger.info(f"  RMSE: {rmse:.6f}")
        else:
            logger.warning(f"Not enough intersections ({len(intersections)}) for ghost circle fitting")
        
        # Estimate gear center
        logger.info("\n--- Gear Center Estimation ---")
        
        center_outer, radius_outer = GearCenterEstimator.from_outer_tips(
            filtered_points, config.r_outer
        )
        logger.info(f"Method 1 (outer tips): center=({center_outer[0]:.4f}, {center_outer[1]:.4f}), "
                   f"radius={radius_outer:.4f}")
        
        center_boundary, radius_boundary = GearCenterEstimator.from_boundary_centroid(
            slice_points, config.r_outer
        )
        logger.info(f"Method 2 (boundary centroid): center=({center_boundary[0]:.4f}, "
                   f"{center_boundary[1]:.4f}), radius={radius_boundary:.4f}")
        
        use_outer_tips = config.gear_center_method == "outer_tips"
        gear_center = GearCenter(
            center=center_outer if use_outer_tips else center_boundary,
            method=config.gear_center_method,
            radius=radius_outer if use_outer_tips else radius_boundary
        )
        logger.info(f"Using method: {gear_center.method}")
        
        # Compute offset analysis
        offset_analysis: Optional[OffsetAnalysis] = None
        if ghost_circle is not None:
            offset_vector = ghost_circle.center - gear_center.center
            magnitude = np.linalg.norm(offset_vector)
            angle_rad = np.arctan2(offset_vector[1], offset_vector[0])
            angle_deg = np.degrees(angle_rad)
            
            offset_analysis = OffsetAnalysis(
                offset_vector=offset_vector,
                magnitude=magnitude,
                angle_deg=angle_deg,
                ghost_center=ghost_circle.center,
                gear_center=gear_center.center
            )
            
            logger.info(f"\n--- Offset Analysis (X-axis Setup Error Proxy) ---")
            logger.info(f"  Offset magnitude: {magnitude:.6f}")
            logger.info(f"  Offset angle: {angle_deg:.2f}° from +X axis")
        
        return AnalysisResult(
            config=config,
            mesh=mesh,
            slice_points=slice_points,
            filtered_points=filtered_points,
            center_shift=center_xy,
            flanks=flanks,
            bisectors=bisectors,
            ghost_circle=ghost_circle,
            gear_center=gear_center,
            offset_analysis=offset_analysis,
        )


# ==================== I/O ====================
class ResultExporter:
    """Exports analysis results to various formats."""
    
    @staticmethod
    def to_json(result: AnalysisResult, output_path: Path) -> None:
        """Save analysis results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(result.to_dict(), fp, indent=2)
        logger.info(f"Saved JSON report to: {output_path}")


# ==================== CLI ====================
def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyze crown gear geometry and compute ghost circle for setup error detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--stl', type=str, required=True,
        help='Path to STL mesh file'
    )
    parser.add_argument(
        '--z-tip', type=float, default=-0.2,
        help='Z-coordinate for slice extraction (default: -0.2)'
    )
    parser.add_argument(
        '--r-inner', type=float, default=2.38,
        help='Inner radius for point filtering (default: 2.38)'
    )
    parser.add_argument(
        '--r-outer', type=float, default=2.58,
        help='Outer radius for point filtering (default: 2.58)'
    )
    parser.add_argument(
        '--n-teeth', type=int, default=38,
        help='Number of teeth in gear (default: 38)'
    )
    parser.add_argument(
        '--units-scale', type=float, default=1.0,
        help='Scale factor for units (default: 1.0)'
    )
    parser.add_argument(
        '--outdir', type=str, default='results',
        help='Output directory for results (default: results/)'
    )
    parser.add_argument(
        '--no-viz', action='store_true',
        help='Skip 3D visualization'
    )
    parser.add_argument(
        '--gear-center-method', type=str, default='outer_tips',
        choices=['outer_tips', 'boundary_centroid'],
        help='Method for gear center estimation (default: outer_tips)'
    )
    
    return parser


def main() -> None:
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1:
        # CLI mode
        parser = create_argument_parser()
        args = parser.parse_args()
        config = AnalysisConfig.from_cli_args(args)
        show_viz = not args.no_viz
    else:
        # Default mode (for testing)
        config = AnalysisConfig()
        show_viz = True
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Crown Gear Ghost Circle Analysis")
    print("=" * 70)
    print(f"Input STL: {config.mesh_path}")
    print(f"Output directory: {config.output_dir.resolve()}")
    print(f"Configuration: Z={config.slice_z}, R_inner={config.r_inner}, "
          f"R_outer={config.r_outer}, N_teeth={config.n_teeth}")
    print("=" * 70 + "\n")
    
    # Run analysis
    analyzer = GearAnalyzer(config)
    result = analyzer.run()
    
    # Save results
    json_path = config.output_dir / "ghost_circle_report.json"
    ResultExporter.to_json(result, json_path)
    
    # Save 2D plot
    plot_path = config.output_dir / "gear_analysis_2d.png"
    Visualizer.plot_2d_analysis(result, plot_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("Analysis Summary:")
    print(f"  Flanks fitted: {len(result.flanks)}")
    print(f"  Bisectors computed: {len(result.bisectors)}")
    if result.ghost_circle:
        print(f"  Ghost circle: center=({result.ghost_circle.center[0]:.4f}, "
              f"{result.ghost_circle.center[1]:.4f}), r={result.ghost_circle.radius:.4f}")
    if result.offset_analysis:
        print(f"  Setup error (offset): {result.offset_analysis.magnitude:.6f} "
              f"@ {result.offset_analysis.angle_deg:.1f}°")
    print("=" * 70 + "\n")
    
    # 3D Visualization
    if show_viz:
        print("Launching 3D viewer...")
        geometries = Visualizer.build_3d_geometries(result)
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Gear Tooth Flank Analysis",
            width=1600,
            height=1000,
            mesh_show_back_face=True,
        )


if __name__ == "__main__":
    main()