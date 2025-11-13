from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
from scipy.optimize import least_squares

# ==================== Configuration ====================
MESH_PATH = Path(
    r"C:\Users\alibi\Documents\Gears Examples\2025_01_22_Kronenrad CT Messung_Ausgerichtet nach neuem Vorgehen.stl"
)
TARGET_TRIANGLES = 1000_000  # Target triangle count for mesh simplification eg.500_000, 1000_000, 2000_000
SLICE_Z = 0.2  # Z-height where gear slice is extracted eg. 0.2, 0.3, 0.4
R_INNER = 2.46  # Inner radius for point filtering
R_OUTER = 2.66  # Outer radius for point filtering eg. 2.46, 2.56, 2.66
OUTPUT_JSON = Path("flank_lines.json")
SLICE_INTERPOLATION_DENSITY = 0.002  # Max distance between interpolated points eg. 0.002, 0.003, 0.004
FLANK_SEGMENT_LENGTH = 0.50  # Visual length of flank lines eg. 0.50, 0.60, 0.70
BISECTOR_LENGTH = 5.0  # Visual length of bisector lines eg. 5.0, 6.0, 7.0
N_TEETH = 38  # Expected number of teeth in gear eg. 38, 39, 40
MIN_POINTS_PER_CLUSTER = 6  # Minimum points to consider a cluster valid eg. 6, 7, 8
MIN_POINTS_PER_FLANK = 5  # Minimum points needed to fit a flank line eg. 5, 6, 7
PARALLEL_THRESHOLD = 0.05  # Skip bisector pairs with angle difference below ~5%
RANSAC_MIN_SAMPLES = 3  # Minimum points to define a circle
RANSAC_RESIDUAL_THRESHOLD = 0.05  # Max distance (in units) for intersection to count as inlier
INTERSECTION_R_MIN_FACTOR = 0.05  # Fraction of R_INNER for minimum intersection radius
INTERSECTION_R_MAX_FACTOR = 0.5  # Fraction of R_INNER for maximum intersection radius


# ==================== Data Structures ====================
@dataclass
class FlankLine:
    """Represents a fitted line for a tooth flank.
    
    Attributes:
        tooth: Tooth number (1-indexed)
        point: 2D centroid of the flank points
        direction: 2D unit vector defining flank direction
        cluster_size: Number of points in this tooth cluster
    """
    tooth: int
    point: np.ndarray
    direction: np.ndarray
    cluster_size: int


@dataclass
class PairBisector:
    """Represents the angle bisector between two adjacent tooth flanks.
    
    Attributes:
        between_teeth: Tuple of two tooth numbers (e.g., (2, 3))
        origin: 2D midpoint between the two flank centroids
        direction: 2D unit vector of the bisector direction
        length: Visual length for rendering
    """
    between_teeth: tuple[int, int]
    origin: np.ndarray
    direction: np.ndarray
    length: float


@dataclass
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
    center: np.ndarray
    radius: float
    inliers: np.ndarray
    outliers: np.ndarray
    rmse: float
    n_intersections: int


@dataclass
class GearCenter:
    """Represents the estimated gear center.
    
    Attributes:
        center: 2D center coordinates (gx, gy)
        method: Method used ('outer_tips' or 'boundary_centroid')
        radius: Associated radius (if applicable)
    """
    center: np.ndarray
    method: str
    radius: Optional[float] = None


@dataclass
class OffsetAnalysis:
    """Represents the offset between ghost circle and gear center.
    
    Attributes:
        offset_vector: 2D vector from gear center to ghost circle center
        magnitude: Magnitude of offset (proxy for X-axis setup error)
        angle_deg: Angle of offset in degrees from +X axis
        ghost_center: Ghost circle center (cx, cy)
        gear_center: Gear center (gx, gy)
    """
    offset_vector: np.ndarray
    magnitude: float
    angle_deg: float
    ghost_center: np.ndarray
    gear_center: np.ndarray


@dataclass
class AnalysisResult:
    """Complete analysis results container.
    
    Attributes:
        mesh: Original 3D mesh (cleaned and potentially simplified)
        slice_points: All points from the horizontal slice at SLICE_Z
        filtered_points: Subset of slice_points within R_INNER and R_OUTER
        center_shift: 2D offset applied to recenter the mesh
        flanks: List of fitted flank lines for each detected tooth
        bisectors: List of bisectors between adjacent tooth pairs
        ghost_circle: Fitted ghost circle from bisector intersections
        gear_center: Estimated gear center
        offset_analysis: Offset analysis between ghost circle and gear center
    """
    mesh: o3d.geometry.TriangleMesh
    slice_points: np.ndarray
    filtered_points: np.ndarray
    center_shift: np.ndarray
    flanks: list[FlankLine]
    bisectors: list[PairBisector]
    ghost_circle: Optional[GhostCircle] = None
    gear_center: Optional[GearCenter] = None
    offset_analysis: Optional[OffsetAnalysis] = None


# ==================== Utility Functions ====================
def unit_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length. Returns original if zero-length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


def to_3d(point_2d: np.ndarray, z: float) -> np.ndarray:
    """Convert 2D point (x, y) to 3D by appending z coordinate."""
    return np.array([point_2d[0], point_2d[1], z])


# ==================== Mesh Processing ====================
def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Load, clean, and optionally simplify a mesh from file.
    
    Steps:
    1. Load STL file
    2. Remove duplicate vertices and degenerate triangles
    3. Compute vertex normals
    4. Simplify to TARGET_TRIANGLES if mesh is too large
    
    Args:
        path: Path to STL file
        
    Returns:
        Cleaned Open3D triangle mesh
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
    if num_triangles > TARGET_TRIANGLES:
        print(f"  Simplifying mesh: {num_triangles} → {TARGET_TRIANGLES} triangles")
        mesh = mesh.simplify_quadric_decimation(TARGET_TRIANGLES)
    
    return mesh


def mesh_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """Convert Open3D mesh to Trimesh format for slicing operations."""
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False  # Skip automatic processing for speed
    )


# ==================== Slice Processing ====================
def interpolate_path(path: np.ndarray, max_step: float) -> np.ndarray:
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
    
    segments = []
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        segment_length = np.linalg.norm(p1 - p0)
        
        if segment_length > max_step:
            # Calculate number of points needed for this segment
            num_points = int(np.ceil(segment_length / max_step)) + 1
            t = np.linspace(0, 1, num_points)
            # Linear interpolation between p0 and p1
            segment = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
            segments.append(segment[:-1])  # Exclude endpoint to avoid duplicates
        else:
            segments.append(p0[None, :])
    
    segments.append(path[-1:])  # Add final point
    return np.vstack(segments)


def extract_slice(tm: trimesh.Trimesh, z: float, max_step: float) -> np.ndarray:
    """Extract a horizontal slice from the mesh and interpolate points.
    
    Creates a 2D cross-section by intersecting the mesh with a horizontal plane.
    The resulting contour is then densified to ensure uniform point spacing.
    
    Args:
        tm: Trimesh object
        z: Z-coordinate of slicing plane
        max_step: Maximum spacing between interpolated points
        
    Returns:
        Array of 2D points (N, 2) representing the slice contour
    """
    # Slice mesh with horizontal plane
    section = tm.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None or section.vertices.size == 0:
        raise RuntimeError(f"No intersection found at Z={z}")
    
    # Extract 2D coordinates from slice
    verts = np.asarray(section.vertices, dtype=float)
    xy = verts[:, :2] if verts.shape[1] >= 2 else verts
    
    # Process connected path entities from the slice
    paths = []
    if hasattr(section, "entities") and len(section.entities) > 0:
        for ent in section.entities:
            if hasattr(ent, "points") and len(ent.points) >= 2:
                pts = xy[np.asarray(ent.points, dtype=int)]
                paths.append(interpolate_path(pts, max_step))
    
    # Fallback: treat all points as a single path if no entities found
    if not paths and len(xy) >= 2:
        paths.append(interpolate_path(xy, max_step))
    
    if not paths:
        raise RuntimeError(f"No valid paths extracted from slice at Z={z}")
    
    return np.vstack(paths)


def filter_by_radius(points: np.ndarray, r_inner: float, r_outer: float) -> np.ndarray:
    """Filter points to keep only those within an annular region.
    
    Keeps points where r_inner <= distance_from_origin <= r_outer.
    If no points found, tries relaxed bounds (±10%).
    
    Args:
        points: 2D point array (N, 2)
        r_inner: Inner radius bound
        r_outer: Outer radius bound
        
    Returns:
        Filtered 2D point array
    """
    radii = np.linalg.norm(points, axis=1)
    mask = (radii >= r_inner) & (radii <= r_outer)
    filtered = points[mask]
    
    # Relaxed bounds fallback if nothing found
    if len(filtered) == 0:
        relaxed_inner, relaxed_outer = r_inner * 0.9, r_outer * 1.1
        mask = (radii >= relaxed_inner) & (radii <= relaxed_outer)
        filtered = points[mask]
        if len(filtered) == 0:
            raise RuntimeError(
                f"No points found within radii {r_inner:.2f}-{r_outer:.2f} "
                f"(relaxed: {relaxed_inner:.2f}-{relaxed_outer:.2f})"
            )
        print(f"  Warning: Using relaxed radii: {relaxed_inner:.2f}-{relaxed_outer:.2f}")
    
    return filtered


# ==================== Tooth Clustering ====================
def partition_points_into_teeth_binned(points_xy: np.ndarray, n_teeth: int) -> list[np.ndarray]:
    """Partition points into exactly n_teeth angular sectors.
    
    Divides the full circle into equal angular bins, one per tooth.
    This guarantees exactly N_TEETH clusters (some may be empty).
    
    Algorithm:
    1. Calculate polar angle for each point
    2. Create n_teeth equal angular bins spanning [0, 2π)
    3. Assign each point to its corresponding bin
    4. Keep bins with at least MIN_POINTS_PER_CLUSTER points
    
    Args:
        points_xy: 2D point array (N, 2)
        n_teeth: Expected number of teeth
        
    Returns:
        List of n_teeth arrays (one per tooth, may be empty)
    """
    if len(points_xy) == 0 or n_teeth < 1:
        return []
    
    # Calculate polar angles and normalize to [0, 2π)
    angles = np.arctan2(points_xy[:, 1], points_xy[:, 0])
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    
    # Create angular bin edges
    edges = np.linspace(0.0, 2 * np.pi, n_teeth + 1)
    
    # Assign points to bins
    bin_idx = np.digitize(angles, edges, right=False) - 1
    bin_idx[bin_idx == n_teeth] = 0  # Wrap last bin to first
    
    # Extract points for each tooth
    clusters = []
    for k in range(n_teeth):
        pts = points_xy[bin_idx == k]
        # Keep cluster only if it has enough points, otherwise return empty array
        if len(pts) >= MIN_POINTS_PER_CLUSTER:
            clusters.append(pts)
        else:
            clusters.append(np.empty((0, 2)))  # Preserve tooth index
    
    return clusters


# ==================== Line Fitting ====================
def fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a line to 2D points using SVD (Principal Component Analysis).
    
    Returns the centroid and principal direction of the point cloud.
    The direction corresponds to the axis of maximum variance.
    
    Args:
        points: 2D point array (N, 2)
        
    Returns:
        Tuple of (centroid, direction_vector)
    """
    if len(points) < 2:
        raise ValueError("Need at least two points to fit a line.")
    
    centroid = points.mean(axis=0)
    # SVD on centered points: first singular vector = principal direction
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    direction = unit_vector(vt[0])
    return centroid, direction


def extract_right_flank(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract and fit a line to the right flank of a tooth cluster.
    
    Strategy:
    1. Calculate cluster center and local coordinate system
    2. Split points into left/right based on tangential projection
    3. Fit line to right-side points only
    
    Args:
        points: 2D points belonging to one tooth (N, 2)
        
    Returns:
        Tuple of (flank_centroid, flank_direction)
    """
    center = points.mean(axis=0)
    
    # Define local radial and tangential directions
    radial = unit_vector(center) if np.linalg.norm(center) > 1e-10 else np.array([1.0, 0.0])
    tangential = np.array([-radial[1], radial[0]])  # 90° rotation
    
    # Project points onto tangential axis and split at median
    projections = (points - center) @ tangential
    median = np.median(projections)
    right = points[projections > median]
    
    # Fallback: ensure enough points for fitting
    if len(right) < MIN_POINTS_PER_FLANK:
        order = np.argsort(projections)
        half = max(MIN_POINTS_PER_FLANK, len(points) // 2)
        right = points[order[-half:]]
        if len(right) < 3:
            raise ValueError("Insufficient points for flank fitting.")
    
    return fit_line(right)


# ==================== Bisector Computation ====================
def compute_bisector(
    point_a: np.ndarray, dir_a: np.ndarray,
    point_b: np.ndarray, dir_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the angle bisector between two lines.
    
    The bisector origin is the midpoint between the two line centers.
    The bisector direction is the normalized sum of the two line directions.
    
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
    
    # Ensure directions point in similar direction (flip if needed)
    if np.dot(dir_a, dir_b) < 0:
        dir_b = -dir_b
    
    # Bisector direction is the average of the two directions
    bisector_dir = unit_vector(dir_a + dir_b)
    origin = 0.5 * (point_a + point_b)
    
    # Fallback for degenerate cases (parallel lines)
    if np.linalg.norm(bisector_dir) < 1e-8:
        bisector_dir = unit_vector(origin) if np.linalg.norm(origin) > 1e-10 else dir_a
    
    return origin, bisector_dir


def compute_pair_bisectors(flanks: list[FlankLine], length: float) -> list[PairBisector]:
    """Compute bisectors for consecutive even-odd tooth pairs with wraparound.
    
    Creates bisectors between: (2-3), (4-5), (6-7), ..., (38-1)
    This pairing strategy is useful for gear measurement applications.
    The last pair wraps around from the highest tooth to tooth 1.
    
    Args:
        flanks: List of fitted flank lines
        length: Visual length for bisector rendering
        
    Returns:
        List of bisector objects
    """
    if len(flanks) < 2:
        return []
    
    # Create a dictionary for fast tooth lookup by tooth number
    flank_dict = {f.tooth: f for f in flanks}
    
    bisectors = []
    # Iterate through pairs: (2,3), (4,5), (6,7), ...
    for tooth_num in range(2, N_TEETH + 1, 2):
        next_tooth_num = tooth_num + 1 if tooth_num < N_TEETH else 1  # Wraparound
        
        # Check if both teeth exist in the fitted flanks
        if tooth_num in flank_dict and next_tooth_num in flank_dict:
            current = flank_dict[tooth_num]
            nxt = flank_dict[next_tooth_num]
            
            origin, direction = compute_bisector(
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
def line_intersection_2d(
    p1: np.ndarray, d1: np.ndarray,
    p2: np.ndarray, d2: np.ndarray
) -> Optional[np.ndarray]:
    """Find intersection point of two 2D lines.
    
    Uses parametric line representation: L = p + t*d
    Returns None if lines are parallel (within numerical tolerance).
    
    Args:
        p1: Point on first line
        d1: Direction of first line (unit vector)
        p2: Point on second line
        d2: Direction of second line (unit vector)
        
    Returns:
        Intersection point or None if parallel
    """
    # Check if lines are nearly parallel
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return None
    
    # Solve for parameter t: p1 + t*d1 = p2 + s*d2
    dp = p2 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
    
    return p1 + t * d1


def compute_bisector_intersections(
    bisectors: list[PairBisector],
    r_min: float,
    r_max: float
) -> list[np.ndarray]:
    """Compute pairwise intersections of bisectors near the center.
    
    Intersects all bisector pairs, filtering out:
    - Nearly parallel bisectors (dot product > threshold)
    - Intersections outside the specified radius range
    
    Args:
        bisectors: List of bisector lines
        r_min: Minimum radius for valid intersections
        r_max: Maximum radius for valid intersections
        
    Returns:
        List of valid intersection points
    """
    if len(bisectors) < 2:
        return []
    
    intersections = []
    
    for i in range(len(bisectors)):
        for j in range(i + 1, len(bisectors)):
            b1, b2 = bisectors[i], bisectors[j]
            
            # Skip nearly parallel bisectors
            dot_product = abs(np.dot(b1.direction, b2.direction))
            if dot_product > (1.0 - PARALLEL_THRESHOLD):
                continue
            
            # Compute intersection
            point = line_intersection_2d(b1.origin, b1.direction, b2.origin, b2.direction)
            
            if point is not None:
                # Check if intersection is within radius bounds
                radius = np.linalg.norm(point)
                if r_min <= radius <= r_max:
                    intersections.append(point)
    
    return intersections


# ==================== Circle Fitting ====================
def fit_circle_kasa(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit circle using Kåsa algebraic method (least squares).
    
    Fast algebraic method that minimizes algebraic distance.
    Good initial estimate but not geometrically optimal.
    
    Args:
        points: 2D points (N, 2)
        
    Returns:
        Tuple of (center, radius)
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a circle")
    
    # Build design matrix: [x, y, 1] for equation x^2 + y^2 + a*x + b*y + c = 0
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([x, y, np.ones(len(points))])
    b = -(x**2 + y**2)
    
    # Solve least squares
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coef, c = params
    
    # Extract center and radius
    center = np.array([-a/2, -b_coef/2])
    radius = np.sqrt(center[0]**2 + center[1]**2 - c)
    
    return center, radius


def fit_circle_taubin(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit circle using Taubin algebraic method.
    
    More accurate than Kåsa for small arcs.
    
    Args:
        points: 2D points (N, 2)
        
    Returns:
        Tuple of (center, radius)
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a circle")
    
    # Center data
    centroid = points.mean(axis=0)
    points_centered = points - centroid
    
    # Build moment matrix
    x, y = points_centered[:, 0], points_centered[:, 1]
    Mxx = (x**2).mean()
    Myy = (y**2).mean()
    Mxy = (x * y).mean()
    Mxz = (x * (x**2 + y**2)).mean()
    Myz = (y * (x**2 + y**2)).mean()
    Mzz = ((x**2 + y**2)**2).mean()
    
    # Build matrices for generalized eigenvalue problem
    M = np.array([[Mxx, Mxy, Mxz],
                  [Mxy, Myy, Myz],
                  [Mxz, Myz, Mzz]])
    
    N = np.array([[0, 0, -2],
                  [0, 0, -2],
                  [-2, -2, 8 * (Mxx + Myy)]])
    
    # Solve generalized eigenvalue problem
    try:
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(N, M))
        idx = eigenvalues.argmin()
        A = eigenvectors[:, idx]
    except np.linalg.LinAlgError:
        # Fallback to Kåsa
        return fit_circle_kasa(points)
    
    # Extract center and radius
    center_offset = A[:2] / (2 * A[2]) if abs(A[2]) > 1e-10 else np.zeros(2)
    center = centroid + center_offset
    radius = np.sqrt(center_offset[0]**2 + center_offset[1]**2 - A[2])
    
    return center, radius


def circle_residuals(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Calculate residuals for circle fitting (geometric distance).
    
    Args:
        params: [cx, cy, r] circle parameters
        points: 2D points (N, 2)
        
    Returns:
        Array of residuals (signed distances)
    """
    cx, cy, r = params
    distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
    return distances - r


def fit_circle_nonlinear(points: np.ndarray, initial: tuple[np.ndarray, float]) -> tuple[np.ndarray, float]:
    """Refine circle fit using nonlinear least squares.
    
    Minimizes geometric distance (more accurate than algebraic methods).
    
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
    
    # Run nonlinear optimization
    result = least_squares(
        circle_residuals,
        params_init,
        args=(points,),
        method='lm'  # Levenberg-Marquardt
    )
    
    center = result.x[:2]
    radius = result.x[2]
    
    return center, radius


def fit_circle_ransac(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
    """Fit circle using RANSAC to handle outliers robustly.
    
    Pipeline:
    1. Initial fit with Taubin method
    2. Refine with nonlinear least squares
    3. RANSAC to identify inliers
    4. Final refit on inliers only
    
    Args:
        points: 2D points (N, 2)
        
    Returns:
        Tuple of (center, radius, inliers, outliers, rmse)
    """
    if len(points) < RANSAC_MIN_SAMPLES:
        raise ValueError(f"Need at least {RANSAC_MIN_SAMPLES} points for RANSAC")
    
    # Initial fit with Taubin
    try:
        center_init, radius_init = fit_circle_taubin(points)
    except (ValueError, np.linalg.LinAlgError):
        center_init, radius_init = fit_circle_kasa(points)
    
    # Refine with nonlinear
    try:
        center_init, radius_init = fit_circle_nonlinear(points, (center_init, radius_init))
    except Exception:
        pass  # Use initial fit if refinement fails
    
    # Custom RANSAC for circle fitting
    best_inliers = []
    best_center = center_init
    best_radius = radius_init
    
    n_iterations = min(100, len(points) * 2)
    
    for _ in range(n_iterations):
        # Sample random subset
        sample_idx = np.random.choice(len(points), RANSAC_MIN_SAMPLES, replace=False)
        sample = points[sample_idx]
        
        try:
            # Fit circle to sample
            center, radius = fit_circle_kasa(sample)
            
            # Find inliers
            distances = np.abs(circle_residuals([center[0], center[1], radius], points))
            inlier_mask = distances < RANSAC_RESIDUAL_THRESHOLD
            inliers = points[inlier_mask]
            
            # Update best model if more inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_center = center
                best_radius = radius
        except Exception:
            continue
    
    # Final refit on all inliers
    if len(best_inliers) >= RANSAC_MIN_SAMPLES:
        try:
            best_center, best_radius = fit_circle_nonlinear(
                best_inliers, 
                (best_center, best_radius)
            )
        except Exception:
            pass
    
    # Separate inliers and outliers
    distances = np.abs(circle_residuals([best_center[0], best_center[1], best_radius], points))
    inlier_mask = distances < RANSAC_RESIDUAL_THRESHOLD
    inliers = points[inlier_mask]
    outliers = points[~inlier_mask]
    
    # Calculate RMSE
    if len(inliers) > 0:
        rmse = np.sqrt(np.mean(circle_residuals([best_center[0], best_center[1], best_radius], inliers)**2))
    else:
        rmse = float('inf')
    
    return best_center, best_radius, inliers, outliers, rmse


# ==================== Gear Center Estimation ====================
def estimate_gear_center_from_outer_tips(filtered_points: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate gear center by fitting circle to outer boundary points.
    
    Uses points near R_OUTER to fit a circle representing the gear's outer profile.
    
    Args:
        filtered_points: Points in the annular region
        
    Returns:
        Tuple of (center, radius)
    """
    if len(filtered_points) == 0:
        return np.zeros(2), R_OUTER
    
    # Select points near outer radius
    radii = np.linalg.norm(filtered_points, axis=1)
    outer_threshold = R_OUTER * 0.95
    outer_points = filtered_points[radii >= outer_threshold]
    
    if len(outer_points) < 3:
        outer_points = filtered_points  # Fallback
    
    # Fit circle
    try:
        center, radius = fit_circle_taubin(outer_points)
        center, radius = fit_circle_nonlinear(outer_points, (center, radius))
    except Exception:
        center = outer_points.mean(axis=0)
        radius = np.linalg.norm(outer_points - center, axis=1).mean()
    
    return center, radius


def estimate_gear_center_from_boundary_centroid(slice_points: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate gear center as centroid of boundary points projected to R_OUTER.
    
    Args:
        slice_points: All slice boundary points
        
    Returns:
        Tuple of (center, radius)
    """
    if len(slice_points) == 0:
        return np.zeros(2), R_OUTER
    
    # Project points to R_OUTER
    radii = np.linalg.norm(slice_points, axis=1)
    radii[radii < 1e-10] = 1.0  # Avoid division by zero
    
    projected = slice_points * (R_OUTER / radii[:, None])
    center = projected.mean(axis=0)
    
    return center, R_OUTER


# ==================== Visualization ====================
def make_lineset(p0: np.ndarray, p1: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    """Create a colored line segment for visualization."""
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([p0, p1]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls


def make_circle(radius: float, z: float, color: tuple[float, float, float], segments: int = 256) -> o3d.geometry.LineSet:
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


def build_visual_geometries(result: AnalysisResult) -> list[o3d.geometry.Geometry]:
    """Build all visualization geometries from analysis results.
    
    Creates:
    - Original mesh
    - Coordinate frame
    - Slice point cloud (gray)
    - Reference circles (inner/outer radii)
    - Flank lines (green, tooth 1 in red)
    - Bisector lines (black)
    - Red sphere marker for tooth 1
    """
    geometries = [
        result.mesh,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]),
    ]
    
    # Add slice points as gray point cloud
    slice_pc = o3d.geometry.PointCloud()
    slice_pc.points = o3d.utility.Vector3dVector(
        np.column_stack([result.filtered_points, np.full(len(result.filtered_points), SLICE_Z)])
    )
    slice_pc.paint_uniform_color([0.8, 0.8, 0.8])
    geometries.append(slice_pc)
    
    # Add reference circles
    geometries.extend([
        make_circle(R_INNER, SLICE_Z, (0.9, 0.4, 0.1)),  # Orange inner circle
        make_circle(R_OUTER, SLICE_Z, (0.1, 0.6, 0.9)),  # Blue outer circle
    ])
    
    # Add flank lines
    for flank in result.flanks:
        center_3d = to_3d(flank.point, SLICE_Z)
        direction_3d = np.append(flank.direction, 0.0)
        start = center_3d - FLANK_SEGMENT_LENGTH * direction_3d
        end = center_3d + FLANK_SEGMENT_LENGTH * direction_3d
        
        if flank.tooth == 1:
            # Special marking for tooth 1 (reference tooth)
            geometries.append(make_lineset(start, end, (1.0, 0.0, 0.0)))  # Red line
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.translate(center_3d)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red sphere
            geometries.append(sphere)
        else:
            geometries.append(make_lineset(start, end, (0.1, 0.8, 0.2)))  # Green lines
    
    # Add bisector lines
    for bisector in result.bisectors:
        origin_3d = to_3d(bisector.origin, SLICE_Z)
        direction_3d = np.append(bisector.direction, 0.0)
        start = origin_3d - 0.5 * bisector.length * direction_3d
        end = origin_3d + 0.5 * bisector.length * direction_3d
        geometries.append(make_lineset(start, end, (0.0, 0.0, 0.0)))  # Black lines
    
    return geometries


def plot_2d_analysis(result: AnalysisResult, output_path: Optional[Path] = None) -> None:
    """Create comprehensive 2D visualization of the analysis.
    
    Shows:
    - Slice boundary points
    - Inner/outer radius circles
    - Tooth flank points and fitted lines
    - Bisectors
    - Bisector intersections (inliers vs outliers)
    - Ghost circle
    - Gear center and ghost circle center
    - Offset vector
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot slice points (light gray background)
    ax.scatter(result.slice_points[:, 0], result.slice_points[:, 1],
               s=1, alpha=0.3, label='Slice boundary')
    
    # Plot filtered points (darker)
    ax.scatter(result.filtered_points[:, 0], result.filtered_points[:, 1],
               s=2, alpha=0.5, label='Filtered points')
    
    # Plot reference circles
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_INNER * np.cos(theta), R_INNER * np.sin(theta),
            '--', linewidth=1.5, label=f'Inner radius ({R_INNER})')
    ax.plot(R_OUTER * np.cos(theta), R_OUTER * np.sin(theta),
            '--', linewidth=1.5, label=f'Outer radius ({R_OUTER})')
    
    # Plot flank lines
    for flank in result.flanks:
        start = flank.point - FLANK_SEGMENT_LENGTH * flank.direction
        end = flank.point + FLANK_SEGMENT_LENGTH * flank.direction
        
        if flank.tooth == 1:
            ax.plot([start[0], end[0]], [start[1], end[1]],
                   linewidth=2, label='Tooth 1 flank' if flank.tooth == 1 else '')
        else:
            ax.plot([start[0], end[0]], [start[1], end[1]],
                   linewidth=1, alpha=0.7)
    
    # Plot bisectors
    for i, bisector in enumerate(result.bisectors):
        start = bisector.origin - 0.5 * bisector.length * bisector.direction
        end = bisector.origin + 0.5 * bisector.length * bisector.direction
        ax.plot([start[0], end[0]], [start[1], end[1]],
               'k-', linewidth=1, alpha=0.6,
               label='Bisectors' if i == 0 else '')
    
    # Plot ghost circle analysis if available
    if result.ghost_circle is not None:
        gc = result.ghost_circle
        
        # Plot intersection points
        if len(gc.outliers) > 0:
            ax.scatter(gc.outliers[:, 0], gc.outliers[:, 1],
                      s=30, marker='x', linewidths=2,
                      label=f'Outlier intersections ({len(gc.outliers)})')
        
        if len(gc.inliers) > 0:
            ax.scatter(gc.inliers[:, 0], gc.inliers[:, 1],
                      s=50, marker='o', linewidths=2, facecolors='none',
                      label=f'Inlier intersections ({len(gc.inliers)})')
        
        # Plot ghost circle
        circle_points = gc.center[:, None] + gc.radius * np.array([np.cos(theta), np.sin(theta)])
        ax.plot(circle_points[0], circle_points[1],
               linewidth=2.5, label=f'Ghost circle (r={gc.radius:.3f}, RMSE={gc.rmse:.4f})')
        ax.annotate(
            f"r = {gc.radius:.3f}",
            xy=(gc.center[0] + gc.radius, gc.center[1]),
            xytext=(gc.center[0] + gc.radius + 0.3, gc.center[1]),
            textcoords="data",
            fontsize=9,
            color="#0b5394",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="->", linewidth=1, color="#0b5394"),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )
        
        # Plot ghost circle center
        ax.scatter([gc.center[0]], [gc.center[1]],
                  s=100, marker='+', linewidths=3,
                  label='Ghost circle center')
    
    # Plot gear center and offset if available
    if result.gear_center is not None and result.offset_analysis is not None:
        gc_center = result.gear_center.center
        offset = result.offset_analysis
        
        # Plot gear center
        ax.scatter([gc_center[0]], [gc_center[1]],
                  s=100, marker='x', linewidths=3,
                  label=f'Gear center ({result.gear_center.method})')
        
        # Plot offset vector
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
        print(f"  Saved 2D plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


# ==================== Analysis Pipeline ====================
def analyse_gear_slice() -> AnalysisResult:
    """Execute complete gear flank analysis pipeline.
    
    Pipeline steps:
    1. Load and clean mesh
    2. Extract horizontal slice at SLICE_Z
    3. Filter points to annular region (R_INNER to R_OUTER)
    4. Recenter geometry to slice centroid
    5. Partition points into N_TEETH angular sectors
    6. Fit flank lines to each tooth cluster
    7. Compute bisectors between adjacent tooth pairs
    8. Compute bisector intersections and fit ghost circle
    9. Estimate gear center and compute offset
    
    Returns:
        AnalysisResult containing all computed geometry
    """
    print("Loading mesh...")
    mesh = load_mesh(MESH_PATH)
    tm = mesh_to_trimesh(mesh)
    
    print(f"Extracting slice at Z={SLICE_Z}...")
    slice_points = extract_slice(tm, SLICE_Z, SLICE_INTERPOLATION_DENSITY)
    
    print(f"Filtering points between radii {R_INNER:.2f} and {R_OUTER:.2f}...")
    filtered_points = filter_by_radius(slice_points, R_INNER, R_OUTER)
    print(f"  Kept {len(filtered_points)} / {len(slice_points)} points")
    
    # Recenter geometry to slice centroid
    center_xy = slice_points[:, :2].mean(axis=0)
    print(f"  Recentering by offset: [{center_xy[0]:.3f}, {center_xy[1]:.3f}]")
    slice_points[:, :2] -= center_xy
    filtered_points -= center_xy
    mesh.translate(np.array([-center_xy[0], -center_xy[1], 0.0]))
    
    # Partition into tooth sectors
    print(f"Partitioning points into {N_TEETH} tooth clusters...")
    tooth_clusters = partition_points_into_teeth_binned(filtered_points, N_TEETH)
    non_empty = sum(1 for c in tooth_clusters if len(c) > 0)
    print(f"  Found {non_empty} / {N_TEETH} non-empty clusters")
    
    # Fit flank lines
    print("Fitting flank lines...")
    flanks = []
    for idx, cluster in enumerate(tooth_clusters, start=1):
        if len(cluster) == 0:
            continue  # Skip empty clusters
        try:
            point, direction = extract_right_flank(cluster)
            flanks.append(FlankLine(
                tooth=idx,
                point=point,
                direction=unit_vector(direction),
                cluster_size=len(cluster),
            ))
        except ValueError as e:
            print(f"  Warning: Skipping tooth {idx}: {e}")
            continue
    
    if not flanks:
        raise RuntimeError("Unable to fit flanks for any tooth.")
    print(f"  Successfully fitted {len(flanks)} flank lines")
    
    # Compute bisectors
    bisectors = compute_pair_bisectors(flanks, BISECTOR_LENGTH)
    print(f"  Computed {len(bisectors)} pair bisectors")
    
    # Compute bisector intersections and fit ghost circle
    print("\n--- Ghost Circle Analysis ---")
    r_min = R_INNER * INTERSECTION_R_MIN_FACTOR
    r_max = R_OUTER * INTERSECTION_R_MAX_FACTOR
    print(f"Computing bisector intersections in radius range [{r_min:.3f}, {r_max:.3f}]...")
    
    intersections = compute_bisector_intersections(bisectors, r_min, r_max)
    print(f"  Found {len(intersections)} valid intersections")
    
    ghost_circle = None
    if len(intersections) >= RANSAC_MIN_SAMPLES:
        print("Fitting ghost circle with RANSAC...")
        intersections_array = np.array(intersections)
        center, radius, inliers, outliers, rmse = fit_circle_ransac(intersections_array)
        
        ghost_circle = GhostCircle(
            center=center,
            radius=radius,
            inliers=inliers,
            outliers=outliers,
            rmse=rmse,
            n_intersections=len(intersections)
        )
        
        print(f"  Ghost circle center: ({center[0]:.4f}, {center[1]:.4f})")
        print(f"  Ghost circle radius: {radius:.4f}")
        print(f"  Inliers: {len(inliers)} / {len(intersections)}")
        print(f"  RMSE: {rmse:.6f}")
    else:
        print(f"  Warning: Not enough intersections ({len(intersections)}) for ghost circle fitting")
    
    # Estimate gear center
    print("\n--- Gear Center Estimation ---")
    
    # Method 1: Outer tips
    center_outer, radius_outer = estimate_gear_center_from_outer_tips(filtered_points)
    print(f"Method 1 (outer tips): center=({center_outer[0]:.4f}, {center_outer[1]:.4f}), radius={radius_outer:.4f}")
    
    # Method 2: Boundary centroid
    center_boundary, radius_boundary = estimate_gear_center_from_boundary_centroid(slice_points)
    print(f"Method 2 (boundary centroid): center=({center_boundary[0]:.4f}, {center_boundary[1]:.4f}), radius={radius_boundary:.4f}")
    
    # Choose method (can be made configurable via CLI)
    USE_OUTER_TIPS = True
    gear_center = GearCenter(
        center=center_outer if USE_OUTER_TIPS else center_boundary,
        method='outer_tips' if USE_OUTER_TIPS else 'boundary_centroid',
        radius=radius_outer if USE_OUTER_TIPS else radius_boundary
    )
    print(f"Using method: {gear_center.method}")
    
    # Compute offset analysis
    offset_analysis = None
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
        
        print(f"\n--- Offset Analysis (X-axis Setup Error Proxy) ---")
        print(f"  Offset magnitude: {magnitude:.6f}")
        print(f"  Offset angle: {angle_deg:.2f}° from +X axis")
        print(f"  Offset vector: ({offset_vector[0]:.6f}, {offset_vector[1]:.6f})")
    
    return AnalysisResult(
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
def save_results(result: AnalysisResult, output_path: Path) -> None:
    """Save analysis results to JSON file.
    
    Output includes:
    - Configuration parameters
    - Slice statistics
    - Flank line parameters (point, direction, cluster size)
    - Bisector parameters (origin, direction, length)
    - Ghost circle parameters (if available)
    - Gear center estimation (if available)
    - Offset analysis (if available)
    """
    data = {
        "configuration": {
            "slice_z": SLICE_Z,
            "inner_radius": R_INNER,
            "outer_radius": R_OUTER,
            "n_teeth": N_TEETH,
        },
        "slice_statistics": {
            "center_shift": result.center_shift.tolist(),
            "slice_point_count": len(result.slice_points),
            "filtered_point_count": len(result.filtered_points),
        },
        "flank_lines": [
            {
                "tooth": f.tooth,
                "side": "right",
                "point": f.point.tolist(),
                "direction": f.direction.tolist(),
                "cluster_size": f.cluster_size,
            }
            for f in result.flanks
        ],
        "pair_bisectors": [
            {
                "between_teeth": list(b.between_teeth),
                "origin": b.origin.tolist(),
                "direction": b.direction.tolist(),
                "length": b.length,
            }
            for b in result.bisectors
        ],
    }
    
    # Add ghost circle data if available
    if result.ghost_circle is not None:
        gc = result.ghost_circle
        data["ghost_circle"] = {
            "center": gc.center.tolist(),
            "radius": float(gc.radius),
            "rmse": float(gc.rmse),
            "n_intersections": gc.n_intersections,
            "n_inliers": len(gc.inliers),
            "n_outliers": len(gc.outliers),
            "inlier_points": gc.inliers.tolist(),
            "outlier_points": gc.outliers.tolist(),
        }
    
    # Add gear center data if available
    if result.gear_center is not None:
        gc = result.gear_center
        data["gear_center"] = {
            "center": gc.center.tolist(),
            "method": gc.method,
            "radius": float(gc.radius) if gc.radius is not None else None,
        }
    
    # Add offset analysis if available
    if result.offset_analysis is not None:
        oa = result.offset_analysis
        data["offset_analysis"] = {
            "offset_vector": oa.offset_vector.tolist(),
            "magnitude": float(oa.magnitude),
            "angle_deg": float(oa.angle_deg),
            "ghost_center": oa.ghost_center.tolist(),
            "gear_center": oa.gear_center.tolist(),
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze crown gear geometry and compute ghost circle for setup error detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--stl', type=str, required=True,
        help='Path to STL mesh file'
    )
    parser.add_argument(
        '--z-tip', type=float, default=SLICE_Z,
        help=f'Z-coordinate for slice extraction (default: {SLICE_Z})'
    )
    parser.add_argument(
        '--r-inner', type=float, default=R_INNER,
        help=f'Inner radius for point filtering (default: {R_INNER})'
    )
    parser.add_argument(
        '--r-outer', type=float, default=R_OUTER,
        help=f'Outer radius for point filtering (default: {R_OUTER})'
    )
    parser.add_argument(
        '--n-teeth', type=int, default=N_TEETH,
        help=f'Number of teeth in gear (default: {N_TEETH})'
    )
    parser.add_argument(
        '--axis', type=str, default='z', choices=['x', 'y', 'z'],
        help='Slice axis (default: z)'
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
    
    return parser.parse_args()


def run_analysis_from_cli() -> None:
    """Run analysis from command line interface."""
    args = parse_arguments()
    
    # Update global configuration from CLI
    global MESH_PATH, SLICE_Z, R_INNER, R_OUTER, N_TEETH
    MESH_PATH = Path(args.stl)
    SLICE_Z = args.z_tip * args.units_scale
    R_INNER = args.r_inner * args.units_scale
    R_OUTER = args.r_outer * args.units_scale
    N_TEETH = args.n_teeth
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"Crown Gear Ghost Circle Analysis")
    print(f"{'='*70}")
    print(f"Input STL: {MESH_PATH}")
    print(f"Output directory: {outdir.resolve()}")
    print(f"Configuration: Z={SLICE_Z}, R_inner={R_INNER}, R_outer={R_OUTER}, N_teeth={N_TEETH}")
    print(f"{'='*70}\n")
    
    # Run analysis
    result = analyse_gear_slice()
    
    # Save JSON report
    json_path = outdir / "ghost_circle_report.json"
    save_results(result, json_path)
    print(f"\n{'='*70}")
    print(f"Saved JSON report to: {json_path.resolve()}")
    
    # Save 2D plot
    plot_path = outdir / "gear_analysis_2d.png"
    plot_2d_analysis(result, plot_path)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Analysis Summary:")
    print(f"  Flanks fitted: {len(result.flanks)}")
    print(f"  Bisectors computed: {len(result.bisectors)}")
    if result.ghost_circle:
        print(f"  Ghost circle: center=({result.ghost_circle.center[0]:.4f}, {result.ghost_circle.center[1]:.4f}), r={result.ghost_circle.radius:.4f}")
    if result.offset_analysis:
        print(f"  Setup error (offset): {result.offset_analysis.magnitude:.6f} @ {result.offset_analysis.angle_deg:.1f}°")
    print(f"{'='*70}\n")
    
    # 3D Visualization
    if not args.no_viz:
        print("Launching 3D viewer...")
        geometries = build_visual_geometries(result)
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Gear Tooth Flank Analysis",
            width=1600,
            height=1000,
            mesh_show_back_face=True,
        )


# ==================== Main ====================
def main() -> None:
    """Main entry point: run analysis, save results, and visualize."""
    result = analyse_gear_slice()
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = output_dir / "ghost_circle_report.json"
    save_results(result, json_path)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {json_path.resolve()}")
    print(f"Summary: {len(result.flanks)} flanks, {len(result.bisectors)} bisectors")
    
    # Display tooth 1 information (reference tooth)
    tooth1 = next((f for f in result.flanks if f.tooth == 1), None)
    if tooth1:
        angle_deg = np.degrees(np.arctan2(tooth1.point[1], tooth1.point[0]))
        print(f"\nTooth 1 (reference, marked in red):")
        print(f"  Position: ({tooth1.point[0]:.3f}, {tooth1.point[1]:.3f})")
        print(f"  Angle: {angle_deg:.1f}° from +X axis")
        print(f"  Cluster size: {tooth1.cluster_size} points")
    
    # Display ghost circle and offset info
    if result.ghost_circle:
        print(f"\nGhost Circle:")
        print(f"  Center: ({result.ghost_circle.center[0]:.4f}, {result.ghost_circle.center[1]:.4f})")
        print(f"  Radius: {result.ghost_circle.radius:.4f}")
        print(f"  RMSE: {result.ghost_circle.rmse:.6f}")
    
    if result.offset_analysis:
        print(f"\nSetup Error (Offset):")
        print(f"  Magnitude: {result.offset_analysis.magnitude:.6f}")
        print(f"  Angle: {result.offset_analysis.angle_deg:.1f}° from +X axis")
    
    print(f"{'='*60}\n")
    
    # Save 2D plot
    plot_path = output_dir / "gear_analysis_2d.png"
    plot_2d_analysis(result, plot_path)
    
    # Launch 3D visualization
    print("Launching 3D viewer...")
    geometries = build_visual_geometries(result)
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Gear Tooth Flank Analysis",
        width=1600,
        height=1000,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    # Check if running from command line with arguments
    import sys
    if len(sys.argv) > 1:
        run_analysis_from_cli()
    else:
        main()