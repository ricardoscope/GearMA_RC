from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

# ==================== Configuration ====================
MESH_PATH = Path(
    r"C:\Users\alibi\Documents\Gears Examples\2025_01_22_Kronenrad CT Messung_Ausgerichtet nach neuem Vorgehen.stl"
)
TARGET_TRIANGLES = 80_000
SLICE_Z = 0.2
R_INNER = 2.45
R_OUTER = 2.9
OUTPUT_JSON = Path("flank_lines.json")
SLICE_INTERPOLATION_DENSITY = 0.002
FLANK_SEGMENT_LENGTH = 1.0
BISECTOR_LENGTH = 5.0
N_TEETH = 38  # Expected number of teeth (set to your gear's tooth count)
MIN_POINTS_PER_CLUSTER = 6
MIN_POINTS_PER_FLANK = 5


# ==================== Data Structures ====================
@dataclass
class FlankLine:
    """Represents a fitted flank line for a single tooth."""
    tooth: int
    point: np.ndarray
    direction: np.ndarray
    cluster_size: int


@dataclass
class PairBisector:
    """Represents an angle bisector between two adjacent teeth."""
    between_teeth: tuple[int, int]
    origin: np.ndarray
    direction: np.ndarray
    length: float


@dataclass
class AnalysisResult:
    """Complete analysis results including mesh, points, flanks, and bisectors."""
    mesh: o3d.geometry.TriangleMesh
    slice_points: np.ndarray
    filtered_points: np.ndarray
    center_shift: np.ndarray
    flanks: list[FlankLine]
    bisectors: list[PairBisector]


# ==================== Utility Functions ====================
def unit_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def to_3d(point_2d: np.ndarray, z: float) -> np.ndarray:
    """Convert 2D point to 3D by adding Z coordinate."""
    return np.array([point_2d[0], point_2d[1], z])


# ==================== Mesh Processing ====================
def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Load and clean mesh from file."""
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError("Loaded mesh has no triangles.")
    
    # Clean mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    
    # Simplify if needed
    if len(mesh.triangles) > TARGET_TRIANGLES:
        mesh = mesh.simplify_quadric_decimation(TARGET_TRIANGLES)
    
    return mesh


def mesh_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """Convert Open3D mesh to Trimesh for slicing operations."""
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False
    )


# ==================== Slice Processing ====================
def interpolate_path(path: np.ndarray, max_step: float) -> np.ndarray:
    """Interpolate points along a path to achieve desired density."""
    if len(path) < 2:
        return path
    
    segments = []
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        segment_length = np.linalg.norm(p1 - p0)
        
        if segment_length > max_step:
            num_points = int(np.ceil(segment_length / max_step)) + 1
            t = np.linspace(0, 1, num_points)
            segment = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
            segments.append(segment[:-1])  # Exclude endpoint (will be added by next segment)
        else:
            segments.append(p0[None, :])
    
    segments.append(path[-1:])  # Add final point
    return np.vstack(segments)


def extract_slice(tm: trimesh.Trimesh, z: float, max_step: float) -> np.ndarray:
    """Extract and interpolate slice points at given Z height."""
    section = tm.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None or section.vertices.size == 0:
        raise RuntimeError(f"No intersection found at Z={z}")
    
    verts = np.asarray(section.vertices, dtype=float)
    xy = verts[:, :2] if verts.shape[1] >= 2 else verts
    
    # Extract paths from entities
    paths: list[np.ndarray] = []
    if hasattr(section, "entities") and len(section.entities) > 0:
        for ent in section.entities:
            if hasattr(ent, "points") and len(ent.points) >= 2:
                pts = xy[np.asarray(ent.points, dtype=int)]
                paths.append(interpolate_path(pts, max_step))
    
    # Fallback if no entities
    if not paths and len(xy) >= 2:
        paths.append(interpolate_path(xy, max_step))
    
    if not paths:
        raise RuntimeError(f"No valid paths extracted from slice at Z={z}")
    
    return np.vstack(paths)


def filter_by_radius(points: np.ndarray, r_inner: float, r_outer: float) -> np.ndarray:
    """Filter points within specified radial bounds."""
    radii = np.linalg.norm(points, axis=1)
    mask = (radii >= r_inner) & (radii <= r_outer)
    filtered = points[mask]
    
    if len(filtered) == 0:
        # Try relaxed bounds
        relaxed_inner, relaxed_outer = r_inner * 0.9, r_outer * 1.1
        mask = (radii >= relaxed_inner) & (radii <= relaxed_outer)
        filtered = points[mask]
        if len(filtered) == 0:
            raise RuntimeError(
                f"No points found within radii {r_inner:.2f}-{r_outer:.2f} "
                f"(relaxed: {relaxed_inner:.2f}-{relaxed_outer:.2f})"
            )
        print(f"Using relaxed radii: {relaxed_inner:.2f}-{relaxed_outer:.2f}")
    
    return filtered


# ==================== Tooth Clustering ====================
def partition_points_into_teeth_binned(points_xy: np.ndarray, n_teeth: int) -> list[np.ndarray]:
    """
    Split points into exactly n_teeth angular sectors.
    Guarantees exactly one cluster per tooth (if points exist).
    """
    if len(points_xy) == 0 or n_teeth < 1:
        return []
    
    angles = np.arctan2(points_xy[:, 1], points_xy[:, 0])
    # Map to [0, 2π)
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    
    # Bin edges
    edges = np.linspace(0.0, 2 * np.pi, n_teeth + 1)
    idx = np.digitize(angles, edges, right=False) - 1
    idx[idx == n_teeth] = 0  # Wrap last bin to 0
    
    clusters = []
    for k in range(n_teeth):
        pts = points_xy[idx == k]
        if len(pts) >= MIN_POINTS_PER_CLUSTER:
            clusters.append(pts)
        else:
            clusters.append(np.empty((0, 2)))  # Keep slot to preserve tooth index
    
    return clusters


# ==================== Line Fitting ====================
def fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a line to points using SVD (returns centroid and direction)."""
    if len(points) < 2:
        raise ValueError("Need at least two points to fit a line.")
    
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    direction = unit_vector(vt[0])
    return centroid, direction


def extract_right_flank(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract and fit right flank line from tooth cluster points."""
    center = points.mean(axis=0)
    radial = unit_vector(center) if np.linalg.norm(center) > 0 else np.array([1.0, 0.0])
    tangential = np.array([-radial[1], radial[0]])
    
    # Split points by tangential projection
    projections = (points - center) @ tangential
    median = np.median(projections)
    right = points[projections > median]
    
    # Fallback if insufficient points
    if len(right) < MIN_POINTS_PER_FLANK:
        order = np.argsort(projections)
        half = max(3, len(points) // 2)
        right = points[order[-half:]]
        if len(right) < 3:
            raise ValueError("Insufficient points for flank fitting.")
    
    return fit_line(right)


# ==================== Bisector Computation ====================
def compute_bisector(
    point_a: np.ndarray, dir_a: np.ndarray,
    point_b: np.ndarray, dir_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute angle bisector between two lines."""
    dir_a = unit_vector(dir_a)
    dir_b = unit_vector(dir_b)
    
    # Ensure directions point in similar direction
    if np.dot(dir_a, dir_b) < 0:
        dir_b = -dir_b
    
    bisector_dir = unit_vector(dir_a + dir_b)
    origin = 0.5 * (point_a + point_b)
    
    # Fallback if bisector is degenerate
    if np.linalg.norm(bisector_dir) < 1e-8:
        bisector_dir = unit_vector(origin) if np.linalg.norm(origin) > 0 else dir_a
    
    return origin, bisector_dir


def compute_pair_bisectors(flanks: list[FlankLine], length: float) -> list[PairBisector]:
    """Compute bisectors for odd-even tooth pairs: (1-2), (3-4), (5-6), etc."""
    if len(flanks) < 2:
        return []
    
    bisectors: list[PairBisector] = []
    for i in range(0, len(flanks) - 1, 2):
        current, nxt = flanks[i], flanks[i + 1]
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


# ==================== Visualization ====================
def make_lineset(p0: np.ndarray, p1: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    """Create a LineSet from two points."""
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([p0, p1]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls


def make_circle(radius: float, z: float, color: tuple[float, float, float], segments: int = 256) -> o3d.geometry.LineSet:
    """Create a circle LineSet at given Z height."""
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
    """Build all geometries for visualization."""
    geometries: list[o3d.geometry.Geometry] = [
        result.mesh,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]),
    ]
    
    # Slice point cloud
    slice_pc = o3d.geometry.PointCloud()
    slice_pc.points = o3d.utility.Vector3dVector(
        np.column_stack([result.filtered_points, np.full(len(result.filtered_points), SLICE_Z)])
    )
    slice_pc.paint_uniform_color([0.8, 0.8, 0.8])
    geometries.append(slice_pc)
    
    # Reference circles
    geometries.extend([
        make_circle(R_INNER, SLICE_Z, (0.9, 0.4, 0.1)),
        make_circle(R_OUTER, SLICE_Z, (0.1, 0.6, 0.9)),
    ])
    
    # Flank lines
    for flank in result.flanks:
        center_3d = to_3d(flank.point, SLICE_Z)
        direction_3d = np.append(flank.direction, 0.0)
        start = center_3d - FLANK_SEGMENT_LENGTH * direction_3d
        end = center_3d + FLANK_SEGMENT_LENGTH * direction_3d
        
        if flank.tooth == 1:
            # Tooth 1: red line + red sphere marker
            geometries.append(make_lineset(start, end, (1.0, 0.0, 0.0)))
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.translate(center_3d)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(sphere)
        else:
            geometries.append(make_lineset(start, end, (0.1, 0.8, 0.2)))  # Green
    
    # Bisectors
    for bisector in result.bisectors:
        origin_3d = to_3d(bisector.origin, SLICE_Z)
        direction_3d = np.append(bisector.direction, 0.0)
        start = origin_3d - 0.5 * bisector.length * direction_3d
        end = origin_3d + 0.5 * bisector.length * direction_3d
        geometries.append(make_lineset(start, end, (0.0, 0.0, 0.0)))  # Black
    
    return geometries


# ==================== Analysis Pipeline ====================
def analyse_gear_slice() -> AnalysisResult:
    """Main analysis pipeline: load mesh, extract slice, fit flanks, compute bisectors."""
    print("Loading mesh...")
    mesh = load_mesh(MESH_PATH)
    tm = mesh_to_trimesh(mesh)
    
    print(f"Extracting slice at Z={SLICE_Z}...")
    slice_points = extract_slice(tm, SLICE_Z, SLICE_INTERPOLATION_DENSITY)
    
    print(f"Filtering points between radii {R_INNER:.2f} and {R_OUTER:.2f}...")
    filtered_points = filter_by_radius(slice_points, R_INNER, R_OUTER)
    print(f"  Kept {len(filtered_points)} points (from {len(slice_points)} total)")
    
    # Recenter
    center_xy = slice_points[:, :2].mean(axis=0)
    print(f"Recentering by {center_xy}")
    slice_points[:, :2] -= center_xy
    filtered_points[:, :2] -= center_xy
    mesh.translate(np.array([-center_xy[0], -center_xy[1], 0.0]))
    
    # Cluster into teeth using binned approach (guarantees exactly N_TEETH clusters)
    print(f"Partitioning points into {N_TEETH} tooth clusters (binned)...")
    tooth_clusters = partition_points_into_teeth_binned(filtered_points, N_TEETH)
    
    # Count non-empty clusters
    non_empty = sum(1 for c in tooth_clusters if len(c) > 0)
    print(f"  Found {non_empty} non-empty clusters out of {N_TEETH} expected teeth")
    
    # Fit flanks (skip empty clusters but preserve tooth numbering)
    print("Fitting flank lines...")
    flanks: list[FlankLine] = []
    for idx, cluster in enumerate(tooth_clusters, start=1):
        if len(cluster) == 0:
            continue  # Skip empty clusters
        try:
            point, direction = extract_right_flank(cluster)
            flanks.append(FlankLine(
                tooth=idx,
                point=point,
                direction=unit_vector(direction),
                cluster_size=int(cluster.shape[0]),
            ))
        except ValueError:
            continue
    
    if not flanks:
        raise RuntimeError("Unable to fit flanks for any tooth.")
    print(f"  Fitted {len(flanks)} flank lines")
    
    # Compute bisectors
    bisectors = compute_pair_bisectors(flanks, BISECTOR_LENGTH)
    print(f"  Computed {len(bisectors)} pair bisectors")
    
    return AnalysisResult(
        mesh=mesh,
        slice_points=slice_points,
        filtered_points=filtered_points,
        center_shift=center_xy,
        flanks=flanks,
        bisectors=bisectors,
    )


# ==================== I/O ====================
def save_results(result: AnalysisResult, output_path: Path) -> None:
    """Save analysis results to JSON file."""
    data = {
        "slice_z": SLICE_Z,
        "inner_radius": R_INNER,
        "outer_radius": R_OUTER,
        "center_shift": result.center_shift.tolist(),
        "slice_point_count": int(result.slice_points.shape[0]),
        "filtered_point_count": int(result.filtered_points.shape[0]),
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
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


# ==================== Main ====================
def main() -> None:
    """Main entry point."""
    result = analyse_gear_slice()
    save_results(result, OUTPUT_JSON)
    
    print(f"\nResults saved to: {OUTPUT_JSON.resolve()}")
    print(f"Summary: {len(result.flanks)} flanks, {len(result.bisectors)} bisectors")
    
    # Identify tooth 1
    tooth1 = next((f for f in result.flanks if f.tooth == 1), None)
    if tooth1:
        angle_deg = np.degrees(np.arctan2(tooth1.point[1], tooth1.point[0]))
        print(f"\nTooth 1 (red marker):")
        print(f"  Position: ({tooth1.point[0]:.3f}, {tooth1.point[1]:.3f})")
        print(f"  Angle: {angle_deg:.1f}° from +X axis")
        print(f"  Cluster size: {tooth1.cluster_size} points")
    
    # Visualize
    geometries = build_visual_geometries(result)
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Tooth Flanks Analysis",
        width=1600,
        height=1000,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
