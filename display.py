from pathlib import Path

import numpy as np
import open3d as o3d

STL_PATH = Path(r"C:\Users\alibi\Documents\Gears Examples\SimRes\DOE_170_AG.stl")


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    if not path.exists():
        raise FileNotFoundError(f"Mesh not found: {path}")

    mesh = o3d.io.reacrown_gear_final_X_0gle_mesh(str(path))
    if mesh.is_empty():
        raise RuntimeError("Loaded mesh has no geometry.")

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def compute_radial_measurements(vertices: np.ndarray) -> dict[str, float]:
    xy = vertices[:, :2]
    center = xy.mean(axis=0)
    shifted = xy - center
    radii = np.linalg.norm(shifted, axis=1)

    stats = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "r_min": float(radii.min()),
        "r_max": float(radii.max()),
        "r_mean": float(radii.mean()),
        "r_median": float(np.median(radii)),
        "r_inner": float(np.percentile(radii, 15)),
        "r_outer": float(np.percentile(radii, 85)),
    }
    return stats


def display_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
    mesh_center = mesh.get_center()
    mesh.translate(-mesh_center)
    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries(
        [mesh],
        window_name="STL Viewer",
        width=1280,
        height=960,
        mesh_show_wireframe=False,
        mesh_show_back_face=True,
    )


def main() -> None:
    print("=" * 70)
    print("Simple STL Viewer")
    print("=" * 70)
    print(f"Input STL: {STL_PATH}")

    mesh = load_mesh(STL_PATH)
    vertices = np.asarray(mesh.vertices)
    print(f"Vertices: {len(vertices):,}")
    print(f"Triangles: {len(mesh.triangles):,}")

    measurements = compute_radial_measurements(vertices)
    print("\nRadial measurements (XY plane):")
    print(f"  Estimated center: ({measurements['center_x']:.3f}, {measurements['center_y']:.3f})")
    print(f"  Inner radius (15th percentile): {measurements['r_inner']:.3f}")
    print(f"  Outer radius (85th percentile): {measurements['r_outer']:.3f}")
    print(f"  Min radius: {measurements['r_min']:.3f}")
    print(f"  Max radius: {measurements['r_max']:.3f}")
    print(f"  Mean radius: {measurements['r_mean']:.3f}")
    print(f"  Median radius: {measurements['r_median']:.3f}")

    display_mesh(mesh)


if __name__ == "__main__":
    main()
