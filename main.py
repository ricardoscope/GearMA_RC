import open3d as o3d
import numpy as np
import os

# ---------- Settings ----------
path = r"C:\Users\alibi\Documents\Gears Examples\2025_01_22_Kronenrad CT Messung_Ausgerichtet nach neuem Vorgehen.stl"
target_triangles = 80_000  # simplify to this many faces if mesh is denser
plane_z = 0.2
circle_radius = 2.49
circle_segments = 128
trim_keep_inside = False  # False = keep outside, True = keep inside

# ---------- Load ----------
if not os.path.exists(path):
    raise FileNotFoundError(f"File not found: {path}")

mesh = o3d.io.read_triangle_mesh(path)
if mesh.is_empty() or len(mesh.triangles) == 0:
    raise RuntimeError("Loaded mesh is empty or has no triangles.")

# ---------- Cleanup & normals ----------
mesh.remove_duplicated_vertices()
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_unreferenced_vertices()
mesh.remove_non_manifold_edges()
mesh.compute_vertex_normals()

# ---------- Optional decimation for speed ----------
if len(mesh.triangles) > target_triangles:
    mesh = mesh.simplify_quadric_decimation(target_triangles)

# ---------- Axis helper (scaled to bbox size) ----------
bbox = mesh.get_axis_aligned_bounding_box()
diag = float(np.max(bbox.get_extent())) if not mesh.is_empty() else 1.0
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1 * diag, origin=[0, 0, 0])

# ---------- Visual aid plane & circle (at z = plane_z) ----------
xy_extent = max(bbox.get_extent()[0], bbox.get_extent()[1]) if diag > 0 else 1.0
half = 0.6 * xy_extent

plane = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector([
        [-half, -half, plane_z],
        [ half, -half, plane_z],
        [ half,  half, plane_z],
        [-half,  half, plane_z],
    ]),
    triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
)
plane.compute_triangle_normals()
plane.paint_uniform_color([0.7, 1.0, 0.7])

theta = np.linspace(0, 2*np.pi, circle_segments, endpoint=False)
circle_pts = np.c_[circle_radius*np.cos(theta), circle_radius*np.sin(theta), np.full_like(theta, plane_z)]
circle = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(circle_pts),
    lines=o3d.utility.Vector2iVector([[i, (i+1) % circle_segments] for i in range(circle_segments)])
)
circle.colors = o3d.utility.Vector3dVector([[0.1, 0.1, 0.1]] * circle_segments)

# ---------- Intersection using Open3D (core simplification!) ----------
# Directly compute the mesh–plane intersection as a LineSet
# (origin on plane, normal along +Z)
plane_origin = np.array([0.0, 0.0, plane_z], dtype=float)
plane_normal = np.array([0.0, 0.0, 1.0], dtype=float)
section_ls = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)  # Open3D does the heavy lifting

# If nothing intersects, bail early
if section_ls is None or len(section_ls.lines) == 0:
    print("No intersection with the plane.")
    o3d.visualization.draw_geometries([mesh, axis, plane, circle])
    raise SystemExit

section_ls.paint_uniform_color([0.0, 0.3, 1.0])

# ---------- Trim section lines against a circle on the same plane ----------
def trim_line_set_by_circle(line_set: o3d.geometry.LineSet,
                            center_xyz: np.ndarray,
                            radius: float,
                            keep_inside: bool = False,
                            eps: float = 1e-12) -> o3d.geometry.LineSet | None:
    """
    Clip each 3D line segment (lying on z = center_xyz[2]) by a circle on that plane.
    keep_inside=False keeps only portions with r >= radius (outside the circle).
    keep_inside=True keeps only portions with r <= radius (inside the circle).
    """
    P = np.asarray(line_set.points)
    L = np.asarray(line_set.lines, dtype=int)

    Cxy = center_xyz[:2].astype(float)
    Z = center_xyz[2]

    out_pts = []
    out_lines = []

    def inside(p):
        # distance in XY to center
        return np.hypot(p[0]-Cxy[0], p[1]-Cxy[1]) <= radius + eps

    def clip_segment(p0, p1):
        """Return 0, 1, or 2 point segments after clipping against circle."""
        i0, i1 = inside(p0), inside(p1)

        # Parametric form in XY
        d = p1 - p0
        dxy = d[:2]
        a = np.dot(dxy, dxy)
        if a < eps:
            # Degenerate tiny segment: keep/discard as a point pair
            return [] if (keep_inside ^ i0) else [(p0, p1)]

        # Solve (p0_xy + t*dxy - Cxy)^2 = r^2
        f = (p0[:2] - Cxy)
        b = 2.0 * np.dot(f, dxy)
        c = np.dot(f, f) - radius**2
        disc = b*b - 4*a*c

        if disc < 0:
            # No intersection with circle boundary
            keep = (i0 and i1) if keep_inside else not (i0 and i1)
            return [(p0, p1)] if keep else []

        t_sqrt = np.sqrt(disc)
        t1 = (-b - t_sqrt) / (2*a)
        t2 = (-b + t_sqrt) / (2*a)

        ts = sorted([t1, t2])
        # classify segment intervals vs circle
        # segments are [0, min(1, t1)], [max(0, t1), min(1, t2)], [max(0, t2), 1]
        candidates = []
        intervals = [(0.0, ts[0]), (ts[0], ts[1]), (ts[1], 1.0)]
        for lo, hi in intervals:
            lo = max(0.0, lo)
            hi = min(1.0, hi)
            if hi - lo <= 1e-9:
                continue
            # midpoint test
            mid = 0.5*(lo+hi)
            pm = p0 + mid * d
            mid_inside = inside(pm)
            keep = mid_inside if keep_inside else (not mid_inside)
            if keep:
                candidates.append((p0 + lo*d, p0 + hi*d))
        return candidates

    for (i, j) in L:
        p0, p1 = P[i], P[j]
        # enforce plane z (robust to tiny noise)
        p0 = p0.copy(); p1 = p1.copy()
        p0[2] = Z; p1[2] = Z

        parts = clip_segment(p0, p1)
        for seg in parts:
            a, b = seg
            base = len(out_pts)
            out_pts.extend([a, b])
            out_lines.append([base, base+1])

    if not out_lines:
        return None

    trimmed = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(out_pts)),
        lines=o3d.utility.Vector2iVector(np.asarray(out_lines, dtype=int))
    )
    trimmed.paint_uniform_color([1.0, 0.0, 0.0])
    return trimmed

trimmed_ls = trim_line_set_by_circle(section_ls, np.array([0.0, 0.0, plane_z]), circle_radius, keep_inside=trim_keep_inside)

# ---------- Report ----------
n_lines = len(section_ls.lines)
print(f"Intersection line segments on z={plane_z}: {n_lines}")
if trimmed_ls is not None:
    print(f"After trimming by circle (keep_inside={trim_keep_inside}): {len(trimmed_ls.lines)} segments")

# ---------- View ----------
geoms = [mesh, axis, plane, circle, section_ls]
if trimmed_ls is not None:
    geoms.append(trimmed_ls)

o3d.visualization.draw_geometries(
    geoms,
    window_name="Open3D Mesh Viewer",
    width=1600,
    height=1000,
    left=50,
    top=50,
    point_show_normal=False,
    mesh_show_wireframe=False,
    mesh_show_back_face=True
)