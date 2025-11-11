import open3d as o3d
import numpy as np
import os

# ========== Settings ==========
path = r"C:\Users\alibi\Documents\Gears Examples\2025_01_22_Kronenrad CT Messung_Ausgerichtet nach neuem Vorgehen.stl"
target_triangles = 80_000
plane_z = 0.2
circle_radius = 2.49
circle_center = np.array([0.0, 0.0, plane_z])
tolerance = 1e-6

# ========== Load and Clean Mesh ==========
if not os.path.exists(path):
    raise FileNotFoundError(f"File not found: {path}")

mesh = o3d.io.read_triangle_mesh(path)
if mesh.is_empty() or len(mesh.triangles) == 0:
    raise RuntimeError("Loaded mesh is empty or has no triangles.")

mesh.remove_duplicated_vertices()
mesh.remove_degenerate_triangles()
mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()

if len(mesh.triangles) > target_triangles:
    mesh = mesh.simplify_quadric_decimation(target_triangles)

# ========== Helper Functions ==========
def create_axis(bbox):
    """Create coordinate axis with transformations."""
    diag = bbox.get_extent().max()
    axis_size = 0.1 * diag if diag > 0 else 1.0
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    
    # Combine transformations: flip Z and swap X/Y
    transform = np.eye(4)
    transform[2, 2] = -1.0  # Flip Z
    transform[0, 0] = 0.0   # Swap X/Y
    transform[1, 1] = 0.0
    transform[0, 1] = 1.0
    transform[1, 0] = 1.0
    axis.transform(transform)
    return axis

def create_plane(bbox, z=plane_z):
    """Create a plane mesh at z."""
    extent = bbox.get_extent()
    plane_half = 0.6 * max(extent[0], extent[1]) if max(extent[0], extent[1]) > 0 else 1.0
    
    vertices = np.array([
        [-plane_half, -plane_half, z],
        [ plane_half, -plane_half, z],
        [ plane_half,  plane_half, z],
        [-plane_half,  plane_half, z],
    ])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    
    plane = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles)
    )
    plane.compute_vertex_normals()
    plane.paint_uniform_color([0.7, 1.0, 0.7])
    return plane

def create_circle(radius, z=plane_z, num_segments=128):
    """Create a circle LineSet on the plane."""
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    points = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.full(num_segments, z)
    ])
    lines = np.column_stack([np.arange(num_segments), (np.arange(num_segments) + 1) % num_segments])
    
    circle = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    circle.paint_uniform_color([0.1, 0.1, 0.1])
    return circle

def extract_plane_intersection(mesh, plane_z, tolerance=1e-6):
    """Extract intersection contours between mesh and plane."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Collect intersection segments
    segments = []
    for tri in triangles:
        v = vertices[tri]
        z = v[:, 2]
        
        # Find edges crossing the plane
        edges = []
        for i in range(3):
            j = (i + 1) % 3
            if (z[i] <= plane_z + tolerance and z[j] >= plane_z - tolerance) or \
               (z[i] >= plane_z - tolerance and z[j] <= plane_z + tolerance):
                if abs(z[i] - z[j]) > tolerance:
                    t = (plane_z - z[i]) / (z[j] - z[i])
                    point = v[i] + t * (v[j] - v[i])
                    edges.append(point)
        
        if len(edges) == 2:
            segments.append((edges[0], edges[1]))
    
    if not segments:
        return None, []
    
    # Connect segments into contours
    def points_equal(p1, p2):
        return np.linalg.norm(p1 - p2) < tolerance
    
    used = set()
    contours = []
    
    for i, (p0, p1) in enumerate(segments):
        if i in used:
            continue
        
        contour = [p0.copy(), p1.copy()]
        used.add(i)
        
        # Extend contour forward and backward
        for direction in [1, -1]:
            while True:
                last_point = contour[-1] if direction == 1 else contour[0]
                found = False
                
                for j, (s0, s1) in enumerate(segments):
                    if j in used:
                        continue
                    
                    if points_equal(last_point, s0):
                        other = s1
                        used.add(j)
                        found = True
                    elif points_equal(last_point, s1):
                        other = s0
                        used.add(j)
                        found = True
                    else:
                        continue
                    
                    if direction == 1:
                        contour.append(other.copy())
                    else:
                        contour.insert(0, other.copy())
                    break
                
                if not found:
                    break
        
        contours.append(np.array(contour))
    
    # Convert to LineSet
    all_points = np.vstack(contours) if contours else np.empty((0, 3))
    all_lines = []
    idx = 0
    for contour in contours:
        for i in range(len(contour) - 1):
            all_lines.append([idx + i, idx + i + 1])
        idx += len(contour)
    
    if len(all_points) == 0:
        return None, []
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(all_lines)
    )
    return line_set, contours

def trim_contours_by_circle(contours, center, radius, keep_inside=False, tolerance=1e-6):
    """Trim contours by removing portions inside/outside the circle."""
    if not contours:
        return [], None
    
    center_xy = center[:2]
    
    def is_inside(p):
        return np.linalg.norm(p[:2] - center_xy) <= radius
    
    def circle_intersection(p0, p1):
        """Find intersection point between line segment and circle."""
        d = p1[:2] - p0[:2]
        diff = p0[:2] - center_xy
        a = np.dot(d, d)
        b = 2 * np.dot(diff, d)
        c = np.dot(diff, diff) - radius**2
        
        disc = b**2 - 4 * a * c
        if disc < 0 or a < 1e-10:
            return None
        
        sqrt_disc = np.sqrt(disc)
        t1, t2 = (-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)
        t = next((t for t in [t1, t2] if 0 <= t <= 1), None)
        return p0 + t * (p1 - p0) if t is not None else None
    
    trimmed_contours = []
    all_points = []
    all_lines = []
    idx = 0
    
    for contour in contours:
        if len(contour) < 2:
            continue
        
        trimmed = []
        for i in range(len(contour) - 1):
            p0, p1 = contour[i], contour[i + 1]
            inside0, inside1 = is_inside(p0), is_inside(p1)
            
            # Segment crosses circle boundary
            if inside0 != inside1:
                p_intersect = circle_intersection(p0, p1)
                if p_intersect is not None:
                    if keep_inside:
                        trimmed.extend([p0, p_intersect] if inside0 else [p_intersect, p1])
                    else:
                        trimmed.extend([p_intersect, p1] if inside0 else [p0, p_intersect])
            # Both points on same side
            elif (keep_inside and inside0 and inside1) or (not keep_inside and not (inside0 and inside1)):
                if not trimmed or not np.allclose(trimmed[-1], p0, atol=tolerance):
                    trimmed.append(p0)
                trimmed.append(p1)
        
        if len(trimmed) >= 2:
            trimmed = np.array(trimmed)
            trimmed_contours.append(trimmed)
            
            start_idx = idx
            all_points.extend(trimmed)
            for i in range(len(trimmed) - 1):
                all_lines.append([start_idx + i, start_idx + i + 1])
            idx += len(trimmed)
    
    if not all_points:
        return trimmed_contours, None
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(all_points)),
        lines=o3d.utility.Vector2iVector(all_lines)
    )
    line_set.paint_uniform_color([1.0, 0.0, 0.0])
    return trimmed_contours, line_set

# ========== Create Visualization Objects ==========
bbox = mesh.get_axis_aligned_bounding_box()
axis = create_axis(bbox)
plane = create_plane(bbox)
circle = create_circle(circle_radius)

# ========== Extract and Trim Contours ==========
intersection_lines, contours = extract_plane_intersection(mesh, plane_z, tolerance)
print(f"Extracted {len(contours)} contours from plane intersection")

trimmed_contours, trimmed_lines = trim_contours_by_circle(
    contours, circle_center, circle_radius, keep_inside=False, tolerance=tolerance
)

if trimmed_lines is not None:
    print(f"After trimming: {len(trimmed_contours)} contours remain")
    print(f"Total line segments: {len(trimmed_lines.lines)}")
    print("\nContour details:")
    for i, c in enumerate(trimmed_contours):
        print(f"  Contour {i+1}: {len(c)} points")

# ========== Interactive Visualization ==========
# Prepare geometries
geometries = [mesh, axis, plane, circle]

if intersection_lines is not None:
    intersection_lines.paint_uniform_color([0.0, 0.0, 1.0])
    geometries.append(intersection_lines)

if trimmed_lines is not None:
    geometries.append(trimmed_lines)


# Use built-in interactive visualizer
o3d.visualization.draw_geometries(
    [g for g in geometries if g is not None],
    window_name="Open3D Mesh Viewer - Interactive",
    width=1600,
    height=1000,
    mesh_show_back_face=True,
    mesh_show_wireframe=False
)