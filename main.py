import matplotlib
# Prefer QtAgg if available; otherwise keep default backend
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh as tm

# STL path (use raw string or forward slashes on Windows)
path = r"C:\Users\alibi\Documents\Gears Examples\2025_01_22_Kronenrad CT Messung_Ausgerichtet nach neuem Vorgehen.stl"

# Load STL; force Trimesh (not Scene)
mesh = tm.load(path, force='mesh')
print(type(mesh), mesh.vertices.shape, mesh.faces.shape)

# If it’s a Scene, merge parts
if isinstance(mesh, tm.Scene):
    mesh = tm.util.concatenate(mesh.dump())

# --- Keep the UI responsive: reduce faces drastically ---
face_count = len(mesh.faces)
target = min(50_000, max(10_000, int(face_count * 0.3)))  # aim 10–50k faces
if face_count > target:
    try:
        mesh = mesh.simplify_quadratic_decimation(target)  # target = final face count
        print(f"Simplified {face_count} → {len(mesh.faces)} faces")
    except Exception as e:
        print("Skipping simplification:", e)

faces = [mesh.vertices[f] for f in mesh.faces]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Fast draw settings: no edges, no AA, no zsort
poly = Poly3DCollection(
    faces,
    linewidths=0,
    edgecolors=None,
    antialiased=False
)
poly.set_facecolor((0.75, 0.78, 0.85, 0.95))
ax.add_collection3d(poly)

vmin, vmax = mesh.bounds
ax.set_xlim(vmin[0], vmax[0]); ax.set_ylim(vmin[1], vmax[1]); ax.set_zlim(vmin[2], vmax[2])
ax.set_box_aspect(vmax - vmin)
ax.set_axis_off()
plt.tight_layout()

# Non-blocking show so the event loop can breathe
plt.show(block=False)
plt.pause(0.5)
input("Press Enter to close...")