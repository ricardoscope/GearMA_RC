"""
Quick script to check your mesh size.
Run this to see if you need mesh simplification.
"""

from pathlib import Path
import open3d as o3d

# Your mesh path from run_analysis.py
mesh_path = Path(r"C:\Users\alibi\Documents\Gears Examples\SimRes\DOE_207_AG.stl")

print("=" * 50)
print("MESH SIZE CHECK")
print("=" * 50)

if not mesh_path.exists():
    print(f"\n❌ File not found: {mesh_path}")
    print("\nUpdate the mesh_path variable to your STL location.")
else:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    
    n_vertices = len(mesh.vertices)
    n_triangles = len(mesh.triangles)
    
    print(f"\nFile: {mesh_path.name}")
    print(f"Vertices:  {n_vertices:,}")
    print(f"Triangles: {n_triangles:,}")
    
    print("\n" + "-" * 50)
    print("RECOMMENDATION:")
    print("-" * 50)
    
    if n_triangles < 100_000:
        print(f"✅ Small mesh ({n_triangles:,} triangles)")
        print("   No simplification needed.")
        print("   target_triangles = 100_000_000  (disabled)")
        
    elif n_triangles < 500_000:
        print(f"✅ Medium mesh ({n_triangles:,} triangles)")
        print("   Simplification optional.")
        print("   target_triangles = 1_000_000")
        
    elif n_triangles < 2_000_000:
        print(f"⚠️  Large mesh ({n_triangles:,} triangles)")
        print("   Consider simplification for faster processing.")
        print("   target_triangles = 500_000")
        
    else:
        print(f"🔴 Very large mesh ({n_triangles:,} triangles)")
        print("   Simplification recommended!")
        print("   target_triangles = 500_000")