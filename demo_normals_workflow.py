"""
Demonstration: Complete workflow with surface normal integration.

This script shows how to:
1. Load a mesh and slice it while preserving normals
2. Filter points by radius (keeping normals together)
3. Cluster points into teeth
4. Fit flanks using both SVD and normal-based methods
5. Validate and flag inconsistent flanks

Run this on your local machine with your gear STL file.
"""

from pathlib import Path
import numpy as np
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_workflow():
    """Demonstrate the complete workflow with normals."""
    
    print("=" * 70)
    print("GEAR ANALYSIS WITH SURFACE NORMALS")
    print("=" * 70)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    # Update this path to your STL file
    mesh_path = Path(r"C:\Users\alibi\Documents\Gears Examples\SimRes\DOE_207_AG.stl")
    
    # Analysis parameters (from your run_analysis.py)
    slice_z = -0.2
    r_inner = 2.46
    r_outer = 2.68
    n_teeth = 19
    max_step = 0.002  # slice_interpolation_density
    min_points_per_flank = 5
    
    print(f"\nConfiguration:")
    print(f"  Mesh: {mesh_path.name}")
    print(f"  Slice Z: {slice_z}")
    print(f"  Radius range: {r_inner} - {r_outer}")
    print(f"  Expected teeth: {n_teeth}")
    
    # =========================================================================
    # STEP 1: Load and slice mesh WITH NORMALS
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: Loading mesh and extracting slice with normals")
    print("-" * 70)
    
    import open3d as o3d
    import trimesh
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    
    # Convert to trimesh
    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False
    )
    
    print(f"  Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    
    # Import our new slicing module
    from gear_analysis.mesh.slicing_with_normals import SliceExtractorWithNormals, SliceData
    
    # Extract slice with normals
    slice_data = SliceExtractorWithNormals.extract(tm, slice_z, max_step)
    
    print(f"  Extracted slice: {len(slice_data)} points with normals")
    print(f"  Points shape: {slice_data.points.shape}")
    print(f"  Normals shape: {slice_data.normals.shape}")
    
    # =========================================================================
    # STEP 2: Filter by radius (keeping normals)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Filtering by radius")
    print("-" * 70)
    
    # Use the built-in filter method
    filtered_data = slice_data.filter_by_radius(r_inner, r_outer)
    
    print(f"  Before filtering: {len(slice_data)} points")
    print(f"  After filtering: {len(filtered_data)} points")
    print(f"  Points shape: {filtered_data.points.shape}")
    print(f"  Normals shape: {filtered_data.normals.shape}")
    
    # =========================================================================
    # STEP 3: Recenter the data
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: Recentering")
    print("-" * 70)
    
    center_xy = slice_data.points.mean(axis=0)
    print(f"  Center offset: ({center_xy[0]:.4f}, {center_xy[1]:.4f})")
    
    # Apply centering (normals don't need to be translated)
    filtered_points = filtered_data.points - center_xy
    filtered_normals = filtered_data.normals  # Normals are directions, no translation
    
    print(f"  Recentered: new centroid at origin")
    
    # =========================================================================
    # STEP 4: Cluster into teeth
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: Clustering into teeth")
    print("-" * 70)
    
    # Use angular binning to cluster points into teeth
    from gear_analysis.geometry.clustering import ToothClusterer
    
    # We need to also cluster the normals along with points
    # ToothClusterer returns indices, so we can use them for normals too
    point_angles = np.arctan2(filtered_points[:, 1], filtered_points[:, 0])
    
    # Define angular bins for each tooth
    tooth_angle_width = 2 * np.pi / n_teeth
    tooth_indices = []
    
    for i in range(n_teeth):
        # Center angle for this tooth
        center_angle = i * tooth_angle_width - np.pi + tooth_angle_width / 2
        
        # Normalize angles to [-π, π]
        angle_diff = point_angles - center_angle
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        # Points within half-width of center
        mask = np.abs(angle_diff) < tooth_angle_width / 2
        tooth_indices.append(np.where(mask)[0])
    
    print(f"  Clustered into {n_teeth} teeth")
    for i, indices in enumerate(tooth_indices[:3]):
        print(f"    Tooth {i+1}: {len(indices)} points")
    print(f"    ...")
    
    # =========================================================================
    # STEP 5: Fit flanks with normal validation
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: Fitting flanks with normal validation")
    print("-" * 70)
    
    from gear_analysis.geometry.line_fitting_with_normals import (
        LineFitterWithNormals,
        ToothFlanksWithNormals,
        summarize_flagged_teeth
    )
    
    tooth_flanks_list: list[ToothFlanksWithNormals] = []
    
    for tooth_num, indices in enumerate(tooth_indices, start=1):
        if len(indices) < 2 * min_points_per_flank:
            print(f"  Tooth {tooth_num}: Skipped (only {len(indices)} points)")
            continue
        
        # Get points and normals for this tooth
        tooth_points = filtered_points[indices]
        tooth_normals = filtered_normals[indices]
        
        # Fit flanks with validation
        result = LineFitterWithNormals.extract_both_flanks_with_normals(
            tooth_points,
            tooth_normals,
            min_points_per_flank,
            tooth_num
        )
        
        if result is not None:
            tooth_flanks_list.append(result)
            
            status = "✓" if result.is_valid and not result.is_flagged else "!"
            flag_str = " FLAGGED" if result.is_flagged else ""
            valid_str = " INVALID" if not result.is_valid else ""
            
            print(f"  [{status}] Tooth {tooth_num}: L={result.left_n_points} pts, "
                  f"R={result.right_n_points} pts{flag_str}{valid_str}")
    
    # =========================================================================
    # STEP 6: Summary of validation results
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 6: Validation Summary")
    print("-" * 70)
    
    summary = summarize_flagged_teeth(tooth_flanks_list)
    
    print(f"\n  Total teeth processed: {summary['total_teeth']}")
    print(f"  Valid teeth: {summary['total_teeth'] - summary['invalid_count']}")
    print(f"  Invalid teeth: {summary['invalid_count']}")
    print(f"  Flagged teeth: {summary['flagged_count']}")
    
    if summary['flagged_teeth']:
        print(f"\n  FLAGGED TEETH (SVD/Normal disagreement):")
        for ft in summary['flagged_teeth']:
            print(f"    Tooth {ft['tooth']}:")
            for reason in ft['reasons']:
                print(f"      - {reason}")
    
    if summary['invalid_teeth']:
        print(f"\n  INVALID TEETH (geometric issues):")
        for it in summary['invalid_teeth']:
            print(f"    Tooth {it['tooth']}: {it['issue']}")
    
    # =========================================================================
    # STEP 7: Detailed look at one tooth
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 7: Detailed example (first tooth)")
    print("-" * 70)
    
    if tooth_flanks_list:
        tooth = tooth_flanks_list[0]
        
        print(f"\n  Tooth {tooth.tooth}:")
        print(f"  ├── Left flank:")
        print(f"  │   ├── Points: {tooth.left_n_points}")
        print(f"  │   ├── Centroid: ({tooth.left_point[0]:.4f}, {tooth.left_point[1]:.4f})")
        print(f"  │   ├── SVD direction: ({tooth.left_direction[0]:.4f}, {tooth.left_direction[1]:.4f})")
        
        if tooth.left_validation:
            lv = tooth.left_validation
            print(f"  │   ├── Normal direction: ({lv.normal_direction[0]:.4f}, {lv.normal_direction[1]:.4f})")
            print(f"  │   ├── Angle difference: {lv.angle_difference_deg:.2f}°")
            print(f"  │   ├── Directions consistent: {lv.is_consistent}")
            print(f"  │   └── Side classification agrees: {lv.sides_agree}")
        
        print(f"  │")
        print(f"  └── Right flank:")
        print(f"      ├── Points: {tooth.right_n_points}")
        print(f"      ├── Centroid: ({tooth.right_point[0]:.4f}, {tooth.right_point[1]:.4f})")
        print(f"      ├── SVD direction: ({tooth.right_direction[0]:.4f}, {tooth.right_direction[1]:.4f})")
        
        if tooth.right_validation:
            rv = tooth.right_validation
            print(f"      ├── Normal direction: ({rv.normal_direction[0]:.4f}, {rv.normal_direction[1]:.4f})")
            print(f"      ├── Angle difference: {rv.angle_difference_deg:.2f}°")
            print(f"      ├── Directions consistent: {rv.is_consistent}")
            print(f"      └── Side classification agrees: {rv.sides_agree}")
    
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    
    return tooth_flanks_list


# =============================================================================
# SIMULATED EXAMPLE (runs without actual mesh file)
# =============================================================================

def simulated_example():
    """Run a simulated example to demonstrate the data structures."""
    
    print("=" * 70)
    print("SIMULATED EXAMPLE (no mesh file needed)")
    print("=" * 70)
    
    # Import the new modules
    from gear_analysis.geometry.line_fitting_with_normals import (
        LineFitterWithNormals,
        NormalBasedClassifier,
        FlankSide,
        summarize_flagged_teeth
    )
    
    np.random.seed(42)
    
    # =========================================================================
    # Create synthetic tooth data
    # =========================================================================
    print("\n1. Creating synthetic tooth data...")
    
    # Tooth at angle 0 (pointing along +X axis)
    # Left flank: points going from outer to inner, normal pointing "backward" (-Y)
    # Right flank: points going from outer to inner, normal pointing "forward" (+Y)
    
    n_points = 30
    tooth_angle = 0.0  # radians
    
    # Left flank (at slightly negative angle from tooth center)
    left_angles = np.linspace(-0.15, -0.05, n_points) + tooth_angle
    left_radii = np.linspace(2.65, 2.50, n_points)  # Outer to inner
    left_points = np.column_stack([
        left_radii * np.cos(left_angles),
        left_radii * np.sin(left_angles)
    ])
    # Add some noise
    left_points += 0.01 * np.random.randn(n_points, 2)
    
    # Left flank normals: should point in -Y direction (tangentially backward)
    left_normals = np.tile(np.array([0.3, -0.95]), (n_points, 1))
    left_normals += 0.1 * np.random.randn(n_points, 2)
    left_normals /= np.linalg.norm(left_normals, axis=1, keepdims=True)
    
    # Right flank (at slightly positive angle from tooth center)
    right_angles = np.linspace(0.05, 0.15, n_points) + tooth_angle
    right_radii = np.linspace(2.65, 2.50, n_points)
    right_points = np.column_stack([
        right_radii * np.cos(right_angles),
        right_radii * np.sin(right_angles)
    ])
    right_points += 0.01 * np.random.randn(n_points, 2)
    
    # Right flank normals: should point in +Y direction (tangentially forward)
    right_normals = np.tile(np.array([0.3, 0.95]), (n_points, 1))
    right_normals += 0.1 * np.random.randn(n_points, 2)
    right_normals /= np.linalg.norm(right_normals, axis=1, keepdims=True)
    
    # Combine into one tooth cluster
    tooth_points = np.vstack([left_points, right_points])
    tooth_normals = np.vstack([left_normals, right_normals])
    
    print(f"   Created tooth with {len(tooth_points)} points")
    print(f"   Points shape: {tooth_points.shape}")
    print(f"   Normals shape: {tooth_normals.shape}")
    
    # =========================================================================
    # Classify flanks using normals
    # =========================================================================
    print("\n2. Classifying flanks using surface normals...")
    
    left_class = NormalBasedClassifier.classify_flank_side(left_points, left_normals)
    right_class = NormalBasedClassifier.classify_flank_side(right_points, right_normals)
    
    print(f"   Left flank classification:")
    print(f"     Side: {left_class.side.value}")
    print(f"     Confidence: {left_class.confidence:.2%}")
    print(f"     Details: {left_class.details}")
    
    print(f"   Right flank classification:")
    print(f"     Side: {right_class.side.value}")
    print(f"     Confidence: {right_class.confidence:.2%}")
    print(f"     Details: {right_class.details}")
    
    # =========================================================================
    # Full flank extraction with validation
    # =========================================================================
    print("\n3. Extracting flanks with validation...")
    
    result = LineFitterWithNormals.extract_both_flanks_with_normals(
        tooth_points,
        tooth_normals,
        min_points=5,
        tooth_number=1
    )
    
    if result:
        print(f"\n   Tooth {result.tooth} results:")
        print(f"   ├── Valid: {result.is_valid}")
        print(f"   ├── Flagged: {result.is_flagged}")
        
        if result.left_validation:
            lv = result.left_validation
            print(f"   ├── Left flank validation:")
            print(f"   │   ├── SVD direction: ({result.left_direction[0]:.3f}, {result.left_direction[1]:.3f})")
            print(f"   │   ├── Normal direction: ({lv.normal_direction[0]:.3f}, {lv.normal_direction[1]:.3f})")
            print(f"   │   ├── Angle difference: {lv.angle_difference_deg:.1f}°")
            print(f"   │   ├── Consistent: {lv.is_consistent}")
            print(f"   │   └── Sides agree: {lv.sides_agree} (angular={lv.svd_side.value}, normal={lv.normal_side.value})")
        
        if result.right_validation:
            rv = result.right_validation
            print(f"   └── Right flank validation:")
            print(f"       ├── SVD direction: ({result.right_direction[0]:.3f}, {result.right_direction[1]:.3f})")
            print(f"       ├── Normal direction: ({rv.normal_direction[0]:.3f}, {rv.normal_direction[1]:.3f})")
            print(f"       ├── Angle difference: {rv.angle_difference_deg:.1f}°")
            print(f"       ├── Consistent: {rv.is_consistent}")
            print(f"       └── Sides agree: {rv.sides_agree} (angular={rv.svd_side.value}, normal={rv.normal_side.value})")
    
    # =========================================================================
    # Create an INCONSISTENT example
    # =========================================================================
    print("\n4. Creating intentionally inconsistent example...")
    
    # Swap the normals to create inconsistency
    bad_tooth_normals = np.vstack([right_normals, left_normals])  # Swapped!
    
    bad_result = LineFitterWithNormals.extract_both_flanks_with_normals(
        tooth_points,
        bad_tooth_normals,  # Wrong normals
        min_points=5,
        tooth_number=2
    )
    
    if bad_result:
        print(f"\n   Tooth {bad_result.tooth} (with wrong normals):")
        print(f"   ├── Valid: {bad_result.is_valid}")
        print(f"   ├── Flagged: {bad_result.is_flagged}")
        if bad_result.flag_reasons:
            print(f"   └── Flag reasons:")
            for reason in bad_result.flag_reasons:
                print(f"       - {reason}")
    
    print("\n" + "=" * 70)
    print("SIMULATED EXAMPLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    # First run simulated example (always works)
    simulated_example()
    
    # Then try real workflow if requested
    if "--real" in sys.argv:
        print("\n\n")
        try:
            demonstrate_workflow()
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("Update the mesh_path variable in demonstrate_workflow()")
        except ImportError as e:
            print(f"\nImport error: {e}")
            print("Make sure you're running from the project directory")
