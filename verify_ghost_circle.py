#!/usr/bin/env python3
"""
Verification script to compare ghost circle calculation between batch_analysis and run_analysis.
This will help identify any differences in configuration or calculation.
"""

from pathlib import Path
from gear_analysis.config import AnalysisConfig
from gear_analysis.analysis.pipeline import GearAnalyzer

def compare_configs():
    """Compare default config vs run_analysis config."""
    
    # Default config (as used in batch_analysis.py)
    default_config = AnalysisConfig(
        mesh_path=Path(r"C:\Users\alibi\Downloads\fixed\fixed\DOE_1_A.stl"),
        use_surface_normals=True,
    )
    
    # Config from run_analysis.py
    run_analysis_config = AnalysisConfig(
        mesh_path=Path(r"C:\Users\alibi\Downloads\fixed\fixed\DOE_1_A.stl"),
        slice_z=-0.2,
        r_inner=2.52,
        r_outer=2.68,
        n_teeth=19,
        use_surface_normals=True,
    )
    
    print("=" * 70)
    print("CONFIGURATION COMPARISON")
    print("=" * 70)
    print(f"\n{'Parameter':<30} {'Default (batch)':<20} {'run_analysis':<20}")
    print("-" * 70)
    
    params = [
        'slice_z', 'r_inner', 'r_outer', 'n_teeth', 
        'use_surface_normals', 'ransac_min_samples', 
        'ransac_residual_threshold', 'ransac_iterations',
        'ghost_radius_tolerance', 'intersection_r_min', 'intersection_r_max'
    ]
    
    for param in params:
        default_val = getattr(default_config, param, 'N/A')
        run_val = getattr(run_analysis_config, param, 'N/A')
        
        if default_val != run_val:
            marker = " ⚠️ DIFFERENT"
        else:
            marker = ""
        
        print(f"{param:<30} {str(default_val):<20} {str(run_val):<20}{marker}")
    
    print("\n" + "=" * 70)
    print("RUNNING ANALYSIS WITH BOTH CONFIGS")
    print("=" * 70)
    
    # Run with default config (batch_analysis style)
    print("\n1. Running with DEFAULT config (batch_analysis.py style)...")
    try:
        analyzer1 = GearAnalyzer(default_config, method='angular')
        result1 = analyzer1.run()
        
        if result1.ghost_circle:
            print(f"   ✓ Ghost circle radius: {result1.ghost_circle.radius:.6f}")
            print(f"   ✓ Center: ({result1.ghost_circle.center[0]:.6f}, {result1.ghost_circle.center[1]:.6f})")
            print(f"   ✓ RMSE: {result1.ghost_circle.rmse:.6f}")
            print(f"   ✓ Inliers: {len(result1.ghost_circle.inliers)} / {result1.ghost_circle.n_intersections}")
        else:
            print("   ✗ Ghost circle not computed")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Run with run_analysis config
    print("\n2. Running with run_analysis.py config...")
    try:
        analyzer2 = GearAnalyzer(run_analysis_config, method='angular')
        result2 = analyzer2.run()
        
        if result2.ghost_circle:
            print(f"   ✓ Ghost circle radius: {result2.ghost_circle.radius:.6f}")
            print(f"   ✓ Center: ({result2.ghost_circle.center[0]:.6f}, {result2.ghost_circle.center[1]:.6f})")
            print(f"   ✓ RMSE: {result2.ghost_circle.rmse:.6f}")
            print(f"   ✓ Inliers: {len(result2.ghost_circle.inliers)} / {result2.ghost_circle.n_intersections}")
        else:
            print("   ✗ Ghost circle not computed")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    if result1.ghost_circle and result2.ghost_circle:
        radius_diff = abs(result1.ghost_circle.radius - result2.ghost_circle.radius)
        print(f"\nRadius difference: {radius_diff:.6f}")
        if radius_diff > 0.0001:
            print("⚠️  WARNING: Radii differ significantly!")
        else:
            print("✓ Radii match (within tolerance)")
    else:
        print("\n⚠️  Cannot compare - one or both ghost circles not computed")

if __name__ == "__main__":
    compare_configs()

