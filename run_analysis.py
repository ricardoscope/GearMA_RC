"""
Run script for gear analysis.
Place this file in the same directory as the gear_analysis package.

Now supports:
- Two flank detection methods: 'radial' (recommended) and 'angular'
- Surface normal extraction for enhanced validation
- Automatic flagging of inconsistent flanks
"""
from pathlib import Path
from gear_analysis import AnalysisConfig, GearAnalyzer, ResultExporter
from gear_analysis.visualization import plot_2d_analysis, show_3d_visualization

# Try to import diagnostics
try:
    from diagnostics import run_full_diagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    print("Note: diagnostics.py not found - diagnostic plots will be skipped")
    DIAGNOSTICS_AVAILABLE = False


def main():
    # ========== CONFIGURE YOUR ANALYSIS HERE ==========
    config = AnalysisConfig(
        # Input file - UPDATE THIS PATH
        mesh_path=Path(r"C:\Users\alibi\Documents\Gears Examples\SimRes\DOE_194_AG.stl"),
        
        # Output directory
        output_dir=Path("results"),
        
        # Slice parameters
        slice_z=-0.2,
        
        # Radius filtering (adjust based on your gear)
        r_inner=2.46,
        r_outer=2.68,
        
        # Gear parameters
        n_teeth=19,
        
        # Optional: mesh simplification
        target_triangles=100_000_000,
        
        # NEW: Enable surface normal validation
        use_surface_normals=True,
    )
    
    # ========== CHOOSE FLANK DETECTION METHOD ==========
    # 'radial' = New method (recommended) - traces from tooth tips inward
    # 'angular' = Old method - divides into angular sectors
    # NOTE: Use 'angular' for full normal validation support
    FLANK_METHOD = 'angular'
    # ===================================================
    
    print(f"Analyzing: {config.mesh_path}")
    print(f"Output to: {config.output_dir}")
    print(f"Flank detection method: {FLANK_METHOD}")
    print(f"Surface normal validation: {'ENABLED' if config.use_surface_normals else 'DISABLED'}")
    
    # Run analysis with chosen method
    analyzer = GearAnalyzer(config, method=FLANK_METHOD)
    result = analyzer.run()
    
    # Save results
    config.output_dir.mkdir(parents=True, exist_ok=True)
    ResultExporter.to_json(result, config.output_dir / "report.json")
    
    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    # Count flanks
    if hasattr(result, 'tooth_flanks') and result.tooth_flanks:
        n_teeth_fitted = len(result.tooth_flanks)
        valid_teeth = sum(1 for tf in result.tooth_flanks if tf.is_valid)
        flagged_teeth = sum(1 for tf in result.tooth_flanks if tf.is_flagged)
        total_flanks = n_teeth_fitted * 2
        print(f"Teeth detected: {n_teeth_fitted}")
        print(f"Valid teeth: {valid_teeth}")
        print(f"Total flanks: {total_flanks}")
        
        # Show invalid teeth
        invalid_flanks = [tf for tf in result.tooth_flanks 
                         if hasattr(tf, 'is_valid') and not tf.is_valid]
        if invalid_flanks:
            print(f"\n⚠️  Invalid teeth: {len(invalid_flanks)}")
            for tf in invalid_flanks[:5]:  # Show first 5
                issue = tf.issue if hasattr(tf, 'issue') else "Unknown"
                print(f"   Tooth {tf.tooth}: {issue}")
            if len(invalid_flanks) > 5:
                print(f"   ... and {len(invalid_flanks) - 5} more")
        
        # NEW: Show flagged teeth (SVD/normal disagreement)
        if flagged_teeth > 0:
            print(f"\n🔍 Flagged teeth (SVD/normal disagreement): {flagged_teeth}")
            flagged = [tf for tf in result.tooth_flanks if tf.is_flagged]
            for tf in flagged[:5]:
                print(f"   Tooth {tf.tooth}:")
                for reason in tf.flag_reasons:
                    print(f"      - {reason}")
            if len(flagged) > 5:
                print(f"   ... and {len(flagged) - 5} more")
    else:
        print(f"Flanks fitted: {len(result.flanks)}")
    
    # NEW: Validation summary
    if hasattr(result, 'validation_summary') and result.validation_summary:
        vs = result.validation_summary
        print(f"\n--- Validation Summary ---")
        print(f"  Surface normals used: Yes")
        print(f"  Flagged teeth: {vs.flagged_teeth}/{vs.total_teeth} ({vs.flagged_percentage:.1f}%)")
        if vs.left_angle_diff_mean is not None:
            print(f"  Avg SVD/Normal difference (left): {vs.left_angle_diff_mean:.1f}°")
        if vs.right_angle_diff_mean is not None:
            print(f"  Avg SVD/Normal difference (right): {vs.right_angle_diff_mean:.1f}°")
    elif hasattr(result, 'has_normals') and not result.has_normals:
        print(f"\n--- Validation Summary ---")
        print(f"  Surface normals used: No (validation disabled)")
    
    # Bisector info
    if hasattr(result, 'tooth_bisectors') and result.tooth_bisectors:
        print(f"\nTooth bisectors: {len(result.tooth_bisectors)}")
    
    # Ghost circle
    if result.ghost_circle:
        gc = result.ghost_circle
        print(f"\nGhost circle:")
        print(f"  Center: ({gc.center[0]:.4f}, {gc.center[1]:.4f})")
        print(f"  Radius: {gc.radius:.4f}")
        print(f"  RMSE: {gc.rmse:.6f}")
        print(f"  Inliers: {len(gc.inliers)} / {gc.n_intersections}")
    
    # Offset
    if result.offset_analysis:
        oa = result.offset_analysis
        print(f"\nOffset analysis:")
        print(f"  Magnitude: {oa.magnitude:.6f}")
        print(f"  Angle: {oa.angle_deg:.1f}°")
    
    # Generate 2D plot
    plot_2d_analysis(result, config.output_dir / "analysis_2d.png")
    print(f"\nSaved plot to: {config.output_dir / 'analysis_2d.png'}")
    
    # Run diagnostics
    if DIAGNOSTICS_AVAILABLE:
        print("\n" + "=" * 50)
        print("RUNNING DIAGNOSTICS...")
        print("=" * 50)
        
        try:
            diagnostics_dir = config.output_dir / "diagnostics"
            run_full_diagnostics(result, config, output_dir=diagnostics_dir)
            print(f"\nDiagnostic plots saved to: {diagnostics_dir}")
        except Exception as e:
            print(f"\n⚠️  Diagnostics failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Show 3D visualization (comment out if not needed)
    show_3d_visualization(result)


if __name__ == "__main__":
    main()
