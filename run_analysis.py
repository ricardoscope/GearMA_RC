"""
Run script for gear analysis.
Place this file in the same directory as the gear_analysis package.
"""
from pathlib import Path
from gear_analysis import AnalysisConfig, GearAnalyzer, ResultExporter
from gear_analysis.visualization import plot_2d_analysis, show_3d_visualization


def main():
    # ========== CONFIGURE YOUR ANALYSIS HERE ==========
    config = AnalysisConfig(
        # Input file - UPDATE THIS PATH
        mesh_path=Path(r"C:\Users\alibi\Documents\Gears Examples\2025_01_22_Kronenrad CT Messung_Ausgerichtet nach neuem Vorgehen.stl"),
        
        # Output directory
        output_dir=Path("results"),
        
        # Slice parameters
        slice_z=0.2,
        
        # Radius filtering (adjust based on your gear)
        r_inner=2.48,
        r_outer=2.64,
        
        # Gear parameters
        n_teeth=19,
        
        # Optional: mesh simplification
        target_triangles=1_000_000,
    )
    # ==================================================
    
    print(f"Analyzing: {config.mesh_path}")
    print(f"Output to: {config.output_dir}")
    
    # Run analysis
    analyzer = GearAnalyzer(config)
    result = analyzer.run()
    
    # Save results
    config.output_dir.mkdir(parents=True, exist_ok=True)
    ResultExporter.to_json(result, config.output_dir / "report.json")
    
    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    # Count flanks correctly: tooth_flanks contains BOTH left and right per tooth
    if hasattr(result, 'tooth_flanks') and result.tooth_flanks:
        n_teeth_fitted = len(result.tooth_flanks)
        total_flanks = n_teeth_fitted * 2  # Each tooth has left + right flank
        print(f"Teeth with both flanks fitted: {n_teeth_fitted}")
        print(f"Total flanks fitted: {total_flanks} ({n_teeth_fitted} teeth × 2)")
    else:
        # Fallback to old method (right flanks only)
        print(f"Flanks fitted (right only): {len(result.flanks)}")
    
    print(f"Bisectors: {len(result.bisectors)}")
    
    if result.ghost_circle:
        gc = result.ghost_circle
        print(f"Ghost circle center: ({gc.center[0]:.4f}, {gc.center[1]:.4f})")
        print(f"Ghost circle radius: {gc.radius:.4f}")
        print(f"RMSE: {gc.rmse:.6f}")
    
    if result.offset_analysis:
        oa = result.offset_analysis
        print(f"Setup error offset: {oa.magnitude:.6f}")
        print(f"Offset angle: {oa.angle_deg:.1f}°")
    
    # Generate 2D plot (now shows BOTH flanks)
    plot_2d_analysis(result, config.output_dir / "analysis_2d.png")
    print(f"\nSaved plot to: {config.output_dir / 'analysis_2d.png'}")
    
    # Show 3D visualization (with improved lighting)
    # Comment out if you don't want the popup
    show_3d_visualization(result)


if __name__ == "__main__":
    main()