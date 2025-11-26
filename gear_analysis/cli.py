"""
Command-line interface module.

This module provides the command-line interface for running
gear analysis from the terminal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gear_analysis.config import AnalysisConfig
from gear_analysis.analysis.pipeline import GearAnalyzer
from gear_analysis.io.export import ResultExporter
from gear_analysis.visualization.plot_2d import plot_2d_analysis
from gear_analysis.visualization.view_3d import show_3d_visualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Analyze crown gear geometry and compute ghost circle for setup error detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --stl gear.stl
  %(prog)s --stl gear.stl --z-tip -0.2 --n-teeth 38
  %(prog)s --stl gear.stl --outdir ./results --no-viz
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--stl',
        type=str,
        required=True,
        help='Path to STL mesh file'
    )
    
    # Slice parameters
    parser.add_argument(
        '--z-tip',
        type=float,
        default=-0.2,
        help='Z-coordinate for slice extraction (default: -0.2)'
    )
    parser.add_argument(
        '--r-inner',
        type=float,
        default=2.38,
        help='Inner radius for point filtering (default: 2.38)'
    )
    parser.add_argument(
        '--r-outer',
        type=float,
        default=2.58,
        help='Outer radius for point filtering (default: 2.58)'
    )
    
    # Gear parameters
    parser.add_argument(
        '--n-teeth',
        type=int,
        default=38,
        help='Number of teeth in gear (default: 38)'
    )
    
    # Output options
    parser.add_argument(
        '--units-scale',
        type=float,
        default=1.0,
        help='Scale factor for units (default: 1.0)'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='results',
        help='Output directory for results (default: results/)'
    )
    
    # Visualization options
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip 3D visualization'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip 2D plot generation'
    )
    
    # Analysis options
    parser.add_argument(
        '--gear-center-method',
        type=str,
        default='outer_tips',
        choices=['outer_tips', 'boundary_centroid'],
        help='Method for gear center estimation (default: outer_tips)'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (debug) output'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-essential output'
    )
    
    return parser


def run_cli() -> int:
    """Run the CLI application.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Create configuration
    try:
        config = AnalysisConfig.from_cli_args(args)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Verify input file exists
    if not config.mesh_path.exists():
        logger.error(f"Input file not found: {config.mesh_path}")
        return 1
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print header
    print("=" * 70)
    print("Crown Gear Ghost Circle Analysis")
    print("=" * 70)
    print(f"Input STL: {config.mesh_path}")
    print(f"Output directory: {config.output_dir.resolve()}")
    print(f"Configuration: Z={config.slice_z}, R_inner={config.r_inner}, "
          f"R_outer={config.r_outer}, N_teeth={config.n_teeth}")
    print("=" * 70 + "\n")
    
    # Run analysis
    try:
        analyzer = GearAnalyzer(config)
        result = analyzer.run()
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1
    
    # Save results
    json_path = config.output_dir / "ghost_circle_report.json"
    ResultExporter.to_json(result, json_path)
    
    # Save CSV summary
    csv_path = config.output_dir / "summary.csv"
    ResultExporter.to_csv_summary(result, csv_path)
    
    # Generate 2D plot
    if not args.no_plot:
        plot_path = config.output_dir / "gear_analysis_2d.png"
        plot_2d_analysis(result, plot_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Analysis Summary:")
    print(f"  Flanks fitted: {len(result.flanks)}")
    print(f"  Bisectors computed: {len(result.bisectors)}")
    
    if result.ghost_circle:
        gc = result.ghost_circle
        print(f"  Ghost circle: center=({gc.center[0]:.4f}, {gc.center[1]:.4f}), "
              f"r={gc.radius:.4f}")
    
    if result.offset_analysis:
        oa = result.offset_analysis
        print(f"  Setup error (offset): {oa.magnitude:.6f} @ {oa.angle_deg:.1f}°")
    
    print("=" * 70)
    print(f"\nResults saved to: {config.output_dir.resolve()}")
    
    # Show 3D visualization
    if not args.no_viz:
        show_3d_visualization(result)
    
    return 0


def main() -> None:
    """Entry point for the CLI."""
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
