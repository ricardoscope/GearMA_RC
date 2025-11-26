# Gear Analysis

Crown gear geometry analysis for detecting manufacturing setup errors.

## Overview

This package analyzes crown gear geometry to detect manufacturing setup errors by computing the "ghost circle" formed by bisector intersections. The deviation of the ghost circle center from the estimated gear center serves as a proxy for X-axis setup errors.

## Features

- Load and process 3D STL mesh files
- Extract horizontal slices from gear meshes
- Detect and cluster tooth regions
- Fit flank lines to each tooth
- Compute angle bisectors between adjacent teeth
- Fit a "ghost circle" through bisector intersections using RANSAC
- Estimate gear center using multiple methods
- Compute offset analysis for setup error detection
- Generate 2D and 3D visualizations
- Export results to JSON and CSV

## Installation

```bash
pip install gear-analysis
```

Or for development:

```bash
git clone https://github.com/example/gear-analysis.git
cd gear-analysis
pip install -e ".[dev]"
```

## Quick Start

### Command Line

```bash
gear-analysis --stl gear.stl --z-tip -0.2 --n-teeth 38
```

### Python API

```python
from pathlib import Path
from gear_analysis import AnalysisConfig, GearAnalyzer, plot_2d_analysis

# Configure analysis
config = AnalysisConfig(
    mesh_path=Path("gear.stl"),
    slice_z=-0.2,
    r_inner=2.38,
    r_outer=2.58,
    n_teeth=38
)

# Run analysis
analyzer = GearAnalyzer(config)
result = analyzer.run()

# Print results
print(f"Flanks fitted: {len(result.flanks)}")
if result.ghost_circle:
    print(f"Ghost circle radius: {result.ghost_circle.radius:.4f}")
if result.offset_analysis:
    print(f"Setup error offset: {result.offset_analysis.magnitude:.6f}")

# Generate visualization
plot_2d_analysis(result, Path("analysis.png"))
```

## Package Structure

```
gear_analysis/
├── __init__.py          # Public API
├── __main__.py          # Module entry point
├── config.py            # Configuration dataclass
├── models.py            # Data structures
├── utils.py             # Utility functions
├── cli.py               # Command-line interface
├── mesh/                # Mesh processing
│   ├── loading.py       # STL loading and cleanup
│   └── slicing.py       # Horizontal slice extraction
├── geometry/            # Geometric algorithms
│   ├── filtering.py     # Point filtering
│   ├── clustering.py    # Tooth clustering
│   ├── line_fitting.py  # Line fitting (SVD)
│   ├── circle_fitting.py # Circle fitting (Kåsa, Taubin, RANSAC)
│   └── bisectors.py     # Bisector computation
├── analysis/            # Analysis pipeline
│   ├── pipeline.py      # Main GearAnalyzer class
│   └── center_estimation.py # Gear center methods
├── visualization/       # Visualization tools
│   ├── plot_2d.py       # Matplotlib 2D plots
│   └── view_3d.py       # Open3D 3D viewer
└── io/                  # Input/Output
    └── export.py        # JSON/CSV export
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mesh_path` | - | Path to STL file |
| `slice_z` | -0.2 | Z-coordinate for slice extraction |
| `r_inner` | 2.38 | Inner radius for filtering |
| `r_outer` | 2.58 | Outer radius for filtering |
| `n_teeth` | 38 | Expected number of teeth |
| `target_triangles` | 1,000,000 | Max triangles (simplification threshold) |
| `gear_center_method` | "outer_tips" | Center estimation method |

## Output Files

When running analysis, the following files are generated:

- `ghost_circle_report.json` - Complete analysis results
- `summary.csv` - Key metrics summary
- `gear_analysis_2d.png` - 2D visualization

## Algorithm Overview

1. **Mesh Loading**: Load STL, remove duplicates, simplify if needed
2. **Slice Extraction**: Create horizontal cross-section at specified Z
3. **Point Filtering**: Keep only points in annular region (r_inner to r_outer)
4. **Tooth Clustering**: Partition points by angular position into N_TEETH bins
5. **Flank Fitting**: Use SVD to fit lines to right side of each tooth cluster
6. **Bisector Computation**: Calculate angle bisectors between tooth pairs
7. **Intersection Finding**: Find pairwise bisector intersections near center
8. **Ghost Circle Fitting**: Use RANSAC to robustly fit circle to intersections
9. **Center Estimation**: Estimate gear center from outer boundary
10. **Offset Analysis**: Compute offset between ghost circle and gear center

## License

MIT License
