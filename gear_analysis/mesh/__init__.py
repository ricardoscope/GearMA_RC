"""
Mesh processing subpackage.

This subpackage handles loading, cleaning, converting, and slicing 3D meshes.

Key classes:
- MeshLoader: Load and clean STL files
- SliceExtractor: Extract 2D cross-sections with optional surface normals
- SliceData: Container for slice points and normals
"""

from gear_analysis.mesh.loading import MeshLoader
from gear_analysis.mesh.slicing import SliceExtractor, SliceData

__all__ = ["MeshLoader", "SliceExtractor", "SliceData"]
