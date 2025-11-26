"""
Mesh loading and preprocessing module.

This module handles loading STL files, cleaning mesh geometry,
and converting between mesh formats.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

logger = logging.getLogger(__name__)


class MeshLoader:
    """Handles mesh loading, cleaning, and format conversion.
    
    This class provides static methods for:
    - Loading STL files with automatic cleanup
    - Mesh simplification for performance
    - Format conversion between Open3D and Trimesh
    
    Example:
        >>> mesh = MeshLoader.load(Path("gear.stl"), target_triangles=500000)
        >>> tm = MeshLoader.to_trimesh(mesh)
    """
    
    @staticmethod
    def load(path: Path, target_triangles: int) -> o3d.geometry.TriangleMesh:
        """Load, clean, and optionally simplify a mesh from file.
        
        Processing steps:
        1. Load STL file
        2. Remove duplicate vertices and degenerate triangles
        3. Remove unreferenced vertices
        4. Compute vertex normals
        5. Simplify to target_triangles if mesh is too large
        
        Args:
            path: Path to STL file
            target_triangles: Target triangle count for simplification.
                             If the mesh has more triangles, it will be
                             simplified using quadric decimation.
            
        Returns:
            Cleaned Open3D triangle mesh
            
        Raises:
            FileNotFoundError: If mesh file doesn't exist
            RuntimeError: If loaded mesh has no triangles
        
        Example:
            >>> mesh = MeshLoader.load(Path("gear.stl"), 1_000_000)
            >>> len(mesh.triangles)
            1000000
        """
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found: {path}")
        
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty() or len(mesh.triangles) == 0:
            raise RuntimeError("Loaded mesh has no triangles.")
        
        # Clean up mesh geometry
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        
        # Simplify if mesh is too complex
        num_triangles = len(mesh.triangles)
        if num_triangles > target_triangles:
            logger.info(f"Simplifying mesh: {num_triangles} → {target_triangles} triangles")
            mesh = mesh.simplify_quadric_decimation(target_triangles)
        
        logger.debug(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        
        return mesh
    
    @staticmethod
    def to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
        """Convert Open3D mesh to Trimesh format for slicing operations.
        
        Trimesh provides more robust slicing functionality for extracting
        2D cross-sections from 3D meshes.
        
        Args:
            mesh: Open3D triangle mesh
            
        Returns:
            Trimesh object with same geometry
        
        Example:
            >>> o3d_mesh = MeshLoader.load(Path("gear.stl"), 1_000_000)
            >>> tm = MeshLoader.to_trimesh(o3d_mesh)
            >>> isinstance(tm, trimesh.Trimesh)
            True
        """
        return trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
            process=False  # Skip automatic processing for speed
        )
    
    @staticmethod
    def get_bounding_box(mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
        """Get the axis-aligned bounding box of the mesh.
        
        Args:
            mesh: Open3D triangle mesh
            
        Returns:
            Tuple of (min_corner, max_corner) as numpy arrays
        
        Example:
            >>> mesh = MeshLoader.load(Path("gear.stl"), 1_000_000)
            >>> min_pt, max_pt = MeshLoader.get_bounding_box(mesh)
            >>> print(f"Z range: {min_pt[2]:.2f} to {max_pt[2]:.2f}")
        """
        bbox = mesh.get_axis_aligned_bounding_box()
        return np.asarray(bbox.min_bound), np.asarray(bbox.max_bound)
    
    @staticmethod
    def get_center(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        """Get the center of the mesh's bounding box.
        
        Args:
            mesh: Open3D triangle mesh
            
        Returns:
            3D center point as numpy array
        """
        min_pt, max_pt = MeshLoader.get_bounding_box(mesh)
        return (min_pt + max_pt) / 2
