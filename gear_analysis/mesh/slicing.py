"""
Mesh slicing module.

This module handles extracting 2D cross-sections from 3D meshes.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import trimesh
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type alias
Points2D = NDArray[np.floating]


class SliceExtractor:
    """Extracts and processes 2D slices from 3D meshes.
    
    This class provides methods for:
    - Extracting horizontal cross-sections from meshes
    - Interpolating points along paths for uniform density
    
    Example:
        >>> tm = trimesh.load("gear.stl")
        >>> points = SliceExtractor.extract(tm, z=-0.2, max_step=0.002)
        >>> points.shape
        (10000, 2)
    """
    
    @staticmethod
    def interpolate_path(path: Points2D, max_step: float) -> Points2D:
        """Add interpolated points to ensure uniform point density along a path.
        
        For each segment in the path longer than max_step, adds intermediate
        points using linear interpolation. This ensures consistent point
        density for accurate tooth detection.
        
        Args:
            path: Array of 2D points forming a path, shape (N, 2)
            max_step: Maximum allowed distance between consecutive points
            
        Returns:
            Densified path with interpolated points, shape (M, 2) where M >= N
        
        Example:
            >>> path = np.array([[0, 0], [10, 0]])  # 10 units apart
            >>> densified = SliceExtractor.interpolate_path(path, max_step=2.0)
            >>> len(densified)
            6  # Points at 0, 2, 4, 6, 8, 10
        """
        if len(path) < 2:
            return path
        
        segments: list[Points2D] = []
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            segment_length = np.linalg.norm(p1 - p0)
            
            if segment_length > max_step:
                # Calculate number of points needed for this segment
                num_points = int(np.ceil(segment_length / max_step)) + 1
                t = np.linspace(0, 1, num_points)
                # Linear interpolation between p0 and p1
                segment = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
                segments.append(segment[:-1])  # Exclude endpoint to avoid duplicates
            else:
                segments.append(p0[None, :])
        
        segments.append(path[-1:])  # Add final point
        return np.vstack(segments)
    
    @classmethod
    def extract(
        cls,
        tm: trimesh.Trimesh,
        z: float,
        max_step: float,
        plane_normal: Optional[NDArray] = None
    ) -> Points2D:
        """Extract a horizontal slice from the mesh and interpolate points.
        
        Creates a 2D cross-section by intersecting the mesh with a horizontal
        plane at the specified z-coordinate. The resulting contour is then
        densified to ensure uniform point spacing.
        
        Args:
            tm: Trimesh object to slice
            z: Z-coordinate of slicing plane
            max_step: Maximum spacing between interpolated points
            plane_normal: Optional custom plane normal (default: [0, 0, 1] for horizontal)
            
        Returns:
            Array of 2D points (N, 2) representing the slice contour
            
        Raises:
            RuntimeError: If no intersection found at specified Z
        
        Example:
            >>> tm = trimesh.load("gear.stl")
            >>> points = SliceExtractor.extract(tm, z=-0.2, max_step=0.002)
            >>> points.shape[1]
            2  # Always returns 2D points
        """
        if plane_normal is None:
            plane_normal = [0, 0, 1]
        
        # Slice mesh with plane
        section = tm.section(
            plane_origin=[0, 0, z],
            plane_normal=plane_normal
        )
        
        if section is None or section.vertices.size == 0:
            raise RuntimeError(f"No intersection found at Z={z}")
        
        # Extract 2D coordinates from slice
        verts = np.asarray(section.vertices, dtype=float)
        xy = verts[:, :2] if verts.shape[1] >= 2 else verts
        
        # Process connected path entities from the slice
        paths: list[Points2D] = []
        if hasattr(section, "entities") and len(section.entities) > 0:
            for ent in section.entities:
                if hasattr(ent, "points") and len(ent.points) >= 2:
                    pts = xy[np.asarray(ent.points, dtype=int)]
                    paths.append(cls.interpolate_path(pts, max_step))
        
        # Fallback: treat all points as a single path if no entities found
        if not paths and len(xy) >= 2:
            paths.append(cls.interpolate_path(xy, max_step))
        
        if not paths:
            raise RuntimeError(f"No valid paths extracted from slice at Z={z}")
        
        result = np.vstack(paths)
        logger.debug(f"Extracted {len(result)} points from slice at Z={z}")
        
        return result
    
    @staticmethod
    def find_slice_range(tm: trimesh.Trimesh) -> tuple[float, float]:
        """Find the valid Z range for slicing.
        
        Args:
            tm: Trimesh object
            
        Returns:
            Tuple of (z_min, z_max) for the mesh bounds
        
        Example:
            >>> tm = trimesh.load("gear.stl")
            >>> z_min, z_max = SliceExtractor.find_slice_range(tm)
            >>> print(f"Valid Z range: {z_min:.2f} to {z_max:.2f}")
        """
        bounds = tm.bounds
        return float(bounds[0, 2]), float(bounds[1, 2])
