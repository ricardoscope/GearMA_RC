"""
Mesh slicing module with surface normal preservation.

This module handles extracting 2D cross-sections from 3D meshes,
optionally preserving surface normal information for enhanced analysis.

The module provides two extraction modes:
1. Points-only (backward compatible): SliceExtractor.extract()
2. Points with normals (enhanced): SliceExtractor.extract_with_normals()

Both modes return data through the SliceData container, which can be
used with or without normals throughout the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import trimesh
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type aliases
Points2D = NDArray[np.floating]
Normals2D = NDArray[np.floating]


@dataclass
class SliceData:
    """Container for slice points and their associated surface normals.
    
    This is the primary data structure that flows through the pipeline.
    It keeps points and normals together, ensuring they stay aligned
    through all transformations (filtering, recentering, etc.).
    
    Attributes:
        points: 2D coordinates of slice points, shape (N, 2)
        normals: 2D surface normals at each point, shape (N, 2) or None
                 Normals are projected from 3D and unit-normalized.
    
    Design rationale:
        - Normals are Optional to maintain backward compatibility
        - All transformation methods preserve the points-normals alignment
        - The class is immutable-ish (transformations return new instances)
    
    Example:
        >>> slice_data = SliceExtractor.extract_with_normals(tm, z=-0.2)
        >>> filtered = slice_data.filter_by_radius(2.46, 2.68)
        >>> centered = filtered.recenter()
        >>> print(f"Points: {centered.points.shape}, Normals: {centered.normals.shape}")
    """
    points: Points2D
    normals: Optional[Normals2D] = None
    
    def __post_init__(self):
        """Validate that points and normals have matching shapes."""
        if self.normals is not None:
            if len(self.points) != len(self.normals):
                raise ValueError(
                    f"Points and normals must have same length. "
                    f"Got {len(self.points)} points and {len(self.normals)} normals."
                )
    
    def __len__(self) -> int:
        return len(self.points)
    
    @property
    def has_normals(self) -> bool:
        """Check if normal data is available."""
        return self.normals is not None
    
    def filter_by_mask(self, mask: NDArray[np.bool_]) -> 'SliceData':
        """Return a new SliceData with only the points matching the mask.
        
        Args:
            mask: Boolean array of same length as points
            
        Returns:
            New SliceData with filtered points (and normals if present)
        """
        new_normals = self.normals[mask] if self.normals is not None else None
        return SliceData(points=self.points[mask].copy(), normals=new_normals)
    
    def filter_by_radius(
        self, 
        r_inner: float, 
        r_outer: float,
        relaxation_factor: float = 0.1
    ) -> 'SliceData':
        """Return a new SliceData with points within the radius range.
        
        Args:
            r_inner: Minimum radius (distance from origin)
            r_outer: Maximum radius
            relaxation_factor: If no points found, try relaxed bounds
            
        Returns:
            New SliceData with filtered points
            
        Raises:
            RuntimeError: If no points found even with relaxed bounds
        """
        radii = np.linalg.norm(self.points, axis=1)
        mask = (radii >= r_inner) & (radii <= r_outer)
        
        if not np.any(mask):
            # Try relaxed bounds
            relaxed_inner = r_inner * (1 - relaxation_factor)
            relaxed_outer = r_outer * (1 + relaxation_factor)
            mask = (radii >= relaxed_inner) & (radii <= relaxed_outer)
            
            if not np.any(mask):
                r_min, r_max = radii.min(), radii.max()
                raise RuntimeError(
                    f"No points found within radii {r_inner:.3f}-{r_outer:.3f}. "
                    f"Actual range: {r_min:.3f}-{r_max:.3f}"
                )
            
            logger.warning(f"Using relaxed radii: {relaxed_inner:.3f}-{relaxed_outer:.3f}")
        
        return self.filter_by_mask(mask)
    
    def recenter(self, center: Optional[NDArray] = None) -> Tuple['SliceData', NDArray]:
        """Recenter points around origin or specified center.
        
        Note: Normals are directional vectors and don't need translation.
        
        Args:
            center: Optional center point. If None, uses centroid of points.
            
        Returns:
            Tuple of (recentered SliceData, center offset that was applied)
        """
        if center is None:
            center = self.points.mean(axis=0)
        
        new_points = self.points - center
        # Normals are directions, not positions - they don't get translated
        new_normals = self.normals.copy() if self.normals is not None else None
        return SliceData(points=new_points, normals=new_normals), center
    
    def copy(self) -> 'SliceData':
        """Create a deep copy of this SliceData."""
        new_normals = self.normals.copy() if self.normals is not None else None
        return SliceData(points=self.points.copy(), normals=new_normals)
    
    def get_points_only(self) -> Points2D:
        """Get just the points array (for backward compatibility)."""
        return self.points
    
    def split_by_indices(self, indices: NDArray[np.integer]) -> 'SliceData':
        """Extract a subset of points by indices.
        
        Args:
            indices: Array of integer indices
            
        Returns:
            New SliceData with only the specified points
        """
        new_normals = self.normals[indices] if self.normals is not None else None
        return SliceData(points=self.points[indices].copy(), normals=new_normals)


class SliceExtractor:
    """Extracts and processes 2D slices from 3D meshes.
    
    This class provides methods for:
    - Extracting horizontal cross-sections from meshes
    - Preserving surface normals during extraction
    - Interpolating points along paths for uniform density
    
    The class supports two extraction modes:
    1. extract(): Returns SliceData with points only (faster, backward compatible)
    2. extract_with_normals(): Returns SliceData with points and normals
    
    Example:
        >>> tm = trimesh.load("gear.stl")
        >>> # Mode 1: Points only
        >>> slice_data = SliceExtractor.extract(tm, z=-0.2, max_step=0.002)
        >>> # Mode 2: With normals
        >>> slice_data = SliceExtractor.extract_with_normals(tm, z=-0.2, max_step=0.002)
    """
    
    @staticmethod
    def interpolate_path(
        path: Points2D,
        max_step: float,
        normals: Optional[Normals2D] = None
    ) -> Union[Points2D, Tuple[Points2D, Normals2D]]:
        """Add interpolated points to ensure uniform point density.
        
        For each segment longer than max_step, adds intermediate points
        using linear interpolation. If normals are provided, they are
        interpolated and renormalized.
        
        Args:
            path: Array of 2D points forming a path, shape (N, 2)
            max_step: Maximum allowed distance between consecutive points
            normals: Optional array of 2D normals, shape (N, 2)
            
        Returns:
            If normals is None: densified points array
            If normals provided: tuple of (points, normals)
        """
        if len(path) < 2:
            if normals is not None:
                return path, normals
            return path
        
        point_segments: list[Points2D] = []
        normal_segments: list[Normals2D] = [] if normals is not None else None
        
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            segment_length = np.linalg.norm(p1 - p0)
            
            if segment_length > max_step:
                num_points = int(np.ceil(segment_length / max_step)) + 1
                t = np.linspace(0, 1, num_points)
                
                # Interpolate positions
                seg_points = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
                point_segments.append(seg_points[:-1])  # Exclude endpoint
                
                # Interpolate normals if provided
                if normals is not None:
                    n0, n1 = normals[i], normals[i + 1]
                    seg_normals = n0[None, :] + t[:, None] * (n1 - n0)[None, :]
                    # Renormalize
                    norms = np.linalg.norm(seg_normals, axis=1, keepdims=True)
                    norms = np.where(norms > 1e-10, norms, 1.0)
                    seg_normals = seg_normals / norms
                    normal_segments.append(seg_normals[:-1])
            else:
                point_segments.append(p0[None, :])
                if normals is not None:
                    normal_segments.append(normals[i:i+1])
        
        # Add final point
        point_segments.append(path[-1:])
        if normals is not None:
            normal_segments.append(normals[-1:])
        
        result_points = np.vstack(point_segments)
        
        if normals is not None:
            result_normals = np.vstack(normal_segments)
            return result_points, result_normals
        
        return result_points
    
    @classmethod
    def extract(
        cls,
        tm: trimesh.Trimesh,
        z: float,
        max_step: float,
        plane_normal: Optional[NDArray] = None
    ) -> SliceData:
        """Extract a horizontal slice from the mesh (points only).
        
        This is the backward-compatible method that returns points without
        normal information. For enhanced analysis with normals, use
        extract_with_normals() instead.
        
        Args:
            tm: Trimesh object to slice
            z: Z-coordinate of slicing plane
            max_step: Maximum spacing between interpolated points
            plane_normal: Optional custom plane normal (default: [0, 0, 1])
            
        Returns:
            SliceData with points (normals will be None)
        """
        if plane_normal is None:
            plane_normal = [0, 0, 1]
        
        section = tm.section(
            plane_origin=[0, 0, z],
            plane_normal=plane_normal
        )
        
        if section is None or section.vertices.size == 0:
            raise RuntimeError(f"No intersection found at Z={z}")
        
        verts = np.asarray(section.vertices, dtype=float)
        xy = verts[:, :2]
        
        # Process path entities
        paths: list[Points2D] = []
        if hasattr(section, "entities") and len(section.entities) > 0:
            for ent in section.entities:
                if hasattr(ent, "points") and len(ent.points) >= 2:
                    pts = xy[np.asarray(ent.points, dtype=int)]
                    paths.append(cls.interpolate_path(pts, max_step))
        
        if not paths and len(xy) >= 2:
            paths.append(cls.interpolate_path(xy, max_step))
        
        if not paths:
            raise RuntimeError(f"No valid paths extracted from slice at Z={z}")
        
        result = np.vstack(paths)
        logger.debug(f"Extracted {len(result)} points from slice at Z={z}")
        
        return SliceData(points=result, normals=None)
    
    @classmethod
    def extract_with_normals(
        cls,
        tm: trimesh.Trimesh,
        z: float,
        max_step: float,
        plane_normal: Optional[NDArray] = None
    ) -> SliceData:
        """Extract a horizontal slice with surface normals.
        
        This enhanced method preserves the surface normal information
        from the original mesh, which enables:
        - More accurate flank side identification
        - Validation of SVD-based line fitting
        - Detection of problematic tooth geometries
        
        Args:
            tm: Trimesh object to slice
            z: Z-coordinate of slicing plane
            max_step: Maximum spacing between interpolated points
            plane_normal: Optional custom plane normal (default: [0, 0, 1])
            
        Returns:
            SliceData with both points and normals
        """
        if plane_normal is None:
            plane_normal = np.array([0, 0, 1])
        
        # Ensure face normals are computed
        if tm.face_normals is None or len(tm.face_normals) == 0:
            tm.fix_normals()
        
        # Get the section
        section = tm.section(
            plane_origin=[0, 0, z],
            plane_normal=plane_normal
        )
        
        if section is None or section.vertices.size == 0:
            raise RuntimeError(f"No intersection found at Z={z}")
        
        slice_verts = np.asarray(section.vertices, dtype=float)
        
        # Find faces crossing the slicing plane
        vertices = tm.vertices
        faces = tm.faces
        face_z_min = vertices[faces].min(axis=1)[:, 2]
        face_z_max = vertices[faces].max(axis=1)[:, 2]
        crossing_mask = (face_z_min <= z) & (face_z_max >= z)
        crossing_indices = np.where(crossing_mask)[0]
        
        if len(crossing_indices) == 0:
            logger.warning("No faces cross the slicing plane, normals unavailable")
            return cls.extract(tm, z, max_step, plane_normal)
        
        # Get centers and normals of crossing faces
        crossing_centers = tm.triangles_center[crossing_indices]
        crossing_normals = tm.face_normals[crossing_indices]
        
        # For each slice vertex, find nearest crossing face and get its normal
        slice_normals_3d = np.zeros_like(slice_verts)
        for i, vert in enumerate(slice_verts):
            xy_dist = np.linalg.norm(crossing_centers[:, :2] - vert[:2], axis=1)
            nearest_idx = np.argmin(xy_dist)
            slice_normals_3d[i] = crossing_normals[nearest_idx]
        
        # Project to 2D and normalize
        points_2d = slice_verts[:, :2]
        normals_2d = slice_normals_3d[:, :2].copy()
        norms = np.linalg.norm(normals_2d, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        normals_2d = normals_2d / norms
        
        # Process path entities with interpolation
        point_paths: list[Points2D] = []
        normal_paths: list[Normals2D] = []
        
        if hasattr(section, "entities") and len(section.entities) > 0:
            for ent in section.entities:
                if hasattr(ent, "points") and len(ent.points) >= 2:
                    indices = np.asarray(ent.points, dtype=int)
                    pts = points_2d[indices]
                    nrm = normals_2d[indices]
                    
                    pts_interp, nrm_interp = cls.interpolate_path(pts, max_step, nrm)
                    point_paths.append(pts_interp)
                    normal_paths.append(nrm_interp)
        
        if not point_paths and len(points_2d) >= 2:
            pts_interp, nrm_interp = cls.interpolate_path(points_2d, max_step, normals_2d)
            point_paths.append(pts_interp)
            normal_paths.append(nrm_interp)
        
        if not point_paths:
            raise RuntimeError(f"No valid paths extracted from slice at Z={z}")
        
        result_points = np.vstack(point_paths)
        result_normals = np.vstack(normal_paths)
        
        logger.debug(f"Extracted {len(result_points)} points with normals from slice at Z={z}")
        
        return SliceData(points=result_points, normals=result_normals)
    
    @staticmethod
    def find_slice_range(tm: trimesh.Trimesh) -> Tuple[float, float]:
        """Find the valid Z range for slicing.
        
        Args:
            tm: Trimesh object
            
        Returns:
            Tuple of (z_min, z_max) for the mesh bounds
        """
        bounds = tm.bounds
        return float(bounds[0, 2]), float(bounds[1, 2])
