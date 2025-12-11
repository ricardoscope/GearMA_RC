"""
Enhanced mesh slicing module with surface normal preservation.

This module handles extracting 2D cross-sections from 3D meshes,
now including the surface normals at each slice point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import trimesh
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type aliases
Points2D = NDArray[np.floating]
Normals2D = NDArray[np.floating]


@dataclass
class SliceData:
    """Container for slice points and their associated normals.
    
    Attributes:
        points: 2D coordinates of slice points, shape (N, 2)
        normals: 2D surface normals at each point, shape (N, 2)
                 Normals are projected from 3D and normalized.
    
    Example:
        >>> slice_data = SliceExtractor.extract_with_normals(tm, z=-0.2)
        >>> print(f"Points: {slice_data.points.shape}")
        >>> print(f"Normals: {slice_data.normals.shape}")
    """
    points: Points2D
    normals: Normals2D
    
    def __len__(self) -> int:
        return len(self.points)
    
    def filter_by_mask(self, mask: NDArray[np.bool_]) -> 'SliceData':
        """Return a new SliceData with only the points matching the mask."""
        return SliceData(
            points=self.points[mask],
            normals=self.normals[mask]
        )
    
    def filter_by_radius(self, r_inner: float, r_outer: float) -> 'SliceData':
        """Return a new SliceData with points within the radius range."""
        radii = np.linalg.norm(self.points, axis=1)
        mask = (radii >= r_inner) & (radii <= r_outer)
        return self.filter_by_mask(mask)


class SliceExtractorWithNormals:
    """Extracts 2D slices from 3D meshes, preserving surface normals.
    
    This enhanced version tracks which mesh faces each slice segment
    comes from, allowing us to retrieve the surface normal at each point.
    
    Example:
        >>> tm = trimesh.load("gear.stl")
        >>> slice_data = SliceExtractorWithNormals.extract(tm, z=-0.2, max_step=0.002)
        >>> print(f"Got {len(slice_data)} points with normals")
    """
    
    @staticmethod
    def interpolate_path_with_normals(
        path: Points2D,
        normals: Normals2D,
        max_step: float
    ) -> Tuple[Points2D, Normals2D]:
        """Add interpolated points while preserving normal information.
        
        For interpolated points, the normal is linearly interpolated
        between the two endpoint normals and re-normalized.
        
        Args:
            path: Array of 2D points forming a path, shape (N, 2)
            normals: Array of 2D normals at each point, shape (N, 2)
            max_step: Maximum allowed distance between consecutive points
            
        Returns:
            Tuple of (densified_points, densified_normals)
        """
        if len(path) < 2:
            return path, normals
        
        point_segments: list[Points2D] = []
        normal_segments: list[Normals2D] = []
        
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            n0, n1 = normals[i], normals[i + 1]
            segment_length = np.linalg.norm(p1 - p0)
            
            if segment_length > max_step:
                # Calculate number of points needed for this segment
                num_points = int(np.ceil(segment_length / max_step)) + 1
                t = np.linspace(0, 1, num_points)
                
                # Linear interpolation for positions
                seg_points = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
                
                # Linear interpolation for normals (then renormalize)
                seg_normals = n0[None, :] + t[:, None] * (n1 - n0)[None, :]
                # Normalize each interpolated normal
                norms = np.linalg.norm(seg_normals, axis=1, keepdims=True)
                norms = np.where(norms > 1e-10, norms, 1.0)  # Avoid division by zero
                seg_normals = seg_normals / norms
                
                # Exclude endpoint to avoid duplicates
                point_segments.append(seg_points[:-1])
                normal_segments.append(seg_normals[:-1])
            else:
                point_segments.append(p0[None, :])
                normal_segments.append(n0[None, :])
        
        # Add final point
        point_segments.append(path[-1:])
        normal_segments.append(normals[-1:])
        
        return np.vstack(point_segments), np.vstack(normal_segments)
    
    @classmethod
    def extract(
        cls,
        tm: trimesh.Trimesh,
        z: float,
        max_step: float,
        plane_normal: Optional[NDArray] = None
    ) -> SliceData:
        """Extract a horizontal slice with surface normals.
        
        This method slices the mesh and preserves the face normal information
        for each point on the slice contour.
        
        Args:
            tm: Trimesh object to slice
            z: Z-coordinate of slicing plane
            max_step: Maximum spacing between interpolated points
            plane_normal: Optional custom plane normal (default: [0, 0, 1])
            
        Returns:
            SliceData containing points and their associated normals
            
        Raises:
            RuntimeError: If no intersection found at specified Z
        """
        if plane_normal is None:
            plane_normal = np.array([0, 0, 1])
        
        # =================================================================
        # STEP 1: Slice the mesh and get face indices
        # =================================================================
        # Use slice_plane which can return face indices
        try:
            # Try the newer trimesh API first
            slice_result = tm.section(
                plane_origin=[0, 0, z],
                plane_normal=plane_normal
            )
            
            if slice_result is None or slice_result.vertices.size == 0:
                raise RuntimeError(f"No intersection found at Z={z}")
            
            # Get vertices from the slice
            verts_3d = np.asarray(slice_result.vertices, dtype=float)
            
            # =================================================================
            # STEP 2: For each slice vertex, find the closest mesh face
            #         and retrieve its normal
            # =================================================================
            # Since trimesh.section doesn't directly give us face indices,
            # we need to find which faces each slice point came from
            
            normals_3d = cls._compute_normals_for_slice_points(tm, verts_3d, z)
            
        except Exception as e:
            logger.warning(f"Advanced slicing failed: {e}, falling back to basic method")
            # Fallback to basic method with estimated normals
            slice_result = tm.section(
                plane_origin=[0, 0, z],
                plane_normal=plane_normal
            )
            if slice_result is None:
                raise RuntimeError(f"No intersection found at Z={z}")
            
            verts_3d = np.asarray(slice_result.vertices, dtype=float)
            normals_3d = cls._estimate_normals_from_contour(verts_3d)
        
        # =================================================================
        # STEP 3: Project to 2D
        # =================================================================
        points_2d = verts_3d[:, :2]
        
        # Project normals to 2D (take X, Y components) and normalize
        normals_2d = normals_3d[:, :2]
        norms = np.linalg.norm(normals_2d, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        normals_2d = normals_2d / norms
        
        # =================================================================
        # STEP 4: Process path entities and interpolate
        # =================================================================
        point_paths: list[Points2D] = []
        normal_paths: list[Normals2D] = []
        
        if hasattr(slice_result, "entities") and len(slice_result.entities) > 0:
            for ent in slice_result.entities:
                if hasattr(ent, "points") and len(ent.points) >= 2:
                    indices = np.asarray(ent.points, dtype=int)
                    pts = points_2d[indices]
                    nrm = normals_2d[indices]
                    
                    # Interpolate with normals
                    pts_interp, nrm_interp = cls.interpolate_path_with_normals(
                        pts, nrm, max_step
                    )
                    point_paths.append(pts_interp)
                    normal_paths.append(nrm_interp)
        
        # Fallback: treat all points as a single path
        if not point_paths and len(points_2d) >= 2:
            pts_interp, nrm_interp = cls.interpolate_path_with_normals(
                points_2d, normals_2d, max_step
            )
            point_paths.append(pts_interp)
            normal_paths.append(nrm_interp)
        
        if not point_paths:
            raise RuntimeError(f"No valid paths extracted from slice at Z={z}")
        
        result_points = np.vstack(point_paths)
        result_normals = np.vstack(normal_paths)
        
        logger.debug(f"Extracted {len(result_points)} points with normals from slice at Z={z}")
        
        return SliceData(points=result_points, normals=result_normals)
    
    @staticmethod
    def _compute_normals_for_slice_points(
        tm: trimesh.Trimesh,
        slice_verts: NDArray,
        z: float
    ) -> NDArray:
        """Compute face normals for each slice vertex.
        
        For each vertex on the slice, finds the closest mesh face
        that intersects the slicing plane and returns its normal.
        
        Args:
            tm: Original trimesh
            slice_verts: 3D vertices from the slice, shape (N, 3)
            z: Z-coordinate of the slicing plane
            
        Returns:
            Array of 3D normals, shape (N, 3)
        """
        # Get all face normals from the mesh
        face_normals = tm.face_normals  # Shape: (num_faces, 3)
        face_centers = tm.triangles_center  # Shape: (num_faces, 3)
        
        # Find faces that are near the slicing plane
        # A face intersects the plane if its vertices span across z
        vertices = tm.vertices
        faces = tm.faces
        
        # For each face, check if it crosses z
        face_z_min = vertices[faces].min(axis=1)[:, 2]
        face_z_max = vertices[faces].max(axis=1)[:, 2]
        crossing_mask = (face_z_min <= z) & (face_z_max >= z)
        crossing_indices = np.where(crossing_mask)[0]
        
        if len(crossing_indices) == 0:
            logger.warning("No faces cross the slicing plane, using estimated normals")
            return SliceExtractorWithNormals._estimate_normals_from_contour(slice_verts)
        
        # Get centers and normals of crossing faces
        crossing_centers = face_centers[crossing_indices]  # Shape: (M, 3)
        crossing_normals = face_normals[crossing_indices]  # Shape: (M, 3)
        
        # For each slice vertex, find the nearest crossing face
        normals = np.zeros_like(slice_verts)
        
        for i, vert in enumerate(slice_verts):
            # Compute distance to each crossing face center (in XY plane)
            xy_dist = np.linalg.norm(crossing_centers[:, :2] - vert[:2], axis=1)
            nearest_idx = np.argmin(xy_dist)
            normals[i] = crossing_normals[nearest_idx]
        
        return normals
    
    @staticmethod
    def _estimate_normals_from_contour(verts: NDArray) -> NDArray:
        """Estimate normals from contour geometry (fallback method).
        
        Uses the contour tangent to estimate the outward normal.
        This is less accurate but works when face data is unavailable.
        
        Args:
            verts: 3D vertices from the slice
            
        Returns:
            Estimated 3D normals (Z component will be ~0)
        """
        n = len(verts)
        normals = np.zeros_like(verts)
        
        for i in range(n):
            # Get neighboring points
            prev_pt = verts[(i - 1) % n]
            next_pt = verts[(i + 1) % n]
            
            # Tangent direction
            tangent = next_pt - prev_pt
            tangent_2d = tangent[:2]
            
            # Normal is perpendicular to tangent (rotate 90°)
            # We assume counterclockwise contour, so normal points outward
            normal_2d = np.array([-tangent_2d[1], tangent_2d[0]])
            norm = np.linalg.norm(normal_2d)
            if norm > 1e-10:
                normal_2d = normal_2d / norm
            
            normals[i, :2] = normal_2d
            normals[i, 2] = 0.0
        
        return normals


# =============================================================================
# Backward compatibility: Keep old SliceExtractor working
# =============================================================================

class SliceExtractor:
    """Original slice extractor (without normals) for backward compatibility."""
    
    @staticmethod
    def interpolate_path(path: Points2D, max_step: float) -> Points2D:
        """Add interpolated points to ensure uniform point density along a path."""
        if len(path) < 2:
            return path
        
        segments: list[Points2D] = []
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            segment_length = np.linalg.norm(p1 - p0)
            
            if segment_length > max_step:
                num_points = int(np.ceil(segment_length / max_step)) + 1
                t = np.linspace(0, 1, num_points)
                segment = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
                segments.append(segment[:-1])
            else:
                segments.append(p0[None, :])
        
        segments.append(path[-1:])
        return np.vstack(segments)
    
    @classmethod
    def extract(
        cls,
        tm: trimesh.Trimesh,
        z: float,
        max_step: float,
        plane_normal: Optional[NDArray] = None
    ) -> Points2D:
        """Extract a horizontal slice (without normals)."""
        # Use the new extractor and return only points
        slice_data = SliceExtractorWithNormals.extract(tm, z, max_step, plane_normal)
        return slice_data.points
    
    @staticmethod
    def find_slice_range(tm: trimesh.Trimesh) -> tuple[float, float]:
        """Find the valid Z range for slicing."""
        bounds = tm.bounds
        return float(bounds[0, 2]), float(bounds[1, 2])
