"""
Gear center estimation module.

This module provides methods for estimating the geometric center
of a gear from its cross-section points.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from gear_analysis.geometry.circle_fitting import CircleFitter
from gear_analysis.utils import compute_radii

logger = logging.getLogger(__name__)

# Type aliases
Points2D = NDArray[np.floating]
Vector2D = NDArray[np.floating]


class GearCenterEstimator:
    """Estimates the gear center using various methods.
    
    The gear center is needed as a reference point to measure
    the offset of the ghost circle center, which indicates
    manufacturing setup errors.
    
    Available methods:
    - outer_tips: Fit circle to outer boundary points
    - boundary_centroid: Use centroid of all boundary points
    
    Example:
        >>> center, radius = GearCenterEstimator.from_outer_tips(
        ...     filtered_points, r_outer=2.58
        ... )
    """
    
    @staticmethod
    def from_outer_tips(
        filtered_points: Points2D,
        r_outer: float
    ) -> tuple[Vector2D, float]:
        """Estimate gear center by fitting a circle to outer boundary points.
        
        This method selects points near the outer radius (within 95%
        of r_outer) and fits a circle to them. The circle center
        represents the gear center.
        
        This method works best when the outer boundary is well-defined
        and has consistent sampling.
        
        Args:
            filtered_points: Points in the annular region, shape (N, 2)
            r_outer: Outer radius threshold for point selection
            
        Returns:
            Tuple of (center, radius) where center is shape (2,)
        
        Example:
            >>> center, radius = GearCenterEstimator.from_outer_tips(
            ...     filtered_points, r_outer=2.58
            ... )
            >>> np.allclose(center, [0, 0], atol=0.1)
            True
        """
        if len(filtered_points) == 0:
            logger.warning("No points provided, returning origin as center")
            return np.zeros(2), r_outer
        
        # Select points near outer radius
        radii = compute_radii(filtered_points)
        outer_threshold = r_outer * 0.95
        outer_points = filtered_points[radii >= outer_threshold]
        
        if len(outer_points) < 3:
            logger.warning("Too few outer points, using all points")
            outer_points = filtered_points
        
        # Fit circle using Taubin + nonlinear refinement
        try:
            center, radius = CircleFitter.fit_taubin(outer_points)
            center, radius = CircleFitter.fit_nonlinear(
                outer_points, (center, radius)
            )
        except Exception as e:
            logger.warning(f"Circle fitting failed: {e}, using centroid")
            center = outer_points.mean(axis=0)
            radius = np.linalg.norm(outer_points - center, axis=1).mean()
        
        logger.debug(f"Outer tips center: ({center[0]:.4f}, {center[1]:.4f}), radius: {radius:.4f}")
        
        return center, radius
    
    @staticmethod
    def from_boundary_centroid(
        slice_points: Points2D,
        r_outer: float
    ) -> tuple[Vector2D, float]:
        """Estimate gear center as centroid of boundary points projected to r_outer.
        
        This method projects all slice points outward to r_outer and
        computes their centroid. This can be more robust when the
        outer boundary is irregular.
        
        Args:
            slice_points: All slice boundary points, shape (N, 2)
            r_outer: Outer radius for projection
            
        Returns:
            Tuple of (center, radius)
        
        Example:
            >>> center, radius = GearCenterEstimator.from_boundary_centroid(
            ...     slice_points, r_outer=2.58
            ... )
        """
        if len(slice_points) == 0:
            logger.warning("No points provided, returning origin as center")
            return np.zeros(2), r_outer
        
        # Compute radial distances
        radii = compute_radii(slice_points)
        
        # Avoid division by zero for points at origin
        radii[radii < 1e-10] = 1.0
        
        # Project all points to r_outer
        scale_factors = r_outer / radii
        projected = slice_points * scale_factors[:, None]
        
        # Centroid of projected points
        center = projected.mean(axis=0)
        
        logger.debug(f"Boundary centroid center: ({center[0]:.4f}, {center[1]:.4f})")
        
        return center, r_outer
    
    @staticmethod
    def from_least_squares_circle(
        slice_points: Points2D
    ) -> tuple[Vector2D, float]:
        """Estimate gear center by fitting a circle to all slice points.
        
        Uses RANSAC to robustly fit a circle to all slice boundary points.
        
        Args:
            slice_points: All slice boundary points, shape (N, 2)
            
        Returns:
            Tuple of (center, radius)
        """
        if len(slice_points) < 3:
            logger.warning("Too few points, returning origin as center")
            return np.zeros(2), 1.0
        
        try:
            center, radius, _, _, rmse = CircleFitter.fit_ransac(
                slice_points,
                min_samples=3,
                residual_threshold=0.1,
                n_iterations=100
            )
            logger.debug(f"LS circle center: ({center[0]:.4f}, {center[1]:.4f}), "
                        f"radius: {radius:.4f}, RMSE: {rmse:.6f}")
        except Exception as e:
            logger.warning(f"RANSAC circle fit failed: {e}")
            center = slice_points.mean(axis=0)
            radius = np.linalg.norm(slice_points - center, axis=1).mean()
        
        return center, radius
