"""
Point filtering module.

This module provides functions for filtering point clouds based on
geometric criteria such as radial distance.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type alias
Points2D = NDArray[np.floating]


class PointFilter:
    """Filters points based on geometric criteria.
    
    This class provides static methods for filtering point clouds,
    primarily used to isolate tooth regions in gear analysis.
    
    Example:
        >>> points = np.random.randn(1000, 2) * 2.5
        >>> filtered = PointFilter.by_radius(points, r_inner=2.0, r_outer=3.0)
    """
    
    @staticmethod
    def by_radius(
        points: Points2D,
        r_inner: float,
        r_outer: float,
        relaxation_factor: float = 0.1
    ) -> Points2D:
        """Filter points to keep only those within an annular region.
        
        Keeps points where r_inner <= distance_from_origin <= r_outer.
        If no points are found within the specified bounds, automatically
        tries with relaxed bounds (expanded by relaxation_factor).
        
        Args:
            points: 2D point array, shape (N, 2)
            r_inner: Inner radius bound (minimum distance from origin)
            r_outer: Outer radius bound (maximum distance from origin)
            relaxation_factor: Factor to relax bounds if no points found.
                              E.g., 0.1 means try ±10% of the bounds.
            
        Returns:
            Filtered 2D point array, shape (M, 2) where M <= N
            
        Raises:
            RuntimeError: If no points found even with relaxed bounds.
                         Error message includes diagnostic info about
                         the actual radius distribution.
        
        Example:
            >>> # Create points in a ring
            >>> angles = np.linspace(0, 2*np.pi, 100)
            >>> r = 2.5 + 0.1 * np.random.randn(100)
            >>> points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
            >>> filtered = PointFilter.by_radius(points, r_inner=2.3, r_outer=2.7)
            >>> len(filtered)
            100
        """
        radii = np.linalg.norm(points, axis=1)
        mask = (radii >= r_inner) & (radii <= r_outer)
        filtered = points[mask]
        
        if len(filtered) == 0:
            # Try relaxed bounds
            relaxed_inner = r_inner * (1 - relaxation_factor)
            relaxed_outer = r_outer * (1 + relaxation_factor)
            mask = (radii >= relaxed_inner) & (radii <= relaxed_outer)
            filtered = points[mask]
            
            if len(filtered) == 0:
                # Provide diagnostic information
                r_min = float(np.min(radii)) if len(radii) > 0 else 0.0
                r_max = float(np.max(radii)) if len(radii) > 0 else 0.0
                r_mean = float(np.mean(radii)) if len(radii) > 0 else 0.0
                r_p25 = float(np.percentile(radii, 25)) if len(radii) > 0 else 0.0
                r_p75 = float(np.percentile(radii, 75)) if len(radii) > 0 else 0.0
                
                raise RuntimeError(
                    f"No points found within radii {r_inner:.2f}-{r_outer:.2f}\n"
                    f"  (relaxed: {relaxed_inner:.2f}-{relaxed_outer:.2f})\n"
                    f"  Actual radius distribution:\n"
                    f"    Min: {r_min:.4f}, Max: {r_max:.4f}, Mean: {r_mean:.4f}\n"
                    f"    25th percentile: {r_p25:.4f}, 75th percentile: {r_p75:.4f}\n"
                    f"  Suggested values:\n"
                    f"    R_INNER ≈ {r_p25:.4f}, R_OUTER ≈ {r_p75:.4f}"
                )
            
            logger.warning(f"Using relaxed radii: {relaxed_inner:.2f}-{relaxed_outer:.2f}")
        
        logger.debug(f"Filtered {len(filtered)}/{len(points)} points within radius [{r_inner}, {r_outer}]")
        
        return filtered
    
    @staticmethod
    def by_angle(
        points: Points2D,
        angle_min: float,
        angle_max: float
    ) -> Points2D:
        """Filter points to keep only those within an angular sector.
        
        Keeps points where angle_min <= angle <= angle_max.
        Angles are measured in radians from the positive x-axis.
        
        Args:
            points: 2D point array, shape (N, 2)
            angle_min: Minimum angle in radians
            angle_max: Maximum angle in radians
            
        Returns:
            Filtered 2D point array
        
        Example:
            >>> # Keep points in the first quadrant
            >>> filtered = PointFilter.by_angle(points, 0, np.pi/2)
        """
        angles = np.arctan2(points[:, 1], points[:, 0])
        # Normalize to [0, 2π)
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        
        # Handle wraparound
        if angle_min < angle_max:
            mask = (angles >= angle_min) & (angles <= angle_max)
        else:
            # Wraparound case (e.g., angle_min=5.5, angle_max=0.5)
            mask = (angles >= angle_min) | (angles <= angle_max)
        
        return points[mask]
    
    @staticmethod
    def remove_outliers(
        points: Points2D,
        std_threshold: float = 3.0
    ) -> Points2D:
        """Remove outlier points based on distance from centroid.
        
        Removes points that are more than std_threshold standard
        deviations from the centroid.
        
        Args:
            points: 2D point array, shape (N, 2)
            std_threshold: Number of standard deviations for outlier detection
            
        Returns:
            Filtered 2D point array with outliers removed
        """
        if len(points) < 3:
            return points
        
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        mean_dist = distances.mean()
        std_dist = distances.std()
        
        mask = distances <= (mean_dist + std_threshold * std_dist)
        return points[mask]
