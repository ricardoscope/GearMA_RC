"""
Line fitting module.

This module provides algorithms for fitting lines to 2D point clouds,
including SVD-based fitting and tooth flank extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from gear_analysis.utils import unit_vector

logger = logging.getLogger(__name__)

# Type aliases
Points2D = NDArray[np.floating]
Vector2D = NDArray[np.floating]


@dataclass
class ToothFlanks:
    """Contains both flanks of a single tooth.
    
    Attributes:
        tooth: Tooth number (1-indexed)
        left_point: Centroid of left flank points
        left_direction: Direction vector of left flank
        left_n_points: Number of points in left flank
        right_point: Centroid of right flank points
        right_direction: Direction vector of right flank
        right_n_points: Number of points in right flank
    """
    tooth: int
    left_point: Vector2D
    left_direction: Vector2D
    left_n_points: int
    right_point: Vector2D
    right_direction: Vector2D
    right_n_points: int


class LineFitter:
    """Fits lines to point clouds using various methods.
    
    This class provides methods for:
    - SVD-based line fitting (principal component analysis)
    - Tooth flank extraction from gear point clusters
    
    Example:
        >>> points = np.array([[1, 1], [2, 2], [3, 3], [4, 4.1]])
        >>> center, direction = LineFitter.fit_svd(points)
        >>> np.allclose(direction, [0.707, 0.707], atol=0.1)
        True
    """
    
    @staticmethod
    def fit_svd(points: Points2D) -> tuple[Vector2D, Vector2D]:
        """Fit a line to 2D points using SVD (Principal Component Analysis).
        
        The line is represented by its centroid and principal direction
        (the direction of maximum variance in the point cloud).
        
        Args:
            points: 2D point array, shape (N, 2) where N >= 2
            
        Returns:
            Tuple of (centroid, direction_vector) where both are 2D arrays.
            The direction vector is normalized to unit length.
            
        Raises:
            ValueError: If fewer than 2 points provided
        
        Example:
            >>> points = np.array([[0, 0], [1, 1], [2, 2]])
            >>> center, direction = LineFitter.fit_svd(points)
            >>> center
            array([1., 1.])
            >>> direction  # Unit vector along y=x line
            array([0.70710678, 0.70710678])
        """
        if len(points) < 2:
            raise ValueError("Need at least two points to fit a line.")
        
        centroid = points.mean(axis=0)
        
        # SVD on centered points
        # The first right singular vector corresponds to the principal direction
        centered = points - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = unit_vector(vt[0])
        
        return centroid, direction
    
    @staticmethod
    def fit_total_least_squares(points: Points2D) -> tuple[Vector2D, Vector2D]:
        """Fit a line using total least squares (orthogonal regression).
        
        This is equivalent to fit_svd but named for clarity.
        Minimizes the orthogonal distance from points to the line.
        
        Args:
            points: 2D point array, shape (N, 2)
            
        Returns:
            Tuple of (point_on_line, direction_vector)
        """
        return LineFitter.fit_svd(points)
    
    @classmethod
    def extract_right_flank(
        cls,
        points: Points2D,
        min_points: int
    ) -> tuple[Vector2D, Vector2D]:
        """Extract and fit a line to the right flank of a tooth cluster.
        
        This method assumes the gear is centered at the origin and uses
        the following strategy:
        1. Calculate the cluster center
        2. Define a local coordinate system (radial/tangential)
        3. Split points into left/right based on tangential projection
        4. Fit a line to the right-side points only
        
        The "right" side is determined by sorting along the tangential
        direction and taking points above the median.
        
        Args:
            points: 2D points belonging to one tooth, shape (N, 2)
            min_points: Minimum points needed for fitting
            
        Returns:
            Tuple of (flank_centroid, flank_direction)
            
        Raises:
            ValueError: If insufficient points for fitting
        
        Example:
            >>> # Create a synthetic tooth cluster
            >>> angles = np.linspace(0.1, 0.3, 50)
            >>> r = 2.5 + 0.05 * np.random.randn(50)
            >>> points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
            >>> center, direction = LineFitter.extract_right_flank(points, min_points=5)
        """
        center = points.mean(axis=0)
        
        # Define local radial direction (from origin to cluster center)
        radial = unit_vector(center) if np.linalg.norm(center) > 1e-10 else np.array([1.0, 0.0])
        
        # Tangential direction: 90° counterclockwise from radial
        tangential = np.array([-radial[1], radial[0]])
        
        # Project points onto tangential axis
        projections = (points - center) @ tangential
        
        # Split at median to get right half
        median = np.median(projections)
        right = points[projections > median]
        
        # Fallback: ensure enough points for fitting
        if len(right) < min_points:
            order = np.argsort(projections)
            half = max(min_points, len(points) // 2)
            right = points[order[-half:]]
            
            if len(right) < 3:
                raise ValueError("Insufficient points for flank fitting.")
        
        return cls.fit_svd(right)
    
    @classmethod
    def extract_left_flank(
        cls,
        points: Points2D,
        min_points: int
    ) -> tuple[Vector2D, Vector2D]:
        """Extract and fit a line to the left flank of a tooth cluster.
        
        Similar to extract_right_flank but takes points below the
        tangential median.
        
        Args:
            points: 2D points belonging to one tooth
            min_points: Minimum points needed for fitting
            
        Returns:
            Tuple of (flank_centroid, flank_direction)
        """
        center = points.mean(axis=0)
        
        radial = unit_vector(center) if np.linalg.norm(center) > 1e-10 else np.array([1.0, 0.0])
        tangential = np.array([-radial[1], radial[0]])
        
        projections = (points - center) @ tangential
        median = np.median(projections)
        left = points[projections < median]
        
        if len(left) < min_points:
            order = np.argsort(projections)
            half = max(min_points, len(points) // 2)
            left = points[order[:half]]
            
            if len(left) < 3:
                raise ValueError("Insufficient points for flank fitting.")
        
        return cls.fit_svd(left)
    
    @classmethod
    def extract_both_flanks(
        cls,
        points: Points2D,
        min_points: int,
        tooth_number: int
    ) -> Optional[ToothFlanks]:
        """Extract and fit lines to BOTH flanks of a tooth cluster.
        
        This is the key method for proper ghost circle analysis.
        It extracts left and right flanks from the same tooth cluster.
        
        Args:
            points: 2D points belonging to one tooth, shape (N, 2)
            min_points: Minimum points needed for each flank
            tooth_number: Tooth index (1-indexed)
            
        Returns:
            ToothFlanks object containing both flanks, or None if fitting fails
        
        Example:
            >>> flanks = LineFitter.extract_both_flanks(points, min_points=5, tooth_number=1)
            >>> if flanks:
            ...     print(f"Left flank at {flanks.left_point}")
            ...     print(f"Right flank at {flanks.right_point}")
        """
        if len(points) < 2 * min_points:
            logger.warning(f"Tooth {tooth_number}: Not enough points ({len(points)}) for both flanks")
            return None
        
        center = points.mean(axis=0)
        
        # Define local coordinate system
        radial = unit_vector(center) if np.linalg.norm(center) > 1e-10 else np.array([1.0, 0.0])
        tangential = np.array([-radial[1], radial[0]])
        
        # Project points onto tangential axis
        projections = (points - center) @ tangential
        
        # Split into left and right based on median
        median = np.median(projections)
        left_mask = projections < median
        right_mask = projections > median
        
        left_points = points[left_mask]
        right_points = points[right_mask]
        
        # Check if we have enough points for both flanks
        if len(left_points) < min_points or len(right_points) < min_points:
            logger.warning(f"Tooth {tooth_number}: Insufficient points for flanks "
                          f"(left={len(left_points)}, right={len(right_points)})")
            return None
        
        try:
            left_center, left_dir = cls.fit_svd(left_points)
            right_center, right_dir = cls.fit_svd(right_points)
            
            return ToothFlanks(
                tooth=tooth_number,
                left_point=left_center,
                left_direction=unit_vector(left_dir),
                left_n_points=len(left_points),
                right_point=right_center,
                right_direction=unit_vector(right_dir),
                right_n_points=len(right_points),
            )
        except ValueError as e:
            logger.warning(f"Tooth {tooth_number}: Flank fitting failed: {e}")
            return None
    
    @staticmethod
    def line_distance(
        point: Vector2D,
        line_point: Vector2D,
        line_direction: Vector2D
    ) -> float:
        """Calculate perpendicular distance from a point to a line.
        
        Args:
            point: Query point
            line_point: A point on the line
            line_direction: Direction vector of the line (should be unit)
            
        Returns:
            Unsigned perpendicular distance
        """
        d = unit_vector(line_direction)
        v = point - line_point
        # Distance = |v - (v·d)d|
        projection = np.dot(v, d) * d
        return float(np.linalg.norm(v - projection))