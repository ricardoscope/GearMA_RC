"""
Circle fitting module with improved RANSAC.

This module provides robust circle fitting using multiple methods:
- Kåsa: Fast algebraic fit
- Taubin: More accurate for small arcs
- Nonlinear: Geometric least squares refinement
- RANSAC: Robust fitting with outlier rejection

Key improvements in this version:
1. Expected radius constraint - prevents convergence to unreasonable radii
2. Fixed random seed - reproducible results
3. Adaptive threshold - based on data scale
4. Pre-filtering of obvious outliers
5. Multiple validation checks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

Points2D = NDArray[np.floating]
Vector2D = NDArray[np.floating]


@dataclass
class CircleFitResult:
    """Detailed result of circle fitting.
    
    Provides more information than the basic tuple return.
    """
    center: Vector2D
    radius: float
    inliers: Points2D
    outliers: Points2D
    rmse: float
    n_iterations: int
    inlier_ratio: float
    method: str = "ransac"
    
    @property
    def is_valid(self) -> bool:
        """Check if fit is valid."""
        return (
            np.isfinite(self.radius) and 
            self.radius > 0 and
            self.inlier_ratio > 0.3
        )
    
    def to_tuple(self) -> Tuple[Vector2D, float, Points2D, Points2D, float]:
        """Convert to legacy tuple format for backward compatibility."""
        return (self.center, self.radius, self.inliers, self.outliers, self.rmse)


class CircleFitter:
    """Fits circles to 2D point clouds using various methods.
    
    Usage:
        # Basic RANSAC (backward compatible)
        center, radius, inliers, outliers, rmse = CircleFitter.fit_ransac(points)
        
        # With expected radius constraint (recommended for ghost circle)
        center, radius, inliers, outliers, rmse = CircleFitter.fit_ransac(
            points,
            expected_radius=0.1,
            radius_tolerance=0.5  # Accept 0.05 to 0.15
        )
        
        # Get detailed result
        result = CircleFitter.fit_ransac_detailed(points, expected_radius=0.1)
        if result.is_valid:
            print(f"Fit quality: {result.inlier_ratio:.1%}")
    """
    
    # Default random seed for reproducibility
    DEFAULT_SEED = 42
    
    @staticmethod
    def fit_kasa(points: Points2D) -> tuple[Vector2D, float]:
        """Fit circle using Kåsa algebraic method (least squares).
        
        Fast algebraic method that minimizes algebraic distance.
        Good initial estimate but not geometrically optimal.
        
        Args:
            points: Nx2 array of 2D points
            
        Returns:
            Tuple of (center, radius)
            
        Raises:
            ValueError: If fewer than 3 points or invalid fit
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        
        x, y = points[:, 0], points[:, 1]
        A = np.column_stack([x, y, np.ones(len(points))])
        b = -(x**2 + y**2)
        
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        a, b_coef, c = params
        
        center = np.array([-a/2, -b_coef/2])
        radius_sq = center[0]**2 + center[1]**2 - c
        
        if radius_sq <= 0:
            raise ValueError("Invalid circle fit (negative radius squared)")
        
        radius = np.sqrt(radius_sq)
        return center, radius
    
    @staticmethod
    def fit_taubin(points: Points2D) -> tuple[Vector2D, float]:
        """Fit circle using Taubin algebraic method.
        
        More accurate than Kåsa for small arcs and partial circles.
        Uses eigenvalue decomposition for better numerical stability.
        
        Args:
            points: Nx2 array of 2D points
            
        Returns:
            Tuple of (center, radius)
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        
        centroid = points.mean(axis=0)
        points_centered = points - centroid
        
        x, y = points_centered[:, 0], points_centered[:, 1]
        Mxx = (x**2).mean()
        Myy = (y**2).mean()
        Mxy = (x * y).mean()
        Mxz = (x * (x**2 + y**2)).mean()
        Myz = (y * (x**2 + y**2)).mean()
        Mzz = ((x**2 + y**2)**2).mean()
        
        M = np.array([[Mxx, Mxy, Mxz],
                      [Mxy, Myy, Myz],
                      [Mxz, Myz, Mzz]])
        
        N = np.array([[0, 0, -2],
                      [0, 0, -2],
                      [-2, -2, 8 * (Mxx + Myy)]])
        
        try:
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(N, M))
            # Get the eigenvector with smallest positive eigenvalue
            positive_mask = eigenvalues.real > 1e-10
            if not np.any(positive_mask):
                raise ValueError("No positive eigenvalues")
            
            positive_eigenvalues = eigenvalues.real[positive_mask]
            positive_eigenvectors = eigenvectors[:, positive_mask]
            idx = positive_eigenvalues.argmin()
            A = positive_eigenvectors[:, idx].real
        except (np.linalg.LinAlgError, ValueError):
            return CircleFitter.fit_kasa(points)
        
        if abs(A[2]) < 1e-10:
            return CircleFitter.fit_kasa(points)
        
        center_offset = A[:2] / (2 * A[2])
        center = centroid + center_offset
        
        radius_sq = center_offset[0]**2 + center_offset[1]**2 - A[2]
        if radius_sq <= 0:
            return CircleFitter.fit_kasa(points)
        
        radius = np.sqrt(radius_sq)
        return center, radius
    
    @staticmethod
    def fit_3_points(p1: Vector2D, p2: Vector2D, p3: Vector2D) -> tuple[Vector2D, float]:
        """Fit circle through exactly 3 points.
        
        Uses the circumcenter formula for efficiency.
        
        Args:
            p1, p2, p3: Three 2D points
            
        Returns:
            Tuple of (center, radius)
            
        Raises:
            ValueError: If points are collinear
        """
        # Translate to p1 as origin
        ax, ay = p2[0] - p1[0], p2[1] - p1[1]
        bx, by = p3[0] - p1[0], p3[1] - p1[1]
        
        d = 2 * (ax * by - ay * bx)
        
        if abs(d) < 1e-10:
            raise ValueError("Points are collinear")
        
        a_sq = ax**2 + ay**2
        b_sq = bx**2 + by**2
        
        cx = (by * a_sq - ay * b_sq) / d
        cy = (ax * b_sq - bx * a_sq) / d
        
        center = np.array([cx + p1[0], cy + p1[1]])
        radius = np.sqrt(cx**2 + cy**2)
        
        return center, radius
    
    @staticmethod
    def _circle_residuals(params: np.ndarray, points: Points2D) -> np.ndarray:
        """Calculate residuals for circle fitting (geometric distance).
        
        Residual = distance_to_center - radius
        """
        cx, cy, r = params
        distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        return distances - r
    
    @classmethod
    def fit_nonlinear(
        cls,
        points: Points2D,
        initial: tuple[Vector2D, float]
    ) -> tuple[Vector2D, float]:
        """Refine circle fit using nonlinear least squares.
        
        Minimizes geometric distance (more accurate than algebraic methods).
        Uses Levenberg-Marquardt algorithm.
        
        Args:
            points: Nx2 array of 2D points
            initial: Initial (center, radius) estimate
            
        Returns:
            Refined (center, radius)
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        
        center_init, radius_init = initial
        params_init = np.array([center_init[0], center_init[1], radius_init])
        
        result = least_squares(
            cls._circle_residuals,
            params_init,
            args=(points,),
            method='lm'
        )
        
        center = result.x[:2]
        radius = abs(result.x[2])
        
        return center, radius
    
    @classmethod
    def _prefilter_outliers(
        cls,
        points: Points2D,
        sigma_threshold: float = 3.0
    ) -> Tuple[Points2D, NDArray[np.bool_]]:
        """Pre-filter obvious outliers based on distance from centroid.
        
        Uses median absolute deviation (MAD) for robust outlier detection.
        
        Args:
            points: Input points
            sigma_threshold: Number of MAD units for outlier threshold
            
        Returns:
            Tuple of (filtered_points, keep_mask)
        """
        if len(points) < 5:
            return points, np.ones(len(points), dtype=bool)
        
        centroid = np.median(points, axis=0)  # Use median for robustness
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # Median Absolute Deviation
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))
        
        if mad < 1e-10:
            return points, np.ones(len(points), dtype=bool)
        
        # Modified Z-score
        modified_z = 0.6745 * (distances - median_dist) / mad
        keep_mask = np.abs(modified_z) < sigma_threshold
        
        # Keep at least 50% of points
        if np.sum(keep_mask) < len(points) * 0.5:
            threshold = np.percentile(np.abs(modified_z), 50)
            keep_mask = np.abs(modified_z) <= threshold
        
        return points[keep_mask], keep_mask
    
    @classmethod
    def _compute_adaptive_threshold(
        cls,
        points: Points2D,
        base_threshold: float = 0.05
    ) -> float:
        """Compute adaptive inlier threshold based on data scale.
        
        Args:
            points: Input points
            base_threshold: Minimum threshold (fraction of median distance)
            
        Returns:
            Adaptive threshold value
        """
        if len(points) < 3:
            return base_threshold
        
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        median_dist = np.median(distances)
        
        # Threshold is 5% of median distance, but at least base_threshold
        adaptive = median_dist * 0.05
        return max(base_threshold, adaptive)
    
    @classmethod
    def _is_radius_acceptable(
        cls,
        radius: float,
        expected_radius: Optional[float],
        radius_tolerance: float
    ) -> bool:
        """Check if radius is within acceptable range.
        
        Args:
            radius: Fitted radius
            expected_radius: Expected radius (None to skip check)
            radius_tolerance: Tolerance as fraction (0.5 = 50%)
            
        Returns:
            True if radius is acceptable
        """
        if expected_radius is None:
            # No constraint - just check for reasonable values
            return 0 < radius < 1e6
        
        r_min = expected_radius * (1 - radius_tolerance)
        r_max = expected_radius * (1 + radius_tolerance)
        
        return r_min <= radius <= r_max
    
    @classmethod
    def fit_ransac(
        cls,
        points: Points2D,
        min_samples: int = 3,
        residual_threshold: float = 0.05,
        max_iterations: int = 100,
        expected_radius: Optional[float] = None,
        radius_tolerance: float = 0.5,
        seed: Optional[int] = None,
        prefilter_outliers: bool = True,
        adaptive_threshold: bool = True
    ) -> tuple[Vector2D, float, Points2D, Points2D, float]:
        """Fit circle using improved RANSAC with outlier rejection.
        
        This improved version includes:
        1. Expected radius constraint - rejects fits with unreasonable radii
        2. Fixed random seed - reproducible results
        3. Adaptive threshold - scales with data
        4. Pre-filtering - removes obvious outliers before RANSAC
        5. Multiple fitting methods - tries Kåsa and 3-point fits
        
        Pipeline:
        1. Pre-filter obvious outliers (optional)
        2. Initial fit with Taubin method
        3. RANSAC iterations with radius constraint
        4. Final refit on inliers with nonlinear optimization
        
        Args:
            points: Nx2 array of 2D points
            min_samples: Minimum samples for fitting (default: 3)
            residual_threshold: Max distance to be considered inlier
            max_iterations: Maximum RANSAC iterations
            expected_radius: Expected circle radius (None to disable constraint)
            radius_tolerance: Tolerance for radius constraint (0.5 = ±50%)
            seed: Random seed for reproducibility (None uses DEFAULT_SEED)
            prefilter_outliers: Whether to pre-filter obvious outliers
            adaptive_threshold: Whether to adapt threshold to data scale
            
        Returns:
            Tuple of (center, radius, inliers, outliers, rmse)
            
        Example:
            # For ghost circle fitting where expected radius is ~0.1:
            center, radius, inliers, outliers, rmse = CircleFitter.fit_ransac(
                intersection_points,
                expected_radius=0.1,
                radius_tolerance=0.5,  # Accept 0.05 to 0.15
                residual_threshold=0.001
            )
        """
        if len(points) < min_samples:
            raise ValueError(f"Need at least {min_samples} points for RANSAC")
        
        # Setup random generator with seed for reproducibility
        if seed is None:
            seed = cls.DEFAULT_SEED
        rng = np.random.default_rng(seed)
        
        # Store original points for final inlier/outlier classification
        original_points = points.copy()
        original_indices = np.arange(len(points))
        
        # Pre-filter obvious outliers
        if prefilter_outliers and len(points) > 10:
            points, keep_mask = cls._prefilter_outliers(points)
            original_indices = original_indices[keep_mask]
            
            n_removed = np.sum(~keep_mask)
            if n_removed > 0:
                logger.debug(f"Pre-filtered {n_removed} obvious outliers")
        
        # Compute adaptive threshold if enabled
        if adaptive_threshold:
            computed_threshold = cls._compute_adaptive_threshold(points, residual_threshold)
            residual_threshold = max(residual_threshold, computed_threshold)
            logger.debug(f"Using adaptive threshold: {residual_threshold:.6f}")
        
        # Initial fit with Taubin (or Kåsa fallback)
        try:
            center_init, radius_init = cls.fit_taubin(points)
        except (ValueError, np.linalg.LinAlgError):
            try:
                center_init, radius_init = cls.fit_kasa(points)
            except ValueError:
                # Last resort: use centroid and median distance
                center_init = points.mean(axis=0)
                radius_init = np.median(np.linalg.norm(points - center_init, axis=1))
        
        # Check if initial fit satisfies radius constraint
        if not cls._is_radius_acceptable(radius_init, expected_radius, radius_tolerance):
            logger.debug(f"Initial fit radius {radius_init:.4f} outside expected range, "
                        f"will search for better fit")
        
        # Refine initial fit with nonlinear optimization
        try:
            center_init, radius_init = cls.fit_nonlinear(points, (center_init, radius_init))
        except Exception:
            pass
        
        # Initialize best model
        best_inlier_count = 0
        best_center = center_init
        best_radius = radius_init
        best_inlier_mask = np.ones(len(points), dtype=bool)
        best_score = -np.inf
        
        # Track statistics
        valid_iterations = 0
        radius_rejected = 0
        
        # RANSAC iterations
        n_iterations = min(max_iterations, max(100, len(points) * 3))
        
        for iteration in range(n_iterations):
            # Sample random subset
            sample_idx = rng.choice(len(points), min_samples, replace=False)
            sample = points[sample_idx]
            
            try:
                # Try 3-point fit first (faster for min_samples=3)
                if min_samples == 3:
                    center, radius = cls.fit_3_points(sample[0], sample[1], sample[2])
                else:
                    center, radius = cls.fit_kasa(sample)
                
                # Validate fit
                if not np.isfinite(radius) or radius <= 0:
                    continue
                
                # Check radius constraint
                if not cls._is_radius_acceptable(radius, expected_radius, radius_tolerance):
                    radius_rejected += 1
                    continue
                
                valid_iterations += 1
                
                # Find inliers
                distances = np.abs(cls._circle_residuals(
                    [center[0], center[1], radius], points
                ))
                inlier_mask = distances < residual_threshold
                inlier_count = np.sum(inlier_mask)
                
                # Score: prioritize inlier count, but also consider fit quality
                if inlier_count >= min_samples:
                    inlier_rmse = np.sqrt(np.mean(distances[inlier_mask]**2))
                    # Score combines inlier count and fit quality
                    score = inlier_count - inlier_rmse * 10
                else:
                    score = inlier_count
                
                # Update best model
                if score > best_score:
                    best_score = score
                    best_inlier_count = inlier_count
                    best_center = center
                    best_radius = radius
                    best_inlier_mask = inlier_mask
                    
            except (ValueError, np.linalg.LinAlgError):
                continue
        
        # Log RANSAC statistics
        if radius_rejected > 0:
            logger.debug(f"RANSAC: {radius_rejected}/{n_iterations} iterations rejected "
                        f"due to radius constraint")
        
        # Final refit on all inliers using nonlinear optimization
        inliers = points[best_inlier_mask]
        
        if len(inliers) >= min_samples:
            try:
                refined_center, refined_radius = cls.fit_nonlinear(
                    inliers, (best_center, best_radius)
                )
                
                # Only accept refinement if radius still acceptable
                if cls._is_radius_acceptable(refined_radius, expected_radius, radius_tolerance):
                    best_center = refined_center
                    best_radius = refined_radius
                else:
                    logger.debug(f"Nonlinear refinement rejected: radius {refined_radius:.4f} "
                                f"outside expected range")
                    
            except Exception as e:
                logger.debug(f"Nonlinear refinement failed: {e}")
        
        # Recalculate inliers on ORIGINAL points after final fit
        distances = np.abs(cls._circle_residuals(
            [best_center[0], best_center[1], best_radius], original_points
        ))
        final_inlier_mask = distances < residual_threshold
        inliers = original_points[final_inlier_mask]
        outliers = original_points[~final_inlier_mask]
        
        # Calculate RMSE on final inliers
        if len(inliers) > 0:
            residuals = cls._circle_residuals(
                [best_center[0], best_center[1], best_radius], inliers
            )
            rmse = np.sqrt(np.mean(residuals**2))
        else:
            rmse = float('inf')
        
        # Log final result
        logger.debug(f"RANSAC result: center=({best_center[0]:.4f}, {best_center[1]:.4f}), "
                    f"radius={best_radius:.4f}, inliers={len(inliers)}/{len(original_points)}, "
                    f"rmse={rmse:.6f}")
        
        return best_center, best_radius, inliers, outliers, rmse
    
    @classmethod
    def fit_ransac_detailed(
        cls,
        points: Points2D,
        **kwargs
    ) -> CircleFitResult:
        """Fit circle with RANSAC and return detailed result object.
        
        Same as fit_ransac but returns a CircleFitResult object with
        additional information.
        
        Args:
            points: Nx2 array of 2D points
            **kwargs: Arguments passed to fit_ransac
            
        Returns:
            CircleFitResult with detailed fit information
        """
        max_iterations = kwargs.get('max_iterations', 100)
        
        center, radius, inliers, outliers, rmse = cls.fit_ransac(points, **kwargs)
        
        return CircleFitResult(
            center=center,
            radius=radius,
            inliers=inliers,
            outliers=outliers,
            rmse=rmse,
            n_iterations=max_iterations,
            inlier_ratio=len(inliers) / len(points) if len(points) > 0 else 0.0,
            method="ransac"
        )