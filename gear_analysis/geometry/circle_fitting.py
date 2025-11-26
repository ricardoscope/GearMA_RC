"""
Circle fitting module.

This module provides algorithms for fitting circles to 2D point clouds,
including algebraic methods (Kåsa, Taubin) and robust fitting (RANSAC).
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

# Type aliases
Points2D = NDArray[np.floating]
Vector2D = NDArray[np.floating]
FloatArray = NDArray[np.floating]


class CircleFitter:
    """Fits circles to point clouds using various methods.
    
    This class provides multiple circle fitting algorithms:
    - Kåsa: Fast algebraic method, good for initial estimates
    - Taubin: More accurate algebraic method for small arcs
    - Nonlinear: Geometric fitting using least squares
    - RANSAC: Robust fitting that handles outliers
    
    Example:
        >>> # Create points on a circle
        >>> angles = np.linspace(0, 2*np.pi, 100)
        >>> r, cx, cy = 5.0, 1.0, 2.0
        >>> points = np.column_stack([cx + r*np.cos(angles), cy + r*np.sin(angles)])
        >>> center, radius = CircleFitter.fit_kasa(points)
        >>> np.allclose(center, [1, 2], atol=0.1)
        True
    """
    
    @staticmethod
    def fit_kasa(points: Points2D) -> tuple[Vector2D, float]:
        """Fit circle using Kåsa algebraic method (least squares).
        
        This is a fast algebraic method that minimizes algebraic distance.
        It provides a good initial estimate but is not geometrically optimal.
        
        The method fits a circle x² + y² + ax + by + c = 0 using
        linear least squares, then extracts center and radius.
        
        Args:
            points: 2D points, shape (N, 2) where N >= 3
            
        Returns:
            Tuple of (center, radius) where center is shape (2,)
            
        Raises:
            ValueError: If fewer than 3 points provided
        
        Example:
            >>> points = np.array([[0, 5], [5, 0], [0, -5], [-5, 0]])
            >>> center, radius = CircleFitter.fit_kasa(points)
            >>> np.allclose(center, [0, 0], atol=0.01)
            True
            >>> np.isclose(radius, 5.0, atol=0.01)
            True
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        
        x, y = points[:, 0], points[:, 1]
        
        # Build design matrix for: x² + y² + a*x + b*y + c = 0
        A = np.column_stack([x, y, np.ones(len(points))])
        b = -(x**2 + y**2)
        
        # Solve least squares: A @ [a, b, c]^T = b
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        a, b_coef, c = params
        
        # Extract center and radius from parameters
        # Center: (-a/2, -b/2)
        # Radius: sqrt(center_x² + center_y² - c)
        center = np.array([-a/2, -b_coef/2])
        radius_sq = center[0]**2 + center[1]**2 - c
        
        if radius_sq < 0:
            logger.warning("Negative radius squared in Kåsa fit, using absolute value")
            radius_sq = abs(radius_sq)
        
        radius = np.sqrt(radius_sq)
        
        return center, radius
    
    @staticmethod
    def fit_taubin(points: Points2D) -> tuple[Vector2D, float]:
        """Fit circle using Taubin algebraic method.
        
        More accurate than Kåsa for small arcs. Uses a generalized
        eigenvalue formulation that better handles partial circles.
        
        Args:
            points: 2D points, shape (N, 2) where N >= 3
            
        Returns:
            Tuple of (center, radius)
            
        Raises:
            ValueError: If fewer than 3 points provided
        
        Example:
            >>> # Small arc (90 degrees)
            >>> angles = np.linspace(0, np.pi/2, 50)
            >>> points = np.column_stack([5*np.cos(angles), 5*np.sin(angles)])
            >>> center, radius = CircleFitter.fit_taubin(points)
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        
        # Center data for numerical stability
        centroid = points.mean(axis=0)
        points_centered = points - centroid
        
        x, y = points_centered[:, 0], points_centered[:, 1]
        
        # Build moment matrix
        Mxx = (x**2).mean()
        Myy = (y**2).mean()
        Mxy = (x * y).mean()
        Mxz = (x * (x**2 + y**2)).mean()
        Myz = (y * (x**2 + y**2)).mean()
        Mzz = ((x**2 + y**2)**2).mean()
        
        # Matrices for generalized eigenvalue problem
        M = np.array([
            [Mxx, Mxy, Mxz],
            [Mxy, Myy, Myz],
            [Mxz, Myz, Mzz]
        ])
        
        N = np.array([
            [0, 0, -2],
            [0, 0, -2],
            [-2, -2, 8 * (Mxx + Myy)]
        ])
        
        # Solve generalized eigenvalue problem: M @ v = λ * N @ v
        try:
            # Equivalent to finding eigenvectors of N^(-1) @ M
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(N, M))
            # Take eigenvector with smallest eigenvalue
            idx = eigenvalues.argmin()
            A = eigenvectors[:, idx].real
        except np.linalg.LinAlgError:
            # Fallback to Kåsa
            logger.warning("Taubin eigenvalue solve failed, falling back to Kåsa")
            return CircleFitter.fit_kasa(points)
        
        # Extract center and radius
        if abs(A[2]) > 1e-10:
            center_offset = A[:2] / (2 * A[2])
        else:
            center_offset = np.zeros(2)
        
        center = centroid + center_offset
        
        radius_sq = center_offset[0]**2 + center_offset[1]**2 - A[2]
        if radius_sq < 0:
            radius_sq = abs(radius_sq)
        radius = np.sqrt(radius_sq)
        
        return center, radius
    
    @staticmethod
    def _circle_residuals(params: FloatArray, points: Points2D) -> FloatArray:
        """Calculate residuals for circle fitting (signed geometric distance).
        
        Args:
            params: Circle parameters [cx, cy, r]
            points: 2D points, shape (N, 2)
            
        Returns:
            Array of signed residuals (distance - radius)
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
        
        Minimizes the sum of squared geometric distances from points
        to the circle boundary. This is more accurate than algebraic
        methods but requires a good initial guess.
        
        Uses Levenberg-Marquardt optimization.
        
        Args:
            points: 2D points, shape (N, 2) where N >= 3
            initial: Initial guess as (center, radius)
            
        Returns:
            Tuple of (center, radius)
            
        Raises:
            ValueError: If fewer than 3 points provided
        
        Example:
            >>> # Start with approximate fit
            >>> center_init, radius_init = CircleFitter.fit_kasa(points)
            >>> # Refine
            >>> center, radius = CircleFitter.fit_nonlinear(
            ...     points, (center_init, radius_init)
            ... )
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        
        center_init, radius_init = initial
        params_init = np.array([center_init[0], center_init[1], radius_init])
        
        # Run Levenberg-Marquardt optimization
        result = least_squares(
            cls._circle_residuals,
            params_init,
            args=(points,),
            method='lm'
        )
        
        center = result.x[:2]
        radius = result.x[2]
        
        return center, radius
    
    @classmethod
    def fit_ransac(
        cls,
        points: Points2D,
        min_samples: int = 3,
        residual_threshold: float = 0.05,
        n_iterations: int = 100
    ) -> tuple[Vector2D, float, Points2D, Points2D, float]:
        """Fit circle using RANSAC to handle outliers robustly.
        
        The pipeline is:
        1. Initial fit with Taubin method
        2. Refine with nonlinear least squares
        3. RANSAC loop to identify inliers
        4. Final refit on inliers only
        
        Args:
            points: 2D points, shape (N, 2) where N >= min_samples
            min_samples: Minimum points to define a circle (default: 3)
            residual_threshold: Maximum distance for a point to be
                               considered an inlier (default: 0.05)
            n_iterations: Number of RANSAC iterations (default: 100)
            
        Returns:
            Tuple of (center, radius, inliers, outliers, rmse) where:
            - center: Circle center, shape (2,)
            - radius: Circle radius
            - inliers: Points classified as inliers, shape (M, 2)
            - outliers: Points classified as outliers, shape (K, 2)
            - rmse: Root mean square error on inliers
            
        Raises:
            ValueError: If fewer than min_samples points provided
        
        Example:
            >>> # Points with some outliers
            >>> angles = np.linspace(0, 2*np.pi, 100)
            >>> good_points = np.column_stack([5*np.cos(angles), 5*np.sin(angles)])
            >>> outliers = 10 * np.random.randn(10, 2)
            >>> all_points = np.vstack([good_points, outliers])
            >>> center, radius, inliers, outliers, rmse = CircleFitter.fit_ransac(
            ...     all_points, residual_threshold=0.5
            ... )
        """
        if len(points) < min_samples:
            raise ValueError(f"Need at least {min_samples} points for RANSAC")
        
        # Get initial estimate
        try:
            center_init, radius_init = cls.fit_taubin(points)
        except (ValueError, np.linalg.LinAlgError):
            center_init, radius_init = cls.fit_kasa(points)
        
        # Refine initial estimate
        try:
            center_init, radius_init = cls.fit_nonlinear(
                points, (center_init, radius_init)
            )
        except Exception:
            pass  # Use algebraic fit if refinement fails
        
        # RANSAC loop
        best_inliers: Points2D = np.empty((0, 2))
        best_center = center_init
        best_radius = radius_init
        
        actual_iterations = min(n_iterations, len(points) * 2)
        
        for _ in range(actual_iterations):
            # Random sample
            sample_idx = np.random.choice(len(points), min_samples, replace=False)
            sample = points[sample_idx]
            
            try:
                # Fit to sample
                center, radius = cls.fit_kasa(sample)
                
                # Find inliers
                distances = np.abs(cls._circle_residuals(
                    [center[0], center[1], radius], points
                ))
                inlier_mask = distances < residual_threshold
                inliers = points[inlier_mask]
                
                # Update best model if we found more inliers
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_center = center
                    best_radius = radius
                    
            except Exception:
                continue
        
        # Final refit on all inliers
        if len(best_inliers) >= min_samples:
            try:
                best_center, best_radius = cls.fit_nonlinear(
                    best_inliers,
                    (best_center, best_radius)
                )
            except Exception:
                pass
        
        # Classify all points as inliers/outliers
        distances = np.abs(cls._circle_residuals(
            [best_center[0], best_center[1], best_radius], points
        ))
        inlier_mask = distances < residual_threshold
        inliers = points[inlier_mask]
        outliers = points[~inlier_mask]
        
        # Calculate RMSE on inliers
        if len(inliers) > 0:
            inlier_residuals = cls._circle_residuals(
                [best_center[0], best_center[1], best_radius], inliers
            )
            rmse = float(np.sqrt(np.mean(inlier_residuals**2)))
        else:
            rmse = float('inf')
        
        logger.debug(f"RANSAC: {len(inliers)} inliers, {len(outliers)} outliers, RMSE={rmse:.6f}")
        
        return best_center, best_radius, inliers, outliers, rmse
    
    @staticmethod
    def circle_fit_quality(
        points: Points2D,
        center: Vector2D,
        radius: float
    ) -> dict:
        """Compute quality metrics for a circle fit.
        
        Args:
            points: 2D points used for fitting
            center: Fitted circle center
            radius: Fitted circle radius
            
        Returns:
            Dictionary with quality metrics:
            - rmse: Root mean square error
            - max_error: Maximum error
            - mean_error: Mean absolute error
            - std_error: Standard deviation of errors
        """
        distances = np.sqrt(
            (points[:, 0] - center[0])**2 + 
            (points[:, 1] - center[1])**2
        )
        errors = np.abs(distances - radius)
        
        return {
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "max_error": float(np.max(errors)),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
        }
