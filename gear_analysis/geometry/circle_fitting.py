"""
Circle fitting module 
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

Points2D = NDArray[np.floating]
Vector2D = NDArray[np.floating]


class CircleFitter:
    """Fits circles to 2D point clouds using various methods."""
    
    @staticmethod
    def fit_kasa(points: Points2D) -> tuple[Vector2D, float]:
        """Fit circle using Kåsa algebraic method (least squares).
        
        Fast algebraic method that minimizes algebraic distance.
        Good initial estimate but not geometrically optimal.
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
        
        More accurate than Kåsa for small arcs.
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
    def _circle_residuals(params: np.ndarray, points: Points2D) -> np.ndarray:
        """Calculate residuals for circle fitting (geometric distance)."""
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
    def fit_ransac(
        cls,
        points: Points2D,
        min_samples: int = 3,
        residual_threshold: float = 0.05,
        max_iterations: int = 100
    ) -> tuple[Vector2D, float, Points2D, Points2D, float]:
        """Fit circle using RANSAC to handle outliers robustly.
        
        Pipeline:
        1. Initial fit with Taubin method
        2. Refine with nonlinear least squares
        3. RANSAC to identify inliers
        4. Final refit on inliers only
        
        Returns:
            Tuple of (center, radius, inliers, outliers, rmse)
        """
        if len(points) < min_samples:
            raise ValueError(f"Need at least {min_samples} points for RANSAC")
        
        # Initial fit with Taubin
        try:
            center_init, radius_init = cls.fit_taubin(points)
        except (ValueError, np.linalg.LinAlgError):
            center_init, radius_init = cls.fit_kasa(points)
        
        # Refine with nonlinear
        try:
            center_init, radius_init = cls.fit_nonlinear(points, (center_init, radius_init))
        except Exception:
            pass
        
        best_inlier_count = 0
        best_center = center_init
        best_radius = radius_init
        best_inlier_mask = np.ones(len(points), dtype=bool)
        
        # RANSAC iterations
        n_iterations = min(max_iterations, len(points) * 2)
        
        for _ in range(n_iterations):
            # Sample random subset
            sample_idx = np.random.choice(len(points), min_samples, replace=False)
            sample = points[sample_idx]
            
            try:
                # Fit circle to sample
                center, radius = cls.fit_kasa(sample)
                
                # Skip invalid fits
                if radius <= 0 or not np.isfinite(radius):
                    continue
                
                # Find inliers
                distances = np.abs(cls._circle_residuals(
                    [center[0], center[1], radius], points
                ))
                inlier_mask = distances < residual_threshold
                inlier_count = np.sum(inlier_mask)
                
                # Update best model if more inliers
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_center = center
                    best_radius = radius
                    best_inlier_mask = inlier_mask
                    
            except Exception:
                continue
        
        # Final refit on all inliers
        inliers = points[best_inlier_mask]
        outliers = points[~best_inlier_mask]
        
        if len(inliers) >= min_samples:
            try:
                best_center, best_radius = cls.fit_nonlinear(
                    inliers, (best_center, best_radius)
                )
            except Exception:
                pass
        
        # Recalculate inliers after final fit
        distances = np.abs(cls._circle_residuals(
            [best_center[0], best_center[1], best_radius], points
        ))
        final_inlier_mask = distances < residual_threshold
        inliers = points[final_inlier_mask]
        outliers = points[~final_inlier_mask]
        
        # Calculate RMSE on inliers
        if len(inliers) > 0:
            residuals = cls._circle_residuals(
                [best_center[0], best_center[1], best_radius], inliers
            )
            rmse = np.sqrt(np.mean(residuals**2))
        else:
            rmse = float('inf')
        
        return best_center, best_radius, inliers, outliers, rmse