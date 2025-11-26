"""
Utility functions for gear analysis.

This module provides common helper functions used throughout the package.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Type aliases
FloatArray = NDArray[np.floating]
Vector2D = NDArray[np.floating]
Vector3D = NDArray[np.floating]


def unit_vector(v: FloatArray) -> FloatArray:
    """Normalize a vector to unit length.
    
    Returns the original vector if it has zero length (within tolerance).
    
    Args:
        v: Input vector of any dimension
        
    Returns:
        Unit vector in the same direction, or original if zero-length
    
    Example:
        >>> v = np.array([3.0, 4.0])
        >>> unit_vector(v)
        array([0.6, 0.8])
        >>> unit_vector(np.array([0.0, 0.0]))
        array([0., 0.])
    """
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


def to_3d(point_2d: Vector2D, z: float) -> Vector3D:
    """Convert a 2D point to 3D by appending a z coordinate.
    
    Args:
        point_2d: 2D point as numpy array [x, y]
        z: Z coordinate to append
        
    Returns:
        3D point as numpy array [x, y, z]
    
    Example:
        >>> to_3d(np.array([1.0, 2.0]), 0.5)
        array([1. , 2. , 0.5])
    """
    return np.array([point_2d[0], point_2d[1], z])


def angle_between_vectors(v1: FloatArray, v2: FloatArray) -> float:
    """Compute the angle between two vectors in radians.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Angle in radians [0, π]
    
    Example:
        >>> v1 = np.array([1.0, 0.0])
        >>> v2 = np.array([0.0, 1.0])
        >>> np.degrees(angle_between_vectors(v1, v2))
        90.0
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot)


def polar_to_cartesian(r: float, theta: float) -> Vector2D:
    """Convert polar coordinates to Cartesian.
    
    Args:
        r: Radial distance from origin
        theta: Angle in radians from positive x-axis
        
    Returns:
        Cartesian coordinates [x, y]
    
    Example:
        >>> polar_to_cartesian(1.0, np.pi/2)
        array([6.123234e-17, 1.000000e+00])  # approximately [0, 1]
    """
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def cartesian_to_polar(point: Vector2D) -> tuple[float, float]:
    """Convert Cartesian coordinates to polar.
    
    Args:
        point: Cartesian coordinates [x, y]
        
    Returns:
        Tuple of (radius, angle) where angle is in radians [-π, π]
    
    Example:
        >>> cartesian_to_polar(np.array([0.0, 1.0]))
        (1.0, 1.5707963267948966)  # (1.0, π/2)
    """
    r = np.linalg.norm(point)
    theta = np.arctan2(point[1], point[0])
    return r, theta


def rotation_matrix_2d(angle: float) -> NDArray[np.floating]:
    """Create a 2D rotation matrix.
    
    Args:
        angle: Rotation angle in radians (counterclockwise positive)
        
    Returns:
        2x2 rotation matrix
    
    Example:
        >>> R = rotation_matrix_2d(np.pi/2)
        >>> R @ np.array([1, 0])  # Rotate [1,0] by 90 degrees
        array([6.123234e-17, 1.000000e+00])  # approximately [0, 1]
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def perpendicular_2d(v: Vector2D, clockwise: bool = False) -> Vector2D:
    """Get a perpendicular vector to the input.
    
    Args:
        v: Input 2D vector
        clockwise: If True, rotate clockwise; otherwise counterclockwise
        
    Returns:
        Perpendicular vector (same magnitude)
    
    Example:
        >>> perpendicular_2d(np.array([1.0, 0.0]))
        array([-0.,  1.])  # [0, 1]
        >>> perpendicular_2d(np.array([1.0, 0.0]), clockwise=True)
        array([ 0., -1.])
    """
    if clockwise:
        return np.array([v[1], -v[0]])
    else:
        return np.array([-v[1], v[0]])
