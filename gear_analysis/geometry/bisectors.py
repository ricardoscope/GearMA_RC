"""
Bisector computation and intersection module.

This module provides algorithms for computing bisectors between
tooth flanks and finding their intersection points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from gear_analysis.models import FlankLine, PairBisector
from gear_analysis.utils import unit_vector, compute_radii, orient_direction_inward

if TYPE_CHECKING:
    from gear_analysis.geometry.line_fitting import ToothFlanks

logger = logging.getLogger(__name__)

# Type aliases
Vector2D = NDArray[np.floating]


@dataclass
class ToothBisector:
    """Represents the angle bisector between left and right flanks of a SINGLE tooth.
    
    This bisector should point toward the gear center for a well-manufactured gear.
    
    Attributes:
        tooth: Tooth number (1-indexed)
        origin: 2D midpoint between the two flank centroids
        direction: 2D unit vector of the bisector direction (pointing inward)
        length: Visual length for rendering
    """
    tooth: int
    origin: Vector2D
    direction: Vector2D
    length: float
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "tooth": self.tooth,
            "origin": self.origin.tolist(),
            "direction": self.direction.tolist(),
            "length": self.length,
        }


class BisectorComputer:
    """Computes angle bisectors between tooth flanks.
    
    Bisectors are used to estimate the gear center: if the gear
    is perfectly manufactured, all bisectors should intersect
    at the gear's true center.
    
    There are two types of bisectors:
    1. **Tooth bisectors**: Between left and right flanks of the SAME tooth
       - These point toward the gear center
       - Used for ghost circle analysis
    
    2. **Pair bisectors**: Between flanks of ADJACENT teeth
       - Used for other types of analysis
    
    Example:
        >>> tooth_bisectors = BisectorComputer.compute_tooth_bisectors(
        ...     tooth_flanks_list, length=5.0
        ... )
    """
    
    @staticmethod
    def compute_bisector(
        point_a: Vector2D,
        dir_a: Vector2D,
        point_b: Vector2D,
        dir_b: Vector2D
    ) -> tuple[Vector2D, Vector2D]:
        """Compute the angle bisector between two lines.
        
        The bisector origin is the midpoint between the two line centers.
        The bisector direction is the normalized sum of the two line
        directions (after ensuring they point in the same general direction).
        
        Args:
            point_a: Point on first line
            dir_a: Direction of first line (will be normalized)
            point_b: Point on second line
            dir_b: Direction of second line (will be normalized)
            
        Returns:
            Tuple of (bisector_origin, bisector_direction) where both
            are 2D arrays and direction is a unit vector.
        
        Example:
            >>> p1, d1 = np.array([0, 0]), np.array([1, 0])
            >>> p2, d2 = np.array([0, 1]), np.array([0, 1])
            >>> origin, direction = BisectorComputer.compute_bisector(p1, d1, p2, d2)
            >>> origin  # Midpoint
            array([0. , 0.5])
            >>> direction  # 45 degree bisector
            array([0.70710678, 0.70710678])
        """
        dir_a = unit_vector(dir_a)
        dir_b = unit_vector(dir_b)
        
        # Ensure directions point in similar direction
        # If they point in opposite directions, flip one
        if np.dot(dir_a, dir_b) < 0:
            dir_b = -dir_b
        
        # Bisector direction is the average of the two directions
        bisector_dir = unit_vector(dir_a + dir_b)
        
        # Origin is midpoint between the two line points
        origin = 0.5 * (point_a + point_b)
        
        # Fallback for degenerate cases (parallel lines with same direction)
        if np.linalg.norm(bisector_dir) < 1e-8:
            # Use radial direction from origin
            bisector_dir = unit_vector(origin) if np.linalg.norm(origin) > 1e-10 else dir_a
        
        return origin, bisector_dir
    
    @classmethod
    def compute_tooth_bisectors(
        cls,
        tooth_flanks_list: list["ToothFlanks"],
        length: float
    ) -> list[ToothBisector]:
        """Compute bisectors between left and right flanks of EACH tooth.
        
        This is the primary method for ghost circle analysis. Each tooth
        produces one bisector that should point toward the gear center.
        
        Args:
            tooth_flanks_list: List of ToothFlanks objects (one per tooth)
            length: Visual length for bisector rendering
            
        Returns:
            List of ToothBisector objects (one per tooth)
        
        Example:
            >>> bisectors = BisectorComputer.compute_tooth_bisectors(flanks, length=5.0)
            >>> len(bisectors)  # One bisector per tooth with valid flanks
            38
        """
        bisectors: list[ToothBisector] = []
        
        for tooth_flanks in tooth_flanks_list:
            # Compute bisector between left and right flank of this tooth
            origin, direction = cls.compute_bisector(
                tooth_flanks.left_point, tooth_flanks.left_direction,
                tooth_flanks.right_point, tooth_flanks.right_direction
            )
            
            # Ensure direction points INWARD (toward gear center, i.e., toward origin)
            direction = orient_direction_inward(direction, origin)
            
            bisectors.append(ToothBisector(
                tooth=tooth_flanks.tooth,
                origin=origin,
                direction=direction,
                length=length,
            ))
        
        logger.info(f"Computed {len(bisectors)} tooth bisectors")
        
        return bisectors
    
    @classmethod
    def compute_pair_bisectors(
        cls,
        flanks: list[FlankLine],
        length: float,
        n_teeth: int
    ) -> list[PairBisector]:
        """Compute bisectors for consecutive even-odd tooth pairs with wraparound.
        
        Creates bisectors between pairs: (2-3), (4-5), (6-7), ..., (N-1)
        where the last pair wraps around from the highest even tooth to tooth 1.
        
        NOTE: This is for bisectors between DIFFERENT teeth, not for ghost circle
        analysis. For ghost circle, use compute_tooth_bisectors() instead.
        
        Args:
            flanks: List of fitted flank lines (FlankLine objects)
            length: Visual length for bisector rendering
            n_teeth: Total number of teeth in the gear
            
        Returns:
            List of PairBisector objects
        
        Example:
            >>> bisectors = BisectorComputer.compute_pair_bisectors(
            ...     flanks, length=5.0, n_teeth=38
            ... )
            >>> len(bisectors)  # One bisector per pair
            19
            >>> bisectors[0].between_teeth
            (2, 3)
        """
        if len(flanks) < 2:
            return []
        
        # Create lookup dictionary for fast access by tooth number
        flank_dict = {f.tooth: f for f in flanks}
        
        bisectors: list[PairBisector] = []
        
        # Iterate through even-odd pairs: (2,3), (4,5), ..., (N, 1)
        for tooth_num in range(2, n_teeth + 1, 2):
            # Wraparound: tooth N pairs with tooth 1
            next_tooth_num = tooth_num + 1 if tooth_num < n_teeth else 1
            
            # Check if both teeth have fitted flanks
            if tooth_num in flank_dict and next_tooth_num in flank_dict:
                current = flank_dict[tooth_num]
                nxt = flank_dict[next_tooth_num]
                
                origin, direction = cls.compute_bisector(
                    current.point, current.direction,
                    nxt.point, nxt.direction
                )
                
                bisectors.append(PairBisector(
                    between_teeth=(current.tooth, nxt.tooth),
                    origin=origin,
                    direction=direction,
                    length=length,
                ))
        
        logger.debug(f"Computed {len(bisectors)} pair bisectors")
        
        return bisectors
    
    @classmethod
    def compute_all_adjacent_bisectors(
        cls,
        flanks: list[FlankLine],
        length: float,
        n_teeth: int
    ) -> list[PairBisector]:
        """Compute bisectors for ALL adjacent tooth pairs.
        
        Unlike compute_pair_bisectors, this creates bisectors between
        every consecutive pair: (1-2), (2-3), (3-4), ..., (N-1)
        
        Args:
            flanks: List of fitted flank lines
            length: Visual length for bisector rendering
            n_teeth: Total number of teeth
            
        Returns:
            List of PairBisector objects (up to n_teeth bisectors)
        """
        if len(flanks) < 2:
            return []
        
        flank_dict = {f.tooth: f for f in flanks}
        bisectors: list[PairBisector] = []
        
        for tooth_num in range(1, n_teeth + 1):
            next_tooth_num = tooth_num + 1 if tooth_num < n_teeth else 1
            
            if tooth_num in flank_dict and next_tooth_num in flank_dict:
                current = flank_dict[tooth_num]
                nxt = flank_dict[next_tooth_num]
                
                origin, direction = cls.compute_bisector(
                    current.point, current.direction,
                    nxt.point, nxt.direction
                )
                
                bisectors.append(PairBisector(
                    between_teeth=(current.tooth, nxt.tooth),
                    origin=origin,
                    direction=direction,
                    length=length,
                ))
        
        return bisectors


class IntersectionFinder:
    """Finds intersections between geometric elements.
    
    This class provides methods for computing intersection points
    of lines, primarily used to find where bisectors meet.
    
    Example:
        >>> point = IntersectionFinder.line_intersection_2d(
        ...     np.array([0, 0]), np.array([1, 1]),
        ...     np.array([1, 0]), np.array([0, 1])
        ... )
        >>> point
        array([0.5, 0.5])
    """
    
    @staticmethod
    def line_intersection_2d(
        p1: Vector2D,
        d1: Vector2D,
        p2: Vector2D,
        d2: Vector2D
    ) -> Optional[Vector2D]:
        """Find intersection point of two 2D lines.
        
        Uses parametric line representation: L = p + t*d
        
        Args:
            p1: Point on first line
            d1: Direction of first line (should be unit vector)
            p2: Point on second line
            d2: Direction of second line (should be unit vector)
            
        Returns:
            Intersection point as 2D array, or None if lines are
            parallel (within numerical tolerance).
        
        Example:
            >>> # Two perpendicular lines
            >>> p1, d1 = np.array([0, 0]), np.array([1, 0])
            >>> p2, d2 = np.array([5, -5]), np.array([0, 1])
            >>> IntersectionFinder.line_intersection_2d(p1, d1, p2, d2)
            array([5., 0.])
        """
        # Check if lines are nearly parallel using cross product
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-10:
            return None
        
        # Solve for parameter t where: p1 + t*d1 = p2 + s*d2
        # t = ((p2-p1) × d2) / (d1 × d2)
        dp = p2 - p1
        t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
        
        return p1 + t * d1
    
    @classmethod
    def _compute_intersections_generic(
        cls,
        bisectors: list,
        r_min: float,
        r_max: float,
        parallel_threshold: float = 0.05,
        log_level: str = "debug"
    ) -> list[Vector2D]:
        """Generic method to compute pairwise intersections of bisector-like objects.
        
        Works with any objects that have 'origin' and 'direction' attributes.
        
        Args:
            bisectors: List of bisector objects (PairBisector or ToothBisector)
            r_min: Minimum radius for valid intersections
            r_max: Maximum radius for valid intersections
            parallel_threshold: Skip pairs where |dot(d1, d2)| > (1 - threshold)
            log_level: Logging level ('debug' or 'info')
            
        Returns:
            List of valid intersection points as 2D arrays
        """
        if len(bisectors) < 2:
            return []
        
        intersections: list[Vector2D] = []
        
        # Check all pairs of bisectors
        for i in range(len(bisectors)):
            for j in range(i + 1, len(bisectors)):
                b1, b2 = bisectors[i], bisectors[j]
                
                # Skip nearly parallel bisectors
                dot_product = abs(np.dot(b1.direction, b2.direction))
                if dot_product > (1.0 - parallel_threshold):
                    continue
                
                # Compute intersection
                point = cls.line_intersection_2d(
                    b1.origin, b1.direction,
                    b2.origin, b2.direction
                )
                
                if point is not None:
                    # Check if intersection is within radius bounds
                    radius = np.linalg.norm(point)
                    if r_min <= radius <= r_max:
                        intersections.append(point)
        
        msg = f"Found {len(intersections)} bisector intersections in radius [{r_min:.3f}, {r_max:.3f}]"
        if log_level == "info":
            logger.info(msg)
        else:
            logger.debug(msg)
        
        return intersections
    
    @classmethod
    def compute_bisector_intersections(
        cls,
        bisectors: list[PairBisector],
        r_min: float,
        r_max: float,
        parallel_threshold: float = 0.05
    ) -> list[Vector2D]:
        """Compute pairwise intersections of bisectors near the center.
        
        Intersects all bisector pairs and filters results to keep only
        intersections within a specified radius range from the origin.
        
        Args:
            bisectors: List of bisector lines
            r_min: Minimum radius for valid intersections
            r_max: Maximum radius for valid intersections
            parallel_threshold: Skip pairs where |dot(d1, d2)| > (1 - threshold).
                              Default 0.05 means skip pairs within ~5% of parallel.
            
        Returns:
            List of valid intersection points as 2D arrays
        
        Example:
            >>> intersections = IntersectionFinder.compute_bisector_intersections(
            ...     bisectors, r_min=0.1, r_max=1.0
            ... )
        """
        return cls._compute_intersections_generic(
            bisectors, r_min, r_max, parallel_threshold, log_level="debug"
        )
    
    @classmethod
    def compute_tooth_bisector_intersections(
        cls,
        bisectors: list[ToothBisector],
        r_min: float,
        r_max: float,
        parallel_threshold: float = 0.05
    ) -> list[Vector2D]:
        """Compute pairwise intersections of TOOTH bisectors.
        
        Same as compute_bisector_intersections but for ToothBisector objects.
        
        Args:
            bisectors: List of ToothBisector objects
            r_min: Minimum radius for valid intersections
            r_max: Maximum radius for valid intersections
            parallel_threshold: Threshold for parallel detection
            
        Returns:
            List of valid intersection points
        """
        return cls._compute_intersections_generic(
            bisectors, r_min, r_max, parallel_threshold, log_level="info"
        )
    
    @staticmethod
    def line_circle_intersection(
        origin: Vector2D,
        direction: Vector2D,
        center: Vector2D,
        radius: float
    ) -> Optional[tuple[Vector2D, Vector2D]]:
        """Find intersection points of a line with a circle.
        
        Args:
            origin: Point on the line
            direction: Direction of the line
            center: Center of the circle
            radius: Radius of the circle
            
        Returns:
            Tuple of two intersection points, or None if no intersection.
            Points are ordered by parameter t (first is "earlier" on line).
        """
        d = unit_vector(direction)
        
        # Translate so circle is at origin
        p = origin - center
        
        # Quadratic coefficients for |p + t*d|² = r²
        a = np.dot(d, d)  # Should be 1 for unit vector
        b = 2 * np.dot(p, d)
        c = np.dot(p, p) - radius**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        point1 = origin + t1 * d
        point2 = origin + t2 * d
        
        return (point1, point2)