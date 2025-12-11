"""
Flank detection using radial tracing.

This module implements a new approach to flank detection:
1. Start at outer radius (tooth tip)
2. Trace inward following aligned points
3. Stop when alignment breaks

This is more robust than angular binning because it follows
the actual geometry of the tooth flanks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Points2D = NDArray[np.floating]
Vector2D = NDArray[np.floating]


@dataclass
class TracedFlank:
    """A flank detected by radial tracing."""
    points: Points2D          # All points in this flank
    start_point: Vector2D     # Outermost point (where tracing started)
    end_point: Vector2D       # Innermost point (where tracing stopped)
    direction: Vector2D       # Fitted line direction (unit vector)
    centroid: Vector2D        # Center of the flank line
    n_points: int
    
    # Quality metrics
    linearity_score: float    # How well points fit a line (0-1, higher=better)
    radial_span: float        # Distance from start to end


@dataclass 
class TracedTooth:
    """Both flanks of a tooth, detected by radial tracing."""
    tooth_id: int
    tip_point: Vector2D       # The outermost point of the tooth
    left_flank: Optional[TracedFlank]
    right_flank: Optional[TracedFlank]
    is_valid: bool = True
    issue: str = ""


class RadialFlankTracer:
    """Detects flanks by tracing from outer radius inward.
    
    Algorithm:
    1. Find tooth tips (local maxima at outer radius)
    2. From each tip, trace LEFT and RIGHT flanks
    3. Tracing follows points that are:
       - More inward (smaller radius)
       - Aligned with current direction (angle threshold)
    4. Stop when no aligned point found
    
    This approach is more robust than angular binning because
    it follows the actual geometry of the flanks.
    """
    
    def __init__(
        self,
        angle_threshold_deg: float = 10.0,
        min_points_per_flank: int = 5,
        radial_step_factor: float = 0.1,
        search_angle_deg: float = 45.0,
    ):
        """Initialize the tracer.
        
        Args:
            angle_threshold_deg: Maximum angle deviation to consider points aligned
            min_points_per_flank: Minimum points needed for a valid flank
            radial_step_factor: How far inward to look for next point (fraction of radius)
            search_angle_deg: Angular range to search for next point (each side)
        """
        self.angle_threshold = np.radians(angle_threshold_deg)
        self.min_points = min_points_per_flank
        self.radial_step_factor = radial_step_factor
        self.search_angle = np.radians(search_angle_deg)
    
    def find_tooth_tips(
        self,
        points: Points2D,
        n_teeth: int,
        r_outer: float,
        tip_radius_fraction: float = 0.95
    ) -> List[Vector2D]:
        """Find the outermost points of each tooth (tooth tips).
        
        Args:
            points: All filtered points
            n_teeth: Expected number of teeth
            r_outer: Outer radius
            tip_radius_fraction: Points above this fraction of r_outer are candidates
            
        Returns:
            List of tip points, one per tooth
        """
        # Get points near outer radius (potential tips)
        radii = np.linalg.norm(points, axis=1)
        tip_threshold = r_outer * tip_radius_fraction
        tip_candidates = points[radii >= tip_threshold]
        
        if len(tip_candidates) == 0:
            logger.warning("No points found near outer radius")
            return []
        
        # Compute angles of tip candidates
        angles = np.arctan2(tip_candidates[:, 1], tip_candidates[:, 0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        
        # Divide into angular sectors and find the outermost point in each
        sector_width = 2 * np.pi / n_teeth
        tips = []
        
        for i in range(n_teeth):
            sector_start = i * sector_width
            sector_end = (i + 1) * sector_width
            
            # Find points in this sector
            if sector_end <= 2 * np.pi:
                mask = (angles >= sector_start) & (angles < sector_end)
            else:
                # Handle wraparound
                mask = (angles >= sector_start) | (angles < (sector_end - 2 * np.pi))
            
            sector_points = tip_candidates[mask]
            
            if len(sector_points) > 0:
                # Find outermost point in sector
                sector_radii = np.linalg.norm(sector_points, axis=1)
                tip_idx = np.argmax(sector_radii)
                tips.append(sector_points[tip_idx])
        
        logger.info(f"Found {len(tips)} tooth tips out of {n_teeth} expected")
        return tips
    
    def trace_flank(
        self,
        points: Points2D,
        start_point: Vector2D,
        direction: str,  # 'left' or 'right'
        r_inner: float,
    ) -> Optional[TracedFlank]:
        """Trace a single flank from start_point inward.
        
        Args:
            points: All available points
            start_point: Starting point (tooth tip)
            direction: 'left' or 'right' - which side of the tooth
            r_inner: Stop tracing when reaching this radius
            
        Returns:
            TracedFlank object or None if tracing failed
        """
        # Initialize
        flank_points = [start_point.copy()]
        current_point = start_point.copy()
        current_radius = np.linalg.norm(current_point)
        current_angle = np.arctan2(current_point[1], current_point[0])
        
        # Initial direction estimate: radially inward with slight angular offset
        if direction == 'left':
            angular_bias = -self.search_angle * 0.5  # Search more to the left
        else:
            angular_bias = self.search_angle * 0.5   # Search more to the right
        
        # Track the evolving flank direction
        flank_direction = None
        
        # Create a set of used point indices to avoid revisiting
        point_distances = np.linalg.norm(points - start_point, axis=1)
        used_mask = point_distances < 1e-10  # Mark start point as used
        
        # Trace until we reach inner radius or can't find next point
        max_iterations = 1000  # Safety limit
        iteration = 0
        
        while current_radius > r_inner and iteration < max_iterations:
            iteration += 1
            
            # Find candidate points for next step
            next_point, next_idx = self._find_next_point(
                points, current_point, current_angle, direction,
                flank_direction, used_mask, r_inner
            )
            
            if next_point is None:
                break  # No valid next point found
            
            # Update flank direction based on accumulated points
            if len(flank_points) >= 2:
                # Direction from first to current point
                flank_direction = next_point - flank_points[0]
                flank_direction = flank_direction / (np.linalg.norm(flank_direction) + 1e-10)
            
            # Add point and update state
            flank_points.append(next_point)
            used_mask[next_idx] = True
            current_point = next_point
            current_radius = np.linalg.norm(current_point)
            current_angle = np.arctan2(current_point[1], current_point[0])
        
        # Check if we have enough points
        if len(flank_points) < self.min_points:
            logger.debug(f"Flank tracing stopped: only {len(flank_points)} points (need {self.min_points})")
            return None
        
        # Convert to array and fit line
        flank_points_array = np.array(flank_points)
        centroid, fitted_direction, linearity = self._fit_line(flank_points_array)
        
        # Orient direction to point outward
        radial = centroid / (np.linalg.norm(centroid) + 1e-10)
        if np.dot(fitted_direction, radial) < 0:
            fitted_direction = -fitted_direction
        
        return TracedFlank(
            points=flank_points_array,
            start_point=flank_points_array[0],
            end_point=flank_points_array[-1],
            direction=fitted_direction,
            centroid=centroid,
            n_points=len(flank_points_array),
            linearity_score=linearity,
            radial_span=np.linalg.norm(flank_points_array[0] - flank_points_array[-1])
        )
    
    def _find_next_point(
        self,
        points: Points2D,
        current: Vector2D,
        current_angle: float,
        direction: str,
        flank_direction: Optional[Vector2D],
        used_mask: NDArray[np.bool_],
        r_inner: float,
    ) -> Tuple[Optional[Vector2D], Optional[int]]:
        """Find the next point to add to the flank.
        
        Criteria:
        1. Not already used
        2. More inward (smaller radius) than current
        3. In the correct angular direction (left or right)
        4. Aligned with current flank direction (if established)
        """
        current_radius = np.linalg.norm(current)
        
        # Compute properties of all points
        radii = np.linalg.norm(points, axis=1)
        angles = np.arctan2(points[:, 1], points[:, 0])
        
        # Angular difference from current point
        angle_diff = angles - current_angle
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # Normalize to [-π, π]
        
        # Build candidate mask
        candidates = np.ones(len(points), dtype=bool)
        
        # 1. Not already used
        candidates &= ~used_mask
        
        # 2. More inward (but not too far)
        min_radius = max(r_inner * 0.95, current_radius - current_radius * self.radial_step_factor * 3)
        candidates &= (radii < current_radius - 1e-6)  # Must be more inward
        candidates &= (radii > min_radius)  # But not too far inward in one step
        
        # 3. Correct angular direction
        if direction == 'left':
            # Left means decreasing angle (clockwise when viewed from above)
            candidates &= (angle_diff < 0) & (angle_diff > -self.search_angle)
        else:
            # Right means increasing angle (counterclockwise)
            candidates &= (angle_diff > 0) & (angle_diff < self.search_angle)
        
        # 4. Aligned with flank direction (if we have one)
        if flank_direction is not None and np.sum(candidates) > 0:
            # Check alignment for all candidates
            for i in np.where(candidates)[0]:
                vec_to_candidate = points[i] - current
                vec_to_candidate = vec_to_candidate / (np.linalg.norm(vec_to_candidate) + 1e-10)
                
                # Angle between flank direction and vector to candidate
                dot = np.clip(np.dot(flank_direction, vec_to_candidate), -1, 1)
                angle_deviation = np.arccos(np.abs(dot))  # Use abs because direction can be flipped
                
                if angle_deviation > self.angle_threshold:
                    candidates[i] = False
        
        # Find the best candidate (closest to current point)
        if np.sum(candidates) == 0:
            return None, None
        
        candidate_indices = np.where(candidates)[0]
        distances = np.linalg.norm(points[candidate_indices] - current, axis=1)
        best_local_idx = np.argmin(distances)
        best_idx = candidate_indices[best_local_idx]
        
        return points[best_idx].copy(), best_idx
    
    def _fit_line(self, points: Points2D) -> Tuple[Vector2D, Vector2D, float]:
        """Fit a line to points using SVD.
        
        Returns:
            Tuple of (centroid, direction, linearity_score)
            linearity_score is ratio of first to second singular value (higher = more linear)
        """
        centroid = points.mean(axis=0)
        centered = points - centroid
        
        if len(points) < 2:
            return centroid, np.array([1.0, 0.0]), 0.0
        
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Linearity score: ratio of singular values
        # High ratio means points are very linear
        if len(s) >= 2 and s[1] > 1e-10:
            linearity = s[0] / s[1]
        else:
            linearity = float('inf')
        
        # Normalize to 0-1 range (cap at 100)
        linearity_score = min(linearity / 100.0, 1.0)
        
        return centroid, direction, linearity_score
    
    def trace_all_teeth(
        self,
        points: Points2D,
        n_teeth: int,
        r_inner: float,
        r_outer: float,
    ) -> List[TracedTooth]:
        """Trace flanks for all teeth.
        
        Args:
            points: All filtered points
            n_teeth: Expected number of teeth
            r_inner: Inner radius
            r_outer: Outer radius
            
        Returns:
            List of TracedTooth objects
        """
        # Step 1: Find tooth tips
        tips = self.find_tooth_tips(points, n_teeth, r_outer)
        
        if len(tips) == 0:
            logger.error("No tooth tips found!")
            return []
        
        # Step 2: Trace flanks from each tip
        teeth = []
        
        for i, tip in enumerate(tips):
            tooth_id = i + 1
            
            # Trace left flank
            left_flank = self.trace_flank(points, tip, 'left', r_inner)
            
            # Trace right flank
            right_flank = self.trace_flank(points, tip, 'right', r_inner)
            
            # Validate
            is_valid = True
            issue = ""
            
            if left_flank is None and right_flank is None:
                is_valid = False
                issue = "No flanks traced"
            elif left_flank is None:
                is_valid = False
                issue = "Left flank not found"
            elif right_flank is None:
                is_valid = False
                issue = "Right flank not found"
            else:
                # Check for orthogonal flanks (bad)
                dot = abs(np.dot(left_flank.direction, right_flank.direction))
                if dot < 0.25:  # > 75 degrees
                    is_valid = False
                    angle = np.degrees(np.arccos(dot))
                    issue = f"Flanks orthogonal ({angle:.0f}°)"
            
            teeth.append(TracedTooth(
                tooth_id=tooth_id,
                tip_point=tip,
                left_flank=left_flank,
                right_flank=right_flank,
                is_valid=is_valid,
                issue=issue
            ))
            
            if not is_valid:
                logger.warning(f"Tooth {tooth_id}: {issue}")
        
        valid_count = sum(1 for t in teeth if t.is_valid)
        logger.info(f"Traced {valid_count}/{len(teeth)} valid teeth")
        
        return teeth


def convert_to_tooth_flanks(traced_teeth: List[TracedTooth]) -> List:
    """Convert TracedTooth objects to ToothFlanks for compatibility.
    
    This allows the new tracing method to work with existing pipeline.
    """
    from gear_analysis.geometry.line_fitting import ToothFlanks
    
    result = []
    for tooth in traced_teeth:
        if tooth.left_flank is None or tooth.right_flank is None:
            continue
        
        result.append(ToothFlanks(
            tooth=tooth.tooth_id,
            left_point=tooth.left_flank.centroid,
            left_direction=tooth.left_flank.direction,
            left_n_points=tooth.left_flank.n_points,
            right_point=tooth.right_flank.centroid,
            right_direction=tooth.right_flank.direction,
            right_n_points=tooth.right_flank.n_points,
            is_valid=tooth.is_valid,
            issue=tooth.issue,
        ))
    
    return result


# ============================================================
# Visualization helper for debugging the tracing
# ============================================================

def visualize_tracing(
    points: Points2D,
    traced_teeth: List[TracedTooth],
    r_inner: float,
    r_outer: float,
    output_path: Optional[str] = None
):
    """Visualize the tracing results for debugging."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Plot all points in light gray
    ax.scatter(points[:, 0], points[:, 1], s=1, c='lightgray', alpha=0.3)
    
    # Plot reference circles
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(r_inner * np.cos(theta), r_inner * np.sin(theta), 'b--', alpha=0.3)
    ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), 'b--', alpha=0.3)
    
    # Plot each tooth
    for tooth in traced_teeth:
        color = 'green' if tooth.is_valid else 'red'
        
        # Plot tip
        ax.scatter([tooth.tip_point[0]], [tooth.tip_point[1]], 
                  s=100, c=color, marker='^', zorder=10)
        ax.annotate(str(tooth.tooth_id), tooth.tip_point, fontsize=8,
                   ha='center', va='bottom')
        
        # Plot left flank
        if tooth.left_flank is not None:
            ax.scatter(tooth.left_flank.points[:, 0], tooth.left_flank.points[:, 1],
                      s=10, c='blue', alpha=0.7)
            # Draw fitted line
            c = tooth.left_flank.centroid
            d = tooth.left_flank.direction
            length = tooth.left_flank.radial_span * 0.6
            ax.plot([c[0] - d[0]*length, c[0] + d[0]*length],
                   [c[1] - d[1]*length, c[1] + d[1]*length],
                   'b-', linewidth=2)
        
        # Plot right flank
        if tooth.right_flank is not None:
            ax.scatter(tooth.right_flank.points[:, 0], tooth.right_flank.points[:, 1],
                      s=10, c='orange', alpha=0.7)
            # Draw fitted line
            c = tooth.right_flank.centroid
            d = tooth.right_flank.direction
            length = tooth.right_flank.radial_span * 0.6
            ax.plot([c[0] - d[0]*length, c[0] + d[0]*length],
                   [c[1] - d[1]*length, c[1] + d[1]*length],
                   'orange', linewidth=2)
    
    ax.set_aspect('equal')
    ax.set_title(f'Radial Flank Tracing\n'
                f'Valid teeth: {sum(1 for t in traced_teeth if t.is_valid)}/{len(traced_teeth)}')
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved tracing visualization to: {output_path}")
    
    plt.show()
    plt.close()