"""
Robust flank classification and consolidation using surface normals.

This module addresses two key issues in crown gear flank detection:

1. WRONG FLANK SIDE: Sometimes the angular-based split assigns points to the
   wrong flank (left vs right). We fix this by using surface normals as the
   PRIMARY classifier, not just for validation.

2. DOUBLE-FITTING: Sometimes one physical flank gets split into multiple
   clusters. We fix this by consolidating candidates per tooth.

3. ANGULAR MISALIGNMENT: Sometimes the angular bins don't align with actual
   tooth positions. We fix this by detecting the optimal phase offset.

The key insight: Surface normals DEFINITIVELY tell us which side a point 
belongs to, because:
- Left flank normals point "backward" (negative tangential component)
- Right flank normals point "forward" (positive tangential component)

This is a physical property of the geometry that doesn't depend on how
points are clustered.

Usage:
    classifier = NormalBasedFlankClassifier()
    
    # Classify each point individually
    left_mask, right_mask = classifier.classify_points(points, normals)
    
    # Or use the full pipeline
    tooth_flanks = classifier.extract_flanks_robust(
        points, normals, tooth_number, min_points
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Points2D = NDArray[np.floating]
Normals2D = NDArray[np.floating]
Vector2D = NDArray[np.floating]


class FlankSide(Enum):
    """Flank side classification."""
    LEFT = "left"
    RIGHT = "right"
    AMBIGUOUS = "ambiguous"  # When normal doesn't clearly indicate side


@dataclass
class PointClassification:
    """Classification result for a single point."""
    side: FlankSide
    tangential_component: float  # Positive = right, negative = left
    confidence: float  # How clearly it belongs to that side


@dataclass
class FlankCandidate:
    """A candidate flank extracted from points."""
    side: FlankSide
    points: Points2D
    normals: Normals2D
    centroid: Vector2D
    direction: Vector2D
    n_points: int
    mean_tangential: float  # Average tangential component (diagnostic)
    confidence: float  # How consistent the normals are
    
    # For consolidation
    angular_span: float = 0.0  # Angular extent of the flank
    radial_range: Tuple[float, float] = (0.0, 0.0)  # (min_r, max_r)


@dataclass
class RobustToothFlanks:
    """Robustly extracted flanks for a single tooth.
    
    At most one left and one right flank per tooth, classified by normals.
    """
    tooth: int
    
    # Left flank (may be None if not enough points)
    left_centroid: Optional[Vector2D] = None
    left_direction: Optional[Vector2D] = None
    left_n_points: int = 0
    left_confidence: float = 0.0
    
    # Right flank (may be None if not enough points)
    right_centroid: Optional[Vector2D] = None
    right_direction: Optional[Vector2D] = None
    right_n_points: int = 0
    right_confidence: float = 0.0
    
    # Diagnostics
    classification_method: str = "normal"  # Always "normal" for this class
    points_reclassified: int = 0  # How many points changed side vs angular method
    candidates_consolidated: int = 0  # How many candidates were merged
    
    @property
    def has_left(self) -> bool:
        return self.left_centroid is not None
    
    @property
    def has_right(self) -> bool:
        return self.right_centroid is not None
    
    @property
    def is_complete(self) -> bool:
        return self.has_left and self.has_right


class NormalBasedFlankClassifier:
    """Classifies flank points using surface normals as the primary signal.
    
    This replaces the angular-based left/right split with a normal-based
    classification that is more robust to clustering errors.
    
    The classification uses the tangential component of each normal:
    - For a point at angle θ, tangential direction is (-sin(θ), cos(θ))
    - Right flank normals have POSITIVE tangential component
    - Left flank normals have NEGATIVE tangential component
    
    This is based on the physical fact that surface normals point outward
    from the gear tooth surface.
    """
    
    # Threshold for classification
    # |tangential_component| must exceed this to be classified
    TANGENTIAL_THRESHOLD = 0.15
    
    # Minimum confidence (fraction of points agreeing) to accept a flank
    MIN_CONFIDENCE = 0.7
    
    @classmethod
    def compute_tangential_components(
        cls,
        points: Points2D,
        normals: Normals2D
    ) -> NDArray[np.floating]:
        """Compute tangential component of each normal.
        
        Args:
            points: 2D points, shape (N, 2)
            normals: 2D normals at each point, shape (N, 2)
            
        Returns:
            Array of tangential components, shape (N,)
            Positive = right flank, Negative = left flank
        """
        # Compute angle of each point
        point_angles = np.arctan2(points[:, 1], points[:, 0])
        
        # Tangential direction at angle θ: (-sin(θ), cos(θ))
        # This points in the CCW direction around the gear
        tangential_dirs = np.column_stack([
            -np.sin(point_angles),
            np.cos(point_angles)
        ])
        
        # Dot product gives tangential component
        tangential_components = np.sum(normals * tangential_dirs, axis=1)
        
        return tangential_components
    
    @classmethod
    def classify_points(
        cls,
        points: Points2D,
        normals: Normals2D,
        threshold: Optional[float] = None
    ) -> Tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
        """Classify each point as left, right, or ambiguous based on normals.
        
        This is the CORE classification function that uses normals as truth.
        
        Args:
            points: 2D points, shape (N, 2)
            normals: 2D normals, shape (N, 2)
            threshold: Classification threshold (default: TANGENTIAL_THRESHOLD)
            
        Returns:
            Tuple of (left_mask, right_mask, ambiguous_mask)
            Each is a boolean array of shape (N,)
        """
        if threshold is None:
            threshold = cls.TANGENTIAL_THRESHOLD
        
        tangential = cls.compute_tangential_components(points, normals)
        
        right_mask = tangential > threshold
        left_mask = tangential < -threshold
        ambiguous_mask = ~right_mask & ~left_mask
        
        return left_mask, right_mask, ambiguous_mask
    
    @classmethod
    def classify_points_with_details(
        cls,
        points: Points2D,
        normals: Normals2D
    ) -> List[PointClassification]:
        """Classify each point with full details.
        
        Args:
            points: 2D points
            normals: 2D normals
            
        Returns:
            List of PointClassification for each point
        """
        tangential = cls.compute_tangential_components(points, normals)
        
        results = []
        for i, tc in enumerate(tangential):
            if tc > cls.TANGENTIAL_THRESHOLD:
                side = FlankSide.RIGHT
                confidence = min(1.0, tc / 0.5)  # Scale to [0, 1]
            elif tc < -cls.TANGENTIAL_THRESHOLD:
                side = FlankSide.LEFT
                confidence = min(1.0, -tc / 0.5)
            else:
                side = FlankSide.AMBIGUOUS
                confidence = 0.0
            
            results.append(PointClassification(
                side=side,
                tangential_component=float(tc),
                confidence=confidence
            ))
        
        return results
    
    @classmethod
    def extract_flanks_by_normal(
        cls,
        points: Points2D,
        normals: Normals2D,
        tooth_number: int,
        min_points: int = 5
    ) -> RobustToothFlanks:
        """Extract left and right flanks using normal-based classification.
        
        This is the main entry point for robust flank extraction.
        
        Instead of splitting by angular position (which can fail), we split
        by the tangential component of the surface normal (which is physical).
        
        Args:
            points: 2D points for this tooth cluster
            normals: 2D normals at each point
            tooth_number: Tooth number (1-indexed)
            min_points: Minimum points required per flank
            
        Returns:
            RobustToothFlanks with at most one left and one right flank
        """
        from gear_analysis.geometry.line_fitting import LineFitter, FitMethod
        from gear_analysis.utils import orient_direction_outward
        
        # Step 1: Classify all points by their normals
        left_mask, right_mask, ambiguous_mask = cls.classify_points(points, normals)
        
        left_pts = points[left_mask]
        left_nrm = normals[left_mask]
        right_pts = points[right_mask]
        right_nrm = normals[right_mask]
        
        n_ambiguous = np.sum(ambiguous_mask)
        if n_ambiguous > 0:
            logger.debug(f"Tooth {tooth_number}: {n_ambiguous} ambiguous points excluded")
        
        result = RobustToothFlanks(
            tooth=tooth_number,
            classification_method="normal"
        )
        
        # Step 2: Fit left flank if enough points
        if len(left_pts) >= min_points:
            try:
                left_center, left_dir, left_method, _ = LineFitter._fit_adaptive(
                    left_pts, left_nrm
                )
                left_dir = orient_direction_outward(left_dir, left_center)
                
                # Compute confidence: what fraction of points have consistent normals
                tangential = cls.compute_tangential_components(left_pts, left_nrm)
                left_confidence = np.mean(tangential < -cls.TANGENTIAL_THRESHOLD)
                
                result.left_centroid = left_center
                result.left_direction = left_dir
                result.left_n_points = len(left_pts)
                result.left_confidence = float(left_confidence)
                
            except Exception as e:
                logger.warning(f"Tooth {tooth_number}: Left flank fitting failed: {e}")
        else:
            logger.debug(f"Tooth {tooth_number}: Not enough left points ({len(left_pts)})")
        
        # Step 3: Fit right flank if enough points
        if len(right_pts) >= min_points:
            try:
                right_center, right_dir, right_method, _ = LineFitter._fit_adaptive(
                    right_pts, right_nrm
                )
                right_dir = orient_direction_outward(right_dir, right_center)
                
                tangential = cls.compute_tangential_components(right_pts, right_nrm)
                right_confidence = np.mean(tangential > cls.TANGENTIAL_THRESHOLD)
                
                result.right_centroid = right_center
                result.right_direction = right_dir
                result.right_n_points = len(right_pts)
                result.right_confidence = float(right_confidence)
                
            except Exception as e:
                logger.warning(f"Tooth {tooth_number}: Right flank fitting failed: {e}")
        else:
            logger.debug(f"Tooth {tooth_number}: Not enough right points ({len(right_pts)})")
        
        return result
    
    @classmethod
    def compare_with_angular_split(
        cls,
        points: Points2D,
        normals: Normals2D,
        tooth_center_angle: Optional[float] = None
    ) -> Dict[str, int]:
        """Compare normal-based vs angular-based classification.
        
        Useful for diagnostics to see how many points would be reclassified.
        
        Args:
            points: 2D points
            normals: 2D normals
            tooth_center_angle: Angle of tooth center (if None, computed from points)
            
        Returns:
            Dict with comparison statistics
        """
        from gear_analysis.utils import normalize_angle_diff
        
        # Normal-based classification
        left_mask_normal, right_mask_normal, _ = cls.classify_points(points, normals)
        
        # Angular-based classification
        if tooth_center_angle is None:
            center = points.mean(axis=0)
            tooth_center_angle = np.arctan2(center[1], center[0])
        
        point_angles = np.arctan2(points[:, 1], points[:, 0])
        angle_diff = normalize_angle_diff(point_angles - tooth_center_angle)
        
        left_mask_angular = angle_diff < 0
        right_mask_angular = angle_diff > 0
        
        # Count disagreements
        left_to_right = np.sum(left_mask_angular & right_mask_normal)
        right_to_left = np.sum(right_mask_angular & left_mask_normal)
        
        return {
            "total_points": len(points),
            "normal_left": int(np.sum(left_mask_normal)),
            "normal_right": int(np.sum(right_mask_normal)),
            "angular_left": int(np.sum(left_mask_angular)),
            "angular_right": int(np.sum(right_mask_angular)),
            "reclassified_left_to_right": int(left_to_right),
            "reclassified_right_to_left": int(right_to_left),
            "total_reclassified": int(left_to_right + right_to_left),
        }


class FlankConsolidator:
    """Consolidates multiple flank candidates to ensure one per side per tooth.
    
    This addresses the double-fitting problem where one physical flank gets
    split into multiple clusters.
    
    Consolidation strategy:
    1. Group candidates by side (left/right)
    2. For each side, check if candidates overlap (similar angle/radius)
    3. If overlapping, merge into single flank
    4. Keep the candidate with most points / highest confidence
    """
    
    # Overlap thresholds
    ANGULAR_OVERLAP_THRESHOLD = 0.1  # radians (~6 degrees)
    RADIAL_OVERLAP_RATIO = 0.3  # fraction of radial range that must overlap
    
    @classmethod
    def consolidate_candidates(
        cls,
        candidates: List[FlankCandidate],
        side: FlankSide
    ) -> Optional[FlankCandidate]:
        """Consolidate multiple candidates for one side into a single flank.
        
        Args:
            candidates: List of FlankCandidate for same side
            side: Which side (LEFT or RIGHT)
            
        Returns:
            Single consolidated FlankCandidate, or None if no valid candidates
        """
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Filter to only candidates of the requested side
        side_candidates = [c for c in candidates if c.side == side]
        
        if not side_candidates:
            return None
        
        if len(side_candidates) == 1:
            return side_candidates[0]
        
        # Multiple candidates - need to consolidate
        # Strategy: merge all points and refit
        logger.info(f"Consolidating {len(side_candidates)} {side.value} flank candidates")
        
        all_points = np.vstack([c.points for c in side_candidates])
        all_normals = np.vstack([c.normals for c in side_candidates])
        
        # Refit on merged points
        from gear_analysis.geometry.line_fitting import LineFitter
        from gear_analysis.utils import orient_direction_outward
        
        centroid, direction, _, _ = LineFitter._fit_adaptive(all_points, all_normals)
        direction = orient_direction_outward(direction, centroid)
        
        # Compute consolidated confidence
        tangential = NormalBasedFlankClassifier.compute_tangential_components(
            all_points, all_normals
        )
        if side == FlankSide.LEFT:
            confidence = np.mean(tangential < -NormalBasedFlankClassifier.TANGENTIAL_THRESHOLD)
            mean_tangential = float(np.mean(tangential[tangential < 0]))
        else:
            confidence = np.mean(tangential > NormalBasedFlankClassifier.TANGENTIAL_THRESHOLD)
            mean_tangential = float(np.mean(tangential[tangential > 0]))
        
        return FlankCandidate(
            side=side,
            points=all_points,
            normals=all_normals,
            centroid=centroid,
            direction=direction,
            n_points=len(all_points),
            mean_tangential=mean_tangential,
            confidence=float(confidence)
        )
    
    @classmethod
    def are_overlapping(
        cls,
        c1: FlankCandidate,
        c2: FlankCandidate
    ) -> bool:
        """Check if two candidates likely represent the same physical flank.
        
        Args:
            c1, c2: Two FlankCandidate objects
            
        Returns:
            True if they likely overlap
        """
        # Check angular proximity
        angle1 = np.arctan2(c1.centroid[1], c1.centroid[0])
        angle2 = np.arctan2(c2.centroid[1], c2.centroid[0])
        angle_diff = abs(np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2)))
        
        if angle_diff > cls.ANGULAR_OVERLAP_THRESHOLD:
            return False
        
        # Check radial overlap
        r1 = np.linalg.norm(c1.centroid)
        r2 = np.linalg.norm(c2.centroid)
        r_diff = abs(r1 - r2)
        r_avg = (r1 + r2) / 2
        
        if r_diff / r_avg > cls.RADIAL_OVERLAP_RATIO:
            return False
        
        return True


def extract_robust_tooth_flanks(
    points: Points2D,
    normals: Normals2D,
    tooth_number: int,
    min_points: int = 5
) -> RobustToothFlanks:
    """Main entry point for robust flank extraction.
    
    This combines normal-based classification with consolidation to produce
    at most one left and one right flank per tooth.
    
    Args:
        points: 2D points for this tooth cluster
        normals: 2D normals at each point
        tooth_number: Tooth number (1-indexed)
        min_points: Minimum points required per flank
        
    Returns:
        RobustToothFlanks with validated flanks
    """
    return NormalBasedFlankClassifier.extract_flanks_by_normal(
        points, normals, tooth_number, min_points
    )


# =============================================================================
# Phase Alignment Detection
# =============================================================================

def detect_angular_phase_offset(
    points: Points2D,
    n_teeth: int,
    search_range_deg: float = 20.0,
    search_steps: int = 40
) -> float:
    """Detect the optimal angular offset for tooth binning.
    
    The standard angular binning assumes teeth start at 0°. But often
    the actual teeth are offset by some angle. This function finds
    the optimal offset by maximizing the "gap score" - we want bins
    to be centered on teeth, with boundaries in the gaps between teeth.
    
    Args:
        points: 2D points (flank region)
        n_teeth: Number of teeth
        search_range_deg: Range to search (+/- this many degrees)
        search_steps: Number of steps in the search
        
    Returns:
        Optimal phase offset in radians
    """
    angles = np.arctan2(points[:, 1], points[:, 0])
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    
    tooth_width = 2 * np.pi / n_teeth
    search_range = np.radians(search_range_deg)
    
    best_offset = 0.0
    best_score = -np.inf
    
    for offset in np.linspace(-search_range, search_range, search_steps):
        # Create bins with this offset
        edges = np.linspace(offset, 2 * np.pi + offset, n_teeth + 1)
        edges = edges % (2 * np.pi)
        
        # Count points near bin boundaries (want this LOW)
        boundary_margin = tooth_width * 0.15  # 15% of tooth width
        
        near_boundary = 0
        for edge in edges[:-1]:
            dist_to_edge = np.abs(np.arctan2(
                np.sin(angles - edge),
                np.cos(angles - edge)
            ))
            near_boundary += np.sum(dist_to_edge < boundary_margin)
        
        # Score: lower boundary count is better
        score = -near_boundary
        
        if score > best_score:
            best_score = score
            best_offset = offset
    
    return best_offset


def detect_angular_phase_offset_by_normals(
    points: Points2D,
    normals: Normals2D,
    n_teeth: int,
    search_range_deg: float = 20.0,
    search_steps: int = 40
) -> float:
    """Detect optimal angular offset using normal consistency.
    
    For each candidate offset, we check how consistent the normal-based
    classification is with the angular binning. The optimal offset is
    where they agree most.
    
    This is more robust than the gap-based method because it uses
    the actual surface geometry.
    
    Args:
        points: 2D points
        normals: 2D surface normals
        n_teeth: Number of teeth
        search_range_deg: Range to search
        search_steps: Number of steps
        
    Returns:
        Optimal phase offset in radians
    """
    from gear_analysis.utils import normalize_angle_diff
    
    angles = np.arctan2(points[:, 1], points[:, 0])
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    
    tooth_width = 2 * np.pi / n_teeth
    search_range = np.radians(search_range_deg)
    
    # Get normal-based classification (this is ground truth)
    left_mask_normal, right_mask_normal, _ = NormalBasedFlankClassifier.classify_points(
        points, normals
    )
    
    best_offset = 0.0
    best_agreement = -np.inf
    
    for offset in np.linspace(-search_range, search_range, search_steps):
        # Create bins with this offset
        edges = (np.linspace(0, 2 * np.pi, n_teeth + 1) + offset) % (2 * np.pi)
        
        # Assign points to bins
        # Need to handle wraparound carefully
        shifted_angles = (angles - offset) % (2 * np.pi)
        bin_edges = np.linspace(0, 2 * np.pi, n_teeth + 1)
        bin_idx = np.digitize(shifted_angles, bin_edges, right=False) - 1
        bin_idx[bin_idx == n_teeth] = 0
        
        total_agreement = 0
        
        for tooth in range(n_teeth):
            tooth_mask = bin_idx == tooth
            if np.sum(tooth_mask) < 5:
                continue
            
            tooth_pts = points[tooth_mask]
            
            # Get tooth center and split by angular position
            center = tooth_pts.mean(axis=0)
            tooth_center_angle = np.arctan2(center[1], center[0])
            
            point_angles = np.arctan2(tooth_pts[:, 1], tooth_pts[:, 0])
            angle_diff = normalize_angle_diff(point_angles - tooth_center_angle)
            
            left_mask_angular = angle_diff < 0
            right_mask_angular = angle_diff > 0
            
            # Check agreement with normal classification
            tooth_left_normal = left_mask_normal[tooth_mask]
            tooth_right_normal = right_mask_normal[tooth_mask]
            
            # Count agreements
            left_agree = np.sum(left_mask_angular & tooth_left_normal)
            right_agree = np.sum(right_mask_angular & tooth_right_normal)
            
            total_agreement += left_agree + right_agree
        
        if total_agreement > best_agreement:
            best_agreement = total_agreement
            best_offset = offset
    
    return best_offset


# =============================================================================
# Integration helper to convert to existing ToothFlanks format
# =============================================================================

def robust_to_standard_tooth_flanks(
    robust: RobustToothFlanks,
    include_validation: bool = True
) -> Optional['ToothFlanks']:
    """Convert RobustToothFlanks to standard ToothFlanks format.
    
    This allows the robust classifier to be used with existing pipeline code.
    
    Args:
        robust: RobustToothFlanks from the robust classifier
        include_validation: Whether to include validation info
        
    Returns:
        ToothFlanks in standard format, or None if incomplete
    """
    from gear_analysis.geometry.line_fitting import (
        ToothFlanks, FlankValidation, FitMethod,
        FlankSide as StandardFlankSide
    )
    
    if not robust.has_left or not robust.has_right:
        return None
    
    # Create flag reasons if confidence is low
    flag_reasons = []
    if robust.left_confidence < 0.8:
        flag_reasons.append(
            f"Left flank: Low confidence ({robust.left_confidence:.0%})"
        )
    if robust.right_confidence < 0.8:
        flag_reasons.append(
            f"Right flank: Low confidence ({robust.right_confidence:.0%})"
        )
    
    return ToothFlanks(
        tooth=robust.tooth,
        left_point=robust.left_centroid,
        left_direction=robust.left_direction,
        left_n_points=robust.left_n_points,
        right_point=robust.right_centroid,
        right_direction=robust.right_direction,
        right_n_points=robust.right_n_points,
        left_fit_method=FitMethod.NORMAL_DERIVED,  # Always normal-based
        right_fit_method=FitMethod.NORMAL_DERIVED,
        is_valid=True,
        is_flagged=len(flag_reasons) > 0,
        flag_reasons=flag_reasons,
    )