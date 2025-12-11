"""
Line fitting module with surface normal validation.

This module provides line fitting for gear tooth flanks with:
1. SVD-based line fitting (robust, well-established method)
2. Normal-based flank side classification (new, uses surface geometry)
3. Validation by comparing SVD vs normal-based methods
4. Flagging of inconsistent flanks for review

The module maintains backward compatibility while adding new capabilities.

Valid tooth shapes:
- Parallel flanks (square teeth)
- V-shaped flanks (tapered teeth)  
- Any angle EXCEPT ~90° (orthogonal = misclassification)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from gear_analysis.utils import (
    unit_vector, normalize_angle_diff, compute_angles, fit_line_svd,
    orient_direction_outward
)

logger = logging.getLogger(__name__)

Points2D = NDArray[np.floating]
Normals2D = NDArray[np.floating]
Vector2D = NDArray[np.floating]


# =============================================================================
# Enums and Data Classes
# =============================================================================

class FlankSide(Enum):
    """Enumeration for flank side classification."""
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class ClassificationMethod(Enum):
    """Method used to classify flank side."""
    ANGULAR = "angular"      # Based on angular position relative to tooth center
    NORMAL = "normal"        # Based on surface normal direction


class FitMethod(Enum):
    """Method used for line direction fitting.
    
    SVD is the fast path, used when normals confirm SVD is correct.
    NORMAL_DERIVED is the fallback, used when SVD disagrees with surface geometry.
    """
    SVD = "svd"              # Standard SVD (fast, usually correct)
    NORMAL_DERIVED = "normal"  # Direction from surface normals (robust fallback)


@dataclass
class FlankClassification:
    """Result of flank side classification.
    
    Attributes:
        side: Classified side (LEFT, RIGHT, or UNKNOWN)
        method: Method used for classification
        confidence: Confidence score (0.0 to 1.0)
        avg_tangential: Average tangential component of normals
    """
    side: FlankSide
    method: ClassificationMethod
    confidence: float
    avg_tangential: float = 0.0


@dataclass
class FlankValidation:
    """Validation result comparing SVD and normal-based methods.
    
    This captures the comparison between the traditional SVD-based fitting
    and the new normal-based approach.
    
    Attributes:
        svd_direction: Direction from SVD fitting
        normal_direction: Direction derived from surface normals
        angle_difference_deg: Angle between the two directions
        is_consistent: True if methods agree within threshold
        angular_side: Side classification from angular position method
        normal_side: Side classification from normal method
        sides_agree: True if both methods classify same side
    """
    svd_direction: Vector2D
    normal_direction: Vector2D
    angle_difference_deg: float
    is_consistent: bool
    angular_side: FlankSide
    normal_side: FlankSide
    sides_agree: bool
    
    @property
    def is_flagged(self) -> bool:
        """Returns True if this flank should be flagged for review."""
        return not self.is_consistent or not self.sides_agree
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "svd_direction": self.svd_direction.tolist(),
            "normal_direction": self.normal_direction.tolist(),
            "angle_difference_deg": float(self.angle_difference_deg),
            "is_consistent": self.is_consistent,
            "angular_side": self.angular_side.value,
            "normal_side": self.normal_side.value,
            "sides_agree": self.sides_agree,
            "is_flagged": self.is_flagged,
        }


@dataclass
class ToothFlanks:
    """Both flanks of a single tooth with optional normal-based validation.
    
    This is the primary output of flank fitting. It contains:
    - Basic geometry: centroid, direction, point count for each flank
    - Validation: comparison of SVD vs normal-based methods (if normals available)
    - Flags: indication of potential problems
    - Fit methods: which method was used for each flank direction
    
    The class maintains backward compatibility - validation fields are optional.
    """
    tooth: int
    
    # Left flank geometry
    left_point: Vector2D
    left_direction: Vector2D
    left_n_points: int
    
    # Right flank geometry
    right_point: Vector2D
    right_direction: Vector2D
    right_n_points: int
    
    # Validation (optional, populated when normals available)
    left_validation: Optional[FlankValidation] = None
    right_validation: Optional[FlankValidation] = None
    
    # Fit method used (optional, populated when auto-correction is active)
    left_fit_method: Optional[FitMethod] = None
    right_fit_method: Optional[FitMethod] = None
    
    # Overall status
    is_valid: bool = True
    issue: str = ""
    is_flagged: bool = False
    flag_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        result = {
            "tooth": self.tooth,
            "is_valid": self.is_valid,
            "is_flagged": self.is_flagged,
            "issue": self.issue,
            "flag_reasons": self.flag_reasons,
            "left": {
                "point": self.left_point.tolist(),
                "direction": self.left_direction.tolist(),
                "n_points": self.left_n_points,
                "fit_method": self.left_fit_method.value if self.left_fit_method else None,
            },
            "right": {
                "point": self.right_point.tolist(),
                "direction": self.right_direction.tolist(),
                "n_points": self.right_n_points,
                "fit_method": self.right_fit_method.value if self.right_fit_method else None,
            },
        }
        
        if self.left_validation:
            result["left"]["validation"] = self.left_validation.to_dict()
        if self.right_validation:
            result["right"]["validation"] = self.right_validation.to_dict()
        
        return result


# =============================================================================
# Normal-Based Classification
# =============================================================================

class NormalBasedClassifier:
    """Classifies flank sides using surface normal information.
    
    The key insight is that surface normals point OUTWARD from the gear:
    - Left flank normals have a negative tangential component (point "backward")
    - Right flank normals have a positive tangential component (point "forward")
    
    In a coordinate system centered on the gear:
    - For a point at angle θ, the tangential direction is (-sin(θ), cos(θ))
    - Left flanks have normals with negative dot product with tangential
    - Right flanks have normals with positive dot product with tangential
    
    This provides an independent verification of the angular-based classification.
    """
    
    # Threshold for classification (absolute value of avg_tangential)
    CLASSIFICATION_THRESHOLD = 0.1
    
    @classmethod
    def classify_flank_side(
        cls,
        points: Points2D,
        normals: Normals2D,
        tooth_center_angle: Optional[float] = None
    ) -> FlankClassification:
        """Classify flank as LEFT or RIGHT based on surface normals.
        
        Args:
            points: 2D points on the flank, shape (N, 2)
            normals: 2D surface normals at each point, shape (N, 2)
            tooth_center_angle: Optional angle of tooth center (radians)
        
        Returns:
            FlankClassification with side, confidence, and details
        """
        if len(points) == 0 or len(normals) == 0:
            return FlankClassification(
                side=FlankSide.UNKNOWN,
                method=ClassificationMethod.NORMAL,
                confidence=0.0,
                avg_tangential=0.0
            )
        
        # Compute tooth center angle if not provided
        if tooth_center_angle is None:
            centroid = points.mean(axis=0)
            tooth_center_angle = np.arctan2(centroid[1], centroid[0])
        
        # Compute tangential direction at each point
        # Tangential direction at angle θ is (-sin(θ), cos(θ)) for CCW rotation
        point_angles = np.arctan2(points[:, 1], points[:, 0])
        tangential_dirs = np.column_stack([
            -np.sin(point_angles),
            np.cos(point_angles)
        ])
        
        # Compute dot product of each normal with its tangential direction
        tangential_components = np.sum(normals * tangential_dirs, axis=1)
        avg_tangential = float(np.mean(tangential_components))
        
        # Compute confidence based on consistency of signs
        n_positive = np.sum(tangential_components > 0)
        n_negative = np.sum(tangential_components < 0)
        n_total = len(tangential_components)
        confidence = max(n_positive, n_negative) / n_total if n_total > 0 else 0.0
        
        # Classify based on average
        if avg_tangential > cls.CLASSIFICATION_THRESHOLD:
            side = FlankSide.RIGHT
        elif avg_tangential < -cls.CLASSIFICATION_THRESHOLD:
            side = FlankSide.LEFT
        else:
            side = FlankSide.UNKNOWN
            confidence *= 0.5  # Reduce confidence for ambiguous cases
        
        return FlankClassification(
            side=side,
            method=ClassificationMethod.NORMAL,
            confidence=confidence,
            avg_tangential=avg_tangential
        )
    
    @staticmethod
    def direction_from_normals(normals: Normals2D) -> Vector2D:
        """Compute flank direction from average surface normal.
        
        The flank direction is perpendicular to the average normal.
        
        Args:
            normals: 2D surface normals, shape (N, 2)
            
        Returns:
            Unit vector in the flank direction
        """
        avg_normal = normals.mean(axis=0)
        norm = np.linalg.norm(avg_normal)
        
        if norm < 1e-10:
            return np.array([1.0, 0.0])
        
        avg_normal = avg_normal / norm
        
        # Flank direction is perpendicular to normal
        return np.array([-avg_normal[1], avg_normal[0]])


# =============================================================================
# Main Line Fitter
# =============================================================================

class LineFitter:
    """Fits lines to point clouds with optional normal-based validation.
    
    This class provides:
    1. SVD-based line fitting (robust, handles any point distribution)
    2. Outlier removal for better fits
    3. Normal-based validation when surface normals are available
    4. Flagging of inconsistent results
    
    The main entry points are:
    - extract_both_flanks(): Basic fitting without normals (backward compatible)
    - extract_both_flanks_with_validation(): Enhanced fitting with normal validation
    """
    
    # Threshold for direction agreement (degrees)
    DIRECTION_AGREEMENT_THRESHOLD = 15.0
    
    @staticmethod
    def fit_svd(points: Points2D) -> Tuple[Vector2D, Vector2D]:
        """Fit line using SVD. Returns (centroid, direction)."""
        return fit_line_svd(points)
    
    @classmethod
    def fit_robust(cls, points: Points2D, threshold: float = 2.0) -> Tuple[Vector2D, Vector2D]:
        """Fit line with outlier removal.
        
        First fits using all points, then removes outliers and refits.
        
        Args:
            points: 2D points array
            threshold: Number of standard deviations for outlier detection
            
        Returns:
            Tuple of (centroid, direction)
        """
        if len(points) < 5:
            return cls.fit_svd(points)
        
        centroid, direction = cls.fit_svd(points)
        
        # Compute perpendicular distances
        perp = np.array([-direction[1], direction[0]])
        distances = np.abs((points - centroid) @ perp)
        
        # Remove outliers
        std = np.std(distances)
        if std > 1e-10:
            mask = distances < threshold * std
            if np.sum(mask) >= 3:
                centroid, direction = cls.fit_svd(points[mask])
        
        return centroid, direction
    
    # Threshold for switching from SVD to normal-derived direction
    # When |dot(svd_direction, avg_normal)| exceeds this, SVD is wrong
    # 
    # The perpendicularity value = sin(angle_error):
    #   - 0.26 = 15° error
    #   - 0.50 = 30° error  
    #   - 0.71 = 45° error
    #
    # We use 0.5 (30°) as a conservative threshold - this is well above
    # normal variation (typically <5°) but catches problems early.
    PERPENDICULARITY_THRESHOLD = 0.5
    
    @classmethod
    def _fit_adaptive(
        cls,
        points: Points2D,
        normals: Normals2D,
    ) -> Tuple[Vector2D, Vector2D, FitMethod, float]:
        """Fit line with automatic method selection.
        
        Uses SVD as primary method, but switches to normal-derived
        direction when SVD is clearly wrong (not perpendicular to normals).
        
        This is the KEY improvement: instead of just flagging bad fits,
        we automatically correct them.
        
        Args:
            points: 2D points, shape (N, 2)
            normals: 2D surface normals, shape (N, 2)
            
        Returns:
            Tuple of (centroid, direction, method_used, angle_difference_deg)
        """
        # Step 1: Standard SVD fit
        svd_centroid, svd_dir = cls.fit_robust(points)
        
        # Step 2: Get direction from normals
        avg_normal = normals.mean(axis=0)
        norm = np.linalg.norm(avg_normal)
        
        if norm < 1e-10:
            # No useful normal info, fall back to SVD
            return svd_centroid, svd_dir, FitMethod.SVD, 0.0
        
        avg_normal = avg_normal / norm
        # Line direction is perpendicular to average normal
        normal_dir = np.array([-avg_normal[1], avg_normal[0]])
        
        # Align directions for comparison (point same way)
        if np.dot(svd_dir, normal_dir) < 0:
            normal_dir = -normal_dir
        
        # Step 3: Compute angle difference
        cos_angle = np.clip(np.dot(svd_dir, normal_dir), -1, 1)
        angle_diff_deg = float(np.degrees(np.arccos(cos_angle)))
        
        # Step 4: Check perpendicularity - THE KEY SIGNAL
        # SVD direction should be perpendicular to average normal (dot ≈ 0)
        # If not, SVD is fitting the wrong direction
        perpendicularity = abs(np.dot(svd_dir, avg_normal))
        
        # Step 5: Choose method
        if perpendicularity > cls.PERPENDICULARITY_THRESHOLD:
            # SVD is wrong! Use normal-derived direction
            logger.debug(f"  Auto-correcting: perp={perpendicularity:.2f}, "
                        f"angle_diff={angle_diff_deg:.1f}° → using normal-derived")
            return svd_centroid, normal_dir, FitMethod.NORMAL_DERIVED, angle_diff_deg
        else:
            # SVD is good
            return svd_centroid, svd_dir, FitMethod.SVD, angle_diff_deg
    
    @classmethod
    def _validate_flank(
        cls,
        points: Points2D,
        normals: Normals2D,
        svd_direction: Vector2D,
        expected_side: FlankSide,
        tooth_center_angle: float
    ) -> FlankValidation:
        """Validate a flank by comparing SVD and normal-based methods.
        
        Args:
            points: 2D flank points
            normals: 2D surface normals
            svd_direction: Direction from SVD fitting
            expected_side: Expected side from angular classification
            tooth_center_angle: Angle of tooth center (radians)
            
        Returns:
            FlankValidation with comparison results
        """
        # Get direction from normals
        normal_direction = NormalBasedClassifier.direction_from_normals(normals)
        
        # Ensure directions point the same way for comparison
        if np.dot(svd_direction, normal_direction) < 0:
            normal_direction = -normal_direction
        
        # Compute angle difference
        cos_angle = np.clip(np.dot(svd_direction, normal_direction), -1, 1)
        angle_diff_deg = float(np.degrees(np.arccos(cos_angle)))
        
        # Check direction consistency
        is_consistent = angle_diff_deg < cls.DIRECTION_AGREEMENT_THRESHOLD
        
        # Classify side using normals
        normal_classification = NormalBasedClassifier.classify_flank_side(
            points, normals, tooth_center_angle
        )
        normal_side = normal_classification.side
        
        # Check side agreement
        sides_agree = (expected_side == normal_side) or (normal_side == FlankSide.UNKNOWN)
        
        return FlankValidation(
            svd_direction=svd_direction,
            normal_direction=normal_direction,
            angle_difference_deg=angle_diff_deg,
            is_consistent=is_consistent,
            angular_side=expected_side,
            normal_side=normal_side,
            sides_agree=sides_agree
        )
    
    @classmethod
    def _validate_flanks_geometry(
        cls,
        left_point: Vector2D,
        left_dir: Vector2D,
        right_point: Vector2D,
        right_dir: Vector2D,
        tooth_angle: float
    ) -> Tuple[bool, str]:
        """Validate flanks using geometric checks.
        
        Checks for:
        1. Orthogonal flanks (bad - indicates misclassification)
        2. Flanks too close together (same flank detected twice)
        3. Large radius difference (one flank not on the tooth)
        
        Returns:
            Tuple of (is_valid, issue_description)
        """
        # Check 1: Flanks should NOT be orthogonal (~90°)
        dir_dot = abs(np.dot(left_dir, right_dir))
        if dir_dot < 0.25:  # > ~75 degrees
            angle_between = np.degrees(np.arccos(np.clip(dir_dot, -1, 1)))
            return False, f"Flanks orthogonal ({angle_between:.0f}° apart)"
        
        # Check 2: Flanks should be separated
        distance = np.linalg.norm(left_point - right_point)
        left_radius = np.linalg.norm(left_point)
        if left_radius > 0 and distance < left_radius * 0.01:
            return False, "Flanks too close (same flank detected twice)"
        
        # Check 3: Both flanks at similar radius
        right_radius = np.linalg.norm(right_point)
        if left_radius > 0 and right_radius > 0:
            radius_ratio = min(left_radius, right_radius) / max(left_radius, right_radius)
            if radius_ratio < 0.85:
                return False, f"Flank radii differ too much ({radius_ratio:.2f} ratio)"
        
        return True, ""
    
    @classmethod
    def extract_both_flanks(
        cls,
        points: Points2D,
        min_points: int,
        tooth_number: int
    ) -> Optional[ToothFlanks]:
        """Extract left and right flanks without normal validation.
        
        This is the backward-compatible method that works without normals.
        For enhanced analysis with validation, use extract_both_flanks_with_validation().
        
        Args:
            points: 2D points for this tooth cluster
            min_points: Minimum points required per flank
            tooth_number: Tooth number (1-indexed)
            
        Returns:
            ToothFlanks if successful, None otherwise
        """
        if len(points) < 2 * min_points:
            logger.warning(f"Tooth {tooth_number}: Not enough points ({len(points)})")
            return None
        
        # Tooth center angle
        center = points.mean(axis=0)
        tooth_angle = compute_angles(center[None, :])[0]
        
        # Split by angular position
        point_angles = compute_angles(points)
        angle_diff = normalize_angle_diff(point_angles - tooth_angle)
        
        right_mask = angle_diff > 0
        left_mask = angle_diff < 0
        
        right_pts = points[right_mask]
        left_pts = points[left_mask]
        
        if len(right_pts) < min_points or len(left_pts) < min_points:
            logger.warning(f"Tooth {tooth_number}: Unbalanced split "
                          f"(left={len(left_pts)}, right={len(right_pts)})")
            return None
        
        try:
            right_center, right_dir = cls.fit_robust(right_pts)
            left_center, left_dir = cls.fit_robust(left_pts)
        except ValueError as e:
            logger.warning(f"Tooth {tooth_number}: Fitting failed: {e}")
            return None
        
        # Orient directions to point outward
        right_dir = orient_direction_outward(right_dir, right_center)
        left_dir = orient_direction_outward(left_dir, left_center)
        
        # Geometric validation
        is_valid, issue = cls._validate_flanks_geometry(
            left_center, left_dir, right_center, right_dir, tooth_angle
        )
        
        if not is_valid:
            logger.warning(f"Tooth {tooth_number}: {issue}")
        
        return ToothFlanks(
            tooth=tooth_number,
            left_point=left_center,
            left_direction=left_dir,
            left_n_points=len(left_pts),
            right_point=right_center,
            right_direction=right_dir,
            right_n_points=len(right_pts),
            is_valid=is_valid,
            issue=issue,
        )
    
    @classmethod
    def extract_both_flanks_with_validation(
        cls,
        points: Points2D,
        normals: Normals2D,
        min_points: int,
        tooth_number: int,
        auto_correct: bool = True
    ) -> Optional[ToothFlanks]:
        """Extract left and right flanks with normal-based validation and auto-correction.
        
        This enhanced method:
        1. Splits points by angular position
        2. Fits lines using adaptive method (SVD with automatic fallback to normals)
        3. Validates using surface normals
        4. AUTO-CORRECTS bad fits when SVD disagrees with normals
        5. Flags corrected flanks for review
        
        The key improvement over the basic method: when SVD produces a wrong
        direction (detected by checking perpendicularity to normals), we
        automatically use the normal-derived direction instead.
        
        Args:
            points: 2D points for this tooth cluster
            normals: 2D surface normals at each point
            min_points: Minimum points required per flank
            tooth_number: Tooth number (1-indexed)
            auto_correct: If True, automatically correct bad SVD fits using normals
            
        Returns:
            ToothFlanks with validation if successful, None otherwise
        """
        if len(points) < 2 * min_points:
            logger.warning(f"Tooth {tooth_number}: Not enough points ({len(points)})")
            return None
        
        # Tooth center angle
        center = points.mean(axis=0)
        tooth_angle = float(np.arctan2(center[1], center[0]))
        
        # Split by angular position
        point_angles = np.arctan2(points[:, 1], points[:, 0])
        angle_diff = normalize_angle_diff(point_angles - tooth_angle)
        
        right_mask = angle_diff > 0
        left_mask = angle_diff < 0
        
        right_pts = points[right_mask]
        left_pts = points[left_mask]
        right_nrm = normals[right_mask]
        left_nrm = normals[left_mask]
        
        if len(right_pts) < min_points or len(left_pts) < min_points:
            logger.warning(f"Tooth {tooth_number}: Unbalanced split "
                          f"(left={len(left_pts)}, right={len(right_pts)})")
            return None
        
        # Fit flanks - use adaptive method if auto_correct is enabled
        try:
            if auto_correct:
                # NEW: Adaptive fitting with automatic correction
                left_center, left_dir, left_method, left_angle_diff = cls._fit_adaptive(
                    left_pts, left_nrm
                )
                right_center, right_dir, right_method, right_angle_diff = cls._fit_adaptive(
                    right_pts, right_nrm
                )
            else:
                # Original behavior: SVD only
                left_center, left_dir = cls.fit_robust(left_pts)
                right_center, right_dir = cls.fit_robust(right_pts)
                left_method = FitMethod.SVD
                right_method = FitMethod.SVD
                left_angle_diff = 0.0
                right_angle_diff = 0.0
        except ValueError as e:
            logger.warning(f"Tooth {tooth_number}: Fitting failed: {e}")
            return None
        
        # Orient directions to point outward
        right_dir = orient_direction_outward(right_dir, right_center)
        left_dir = orient_direction_outward(left_dir, left_center)
        
        # Validate with normals (still useful even with auto-correction)
        left_validation = cls._validate_flank(
            left_pts, left_nrm, left_dir, FlankSide.LEFT, tooth_angle
        )
        right_validation = cls._validate_flank(
            right_pts, right_nrm, right_dir, FlankSide.RIGHT, tooth_angle
        )
        
        # Collect flag reasons
        flag_reasons = []
        
        # Flag if we had to use normal-derived (indicates SVD was wrong)
        if left_method == FitMethod.NORMAL_DERIVED:
            flag_reasons.append(
                f"Left flank: Auto-corrected (SVD was {left_angle_diff:.1f}° off)"
            )
        
        if right_method == FitMethod.NORMAL_DERIVED:
            flag_reasons.append(
                f"Right flank: Auto-corrected (SVD was {right_angle_diff:.1f}° off)"
            )
        
        # Also flag if validation shows issues (side disagreement)
        if left_validation.is_flagged and not left_validation.sides_agree:
            flag_reasons.append(
                f"Left flank: Angular says LEFT, normals say "
                f"{left_validation.normal_side.value.upper()}"
            )
        
        if right_validation.is_flagged and not right_validation.sides_agree:
            flag_reasons.append(
                f"Right flank: Angular says RIGHT, normals say "
                f"{right_validation.normal_side.value.upper()}"
            )
        
        is_flagged = len(flag_reasons) > 0
        
        # Geometric validation
        is_valid, issue = cls._validate_flanks_geometry(
            left_center, left_dir, right_center, right_dir, tooth_angle
        )
        
        if not is_valid:
            logger.warning(f"Tooth {tooth_number}: {issue}")
        
        if is_flagged:
            logger.info(f"Tooth {tooth_number}: FLAGGED - {'; '.join(flag_reasons)}")
        
        return ToothFlanks(
            tooth=tooth_number,
            left_point=left_center,
            left_direction=left_dir,
            left_n_points=len(left_pts),
            right_point=right_center,
            right_direction=right_dir,
            right_n_points=len(right_pts),
            left_validation=left_validation,
            right_validation=right_validation,
            left_fit_method=left_method,
            right_fit_method=right_method,
            is_valid=is_valid,
            issue=issue,
            is_flagged=is_flagged,
            flag_reasons=flag_reasons,
        )
    
    @classmethod
    def extract_right_flank(cls, points: Points2D, min_points: int) -> Tuple[Vector2D, Vector2D]:
        """Extract right flank only (backward compatibility)."""
        center = points.mean(axis=0)
        tooth_angle = compute_angles(center[None, :])[0]
        point_angles = compute_angles(points)
        angle_diff = normalize_angle_diff(point_angles - tooth_angle)
        
        right_pts = points[angle_diff > 0]
        if len(right_pts) < min_points:
            right_pts = points[np.argsort(angle_diff)[-min_points:]]
        
        centroid, direction = cls.fit_robust(right_pts)
        return centroid, orient_direction_outward(direction, centroid)
    
    @classmethod
    def extract_left_flank(cls, points: Points2D, min_points: int) -> Tuple[Vector2D, Vector2D]:
        """Extract left flank only (backward compatibility)."""
        center = points.mean(axis=0)
        tooth_angle = compute_angles(center[None, :])[0]
        point_angles = compute_angles(points)
        angle_diff = normalize_angle_diff(point_angles - tooth_angle)
        
        left_pts = points[angle_diff < 0]
        if len(left_pts) < min_points:
            left_pts = points[np.argsort(angle_diff)[:min_points]]
        
        centroid, direction = cls.fit_robust(left_pts)
        return centroid, orient_direction_outward(direction, centroid)


# =============================================================================
# Utility Functions
# =============================================================================

def summarize_validation_results(teeth: List[ToothFlanks]) -> dict:
    """Summarize validation results across all teeth.
    
    Args:
        teeth: List of ToothFlanks from analysis
        
    Returns:
        Dictionary with summary statistics
    """
    total = len(teeth)
    valid = sum(1 for t in teeth if t.is_valid)
    flagged = sum(1 for t in teeth if t.is_flagged)
    
    # Collect angle differences where validation exists
    left_diffs = []
    right_diffs = []
    for t in teeth:
        if t.left_validation:
            left_diffs.append(t.left_validation.angle_difference_deg)
        if t.right_validation:
            right_diffs.append(t.right_validation.angle_difference_deg)
    
    return {
        "total_teeth": total,
        "valid_teeth": valid,
        "invalid_teeth": total - valid,
        "flagged_teeth": flagged,
        "flagged_percentage": 100 * flagged / total if total > 0 else 0,
        "left_angle_diff_mean": float(np.mean(left_diffs)) if left_diffs else None,
        "left_angle_diff_max": float(np.max(left_diffs)) if left_diffs else None,
        "right_angle_diff_mean": float(np.mean(right_diffs)) if right_diffs else None,
        "right_angle_diff_max": float(np.max(right_diffs)) if right_diffs else None,
        "flagged_details": [
            {
                "tooth": t.tooth,
                "reasons": t.flag_reasons,
                "is_valid": t.is_valid,
                "issue": t.issue
            }
            for t in teeth if t.is_flagged
        ]
    }