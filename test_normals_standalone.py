"""
Standalone test of the normal-based classification logic.
This version doesn't require open3d/trimesh to demonstrate the core algorithms.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum


# =============================================================================
# Core classes (extracted from line_fitting_with_normals.py)
# =============================================================================

class FlankSide(Enum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class ClassificationMethod(Enum):
    ANGULAR = "angular"
    NORMAL = "normal"
    HYBRID = "hybrid"


@dataclass
class FlankClassification:
    side: FlankSide
    method: ClassificationMethod
    confidence: float
    details: str = ""


@dataclass
class FlankValidation:
    svd_direction: np.ndarray
    normal_direction: np.ndarray
    angle_difference_deg: float
    is_consistent: bool
    svd_side: FlankSide
    normal_side: FlankSide
    sides_agree: bool
    
    @property
    def is_flagged(self) -> bool:
        return not self.is_consistent or not self.sides_agree


@dataclass
class ToothFlanksWithNormals:
    tooth: int
    left_point: np.ndarray
    left_direction: np.ndarray
    left_n_points: int
    left_validation: Optional[FlankValidation] = None
    right_point: np.ndarray = None
    right_direction: np.ndarray = None
    right_n_points: int = 0
    right_validation: Optional[FlankValidation] = None
    is_valid: bool = True
    issue: str = ""
    is_flagged: bool = False
    flag_reasons: List[str] = field(default_factory=list)


# =============================================================================
# Core algorithms
# =============================================================================

def normalize_angle_diff(angle_diff):
    """Normalize angle difference to [-π, π]."""
    return np.arctan2(np.sin(angle_diff), np.cos(angle_diff))


def fit_line_svd(points):
    """Fit line using SVD. Returns (centroid, direction)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered)
    direction = Vt[0]
    return centroid, direction


def orient_direction_outward(direction, point):
    """Orient direction to point away from origin."""
    if np.dot(direction, point) < 0:
        return -direction
    return direction


class NormalBasedClassifier:
    """Classifies flank sides using surface normal information."""
    
    @staticmethod
    def classify_flank_side(points, normals, tooth_center_angle=None):
        """Classify flank as LEFT or RIGHT based on surface normals."""
        if len(points) == 0 or len(normals) == 0:
            return FlankClassification(
                side=FlankSide.UNKNOWN,
                method=ClassificationMethod.NORMAL,
                confidence=0.0,
                details="No points provided"
            )
        
        if tooth_center_angle is None:
            centroid = points.mean(axis=0)
            tooth_center_angle = np.arctan2(centroid[1], centroid[0])
        
        # Tangential direction at each point
        point_angles = np.arctan2(points[:, 1], points[:, 0])
        tangential_dirs = np.column_stack([
            -np.sin(point_angles),
            np.cos(point_angles)
        ])
        
        # Dot product of normals with tangential directions
        tangential_components = np.sum(normals * tangential_dirs, axis=1)
        avg_tangential = np.mean(tangential_components)
        
        # Confidence based on consistency
        n_positive = np.sum(tangential_components > 0)
        n_negative = np.sum(tangential_components < 0)
        n_total = len(tangential_components)
        confidence = max(n_positive, n_negative) / n_total if n_total > 0 else 0.0
        
        # Classify
        if avg_tangential > 0.1:
            side = FlankSide.RIGHT
        elif avg_tangential < -0.1:
            side = FlankSide.LEFT
        else:
            side = FlankSide.UNKNOWN
            confidence *= 0.5
        
        details = f"avg_tangential={avg_tangential:.3f}, pos={n_positive}, neg={n_negative}"
        
        return FlankClassification(
            side=side,
            method=ClassificationMethod.NORMAL,
            confidence=confidence,
            details=details
        )
    
    @staticmethod
    def direction_from_normals(normals):
        """Compute flank direction from average surface normal."""
        avg_normal = normals.mean(axis=0)
        norm = np.linalg.norm(avg_normal)
        
        if norm < 1e-10:
            return np.array([1.0, 0.0])
        
        avg_normal = avg_normal / norm
        # Perpendicular direction
        return np.array([-avg_normal[1], avg_normal[0]])


class LineFitterWithNormals:
    """Line fitting with normal-based validation."""
    
    DIRECTION_AGREEMENT_THRESHOLD = 15.0  # degrees
    
    @classmethod
    def validate_flank(cls, points, normals, svd_direction, expected_side, tooth_center_angle):
        """Validate a flank by comparing SVD and normal-based methods."""
        normal_direction = NormalBasedClassifier.direction_from_normals(normals)
        
        # Ensure same direction for comparison
        if np.dot(svd_direction, normal_direction) < 0:
            normal_direction = -normal_direction
        
        # Angle difference
        cos_angle = np.clip(np.dot(svd_direction, normal_direction), -1, 1)
        angle_diff_deg = np.degrees(np.arccos(cos_angle))
        is_consistent = angle_diff_deg < cls.DIRECTION_AGREEMENT_THRESHOLD
        
        # Side classification from normals
        normal_classification = NormalBasedClassifier.classify_flank_side(
            points, normals, tooth_center_angle
        )
        normal_side = normal_classification.side
        sides_agree = (expected_side == normal_side) or (normal_side == FlankSide.UNKNOWN)
        
        return FlankValidation(
            svd_direction=svd_direction,
            normal_direction=normal_direction,
            angle_difference_deg=angle_diff_deg,
            is_consistent=is_consistent,
            svd_side=expected_side,
            normal_side=normal_side,
            sides_agree=sides_agree
        )
    
    @classmethod
    def extract_both_flanks_with_normals(cls, points, normals, min_points, tooth_number):
        """Extract left and right flanks with normal-based validation."""
        if len(points) < 2 * min_points:
            return None
        
        # Split by angular position
        center = points.mean(axis=0)
        tooth_angle = np.arctan2(center[1], center[0])
        point_angles = np.arctan2(points[:, 1], points[:, 0])
        angle_diff = normalize_angle_diff(point_angles - tooth_angle)
        
        right_mask = angle_diff > 0
        left_mask = angle_diff < 0
        
        right_pts, left_pts = points[right_mask], points[left_mask]
        right_nrm, left_nrm = normals[right_mask], normals[left_mask]
        
        if len(right_pts) < min_points or len(left_pts) < min_points:
            return None
        
        # Fit using SVD
        right_center, right_dir = fit_line_svd(right_pts)
        left_center, left_dir = fit_line_svd(left_pts)
        
        right_dir = orient_direction_outward(right_dir, right_center)
        left_dir = orient_direction_outward(left_dir, left_center)
        
        # Validate with normals
        left_validation = cls.validate_flank(
            left_pts, left_nrm, left_dir, FlankSide.LEFT, tooth_angle
        )
        right_validation = cls.validate_flank(
            right_pts, right_nrm, right_dir, FlankSide.RIGHT, tooth_angle
        )
        
        # Check for flags
        flag_reasons = []
        if left_validation.is_flagged:
            if not left_validation.is_consistent:
                flag_reasons.append(
                    f"Left: SVD/normal differ by {left_validation.angle_difference_deg:.1f}°"
                )
            if not left_validation.sides_agree:
                flag_reasons.append(
                    f"Left: Angular=LEFT, Normal={left_validation.normal_side.value}"
                )
        
        if right_validation.is_flagged:
            if not right_validation.is_consistent:
                flag_reasons.append(
                    f"Right: SVD/normal differ by {right_validation.angle_difference_deg:.1f}°"
                )
            if not right_validation.sides_agree:
                flag_reasons.append(
                    f"Right: Angular=RIGHT, Normal={right_validation.normal_side.value}"
                )
        
        return ToothFlanksWithNormals(
            tooth=tooth_number,
            left_point=left_center,
            left_direction=left_dir,
            left_n_points=len(left_pts),
            left_validation=left_validation,
            right_point=right_center,
            right_direction=right_dir,
            right_n_points=len(right_pts),
            right_validation=right_validation,
            is_valid=True,
            is_flagged=len(flag_reasons) > 0,
            flag_reasons=flag_reasons,
        )


# =============================================================================
# Demonstration
# =============================================================================

def run_demo():
    print("=" * 70)
    print("SURFACE NORMAL INTEGRATION - DEMONSTRATION")
    print("=" * 70)
    
    np.random.seed(42)
    
    # =========================================================================
    # 1. Create synthetic tooth data
    # =========================================================================
    print("\n1. CREATING SYNTHETIC TOOTH DATA")
    print("-" * 70)
    
    n_points = 30
    tooth_angle = 0.0
    
    # Left flank (negative angle, normals pointing "backward")
    left_angles = np.linspace(-0.15, -0.05, n_points) + tooth_angle
    left_radii = np.linspace(2.65, 2.50, n_points)
    left_points = np.column_stack([
        left_radii * np.cos(left_angles),
        left_radii * np.sin(left_angles)
    ])
    left_points += 0.01 * np.random.randn(n_points, 2)
    
    # Left normals: tangentially backward (-Y direction roughly)
    left_normals = np.tile(np.array([0.3, -0.95]), (n_points, 1))
    left_normals += 0.1 * np.random.randn(n_points, 2)
    left_normals /= np.linalg.norm(left_normals, axis=1, keepdims=True)
    
    # Right flank (positive angle, normals pointing "forward")
    right_angles = np.linspace(0.05, 0.15, n_points) + tooth_angle
    right_radii = np.linspace(2.65, 2.50, n_points)
    right_points = np.column_stack([
        right_radii * np.cos(right_angles),
        right_radii * np.sin(right_angles)
    ])
    right_points += 0.01 * np.random.randn(n_points, 2)
    
    # Right normals: tangentially forward (+Y direction roughly)
    right_normals = np.tile(np.array([0.3, 0.95]), (n_points, 1))
    right_normals += 0.1 * np.random.randn(n_points, 2)
    right_normals /= np.linalg.norm(right_normals, axis=1, keepdims=True)
    
    tooth_points = np.vstack([left_points, right_points])
    tooth_normals = np.vstack([left_normals, right_normals])
    
    print(f"   Points array shape: {tooth_points.shape}")
    print(f"   Normals array shape: {tooth_normals.shape}")
    print(f"   Sample point: {tooth_points[0]}")
    print(f"   Sample normal: {tooth_normals[0]}")
    
    # =========================================================================
    # 2. Classify using normals
    # =========================================================================
    print("\n2. NORMAL-BASED CLASSIFICATION")
    print("-" * 70)
    
    left_class = NormalBasedClassifier.classify_flank_side(left_points, left_normals)
    right_class = NormalBasedClassifier.classify_flank_side(right_points, right_normals)
    
    print(f"   Left flank:")
    print(f"     Classified as: {left_class.side.value}")
    print(f"     Confidence: {left_class.confidence:.1%}")
    print(f"     Details: {left_class.details}")
    
    print(f"\n   Right flank:")
    print(f"     Classified as: {right_class.side.value}")
    print(f"     Confidence: {right_class.confidence:.1%}")
    print(f"     Details: {right_class.details}")
    
    # =========================================================================
    # 3. Full extraction with validation
    # =========================================================================
    print("\n3. FULL EXTRACTION WITH VALIDATION")
    print("-" * 70)
    
    result = LineFitterWithNormals.extract_both_flanks_with_normals(
        tooth_points, tooth_normals, min_points=5, tooth_number=1
    )
    
    if result:
        print(f"\n   TOOTH {result.tooth} RESULTS:")
        print(f"   ├── Overall valid: {result.is_valid}")
        print(f"   ├── Flagged: {result.is_flagged}")
        
        lv = result.left_validation
        print(f"   │")
        print(f"   ├── LEFT FLANK:")
        print(f"   │   ├── Points: {result.left_n_points}")
        print(f"   │   ├── SVD direction: ({result.left_direction[0]:.3f}, {result.left_direction[1]:.3f})")
        print(f"   │   ├── Normal direction: ({lv.normal_direction[0]:.3f}, {lv.normal_direction[1]:.3f})")
        print(f"   │   ├── Angle difference: {lv.angle_difference_deg:.1f}°")
        print(f"   │   ├── Directions consistent: {'✓' if lv.is_consistent else '✗'}")
        print(f"   │   └── Side classification: Angular={lv.svd_side.value}, Normal={lv.normal_side.value} {'✓' if lv.sides_agree else '✗'}")
        
        rv = result.right_validation
        print(f"   │")
        print(f"   └── RIGHT FLANK:")
        print(f"       ├── Points: {result.right_n_points}")
        print(f"       ├── SVD direction: ({result.right_direction[0]:.3f}, {result.right_direction[1]:.3f})")
        print(f"       ├── Normal direction: ({rv.normal_direction[0]:.3f}, {rv.normal_direction[1]:.3f})")
        print(f"       ├── Angle difference: {rv.angle_difference_deg:.1f}°")
        print(f"       ├── Directions consistent: {'✓' if rv.is_consistent else '✗'}")
        print(f"       └── Side classification: Angular={rv.svd_side.value}, Normal={rv.normal_side.value} {'✓' if rv.sides_agree else '✗'}")
    
    # =========================================================================
    # 4. Intentionally BAD example
    # =========================================================================
    print("\n4. INTENTIONALLY INCONSISTENT EXAMPLE")
    print("-" * 70)
    print("   (Swapping normals to create disagreement)")
    
    # Swap normals to create inconsistency
    bad_normals = np.vstack([right_normals, left_normals])  # Swapped!
    
    bad_result = LineFitterWithNormals.extract_both_flanks_with_normals(
        tooth_points, bad_normals, min_points=5, tooth_number=2
    )
    
    if bad_result:
        print(f"\n   TOOTH {bad_result.tooth} (with swapped normals):")
        print(f"   ├── Flagged: {'✗ YES!' if bad_result.is_flagged else '✓ No'}")
        if bad_result.flag_reasons:
            print(f"   └── Reasons:")
            for reason in bad_result.flag_reasons:
                print(f"       ⚠ {reason}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The normal-based validation system:

1. PRESERVES NORMALS during slicing
   - Each 2D point keeps its associated surface normal
   - Shape: points (N, 2), normals (N, 2)

2. CLASSIFIES FLANKS using normals
   - Left flanks: normals point tangentially backward (avg_tangential < 0)
   - Right flanks: normals point tangentially forward (avg_tangential > 0)
   - Provides confidence score based on consistency

3. VALIDATES by comparing methods
   - SVD-based: fits line to points, uses angular position for side
   - Normal-based: uses surface normal direction for side
   - Flags flanks where methods disagree by > 15°

4. FLAGS INCONSISTENT flanks for review
   - is_flagged=True when SVD and normal methods disagree
   - flag_reasons list explains what went wrong
""")


if __name__ == "__main__":
    run_demo()
