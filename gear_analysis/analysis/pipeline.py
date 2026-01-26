"""
Main analysis pipeline module with surface normal integration.

This module contains the GearAnalyzer class which orchestrates
the complete gear flank analysis workflow.

Key features:
1. Two flank detection methods: 'angular' (fast) and 'radial' (robust)
2. Optional surface normal extraction for enhanced validation
3. Automatic flagging of inconsistent flanks
4. Full backward compatibility with existing code

The pipeline supports two modes:
- Standard mode (use_surface_normals=False): Original behavior
- Enhanced mode (use_surface_normals=True): Adds normal-based validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Any

import numpy as np

from gear_analysis.config import AnalysisConfig
from gear_analysis.models import (
    AnalysisResult,
    FlankLine,
    GearCenter,
    GhostCircle,
    OffsetAnalysis,
    PairBisector,
)
from gear_analysis.mesh import MeshLoader, SliceExtractor
from gear_analysis.mesh.slicing import SliceData
from gear_analysis.geometry import (
    PointFilter,
    ToothClusterer,
    LineFitter,
    CircleFitter,
    BisectorComputer,
    IntersectionFinder,
)
from gear_analysis.geometry.line_fitting import (
    ToothFlanks, 
    summarize_validation_results
)
from gear_analysis.geometry.robust_classification import (
    NormalBasedFlankClassifier,
    extract_robust_tooth_flanks,
    robust_to_standard_tooth_flanks,
)
from gear_analysis.geometry.bisectors import ToothBisector
from gear_analysis.analysis.center_estimation import GearCenterEstimator
from gear_analysis.utils import unit_vector

logger = logging.getLogger(__name__)


@dataclass
class ValidationSummary:
    """Summary of normal-based validation results.
    
    This captures the overall quality of the analysis based on
    comparing SVD-based fitting with surface normal information.
    """
    total_teeth: int = 0
    valid_teeth: int = 0
    flagged_teeth: int = 0
    flagged_percentage: float = 0.0
    left_angle_diff_mean: Optional[float] = None
    right_angle_diff_mean: Optional[float] = None
    flagged_details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "total_teeth": self.total_teeth,
            "valid_teeth": self.valid_teeth,
            "flagged_teeth": self.flagged_teeth,
            "flagged_percentage": self.flagged_percentage,
            "left_angle_diff_mean": self.left_angle_diff_mean,
            "right_angle_diff_mean": self.right_angle_diff_mean,
            "flagged_details": self.flagged_details,
        }


@dataclass
class ExtendedAnalysisResult(AnalysisResult):
    """Extended analysis results with validation data.
    
    This extends the base AnalysisResult with:
    - tooth_flanks: Detailed flank data including validation
    - tooth_bisectors: Bisectors for ghost circle computation
    - validation_summary: Overall validation statistics
    - has_normals: Whether surface normals were used
    """
    tooth_flanks: List[ToothFlanks] = None
    tooth_bisectors: List[ToothBisector] = None
    validation_summary: Optional[ValidationSummary] = None
    has_normals: bool = False
    
    def __post_init__(self):
        if self.tooth_flanks is None:
            self.tooth_flanks = []
        if self.tooth_bisectors is None:
            self.tooth_bisectors = []
    
    def get_flagged_teeth(self) -> List[ToothFlanks]:
        """Get list of teeth that were flagged during validation."""
        return [t for t in self.tooth_flanks if t.is_flagged]
    
    def get_invalid_teeth(self) -> List[ToothFlanks]:
        """Get list of teeth that failed geometric validation."""
        return [t for t in self.tooth_flanks if not t.is_valid]


class GearAnalyzer:
    """Main analysis pipeline for gear flank analysis.
    
    This class orchestrates the complete analysis workflow:
    1. Load and prepare mesh
    2. Extract slice (optionally with surface normals)
    3. Filter points to tooth region
    4. Detect and fit flanks
    5. Validate flanks (if normals available)
    6. Compute bisectors and ghost circle
    7. Analyze offset
    
    Two flank detection methods are supported:
    - 'angular': Partitions by angular sectors (faster, less robust)
    - 'radial': Traces from tooth tips inward (slower, more robust)
    
    Example:
        >>> config = AnalysisConfig(
        ...     mesh_path=Path("gear.stl"),
        ...     use_surface_normals=True  # Enable validation
        ... )
        >>> analyzer = GearAnalyzer(config, method='radial')
        >>> result = analyzer.run()
        >>> 
        >>> # Check validation results
        >>> if result.validation_summary:
        ...     print(f"Flagged teeth: {result.validation_summary.flagged_teeth}")
    """
    
    def __init__(
        self, 
        config: AnalysisConfig,
        method: Literal['angular', 'radial'] = 'radial'
    ):
        """Initialize analyzer.
        
        Args:
            config: Analysis configuration
            method: Flank detection method - 'angular' or 'radial'
        """
        self.config = config
        self.method = method
    
    def run(self) -> ExtendedAnalysisResult:
        """Execute complete gear flank analysis pipeline.
        
        Returns:
            ExtendedAnalysisResult with all analysis data including validation
        """
        config = self.config
        
        # Step 1: Load mesh
        logger.info("Loading mesh...")
        mesh = MeshLoader.load(config.mesh_path, config.target_triangles)
        tm = MeshLoader.to_trimesh(mesh)
        
        # Step 2: Extract slice (with or without normals based on config)
        logger.info(f"Extracting slice at Z={config.slice_z}...")
        
        if config.use_surface_normals:
            logger.info("  Using enhanced extraction with surface normals")
            slice_data = SliceExtractor.extract_with_normals(
                tm, config.slice_z, config.slice_interpolation_density
            )
        else:
            slice_data = SliceExtractor.extract(
                tm, config.slice_z, config.slice_interpolation_density
            )
        
        logger.info(f"  Extracted {len(slice_data)} points")
        if slice_data.has_normals:
            logger.info(f"  Surface normals: available")
        
        # Step 3: Filter by radius
        logger.info(f"Filtering points between radii {config.r_inner:.2f} and {config.r_outer:.2f}...")
        filtered_data = slice_data.filter_by_radius(config.r_inner, config.r_outer)
        logger.info(f"  Kept {len(filtered_data)} / {len(slice_data)} points")
        
        # Step 4: Recenter geometry
        # Get center from full slice for consistency
        full_slice_center = slice_data.points.mean(axis=0)
        filtered_data, center_xy = filtered_data.recenter(full_slice_center)
        
        logger.info(f"  Recentering by offset: [{center_xy[0]:.3f}, {center_xy[1]:.3f}]")
        
        # Also recenter the full slice and mesh
        slice_data_centered, _ = slice_data.recenter(full_slice_center)
        mesh.translate(np.array([-center_xy[0], -center_xy[1], 0.0]))
        
        # Step 5 & 6: Detect flanks (choose method)
        logger.info(f"Detecting flanks using '{self.method}' method...")
        
        if self.method == 'radial':
            tooth_flanks_list, flanks = self._fit_flanks_radial(filtered_data)
        else:
            tooth_flanks_list, flanks = self._fit_flanks_angular(filtered_data)
        
        if not tooth_flanks_list:
            raise RuntimeError("Unable to fit flanks for any tooth.")
        
        valid_count = sum(1 for tf in tooth_flanks_list if tf.is_valid)
        flagged_count = sum(1 for tf in tooth_flanks_list if tf.is_flagged)
        logger.info(f"  Successfully fitted {valid_count} valid teeth (out of {len(tooth_flanks_list)})")
        
        if flagged_count > 0:
            logger.warning(f"  {flagged_count} teeth flagged for review (SVD/normal disagreement)")
        
        # Create validation summary if normals were used
        validation_summary = None
        if filtered_data.has_normals:
            summary_dict = summarize_validation_results(tooth_flanks_list)
            validation_summary = ValidationSummary(
                total_teeth=summary_dict["total_teeth"],
                valid_teeth=summary_dict["valid_teeth"],
                flagged_teeth=summary_dict["flagged_teeth"],
                flagged_percentage=summary_dict["flagged_percentage"],
                left_angle_diff_mean=summary_dict["left_angle_diff_mean"],
                right_angle_diff_mean=summary_dict["right_angle_diff_mean"],
                flagged_details=summary_dict["flagged_details"],
            )
        
        # Step 7: Compute tooth bisectors
        logger.info("Computing tooth bisectors...")
        tooth_bisectors = BisectorComputer.compute_tooth_bisectors(
            tooth_flanks_list, config.bisector_length
        )
        logger.info(f"  Computed {len(tooth_bisectors)} tooth bisectors")
        
        pair_bisectors = self._convert_to_pair_bisectors(tooth_bisectors)
        
        # Step 8: Ghost circle analysis
        logger.info("\n--- Ghost Circle Analysis ---")
        ghost_circle = self._fit_ghost_circle_from_tooth_bisectors(tooth_bisectors)
        
        # Step 9: Gear center and offset
        logger.info("\n--- Gear Center Estimation ---")
        gear_center = self._estimate_gear_center(
            filtered_data.points, slice_data_centered.points
        )
        offset_analysis = self._compute_offset(ghost_circle, gear_center)
        
        # Create result
        analysis_result = ExtendedAnalysisResult(
            config=config,
            mesh=mesh,
            slice_points=slice_data_centered.points,
            filtered_points=filtered_data.points,
            center_shift=center_xy,
            flanks=flanks,
            bisectors=pair_bisectors,
            ghost_circle=ghost_circle,
            gear_center=gear_center,
            offset_analysis=offset_analysis,
            tooth_flanks=tooth_flanks_list,
            tooth_bisectors=tooth_bisectors,
            validation_summary=validation_summary,
            has_normals=filtered_data.has_normals,
        )
        
        total_flanks = len(tooth_flanks_list) * 2
        logger.info(f"Total flanks fitted: {total_flanks} ({len(tooth_flanks_list)} teeth × 2)")
        
        return analysis_result
    
    # =========================================================================
    # Flank Detection Methods
    # =========================================================================
    
    def _fit_flanks_radial(
        self,
        data: SliceData
    ) -> tuple[list[ToothFlanks], list[FlankLine]]:
        """Fit flanks using radial tracing method.
        
        This method traces from tooth tips inward, following aligned points.
        More robust than angular binning for irregular tooth shapes.
        
        If surface normals are available, performs validation.
        """
        from gear_analysis.geometry.radial_tracing import (
            RadialFlankTracer,
            convert_to_tooth_flanks
        )
        
        config = self.config
        
        # Create tracer with parameters
        tracer = RadialFlankTracer(
            angle_threshold_deg=10.0,
            min_points_per_flank=config.min_points_per_flank,
            radial_step_factor=0.15,
            search_angle_deg=45.0,
        )
        
        # Trace all teeth
        traced_teeth = tracer.trace_all_teeth(
            data.points,
            config.n_teeth,
            config.r_inner,
            config.r_outer
        )
        
        # Convert to ToothFlanks format
        tooth_flanks_list = convert_to_tooth_flanks(traced_teeth)
        
        # If we have normals, add validation to the ToothFlanks
        # Note: For radial tracing, we'd need to track which points went to which flank
        # This is a limitation - radial tracing doesn't easily support per-point normals
        # For now, radial method doesn't validate with normals
        if data.has_normals:
            logger.info("  Note: Radial tracing method doesn't yet support normal validation")
        
        # Create backward-compatible FlankLine list
        flanks_compat = []
        for tf in tooth_flanks_list:
            flanks_compat.append(FlankLine(
                tooth=tf.tooth,
                point=tf.right_point,
                direction=tf.right_direction,
                cluster_size=tf.right_n_points,
            ))
        
        return tooth_flanks_list, flanks_compat
    
    def _fit_flanks_angular(
        self,
        data: SliceData
    ) -> tuple[list[ToothFlanks], list[FlankLine]]:
        """Fit flanks using angular binning method.
        
        This partitions points into angular sectors, then fits lines
        to left/right halves of each sector.
        
        Two classification modes (controlled by config.use_normal_based_classification):
        1. Angular (default): Split left/right by angular position relative to tooth center
        2. Normal-based: Split left/right by surface normal tangential component
        
        The normal-based mode is more robust when angular binning misclassifies
        points near the boundary between flanks.
        
        When normals are available, we also detect and correct for angular
        phase misalignment (when teeth don't start at 0°).
        """
        config = self.config
        has_normals = data.has_normals
        use_robust = config.use_normal_based_classification and has_normals
        
        # Partition into tooth clusters using index-based approach
        # This ensures points and normals stay aligned
        from gear_analysis.utils import normalize_angle, compute_angles
        from gear_analysis.geometry.robust_classification import (
            detect_angular_phase_offset_by_normals
        )
        
        points = data.points
        normals = data.normals if has_normals else None
        n_teeth = config.n_teeth
        min_points = config.min_points_per_cluster
        
        # Detect and apply phase offset if normals available
        phase_offset = 0.0
        if has_normals:
            phase_offset = detect_angular_phase_offset_by_normals(
                points, normals, n_teeth
            )
            if abs(phase_offset) > 0.01:  # More than ~0.5 degrees
                logger.info(f"Detected angular phase offset: {np.degrees(phase_offset):.1f}°")
        
        # Calculate polar angles [0, 2π) with phase offset
        angles = normalize_angle(compute_angles(points))
        shifted_angles = (angles - phase_offset) % (2 * np.pi)
        
        # Create bin edges
        edges = np.linspace(0.0, 2 * np.pi, n_teeth + 1)
        
        # Assign points to bins using shifted angles
        bin_idx = np.digitize(shifted_angles, edges, right=False) - 1
        bin_idx[bin_idx == n_teeth] = 0  # Wrap last bin
        
        tooth_flanks_list: list[ToothFlanks] = []
        flanks_compat: list[FlankLine] = []
        
        for tooth_num in range(n_teeth):
            # Get indices for this tooth
            mask = bin_idx == tooth_num
            
            if np.sum(mask) < min_points:
                continue
            
            # Extract points (and normals if available)
            cluster_points = points[mask]
            cluster_normals = normals[mask] if has_normals else None
            
            idx = tooth_num + 1  # 1-indexed tooth number
            
            # Fit flanks with appropriate method
            if use_robust and cluster_normals is not None:
                # ROBUST MODE: Use normal-based classification
                robust_result = extract_robust_tooth_flanks(
                    cluster_points, cluster_normals, idx, config.min_points_per_flank
                )
                tooth_flanks = robust_to_standard_tooth_flanks(robust_result)
                
            elif has_normals and cluster_normals is not None:
                # STANDARD MODE with validation
                tooth_flanks = LineFitter.extract_both_flanks_with_validation(
                    cluster_points, cluster_normals, config.min_points_per_flank, idx
                )
            else:
                # BASIC MODE without normals
                tooth_flanks = LineFitter.extract_both_flanks(
                    cluster_points, config.min_points_per_flank, idx
                )
            
            if tooth_flanks is not None:
                tooth_flanks_list.append(tooth_flanks)
                flanks_compat.append(FlankLine(
                    tooth=tooth_flanks.tooth,
                    point=tooth_flanks.right_point,
                    direction=tooth_flanks.right_direction,
                    cluster_size=tooth_flanks.right_n_points,
                ))
            else:
                # Fallback: try to extract just right flank
                try:
                    point, direction = LineFitter.extract_right_flank(
                        cluster_points, config.min_points_per_flank
                    )
                    flanks_compat.append(FlankLine(
                        tooth=idx,
                        point=point,
                        direction=unit_vector(direction),
                        cluster_size=len(cluster_points),
                    ))
                except ValueError:
                    pass
        
        return tooth_flanks_list, flanks_compat
    
    # =========================================================================
    # Shared Methods
    # =========================================================================
    
    def _convert_to_pair_bisectors(
        self, 
        tooth_bisectors: list[ToothBisector]
    ) -> list[PairBisector]:
        """Convert ToothBisector to PairBisector for backward compatibility."""
        pair_bisectors: list[PairBisector] = []
        
        for tb in tooth_bisectors:
            pair_bisectors.append(PairBisector(
                between_teeth=(tb.tooth, tb.tooth),
                origin=tb.origin,
                direction=tb.direction,
                length=tb.length,
            ))
        
        return pair_bisectors
    
    def _fit_ghost_circle_from_tooth_bisectors(
        self, 
        tooth_bisectors: list[ToothBisector]
    ) -> Optional[GhostCircle]:
        """Fit ghost circle from tooth bisector intersections."""
        config = self.config
        
        logger.info(f"Computing intersections in radius range "
                   f"[{config.intersection_r_min:.3f}, {config.intersection_r_max:.3f}]...")
        
        intersections = IntersectionFinder.compute_tooth_bisector_intersections(
            tooth_bisectors,
            config.intersection_r_min,
            config.intersection_r_max,
            config.parallel_threshold
        )
        logger.info(f"  Found {len(intersections)} valid intersections")
        
        if len(intersections) < config.ransac_min_samples:
            logger.warning(f"Not enough intersections ({len(intersections)})")
            return None
        
        logger.info("Fitting ghost circle with RANSAC...")
        intersections_array = np.array(intersections)
        
        # Estimate expected ghost circle radius from intersection spread
        # Ghost circle should be small (intersections cluster near center)
        expected_radius = None
        if len(intersections_array) > 5:
            # Use median distance from centroid as expected radius
            centroid = intersections_array.mean(axis=0)
            distances = np.linalg.norm(intersections_array - centroid, axis=1)
            expected_radius = np.median(distances)
            logger.info(f"  Expected ghost circle radius: {expected_radius:.4f}")
        
        # Use config expected_ghost_radius if set, otherwise use computed value
        if hasattr(config, 'expected_ghost_radius') and config.expected_ghost_radius is not None:
            expected_radius = config.expected_ghost_radius
            logger.info(f"  Using configured expected radius: {expected_radius:.4f}")
        
        center, radius, inliers, outliers, rmse = CircleFitter.fit_ransac(
            intersections_array,
            min_samples=config.ransac_min_samples,
            residual_threshold=config.ransac_residual_threshold,
            max_iterations=config.ransac_iterations,
            expected_radius=expected_radius,
            radius_tolerance=getattr(config, 'ghost_radius_tolerance', 0.5),
        )
        
        ghost_circle = GhostCircle(
            center=center,
            radius=radius,
            inliers=inliers,
            outliers=outliers,
            rmse=rmse,
            n_intersections=len(intersections)
        )
        
        logger.info(f"  Center: ({center[0]:.4f}, {center[1]:.4f})")
        logger.info(f"  Radius: {radius:.4f}")
        logger.info(f"  Inliers: {len(inliers)} / {len(intersections)}")
        logger.info(f"  RMSE: {rmse:.6f}")
        
        return ghost_circle
    
    def _estimate_gear_center(
        self,
        filtered_points: np.ndarray,
        slice_points: np.ndarray
    ) -> GearCenter:
        """Estimate gear center."""
        config = self.config
        
        center_outer, radius_outer = GearCenterEstimator.from_outer_tips(
            filtered_points, config.r_outer
        )
        logger.info(f"Method 1 (outer tips): center=({center_outer[0]:.4f}, {center_outer[1]:.4f})")
        
        center_boundary, radius_boundary = GearCenterEstimator.from_boundary_centroid(
            slice_points, config.r_outer
        )
        logger.info(f"Method 2 (boundary): center=({center_boundary[0]:.4f}, {center_boundary[1]:.4f})")
        
        use_outer_tips = config.gear_center_method == "outer_tips"
        gear_center = GearCenter(
            center=center_outer if use_outer_tips else center_boundary,
            method=config.gear_center_method,
            radius=radius_outer if use_outer_tips else radius_boundary
        )
        
        return gear_center
    
    def _compute_offset(
        self,
        ghost_circle: Optional[GhostCircle],
        gear_center: GearCenter
    ) -> Optional[OffsetAnalysis]:
        """Compute offset between ghost circle and gear center."""
        if ghost_circle is None:
            return None
        
        offset_vector = ghost_circle.center - gear_center.center
        magnitude = float(np.linalg.norm(offset_vector))
        angle_rad = float(np.arctan2(offset_vector[1], offset_vector[0]))
        angle_deg = float(np.degrees(angle_rad))
        
        offset_analysis = OffsetAnalysis(
            offset_vector=offset_vector,
            magnitude=magnitude,
            angle_deg=angle_deg,
            ghost_center=ghost_circle.center,
            gear_center=gear_center.center
        )
        
        logger.info(f"\n--- Offset Analysis ---")
        logger.info(f"  Magnitude: {magnitude:.6f}")
        logger.info(f"  Angle: {angle_deg:.2f}°")
        
        return offset_analysis