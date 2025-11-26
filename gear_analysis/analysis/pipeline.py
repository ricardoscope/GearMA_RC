"""
Main analysis pipeline module.

This module contains the GearAnalyzer class which orchestrates
the complete gear flank analysis workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

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
from gear_analysis.geometry import (
    PointFilter,
    ToothClusterer,
    LineFitter,
    CircleFitter,
    BisectorComputer,
    IntersectionFinder,
)
from gear_analysis.geometry.line_fitting import ToothFlanks
from gear_analysis.geometry.bisectors import ToothBisector
from gear_analysis.analysis.center_estimation import GearCenterEstimator
from gear_analysis.utils import unit_vector

logger = logging.getLogger(__name__)


@dataclass
class ExtendedAnalysisResult(AnalysisResult):
    """Extended analysis results with both flank types and tooth bisectors.
    
    This extends the base AnalysisResult to include:
    - tooth_flanks: Both left and right flanks for each tooth
    - tooth_bisectors: Bisectors between flanks of each tooth
    """
    tooth_flanks: list[ToothFlanks] = None
    tooth_bisectors: list[ToothBisector] = None
    
    def __post_init__(self):
        if self.tooth_flanks is None:
            self.tooth_flanks = []
        if self.tooth_bisectors is None:
            self.tooth_bisectors = []


class GearAnalyzer:
    """Main analysis pipeline for gear flank analysis.
    
    This class orchestrates the complete analysis workflow:
    1. Load and clean the mesh
    2. Extract a horizontal slice
    3. Filter points to the annular region
    4. Partition points into tooth clusters
    5. Fit BOTH flanks (left and right) to each tooth
    6. Compute bisectors between the flanks of EACH tooth
    7. Find bisector intersections
    8. Fit the ghost circle
    9. Estimate gear center and compute offset
    
    Attributes:
        config: Analysis configuration parameters
    
    Example:
        >>> config = AnalysisConfig(
        ...     mesh_path=Path("gear.stl"),
        ...     slice_z=-0.2,
        ...     r_inner=2.38,
        ...     r_outer=2.58,
        ...     n_teeth=38
        ... )
        >>> analyzer = GearAnalyzer(config)
        >>> result = analyzer.run()
        >>> print(f"Ghost circle radius: {result.ghost_circle.radius:.4f}")
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize analyzer with configuration.
        
        Args:
            config: Analysis configuration parameters
        """
        self.config = config
    
    def run(self) -> AnalysisResult:
        """Execute complete gear flank analysis pipeline.
        
        Returns:
            AnalysisResult containing all computed geometry and metrics
            
        Raises:
            FileNotFoundError: If mesh file doesn't exist
            RuntimeError: If analysis fails (e.g., no valid slices)
        """
        config = self.config
        
        # Step 1: Load mesh
        logger.info("Loading mesh...")
        mesh = MeshLoader.load(config.mesh_path, config.target_triangles)
        tm = MeshLoader.to_trimesh(mesh)
        
        # Step 2: Extract slice
        logger.info(f"Extracting slice at Z={config.slice_z}...")
        slice_points = SliceExtractor.extract(
            tm, config.slice_z, config.slice_interpolation_density
        )
        
        # Step 3: Filter by radius
        logger.info(f"Filtering points between radii {config.r_inner:.2f} and {config.r_outer:.2f}...")
        filtered_points = PointFilter.by_radius(
            slice_points, config.r_inner, config.r_outer
        )
        logger.info(f"  Kept {len(filtered_points)} / {len(slice_points)} points")
        
        # Step 4: Recenter geometry to slice centroid
        center_xy = slice_points[:, :2].mean(axis=0)
        logger.info(f"  Recentering by offset: [{center_xy[0]:.3f}, {center_xy[1]:.3f}]")
        slice_points[:, :2] -= center_xy
        filtered_points -= center_xy
        mesh.translate(np.array([-center_xy[0], -center_xy[1], 0.0]))
        
        # Step 5: Partition into tooth sectors
        logger.info(f"Partitioning points into {config.n_teeth} tooth clusters...")
        tooth_clusters = ToothClusterer.partition_by_angle(
            filtered_points, config.n_teeth, config.min_points_per_cluster
        )
        non_empty = sum(1 for c in tooth_clusters if len(c) > 0)
        logger.info(f"  Found {non_empty} / {config.n_teeth} non-empty clusters")
        
        # Step 6: Fit BOTH flanks for each tooth
        logger.info("Fitting both flanks for each tooth...")
        tooth_flanks_list, flanks = self._fit_both_flanks(tooth_clusters)
        
        if not tooth_flanks_list:
            raise RuntimeError("Unable to fit flanks for any tooth.")
        logger.info(f"  Successfully fitted both flanks for {len(tooth_flanks_list)} teeth")
        
        # Step 7: Compute tooth bisectors (between left and right flank of SAME tooth)
        logger.info("Computing tooth bisectors...")
        tooth_bisectors = BisectorComputer.compute_tooth_bisectors(
            tooth_flanks_list, config.bisector_length
        )
        logger.info(f"  Computed {len(tooth_bisectors)} tooth bisectors")
        
        # Also compute pair bisectors for backward compatibility
        pair_bisectors = self._convert_to_pair_bisectors(tooth_bisectors)
        
        # Step 8: Ghost circle analysis using tooth bisector intersections
        logger.info("\n--- Ghost Circle Analysis ---")
        ghost_circle = self._fit_ghost_circle_from_tooth_bisectors(tooth_bisectors)
        
        # Step 9: Gear center and offset
        logger.info("\n--- Gear Center Estimation ---")
        gear_center = self._estimate_gear_center(filtered_points, slice_points)
        offset_analysis = self._compute_offset(ghost_circle, gear_center)
        
        # Create result with tooth_flanks included
        analysis_result = AnalysisResult(
            config=config,
            mesh=mesh,
            slice_points=slice_points,
            filtered_points=filtered_points,
            center_shift=center_xy,
            flanks=flanks,  # For backward compatibility (right flanks only)
            bisectors=pair_bisectors,  # Converted tooth bisectors
            ghost_circle=ghost_circle,
            gear_center=gear_center,
            offset_analysis=offset_analysis,
        )
        
        # Attach tooth_flanks as an additional attribute for visualization
        # This contains BOTH left and right flanks for each tooth
        analysis_result.tooth_flanks = tooth_flanks_list
        
        # Log the correct count: each tooth has 2 flanks (left + right)
        total_flanks = len(tooth_flanks_list) * 2
        logger.info(f"Total flanks fitted: {total_flanks} ({len(tooth_flanks_list)} teeth × 2 flanks)")
        
        return analysis_result
    
    def _fit_both_flanks(
        self, 
        tooth_clusters: list[np.ndarray]
    ) -> tuple[list[ToothFlanks], list[FlankLine]]:
        """Fit BOTH flanks (left and right) to each tooth cluster.
        
        Args:
            tooth_clusters: List of point arrays, one per tooth
            
        Returns:
            Tuple of:
            - List of ToothFlanks objects (with both flanks)
            - List of FlankLine objects (for backward compatibility, right flank only)
        """
        config = self.config
        tooth_flanks_list: list[ToothFlanks] = []
        flanks_compat: list[FlankLine] = []  # For backward compatibility
        
        for idx, cluster in enumerate(tooth_clusters, start=1):
            if len(cluster) == 0:
                continue
            
            # Try to fit both flanks
            tooth_flanks = LineFitter.extract_both_flanks(
                cluster, config.min_points_per_flank, tooth_number=idx
            )
            
            if tooth_flanks is not None:
                tooth_flanks_list.append(tooth_flanks)
                
                # Also create a FlankLine for backward compatibility (using right flank)
                flanks_compat.append(FlankLine(
                    tooth=idx,
                    point=tooth_flanks.right_point,
                    direction=tooth_flanks.right_direction,
                    cluster_size=len(cluster),
                ))
                
                logger.debug(f"Tooth {idx}: left={tooth_flanks.left_n_points} pts, "
                           f"right={tooth_flanks.right_n_points} pts")
            else:
                # Fallback: try to fit at least the right flank
                try:
                    point, direction = LineFitter.extract_right_flank(
                        cluster, config.min_points_per_flank
                    )
                    flanks_compat.append(FlankLine(
                        tooth=idx,
                        point=point,
                        direction=unit_vector(direction),
                        cluster_size=len(cluster),
                    ))
                    logger.warning(f"Tooth {idx}: Only right flank fitted (fallback)")
                except ValueError as e:
                    logger.warning(f"Skipping tooth {idx}: {e}")
        
        return tooth_flanks_list, flanks_compat
    
    def _convert_to_pair_bisectors(
        self, 
        tooth_bisectors: list[ToothBisector]
    ) -> list[PairBisector]:
        """Convert ToothBisector objects to PairBisector for backward compatibility.
        
        Args:
            tooth_bisectors: List of ToothBisector objects
            
        Returns:
            List of PairBisector objects
        """
        pair_bisectors: list[PairBisector] = []
        
        for tb in tooth_bisectors:
            pair_bisectors.append(PairBisector(
                between_teeth=(tb.tooth, tb.tooth),  # Same tooth for both flanks
                origin=tb.origin,
                direction=tb.direction,
                length=tb.length,
            ))
        
        return pair_bisectors
    
    def _fit_ghost_circle_from_tooth_bisectors(
        self, 
        tooth_bisectors: list[ToothBisector]
    ) -> Optional[GhostCircle]:
        """Fit ghost circle from tooth bisector intersections.
        
        Args:
            tooth_bisectors: List of ToothBisector objects
            
        Returns:
            GhostCircle object or None if fitting fails
        """
        config = self.config
        
        logger.info(f"Computing tooth bisector intersections in radius range "
                   f"[{config.intersection_r_min:.3f}, {config.intersection_r_max:.3f}]...")
        
        intersections = IntersectionFinder.compute_tooth_bisector_intersections(
            tooth_bisectors,
            config.intersection_r_min,
            config.intersection_r_max,
            config.parallel_threshold
        )
        logger.info(f"  Found {len(intersections)} valid intersections")
        
        if len(intersections) < config.ransac_min_samples:
            logger.warning(f"Not enough intersections ({len(intersections)}) for ghost circle fitting")
            return None
        
        logger.info("Fitting ghost circle with RANSAC...")
        intersections_array = np.array(intersections)
        
        center, radius, inliers, outliers, rmse = CircleFitter.fit_ransac(
            intersections_array,
            config.ransac_min_samples,
            config.ransac_residual_threshold,
            config.ransac_iterations
        )
        
        ghost_circle = GhostCircle(
            center=center,
            radius=radius,
            inliers=inliers,
            outliers=outliers,
            rmse=rmse,
            n_intersections=len(intersections)
        )
        
        logger.info(f"  Ghost circle center: ({center[0]:.4f}, {center[1]:.4f})")
        logger.info(f"  Ghost circle radius: {radius:.4f}")
        logger.info(f"  Inliers: {len(inliers)} / {len(intersections)}")
        logger.info(f"  RMSE: {rmse:.6f}")
        
        return ghost_circle
    
    def _fit_ghost_circle(self, bisectors: list) -> Optional[GhostCircle]:
        """Fit ghost circle from bisector intersections (legacy method).
        
        Args:
            bisectors: List of PairBisector objects
            
        Returns:
            GhostCircle object or None if fitting fails
        """
        config = self.config
        
        logger.info(f"Computing bisector intersections in radius range "
                   f"[{config.intersection_r_min:.3f}, {config.intersection_r_max:.3f}]...")
        
        intersections = IntersectionFinder.compute_bisector_intersections(
            bisectors,
            config.intersection_r_min,
            config.intersection_r_max,
            config.parallel_threshold
        )
        logger.info(f"  Found {len(intersections)} valid intersections")
        
        if len(intersections) < config.ransac_min_samples:
            logger.warning(f"Not enough intersections ({len(intersections)}) for ghost circle fitting")
            return None
        
        logger.info("Fitting ghost circle with RANSAC...")
        intersections_array = np.array(intersections)
        
        center, radius, inliers, outliers, rmse = CircleFitter.fit_ransac(
            intersections_array,
            config.ransac_min_samples,
            config.ransac_residual_threshold,
            config.ransac_iterations
        )
        
        ghost_circle = GhostCircle(
            center=center,
            radius=radius,
            inliers=inliers,
            outliers=outliers,
            rmse=rmse,
            n_intersections=len(intersections)
        )
        
        logger.info(f"  Ghost circle center: ({center[0]:.4f}, {center[1]:.4f})")
        logger.info(f"  Ghost circle radius: {radius:.4f}")
        logger.info(f"  Inliers: {len(inliers)} / {len(intersections)}")
        logger.info(f"  RMSE: {rmse:.6f}")
        
        return ghost_circle
    
    def _estimate_gear_center(
        self,
        filtered_points: np.ndarray,
        slice_points: np.ndarray
    ) -> GearCenter:
        """Estimate gear center using configured method.
        
        Args:
            filtered_points: Points in annular region
            slice_points: All slice points
            
        Returns:
            GearCenter object
        """
        config = self.config
        
        # Method 1: Outer tips
        center_outer, radius_outer = GearCenterEstimator.from_outer_tips(
            filtered_points, config.r_outer
        )
        logger.info(f"Method 1 (outer tips): center=({center_outer[0]:.4f}, {center_outer[1]:.4f}), "
                   f"radius={radius_outer:.4f}")
        
        # Method 2: Boundary centroid
        center_boundary, radius_boundary = GearCenterEstimator.from_boundary_centroid(
            slice_points, config.r_outer
        )
        logger.info(f"Method 2 (boundary centroid): center=({center_boundary[0]:.4f}, "
                   f"{center_boundary[1]:.4f}), radius={radius_boundary:.4f}")
        
        # Select method based on configuration
        use_outer_tips = config.gear_center_method == "outer_tips"
        gear_center = GearCenter(
            center=center_outer if use_outer_tips else center_boundary,
            method=config.gear_center_method,
            radius=radius_outer if use_outer_tips else radius_boundary
        )
        
        logger.info(f"Using method: {gear_center.method}")
        
        return gear_center
    
    def _compute_offset(
        self,
        ghost_circle: Optional[GhostCircle],
        gear_center: GearCenter
    ) -> Optional[OffsetAnalysis]:
        """Compute offset between ghost circle and gear center.
        
        Args:
            ghost_circle: Fitted ghost circle (may be None)
            gear_center: Estimated gear center
            
        Returns:
            OffsetAnalysis object or None if ghost circle is not available
        """
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
        
        logger.info(f"\n--- Offset Analysis (X-axis Setup Error Proxy) ---")
        logger.info(f"  Offset magnitude: {magnitude:.6f}")
        logger.info(f"  Offset angle: {angle_deg:.2f}° from +X axis")
        
        return offset_analysis