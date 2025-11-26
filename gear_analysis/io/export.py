"""
Result export module.

This module provides functions for exporting analysis results
to various file formats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from gear_analysis.models import AnalysisResult

logger = logging.getLogger(__name__)


class ResultExporter:
    """Exports analysis results to various formats.
    
    Currently supported formats:
    - JSON: Complete structured export of all results
    
    Example:
        >>> ResultExporter.to_json(result, Path("report.json"))
    """
    
    @staticmethod
    def to_json(
        result: AnalysisResult,
        output_path: Path,
        indent: int = 2
    ) -> None:
        """Save analysis results to JSON file.
        
        Exports all analysis results including:
        - Configuration parameters
        - Slice statistics
        - Flank line parameters
        - Bisector parameters
        - Ghost circle parameters (if available)
        - Gear center estimation (if available)
        - Offset analysis (if available)
        
        Args:
            result: AnalysisResult from GearAnalyzer
            output_path: Path to output JSON file
            indent: JSON indentation level (default: 2)
        
        Example:
            >>> result = analyzer.run()
            >>> ResultExporter.to_json(result, Path("results/report.json"))
        """
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert result to dictionary
        data = result.to_dict()
        
        # Write JSON file
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=indent)
        
        logger.info(f"Saved JSON report to: {output_path}")
    
    @staticmethod
    def to_csv_summary(
        result: AnalysisResult,
        output_path: Path
    ) -> None:
        """Save a summary of key metrics to CSV.
        
        Creates a simple CSV file with key analysis metrics.
        
        Args:
            result: AnalysisResult from GearAnalyzer
            output_path: Path to output CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = ["metric,value"]
        
        # Add basic metrics
        lines.append(f"n_flanks,{len(result.flanks)}")
        lines.append(f"n_bisectors,{len(result.bisectors)}")
        lines.append(f"slice_points,{len(result.slice_points)}")
        lines.append(f"filtered_points,{len(result.filtered_points)}")
        
        # Add ghost circle metrics if available
        if result.ghost_circle is not None:
            gc = result.ghost_circle
            lines.append(f"ghost_circle_center_x,{gc.center[0]:.6f}")
            lines.append(f"ghost_circle_center_y,{gc.center[1]:.6f}")
            lines.append(f"ghost_circle_radius,{gc.radius:.6f}")
            lines.append(f"ghost_circle_rmse,{gc.rmse:.6f}")
            lines.append(f"ghost_circle_inliers,{len(gc.inliers)}")
        
        # Add offset metrics if available
        if result.offset_analysis is not None:
            oa = result.offset_analysis
            lines.append(f"offset_magnitude,{oa.magnitude:.6f}")
            lines.append(f"offset_angle_deg,{oa.angle_deg:.2f}")
        
        with output_path.open("w", encoding="utf-8") as fp:
            fp.write("\n".join(lines))
        
        logger.info(f"Saved CSV summary to: {output_path}")
    
    @staticmethod
    def to_flank_csv(
        result: AnalysisResult,
        output_path: Path
    ) -> None:
        """Export flank line data to CSV.
        
        Args:
            result: AnalysisResult from GearAnalyzer
            output_path: Path to output CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = ["tooth,point_x,point_y,direction_x,direction_y,cluster_size"]
        
        for flank in result.flanks:
            lines.append(
                f"{flank.tooth},"
                f"{flank.point[0]:.6f},{flank.point[1]:.6f},"
                f"{flank.direction[0]:.6f},{flank.direction[1]:.6f},"
                f"{flank.cluster_size}"
            )
        
        with output_path.open("w", encoding="utf-8") as fp:
            fp.write("\n".join(lines))
        
        logger.info(f"Saved flank data to: {output_path}")
