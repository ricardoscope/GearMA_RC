"""
2D visualization module.

This module provides matplotlib-based 2D visualization of gear
analysis results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from gear_analysis.models import AnalysisResult, PairBisector
from gear_analysis.utils import unit_vector

logger = logging.getLogger(__name__)

# Type alias
Vector2D = NDArray[np.floating]


def _bisector_display_segment(
    bisector: PairBisector,
    r_outer: float,
    length_scale: float = 0.8  # 20% shorter
) -> tuple[Vector2D, Vector2D]:
    """Calculate start/end points for displaying a bisector line.
    
    The bisector is drawn from near the flank midpoint toward
    the gear center for a fixed length.
    
    Args:
        bisector: Bisector to display
        r_outer: Outer radius for clipping
        length_scale: Scale factor for bisector length (default 0.8 = 20% shorter)
        
    Returns:
        Tuple of (start_point, end_point) as 2D arrays
    """
    direction = unit_vector(bisector.direction)
    if np.linalg.norm(direction) < 1e-10:
        direction = np.array([1.0, 0.0])
    
    # Ensure direction points inward (toward center)
    if np.dot(direction, bisector.origin) > 0:
        direction = -direction
    
    start = bisector.origin.copy()
    start_radius = np.linalg.norm(start)
    
    # If start is outside r_outer, clip to the circle
    if start_radius > r_outer:
        # Find intersection with outer circle
        d = direction
        p = bisector.origin
        a = np.dot(d, d)
        b = 2 * np.dot(p, d)
        c = np.dot(p, p) - r_outer**2
        discriminant = b**2 - 4*a*c
        
        if discriminant > 0:
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2*a)
            t2 = (-b + sqrt_disc) / (2*a)
            # Pick the closer intersection
            p1 = p + t1 * d
            p2 = p + t2 * d
            if np.linalg.norm(p1 - bisector.origin) < np.linalg.norm(p2 - bisector.origin):
                start = p1
            else:
                start = p2
    
    # Apply length scaling (20% shorter)
    scaled_length = bisector.length * length_scale
    end = start + direction * scaled_length
    return start, end


def plot_2d_analysis(
    result: AnalysisResult,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 12),
    dpi: int = 150,
    show_legend: bool = True,
    show_grid: bool = True
) -> plt.Figure:
    """Create comprehensive 2D visualization of the analysis.
    
    The plot shows:
    - Slice boundary points (light gray)
    - Filtered points in annular region (darker)
    - Inner/outer radius reference circles
    - Tooth flank lines: right flanks (green, tooth 1 in red), left flanks (cyan)
    - Bisector lines (black)
    - Bisector intersections (inliers marked)
    - Ghost circle (if fitted)
    - Gear center and offset vector
    
    Args:
        result: AnalysisResult from GearAnalyzer
        output_path: If provided, save figure to this path
        figsize: Figure size in inches (width, height)
        dpi: Resolution for saved figure
        show_legend: Whether to show the legend
        show_grid: Whether to show grid lines
        
    Returns:
        matplotlib Figure object
    
    Example:
        >>> result = analyzer.run()
        >>> fig = plot_2d_analysis(result, Path("analysis.png"))
    """
    config = result.config
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot slice points (light gray background)
    ax.scatter(
        result.slice_points[:, 0],
        result.slice_points[:, 1],
        s=1,
        alpha=0.3,
        label='Slice boundary'
    )
    
    # Plot filtered points (darker)
    ax.scatter(
        result.filtered_points[:, 0],
        result.filtered_points[:, 1],
        s=2,
        alpha=0.5,
        label='Filtered points'
    )
    
    # Plot reference circles
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(
        config.r_inner * np.cos(theta),
        config.r_inner * np.sin(theta),
        '--',
        linewidth=1.5,
        label=f'Inner radius ({config.r_inner})'
    )
    ax.plot(
        config.r_outer * np.cos(theta),
        config.r_outer * np.sin(theta),
        '--',
        linewidth=1.5,
        label=f'Outer radius ({config.r_outer})'
    )
    
    # Plot RIGHT flank lines (from result.flanks - original behavior)
    for flank in result.flanks:
        start = flank.point - config.flank_segment_length * flank.direction
        end = flank.point + config.flank_segment_length * flank.direction
        
        if flank.tooth == 1:
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                'r-',
                linewidth=2,
                label='Tooth 1 right flank'
            )
        else:
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                'g-',
                linewidth=1,
                alpha=0.7
            )
    
    # Plot LEFT flank lines (from result.tooth_flanks if available)
    left_flank_plotted = False
    if hasattr(result, 'tooth_flanks') and result.tooth_flanks:
        for tf in result.tooth_flanks:
            start = tf.left_point - config.flank_segment_length * tf.left_direction
            end = tf.left_point + config.flank_segment_length * tf.left_direction
            
            if tf.tooth == 1:
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    'm-',  # Magenta for tooth 1 left flank
                    linewidth=2,
                    label='Tooth 1 left flank'
                )
            else:
                label = 'Left flanks' if not left_flank_plotted else ''
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    'c-',  # Cyan for left flanks
                    linewidth=1,
                    alpha=0.7,
                    label=label
                )
                left_flank_plotted = True
    
    # Plot bisectors (20% shorter)
    for i, bisector in enumerate(result.bisectors):
        start, end = _bisector_display_segment(bisector, config.r_outer, length_scale=0.8)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            'k-',
            linewidth=1,
            alpha=0.6,
            label='Bisectors' if i == 0 else ''
        )
    
    # Plot ghost circle analysis
    if result.ghost_circle is not None:
        gc = result.ghost_circle
        
        # Plot inlier intersection points
        if len(gc.inliers) > 0:
            ax.scatter(
                gc.inliers[:, 0],
                gc.inliers[:, 1],
                s=50,
                marker='o',
                linewidths=2,
                facecolors='none',
                edgecolors='blue',
                label=f'Inlier intersections ({len(gc.inliers)})'
            )
        
        # Plot ghost circle
        circle_x = gc.center[0] + gc.radius * np.cos(theta)
        circle_y = gc.center[1] + gc.radius * np.sin(theta)
        ax.plot(
            circle_x,
            circle_y,
            'b-',
            linewidth=2.5,
            label=f'Ghost circle (r={gc.radius:.3f}, RMSE={gc.rmse:.4f})'
        )
        
        # Annotate radius
        ax.annotate(
            f"r = {gc.radius:.3f}",
            xy=(gc.center[0] + gc.radius, gc.center[1]),
            xytext=(gc.center[0] + gc.radius + 0.3, gc.center[1]),
            fontsize=9,
            color="#0b5394",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="->", linewidth=1, color="#0b5394"),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )
        
        # Plot ghost circle center
        ax.scatter(
            [gc.center[0]],
            [gc.center[1]],
            s=100,
            marker='+',
            linewidths=3,
            c='blue',
            label='Ghost circle center'
        )
    
    # Plot offset vector
    if result.gear_center is not None and result.offset_analysis is not None:
        gc_center = result.gear_center.center
        offset = result.offset_analysis
        
        ax.arrow(
            gc_center[0],
            gc_center[1],
            offset.offset_vector[0],
            offset.offset_vector[1],
            head_width=0.05,
            head_length=0.03,
            linewidth=2,
            fc='red',
            ec='red',
            length_includes_head=True,
            label=f'Offset ({offset.magnitude:.4f} @ {offset.angle_deg:.1f}°)'
        )
    
    # Configure axes
    ax.set_aspect('equal')
    if show_grid:
        ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Gear Flank Analysis with Ghost Circle')
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved 2D plot to: {output_path}")
    
    return fig


def plot_tooth_details(
    result: AnalysisResult,
    tooth_numbers: Optional[list[int]] = None,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Plot detailed view of specific teeth.
    
    Args:
        result: AnalysisResult from GearAnalyzer
        tooth_numbers: List of tooth numbers to show (default: first 4)
        output_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure object
    """
    if tooth_numbers is None:
        tooth_numbers = [1, 2, 3, 4]
    
    n_teeth = len(tooth_numbers)
    fig, axes = plt.subplots(1, n_teeth, figsize=(4*n_teeth, 4))
    if n_teeth == 1:
        axes = [axes]
    
    flank_dict = {f.tooth: f for f in result.flanks}
    
    # Also get tooth_flanks if available
    tooth_flanks_dict = {}
    if hasattr(result, 'tooth_flanks') and result.tooth_flanks:
        tooth_flanks_dict = {tf.tooth: tf for tf in result.tooth_flanks}
    
    for ax, tooth_num in zip(axes, tooth_numbers):
        if tooth_num in flank_dict:
            flank = flank_dict[tooth_num]
            
            # Plot right flank line
            start = flank.point - 0.3 * flank.direction
            end = flank.point + 0.3 * flank.direction
            ax.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=2, label='Right')
            ax.scatter([flank.point[0]], [flank.point[1]], s=50, c='green')
            
            # Plot left flank if available
            if tooth_num in tooth_flanks_dict:
                tf = tooth_flanks_dict[tooth_num]
                start = tf.left_point - 0.3 * tf.left_direction
                end = tf.left_point + 0.3 * tf.left_direction
                ax.plot([start[0], end[0]], [start[1], end[1]], 'c-', linewidth=2, label='Left')
                ax.scatter([tf.left_point[0]], [tf.left_point[1]], s=50, c='cyan')
            
            ax.set_title(f'Tooth {tooth_num} ({flank.cluster_size} pts)')
            ax.legend(fontsize=7)
        else:
            ax.set_title(f'Tooth {tooth_num} (no data)')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig