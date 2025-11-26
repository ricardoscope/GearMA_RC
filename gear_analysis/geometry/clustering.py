"""
Tooth clustering module.

This module provides algorithms for partitioning gear slice points
into individual tooth clusters based on angular position.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type alias
Points2D = NDArray[np.floating]


class ToothClusterer:
    """Partitions points into tooth clusters based on angular position.
    
    This class provides methods for dividing a gear's cross-section
    points into clusters, one per tooth, using angular binning.
    
    Example:
        >>> points = np.random.randn(1000, 2) * 2.5
        >>> clusters = ToothClusterer.partition_by_angle(points, n_teeth=38, min_points=5)
        >>> len(clusters)
        38
    """
    
    @staticmethod
    def partition_by_angle(
        points_xy: Points2D,
        n_teeth: int,
        min_points_per_cluster: int
    ) -> list[Points2D]:
        """Partition points into exactly n_teeth angular sectors.
        
        Divides the full circle (2π) into n_teeth equal angular bins,
        one per tooth. Points are assigned to bins based on their
        polar angle from the origin.
        
        This method guarantees exactly n_teeth clusters in the output,
        though some may be empty if they contain fewer than
        min_points_per_cluster points.
        
        Args:
            points_xy: 2D point array, shape (N, 2)
            n_teeth: Expected number of teeth (number of bins to create)
            min_points_per_cluster: Minimum points for a valid cluster.
                                   Clusters with fewer points are returned
                                   as empty arrays.
            
        Returns:
            List of n_teeth arrays. Each array contains points belonging
            to that tooth, or is empty if the cluster is invalid.
        
        Example:
            >>> # Create synthetic gear-like point distribution
            >>> n_teeth = 38
            >>> points_per_tooth = 50
            >>> all_points = []
            >>> for i in range(n_teeth):
            ...     angle = 2 * np.pi * i / n_teeth
            ...     r = 2.5 + 0.1 * np.random.randn(points_per_tooth)
            ...     theta = angle + 0.05 * np.random.randn(points_per_tooth)
            ...     pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            ...     all_points.append(pts)
            >>> points = np.vstack(all_points)
            >>> clusters = ToothClusterer.partition_by_angle(points, n_teeth, min_points=5)
            >>> non_empty = sum(1 for c in clusters if len(c) > 0)
            >>> non_empty
            38
        """
        if len(points_xy) == 0 or n_teeth < 1:
            return []
        
        # Calculate polar angles and normalize to [0, 2π)
        angles = np.arctan2(points_xy[:, 1], points_xy[:, 0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        
        # Create angular bin edges (n_teeth + 1 edges for n_teeth bins)
        edges = np.linspace(0.0, 2 * np.pi, n_teeth + 1)
        
        # Assign points to bins using digitize
        # digitize returns indices 1 to n_teeth+1, we want 0 to n_teeth-1
        bin_idx = np.digitize(angles, edges, right=False) - 1
        # Handle edge case: points exactly at 2π should go to bin 0
        bin_idx[bin_idx == n_teeth] = 0
        
        # Extract points for each tooth
        clusters: list[Points2D] = []
        for k in range(n_teeth):
            pts = points_xy[bin_idx == k]
            if len(pts) >= min_points_per_cluster:
                clusters.append(pts)
            else:
                # Return empty array to preserve tooth index alignment
                clusters.append(np.empty((0, 2)))
        
        non_empty = sum(1 for c in clusters if len(c) > 0)
        logger.debug(f"Partitioned into {non_empty}/{n_teeth} non-empty clusters")
        
        return clusters
    
    @staticmethod
    def partition_by_angle_with_offset(
        points_xy: Points2D,
        n_teeth: int,
        min_points_per_cluster: int,
        angular_offset: float = 0.0
    ) -> list[Points2D]:
        """Partition points with an angular offset for bin boundaries.
        
        Same as partition_by_angle but allows shifting the bin boundaries
        by an angular offset. This can be useful when the default binning
        splits a tooth across two bins.
        
        Args:
            points_xy: 2D point array, shape (N, 2)
            n_teeth: Expected number of teeth
            min_points_per_cluster: Minimum points for a valid cluster
            angular_offset: Offset in radians to shift bin boundaries
            
        Returns:
            List of n_teeth arrays
        
        Example:
            >>> # Shift bins by half a tooth width
            >>> tooth_width = 2 * np.pi / n_teeth
            >>> clusters = ToothClusterer.partition_by_angle_with_offset(
            ...     points, n_teeth, min_points=5, angular_offset=tooth_width/2
            ... )
        """
        if len(points_xy) == 0 or n_teeth < 1:
            return []
        
        # Calculate polar angles with offset
        angles = np.arctan2(points_xy[:, 1], points_xy[:, 0])
        angles = (angles - angular_offset + 2 * np.pi) % (2 * np.pi)
        
        edges = np.linspace(0.0, 2 * np.pi, n_teeth + 1)
        bin_idx = np.digitize(angles, edges, right=False) - 1
        bin_idx[bin_idx == n_teeth] = 0
        
        clusters: list[Points2D] = []
        for k in range(n_teeth):
            pts = points_xy[bin_idx == k]
            if len(pts) >= min_points_per_cluster:
                clusters.append(pts)
            else:
                clusters.append(np.empty((0, 2)))
        
        return clusters
    
    @staticmethod
    def get_cluster_statistics(clusters: list[Points2D]) -> dict:
        """Compute statistics about the tooth clusters.
        
        Args:
            clusters: List of point arrays from partition_by_angle
            
        Returns:
            Dictionary with cluster statistics
        """
        sizes = [len(c) for c in clusters]
        non_empty_sizes = [s for s in sizes if s > 0]
        
        return {
            "n_clusters": len(clusters),
            "n_non_empty": len(non_empty_sizes),
            "n_empty": len(sizes) - len(non_empty_sizes),
            "min_size": min(non_empty_sizes) if non_empty_sizes else 0,
            "max_size": max(non_empty_sizes) if non_empty_sizes else 0,
            "mean_size": np.mean(non_empty_sizes) if non_empty_sizes else 0,
            "total_points": sum(sizes),
        }
