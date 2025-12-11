"""
Tooth clustering module - improved version with validation.

This module partitions gear slice points into tooth clusters,
with better handling of edge cases and misalignment.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from gear_analysis.utils import normalize_angle, compute_angles

logger = logging.getLogger(__name__)

Points2D = NDArray[np.floating]


class ToothClusterer:
    """Partitions points into tooth clusters with validation."""
    
    @staticmethod
    def partition_by_angle(
        points: Points2D,
        n_teeth: int,
        min_points: int
    ) -> list[np.ndarray]:
        """Partition points into n_teeth angular bins.
        
        Args:
            points: 2D points (N, 2)
            n_teeth: Expected number of teeth
            min_points: Minimum points per valid cluster
            
        Returns:
            List of n_teeth point arrays (some may be empty)
        """
        if len(points) == 0 or n_teeth < 1:
            return [np.empty((0, 2)) for _ in range(n_teeth)]
        
        # Calculate polar angles [0, 2π)
        angles = normalize_angle(compute_angles(points))
        
        # Create bin edges
        edges = np.linspace(0.0, 2 * np.pi, n_teeth + 1)
        
        # Assign points to bins
        bin_idx = np.digitize(angles, edges, right=False) - 1
        bin_idx[bin_idx == n_teeth] = 0  # Wrap last bin
        
        # Extract clusters
        clusters = []
        for k in range(n_teeth):
            pts = points[bin_idx == k]
            if len(pts) >= min_points:
                clusters.append(pts)
            else:
                clusters.append(np.empty((0, 2)))
        
        return clusters
    
    @classmethod
    def partition_adaptive(
        cls,
        points: Points2D,
        n_teeth: int,
        min_points: int
    ) -> list[np.ndarray]:
        """Partition with automatic bin offset detection.
        
        Tries multiple angular offsets to find the best alignment
        where bins don't split teeth in half.
        
        Args:
            points: 2D points (N, 2)
            n_teeth: Expected number of teeth
            min_points: Minimum points per valid cluster
            
        Returns:
            List of n_teeth point arrays
        """
        if len(points) == 0 or n_teeth < 1:
            return [np.empty((0, 2)) for _ in range(n_teeth)]
        
        angles = normalize_angle(compute_angles(points))
        
        bin_width = 2 * np.pi / n_teeth
        best_score = -1
        best_offset = 0
        
        # Try different offsets (10 steps within one bin width)
        for offset_frac in np.linspace(0, 1, 10, endpoint=False):
            offset = offset_frac * bin_width
            edges = np.linspace(offset, 2 * np.pi + offset, n_teeth + 1)
            edges = edges % (2 * np.pi)
            
            # Wrap angles with offset
            shifted_angles = (angles - offset + 2 * np.pi) % (2 * np.pi)
            bin_idx = np.digitize(shifted_angles, np.linspace(0, 2*np.pi, n_teeth+1)) - 1
            bin_idx[bin_idx == n_teeth] = 0
            
            # Score: prefer when bins have similar point counts
            counts = np.array([np.sum(bin_idx == k) for k in range(n_teeth)])
            valid_counts = counts[counts >= min_points]
            
            if len(valid_counts) == 0:
                continue
            
            # Score = number of valid bins - variance in counts
            score = len(valid_counts) - np.std(valid_counts) / (np.mean(valid_counts) + 1)
            
            if score > best_score:
                best_score = score
                best_offset = offset
        
        # Use best offset
        shifted_angles = (angles - best_offset + 2 * np.pi) % (2 * np.pi)
        bin_idx = np.digitize(shifted_angles, np.linspace(0, 2*np.pi, n_teeth+1)) - 1
        bin_idx[bin_idx == n_teeth] = 0
        
        clusters = []
        for k in range(n_teeth):
            pts = points[bin_idx == k]
            if len(pts) >= min_points:
                clusters.append(pts)
            else:
                clusters.append(np.empty((0, 2)))
        
        logger.debug(f"Best angular offset: {np.degrees(best_offset):.1f}°")
        
        return clusters
    
    @staticmethod
    def validate_cluster(
        cluster: Points2D,
        expected_span_deg: float,
        min_points: int
    ) -> tuple[bool, str]:
        """Validate a tooth cluster.
        
        Args:
            cluster: Points in the cluster
            expected_span_deg: Expected angular span in degrees
            min_points: Minimum required points
            
        Returns:
            Tuple of (is_valid, issue_description)
        """
        if len(cluster) < min_points:
            return False, f"Too few points ({len(cluster)})"
        
        # Compute angular span
        angles = np.degrees(np.arctan2(cluster[:, 1], cluster[:, 0]))
        angles = (angles + 360) % 360
        
        # Handle wraparound
        angle_range = np.max(angles) - np.min(angles)
        if angle_range > 180:
            angles[angles < 180] += 360
            angle_range = np.max(angles) - np.min(angles)
        
        if angle_range > expected_span_deg * 1.8:
            return False, f"Too wide ({angle_range:.1f}°)"
        
        if angle_range < expected_span_deg * 0.2:
            return False, f"Too narrow ({angle_range:.1f}°)"
        
        return True, ""
    
    @classmethod
    def find_gaps(
        cls,
        points: Points2D,
        n_teeth: int
    ) -> np.ndarray:
        """Find angular positions of gaps between teeth.
        
        This can be used to better align bin boundaries with gaps
        instead of splitting teeth.
        
        Args:
            points: 2D points
            n_teeth: Expected number of teeth
            
        Returns:
            Array of gap angles in radians
        """
        if len(points) < 10:
            return np.array([])
        
        # Create fine histogram of angles
        angles = normalize_angle(compute_angles(points))
        
        n_bins = n_teeth * 10  # 10 bins per tooth
        hist, edges = np.histogram(angles, bins=n_bins, range=(0, 2*np.pi))
        bin_centers = (edges[:-1] + edges[1:]) / 2
        
        # Smooth histogram
        kernel = np.ones(3) / 3
        hist_smooth = np.convolve(hist, kernel, mode='same')
        
        # Find local minima (gaps)
        gaps = []
        for i in range(1, len(hist_smooth) - 1):
            if hist_smooth[i] < hist_smooth[i-1] and hist_smooth[i] < hist_smooth[i+1]:
                if hist_smooth[i] < np.mean(hist_smooth) * 0.5:  # Significant dip
                    gaps.append(bin_centers[i])
        
        return np.array(gaps)
    
    @classmethod
    def partition_by_gaps(
        cls,
        points: Points2D,
        n_teeth: int,
        min_points: int
    ) -> list[np.ndarray]:
        """Partition using detected gaps between teeth.
        
        More robust than fixed bins when teeth don't align with
        expected positions.
        
        Args:
            points: 2D points
            n_teeth: Expected number of teeth
            min_points: Minimum points per cluster
            
        Returns:
            List of point arrays (one per tooth)
        """
        gaps = cls.find_gaps(points, n_teeth)
        
        if len(gaps) < n_teeth - 2:
            # Not enough gaps found, fall back to adaptive
            logger.warning(f"Only found {len(gaps)} gaps, expected ~{n_teeth}. Using adaptive binning.")
            return cls.partition_adaptive(points, n_teeth, min_points)
        
        # Sort gaps
        gaps = np.sort(gaps)
        
        # Use gaps as bin edges
        angles = normalize_angle(compute_angles(points))
        
        # Assign points to bins between gaps
        bin_idx = np.digitize(angles, gaps) % len(gaps)
        
        clusters = []
        for k in range(len(gaps)):
            pts = points[bin_idx == k]
            if len(pts) >= min_points:
                clusters.append(pts)
            else:
                clusters.append(np.empty((0, 2)))
        
        # Pad or trim to n_teeth
        while len(clusters) < n_teeth:
            clusters.append(np.empty((0, 2)))
        clusters = clusters[:n_teeth]
        
        return clusters