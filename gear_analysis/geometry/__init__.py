"""
Geometry processing subpackage.

This subpackage provides geometric algorithms for:
- Point filtering by radius
- Tooth clustering by angular position
- Line fitting using SVD
- Circle fitting (Kåsa, Taubin, RANSAC)
- Bisector computation and intersection
"""

from gear_analysis.geometry.filtering import PointFilter
from gear_analysis.geometry.clustering import ToothClusterer
from gear_analysis.geometry.line_fitting import LineFitter
from gear_analysis.geometry.circle_fitting import CircleFitter
from gear_analysis.geometry.bisectors import BisectorComputer, IntersectionFinder

__all__ = [
    "PointFilter",
    "ToothClusterer",
    "LineFitter",
    "CircleFitter",
    "BisectorComputer",
    "IntersectionFinder",
]
