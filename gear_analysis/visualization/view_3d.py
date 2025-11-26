"""
3D Visualization module.

This module provides functions for creating 3D visualizations
of gear analysis results using Open3D.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

if TYPE_CHECKING:
    from gear_analysis.models import AnalysisResult

logger = logging.getLogger(__name__)


class Visualizer3D:
    """3D visualization handler using Open3D.
    
    This class provides methods for creating and displaying
    3D visualizations of gear geometry and analysis results.
    
    Example:
        >>> vis = Visualizer3D()
        >>> vis.add_mesh(result.mesh)
        >>> vis.add_ghost_circle(result.ghost_circle, z_height=-0.2)
        >>> vis.show()
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required for 3D visualization. "
                            "Install with: pip install open3d")
        
        self.geometries: list = []
    
    def add_mesh(
        self,
        mesh,
        color: tuple[float, float, float] = (0.7, 0.7, 0.8),
        show_wireframe: bool = False
    ) -> None:
        """Add a mesh to the visualization.
        
        Args:
            mesh: Open3D TriangleMesh object
            color: RGB color tuple (0-1 range)
            show_wireframe: Whether to show wireframe overlay
        """
        # Create a copy to avoid modifying the original
        vis_mesh = o3d.geometry.TriangleMesh(mesh)
        
        # Compute normals for proper shading
        vis_mesh.compute_vertex_normals()
        vis_mesh.compute_triangle_normals()
        
        # Paint the mesh with a uniform color
        vis_mesh.paint_uniform_color(color)
        
        self.geometries.append(vis_mesh)
        
        if show_wireframe:
            wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(vis_mesh)
            wireframe.paint_uniform_color((0.3, 0.3, 0.3))
            self.geometries.append(wireframe)
    
    def add_points(
        self,
        points: np.ndarray,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        point_size: float = 5.0
    ) -> None:
        """Add points to the visualization.
        
        Args:
            points: Nx3 array of 3D points
            color: RGB color tuple
            point_size: Size of points (note: may not work in all viewers)
        """
        if len(points) == 0:
            return
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        
        self.geometries.append(pcd)
    
    def add_circle(
        self,
        center: np.ndarray,
        radius: float,
        z_height: float,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        n_segments: int = 100
    ) -> None:
        """Add a circle to the visualization.
        
        Args:
            center: 2D center point (x, y)
            radius: Circle radius
            z_height: Z coordinate for the circle
            color: RGB color tuple
            n_segments: Number of line segments
        """
        # Create circle points
        angles = np.linspace(0, 2 * np.pi, n_segments + 1)
        points = np.zeros((n_segments + 1, 3))
        points[:, 0] = center[0] + radius * np.cos(angles)
        points[:, 1] = center[1] + radius * np.sin(angles)
        points[:, 2] = z_height
        
        # Create line set
        lines = [[i, i + 1] for i in range(n_segments)]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        
        self.geometries.append(line_set)
    
    def add_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        color: tuple[float, float, float] = (0.0, 1.0, 0.0)
    ) -> None:
        """Add a line segment to the visualization.
        
        Args:
            start: 3D start point
            end: 3D end point
            color: RGB color tuple
        """
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector([start, end])
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_set.paint_uniform_color(color)
        
        self.geometries.append(line_set)
    
    def add_coordinate_frame(self, size: float = 1.0) -> None:
        """Add a coordinate frame at the origin.
        
        Args:
            size: Size of the coordinate frame axes
        """
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self.geometries.append(frame)
    
    def show(self, window_name: str = "Gear Analysis 3D") -> None:
        """Display the visualization with proper lighting.
        
        Args:
            window_name: Title of the visualization window
        """
        if not self.geometries:
            logger.warning("No geometries to display")
            return
        
        # Create visualizer with proper settings
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1280, height=960)
        
        # Add all geometries
        for geom in self.geometries:
            vis.add_geometry(geom)
        
        # Get render options and configure for better visibility
        render_opt = vis.get_render_option()
        render_opt.background_color = np.array([1.0, 1.0, 1.0])  # White background (matching 2D plot)
        render_opt.point_size = 5.0
        render_opt.line_width = 2.0
        render_opt.light_on = True
        render_opt.mesh_show_back_face = True
        # Smooth shading is achieved through computed vertex normals
        
        # Set up the view
        view_ctrl = vis.get_view_control()
        view_ctrl.set_zoom(0.8)
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()


def build_3d_geometries(
    result: "AnalysisResult",
    z_height: Optional[float] = None
) -> list:
    """Build Open3D geometries from analysis result.
    
    Args:
        result: AnalysisResult object
        z_height: Z coordinate for 2D elements (default: config.slice_z)
        
    Returns:
        List of Open3D geometry objects
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D required for 3D visualization")
    
    geometries = []
    
    if z_height is None:
        z_height = result.config.slice_z
    
    # Add mesh with proper shading and silver/metallic color
    if result.mesh is not None:
        mesh = o3d.geometry.TriangleMesh(result.mesh)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # Silver/metallic color (light gray with slight blue tint)
        # RGB values: [0.75, 0.75, 0.8] for a silver appearance
        mesh.paint_uniform_color([0.75, 0.75, 0.8])
        geometries.append(mesh)
    
    # Add slice points (matching 2D plot: darker gray for filtered points)
    if result.filtered_points is not None and len(result.filtered_points) > 0:
        # Convert 2D to 3D
        points_3d = np.zeros((len(result.filtered_points), 3))
        points_3d[:, :2] = result.filtered_points
        points_3d[:, 2] = z_height
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        # Darker gray to match 2D plot filtered points
        pcd.paint_uniform_color([0.4, 0.4, 0.4])
        geometries.append(pcd)
    
    # Add RIGHT flank lines (matching 2D: green for most, red for tooth 1)
    if result.flanks:
        for flank in result.flanks:
            center_3d = np.array([flank.point[0], flank.point[1], z_height])
            direction_3d = np.array([flank.direction[0], flank.direction[1], 0.0])
            
            # Use config flank segment length if available
            length = getattr(result.config, 'flank_segment_length', 0.15)
            start = center_3d - length * direction_3d
            end = center_3d + length * direction_3d
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([start, end])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            
            # Match 2D colors: red for tooth 1, green for others
            if flank.tooth == 1:
                line_set.paint_uniform_color([1.0, 0.0, 0.0])  # Red
            else:
                line_set.paint_uniform_color([0.0, 0.8, 0.0])  # Green
            geometries.append(line_set)
    
    # Add LEFT flank lines (matching 2D: cyan for most, magenta for tooth 1)
    if hasattr(result, 'tooth_flanks') and result.tooth_flanks:
        for tf in result.tooth_flanks:
            left_point_3d = np.array([tf.left_point[0], tf.left_point[1], z_height])
            left_direction_3d = np.array([tf.left_direction[0], tf.left_direction[1], 0.0])
            
            length = getattr(result.config, 'flank_segment_length', 0.15)
            start = left_point_3d - length * left_direction_3d
            end = left_point_3d + length * left_direction_3d
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([start, end])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            
            # Match 2D colors: magenta for tooth 1, cyan for others
            if tf.tooth == 1:
                line_set.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta
            else:
                line_set.paint_uniform_color([0.0, 1.0, 1.0])  # Cyan
            geometries.append(line_set)
    
    # Add bisectors (matching 2D: black lines, 20% shorter)
    if result.bisectors:
        from gear_analysis.utils import unit_vector
        r_outer = result.config.r_outer
        
        for bisector in result.bisectors:
            # Use the same bisector display logic as 2D plot
            direction = unit_vector(bisector.direction)
            if np.linalg.norm(direction) < 1e-10:
                direction = np.array([1.0, 0.0])
            
            # Ensure direction points inward (toward center)
            if np.dot(direction, bisector.origin) > 0:
                direction = -direction
            
            start_2d = bisector.origin.copy()
            start_radius = np.linalg.norm(start_2d)
            
            # If start is outside r_outer, clip to the circle
            if start_radius > r_outer:
                # Find intersection with outer circle
                a = np.dot(direction, direction)
                b = 2.0 * np.dot(bisector.origin, direction)
                c = np.dot(bisector.origin, bisector.origin) - r_outer**2
                discriminant = b**2 - 4 * a * c
                if discriminant > 0:
                    sqrt_disc = np.sqrt(discriminant)
                    t1 = (-b - sqrt_disc) / (2 * a)
                    t2 = (-b + sqrt_disc) / (2 * a)
                    # Use the point closer to origin
                    p1 = bisector.origin + t1 * direction
                    p2 = bisector.origin + t2 * direction
                    start_2d = p1 if np.linalg.norm(p1) < np.linalg.norm(p2) else p2
            
            # Scale length by 0.8 (20% shorter as in 2D plot)
            scaled_length = bisector.length * 0.8
            end_2d = start_2d + direction * scaled_length
            
            # Convert to 3D
            start_3d = np.array([start_2d[0], start_2d[1], z_height])
            end_3d = np.array([end_2d[0], end_2d[1], z_height])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([start_3d, end_3d])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color([0.0, 0.0, 0.0])  # Black bisectors (matching 2D)
            geometries.append(line_set)
    
    # Add ghost circle
    if result.ghost_circle is not None:
        gc = result.ghost_circle
        n_segments = 100
        angles = np.linspace(0, 2 * np.pi, n_segments + 1)
        
        points = np.zeros((n_segments + 1, 3))
        points[:, 0] = gc.center[0] + gc.radius * np.cos(angles)
        points[:, 1] = gc.center[1] + gc.radius * np.sin(angles)
        points[:, 2] = z_height
        
        lines = [[i, i + 1] for i in range(n_segments)]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        # Match 2D plot ghost circle color: blue (#0b5394 = RGB 0.043, 0.329, 0.580)
        line_set.paint_uniform_color([0.043, 0.329, 0.580])
        geometries.append(line_set)
        
        # Add ghost circle center marker (small sphere) - matching 2D plot
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        center_sphere.translate([gc.center[0], gc.center[1], z_height])
        center_sphere.paint_uniform_color([0.043, 0.329, 0.580])  # Same blue as circle
        center_sphere.compute_vertex_normals()
        geometries.append(center_sphere)
        
        # Add inliers (matching 2D: hollow circles, use white/light color)
        if gc.inliers is not None and len(gc.inliers) > 0:
            inliers_3d = np.zeros((len(gc.inliers), 3))
            inliers_3d[:, :2] = gc.inliers
            inliers_3d[:, 2] = z_height
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(inliers_3d)
            # Light color for inlier points (hollow circles in 2D)
            pcd.paint_uniform_color([0.8, 0.8, 0.9])
            geometries.append(pcd)
    
    # Add gear center marker
    if result.gear_center is not None:
        center = result.gear_center.center
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
        center_sphere.translate([center[0], center[1], z_height])
        center_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue gear center
        center_sphere.compute_vertex_normals()
        geometries.append(center_sphere)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    coord_frame.translate([0, 0, z_height])
    geometries.append(coord_frame)
    
    return geometries


def show_3d_visualization(
    result: "AnalysisResult",
    window_name: str = "Gear Analysis 3D"
) -> None:
    """Display 3D visualization of analysis results with proper lighting.
    
    Args:
        result: AnalysisResult object
        window_name: Window title
    """
    if not HAS_OPEN3D:
        logger.error("Open3D not installed. Install with: pip install open3d")
        return
    
    logger.info("Building 3D visualization...")
    
    geometries = build_3d_geometries(result)
    
    if not geometries:
        logger.warning("No geometries to display")
        return
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=1000)
    
    # Add geometries
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Configure render options for proper visibility
    render_opt = vis.get_render_option()
    
    # White background (matching 2D plot)
    render_opt.background_color = np.array([1.0, 1.0, 1.0])
    
    # Enable lighting for proper shading
    render_opt.light_on = True
    
    # Point and line sizes (thicker for visibility)
    render_opt.point_size = 6.0
    render_opt.line_width = 3.0
    
    # Mesh rendering options - ensure proper shading
    render_opt.mesh_show_back_face = True
    # Smooth shading is achieved through computed vertex normals
    
    # Set view control
    view_ctrl = vis.get_view_control()
    view_ctrl.set_zoom(0.6)
    
    # Set front view (looking down at the gear)
    view_ctrl.set_front([0, 0, -1])
    view_ctrl.set_up([0, 1, 0])
    view_ctrl.set_lookat([0, 0, result.config.slice_z])
    
    logger.info("3D visualization ready. Close the window to continue.")
    
    # Run visualization
    vis.run()
    vis.destroy_window()


def show_3d_with_custom_lighting(
    result: "AnalysisResult",
    window_name: str = "Gear Analysis 3D (Enhanced)"
) -> None:
    """Display 3D visualization with enhanced custom lighting setup.
    
    This provides better contrast and visibility than the default lighting.
    
    Args:
        result: AnalysisResult object
        window_name: Window title
    """
    if not HAS_OPEN3D:
        logger.error("Open3D not installed")
        return
    
    geometries = build_3d_geometries(result)
    
    if not geometries:
        return
    
    # Use the VisualizerWithEditing for more control
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name, width=1400, height=1000)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Render options
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1.0, 1.0, 1.0])  # White background (matching 2D plot)
    render_opt.light_on = True
    render_opt.point_size = 8.0
    render_opt.line_width = 4.0
    render_opt.mesh_show_back_face = True
    # Smooth shading is achieved through computed vertex normals
    
    # Run
    vis.run()
    vis.destroy_window()