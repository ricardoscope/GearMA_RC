"""
Ghost circle detection for crown gears.

This script implements a reproducible pipeline which
 1. loads and orients an STL gear model,
 2. finds the tooth-tip height,
 3. extracts a planar slice,
 4. isolates tooth flanks,
 5. fits centre lines for each tooth,
 6. estimates the "ghost circle" caused by X-axis cutter misalignment,
 7. reports results and produces diagnostic plots.

Usage (defaults make it runnable out of the box once dependencies are installed):
    python test.py --stl path/to/gear.stl --outdir ./analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import trimesh

# -------------------- Configuration defaults --------------------

DEFAULT_STL = Path(
    r"C:\Users\alibi\Documents\Gears Examples\2025_01_22_Kronenrad CT Messung_Ausgerichtet nach neuem Vorgehen.stl"
)
DEFAULT_OUTPUT = Path("./analysis_output")
MIN_POINTS_PER_TOOTH = 30


# -------------------- Data containers --------------------


@dataclass
class ZTipEstimate:
    z_tip: float
    method: str
    confidence: float


@dataclass
class LineModel:
    point: np.ndarray
    direction: np.ndarray


@dataclass
class GhostCircleResult:
    stl: str
    z_tip: float
    r_inner: float
    r_outer: float
    circle_center: Sequence[float] | None
    circle_radius: float | None
    offset_vector: Sequence[float] | None
    offset_magnitude: float | None
    n_intersections: int
    n_bisectors_used: int


# -------------------- Utilities --------------------


def _ensure_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _least_squares_intersection(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
    A = np.column_stack((d1, -d2))
    if np.linalg.matrix_rank(A) < 2:
        return None
    t = np.linalg.lstsq(A, p2 - p1, rcond=None)[0]
    return p1 + t[0] * d1


def _taubin_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimal Taubin circle fit (no weights)."""
    x = points[:, 0]
    y = points[:, 1]
    x_m = x.mean()
    y_m = y.mean()
    u = x - x_m
    v = y - y_m
    Suu = np.dot(u, u)
    Suv = np.dot(u, v)
    Svv = np.dot(v, v)
    Suuu = np.dot(u, u * u)
    Svvv = np.dot(v, v * v)
    Suvv = np.dot(u, v * v)
    Svuu = np.dot(v, u * u)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu])
    uc, vc = np.linalg.solve(A, b)
    center = np.array([x_m + uc, y_m + vc])
    radius = np.sqrt(uc * uc + vc * vc + (Suu + Svv) / len(points))
    return center, float(radius)


# -------------------- Mesh loading --------------------


def load_mesh(path: Path, units_scale: float = 1.0, axis: str = "z") -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(path))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded mesh is not a triangular mesh.")
    if units_scale != 1.0:
        mesh.apply_scale(units_scale)
    if axis.lower() == "x":
        mesh.apply_transform(
            np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
        )
    elif axis.lower() == "y":
        mesh.apply_transform(
            np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
        )
    return mesh


# -------------------- Z_tip detection --------------------


def _max_radius_of_slice(mesh: trimesh.Trimesh, z: float) -> float:
    sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is None or sec.vertices.shape[0] == 0:
        return np.nan
    xy = np.asarray(sec.vertices, dtype=float)[:, :2]
    r = np.linalg.norm(xy, axis=1)
    return float(np.nanmax(r)) if len(r) else np.nan


def detect_z_tip(mesh: trimesh.Trimesh, explicit_z: Optional[float] = None) -> ZTipEstimate:
    if explicit_z is not None:
        return ZTipEstimate(float(explicit_z), "explicit", 1.0)
    z_vals = mesh.vertices[:, 2]
    z_lo, z_hi = np.quantile(z_vals, [0.90, 0.995])
    Z = np.linspace(z_lo, z_hi, 45)
    Rmax = np.array([_max_radius_of_slice(mesh, float(z)) for z in Z])
    m = np.nanmax(Rmax)
    if not np.isfinite(m):
        fallback = float(np.quantile(z_vals, 0.98))
        return ZTipEstimate(fallback, "quantile98", 0.3)
    within = np.where(Rmax >= 0.99 * m)[0]
    idx = int(within[-1]) if len(within) else int(np.nanargmax(Rmax))
    conf = float(min(1.0, len(within) / max(1, len(Z)) * 4.0 + 0.2))
    return ZTipEstimate(float(Z[idx]), "radius-plateau", conf)


# -------------------- Slice extraction --------------------


def slice_mesh(mesh: trimesh.Trimesh, z: float) -> list[np.ndarray]:
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None or section.vertices.shape[0] == 0:
        return []
    xy = np.asarray(section.vertices, dtype=float)[:, :2]
    polylines: list[np.ndarray] = []
    if hasattr(section, "entities") and len(section.entities) > 0:
        for ent in section.entities:
            if hasattr(ent, "points"):
                pts = xy[np.asarray(ent.points, dtype=int)]
                if pts.shape[0] >= 2:
                    polylines.append(pts)
    if not polylines:
        polylines.append(xy)
    return polylines


# -------------------- Tooth segmentation and flank fitting --------------------


def estimate_radial_bounds(points_xy: np.ndarray, r_inner: float | None, r_outer: float | None) -> tuple[float, float]:
    if r_inner is not None and r_outer is not None:
        return float(r_inner), float(r_outer)
    if len(points_xy) == 0:
        return 0.0, 1.0
    r = np.linalg.norm(points_xy, axis=1)
    outer = float(np.quantile(r, 0.97)) if r_outer is None else float(r_outer)
    inner = float(np.quantile(r, 0.70)) if r_inner is None else float(r_inner)
    if inner >= outer:
        inner = 0.9 * outer
    return inner, outer


def partition_points_into_teeth(points_xy: np.ndarray, n_teeth: Optional[int] = None) -> list[np.ndarray]:
    if len(points_xy) == 0:
        return []
    angles = np.arctan2(points_xy[:, 1], points_xy[:, 0])
    order = np.argsort(angles)
    angles_sorted = angles[order]
    pts_sorted = points_xy[order]
    diffs = np.diff(np.unwrap(angles_sorted))
    gap_threshold = 1.8 * np.median(np.abs(diffs)) if len(diffs) else 0
    split_indices = list(np.where(diffs > gap_threshold)[0] + 1)
    clusters = np.split(pts_sorted, split_indices)
    if n_teeth and len(clusters) != n_teeth:
        logging.warning("Detected %d teeth clusters; expected %d", len(clusters), n_teeth)
    clusters = [cluster for cluster in clusters if len(cluster) >= MIN_POINTS_PER_TOOTH]
    return clusters


def fit_line(points: np.ndarray, iters: int = 200, thresh: float = 0.05) -> LineModel:
    if len(points) < 2:
        raise ValueError("Not enough points to fit a line")
    rng = np.random.default_rng(0)
    best_mask = None
    best_score = -1
    best_direction = None
    best_point = None
    for _ in range(iters):
        i, j = rng.choice(len(points), size=2, replace=False)
        p1, p2 = points[i], points[j]
        d = p2 - p1
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        d /= n
        v = points - p1
        dist = np.abs(v[:, 0] * (-d[1]) + v[:, 1] * d[0])
        mask = dist < thresh
        score = int(mask.sum())
        if score > best_score:
            best_score = score
            best_mask = mask
            best_direction = d.copy()
            best_point = p1
    if best_mask is None or best_score < 2:
        C = points.mean(axis=0)
        U, _, Vt = np.linalg.svd(points - C, full_matrices=False)
        direction = _ensure_unit(Vt[0])
        return LineModel(C, direction)
    inliers = points[best_mask]
    C = inliers.mean(axis=0)
    _, _, Vt = np.linalg.svd(inliers - C, full_matrices=False)
    direction = _ensure_unit(Vt[0])
    return LineModel(C, direction)


def flank_lines_from_tooth(points: np.ndarray) -> Optional[tuple[LineModel, LineModel]]:
    if len(points) < MIN_POINTS_PER_TOOTH:
        return None
    center = points.mean(axis=0)
    radial = _ensure_unit(center)
    tangential = np.array([-radial[1], radial[0]])
    proj = (points - center) @ tangential
    median = np.median(proj)
    left = points[proj <= median]
    right = points[proj > median]
    if len(left) < 5 or len(right) < 5:
        return None
    radius_est = np.linalg.norm(center)
    thresh = 0.05 * radius_est if radius_est > 0 else 0.05
    left_model = fit_line(left, thresh=thresh)
    right_model = fit_line(right, thresh=thresh)
    if np.linalg.norm(left_model.direction) == 0 or np.linalg.norm(right_model.direction) == 0:
        return None
    return left_model, right_model


# -------------------- Pipeline --------------------


def analyse_ghost_circle(
    stl_path: Path,
    outdir: Path,
    z_tip: Optional[float] = None,
    r_inner: Optional[float] = None,
    r_outer: Optional[float] = None,
    axis: str = "z",
    units_scale: float = 1.0,
    expected_teeth: Optional[int] = None,
) -> tuple[GhostCircleResult, np.ndarray, list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], list[tuple[LineModel, LineModel]]]:
    outdir.mkdir(parents=True, exist_ok=True)
    mesh = load_mesh(stl_path, units_scale=units_scale, axis=axis)
    z_est = detect_z_tip(mesh, z_tip)
    polylines = slice_mesh(mesh, z_est.z_tip)
    slice_points = np.vstack(polylines) if polylines else np.zeros((0, 2))
    r_in, r_out = estimate_radial_bounds(slice_points, r_inner, r_outer)
    mask = (np.linalg.norm(slice_points, axis=1) >= r_in) & (np.linalg.norm(slice_points, axis=1) <= r_out)
    filtered_points = slice_points[mask]
    tooth_clusters = partition_points_into_teeth(filtered_points, expected_teeth)
    bisectors: list[tuple[np.ndarray, np.ndarray]] = []
    all_flanks: list[tuple[np.ndarray, LineModel, LineModel]] = []  # Store cluster and flank lines
    for cluster in tooth_clusters:
        flanks = flank_lines_from_tooth(cluster)
        if not flanks:
            continue
        left_model, right_model = flanks
        all_flanks.append((cluster, left_model, right_model))  # Store for visualization
        bdir = _ensure_unit(left_model.direction + right_model.direction)
        if np.linalg.norm(bdir) == 0:
            continue
        origin = 0.5 * (left_model.point + right_model.point)
        bisectors.append((origin, bdir))
    intersections: list[np.ndarray] = []
    for i in range(len(bisectors)):
        p1, d1 = bisectors[i]
        for j in range(i + 1, len(bisectors)):
            p2, d2 = bisectors[j]
            q = _least_squares_intersection(p1, d1, p2, d2)
            if q is not None and np.isfinite(q).all():
                intersections.append(q)
    circle_center = circle_radius = offset_vec = offset_mag = None
    if len(intersections) >= 6:
        points = np.vstack(intersections)
        center, radius = _taubin_circle(points)
        circle_center = center.tolist()
        circle_radius = radius
        offset_vec = center.tolist()
        offset_mag = float(np.linalg.norm(center))
    result = GhostCircleResult(
        stl=str(stl_path),
        z_tip=z_est.z_tip,
        r_inner=r_in,
        r_outer=r_out,
        circle_center=circle_center,
        circle_radius=circle_radius,
        offset_vector=offset_vec,
        offset_magnitude=offset_mag,
        n_intersections=len(intersections),
        n_bisectors_used=len(bisectors),
    )
    
    # Visualize with flanks
    return result, slice_points, tooth_clusters, bisectors, all_flanks


# -------------------- Visualisation --------------------


def plot_results(
    outdir: Path,
    slice_points: np.ndarray,
    tooth_clusters: list[np.ndarray],
    bisectors: list[tuple[np.ndarray, np.ndarray]],
    ghost_center: Optional[Sequence[float]],
    ghost_radius: Optional[float],
    r_inner: float,
    r_outer: float,
    flank_lines: Optional[list[tuple[np.ndarray, LineModel, LineModel]]] = None,
) -> None:
    if slice_points.size == 0:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(slice_points[:, 0], slice_points[:, 1], s=3, c="lightgrey", label="Slice points", alpha=0.3)
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(tooth_clusters))))
    for i, cluster in enumerate(tooth_clusters):
        ax.scatter(cluster[:, 0], cluster[:, 1], s=6, color=colors[i % len(colors)], label=f"Tooth {i+1}" if i < 10 else None)
    
    # Draw all tooth flanks
    if flank_lines:
        flank_drawn = False
        for _, left_model, right_model in flank_lines:
            # Draw left flank line
            left_seg_length = 30.0
            left_seg = np.vstack((
                left_model.point - left_seg_length * left_model.direction,
                left_model.point + left_seg_length * left_model.direction
            ))
            ax.plot(left_seg[:, 0], left_seg[:, 1], color="blue", linewidth=1.2, alpha=0.7, 
                   label="Left flanks" if not flank_drawn else "")
            
            # Draw right flank line
            right_seg_length = 30.0
            right_seg = np.vstack((
                right_model.point - right_seg_length * right_model.direction,
                right_model.point + right_seg_length * right_model.direction
            ))
            ax.plot(right_seg[:, 0], right_seg[:, 1], color="green", linewidth=1.2, alpha=0.7,
                   label="Right flanks" if not flank_drawn else "")
            flank_drawn = True
        # Save individual flank visualisations
        flank_dir = outdir / "tooth_flanks"
        flank_dir.mkdir(parents=True, exist_ok=True)
        for idx, (cluster_pts, left_model, right_model) in enumerate(flank_lines, start=1):
            fig_tooth, ax_tooth = plt.subplots(figsize=(5, 5))
            ax_tooth.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=8, color="tab:blue", label="Tooth flank points")
            for model, color, label in (
                (left_model, "blue", "Left flank"),
                (right_model, "green", "Right flank"),
            ):
                seg = np.vstack((
                    model.point - 30.0 * model.direction,
                    model.point + 30.0 * model.direction
                ))
                ax_tooth.plot(seg[:, 0], seg[:, 1], color=color, linewidth=1.5, label=label)
            ax_tooth.set_aspect("equal", "box")
            ax_tooth.set_title(f"Tooth {idx}")
            ax_tooth.legend(loc="best")
            fig_tooth.tight_layout()
            fig_tooth.savefig(flank_dir / f"tooth_{idx:03d}.png", dpi=160)
            plt.close(fig_tooth)
            # Save flank points to CSV
            np.savetxt(flank_dir / f"tooth_{idx:03d}_points.csv", cluster_pts, delimiter=",", header="x,y", comments="")
    
    # Draw bisectors
    for idx, (origin, direction) in enumerate(bisectors):
        seg = np.vstack((origin - 50 * direction, origin + 50 * direction))
        ax.plot(seg[:, 0], seg[:, 1], color="black", linewidth=0.6, alpha=0.6, linestyle="--", 
               label="Bisectors" if idx == 0 else "")
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(r_inner * np.cos(theta), r_inner * np.sin(theta), linestyle="--", color="grey", linewidth=1.0, label="r_inner")
    ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), linestyle="--", color="grey", linewidth=1.0, label="r_outer")
    if ghost_center is not None and ghost_radius is not None:
        ax.plot(
            ghost_center[0] + ghost_radius * np.cos(theta),
            ghost_center[1] + ghost_radius * np.sin(theta),
            color="red",
            linewidth=2,
            label="Ghost circle",
        )
        ax.scatter([ghost_center[0]], [ghost_center[1]], color="red", s=40, label="Ghost centre")
    ax.set_aspect("equal", "box")
    ax.set_title("Crown gear ghost circle analysis")
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.set_xlabel("X [model units]")
    ax.set_ylabel("Y [model units]")
    fig.tight_layout()
    fig.savefig(outdir / "ghost_circle_overview.png", dpi=220)
    plt.close(fig)


# -------------------- CLI --------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect X-axis cutter setup error in a crown gear via ghost circle analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stl", type=Path, default=DEFAULT_STL, help="Input STL file.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT, help="Output directory for results.")
    parser.add_argument("--z-tip", type=float, default=None, help="Override automatic tooth-tip Z height.")
    parser.add_argument("--r-inner", type=float, default=None, help="Inner radius bound for flank isolation.")
    parser.add_argument("--r-outer", type=float, default=None, help="Outer radius bound for flank isolation.")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z", help="Mesh axis aligned with gear axis.")
    parser.add_argument("--units-scale", type=float, default=1.0, help="Scale factor to apply to the mesh.")
    parser.add_argument("--expected-teeth", type=int, default=None, help="Expected number of teeth (optional).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    if not args.stl.exists():
        logging.error("STL file not found: %s", args.stl)
        return 2
    result, slice_points, tooth_clusters, bisectors, flank_details = analyse_ghost_circle(
        args.stl,
        args.outdir,
        z_tip=args.z_tip,
        r_inner=args.r_inner,
        r_outer=args.r_outer,
        axis=args.axis,
        units_scale=args.units_scale,
        expected_teeth=args.expected_teeth,
    )
    filtered_points = slice_points[
        (np.linalg.norm(slice_points, axis=1) >= result.r_inner)
        & (np.linalg.norm(slice_points, axis=1) <= result.r_outer)
    ]
    plot_results(
        args.outdir,
        filtered_points,
        tooth_clusters,
        bisectors,
        result.circle_center,
        result.circle_radius,
        result.r_inner,
        result.r_outer,
        flank_details,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    report_path = args.outdir / "ghost_circle_report.json"
    report_path.write_text(json.dumps(asdict(result), indent=2))
    logging.info("Analysis complete. Report saved to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
