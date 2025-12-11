"""
Diagnostic tool for gear analysis pipeline.

Detects:
1. Tooth clustering issues (bin alignment)
2. Flank misclassification (orthogonal flanks = wrong)
3. Bisector issues

Valid tooth shapes: parallel, V-shaped, square
Invalid: orthogonal flanks (~90°), same flank twice
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ToothDiagnostics:
    """Diagnostic information for a single tooth."""
    tooth_number: int
    n_points: int
    angular_center: float
    angular_span: float
    is_valid: bool = True
    issue: str = ""


def diagnose_clustering(
    filtered_points: np.ndarray,
    n_teeth: int,
    output_path: Path
) -> list[ToothDiagnostics]:
    """Diagnose tooth clustering issues."""
    print(f"  Generating clustering diagnostic...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1 = axes[0]
    
    angles = np.degrees(np.arctan2(filtered_points[:, 1], filtered_points[:, 0]))
    angles = (angles + 360) % 360
    
    bin_edges_deg = np.linspace(0, 360, n_teeth + 1)
    ax1.hist(angles, bins=72, alpha=0.7, color='blue', label='Point distribution')
    
    for edge in bin_edges_deg:
        ax1.axvline(edge, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    bin_centers = (bin_edges_deg[:-1] + bin_edges_deg[1:]) / 2
    ylim = ax1.get_ylim()
    for i, center in enumerate(bin_centers):
        ax1.text(center, ylim[1] * 0.95 if ylim[1] > 0 else 1, 
                str(i+1), ha='center', fontsize=7, color='green', fontweight='bold')
    
    ax1.set_xlabel('Angle (degrees)', fontsize=10)
    ax1.set_ylabel('Point count', fontsize=10)
    ax1.set_title('Angular Distribution vs Bin Boundaries\n(Red lines should fall in valleys/gaps)', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    
    angles_rad = np.arctan2(filtered_points[:, 1], filtered_points[:, 0])
    angles_rad = (angles_rad + 2 * np.pi) % (2 * np.pi)
    edges_rad = np.linspace(0.0, 2 * np.pi, n_teeth + 1)
    bin_idx = np.digitize(angles_rad, edges_rad, right=False) - 1
    bin_idx[bin_idx == n_teeth] = 0
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_teeth))
    diagnostics = []
    
    for i in range(n_teeth):
        mask = bin_idx == i
        cluster = filtered_points[mask]
        tooth_num = i + 1
        
        if len(cluster) == 0:
            diagnostics.append(ToothDiagnostics(
                tooth_number=tooth_num, n_points=0,
                angular_center=bin_centers[i], angular_span=0,
                is_valid=False, issue="Empty cluster"
            ))
            continue
        
        ax2.scatter(cluster[:, 0], cluster[:, 1], s=8, c=[colors[i]], alpha=0.7)
        
        cluster_angles = np.degrees(np.arctan2(cluster[:, 1], cluster[:, 0]))
        cluster_angles = (cluster_angles + 360) % 360
        
        angle_min, angle_max = np.min(cluster_angles), np.max(cluster_angles)
        if angle_max - angle_min > 180:
            cluster_angles[cluster_angles < 180] += 360
            angle_min, angle_max = np.min(cluster_angles), np.max(cluster_angles)
        
        angular_center = np.mean(cluster_angles) % 360
        angular_span = angle_max - angle_min
        
        expected_span = 360 / n_teeth
        issue = ""
        is_valid = True
        
        if angular_span > expected_span * 1.8:
            issue = f"Too wide ({angular_span:.1f}°)"
            is_valid = False
        elif len(cluster) < 10:
            issue = f"Few points ({len(cluster)})"
            is_valid = False
        
        center = cluster.mean(axis=0)
        ax2.annotate(str(tooth_num), center, fontsize=8, ha='center', va='center',
                    color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
        
        diagnostics.append(ToothDiagnostics(
            tooth_number=tooth_num, n_points=len(cluster),
            angular_center=angular_center, angular_span=angular_span,
            is_valid=is_valid, issue=issue
        ))
    
    ax2.set_aspect('equal')
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_title('Tooth Clusters (each color = one bin)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_path}")
    return diagnostics


def diagnose_flanks(
    tooth_flanks_list: list,
    filtered_points: np.ndarray,
    n_teeth: int,
    output_path: Path
) -> list[tuple[int, str]]:
    """Diagnose flank fitting issues.
    
    Checks for:
    - Orthogonal flanks (~90°) = INVALID (misclassification)
    - Parallel or V-shaped flanks = VALID
    """
    print(f"  Generating flank diagnostic...")
    
    if not tooth_flanks_list:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No tooth flanks data available", 
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title("Flank Diagnostic - No Data")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {output_path} (no data)")
        return []
    
    n_flanks = len(tooth_flanks_list)
    n_cols = min(6, n_flanks)
    n_rows = max(1, (n_flanks + n_cols - 1) // n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()
    
    angles_rad = np.arctan2(filtered_points[:, 1], filtered_points[:, 0])
    angles_rad = (angles_rad + 2 * np.pi) % (2 * np.pi)
    edges_rad = np.linspace(0.0, 2 * np.pi, n_teeth + 1)
    bin_idx = np.digitize(angles_rad, edges_rad, right=False) - 1
    bin_idx[bin_idx == n_teeth] = 0
    
    issues = []
    
    for idx, tf in enumerate(tooth_flanks_list):
        ax = axes[idx]
        tooth_num = tf.tooth
        
        cluster = filtered_points[bin_idx == (tooth_num - 1)]
        
        if len(cluster) > 0:
            ax.scatter(cluster[:, 0], cluster[:, 1], s=3, c='lightgray', alpha=0.6)
        
        if len(cluster) > 0:
            cluster_radius = np.mean(np.linalg.norm(cluster, axis=1))
            line_len = cluster_radius * 0.15
        else:
            line_len = 0.2
        
        # Plot left flank (green)
        left_start = tf.left_point - line_len * tf.left_direction
        left_end = tf.left_point + line_len * tf.left_direction
        ax.plot([left_start[0], left_end[0]], [left_start[1], left_end[1]], 
               'g-', linewidth=2.5, label='Left')
        ax.scatter([tf.left_point[0]], [tf.left_point[1]], c='green', s=40, zorder=5)
        
        # Plot right flank (blue)
        right_start = tf.right_point - line_len * tf.right_direction
        right_end = tf.right_point + line_len * tf.right_direction
        ax.plot([right_start[0], right_end[0]], [right_start[1], right_end[1]], 
               'b-', linewidth=2.5, label='Right')
        ax.scatter([tf.right_point[0]], [tf.right_point[1]], c='blue', s=40, zorder=5)
        
        # Check for issues
        issue = None
        title_color = 'black'
        
        # Use is_valid from ToothFlanks if available
        if hasattr(tf, 'is_valid') and not tf.is_valid:
            issue = tf.issue if hasattr(tf, 'issue') else "Invalid"
            title_color = 'red'
        else:
            # Manual check: orthogonal flanks are BAD
            dir_dot = abs(np.dot(tf.left_direction, tf.right_direction))
            angle_between = np.degrees(np.arccos(np.clip(dir_dot, 0, 1)))
            
            if dir_dot < 0.25:  # ~75-90° = orthogonal = BAD
                issue = f"Orthogonal! ({angle_between:.0f}°)"
                title_color = 'red'
            
            # Check if flanks are too close (same flank twice)
            flank_distance = np.linalg.norm(tf.left_point - tf.right_point)
            if flank_distance < 0.01:
                issue = "Same flank twice"
                title_color = 'red'
        
        if issue:
            issues.append((tooth_num, issue))
        
        ax.set_aspect('equal')
        
        # Show angle between flanks
        dir_dot = abs(np.dot(tf.left_direction, tf.right_direction))
        angle_between = np.degrees(np.arccos(np.clip(dir_dot, 0, 1)))
        
        title = f"Tooth {tooth_num} ({angle_between:.0f}°)"
        if issue:
            title += f"\n⚠ {issue}"
        ax.set_title(title, fontsize=9, color=title_color, 
                    fontweight='bold' if issue else 'normal')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    for idx in range(len(tooth_flanks_list), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Flank Diagnostic: Angle shown is between L/R flanks\n"
                "Parallel (0°) or V-shape (<75°) = OK | Orthogonal (>75°) = BAD", 
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_path}")
    return issues


def diagnose_bisectors(
    tooth_bisectors: list,
    r_inner: float,
    r_outer: float,
    output_path: Path
) -> list[tuple[int, str]]:
    """Diagnose bisector issues."""
    print(f"  Generating bisector diagnostic...")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(r_inner * np.cos(theta), r_inner * np.sin(theta), 
           'b--', alpha=0.3, linewidth=1, label='Inner radius')
    ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), 
           'b--', alpha=0.3, linewidth=1, label='Outer radius')
    
    if not tooth_bisectors:
        ax.text(0, 0, "No bisector data available\n\nCheck if tooth_bisectors\nis being computed", 
               ha='center', va='center', fontsize=14)
        ax.set_xlim(-r_outer * 1.2, r_outer * 1.2)
        ax.set_ylim(-r_outer * 1.2, r_outer * 1.2)
        ax.set_title("Bisector Diagnostic - No Data", fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {output_path} (no data)")
        return []
    
    issues = []
    arrow_length = r_inner * 0.4
    
    good_count = 0
    warn_count = 0
    bad_count = 0
    
    for bisector in tooth_bisectors:
        origin = bisector.origin
        direction = bisector.direction
        
        # Check if direction points inward (toward center)
        origin_radius = np.linalg.norm(origin)
        if origin_radius > 1e-10:
            radial_dir = origin / origin_radius
        else:
            radial_dir = np.array([1, 0])
        
        # Negative dot = pointing inward (good), Positive = pointing outward (bad)
        radial_component = np.dot(direction, radial_dir)
        
        if radial_component > 0.3:  # Pointing outward
            color = 'red'
            issues.append((bisector.tooth, "Points outward"))
            bad_count += 1
        elif radial_component > -0.3:  # Too tangential
            color = 'orange'
            issues.append((bisector.tooth, "Too tangential"))
            warn_count += 1
        else:  # Good - pointing inward
            color = 'green'
            good_count += 1
        
        # Draw arrow
        ax.arrow(origin[0], origin[1], 
                direction[0] * arrow_length * 0.8, 
                direction[1] * arrow_length * 0.8,
                head_width=arrow_length * 0.12, 
                head_length=arrow_length * 0.08, 
                fc=color, ec=color, alpha=0.8, linewidth=1.5)
        
        # Label
        label_pos = origin - direction * arrow_length * 0.2
        ax.annotate(str(bisector.tooth), label_pos,
                   fontsize=8, ha='center', va='center', 
                   fontweight='bold', alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7))
    
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_title(f'Bisector Diagnostic\n'
                f'Green={good_count} (inward) | Orange={warn_count} (tangential) | Red={bad_count} (outward)\n'
                f'All arrows should point INWARD toward center', 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_path}")
    return issues


def run_full_diagnostics(result, config, output_dir: Path):
    """Run all diagnostics on analysis result."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 60)
    
    all_issues = []
    
    # 1. Clustering diagnostic
    print("\n[1/3] Clustering diagnostic...")
    clustering_path = output_dir / "diagnostic_1_clustering.png"
    cluster_diagnostics = diagnose_clustering(
        result.filtered_points,
        config.n_teeth,
        clustering_path
    )
    
    cluster_issues = [d for d in cluster_diagnostics if not d.is_valid]
    if cluster_issues:
        print(f"      ⚠️  {len(cluster_issues)} cluster issues found")
        for d in cluster_issues[:5]:
            all_issues.append(f"Cluster {d.tooth_number}: {d.issue}")
    
    # 2. Flank diagnostic
    print("\n[2/3] Flank diagnostic...")
    flank_path = output_dir / "diagnostic_2_flanks.png"
    
    tooth_flanks = result.tooth_flanks if hasattr(result, 'tooth_flanks') else []
    flank_issues = diagnose_flanks(
        tooth_flanks,
        result.filtered_points,
        config.n_teeth,
        flank_path
    )
    
    if flank_issues:
        print(f"      ⚠️  {len(flank_issues)} flank issues found")
        for tooth_num, issue in flank_issues[:5]:
            all_issues.append(f"Tooth {tooth_num} flank: {issue}")
    
    # 3. Bisector diagnostic
    print("\n[3/3] Bisector diagnostic...")
    bisector_path = output_dir / "diagnostic_3_bisectors.png"
    
    tooth_bisectors = result.tooth_bisectors if hasattr(result, 'tooth_bisectors') else []
    
    # Debug info
    print(f"      tooth_bisectors count: {len(tooth_bisectors)}")
    
    bisector_issues = diagnose_bisectors(
        tooth_bisectors,
        config.r_inner,
        config.r_outer,
        bisector_path
    )
    
    if bisector_issues:
        print(f"      ⚠️  {len(bisector_issues)} bisector issues found")
        for tooth_num, issue in bisector_issues[:5]:
            all_issues.append(f"Tooth {tooth_num} bisector: {issue}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"\nFiles saved to: {output_dir}")
    print(f"  • diagnostic_1_clustering.png")
    print(f"  • diagnostic_2_flanks.png")
    print(f"  • diagnostic_3_bisectors.png")
    
    if all_issues:
        print(f"\n⚠️  Total issues found: {len(all_issues)}")
        for issue in all_issues[:10]:
            print(f"   • {issue}")
        if len(all_issues) > 10:
            print(f"   ... and {len(all_issues) - 10} more")
    else:
        print("\n✅ No issues detected!")
    
    print("=" * 60)