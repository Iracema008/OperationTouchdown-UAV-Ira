''' Visualize lawnmower waypoints for grid and SA planners.
Plots the path on a 2D field map so you can confirm the order visually.

Usage:
    python tools/visualize_path.py --planner grid
    python tools/visualize_path.py --planner sa
'''

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIELD_CONFIG = {
    "north_min_m": 0.0,
    "north_max_m": 9.2,
    "east_min_m":  0.0,
    "east_max_m":  9.2,
}

LAWNMOWER_CONFIG = {
    "col_spacing_m": 2.8,
    "row_spacing_m": 2.5,
}


def build_grid_waypoints() -> list:
    waypoints   = []
    col_spacing = LAWNMOWER_CONFIG["col_spacing_m"]
    row_spacing = LAWNMOWER_CONFIG["row_spacing_m"]
    north_min   = FIELD_CONFIG["north_min_m"]
    north_max   = FIELD_CONFIG["north_max_m"]
    east_min    = FIELD_CONFIG["east_min_m"]
    east_max    = FIELD_CONFIG["east_max_m"]

    east    = east_min
    col_idx = 0
    while east <= east_max + 1e-6:
        if col_idx % 2 == 0:
            north = north_min
            while north <= north_max + 1e-6:
                waypoints.append((north, east))
                north += row_spacing
        else:
            north = north_max
            while north >= north_min - 1e-6:
                waypoints.append((north, east))
                north -= row_spacing
        east += col_spacing
        col_idx += 1
    return waypoints


def plot_path(waypoints: list, title: str):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(-0.5, FIELD_CONFIG["east_max_m"] + 0.5)
    ax.set_ylim(-0.5, FIELD_CONFIG["north_max_m"] + 0.5)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Field boundary outline
    rect = plt.Rectangle(
        (FIELD_CONFIG["east_min_m"], FIELD_CONFIG["north_min_m"]),
        FIELD_CONFIG["east_max_m"] - FIELD_CONFIG["east_min_m"],
        FIELD_CONFIG["north_max_m"] - FIELD_CONFIG["north_min_m"],
        fill=False, edgecolor="gray", linewidth=2, linestyle="--"
    )
    ax.add_patch(rect)

    # Path lines
    for i in range(len(waypoints) - 1):
        n0, e0 = waypoints[i]
        n1, e1 = waypoints[i + 1]
        ax.plot([e0, e1], [n0, n1], "b-", linewidth=1.5, alpha=0.6)

    # Waypoint markers with numbers
    for i, (north, east) in enumerate(waypoints):
        color = "green" if i == 0 else "red" if i == len(waypoints) - 1 else "dodgerblue"
        size  = 120 if i in (0, len(waypoints) - 1) else 60
        ax.scatter(east, north, c=color, s=size, zorder=5)
        ax.annotate(
            str(i + 1),
            (east, north),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="black"
        )

    start_patch = mpatches.Patch(color="green",      label="Start (WP 1)")
    end_patch   = mpatches.Patch(color="red",        label=f"End (WP {len(waypoints)})")
    wp_patch    = mpatches.Patch(color="dodgerblue", label="Waypoints")
    ax.legend(handles=[start_patch, end_patch, wp_patch], loc="upper right")

    total_dist = sum(
        ((waypoints[i][0] - waypoints[i+1][0])**2 +
         (waypoints[i][1] - waypoints[i+1][1])**2) ** 0.5
        for i in range(len(waypoints) - 1)
    )
    ax.text(
        0.02, 0.02,
        f"Waypoints: {len(waypoints)}\nTotal distance: {total_dist:.1f}m",
        transform=ax.transAxes,
        fontsize=9, verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize UAV search path")
    parser.add_argument(
        "--planner", choices=["grid", "sa", "sac"], default="grid",
        help="grid = boustrophedon | sa = simulated annealing | sac = center-biased simulated annealing"
    )
    args = parser.parse_args()

    grid = build_grid_waypoints()

    if args.planner == "sa":
        try:
            from path_planning.simulated_annealing import build_sa_waypoints
            waypoints = build_sa_waypoints(grid, start_pos=(0.0, 0.0))
            title = f"SA Optimized Path — {len(waypoints)} waypoints"
        except ImportError:
            print("Could not import simulated_annealing — showing grid instead")
            waypoints = grid
            title = f"Grid Path (SA import failed) — {len(waypoints)} waypoints"
    elif args.planner == "sac":
        try:
            from path_planning.simulated_annealing_center import build_sa_center_waypoints
            waypoints = build_sa_center_waypoints(grid, start_pos=(0.0, 0.0))
            title = f"SAC Optimized Path — {len(waypoints)} waypoints"
        except ImportError:
            print("Could not import simulated_annealing_center — showing grid instead")
            waypoints = grid
            title = f"Grid Path (SAC import failed) — {len(waypoints)} waypoints"
    else:
        waypoints = grid
        title = f"Grid Path — {len(waypoints)} waypoints"

    print(f"\n{'='*40}")
    print(f"Planner: {args.planner.upper()}")
    print(f"{'='*40}")
    for i, (n, e) in enumerate(waypoints):
        print(f"  WP {i+1:>2}/{len(waypoints)} → N={n:.1f} E={e:.1f}")

    total_dist = sum(
        ((waypoints[i][0] - waypoints[i+1][0])**2 +
         (waypoints[i][1] - waypoints[i+1][1])**2) ** 0.5
        for i in range(len(waypoints) - 1)
    )
    print(f"\nTotal path distance: {total_dist:.2f}m")
    print(f"Waypoints: {len(waypoints)}")
    print(f"Est. time at 3s dwell: {len(waypoints) * 3}s ({len(waypoints) * 3 / 60:.1f} min)")
    print(f"{'='*40}\n")

    plot_path(waypoints, title)


if __name__ == "__main__":
    main()