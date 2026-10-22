"""
scan3d.py - turn an ultrasonic scan log into a 3D point array and figures.

Usage:
    python scan3d.py log.txt                 # last complete scan in the file
    python scan3d.py log.txt --scan 2        # the 2nd scan in the file (1 = first)
    python scan3d.py log.txt --list          # list the scans found in the file
    python scan3d.py log.txt --R 6.5 --H 12  # override the geometry from the log

The log can be pasted straight from the Serial Monitor: comment lines (#),
repeated menus and interrupted scans are handled automatically.

Outputs (next to the log file, named <log>_scanN_*):
    _pontos.csv / _pontos.npy   point array (columns below)
    _pontos.ply                 point cloud for MeshLab
    _figura.png                 3D view, top view and side profile
    _figura.html                interactive 3D view (only if plotly is installed)

Point array columns:
    x, y, z (cm, object frame)   plate_angle (deg)   inclination (deg)
    distance (cm, corrected)     repeats (readings merged into the point)
"""

import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =================== SETTINGS (change here) ===================
# Geometry. None = read R and H from the "# Geometria" line of the log.
R_SENSOR = None        # cm: plate axis -> servo axis, horizontal
H_SENSOR = None        # cm: plate top -> servo axis, vertical

# Where the transducers sit relative to the servo axis (measure with a ruler).
PIVOT_FRONT = 0.0      # cm along the beam (towards the object = positive)
PIVOT_UP = 0.0         # cm perpendicular to the beam (up when level = positive)

# Distance corrections. Leave DIST_OFFSET = 0 if it is already applied in the Arduino.
DIST_OFFSET = 0.0      # cm, real - measured
DIST_SCALE = 1.0       # real = DIST_SCALE * measured + DIST_OFFSET
TR_SEPARATION = 2.6    # cm between speaker and microphone (triangle correction)

# Plate centre offset, if the plate is off-centre on the motor (cm).
PLATE_CX = 0.0
PLATE_CY = 0.0

# Filtering
Z_MIN = 1.0            # cm: points lower than this are treated as plate echoes
MAX_DIST = None        # cm: ignore readings further than this (None = no limit)

# Merge consecutive tilts at the same plate angle that return the same distance
MERGE = True
MERGE_TOL = 0.35       # cm
# ===============================================================


def parse_log(path):
    """Return a list of scans: dict(rows, complete, R, H, start_line)."""
    scans = []
    R = H = None
    current = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            m = re.search(r"Geometria:\s*R=([\d.]+)\s*cm,\s*H=([\d.]+)", line)
            if m:
                R, H = float(m.group(1)), float(m.group(2))
            if line.startswith("# posicao") or line.startswith("# posição"):
                if current and current["rows"]:
                    scans.append(current)
                current = dict(rows=[], complete=False, R=R, H=H, start_line=n)
                continue
            if current is None:
                continue
            if line.startswith("# Fim"):
                current["complete"] = True
                scans.append(current)
                current = None
                continue
            parts = line.split(",")
            if len(parts) == 4:
                try:
                    current["rows"].append([float(p) for p in parts])
                    continue
                except ValueError:
                    pass
            # any other text after data has started = the scan was interrupted
            if current["rows"] and line:
                scans.append(current)
                current = None
    if current and current["rows"]:
        scans.append(current)
    for s in scans:
        s["rows"] = np.array(s["rows"])
    return scans


def merge_repeats(d):
    """Merge consecutive tilts (same plate position) with the same distance."""
    out = []
    for k in np.unique(d[:, 0]):
        s = d[d[:, 0] == k]
        s = s[np.argsort(s[:, 2])]
        group = [s[0]]
        for row in s[1:]:
            same = abs(row[3] - group[-1][3]) < MERGE_TOL
            next_tilt = row[2] - group[-1][2] <= 5.01
            if same and next_tilt:
                group.append(row)
            else:
                out.append(group)
                group = [row]
        out.append(group)
    merged = [[g[0][0], g[0][1], np.mean([r[2] for r in g]),
               np.mean([r[3] for r in g]), len(g)] for g in out]
    return np.array(merged)


def to_points(d, R, H):
    """d columns: position, plate_angle, inclination, distance, repeats."""
    ang = np.radians(d[:, 1])
    inc = np.radians(d[:, 2])
    dist = DIST_SCALE * d[:, 3] + DIST_OFFSET
    # triangle correction (speaker and microphone are side by side)
    dist = np.sqrt(np.maximum(dist**2 - (TR_SEPARATION / 2) ** 2, 0))

    # beam direction (towards the axis, tilted down) and its "up" normal, in the x-z plane
    ux, uz = -np.cos(inc), -np.sin(inc)
    nx, nz = -np.sin(inc), np.cos(inc)
    sx = R + PIVOT_FRONT * ux + PIVOT_UP * nx
    sz = H + PIVOT_FRONT * uz + PIVOT_UP * nz

    x = sx + dist * ux
    z = sz + dist * uz
    # the plate turned by 'ang': rotate back into the object frame
    xo = x * np.cos(ang) - PLATE_CX
    yo = -x * np.sin(ang) - PLATE_CY
    return np.c_[xo, yo, z, d[:, 1], d[:, 2], dist, d[:, 4]]


def save_ply(path, P):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(P)}\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in P:
            f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")


def make_figure(path, P, rejected, R, H, title):
    fig = plt.figure(figsize=(18, 5.5))
    fig.subplots_adjust(wspace=0.55)
    th = np.linspace(0, 2 * np.pi, 120)
    lim = max(R, np.abs(P[:, :2]).max() if len(P) else 1) + 1
    zmax = max(H, P[:, 2].max() if len(P) else 1) + 1

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=8, c=P[:, 4], cmap="viridis", depthshade=False)
    if len(rejected):
        ax.scatter(rejected[:, 0], rejected[:, 1], rejected[:, 2], s=6, c="#aaaaaa",
                   depthshade=False, label="rejected (plate)")
    ax.plot(R * np.cos(th), R * np.sin(th), np.full(th.size, H), ":", c="gray", label="sensor path")
    ax.plot([0, 0], [0, 0], [0, zmax], "k--", lw=1, label="plate axis")
    ax.scatter([R], [0], [H], marker="s", s=50, c="k", label="sensor")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, zmax)
    ax.set_box_aspect((2 * lim, 2 * lim, zmax))
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)"); ax.set_zlabel("z (cm)")
    ax.set_title("3D"); ax.legend(fontsize=7, loc="upper left")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(P[:, 0], P[:, 1], s=8 + 6 * (P[:, 6] - 1), c=P[:, 4], cmap="viridis")
    ax2.plot(0, 0, "k+", ms=12)
    ax2.set_aspect("equal"); ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("x (cm)"); ax2.set_ylabel("y (cm)"); ax2.set_title("Top view")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(np.hypot(P[:, 0], P[:, 1]), P[:, 2], s=8 + 6 * (P[:, 6] - 1), c=P[:, 4], cmap="viridis")
    ax3.plot(R, H, "ks", ms=8, label="sensor")
    ax3.set_xlim(0, lim); ax3.set_ylim(0, zmax); ax3.set_aspect("equal")
    ax3.set_xlabel("distance from plate axis (cm)"); ax3.set_ylabel("z (cm)")
    ax3.set_title("Side profile (all rotations)"); ax3.legend(fontsize=7)

    fig.colorbar(sc, ax=[ax, ax2, ax3], shrink=0.7, label="inclination (deg)")
    fig.suptitle(title)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_html(path, P, rejected, R, H, title):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False
    th = np.linspace(0, 2 * np.pi, 120)
    fig = go.Figure()
    fig.add_scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="points",
        marker=dict(size=3, color=P[:, 4], colorscale="Viridis", colorbar=dict(title="inc (deg)")),
        text=[f"plate {a:.0f}°, inc {i:.0f}°, d={r:.2f} cm, x{int(n)}" for a, i, r, n in P[:, 3:7]],
        hoverinfo="text")
    if len(rejected):
        fig.add_scatter3d(x=rejected[:, 0], y=rejected[:, 1], z=rejected[:, 2], mode="markers",
                          name="rejected (plate)", marker=dict(size=2, color="#aaaaaa"))
    fig.add_scatter3d(x=R * np.cos(th), y=R * np.sin(th), z=np.full(th.size, H), mode="lines",
                      name="sensor path", line=dict(color="gray", dash="dot"))
    fig.add_scatter3d(x=[0, 0], y=[0, 0], z=[0, max(H, 1) + 1], mode="lines", name="plate axis",
                      line=dict(color="black", dash="dash"))
    fig.update_layout(title=title, scene=dict(aspectmode="data", xaxis_title="x (cm)",
                      yaxis_title="y (cm)", zaxis_title="z (cm)"), margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(path, include_plotlyjs=True)
    return True


def main():
    ap = argparse.ArgumentParser(description="Ultrasonic scan log -> 3D points and figures")
    ap.add_argument("log")
    ap.add_argument("--scan", type=int, help="which scan in the file (1 = first); default = last complete")
    ap.add_argument("--list", action="store_true", help="list the scans in the file and exit")
    ap.add_argument("--R", type=float, help="override R (cm)")
    ap.add_argument("--H", type=float, help="override H (cm)")
    args = ap.parse_args()

    scans = parse_log(args.log)
    if not scans:
        raise SystemExit("No scan data found in the file.")

    if args.list:
        for i, s in enumerate(scans, 1):
            print(f"{i}: line {s['start_line']}, {len(s['rows'])} readings, "
                  f"{'complete' if s['complete'] else 'INTERRUPTED'}, R={s['R']} H={s['H']}")
        return

    if args.scan:
        idx = args.scan - 1
    else:
        complete = [i for i, s in enumerate(scans) if s["complete"]]
        idx = complete[-1] if complete else len(scans) - 1
    scan = scans[idx]

    R = args.R or R_SENSOR or scan["R"]
    H = args.H or H_SENSOR or scan["H"]
    if R is None or H is None:
        raise SystemExit("R/H not found in the log: set R_SENSOR/H_SENSOR or use --R/--H.")

    d = scan["rows"]
    valid = d[:, 3] > 0
    if MAX_DIST is not None:
        valid &= d[:, 3] <= MAX_DIST
    d = d[valid]
    d = merge_repeats(d) if MERGE else np.c_[d, np.ones(len(d))]

    P_all = to_points(d, R, H)
    keep = P_all[:, 2] >= Z_MIN
    P, rejected = P_all[keep], P_all[~keep]

    base = os.path.splitext(args.log)[0] + f"_scan{idx + 1}"
    header = "x_cm,y_cm,z_cm,plate_angle_deg,inclination_deg,distance_cm,repeats"
    np.savetxt(base + "_pontos.csv", P, delimiter=",", header=header, comments="", fmt="%.3f")
    np.save(base + "_pontos.npy", P)
    save_ply(base + "_pontos.ply", P)

    title = f"Scan {idx + 1}: R={R} cm, H={H} cm, {len(P)} points"
    make_figure(base + "_figura.png", P, rejected, R, H, title)
    html = make_html(base + "_figura.html", P, rejected, R, H, title)

    r = np.hypot(P[:, 0], P[:, 1])
    print(f"Scan {idx + 1} of {len(scans)} ({'complete' if scan['complete'] else 'interrupted'}), R={R}, H={H}")
    print(f"Readings with echo: {int(valid.sum())} of {len(scan['rows'])}")
    print(f"Points kept: {len(P)} (rejected as plate: {len(rejected)})"
          + (f", merged from {int(P[:, 6].sum())} readings" if MERGE else ""))
    if len(P):
        print(f"Radius from axis: median {np.median(r):.2f} cm (10-90%: {np.percentile(r, 10):.2f}-{np.percentile(r, 90):.2f})")
        print(f"Height range:     {P[:, 2].min():.2f} - {P[:, 2].max():.2f} cm")
    print("Saved:", base + "_pontos.csv/.npy/.ply,", base + "_figura.png" + (", .html" if html else ""))
    return P


if __name__ == "__main__":
    main()
