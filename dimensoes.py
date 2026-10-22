"""
dimensoes.py - object dimensions with uncertainties from one or more scan logs,
compared with the real (caliper) values.

Usage:
    python dimensoes.py scan_a.txt                         # one run
    python dimensoes.py scan_a.txt scan_b.txt scan_c.txt   # repeats and/or other heights
    python dimensoes.py scan_a.txt --D 2.60 0.02 --h 11.0 0.05
                                   (caliper diameter and height, each value and sigma, cm)

Each log can be pasted straight from the Serial Monitor; the last complete scan in
each file is used, with R and H read from its "# Geometria" line. Needs scan3d.py
in the same folder, plus numpy, scipy and matplotlib.

Method (what goes in the report):
  1. Each reading's distance is corrected (offset, scale, speaker-microphone triangle).
  2. The object is modelled as an upright cylinder (radius r, height h, centre per scan).
     For every reading the model predicts the distance to the NEAREST surface point
     inside the beam cone (half-angle beta), which is what the HC-SR04 reports.
     r, h, beta and the centres are fitted to all readings together (robust least squares).
  3. Uncertainty of each result, combined in quadrature:
       fit         from the fit's covariance matrix
       scan-scan   standard deviation of the result fitted to each scan alone (>= 2 scans)
       calibration change of the result when each calibration value moves by its sigma
                   (R, H, servo offset, distance offset, pivot), refitting each time
  4. z = |scan - real| / sqrt(sigma_scan^2 + sigma_real^2).
"""

import argparse
import os

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scan3d import parse_log

# ========================= SETTINGS =========================
# Calibration values and their sigma (cm or degrees). Value = correction you apply.
CAL = {
    #  name           value  sigma
    "R":            (None,  0.10),   # None = from the log
    "H":            (None,  0.10),   # None = from the log
    "servo_offset": (0.0,   1.0),    # deg added to the logged inclination
    "dist_offset":  (0.0,   0.05),   # cm, real - measured (0 if already in the Arduino)
    "pivot_front":  (0.0,   0.10),   # cm, transducers in front of the servo axis
    "pivot_up":     (0.0,   0.10),   # cm, transducers above the servo axis (when level)
    "beta":         (None,  2.0),    # deg, beam half-angle from the rod sweep. None = fit it
                                     # from the scan (only trustworthy for wide objects: for a
                                     # thin one, object width and beam width trade off)
}
DIST_SCALE = 1.0          # real = DIST_SCALE * measured + dist_offset
TR_SEPARATION = 2.6       # cm, speaker to microphone

# Which readings are the object
MAX_DIST = 12.0           # cm: further readings are plate / background
Z_MIN = 1.0               # cm: reconstructed points below this are plate echoes

# Fit
BETA_START = 12.0         # deg, starting beam half-angle
ROBUST_SCALE = 0.3        # cm, residuals much larger than this count less (outliers)
H_UNCONSTRAINED = 1.0     # cm: if the total sigma(h) exceeds this, height is reported as not measured
# ============================================================


def load(path, R=None, H=None):
    scans = parse_log(path)
    complete = [s for s in scans if s["complete"]] or scans
    s = complete[-1]
    d = s["rows"]
    return dict(name=os.path.basename(path), d=d[d[:, 3] > 0],
                R=R or s["R"], H=H or s["H"], n_total=len(d))


def geometry(d, R, H, cal):
    """Sensor position, beam direction and corrected distance for each reading (lab x-z plane)."""
    inc = np.radians(d[:, 2] + cal["servo_offset"])
    dist = DIST_SCALE * d[:, 3] + cal["dist_offset"]
    dist = np.sqrt(np.maximum(dist**2 - (TR_SEPARATION / 2) ** 2, 0))
    ux, uz = -np.cos(inc), -np.sin(inc)
    nx, nz = -np.sin(inc), np.cos(inc)
    sx = R + cal["pivot_front"] * ux + cal["pivot_up"] * nx
    sz = H + cal["pivot_front"] * uz + cal["pivot_up"] * nz
    return sx, sz, ux, uz, dist


# Fixed grid on a unit cylinder, scaled by (r, h): the surface moves smoothly with r and h
_TH = np.linspace(0, 2 * np.pi, 180, endpoint=False)
_T, _ZF = np.meshgrid(_TH, np.linspace(0, 1, 60))
_T2, _RF = np.meshgrid(_TH, np.linspace(0, 1, 14))
SOFT_MIN = 0.03           # cm: smoothing of "nearest point" (keeps the fit from sticking)
CONE_EDGE = 0.002         # smoothing of the cone edge, in cos(angle)


def cylinder_surface(r, h):
    z0 = 0.3
    side = np.c_[r * np.cos(_T).ravel(), r * np.sin(_T).ravel(), (z0 + _ZF * (h - z0)).ravel()]
    top = np.c_[(r * _RF * np.cos(_T2)).ravel(), (r * _RF * np.sin(_T2)).ravel(), np.full(_T2.size, h)]
    return np.vstack([side, top])


def predict(scan, geo, r, h, beta, cx, cy):
    """Distance to the nearest surface point inside the cone, for every reading.

    'Nearest' and 'inside' are smoothed (soft minimum, soft cone edge) so the result
    changes continuously with the parameters; otherwise the fit gets stuck."""
    sx, sz, ux, uz, _ = geo
    P = cylinder_surface(r, h)
    P[:, 0] += cx
    P[:, 1] += cy
    cb = np.cos(np.radians(beta))
    ang = np.radians(scan["d"][:, 1])
    out = np.empty(len(ang))
    for a in np.unique(ang):
        idx = np.where(ang == a)[0]
        # object turned by the plate angle a; sensor fixed on the +x axis (y = 0)
        x = P[:, 0] * np.cos(a) - P[:, 1] * np.sin(a)
        y = -(P[:, 0] * np.sin(a) + P[:, 1] * np.cos(a))
        vx = x[None, :] - sx[idx, None]
        vz = P[None, :, 2] - sz[idx, None]
        n = np.sqrt(vx**2 + y[None, :] ** 2 + vz**2)
        c = (vx * ux[idx, None] + vz * uz[idx, None]) / n
        # outside the cone -> distance pushed up smoothly (about 2 cm per 3 deg outside)
        outside = CONE_EDGE * np.logaddexp(0, (cb - c) / CONE_EDGE)
        n_eff = n + 200.0 * outside
        m = n_eff.min(axis=1, keepdims=True)
        w = np.exp(-(n_eff - m) / SOFT_MIN)            # weighted mean of the nearest points
        out[idx] = (w * n_eff).sum(axis=1) / w.sum(axis=1)
    return out


def select_object(scan, cal):
    """Keep readings that are the object (not plate, not background)."""
    d = scan["d"]
    d = d[d[:, 3] <= MAX_DIST]
    sx, sz, ux, uz, dist = geometry(d, scan["R"], scan["H"], cal)
    z = sz + dist * uz
    return dict(scan, d=d[z >= Z_MIN])


def fit(scans, cal, start=None):
    """Parameters p = [r, h, beta, cx1, cy1, cx2, cy2, ...]; beta is held fixed if cal['beta'] is set."""
    geos = [geometry(s["d"], s["R"], s["H"], cal) for s in scans]
    ns = len(scans)
    fixed_beta = cal.get("beta")
    free = [0, 1] + ([] if fixed_beta is not None else [2]) + list(range(3, 3 + 2 * ns))

    def full(q):
        p = np.empty(3 + 2 * ns)
        p[free] = q
        if fixed_beta is not None:
            p[2] = fixed_beta
        return p

    def residuals(q):
        p = full(q)
        r, h, beta = p[:3]
        res = []
        for k, (s, g) in enumerate(zip(scans, geos)):
            cx, cy = p[3 + 2 * k], p[4 + 2 * k]
            res.append(predict(s, g, r, h, beta, cx, cy) - g[4])
        return np.concatenate(res)

    if start is None:
        # starting guess from the raw cloud
        raw_r, zmax = [], []
        for s, g in zip(scans, geos):
            sx, sz, ux, uz, dist = g
            raw_r.append(np.abs(sx + dist * ux))
            zmax.append(sz + dist * uz)
        r0 = 0.8 * np.median(np.concatenate(raw_r))
        h0 = np.max(np.concatenate(zmax)) + 1.0
        starts = [[r0, h0, BETA_START], [0.6 * r0 / 0.8, h0 + 2, BETA_START + 5]]
        starts = [s0 + [0.0, 0.0] * ns for s0 in starts]
    else:
        starts = [start]

    lo = np.array([0.3, 1.0, 3.0] + [-3.0] * 2 * ns)[free]
    hi = np.array([10.0, 25.0, 40.0] + [3.0] * 2 * ns)[free]
    best = None
    for p0 in starts:
        q0 = np.clip(np.array(p0, float)[free], lo + 1e-3, hi - 1e-3)
        sol = least_squares(residuals, q0, bounds=(lo, hi), loss="soft_l1",
                            f_scale=ROBUST_SCALE, diff_step=0.01)
        if best is None or sol.cost < best.cost:
            best = sol

    # covariance from the Jacobian, scaled by the residual scatter
    J = best.jac
    res = best.fun
    dof = max(len(res) - len(best.x), 1)
    s2 = np.sum(res**2) / dof
    try:
        cov_q = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        cov_q = np.full((len(best.x), len(best.x)), np.inf)
    cov = np.zeros((3 + 2 * ns, 3 + 2 * ns))
    cov[np.ix_(free, free)] = cov_q
    return full(best.x), np.sqrt(np.abs(np.diag(cov))), res, geos, cov


def results_from(p):
    r, h = p[0], p[1]
    return dict(D=2 * r, h=h, V=np.pi * r**2 * h, beta=p[2])


def main():
    ap = argparse.ArgumentParser(description="Object dimensions with uncertainties from scan logs")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--D", nargs=2, type=float, metavar=("VALUE", "SIGMA"), help="caliper diameter (cm)")
    ap.add_argument("--h", nargs=2, type=float, metavar=("VALUE", "SIGMA"), help="caliper height (cm)")
    ap.add_argument("--R", type=float, help="override R for all logs (cm)")
    ap.add_argument("--H", type=float, help="override H for all logs (cm)")
    ap.add_argument("--beta", nargs=2, type=float, metavar=("VALUE", "SIGMA"),
                    help="beam half-angle from the rod sweep (deg); overrides CAL['beta']")
    ap.add_argument("--fast", action="store_true", help="skip the calibration part (quick look)")
    args = ap.parse_args()

    cal = {k: v[0] for k, v in CAL.items()}
    if args.beta:
        CAL["beta"] = tuple(args.beta)
        cal["beta"] = args.beta[0]
    scans_all = [load(f, args.R or cal["R"], args.H or cal["H"]) for f in args.logs]
    scans = [select_object(s, cal) for s in scans_all]
    for s in scans:
        print(f"{s['name']}: R={s['R']} H={s['H']}, {len(s['d'])} object readings "
              f"of {s['n_total']}")

    # 1. joint fit
    p, sp, res, geos, cov = fit(scans, cal)
    best = results_from(p)
    sig = {"D": {"fit": 2 * sp[0]}, "h": {"fit": sp[1]}}

    # raw cloud estimate, for the bias
    raw_r = np.concatenate([np.abs(g[0] + g[4] * g[2]) for g in geos])
    D_raw = 2 * np.median(raw_r)

    # 2. scan-to-scan
    if len(scans) >= 2:
        per = []
        for k, s in enumerate(scans):
            start = list(p[:3]) + list(p[3 + 2 * k: 5 + 2 * k])
            pk = fit([s], cal, start=start)[0]
            per.append(results_from(pk))
        sig["D"]["scan-scan"] = np.std([q["D"] for q in per], ddof=1)
        sig["h"]["scan-scan"] = np.std([q["h"] for q in per], ddof=1)
    else:
        per = None

    # 3. calibration sensitivities: move each value by +sigma, refit from the best solution
    contrib = {}
    if not args.fast:
        for name, (value, sigma) in CAL.items():
            cal2 = dict(cal)
            if name == "beta" and cal["beta"] is None:
                continue                      # beta was fitted: already in the fit part
            if name in ("R", "H"):
                scans2 = [dict(s, **{name: s[name] + sigma}) for s in scans]
            else:
                cal2[name] = (value or 0.0) + sigma
                scans2 = scans
            p2 = fit(scans2, cal2, start=list(p))[0]
            q = results_from(p2)
            contrib[name] = (abs(q["D"] - best["D"]), abs(q["h"] - best["h"]))
        sig["D"]["calibration"] = np.sqrt(sum(c[0] ** 2 for c in contrib.values()))
        sig["h"]["calibration"] = np.sqrt(sum(c[1] ** 2 for c in contrib.values()))

    total = {k: np.sqrt(sum(v ** 2 for v in sig[k].values())) for k in sig}
    h_ok = total["h"] < H_UNCONSTRAINED

    # ---------------- report ----------------
    print("\n=== Fit ===")
    print(f"Readings fitted: {len(res)} | median |residual| {np.median(np.abs(res)):.2f} cm | "
          f"within 0.5 cm: {100 * np.mean(np.abs(res) < 0.5):.0f}%")
    if cal["beta"] is None:
        rho = cov[0, 2] / np.sqrt(cov[0, 0] * cov[2, 2]) if cov[0, 0] * cov[2, 2] > 0 else 0
        print(f"Beam half-angle beta = {p[2]:.1f} ± {sp[2]:.1f} deg (fitted; correlation with r: {rho:+.2f})")
        if abs(rho) > 0.7:
            print("  WARNING: radius and beam width trade off in this fit. Measure beta with the rod")
            print("  sweep and set CAL['beta'] = (value, sigma) for a trustworthy diameter.")
    else:
        print(f"Beam half-angle beta = {p[2]:.1f} ± {CAL['beta'][1]:.1f} deg (fixed, from the rod sweep)")
    for k in range(len(scans)):
        print(f"Centre offset, {scans[k]['name']}: {np.hypot(p[3 + 2 * k], p[4 + 2 * k]):.2f} cm")

    print("\n=== Uncertainty budget (cm) ===")
    print(f"{'part':14s}{'diameter':>10s}{'height':>10s}")
    for part in sig["D"]:
        print(f"{part:14s}{sig['D'][part]:10.2f}{sig['h'][part]:10.2f}")
    if contrib:
        print("  calibration, by source:")
        for name, (cD, ch) in contrib.items():
            print(f"    {name:12s}{cD:10.2f}{ch:10.2f}")
    print(f"{'TOTAL':14s}{total['D']:10.2f}{total['h']:10.2f}")

    print("\n=== Result ===")
    print(f"Diameter D = {best['D']:.2f} ± {total['D']:.2f} cm   (raw point cloud: {D_raw:.2f} cm)")
    if h_ok:
        sV = best["V"] * np.hypot(2 * total["D"] / best["D"], total["h"] / best["h"])
        print(f"Height   h = {best['h']:.2f} ± {total['h']:.2f} cm")
        print(f"Volume   V = {best['V']:.1f} ± {sV:.1f} cm^3")
    else:
        print(f"Height: NOT constrained by these scans (sigma {total['h']:.1f} cm) - "
              f"add a scan from another height or flatter tilts")
    if per:
        print("Per scan:", ", ".join(f"{s['name']}: D={q['D']:.2f} h={q['h']:.2f}"
                                    for s, q in zip(scans, per)))

    if args.D:
        Dr, sDr = args.D
        z = abs(best["D"] - Dr) / np.hypot(total["D"], sDr)
        zr = abs(D_raw - Dr) / np.hypot(total["D"], sDr)
        print(f"\nvs caliper D = {Dr:.2f} ± {sDr:.2f} cm: difference {best['D'] - Dr:+.2f} cm, z = {z:.1f}"
              f"   (raw cloud: {D_raw - Dr:+.2f} cm, z = {zr:.1f})")
    if args.h and h_ok:
        hr, shr = args.h
        z = abs(best["h"] - hr) / np.hypot(total["h"], shr)
        print(f"vs caliper h = {hr:.2f} ± {shr:.2f} cm: difference {best['h'] - hr:+.2f} cm, z = {z:.1f}")
    print("z < 2 consistent, 2-3 tension, > 3 find the cause")

    # ---------------- figure ----------------
    base = os.path.splitext(args.logs[0])[0] + "_dimensoes"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    off = 0
    for k, (s, g) in enumerate(zip(scans, geos)):
        sx, sz, ux, uz, dist = g
        x = sx + dist * ux
        z = sz + dist * uz
        ang = np.radians(s["d"][:, 1])
        xo = x * np.cos(ang) - p[3 + 2 * k]
        yo = -x * np.sin(ang) - p[4 + 2 * k]
        a1.scatter(np.hypot(xo, yo), z, s=8, alpha=0.6, label=f"{s['name']} (H={s['H']})")
        n = len(s["d"])
        a2.scatter(s["d"][:, 1], res[off:off + n], s=8, alpha=0.6)
        off += n
        a1.plot(s["R"], s["H"], "ks", ms=7)
    D, h = best["D"], best["h"]
    a1.plot([D / 2, D / 2, 0], [0, h, h], "r-", lw=2, label=f"fit: D = {D:.2f} ± {total['D']:.2f} cm")
    a1.axvspan(D / 2 - total["D"] / 2, D / 2 + total["D"] / 2, color="r", alpha=0.12)
    if args.D:
        hr = args.h[0] if args.h else h
        a1.plot([args.D[0] / 2, args.D[0] / 2, 0], [0, hr, hr], "k--", lw=1.5,
                label=f"caliper: D = {args.D[0]:.2f} cm")
    a1.set_xlabel("distance from the object axis (cm)")
    a1.set_ylabel("height z (cm)")
    a1.set_title("Side profile: points (raw, on the beam axis) and fit")
    a1.set_xlim(0, None); a1.set_ylim(0, None); a1.set_aspect("equal"); a1.legend(fontsize=8)
    a2.axhline(0, color="k", lw=1)
    a2.axhspan(-ROBUST_SCALE, ROBUST_SCALE, color="gray", alpha=0.2)
    a2.set_xlabel("plate angle (deg)")
    a2.set_ylabel("measured - model (cm)")
    a2.set_title("Residuals: no trend expected")
    plt.tight_layout()
    plt.savefig(base + ".png", dpi=130)
    print(f"\nFigure: {base}.png")


if __name__ == "__main__":
    main()
