"""
prisma.py - dimensions of a PRISM with uncertainties from one or more scan logs,
compared with the real (caliper) values. Same method as dimensoes.py (cylinder).

Shapes:
    box           upright rectangular prism: length a, width b, height h   (default)
    regular N     upright regular prism with N sides of length s (3 = triangle, 6 = hexagon)

Usage:
    python prisma.py box_H12.txt                                    # one run
    python prisma.py box_H12.txt box_H155.txt                       # repeats and/or other heights
    python prisma.py box_H12.txt --a 6.00 0.02 --b 4.00 0.02 --h 10.0 0.05 --beta 13 1.5
    python prisma.py tri_H12.txt --sides 3 --s 5.00 0.02 --h 8.0 0.05 --beta 13 1.5

    --a/--b/--s/--h  caliper values: value and sigma, in cm (order of a and b does not matter)
    --beta           beam half-angle from the rod sweep: value and sigma, in degrees
    --fast           skip the calibration part (quick look)

Each log can be pasted straight from the Serial Monitor; the last complete scan in each
file is used, with R and H from its "# Geometria" line. Needs scan3d.py in the same folder,
plus numpy, scipy and matplotlib. A full run takes a few minutes.

Method (what goes in the report):
  1. Each reading's distance is corrected (offset, scale, speaker-microphone triangle).
  2. The object is an upright prism: its size and height are shared by all scans; its
     centre and its rotation angle on the plate are fitted separately for each scan (it is
     put down anew each time). For every reading the model predicts the distance to the
     NEAREST surface point inside the beam cone (half-angle beta).
  3. Uncertainty of each result, combined in quadrature:
       fit         from the fit's covariance matrix
       scan-scan   standard deviation of the result fitted to each scan alone (>= 2 scans)
       calibration change of the result when each calibration value moves by its sigma
  4. z = |scan - real| / sqrt(sigma_scan^2 + sigma_real^2).

Physics note: a flat face square to the beam returns its distance exactly, whatever the
beam width; edges and corners are where the beam width matters. So a prism's face-to-face
size is usually better determined than a cylinder's diameter.
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
CAL = {
    #  name           value  sigma
    "R":            (None,  0.10),   # cm; None = from the log
    "H":            (None,  0.10),   # cm; None = from the log
    "servo_offset": (0.0,   1.0),    # deg added to the logged inclination
    "dist_offset":  (0.0,   0.05),   # cm, real - measured (0 if already in the Arduino)
    "pivot_front":  (0.0,   0.10),   # cm, transducers in front of the servo axis
    "pivot_up":     (0.0,   0.10),   # cm, transducers above the servo axis (when level)
    "beta":         (None,  2.0),    # deg, from the rod sweep. None = fit it from the scan
}
DIST_SCALE = 1.0          # real = DIST_SCALE * measured + dist_offset
TR_SEPARATION = 2.6       # cm, speaker to microphone

MAX_DIST = 12.0           # cm: further readings are plate / background
Z_MIN = 1.0               # cm: reconstructed points below this are plate echoes

BETA_START = 12.0         # deg
ROBUST_SCALE = 0.3        # cm
H_UNCONSTRAINED = 1.0     # cm: if the total sigma(h) exceeds this, height is not reported
SOFT_MIN = 0.03           # cm, smoothing of "nearest point"
CONE_EDGE = 0.002         # smoothing of the cone edge, in cos(angle)
# ============================================================

_E = np.linspace(0, 1, 28, endpoint=False)   # samples along each edge
_ZF = np.linspace(0, 1, 40)                  # samples up each face
_RF = np.linspace(0, 1, 10)                  # samples from the centre out, on the top


class Shape:
    def __init__(self, sides, box):
        self.box = box
        self.N = 4 if box else sides
        self.n_size = 2 if self.box else 1
        self.names = ["a", "b"] if self.box else ["s"]

    def vertices(self, size, phi):
        if self.box:
            a, b = size
            v = np.array([[a / 2, b / 2], [-a / 2, b / 2], [-a / 2, -b / 2], [a / 2, -b / 2]])
        else:
            s = size[0]
            Rc = s / (2 * np.sin(np.pi / self.N))
            t = 2 * np.pi * np.arange(self.N) / self.N
            v = np.c_[Rc * np.cos(t), Rc * np.sin(t)]
        c, sn = np.cos(phi), np.sin(phi)
        return v @ np.array([[c, sn], [-sn, c]])

    def surface(self, size, h, phi):
        v = self.vertices(size, phi)
        w = np.roll(v, -1, axis=0)
        edge = v[:, None, :] + _E[None, :, None] * (w - v)[:, None, :]      # (N, E, 2)
        edge = edge.reshape(-1, 2)
        z0 = 0.3
        Z = z0 + _ZF * (h - z0)
        side = np.c_[np.repeat(edge, len(Z), axis=0), np.tile(Z, len(edge))]
        top_xy = (_RF[:, None, None] * edge[None, :, :]).reshape(-1, 2)
        top = np.c_[top_xy, np.full(len(top_xy), h)]
        return np.vstack([side, top])

    def volume(self, size, h):
        if self.box:
            return size[0] * size[1] * h
        s = size[0]
        return self.N * s**2 * h / (4 * np.tan(np.pi / self.N))

    def volume_rel_sigma(self, size, ssize, h, sh):
        if self.box:
            return np.sqrt((ssize[0] / size[0]) ** 2 + (ssize[1] / size[1]) ** 2 + (sh / h) ** 2)
        return np.sqrt((2 * ssize[0] / size[0]) ** 2 + (sh / h) ** 2)


def load(path, R=None, H=None):
    scans = parse_log(path)
    complete = [s for s in scans if s["complete"]] or scans
    s = complete[-1]
    d = s["rows"]
    return dict(name=os.path.basename(path), d=d[d[:, 3] > 0],
                R=R or s["R"], H=H or s["H"], n_total=len(d))


def geometry(d, R, H, cal):
    inc = np.radians(d[:, 2] + cal["servo_offset"])
    dist = DIST_SCALE * d[:, 3] + cal["dist_offset"]
    dist = np.sqrt(np.maximum(dist**2 - (TR_SEPARATION / 2) ** 2, 0))
    ux, uz = -np.cos(inc), -np.sin(inc)
    nx, nz = -np.sin(inc), np.cos(inc)
    sx = R + cal["pivot_front"] * ux + cal["pivot_up"] * nx
    sz = H + cal["pivot_front"] * uz + cal["pivot_up"] * nz
    return sx, sz, ux, uz, dist


def select_object(scan, cal):
    d = scan["d"]
    d = d[d[:, 3] <= MAX_DIST]
    sx, sz, ux, uz, dist = geometry(d, scan["R"], scan["H"], cal)
    return dict(scan, d=d[sz + dist * uz >= Z_MIN])


def predict(scan, geo, P, beta):
    """Soft distance to the nearest surface point inside the cone, for each reading."""
    sx, sz, ux, uz, _ = geo
    cb = np.cos(np.radians(beta))
    ang = np.radians(scan["d"][:, 1])
    out = np.empty(len(ang))
    for a in np.unique(ang):
        idx = np.where(ang == a)[0]
        x = P[:, 0] * np.cos(a) - P[:, 1] * np.sin(a)
        y = -(P[:, 0] * np.sin(a) + P[:, 1] * np.cos(a))
        vx = x[None, :] - sx[idx, None]
        vz = P[None, :, 2] - sz[idx, None]
        n = np.sqrt(vx**2 + y[None, :] ** 2 + vz**2)
        c = (vx * ux[idx, None] + vz * uz[idx, None]) / n
        n_eff = n + 200.0 * CONE_EDGE * np.logaddexp(0, (cb - c) / CONE_EDGE)
        m = n_eff.min(axis=1, keepdims=True)
        w = np.exp(-(n_eff - m) / SOFT_MIN)
        out[idx] = (w * n_eff).sum(axis=1) / w.sum(axis=1)
    return out


def fit(scans, cal, shape, starts):
    """p = [size..., h, beta, (cx, cy, phi) per scan]. beta fixed if cal['beta'] is set."""
    geos = [geometry(s["d"], s["R"], s["H"], cal) for s in scans]
    ns, k0 = len(scans), shape.n_size
    npar = k0 + 2 + 3 * ns
    fixed_beta = cal.get("beta")
    free = [i for i in range(npar) if not (i == k0 + 1 and fixed_beta is not None)]

    def full(q):
        p = np.empty(npar)
        p[free] = q
        if fixed_beta is not None:
            p[k0 + 1] = fixed_beta
        return p

    def residuals(q):
        p = full(q)
        size, h, beta = p[:k0], p[k0], p[k0 + 1]
        res = []
        for k, (s, g) in enumerate(zip(scans, geos)):
            cx, cy, phi = p[k0 + 2 + 3 * k: k0 + 5 + 3 * k]
            P = shape.surface(size, h, phi)
            P[:, 0] += cx
            P[:, 1] += cy
            res.append(predict(s, g, P, beta) - g[4])
        return np.concatenate(res)

    lo = np.array([0.5] * k0 + [1.0, 3.0] + [-3.0, -3.0, -2 * np.pi] * ns)[free]
    hi = np.array([20.0] * k0 + [25.0, 40.0] + [3.0, 3.0, 2 * np.pi] * ns)[free]
    best = None
    for p0 in starts:
        q0 = np.clip(np.array(p0, float)[free], lo + 1e-3, hi - 1e-3)
        sol = least_squares(residuals, q0, bounds=(lo, hi), loss="soft_l1",
                            f_scale=ROBUST_SCALE, diff_step=0.01)
        if best is None or sol.cost < best.cost:
            best = sol
    J, res = best.jac, best.fun
    s2 = np.sum(res**2) / max(len(res) - len(best.x), 1)
    try:
        cov_q = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        cov_q = np.full((len(best.x), len(best.x)), np.inf)
    cov = np.zeros((npar, npar))
    cov[np.ix_(free, free)] = cov_q
    return full(best.x), np.sqrt(np.abs(np.diag(cov))), res, geos, cov


def first_guess(scan, cal, shape):
    sx, sz, ux, uz, dist = geometry(scan["d"], scan["R"], scan["H"], cal)
    r0 = 0.8 * np.median(np.abs(sx + dist * ux))
    h0 = np.max(sz + dist * uz) + 1.0
    if shape.box:
        size = [1.7 * r0, 1.7 * r0 * 0.8]
        phis = np.radians([0, 30, 60])                    # box: phi + 90 = a and b swapped
    else:
        size = [2 * r0 * np.tan(np.pi / shape.N)]          # side of an N-gon with inradius r0
        phis = np.radians(np.arange(3) * 120 / shape.N)
    return size, h0, phis


def results_from(p, shape):
    k0 = shape.n_size
    size = sorted(p[:k0], reverse=True)
    return dict(size=np.array(size), h=p[k0], V=shape.volume(size, p[k0]))


def main():
    ap = argparse.ArgumentParser(description="Prism dimensions with uncertainties from scan logs")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--sides", type=int, default=0, help="regular prism with N sides (default: box)")
    ap.add_argument("--a", nargs=2, type=float, metavar=("VALUE", "SIGMA"), help="box: caliper length (cm)")
    ap.add_argument("--b", nargs=2, type=float, metavar=("VALUE", "SIGMA"), help="box: caliper width (cm)")
    ap.add_argument("--s", nargs=2, type=float, metavar=("VALUE", "SIGMA"), help="regular: caliper side (cm)")
    ap.add_argument("--h", nargs=2, type=float, metavar=("VALUE", "SIGMA"), help="caliper height (cm)")
    ap.add_argument("--R", type=float, help="override R for all logs (cm)")
    ap.add_argument("--H", type=float, help="override H for all logs (cm)")
    ap.add_argument("--beta", nargs=2, type=float, metavar=("VALUE", "SIGMA"),
                    help="beam half-angle from the rod sweep (deg)")
    ap.add_argument("--fast", action="store_true", help="skip the calibration part")
    args = ap.parse_args()

    shape = Shape(args.sides, box=(args.sides == 0))
    kind = "box" if shape.box else f"regular prism, {shape.N} sides"
    k0 = shape.n_size

    cal = {k: v[0] for k, v in CAL.items()}
    if args.beta:
        CAL["beta"] = tuple(args.beta)
        cal["beta"] = args.beta[0]
    scans = [select_object(load(f, args.R or cal["R"], args.H or cal["H"]), cal) for f in args.logs]
    print(f"Shape: {kind}")
    for s in scans:
        print(f"{s['name']}: R={s['R']} H={s['H']}, {len(s['d'])} object readings of {s['n_total']}")

    # 1. each scan alone (several starting rotations), then all together
    per, per_p = [], []
    for s in scans:
        size0, h0, phis = first_guess(s, cal, shape)
        starts = [list(size0) + [h0, BETA_START, 0.0, 0.0, ph] for ph in phis]
        pk = fit([s], cal, shape, starts)[0]
        per_p.append(pk)
        per.append(results_from(pk, shape))
    if len(scans) > 1:
        start = list(np.mean([q[:k0 + 2] for q in per_p], axis=0))
        for pk in per_p:
            start += list(pk[k0 + 2:k0 + 5])
        p, sp, res, geos, cov = fit(scans, cal, shape, [start])
    else:
        p, sp, res, geos, cov = fit(scans, cal, shape, [per_p[0]])
    best = results_from(p, shape)
    order = np.argsort(-p[:k0])                             # sorted (largest first), like results_from
    sig = {n: {"fit": sp[:k0][order][i]} for i, n in enumerate(shape.names)}
    sig["h"] = {"fit": sp[k0]}

    # 2. scan-to-scan
    if len(scans) >= 2:
        for i, n in enumerate(shape.names):
            sig[n]["scan-scan"] = np.std([q["size"][i] for q in per], ddof=1)
        sig["h"]["scan-scan"] = np.std([q["h"] for q in per], ddof=1)

    # 3. calibration sensitivities
    contrib = {}
    if not args.fast:
        for name, (value, sigma) in CAL.items():
            if name == "beta" and cal["beta"] is None:
                continue
            cal2 = dict(cal)
            if name in ("R", "H"):
                scans2 = [dict(s, **{name: s[name] + sigma}) for s in scans]
            else:
                cal2[name] = (value or 0.0) + sigma
                scans2 = scans
            q = results_from(fit(scans2, cal2, shape, [list(p)])[0], shape)
            contrib[name] = [abs(q["size"][i] - best["size"][i]) for i in range(k0)] + [abs(q["h"] - best["h"])]
        for i, n in enumerate(shape.names + ["h"]):
            sig[n]["calibration"] = np.sqrt(sum(c[i] ** 2 for c in contrib.values()))

    total = {n: np.sqrt(sum(v ** 2 for v in sig[n].values())) for n in sig}
    h_ok = total["h"] < H_UNCONSTRAINED

    # raw-cloud size (points placed on the beam axis), for the bias
    raw = []
    for k, (s, g) in enumerate(zip(scans, geos)):
        sx, sz, ux, uz, dist = g
        x = sx + dist * ux
        ang = np.radians(s["d"][:, 1])
        cx, cy, phi = p[k0 + 2 + 3 * k: k0 + 5 + 3 * k]
        xo, yo = x * np.cos(ang) - cx, -x * np.sin(ang) - cy
        c, sn = np.cos(-phi), np.sin(-phi)                  # into the prism's own axes
        raw.append(np.c_[xo * c - yo * sn, xo * sn + yo * c])
    raw = np.vstack(raw)
    if shape.box:
        a_, b_ = p[0], p[1]
        on_x = np.abs(raw[:, 0]) / a_ > np.abs(raw[:, 1]) / b_
        raw_size = sorted([2 * np.median(np.abs(raw[on_x, 0])), 2 * np.median(np.abs(raw[~on_x, 1]))], reverse=True)
    else:
        rin = np.median(np.hypot(raw[:, 0], raw[:, 1]))     # roughly the inradius
        raw_size = [2 * rin * np.tan(np.pi / shape.N)]

    # ---------------- report ----------------
    print("\n=== Fit ===")
    print(f"Readings fitted: {len(res)} | median |residual| {np.median(np.abs(res)):.2f} cm | "
          f"within 0.5 cm: {100 * np.mean(np.abs(res) < 0.5):.0f}%")
    if cal["beta"] is None:
        rho = cov[0, k0 + 1] / np.sqrt(cov[0, 0] * cov[k0 + 1, k0 + 1]) if cov[0, 0] * cov[k0 + 1, k0 + 1] > 0 else 0
        print(f"Beam half-angle beta = {p[k0 + 1]:.1f} ± {sp[k0 + 1]:.1f} deg (fitted; correlation with size: {rho:+.2f})")
        if abs(rho) > 0.7:
            print("  WARNING: size and beam width trade off. Measure beta with the rod sweep (--beta).")
    else:
        print(f"Beam half-angle beta = {p[k0 + 1]:.1f} ± {CAL['beta'][1]:.1f} deg (fixed, from the rod sweep)")
    for k, s in enumerate(scans):
        cx, cy, phi = p[k0 + 2 + 3 * k: k0 + 5 + 3 * k]
        print(f"{s['name']}: centre offset {np.hypot(cx, cy):.2f} cm, rotation {np.degrees(phi) % 360:.0f} deg")

    names = shape.names + ["h"]
    print("\n=== Uncertainty budget (cm) ===")
    print(f"{'part':14s}" + "".join(f"{n:>10s}" for n in names))
    for part in sig["h"]:
        print(f"{part:14s}" + "".join(f"{sig[n].get(part, 0):10.2f}" for n in names))
    if contrib:
        print("  calibration, by source:")
        for name, c in contrib.items():
            print(f"    {name:12s}" + "".join(f"{x:10.2f}" for x in c))
    print(f"{'TOTAL':14s}" + "".join(f"{total[n]:10.2f}" for n in names))

    print("\n=== Result ===")
    for i, n in enumerate(shape.names):
        label = {"a": "Length a", "b": "Width  b", "s": "Side   s"}[n]
        print(f"{label} = {best['size'][i]:.2f} ± {total[n]:.2f} cm   (raw point cloud: {raw_size[i]:.2f} cm)")
    if h_ok:
        ssize = [total[n] for n in shape.names]
        sV = best["V"] * shape.volume_rel_sigma(best["size"], ssize, best["h"], total["h"])
        print(f"Height   h = {best['h']:.2f} ± {total['h']:.2f} cm")
        print(f"Volume   V = {best['V']:.1f} ± {sV:.1f} cm^3")
    else:
        print(f"Height: NOT constrained by these scans (sigma {total['h']:.1f} cm) - "
              f"add a scan from another height or flatter tilts")
    if len(scans) > 1:
        for s, q in zip(scans, per):
            print(f"  alone, {s['name']}: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(shape.names, q["size"]))
                  + f", h={q['h']:.2f}")

    real = []
    if shape.box and (args.a or args.b):
        real = sorted([x for x in (args.a, args.b) if x], key=lambda t: -t[0])
    elif not shape.box and args.s:
        real = [args.s]
    if real:
        print()
        for i, (val, sv) in enumerate(real):
            n = shape.names[i]
            z = abs(best["size"][i] - val) / np.hypot(total[n], sv)
            zr = abs(raw_size[i] - val) / np.hypot(total[n], sv)
            print(f"vs caliper {n} = {val:.2f} ± {sv:.2f} cm: difference {best['size'][i] - val:+.2f} cm, z = {z:.1f}"
                  f"   (raw cloud: {raw_size[i] - val:+.2f} cm, z = {zr:.1f})")
    if args.h and h_ok:
        hr, shr = args.h
        z = abs(best["h"] - hr) / np.hypot(total["h"], shr)
        print(f"vs caliper h = {hr:.2f} ± {shr:.2f} cm: difference {best['h'] - hr:+.2f} cm, z = {z:.1f}")
    print("z < 2 consistent, 2-3 tension, > 3 find the cause")

    # ---------------- figure ----------------
    base = os.path.splitext(args.logs[0])[0] + "_prisma"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5))
    a1.scatter(raw[:, 0], raw[:, 1], s=6, alpha=0.5, c="#4c78a8", label="points (raw, on the beam axis)")
    v = shape.vertices(p[:k0], 0.0)
    v = np.vstack([v, v[:1]])
    a1.plot(v[:, 0], v[:, 1], "r-", lw=2, label="fit: " + ", ".join(
        f"{n} = {best['size'][i]:.2f} ± {total[n]:.2f}" for i, n in enumerate(shape.names)) + " cm")
    if real:
        rs = [x[0] for x in real]
        if shape.box and len(rs) == 2:
            big_first = p[0] >= p[1]
            vr = shape.vertices(rs if big_first else rs[::-1], 0.0)
        else:
            vr = shape.vertices([rs[0]], 0.0)
        vr = np.vstack([vr, vr[:1]])
        a1.plot(vr[:, 0], vr[:, 1], "k--", lw=1.5, label="caliper")
    a1.plot(0, 0, "k+", ms=10)
    a1.set_aspect("equal")
    a1.set_xlabel("x along the prism (cm)"); a1.set_ylabel("y across the prism (cm)")
    a1.set_title("Top view, in the prism's own axes"); a1.legend(fontsize=8, loc="lower left")
    off = 0
    for s in scans:
        n = len(s["d"])
        a2.scatter(s["d"][:, 1], res[off:off + n], s=8, alpha=0.6, label=s["name"])
        off += n
    a2.axhline(0, color="k", lw=1)
    a2.axhspan(-ROBUST_SCALE, ROBUST_SCALE, color="gray", alpha=0.2)
    a2.set_xlabel("plate angle (deg)"); a2.set_ylabel("measured - model (cm)")
    a2.set_title("Residuals: no trend expected"); a2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(base + ".png", dpi=130)
    print(f"\nFigure: {base}.png")


if __name__ == "__main__":
    main()
