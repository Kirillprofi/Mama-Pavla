"""Ashen Candlewax Top Hat: Hat accessory (HatAttachment), fitted to the R6 avatar.

The visual-kei aristocrat's mini top hat, being consumed by its own candles. A crooked oxblood
crushed-velvet crown sits tilted over the right side of the head; three black taper candles are fused
into its top (one sagging over from the heat), black wax pools over the rim and runs down the crown in
long beaded drips that follow gravity on the tilted hat, two of them across the curled brim and over
its edge, where they hang as droplets. A tarnished silver band engraved with line-and-dot sigils and set
with irregular studs is stabbed through by jet and ivory hatpins. One closed mesh, < 4k triangles.
"""
import math
import os
import sys

import bpy  # noqa: F401  (must precede mathutils)
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbxkit as K  # noqa: E402

NAME = 'AshenCandlewaxTopHat'
np.random.seed(37)                 # Asm draws per-part shading noise from the global generator
rng = np.random.default_rng(37)
Z = np.array([0.0, 0.0, 1.0])

# Hat-local frame: z runs up the crown axis from the brim top, phi = 0 at the front (-Y), +pi/2 at +X.
H = 0.45                                             # crown height
PROF = CubicSpline([0.0, 0.11, 0.24, 0.36, H], [0.186, 0.179, 0.183, 0.192, 0.197])   # crown radius by height
WALL_H = [0.29, 0.16, -0.008]                        # velvet rings below the wax lip (the band hides the foot)
OVAL, OVAL_B = 0.95, 0.955                           # crown / brim side-to-side squash
BEND = np.array([0.014, 0.016])                      # crooked crown: how far its top drifts (x, y)
BULGE = (0.012, 2.3, 0.75)                           # soft bulge: amplitude, angle, angular width
WAX_T, FIL = 0.01, 0.016                             # wax cap thickness, crown top edge radius
BAND_H, BAND_T = 0.075, 0.011                        # silver band height and thickness
R0, R1, ROLL = 0.205, 0.336, 0.012                   # brim: curl start, outer radius, rolled edge
RC = (R1 - ROLL, -0.009)                             # rolled edge centre in the brim profile
CURL_SIDE, CURL_FB = 0.062, 0.016                    # brim rise at the sides, dip front and back
TILT = (16.0, 5.0)                                   # degrees towards the character's right (-X) / forward
HAT_XY = (-0.18, -0.1)                              # crown base centre: right of and ahead of the head top
CLEAR = 0.018                                        # brim-to-head clearance
SEGS_C, SEGS_B = 28, 32
VIEWS = [('preview', (-0.2, 0.0, 3.3), (0.55, -1.0, 0.4), 2.7, 60),
         ('front', (-0.18, 0.0, 3.28), (0.0, -1.0, 0.1), 2.8, 60),
         ('side', (-0.2, 0.0, 3.3), (-1.0, -0.2, 0.15), 2.8, 60),
         ('closeup', (-0.3, -0.2, 3.3), (0.25, -1.0, 0.25), 1.35, 60)]

# crown drips: start angle, trail length, half-width, height, 'stop' on the crown or run over the 'brim'
DRIPS = [(-0.25, 0.66, 0.019, 0.013, 'brim'), (-0.9, 0.5, 0.015, 0.011, 'brim'),
         (0.4, 0.37, 0.016, 0.012, 'stop'), (0.72, 0.13, 0.011, 0.008, 'stop'),
         (1.2, 0.24, 0.013, 0.009, 'stop'), (1.75, 0.41, 0.02, 0.014, 'stop'),
         (2.55, 0.12, 0.012, 0.009, 'stop'), (3.0, 0.3, 0.016, 0.011, 'stop'),
         (-1.6, 0.075, 0.022, 0.015, 'stop'), (-2.2, 0.11, 0.017, 0.012, 'stop')]
# hatpins: entry angle, entry height, angle to the band tangent, slope, travel (+1 = towards +phi),
# length outside, head
PINS = [(0.7, 0.064, 28.0, -0.18, -1, 0.21, 'jet'), (-2.4, 0.066, 20.0, -0.3, 1, 0.19, 'ivory'),
        (2.1, 0.06, 24.0, -0.08, -1, 0.23, 'double')]


# ------------------------------------------------------------------------------------ materials
def _velvet(nb):
    """Oxblood crushed velvet: swirled crushed-sheen patches, pile texture brushed along the surface
    (SrcUV V runs down the crown and out over the brim), dark creases, soft sheen on the edges."""
    uv = nb.uvmap()
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    crush = nb.noise(nb.mapping(uv, scale=(1.0, 0.45, 1.0)), 9.0, 5, 0.65, distortion=3.0)
    patch = nb.noise(co, 5.0, 4, 0.55, distortion=1.2)
    pile = nb.noise(nb.mapping(uv, scale=(1.0, 0.3, 1.0)), 120.0, 3, 0.6)
    grain = nb.noise(co, 380.0, 2, 0.5)
    open_ = nb.mapr(nb.ao(0.035), 0.3, 0.95)
    sheen = nb.mapr(nb.add(nb.add(nb.mul(crush, 0.65), nb.mul(patch, 0.5)), nb.mul(rnd, 0.08)), 0.55, 0.9,
                    smooth=True)
    crease = nb.mapr(nb.absf(nb.sub(crush, 0.5)), 0.022, 0.0)
    c = nb.mixc(sheen, (0.038, 0.0032, 0.0065), (0.15, 0.026, 0.032))
    c = nb.mixc(nb.mul(nb.mapr(pile, 0.4, 0.75), 0.35), c, (0.075, 0.009, 0.013))
    c = nb.mixc(nb.mul(crease, 0.6), c, (0.012, 0.001, 0.002))
    c = nb.mixc(0.3, c, nb.mapr(grain, 0.25, 0.75, 0.7, 1.0), 'MULTIPLY')
    edge = nb.mul(K.edge_mask(nb, 0.01, 0.03, 0.3), open_)
    c = nb.mixc(nb.mul(edge, 0.45), c, (0.24, 0.06, 0.065))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.9), c, (0.005, 0.0008, 0.0012))
    rough = nb.mapr(sheen, 0.0, 1.0, 0.9, 0.74)
    h = nb.sub(nb.add(nb.mul(crush, 0.45), nb.add(nb.mul(grain, 0.25), nb.mul(pile, 0.2))), nb.mul(crease, 0.5))
    return nb.finish(c, rough, 0.0, nb.bump(h, 0.22, 0.003))


def _wax(nb):
    """Black wax: glossy fresh runs, dust settled on the upward faces, faintly warm at thin tips."""
    co = nb.coord('Object')
    _, _, nz = nb.xyz(nb.coord('Normal'))
    rnd = nb.attr('part_rnd')
    thin = nb.attr('vfac')
    flow = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 0.35)), 30.0, 4, 0.55, distortion=0.8)
    mott = nb.noise(co, 9.0, 4, 0.6)
    fine = nb.noise(co, 160.0, 3, 0.5)
    open_ = nb.mapr(nb.ao(0.025), 0.3, 0.95)
    dust = nb.mul(nb.mul(nb.mapr(nz, 0.3, 0.9), nb.mapr(nb.add(mott, nb.mul(fine, 0.35)), 0.5, 0.85)), open_)
    c = nb.mixc(nb.mapr(nb.add(flow, nb.mul(rnd, 0.4)), 0.3, 0.95), (0.004, 0.0037, 0.0037), (0.016, 0.0145, 0.014))
    c = nb.mixc(nb.mul(nb.mul(nb.mapr(thin, 0.7, 1.0), open_), 0.4), c, (0.05, 0.018, 0.008))
    c = nb.mixc(nb.mul(dust, 0.7), c, (0.07, 0.066, 0.062))
    edge = nb.mul(K.edge_mask(nb, 0.004, 0.04, 0.3), open_)
    c = nb.mixc(nb.mul(edge, 0.25), c, (0.045, 0.042, 0.04))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.7), c, (0.0015, 0.0015, 0.0015))
    rough = nb.add(nb.mapr(flow, 0.3, 0.7, 0.2, 0.34), nb.mul(dust, 0.45))
    h = nb.add(nb.mul(flow, 0.4), nb.mul(fine, 0.15))
    return nb.finish(c, rough, 0.0, nb.bump(h, 0.15, 0.002))


def _candle(nb):
    """Dipped black taper: fine layer rings along the candle (SrcUV V), satin sheen, faint streaks."""
    uv = nb.uvmap()
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    thin = nb.attr('vfac')
    _, v, _ = nb.xyz(uv)
    wob = nb.noise(co, 25.0, 2, 0.5)
    layer = nb.mapr(nb.absf(nb.sub(nb.frac(nb.add(nb.mul(v, 85.0), nb.mul(wob, 0.8))), 0.5)), 0.32, 0.5)
    streak = nb.noise(nb.mapping(uv, scale=(1.0, 0.1, 1.0)), 50.0, 3, 0.5)
    open_ = nb.mapr(nb.ao(0.02), 0.3, 0.95)
    c = nb.mixc(nb.mapr(nb.add(streak, nb.mul(rnd, 0.3)), 0.3, 0.85), (0.006, 0.0055, 0.006), (0.019, 0.017, 0.017))
    c = nb.mixc(nb.mul(layer, 0.45), c, (0.026, 0.023, 0.022))
    c = nb.mixc(nb.mul(nb.mul(thin, open_), 0.5), c, (0.05, 0.019, 0.009))
    edge = nb.mul(K.edge_mask(nb, 0.003, 0.04, 0.3), open_)
    c = nb.mixc(nb.mul(edge, 0.35), c, (0.06, 0.055, 0.052))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.75), c, (0.002, 0.002, 0.002))
    rough = nb.add(nb.mapr(streak, 0.3, 0.7, 0.3, 0.45), nb.mul(layer, 0.08))
    return nb.finish(c, rough, 0.0, nb.bump(nb.add(nb.mul(layer, 0.3), nb.mul(streak, 0.3)), 0.12, 0.002))


def _wick(nb):
    """Charred cotton wick: fibrous black, ashen towards the curled tip (vfac)."""
    co = nb.coord('Object')
    t = nb.attr('vfac')
    fib = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 4.0)), 300.0, 3, 0.6)
    c = nb.mixc(nb.mapr(fib, 0.3, 0.75), (0.008, 0.007, 0.0065), (0.03, 0.027, 0.025))
    c = nb.mixc(nb.mul(nb.mapr(t, 0.55, 1.0), nb.mapr(fib, 0.35, 0.7)), c, (0.13, 0.125, 0.118))
    return nb.finish(c, nb.mapr(fib, 0.3, 0.7, 0.78, 0.95), 0.0, nb.bump(fib, 0.4, 0.002))


def _jet(nb):
    co = nb.coord('Object')
    scr = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 9.0)), 140.0, 2, 0.5)
    c = nb.mixc(nb.mapr(scr, 0.62, 0.8), (0.0035, 0.0035, 0.0045), (0.02, 0.02, 0.025))
    c = nb.mixc(nb.mul(nb.inv(nb.mapr(nb.ao(0.01), 0.3, 0.95)), 0.8), c, (0.001, 0.001, 0.001))
    return nb.finish(c, nb.mapr(scr, 0.3, 0.8, 0.05, 0.18), 0.0)


def _ivory(nb):
    """Aged bone / ivory: warm grain, fine crazing, yellowed grime in the recesses."""
    co = nb.coord('Object')
    n1 = nb.noise(co, 50.0, 4, 0.55)
    grain = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 5.0)), 60.0, 3, 0.5)
    craze = nb.mapr(nb.voronoi(co, 110.0, 'DISTANCE_TO_EDGE', 'Distance'), 0.022, 0.0)
    open_ = nb.mapr(nb.ao(0.015), 0.3, 0.95)
    c = nb.mixc(nb.mapr(nb.add(n1, nb.mul(grain, 0.4)), 0.3, 0.85), (0.4, 0.32, 0.2), (0.62, 0.55, 0.4))
    c = nb.mixc(nb.mul(craze, 0.45), c, (0.16, 0.11, 0.06))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.85), c, (0.07, 0.045, 0.022))
    rough = nb.add(nb.mapr(n1, 0.3, 0.7, 0.35, 0.52), nb.mul(craze, 0.2))
    return nb.finish(c, rough, 0.0, nb.bump(nb.sub(nb.mul(n1, 0.3), craze), 0.25, 0.002))


BAND = {'len': 1.2, 'v0': 0.0, 'v1': 0.075}            # set when the band is built


def engraving(nb, c, rough, met, h):
    """Line-and-dot sigils in band space: border grooves, rope hatching, bars with dots between."""
    u, v, _ = nb.xyz(nb.uvmap())
    reps = max(8, round(BAND['len'] / 0.05))
    cu = nb.frac(nb.mul(u, reps / BAND['len']))
    vr = nb.mapr(v, BAND['v0'], BAND['v1'])
    dv = nb.absf(nb.sub(vr, 0.5))
    border = nb.mapr(nb.absf(nb.sub(dv, 0.38)), 0.026, 0.014)
    hatch = nb.mul(nb.mapr(nb.absf(nb.sub(nb.frac(nb.add(nb.mul(cu, 3.0), nb.mul(vr, 1.6))), 0.5)), 0.1, 0.05),
                   nb.mapr(nb.absf(nb.sub(dv, 0.29)), 0.035, 0.022))
    bar = nb.mul(nb.mapr(nb.absf(nb.sub(cu, 0.5)), 0.055, 0.03), nb.mapr(dv, 0.17, 0.14))
    dx = nb.sub(0.5, nb.absf(nb.sub(cu, 0.5)))
    dy = nb.mul(nb.sub(vr, 0.5), 0.75)
    dots = nb.mapr(nb.math('SQRT', nb.add(nb.mul(dx, dx), nb.mul(dy, dy))), 0.1, 0.07)
    groove = nb.maxf(nb.maxf(border, hatch), bar)
    c = nb.mixc(nb.mul(groove, 0.9), c, (0.006, 0.0055, 0.005))
    c = nb.mixc(nb.mul(dots, 0.7), c, (0.62, 0.62, 0.64))
    h = nb.add(nb.sub(h, nb.mul(groove, 0.9)), nb.mul(dots, 0.6))
    return c, nb.add(rough, nb.mul(groove, 0.2)), met, h


K.RECIPES.update({
    'velvet': _velvet, 'wax': _wax, 'candle': _candle, 'wick': _wick, 'jet': _jet, 'ivory': _ivory,
    'band_silver': lambda nb: K.metal_recipe(nb, (0.42, 0.42, 0.43), (0.12, 0.115, 0.108), (0.008, 0.0075, 0.007),
                                             (0.72, 0.72, 0.74), 0.22, 0.62, 1.0, extra=engraving),
    'aged_gold': lambda nb: K.metal_recipe(nb, (0.34, 0.22, 0.075), (0.09, 0.06, 0.024), (0.012, 0.009, 0.005),
                                           (0.62, 0.46, 0.2), 0.3, 0.7, 1.0)})


# -------------------------------------------------------------------------------------- helpers
def sstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def unit(v):
    return K._unit(v)


def revolve(rings):
    """Closed surface through `rings` (top to bottom; each an (n, 3) array with the same n, or one point
    for a pole), open ends capped. Per-corner UVs in studs, U around each ring and V down each meridian
    (like K.lathe). Returns the primitive and, per face, the index of its upper ring (-1 / len(rings) - 1
    for the top / bottom cap)."""
    rings = [np.atleast_2d(np.asarray(r, float)) for r in rings]
    segs = max(len(r) for r in rings)
    full = [np.repeat(r, segs, 0) if len(r) == 1 else r for r in rings]
    idx, n = [], 0
    for r in rings:
        idx.append(list(range(n, n + len(r))))
        n += len(r)
    verts = np.vstack(rings)
    vco = np.zeros((len(rings), segs + 1))
    for j in range(1, len(rings)):
        vco[j, :segs] = vco[j - 1, :segs] + np.linalg.norm(full[j] - full[j - 1], axis=1)
    vco[:, segs] = vco[:, 0]

    def ucoord(j):
        r = full[j] if len(rings[j]) > 1 else full[j + 1 if j + 1 < len(rings) else j - 1]
        u = np.r_[0.0, np.cumsum(np.linalg.norm(np.roll(r, -1, 0) - r, axis=1))]
        return u if len(rings[j]) > 1 else np.r_[(u[:-1] + u[1:]) / 2, 0.0]

    faces, uvs, band = [], [], []
    for j in range(len(rings) - 1):
        a, b, ua, ub, va, vb = idx[j], idx[j + 1], ucoord(j), ucoord(j + 1), vco[j], vco[j + 1]
        for k in range(segs):
            k1 = (k + 1) % segs
            if len(a) == 1:
                faces.append([a[0], b[k1], b[k]])
                uvs.append([(ua[k], va[k]), (ub[k + 1], vb[k + 1]), (ub[k], vb[k])])
            elif len(b) == 1:
                faces.append([a[k], a[k1], b[0]])
                uvs.append([(ua[k], va[k]), (ua[k + 1], va[k + 1]), (ub[k], vb[k])])
            else:
                faces.append([a[k], a[k1], b[k1], b[k]])
                uvs.append([(ua[k], va[k]), (ua[k + 1], va[k + 1]), (ub[k + 1], vb[k + 1]), (ub[k], vb[k])])
            band.append(j)
    for end, tag in ((0, -1), (len(rings) - 1, len(rings) - 1)):
        if len(rings[end]) > 1:
            face = idx[end][::-1] if end == 0 else idx[end]
            faces.append(face)
            uvs.append([(verts[i][0], verts[i][1]) for i in face])
            band.append(tag)
    return (verts, faces, uvs), np.array(band)


def subprim(prim, mask):
    """Faces of `prim` where mask holds (the weld drops the vertices no face uses)."""
    v, f, uv = prim
    return v, [x for x, m in zip(f, mask) if m], [u for u, m in zip(uv, mask) if m]


def speck_uv(prim, mask, s=0.03):
    """Shrink the UVs of hidden faces to specks so they take no texture space."""
    v, f, uv = prim
    return v, f, [[(a * s, b * s) for a, b in u] if m else u for u, m in zip(uv, mask)]


def resample(P, step):
    s = K.rbxgeo._arc(P)
    t = np.linspace(0.0, s[-1], max(2, int(round(s[-1] / step)) + 1))
    return np.array([[np.interp(x, s, P[:, i]) for i in range(P.shape[1])] for x in t])


# ------------------------------------------------------------------------------- crown and brim
def bulge(ph, h):
    a, c, w = BULGE
    return a * math.exp(-(math.remainder(ph - c, 2 * math.pi) / w) ** 2) * math.sin(math.pi * min(max(h / H, 0), 1))


def cyl(ph, rho, h):
    """Crown space (angle, radius, height) -> hat-local point, with the oval squash and the crooked bend."""
    t = (max(h, 0.0) / H) ** 2
    return np.array([rho * math.sin(ph) * OVAL + BEND[0] * t, -rho * math.cos(ph) + BEND[1] * t, h])


def crown_rad(ph, h):
    return float(PROF(min(max(h, 0.0), H))) + bulge(ph, h)


def band_lift(h):
    return BAND_T * sstep((BAND_H + 0.003 - h) / 0.009)


def support_pt(ph, h, off=0.0):
    """Crown wall (or the band's face where it covers the wall), offset outwards."""
    return cyl(ph, crown_rad(ph, h) + band_lift(h) + off, h)


def lip_drop(ph):
    d = 0.03 + 0.007 * math.sin(3 * ph + 0.7) + 0.003 * math.sin(5 * ph + 2.1)
    for p0, _, w0, *_ in DRIPS:
        d += (0.016 + 1.3 * w0) * math.exp(-(math.remainder(ph - p0, 2 * math.pi) / 0.15) ** 2)
    return d


def crown_rings():
    """Wax cap (pole, over the rounded edge, the lip bead) then the velvet wall down into the brim."""
    rings, cols = [[cyl(0.0, 0.0, H + WAX_T + 0.003)]], []
    for ph in 2 * math.pi * np.arange(SEGS_C) / SEGS_C:
        rt, d = crown_rad(ph, H), lip_drop(ph)
        h4 = H - FIL * 0.9
        h5 = min(H - d + 0.007, h4 - 0.006)
        hl = H - d
        cols.append([cyl(ph, rt - FIL, H + WAX_T + 0.0015 * math.sin(2 * ph + 1.0)),
                     cyl(ph, rt - FIL + (FIL + WAX_T) * 0.7071, H - FIL + (FIL + WAX_T) * 0.7071),
                     cyl(ph, crown_rad(ph, h4) + WAX_T, h4),
                     cyl(ph, crown_rad(ph, h5) + WAX_T * 1.3, h5),
                     cyl(ph, crown_rad(ph, hl - 0.005) + WAX_T * 0.55, hl - 0.005),
                     cyl(ph, crown_rad(ph, hl) - 0.0005, hl + 0.001)]
                    + [cyl(ph, crown_rad(ph, h), h) for h in WALL_H])
    cols = np.array(cols)
    return revolve(rings + [cols[:, j] for j in range(cols.shape[1])])


def curl_amp(ph):
    return CURL_SIDE * math.sin(ph) ** 2 - CURL_FB * math.cos(ph) ** 2 + 0.006 * math.sin(ph + 0.9)


def brim_pt(ph, r, z):
    """Brim profile point (r, z) at angle ph: the flat profile bent up along the curl (sides up, front and
    back down), so the rolled edge turns with the slope."""
    a, s = curl_amp(ph), (r - R0) / (R1 - R0)
    c, dc = (a * s ** 2.2, a * 2.2 * s ** 1.2 / (R1 - R0)) if s > 0 else (0.0, 0.0)
    k = math.hypot(1.0, dc)
    rr, zz = r - z * dc / k, c + z / k
    return np.array([rr * math.sin(ph) * OVAL_B, -rr * math.cos(ph), zz])


def roll_rz(a, extra=0.0):
    return RC[0] + (ROLL + extra) * math.cos(math.radians(a)), RC[1] + (ROLL + extra) * math.sin(math.radians(a))


BRIM = ([(0.0, 0.0), (0.2, 0.0), (0.245, 0.0), (0.285, 0.0005)] + [roll_rz(a) for a in (95, 30, -35, -100)]
        + [(0.285, -0.0165), (0.245, -0.016), (0.19, -0.0155), (0.0, -0.007)])


def brim_rings():
    phs = 2 * math.pi * np.arange(SEGS_B) / SEGS_B
    return revolve([[brim_pt(0.0, 0.0, z)] if r == 0 else [brim_pt(ph, r, z) for ph in phs] for r, z in BRIM])


# --------------------------------------------------------------------------------- fit on the head
def hat_matrix(cz):
    a, b = map(math.radians, TILT)
    rot = Matrix.Rotation(-a, 4, 'Y') @ Matrix.Rotation(b, 4, 'X')
    return np.array(Matrix.Translation((HAT_XY[0], HAT_XY[1], cz)) @ rot)


def head_bvh():
    ob = K.load_mannequin(['Head'])[0]
    tree = BVHTree.FromObject(ob, bpy.context.evaluated_depsgraph_get())
    bpy.data.objects.remove(ob)
    return tree


def clearance(W, tree):
    best = 1.0
    for v in W:
        loc, nrm, _, dist = tree.find_nearest(Vector(v), 0.3)
        if loc is not None:
            best = min(best, dist if (Vector(v) - loc).dot(nrm) >= 0 else -dist)
    return best


def fit_height(V, tree):
    """Lowest crown-base height that keeps the hat CLEAR off the head."""
    lo, hi = 3.0, 3.5
    for _ in range(22):
        cz = (lo + hi) / 2
        if clearance(K.placed((V,), hat_matrix(cz)), tree) < CLEAR:
            lo = cz
        else:
            hi = cz
    return hi


# ----------------------------------------------------------------------------------------- drips
def crown_frame(ph, h):
    e = 1e-4
    p = support_pt(ph, h)
    eph = (support_pt(ph + e, h) - support_pt(ph - e, h)) / (2 * e)
    eh = (support_pt(ph, h + e) - support_pt(ph, h - e)) / (2 * e)
    return p, eph, eh, unit(np.cross(eph, eh))


def brim_frame(ph, r):
    e = 1e-4
    p = brim_pt(ph, r, 0.0)
    er = (brim_pt(ph, r + e, 0.0) - brim_pt(ph, r - e, 0.0)) / (2 * e)
    eph = (brim_pt(ph + e, r, 0.0) - brim_pt(ph - e, r, 0.0)) / (2 * e)
    return p, er, eph, unit(np.cross(er, eph))


def flow_step(e1, e2, n, s, wob):
    """Parameter step (along e1, e2) for unit travel down the gravity slope, meandering a little."""
    g = G - n * (G @ n)
    if np.linalg.norm(g) < 1e-6:
        return None
    d = unit(g)
    w = wob[0] * math.sin(wob[1] * s + wob[2])
    d = math.cos(w) * d + math.sin(w) * np.cross(n, d)
    return d @ e1 / (e1 @ e1), d @ e2 / (e2 @ e2)


def trail(ph, length, kind, wob):
    """Surface points, normals and zone labels of a wax run from the lip at angle ph down the crown (and for
    'brim' runs on across the brim and round its rolled edge). Also returns the outward direction when the
    run leaves the edge at its lowest point, else None."""
    P, N, zone = [], [], []
    h, s, step = H - lip_drop(ph) + 0.012, 0.0, 0.006
    h_end = 0.016 if kind == 'brim' else 0.045
    while s < length:
        p, eph, eh, n = crown_frame(ph, h)
        P.append(p), N.append(n), zone.append('crown')
        if h <= h_end:
            break
        a, b = flow_step(eph, eh, n, s, wob)
        ph, h, s = ph + a * step, h + b * step, s + step
    if kind != 'brim' or s >= length:
        return np.array(P), np.array(N), zone, None
    r = crown_rad(ph, 0.0) + BAND_T + 0.004
    pc, _, _, nc = crown_frame(ph, 0.006)
    _, _, _, nb_ = brim_frame(ph, r + 0.006)
    P.append(pc + nc * 0.004 + Z * 0.004), N.append(unit(nc + nb_)), zone.append('corner')
    r += 0.006
    while r < R1 - ROLL - 0.004 and s < length:
        p, er, eph, n = brim_frame(ph, r)
        P.append(p), N.append(n), zone.append('brim')
        st = flow_step(er, eph, n, s, wob)
        if st is None or st[0] <= 0.15:           # the slope runs back to the crown: the wax pools here
            print(f'DRIP {ph:.2f}: pools on the brim at r={r:.3f}')
            return np.array(P), np.array(N), zone, None
        r, ph, s = r + st[0] * step, ph + st[1] * step, s + step
    best = None
    for a in (75, 35, 0, -40, -75):                # round the rolled edge to its lowest point
        rr, zz = roll_rz(a)
        p = brim_pt(ph, rr, zz)
        wz = (ROT @ p)[2]
        if best is not None and wz > best:
            break
        P.append(p), N.append(unit(brim_pt(ph, *roll_rz(a, 0.002)) - p)), zone.append('roll')
        best = wz
    return np.array(P), np.array(N), zone, unit(brim_pt(ph, R1 + 0.05, 0.0) - brim_pt(ph, R1, 0.0))


def wax_tube(group, P, N, zone, w0, t0, bead, hang_dir=None, hang_len=0.0, sides=4, step=0.026):
    """Beaded run: wide where it leaves the lip, thinning, wider and flatter where it spreads on the brim,
    ending in a fat rounded tip, or hanging off the brim edge as a teardrop."""
    cuts = sorted({0, len(P) - 1} | {i for i in range(1, len(P)) if zone[i] != zone[i - 1]})
    pts, nrm, zn = [], [], []
    for i0, i1 in zip(cuts[:-1], cuts[1:]):
        if zone[i0] in ('crown', 'brim') and i1 - i0 > 2:
            for q in resample(P[i0:i1 + 1], step)[:-1]:
                j = int(np.argmin(np.linalg.norm(P - q, axis=1)))
                pts.append(q), nrm.append(N[j]), zn.append(zone[j])
        else:
            for j in range(i0, i1):
                pts.append(P[j]), nrm.append(N[j]), zn.append(zone[j])
    pts.append(P[-1]), nrm.append(N[-1]), zn.append(zone[-1])
    pts, nrm = np.array(pts), np.array(nrm)
    s = K.rbxgeo._arc(pts)
    L = max(s[-1], 1e-6)
    kz = {'crown': (1.0, 1.0), 'corner': (1.4, 1.25), 'brim': (1.3, 0.75), 'roll': (0.95, 0.9)}
    W, T, vf = [], [], []
    for si, z in zip(s, zn):
        k = (1 + 0.8 * math.exp(-si / 0.02)) * (1 + bead[0] * math.sin(2 * math.pi * si / bead[1] + bead[2]))
        k *= 1 - 0.22 * si / L
        W.append(w0 * k * kz[z][0]), T.append(t0 * k * kz[z][1]), vf.append(0.1 + 0.5 * si / L)
    W[-1], T[-1] = W[-1] * 0.85, T[-1] * 0.85
    path, ups = [p + n * t * 0.3 for p, n, t in zip(pts, nrm, T)], list(nrm)
    d = unit(pts[-1] - pts[-2])
    if hang_dir is None:                             # fat rounded tip resting on the surface
        wt, tt = W[-1] * 1.7, T[-1] * 1.75
        c = pts[-1] + nrm[-1] * tt * 0.3 + d * wt * 0.85
        for a in (-50, 0, 40, 70, 90):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            path.append(c + d * wt * sa), ups.append(nrm[-1]), W.append(wt * ca), T.append(tt * ca)
            vf.append(0.8 + 0.2 * sa)
    else:                                            # teardrop hanging off the rolled edge
        p0, wd = path[-1], W[-1]
        tip = wd * 1.2
        for f, k in ((0.35, 0.68), (0.7, 0.56)):
            path.append(p0 + G * hang_len * f), ups.append(hang_dir), W.append(wd * k), T.append(wd * k)
            vf.append(0.85)
        c = p0 + G * (hang_len + tip * 0.35)
        for a in (-45, 0, 45, 75, 90):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            path.append(c + G * tip * sa), ups.append(hang_dir), W.append(tip * ca), T.append(tip * ca)
            vf.append(1.0)
    prim = K.tube(np.array(path), np.array(T), sides=sides, caps=('point', 'point'), ry=np.array(W),
                  ups=np.array(ups), chunk=10.0)
    A.part(group, prim, 'wax', M, smooth=180, vfac=np.r_[vf[0], np.repeat(vf[1:-1], sides), vf[-1]])
    return path


def stalactite(group, p, out, length, w):
    """Drop of wax hanging straight down from p (hat-local), a narrowing neck and a teardrop."""
    pts, rad = [p - G * 0.006, p + G * 0.004], [w, w * 1.05]
    for f, k in ((0.4, 0.75), (0.75, 0.6)):
        pts.append(p + G * length * f), rad.append(w * k)
    tip, c = w * 1.15, p + G * (length + w * 0.4)
    for a in (-45, 0, 45, 75, 90):
        pts.append(c + G * tip * math.sin(math.radians(a))), rad.append(tip * math.cos(math.radians(a)))
    vf = np.r_[0.3, np.repeat(np.linspace(0.4, 1.0, len(pts) - 2), 4), 1.0]
    A.part(group, K.tube(pts, rad, sides=4, caps=('point', 'point'), up=out, chunk=10.0), 'wax', M, vfac=vf)


# --------------------------------------------------------------------------------------- candles
def centreline(base, axis, length, sag=None, n=48):
    """Samples, points and frames of a candle axis; `sag` = (start, angle, direction, power) bends it over."""
    zs = np.linspace(0.0, length, n)
    if sag:
        z0, ang, dirn, pw = sag
        th = ang * np.clip((zs - z0) / (length - z0), 0.0, 1.0) ** pw
        dirs = np.outer(np.cos(th), axis) + np.outer(np.sin(th), unit(dirn - axis * (dirn @ axis)))
    else:
        dirs = np.tile(axis, (n, 1))
    steps = (dirs[1:] + dirs[:-1]) / 2 * np.diff(zs)[:, None]
    C = base + np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])
    T, Nn, B = K.frames(C, up=unit(np.cross(axis, (0.31, 0.95, 0.0))))
    return zs, C, T, Nn, B


def along(cl, x, y, z):
    """Candle-local (x, y, z along the axis) -> hat-local point."""
    zs, C, T, Nn, B = cl
    i = max(min(int(np.searchsorted(zs, z)) - 1, len(zs) - 2), 0)
    f = (z - zs[i]) / (zs[i + 1] - zs[i])
    c = C[0] + T[0] * z if z < 0 else C[i] + (C[i + 1] - C[i]) * f
    return c + unit(Nn[i] + (Nn[i + 1] - Nn[i]) * f) * x + unit(B[i] + (B[i + 1] - B[i]) * f) * y


def candle(group, base, axis, height, rad, segs=10, sag=None, crater=1.0, body=3):
    """Black taper fused at `base`: tapering body, melted crater with an uneven rim that is lowest where
    gravity spills the wax, a flared foot sunk into the pool. Returns its centreline and that spill angle."""
    cl = centreline(base, axis, height + 0.02, sag)
    zs, C, T, Nn, B = cl
    k = int(np.searchsorted(zs, height)) - 1
    g = G - T[k] * (G @ T[k])
    spill = math.atan2(g @ B[k], g @ Nn[k])
    prof = [(0.0, height - 0.017 * crater), (0.8 * rad, height - 0.005), (0.96 * rad, height + 0.003),
            (1.07 * rad, height - 0.008), (1.0 * rad, height - 0.026)]
    prof += [(rad * (1 + 0.12 * (1 - z / height)), z) for z in np.linspace(height - 0.055, 0.035, body)]
    prof += [(rad * 1.35, 0.01), (rad * 1.75, -0.003), (0.0, -0.014)]
    phase = rng.uniform(0, 2 * math.pi)
    rings = []
    for j, (r, z) in enumerate(prof):
        if r == 0:
            rings.append([along(cl, 0.0, 0.0, z)])
            continue
        ring = []
        for a in 2 * math.pi * np.arange(segs) / segs:
            notch = math.exp(-(math.remainder(a - spill, 2 * math.pi) / 0.55) ** 2)
            rr, zz = r, z
            if j in (1, 2, 3):
                zz += 0.005 * math.sin(3 * a + phase) + 0.003 * math.sin(5 * a + 2 * phase) - 0.015 * crater * notch
                rr *= 1 + (0.1 if j == 3 else 0.04) * notch
            ring.append(along(cl, rr * math.cos(a), rr * math.sin(a), zz))
        rings.append(ring)
    prim, _ = revolve(rings)
    vf = np.zeros(len(prim[0]))
    vf[1:1 + 3 * segs] = 0.55                         # thin, translucent crater rim
    A.part(group, prim, 'candle', M, smooth=70, vfac=vf)
    return cl, spill


def wick(group, cl, height, crater, curl_dir):
    top = along(cl, 0.0, 0.0, height - 0.017 * crater)
    zs, C, T, Nn, B = cl
    ax = T[int(np.searchsorted(zs, height)) - 1]
    sd = unit(curl_dir - ax * (curl_dir @ ax))
    pts = [top - ax * 0.006, top + ax * 0.012, top + ax * 0.024 + sd * 0.004, top + ax * 0.031 + sd * 0.012,
           top + ax * 0.03 + sd * 0.02, top + ax * 0.025 + sd * 0.026]
    vf = np.r_[np.repeat([0.0, 0.2, 0.45, 0.7, 0.9], 4), 1.0]
    A.part(group, K.tube(pts, [0.0034, 0.0032, 0.003, 0.0028, 0.0034, 0.0], sides=4, caps=('flat', 'point')),
           'wick', M, vfac=vf)


def candle_drip(group, cl, height, rad, a, length, w0, t0):
    """Run from the crater rim down the candle body."""
    P, N = [], []
    for z in np.linspace(height - 0.006, height - length, 9):
        aa = a + 0.12 * math.sin(z * 40.0)
        r = rad * (1 + 0.12 * (1 - z / height))
        p = along(cl, r * math.cos(aa), r * math.sin(aa), z)
        P.append(p), N.append(unit(p - along(cl, 0.0, 0.0, z)))
    wax_tube(group, np.array(P), np.array(N), ['crown'] * len(P), w0, t0, (0.15, 0.06, rng.uniform(0, 6)),
             step=0.022)


# --------------------------------------------------------------------------------------- hatpins
def hatpin(group, ph, h, ang, slope, sgn, lout, head):
    """Long pin stabbed into the band at (ph, h), `ang` off its tangent: it runs along a chord inside the
    crown, its point comes out a little on the far side and the bead head stands off the hat."""
    def face(a):
        return cyl(a, crown_rad(a, h) + BAND_T, h)
    e = 1e-4
    p = face(ph)
    t = unit(face(ph + e) - face(ph - e))
    n = unit(np.cross(t, Z))
    d = unit(math.cos(math.radians(ang)) * t * sgn - math.sin(math.radians(ang)) * n + slope * Z)
    q, inside = p.copy(), False
    for _ in range(400):
        q = q + d * 0.002
        phq = math.atan2(q[0] / OVAL, -q[1])
        if math.hypot(q[0] / OVAL, q[1]) < crown_rad(phq, q[2]) + band_lift(q[2]) - 0.004:
            inside = True
        elif inside:
            break
    if q[2] < 0.012:
        print(f'WARN pin at {ph:.2f} leaves the crown at h={q[2]:.3f}')
    head_c = p - d * lout
    A.part(group, K.tube([head_c, p, q + d * 0.034], [0.0045, 0.0045, 0.0], sides=4, caps=('flat', 'point')),
           'silver', M, smooth=60)
    look = (tuple(d), (0, 0, 1))
    if head == 'jet':
        bead = K.lathe([(0, 0.034), (0.024, 0.023), (0.034, 0.0), (0.024, -0.023), (0, -0.034)], segs=8)
        A.part(group, bead, 'jet', M @ K.mat_trs(head_c - d * 0.026, look), smooth=25)
    elif head == 'ivory':
        A.part(group, K.sphere(0.026, 8, 5), 'ivory', M @ K.mat_trs(head_c - d * 0.03, look, (1.0, 1.0, 1.35)))
    else:
        A.part(group, K.sphere(0.024, 8, 5), 'ivory', M @ K.mat_trs(head_c - d * 0.02, look))
        bead = K.lathe([(0, 0.02), (0.015, 0.013), (0.02, 0.0), (0.015, -0.013), (0, -0.02)], segs=6)
        A.part(group, bead, 'jet', M @ K.mat_trs(head_c - d * 0.062, look), smooth=25)
    A.part(group, K.lathe([(0, 0.004), (0.009, 0.002), (0.006, -0.007), (0, -0.008)], segs=6), 'aged_gold',
           M @ K.mat_trs(head_c + d * 0.004, (tuple(-d), (0, 0, 1))), smooth=60)


# ------------------------------------------------------------------------------------------ build
K.use_rig('R6')
K.ensure_mannequin()
K.new_scene()
A = K.Asm()
TREE = head_bvh()

crown, cband = crown_rings()
brim, bband = brim_rings()
CZ = fit_height(np.vstack([crown[0], brim[0]]), TREE)
M = hat_matrix(CZ)
ROT = M[:3, :3]
G = ROT.T @ np.array([0.0, 0.0, -1.0])               # gravity in the hat frame
print(f'FIT crown base z={CZ:.4f}  gravity(local)={np.round(G, 3)}')
if os.environ.get('DIAG'):
    for ph in np.linspace(-1.6, 1.0, 14):
        p, er, eph, n = brim_frame(ph, 0.29)
        g = G - n * (G @ n)
        lo = min((K.xf(M, brim_pt(ph, *roll_rz(a)))[2], a) for a in range(90, -100, -10))
        pw = K.xf(M, brim_pt(ph, *roll_rz(lo[1])))
        hit = TREE.ray_cast(Vector(pw), Vector((0, 0, -1)))
        print(f'DIAG ph={ph:+.2f} outflow={g @ unit(er):+.3f} edge_z={lo[0]:.3f} room={hit[3] if hit[0] else 9:.3f}')

N_CAP = 6                                             # wax bands of the crown revolve (pole fan .. lip underside)
wax_cap = (cband >= 0) & (cband < N_CAP)
A.part('crown', subprim(crown, wax_cap), 'wax', M, smooth=180)
vel = speck_uv(crown, cband == cband.max())           # bottom cap, buried in the brim
A.part('crown', subprim(vel, ~wax_cap), 'velvet', M, smooth=55)
A.part('brim', speck_uv(brim, bband <= 0), 'velvet', M, smooth=75)   # top centre fan, under the crown

# --- silver band with irregular studs
phs = 2 * math.pi * np.arange(28) / 28
path = [cyl(ph, crown_rad(ph, BAND_H / 2), BAND_H / 2) for ph in phs]
nrms = [unit(np.cross(cyl(ph + 1e-3, 0.2, 0) - cyl(ph - 1e-3, 0.2, 0), Z)) for ph in phs]
bprof = [(-0.005, -BAND_H / 2), (BAND_T, -BAND_H / 2), (BAND_T, BAND_H / 2), (-0.005, BAND_H / 2)]
BAND['len'] = float(K.rbxgeo._arc(np.vstack([path, path[:1]]))[-1])
BAND['v0'], BAND['v1'] = BAND_T + 0.005, BAND_T + 0.005 + BAND_H
A.part('band', K.sweep_profile(path, bprof, nrms, [Z] * len(phs), chunk=10.0), 'band_silver', M, smooth=40)

avoid = [p[0] for p in PINS] + [d[0] for d in DRIPS if d[4] == 'brim' or d[1] > 0.33]
for k in range(10):
    ph = 2 * math.pi * (k + 0.3) / 10 + rng.uniform(-0.1, 0.1)
    if min(abs(math.remainder(ph - q, 2 * math.pi)) for q in avoid) < 0.24:
        continue
    h = BAND_H / 2 + rng.uniform(-0.007, 0.007)
    p = cyl(ph, crown_rad(ph, h) + BAND_T - 0.002, h)
    n = unit(np.cross(cyl(ph + 1e-3, 0.2, 0) - cyl(ph - 1e-3, 0.2, 0), Z))
    sq = (rng.uniform(0.85, 1.2), rng.uniform(0.85, 1.2), rng.uniform(0.5, 0.75))
    A.part('studs', K.sphere(rng.uniform(0.009, 0.0125), 6, 3), 'silver',
           M @ K.mat_trs(p, (tuple(n), (rng.uniform(-0.3, 0.3), 0.0, 1.0)), sq), smooth=35)

# --- crown drips (they follow gravity on the tilted hat; two run over the brim edge)
for ph0, length, w0, t0, kind in DRIPS:
    wob = (rng.uniform(0.15, 0.35), rng.uniform(14, 24), rng.uniform(0, 6))
    P, N, zone, out = trail(ph0, length, kind, wob)
    bead = (rng.uniform(0.15, 0.3), rng.uniform(0.05, 0.09), rng.uniform(0, 6))
    hang = 0.0
    if out is not None:
        hit = TREE.ray_cast(Vector(K.xf(M, P[-1])), Vector((0, 0, -1)))
        room = (hit[3] if hit[0] is not None else 1.0) - CLEAR - 0.006
        hang = min(0.05, room - w0 * 1.9)
        print(f'DRIP {ph0:.2f}: over the edge, room {room:.3f}, hang {hang:.3f}')
        if hang < 0.012:
            out = None
    wax_tube('drips', P, N, zone, w0, t0, bead, hang_dir=out, hang_len=hang, sides=5 if w0 > 0.018 else 4)

# --- candles fused into the wax pool, the middle one sagging over the right-hand edge
top = H + WAX_T - 0.003
tall = candle('candles', cyl(0, 0, top) + np.array([0.03, 0.075, 0.0]), unit(Z + np.array([0.04, 0.05, 0.0])),
              0.34, 0.033)
bent = candle('candles', cyl(0, 0, top) + np.array([-0.095, -0.045, 0.0]), unit(Z + np.array([-0.06, 0.0, 0.0])),
              0.23, 0.03, sag=(0.03, math.radians(140), np.array([-0.85, -0.53, 0.0]), 1.15), body=10)
stub = candle('candles', cyl(0, 0, top) + np.array([0.095, -0.06, 0.0]), unit(Z + np.array([0.08, -0.06, 0.0])),
              0.11, 0.04, crater=1.4, body=2)
for cl, hgt, cr in ((tall[0], 0.34, 1.0), (bent[0], 0.23, 1.0), (stub[0], 0.11, 1.4)):
    wick('wicks', cl, hgt, cr, np.array([0.3, -0.9, 0.2]))
candle_drip('candle_drips', tall[0], 0.34, 0.033, tall[1], 0.15, 0.0095, 0.0068)
candle_drip('candle_drips', tall[0], 0.34, 0.033, tall[1] + 0.9, 0.08, 0.008, 0.0058)
candle_drip('candle_drips', stub[0], 0.11, 0.04, stub[1], 0.08, 0.012, 0.0085)
cl = bent[0]
rim = along(cl, 1.02 * 0.03 * math.cos(bent[1]), 1.02 * 0.03 * math.sin(bent[1]), 0.23 - 0.01)
stalactite('candle_drips', rim, unit(rim - along(cl, 0.0, 0.0, 0.22)), 0.05, 0.0085)
for a, b in ((tall, stub), (tall, bent)):            # wax fusing the candle feet together
    pa, pb = a[0][1][0], b[0][1][0]
    u = unit(pb - pa)
    mid = (pa + pb) / 2 + Z * 0.003
    A.part('candles', K.tube([pa + u * 0.035, mid, pb - u * 0.04], [0.007, 0.009, 0.007], sides=4, ry=[0.02, 0.026, 0.02],
                             caps=('point', 'point'), up=Z), 'wax', M, vfac=np.full(6, 0.2))

# --- hatpins stabbed through the band
for spec in PINS:
    hatpin('pins', *spec)

body = K.dev_modes(A, NAME, VIEWS)
ob, rep = K.finish_item(A, NAME, 'Hat', 'Hat', 'Ashen Candlewax Top Hat', meta=body)
K.previews(NAME, VIEWS, spp=int(os.environ.get('PREVIEW_SPP', 40)))
print('DONE', NAME)
