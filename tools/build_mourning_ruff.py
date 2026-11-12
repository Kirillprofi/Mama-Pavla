"""Ashen Mourning Ruff: Neck accessory (NeckAttachment), fitted to the R6 avatar.

An Elizabethan millstone ruff reworked as visual-kei mourning wear: black chiffon set in deep
figure-eight pleats with an oxblood binding, around a taller box-pleated frill of ivory lace. The left
shoulder is torn (shorter, frayed pleats, one hanging loose over the chest). Three black glass shards
in a tarnished silver bezel with pearl studs close it at the throat, and two uneven strands of pearls
and jet beads fall from the bezel onto the chest, each ending in a drop. One closed mesh, < 4k triangles.
"""
import math
import os
import sys

import bpy  # noqa: F401  (must precede mathutils)
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbxkit as K  # noqa: E402

NAME = 'AshenMourningRuff'
np.random.seed(37)                 # Asm draws per-part shading noise from the global generator
rng = np.random.default_rng(37)
X, Y, Z = np.eye(3)
TOP, CLR = 2.0, 0.014              # torso top, clearance kept above it
THK, EDGE = 0.012, 0.016           # lace / chiffon thickness, bound edge of the ruff
CH = K.CHUNK                       # UV island length: periodic prints repeat a whole number of times per island

# black ruff: a strip whose outer edge traces x = t - BETA sin 2t, z = AMP sin t (upper and lower tubes)
NP, SPP = 24, 15                   # pleat periods (48 tubes), samples per period
BETA, AMP, AIN = 1.85, 0.1, 0.45   # S-curve overhang (narrow tube mouths), pleat half height, gathered at the neck
RX, RY, RIN = 1.02, 0.90, 0.705    # outer half-extents, inner radius
GAPA = math.radians(11)            # half-opening at the throat, closed by the clasp
ZC = TOP + CLR + 1.12 * AMP + EDGE / 2   # mid height where it rests on the shoulders
DROOP, DY0 = 0.09, 0.46            # sag beyond the torso's front and back faces
TEAR = (math.radians(34), math.radians(118))   # torn sector on the left shoulder
LOOSE = math.radians(47)           # where the torn pleat hangs loose
# ivory lace ruffle: gathered at the neck, flaring over the ruff's inner edge
NF, FZ0, FZ1 = 16, 2.03, 2.29
VIEWS = [('preview', (0.1, -0.3, 2.0), (0.55, -1.0, 0.42), 4.4, 60),
         ('front', (0.0, -0.2, 1.85), (0.0, -1.0, 0.1), 4.8, 60),
         ('back', (0.0, 0.2, 2.05), (-0.5, 1.0, 0.5), 4.4, 60),
         ('closeup', (0.22, -0.62, 1.95), (0.35, -1.0, 0.3), 2.2, 60)]
XVIEWS = [('side', (-0.95, -0.2, 2.12), (-1.0, -0.4, 0.15), 1.7, 60)]    # iteration only (LOOK_VIEWS=side)


# ------------------------------------------------------------------------------------ materials
def _chiffon(nb):
    """Black crepe chiffon: crinkle streaks along the folds, soft sheen, sheer lighter zone inside an
    oxblood satin binding with stitches (v = studs from the bound edge), frayed where torn (vfac)."""
    co = nb.coord('Object')
    u, v, _ = nb.xyz(nb.uvmap())
    rnd = nb.attr('part_rnd')
    fray = nb.attr('vfac')
    tau = 2 * math.pi / CH
    warp = nb.noise(co, 7.0, 3, 0.5)
    crk = nb.add(nb.mul(nb.math('SINE', nb.add(nb.mul(u, tau * 23), nb.mul(warp, 9.0))), 0.6),
                 nb.mul(nb.math('SINE', nb.add(nb.mul(u, tau * 57), nb.mul(warp, 15.0))), 0.4))
    crk = nb.mapr(crk, -0.9, 0.9)
    sheen = nb.noise(co, 3.5, 3, 0.55)
    grain = nb.noise(co, 160.0, 2, 0.5)
    open_ = nb.mapr(nb.ao(0.06), 0.3, 0.95)
    wear = nb.mul(K.edge_mask(nb, 0.004, 0.04, 0.3), open_)
    sheer = nb.mapr(v, 0.13, 0.035, smooth=True)
    keep = nb.inv(nb.mapr(fray, 0.15, 0.5))
    bind = nb.mul(nb.mapr(v, 0.031, 0.026), keep)
    dash = nb.mapr(nb.absf(nb.sub(nb.frac(nb.mul(u, 40 / CH)), 0.5)), 0.3, 0.18)
    stitch = nb.mul(nb.mul(dash, nb.mapr(nb.absf(nb.sub(v, 0.037)), 0.0032, 0.0018)), keep)
    threads = nb.mapr(nb.math('SINE', nb.add(nb.mul(u, tau * 71), nb.mul(warp, 23.0))), 0.1, 0.9)
    frayed = nb.mul(nb.mul(nb.mapr(fray, 0.2, 0.7), nb.mapr(v, 0.1, 0.0)), threads)
    c = nb.mixc(nb.mapr(nb.add(sheen, nb.mul(rnd, 0.3)), 0.35, 0.85), (0.0035, 0.0033, 0.004), (0.011, 0.01, 0.0125))
    c = nb.mixc(nb.mul(crk, 0.35), c, (0.02, 0.018, 0.022))
    c = nb.mixc(nb.mul(sheer, 0.5), c, (0.026, 0.021, 0.028))
    c = nb.mixc(nb.mul(wear, 0.5), c, (0.04, 0.037, 0.04))
    bc = nb.mixc(nb.mapr(nb.add(sheen, nb.mul(crk, 0.2)), 0.3, 0.85), (0.042, 0.005, 0.007), (0.1, 0.012, 0.016))
    c = nb.mixc(bind, c, bc)
    c = nb.mixc(nb.mul(stitch, 0.9), c, (0.14, 0.03, 0.03))
    c = nb.mixc(nb.mul(frayed, 0.85), c, (0.095, 0.084, 0.072))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.85), c, (0.0015, 0.0015, 0.002))
    rough = nb.add(nb.sub(nb.mapr(nb.add(sheen, nb.mul(crk, 0.3)), 0.3, 0.9, 0.52, 0.8), nb.mul(bind, 0.2)),
                   nb.mul(frayed, 0.15))
    h = nb.add(nb.add(nb.mul(crk, 0.35), nb.mul(grain, 0.25)), nb.add(nb.mul(bind, 0.5), nb.mul(stitch, 0.4)))
    return nb.finish(c, rough, 0.0, nb.bump(nb.add(h, nb.mul(frayed, 0.3)), 0.3, 0.002))


def _lace(nb):
    """Aged ivory lace: scalloped cut edge with a corded outline, an eyelet row, then staggered thread
    rings over a fine net (perforations show black); v = studs down from the top edge."""
    co = nb.coord('Object')
    u, v, _ = nb.xyz(nb.uvmap())
    rnd = nb.attr('part_rnd')

    def cell(x, pitch, off=0.0):
        return nb.mul(nb.sub(nb.frac(nb.add(nb.mul(x, 1 / pitch), off)), 0.5), pitch)

    def dist(a, b):
        return nb.math('SQRT', nb.add(nb.mul(a, a), nb.mul(b, b)))

    PS = CH / 10
    su = cell(u, PS)
    bump = nb.math('SQRT', nb.maxf(nb.sub(1.0, nb.mul(nb.mul(su, su), 4 / PS ** 2)), 0.0))
    edge = nb.sub(0.034, nb.mul(bump, 0.026))
    cut = nb.mapr(nb.sub(edge, v), -0.0015, 0.0015)
    cord = nb.mapr(nb.absf(nb.sub(v, nb.add(edge, 0.005))), 0.0035, 0.002)
    eyelet = nb.maxf(nb.mapr(dist(su, nb.sub(v, 0.056)), 0.0095, 0.0072),
                     nb.mapr(dist(cell(u, PS, 0.5), nb.sub(v, 0.05)), 0.0042, 0.003))
    body = nb.mul(nb.mapr(v, 0.074, 0.082), nb.mapr(v, 0.27, 0.25))
    PU, PV = CH / 7, 0.064
    d1 = dist(cell(u, PU), cell(nb.sub(v, 0.11), PV))
    d2 = dist(cell(u, PU, 0.5), cell(nb.sub(v, 0.11), PV, 0.5))
    ring = nb.maxf(nb.mapr(nb.absf(nb.sub(d1, 0.022)), 0.0042, 0.0024), nb.mapr(nb.absf(nb.sub(d2, 0.013)), 0.0036, 0.002))
    core = nb.maxf(nb.mapr(d1, 0.0105, 0.008), nb.mapr(d2, 0.0045, 0.003))
    PN = CH / 30
    net = nb.mapr(dist(cell(u, PN), cell(v, PN)), 0.0068, 0.005)
    solid_motif = nb.maxf(ring, nb.mapr(nb.minf(d1, d2), 0.03, 0.027))
    holes = nb.maxf(nb.maxf(cut, eyelet), nb.mul(body, nb.maxf(core, nb.mul(net, nb.inv(solid_motif)))))
    n1 = nb.noise(co, 30.0, 4, 0.6)
    open_ = nb.mapr(nb.ao(0.05), 0.3, 0.95)
    c = nb.mixc(nb.mapr(nb.add(n1, nb.mul(rnd, 0.3)), 0.3, 0.8), (0.47, 0.4, 0.29), (0.62, 0.56, 0.43))
    c = nb.mixc(nb.mul(nb.maxf(cord, ring), 0.6), c, (0.74, 0.68, 0.55))
    c = nb.mixc(nb.mul(nb.mapr(v, 0.2, 0.3), 0.55), c, (0.3, 0.24, 0.16))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.75), c, (0.07, 0.055, 0.04))
    c = nb.mixc(nb.mul(holes, 0.95), c, (0.012, 0.01, 0.011))
    h = nb.sub(nb.add(nb.mul(nb.maxf(cord, ring), 0.5), nb.mul(n1, 0.2)), nb.mul(holes, 0.7))
    return nb.finish(c, nb.add(nb.mapr(n1, 0.3, 0.7, 0.6, 0.8), nb.mul(holes, 0.15)), 0.0, nb.bump(h, 0.35, 0.002))


def _pearl(nb):
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    n1 = nb.noise(co, 90.0, 3, 0.5)
    ov = nb.noise(co, 22.0, 2, 0.5)
    c = nb.mixc(rnd, (0.66, 0.6, 0.49), (0.6, 0.585, 0.56))
    c = nb.mixc(nb.mul(nb.mapr(ov, 0.5, 0.72), 0.4), c, (0.66, 0.5, 0.52))
    c = nb.mixc(nb.mul(nb.mapr(ov, 0.45, 0.25), 0.3), c, (0.5, 0.6, 0.55))
    c = nb.mixc(nb.mul(nb.mapr(n1, 0.66, 0.8), 0.45), c, (0.3, 0.26, 0.2))
    c = nb.mixc(nb.mul(nb.inv(nb.mapr(nb.ao(0.02), 0.35, 0.95)), 0.7), c, (0.08, 0.07, 0.06))
    return nb.finish(c, nb.mapr(n1, 0.3, 0.7, 0.16, 0.3), 0.0, nb.bump(n1, 0.1, 0.001))


def _jet(nb):
    co = nb.coord('Object')
    n1 = nb.noise(co, 50.0, 3, 0.5)
    c = nb.mixc(nb.mapr(n1, 0.35, 0.7), (0.004, 0.004, 0.005), (0.013, 0.012, 0.014))
    c = nb.mixc(nb.mul(K.edge_mask(nb, 0.002, 0.05, 0.3), 0.6), c, (0.07, 0.068, 0.072))
    return nb.finish(c, nb.mapr(n1, 0.3, 0.7, 0.07, 0.16), 0.0, nb.bump(n1, 0.1, 0.001))


def _glass(nb):
    """Black glass shards: smoky oxblood depth, hairline fractures, chipped grey edges."""
    co = nb.coord('Object')
    smoke = nb.noise(co, 14.0, 4, 0.6, distortion=0.8)
    lines = nb.mul(nb.mapr(nb.voronoi(co, 40.0, 'DISTANCE_TO_EDGE', 'Distance'), 0.02, 0.0),
                   nb.mapr(nb.noise(co, 9.0, 2, 0.5), 0.45, 0.6))
    edge = K.edge_mask(nb, 0.002, 0.05, 0.3)
    c = nb.mixc(nb.mapr(smoke, 0.35, 0.75), (0.004, 0.003, 0.0035), (0.034, 0.006, 0.01))
    c = nb.mixc(nb.mul(lines, 0.7), c, (0.075, 0.07, 0.075))
    c = nb.mixc(nb.mul(edge, 0.7), c, (0.14, 0.135, 0.14))
    rough = nb.add(nb.mapr(smoke, 0.3, 0.7, 0.03, 0.1), nb.mul(nb.add(lines, edge), 0.3))
    return nb.finish(c, rough, 0.0, nb.bump(nb.mul(lines, -1.0), 0.2, 0.001))


def hammered(nb, c, rough, met, h):
    """metal_recipe `extra`: hammer dents, their rims polished, black oxide left in the hollows."""
    cell = nb.voronoi(nb.coord('Object'), 140.0, 'F1', 'Distance')
    dent = nb.mapr(cell, 0.05, 0.45)
    c = nb.mixc(nb.mul(nb.inv(dent), 0.55), c, (0.012, 0.011, 0.01))
    return c, nb.add(rough, nb.mul(nb.inv(dent), 0.15)), met, nb.add(h, nb.mul(dent, 0.6))


def _thread(nb):
    co = nb.coord('Object')
    tw = nb.noise(co, 300.0, 2, 0.5)
    c = nb.mixc(nb.mapr(tw, 0.3, 0.7), (0.006, 0.005, 0.006), (0.03, 0.026, 0.028))
    return nb.finish(c, 0.55, 0.0, nb.bump(tw, 0.2, 0.001))


K.RECIPES.update({'chiffon': _chiffon, 'lace': _lace, 'pearl': _pearl, 'jet': _jet, 'glass': _glass,
                  'thread': _thread,
                  'silver_old': lambda nb: K.metal_recipe(nb, (0.62, 0.62, 0.63), (0.14, 0.128, 0.115),
                                                          (0.008, 0.007, 0.006), (0.9, 0.9, 0.92), 0.16, 0.55, 1.0),
                  'silver_plate': lambda nb: K.metal_recipe(nb, (0.34, 0.335, 0.33), (0.045, 0.04, 0.036),
                                                            (0.006, 0.005, 0.004), (0.75, 0.75, 0.77), 0.3, 0.75,
                                                            1.0, bump=0.5, sc=4.0, cav=0.025, extra=hammered)})


# -------------------------------------------------------------------------------------- helpers
def at(r, th, z):
    """(radius, angle from the front towards +X, height) -> xyz; broadcasts."""
    r, th, z = np.broadcast_arrays(np.asarray(r, float), np.asarray(th, float), np.asarray(z, float))
    return np.stack([r * np.sin(th), -r * np.cos(th), z], -1)


def smooth(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def sag(y):
    s = np.clip((np.abs(y) - DY0) / (RY - DY0), 0.0, 1.0)
    return DROOP * s * s


def droop(P):
    """The starched ruff sags where nothing holds it up: beyond the torso's front and back faces."""
    P = np.array(P, float)
    P[..., 2] -= sag(P[..., 1])
    return P


def strip(P, U, V, thk, loop=None, vfac=None):
    """Closed shell of the quad grid P[i, j] (i along, j across; closed in i when `loop` gives its U length).
    Row j is thk[j] thick along the surface normal; 0 makes a knife edge, which saves the rim triangles
    where nobody looks. Thick edge rows get a rim, open strips end caps. Faces are UV'd by U[i] and V[i, j]
    (studs) and cut into islands every CH of U. Returns the primitive and per-vertex vfac."""
    n, m = P.shape[:2]
    ti = np.roll(P, -1, 0) - np.roll(P, 1, 0) if loop else np.gradient(P, axis=0)
    N = K._unit(np.cross(ti, np.gradient(P, axis=1)))
    fi, bi, verts = np.zeros((n, m), int), np.zeros((n, m), int), []
    for i in range(n):
        for j in range(m):
            fi[i, j] = bi[i, j] = len(verts)
            verts.append(P[i, j] + N[i, j] * thk[j] / 2)
            if thk[j] > 0:
                bi[i, j] = len(verts)
                verts.append(P[i, j] - N[i, j] * thk[j] / 2)
    faces, uvs = [], []

    def face(idx, us, vs, sgn=1.0):
        us = np.asarray(us, float) * sgn
        us -= math.floor(us.mean() / CH) * CH
        faces.append([int(x) for x in idx])
        uvs.append([(float(a), float(b)) for a, b in zip(us, vs)])

    for i in range(n if loop else n - 1):
        i1 = (i + 1) % n
        u0, u1 = U[i], (U[i + 1] if i + 1 < n else loop)
        for j in range(m - 1):
            vs = (V[i, j], V[i1, j], V[i1, j + 1], V[i, j + 1])
            face((fi[i, j], fi[i1, j], fi[i1, j + 1], fi[i, j + 1]), (u0, u1, u1, u0), vs)
            face((bi[i, j], bi[i, j + 1], bi[i1, j + 1], bi[i1, j]), (u0, u0, u1, u1), (vs[0], vs[3], vs[2], vs[1]), -1)
        for j, q in ((m - 1, (fi[i, m - 1], bi[i, m - 1], bi[i1, m - 1], fi[i1, m - 1])),
                     (0, (fi[i1, 0], bi[i1, 0], bi[i, 0], fi[i, 0]))):
            if thk[j] > 0:
                face(q, (u0, u0, u1, u1) if j else (u1, u1, u0, u0), (0.0, thk[j], thk[j], 0.0))
    if not loop:
        for i in (0, n - 1):
            ring = [(fi[i, j], thk[j] / 2) for j in range(m)] + [(bi[i, j], -thk[j] / 2) for j in range(m - 1, -1, -1)
                                                                  if bi[i, j] != fi[i, j]]
            ring = ring if i else ring[::-1]
            faces.append([int(a) for a, _ in ring])
            vv = {fi[i, j]: V[i, j] for j in range(m)} | {bi[i, j]: V[i, j] for j in range(m)}
            uvs.append([(float(vv[a]), float(h)) for a, h in ring])
    vf = None
    if vfac is not None:
        vf = np.zeros(len(verts))
        vf[fi.ravel()], vf[bi.ravel()] = np.ravel(vfac), np.ravel(vfac)
    return (np.array(verts), faces, uvs), vf


def rows_uv(P):
    """U: mean arc length along the rows; V: distance from the last row (the bound / top edge)."""
    U = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=-1).mean(1))]
    V = np.linalg.norm(P - P[:, -1:], axis=-1)
    return U, V


# ------------------------------------------------------------------------------------- geometry
def head_radius():
    """Largest horizontal radius of the head per height, from ray casts all round it (its vertices are
    too sparse for polar_profile near the rounded chin)."""
    ob = K.load_mannequin(['Head'])[0]
    tree = BVHTree.FromObject(ob, bpy.context.evaluated_depsgraph_get())
    zs = np.linspace(1.92, 2.5, 30)
    rr = [max(Vector(tree.ray_cast(Vector((2 * math.sin(a), -2 * math.cos(a), z)),
                                   Vector((-math.sin(a), math.cos(a), 0.0)))[0]).xy.length
              for a in np.linspace(0, 2 * math.pi, 48, endpoint=False)) for z in zs]
    bpy.data.objects.remove(ob)
    return lambda z: np.interp(z, zs, rr)


K.use_rig('R6')
K.ensure_mannequin()
K.new_scene()
HR = head_radius()
A = K.Asm()


def xs(t):
    return t - BETA * np.sin(2 * t)


def ruff():
    """Black chiffon strip from the throat round the neck and back to the throat, set in figure-eight
    pleats: gathered tight at the neck (a knife edge), bound and deepest at the outer edge, torn short and
    frayed on the left shoulder. Returns the torn edge radius and angle of the lobe that hangs loose."""
    t0 = 0.5 * math.acos(1 / (2 * BETA))            # minima of x(t): the strip ends at the extremes of its sweep
    t1 = 2 * math.pi * NP - t0
    k = (2 * math.pi - 2 * GAPA) / (xs(t1) - xs(t0))
    tt = np.linspace(t0, t1, 60001)
    px, pz = xs(tt) * k * RX, AMP * np.sin(tt)
    dx, dz = np.gradient(px), np.gradient(pz)
    w = np.hypot(dx, dz) + 0.05 * np.abs(np.gradient(np.unwrap(np.arctan2(dz, dx))))   # length + bending
    s = np.cumsum(w) - w[0]
    n = int(round((t1 - t0) / (2 * math.pi) * SPP)) + 1
    t = np.interp(np.linspace(0, s[-1], n), s, tt)
    th = GAPA + (xs(t) - xs(t0)) * k
    amp = AMP * (1 + 0.07 * np.sin(0.41 * t + 1.3) + 0.05 * np.sin(0.93 * t + 0.4))
    rout = (1 + 0.012 * np.sin(0.5 * t + 2.0)) / np.sqrt((np.sin(th) / RX) ** 2 + (np.cos(th) / RY) ** 2)
    lobe = np.floor((t - t0) / math.pi).astype(int)
    lobe_th = np.array([th[lobe == i].mean() for i in range(lobe.max() + 1)])
    env = smooth(TEAR[0], TEAR[0] + 0.25, th) * smooth(TEAR[1], TEAR[1] - 0.3, th)
    depth = rng.uniform(0.05, 0.15, len(lobe_th))
    loose = int(np.argmin(np.abs(lobe_th - LOOSE)))
    depth[loose] = 0.15
    jag = rng.uniform(-0.022, 0.022, n) + 0.016 * (-1.0) ** np.arange(n)
    tear = env * np.maximum(depth[lobe] + jag, 0.0)
    P = droop(np.stack([at(RIN, th, ZC + AIN * amp * np.sin(t)), at(rout - tear, th, ZC + amp * np.sin(t))], 1))
    U, V = rows_uv(P)
    prim, vf = strip(P, U, V, (0.0, EDGE), vfac=np.stack([np.zeros(n), np.clip(env * 1.5, 0, 1)], 1))
    A.part('ruff', prim, 'chiffon', smooth=80, vfac=vf)
    return float((rout - tear)[lobe == loose].min()), float(lobe_th[loose])


def frill():
    """Ivory lace ruffle standing inside the ruff: gathered flat at the neck (a knife edge hidden in the
    ruff), fluted and flaring towards the scalloped top edge, a little higher at the back."""
    n = NF * 5
    k = np.arange(n)
    th = (k / 5 - 0.8 + 0.12 * np.sin(k * 0.9) * (k % 5 > 0)) * 2 * math.pi / NF   # a trough at the front (clasp)
    fl = np.sin(2 * math.pi * k / 5) * np.repeat(rng.uniform(0.7, 1.25, NF), 5)
    jit = rng.uniform(-1, 1, n)
    back = (1 - np.cos(th)) / 2                    # flares less at the throat, where it dips into the ruff
    rt = 0.69 + 0.02 * back + 0.035 * (0.6 + 0.4 * back) * fl + 0.004 * jit
    zt = FZ1 - 0.04 * np.cos(th) + 0.01 * fl + 0.004 * jit           # dips at the throat, stands at the back
    P = droop(np.stack([at(0.62, th, FZ0), at(rt, th, zt)], 1))
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=-1).mean(1)
    loop = round(seg.sum() / CH) * CH
    U = np.r_[0.0, np.cumsum(seg[:-1])] * loop / seg.sum()    # whole islands round the loop: the print meets
    V = np.linalg.norm(P - P[:, -1:], axis=-1)
    prim, _ = strip(P, U, V, (0.0, THK), loop=loop)
    A.part('frill', prim, 'lace', smooth=80)
    q = P[:, :1] + (P[:, 1:] - P[:, :1]) * np.linspace(0, 1, 25)[:, None]      # the faces, not just the rows
    r = np.hypot(q[..., 0], q[..., 1])
    low = q[..., 2] < ZC + AIN * AMP * 1.12 + 0.01 - sag(RIN * q[..., 1] / np.maximum(r, 1e-9))
    print(f'frill clearance: head {np.min(r - HR(q[..., 2])) - THK / 2:.4f}, ruff {RIN - r[low].max() - THK / 2:.4f}')


def loose_pleat(r_edge, th_l):
    """The torn end of a pleat hanging out of its tube down over the left chest: a twisting strip, its
    bound edge outward, its torn edge and tip frayed."""
    x0 = r_edge * math.sin(th_l)
    y0 = min(-r_edge * math.cos(th_l), -0.552)
    z0 = ZC - 0.6 * AMP - sag(y0)
    ctrl = np.array([(x0 - 0.02, y0 + 0.01, z0 + 0.02), (x0, y0 - 0.006, z0 - 0.045), (x0 + 0.03, y0 - 0.018, z0 - 0.13),
                     (x0 + 0.008, y0 - 0.012, z0 - 0.21), (x0 + 0.035, y0 - 0.02, z0 - 0.28),
                     (x0 + 0.065, y0 - 0.028, z0 - 0.34)])
    c = K.spline(ctrl, 3)
    T = K._unit(np.gradient(c, axis=0))
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    f = s / s[-1]
    phi = 0.35 * np.sin(f * 5.0 + 0.4)
    W = K._unit(np.outer(np.cos(phi), X) - np.outer(np.sin(phi), Y))
    W = K._unit(W - T * np.sum(W * T, 1, keepdims=True))
    b = 0.105 - 0.035 * f
    tip = smooth(1.0, 0.8, f)
    P = np.stack([c - W * (b / 2 * (0.15 + 0.85 * tip))[:, None], c + W * (b / 2 * (0.35 + 0.65 * tip))[:, None]], 1)
    U, V = rows_uv(P)
    prim, vf = strip(P, U, V, (THK, THK), vfac=np.stack([np.ones(len(c)), 1 - tip], 1))
    A.part('loose', prim, 'chiffon', smooth=80, vfac=vf)


def shard(M, base, ang, length, width, height):
    """Convex faceted glass shard in the bezel plane (local XY of M), pointing along `ang`."""
    d = np.array([math.cos(ang), math.sin(ang)])
    sd = np.array([-d[1], d[0]])
    pts = [(*(base + d * length * f + sd * side * width / 2 * w * rng.uniform(0.75, 1.15)), 0.0)
           for f, w in ((-0.12, 0.55), (0.18, 1.0), (0.55, 0.7), (0.97, 0.12)) for side in (-1, 1)]
    for f, hh, o in ((0.12, 1.0, 0.18), (0.5, 0.7, -0.12), (0.85, 0.25, 0.05)):
        pts.append((*(base + d * length * f + sd * width * o), height * hh))
    pts = np.array(pts)
    hull = ConvexHull(pts)
    used = np.unique(hull.simplices)
    remap = np.full(len(pts), -1)
    remap[used] = np.arange(len(used))
    A.part('clasp', (K.placed((pts[used], []), M), [list(remap[s]) for s in hull.simplices]), 'glass', smooth=15)


BEZEL = [(-0.118, 0.03), (-0.072, 0.094), (0.004, 0.08), (0.062, 0.1), (0.122, 0.014), (0.088, -0.06),
         (0.018, -0.098), (-0.05, -0.07), (-0.104, -0.052)]


def clasp():
    """Three black glass shards in a tarnished silver bezel closing the ruff at the throat, pearl studs on
    the rim and two loops under it for the strands. Returns the loop bottoms."""
    o = np.array([0.0, -0.738, ZC - sag(0.74)])
    nrm = K._unit((0.0, -1.0, 0.1))
    up = np.cross(nrm, X)
    M = K.axes(o, X, up)
    out = np.array(K.fillet(BEZEL, 0.014, 2))
    inner = out[::2] * 0.55 + (0.0, -0.006)
    v2, tris = K.triangulate_region(out, interior=list(inner) + [(0.0, -0.01)])
    z = np.r_[np.full(len(out), 0.016), np.full(len(inner), 0.027), 0.031]       # domed front, flat back
    A.part('clasp', K.solid_from_surface(np.c_[v2, z], tris, np.c_[v2, np.zeros(len(v2))], v2), 'silver_plate', M,
           smooth=40)
    rim = [(x, y, 0.017) for x, y in out]
    A.part('clasp', K.tube(rim, 0.0085, sides=3, closed=True, up=(0, 0, 1)), 'silver_old', M, smooth=60)
    for ang, ln, wd, ht in ((1.2, 0.2, 0.06, 0.07), (2.95, 0.15, 0.052, 0.058), (-1.05, 0.11, 0.042, 0.046)):
        shard(M @ K.mat_trs((0, 0, 0.014)), np.array([0.0, -0.012]), ang, ln, wd, ht)
    for i in (2, 9, 13, 16):
        A.part('clasp', K.sphere(0.0125, 5, 3), 'pearl', M @ K.mat_trs(np.array(rim[i]) + (0, 0, 0.013)), smooth=180)
    for x, y in ((-0.075, 0.02), (0.07, -0.03), (0.03, 0.06)):
        q = np.array([x, y, 0.02])
        A.part('clasp', K.tube([q, q * (0.55, 0.55, 1.0) + (0, 0, 0.022)], [0.0045, 0.0], sides=3,
                               caps=('flat', 'point')), 'silver_old', M)
    tops = []
    for x in (0.042, -0.025):
        cr = K.xf(M, (x, -0.084 + 0.6 * abs(x - 0.018), 0.006)) - up * 0.006
        A.part('clasp', K.ring(0.012, 0.012, 0.0032, segs=5, sides=3), 'silver_old', K.frame(cr, -up, X))
        tops.append(cr - up * 0.012)
    return tops


def bead(p, t, kind, r):
    if kind == 'pearl':
        A.part('strands', K.sphere(r, 6, 4), 'pearl', K.mat_trs(p, (tuple(t), (0, -1, 0)), (1.0, 1.0, 0.92)),
               smooth=180)
    else:
        A.part('strands', K.lathe([(0, r * 1.05), (r, 0.0), (0, -r * 1.05)], segs=5), 'jet',
               K.mat_trs(p, (tuple(t), (0, -1, 0))), smooth=30)


def strand(top, ctrl, drop):
    """Knotted clusters of pearls and jet beads on a black thread hanging from a clasp loop, uneven gaps
    of bare thread between them; ends in a drop on a silver cap."""
    pts = K.spline(np.vstack([top, ctrl]), 2)
    path = K.Path(pts)
    A.part('strands', K.tube(pts, 0.0032, sides=3), 'thread', smooth=60)
    d, left = 0.018, 0
    while True:
        if left == 0:
            left = int(rng.integers(2, 4))
            d += rng.uniform(0.04, 0.1)
        kind = 'pearl' if rng.random() < 0.55 else 'jet'
        r = rng.uniform(0.02, 0.026) if kind == 'pearl' else rng.uniform(0.017, 0.021)
        if d + 2 * r > path.L - 0.012:
            break
        bead(path.at(d + r), path.tan(d + r), kind, r)
        d += 2 * r + rng.uniform(0.0, 0.004)
        left -= 1
    end, tn = pts[-1], K._unit(pts[-1] - pts[-2])
    m = K.mat_trs(end, (tuple(-tn), (0, -1, 0)))
    A.part('strands', K.lathe([(0, 0.004), (0.007, 0.0), (0.006, -0.008), (0, -0.01)], segs=6), 'silver_old', m,
           smooth=50)
    if drop == 'pearl':
        prof = [(0, -0.006), (0.009, -0.014), (0.016, -0.03), (0.019, -0.048), (0.014, -0.064), (0, -0.07)]
        A.part('strands', K.lathe(prof, segs=6), 'pearl', m, smooth=180)
    else:
        prof = [(0, -0.006), (0.015, -0.032), (0.011, -0.062), (0, -0.1)]
        A.part('strands', K.lathe(prof, segs=6), 'jet', m, smooth=25)


r_edge, th_l = ruff()
frill()
loose_pleat(r_edge, th_l)
top_a, top_b = clasp()
strand(top_a, [(0.04, -0.705, 1.93), (0.048, -0.61, 1.85), (0.056, -0.548, 1.75), (0.07, -0.537, 1.62),
               (0.08, -0.537, 1.5)], 'pearl')
strand(top_b, [(-0.04, -0.705, 1.93), (-0.055, -0.61, 1.85), (-0.095, -0.548, 1.73), (-0.17, -0.537, 1.56),
               (-0.235, -0.537, 1.38), (-0.25, -0.537, 1.22), (-0.2, -0.537, 1.1)], 'jet')
print(f'loose lobe r {r_edge:.3f} at {math.degrees(th_l):.1f} deg')

if os.environ.get('DBG'):
    hd = K.load_mannequin(['Head'])[0]
    tr = BVHTree.FromObject(hd, bpy.context.evaluated_depsgraph_get())
    for g, vv in zip(A.part_group, A.v):
        for p in vv:
            loc, nrm, fidx, dist = tr.find_nearest(Vector(p))
            if (Vector(p) - loc).dot(nrm) < 0:
                print('INSIDE', g, np.round(p, 4), 'nearest', np.round(loc, 4), 'n', np.round(nrm, 3), 'face', fidx)
    bpy.data.objects.remove(hd)
LV = os.environ.get('LOOK_VIEWS')
body = K.dev_modes(A, NAME, [v for v in VIEWS + XVIEWS if v[0] in LV.split(',')] if LV else VIEWS)
ob, rep = K.finish_item(A, NAME, 'Neck', 'Neck', 'Ashen Mourning Ruff', meta=body)
K.previews(NAME, VIEWS, spp=int(os.environ.get('PREVIEW_SPP', 40)))
print('DONE', NAME)
