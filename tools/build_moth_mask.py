"""Ashen Moth Mask: Face accessory (FaceFrontAttachment), fitted to the R6 head.

A dusky moth spread across the eyes as a masquerade mask, drawn to the pack's candle flame: ash-grey
forewings sweep up and out from the nose bridge past the sides of the head, their hooked tips lifting
off the face; smaller oxblood-flushed hindwings curl down over the cheeks on slender tails. The furry
thorax and segmented abdomen run down the nose bridge as a raised ridge and two feathery (bipectinate)
antennae curl up over the forehead, one scrolled tighter than the other. The eye openings are cut
through the forewings around the mannequin's eyes. Torn notches and moth-eaten holes fray the wings;
abstract oxblood eye-spots of concentric irregular ovals are each set with a pearl, and a scatter of
tarnished silver studs follows one vein. Veins, scale rows and eye-spots are printed in each wing's
own front-view outline space. One closed mesh, < 4k triangles.
"""
import math
import os
import sys

import bpy  # noqa: F401  (must precede mathutils)
import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils import Vector
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, maximum_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbxkit as K  # noqa: E402
import rbxsdf as S  # noqa: E402

K.use_rig('R6')
NAME = 'AshenMothMask'
np.random.seed(97)                 # Asm draws per-part shading noise from the global generator
rng = np.random.default_rng(97)
Z = np.array([0.0, 0.0, 1.0])

# Front-view design space (x, z) as seen from the front, x > 0 the character's left; the right-hand parts
# are mirrored with deliberate differences. The mannequin's eyes sit at x 0.08-0.15, z 2.58-2.72.
FW_ROOT, HW_ROOT = (0.035, 2.645), (0.03, 2.52)    # vein origins of the fore/hind wing patterns
SPOT = (0.47, 2.785, 0.1, 0.068, 0.45)             # forewing eye-spot: centre, radii, tilt
TAIL_SPOT = (0.44, 2.36)                           # pale crescent where the hindwing turns into its tail
FW_VEINS, HW_VEINS = (-0.1, 0.085, 9, 0.15), (-1.0, 0.2, 7, 0.0)   # first angle, step, count, bend
STUD_VEIN = 7                                      # the vein above the eye-spot carries the studs
ROWS = 105.0                                       # wing scale rows per stud
BODY_Y = -0.664                                    # depth of the thorax axis
SEGS = [(2.552, 0.037, 0.03, 0.027), (2.511, 0.036, 0.029, 0.026), (2.471, 0.033, 0.027, 0.025),
        (2.433, 0.029, 0.024, 0.024), (2.397, 0.024, 0.02, 0.023), (2.363, 0.019, 0.016, 0.021),
        (2.331, 0.014, 0.012, 0.019)]              # abdomen segments: z, half-width, depth, half-length
GROOVES = [(a[0] + b[0]) / 2 for a, b in zip(SEGS, SEGS[1:])]
VIEWS = [('preview', (0.05, -0.4, 2.6), (0.62, -1.0, 0.22), 3.4, 60),
         ('front', (0.0, -0.4, 2.6), (0.0, -1.0, 0.05), 3.3, 60),
         ('side', (0.0, -0.35, 2.6), (1.0, -0.2, 0.06), 3.3, 60)]

FW = [  # left forewing: inner margin + termen (ragged towards the tip), costa over the brows, root
    ([(0.03, 2.55), (0.08, 2.515), (0.16, 2.5), (0.26, 2.512), (0.37, 2.548), (0.48, 2.603), (0.58, 2.668),
      (0.665, 2.738), (0.735, 2.818), (0.79, 2.903), (0.828, 2.985), (0.848, 3.04), (0.866, 3.093)], 0.026, 0.0045),
    ([(0.866, 3.093), (0.8, 3.08), (0.7, 3.052), (0.59, 3.015), (0.47, 2.976), (0.35, 2.942), (0.24, 2.914),
      (0.15, 2.893), (0.085, 2.874), (0.045, 2.847), (0.03, 2.79)], 0.03, 0.0),
    ([(0.03, 2.79), (0.025, 2.67), (0.03, 2.55)], 0.05, 0.0)]
HW = [  # left hindwing: costa (under the forewing), outer margin curling into a tail, inner margin, root
    ([(0.03, 2.585), (0.1, 2.545), (0.2, 2.537), (0.3, 2.558), (0.4, 2.598), (0.455, 2.625)], 0.05, 0.0),
    ([(0.455, 2.625), (0.49, 2.59), (0.507, 2.535), (0.503, 2.475), (0.488, 2.415), (0.462, 2.36), (0.432, 2.312),
      (0.416, 2.262), (0.402, 2.215), (0.383, 2.175), (0.357, 2.148), (0.332, 2.146), (0.322, 2.165),
      (0.336, 2.183), (0.352, 2.203), (0.357, 2.238), (0.352, 2.275)], 0.026, 0.0035),
    ([(0.352, 2.275), (0.29, 2.308), (0.22, 2.345), (0.15, 2.386), (0.09, 2.426), (0.045, 2.46), (0.03, 2.48)],
     0.034, 0.0),
    ([(0.03, 2.48), (0.026, 2.53), (0.03, 2.585)], 0.05, 0.0)]
ANT = [(0.012, 2.788), (0.022, 2.85), (0.045, 2.915), (0.085, 2.97), (0.14, 3.012), (0.2, 3.042), (0.265, 3.062),
       (0.33, 3.07)]                               # left antenna in front view, before its end scroll
EYE_HOLE = ((0.123, 2.652), 0.106, 0.06, 0.054, math.radians(60))   # centre, half-length, lid heights, tilt

# ------------------------------------------------------------------------------------ materials
SOOT, ASH_D, ASH, DUST = (0.006, 0.0055, 0.005), (0.022, 0.021, 0.02), (0.07, 0.066, 0.061), (0.17, 0.158, 0.14)
OX, OX_D, BONE, FRINGE = (0.12, 0.012, 0.014), (0.04, 0.004, 0.005), (0.38, 0.335, 0.26), (0.2, 0.185, 0.16)


def vec(nb, x, y, z=0.0):
    n = nb.node('ShaderNodeCombineXYZ')
    for s, val in zip(n.inputs, (x, y, z)):
        nb.feed(s, val)
    return n.outputs[0]


def hyp(nb, a, b):
    return nb.math('SQRT', nb.add(nb.mul(a, a), nb.mul(b, b)))


def band(nb, q, lo, hi, soft):
    """1 for lo <= q <= hi with soft edges."""
    return nb.mul(nb.mapr(q, lo - soft, lo), nb.mapr(q, hi + soft, hi))


def wing_fields(nb, root):
    """Pattern space of a wing: SrcUV holds the front-view (x, z) on its faces; polar coords about the
    vein origin. Rim faces carry unrolled UVs (v = thickness), flagged by `rim`."""
    u, v, _ = nb.xyz(nb.uvmap())
    au = nb.absf(u)
    dx, dz = nb.sub(au, root[0]), nb.sub(v, root[1])
    return {'u': u, 'v': v, 'au': au, 'r': hyp(nb, dx, dz), 'phi': nb.math('ARCTAN2', dz, dx),
            'uv': vec(nb, u, v), 'rim': nb.mapr(v, 1.2, 0.8)}


def veins(nb, f, spec, width=0.0032, r0=0.07):
    phi0, step, n, bend = spec
    t = nb.mul(nb.sub(nb.sub(f['phi'], nb.mul(f['r'], bend)), phi0), 1.0 / step)
    d = nb.mul(nb.absf(nb.sub(nb.frac(nb.add(t, 0.5)), 0.5)), nb.mul(f['r'], step))
    on = nb.mul(nb.mapr(t, -0.5, -0.3), nb.mapr(t, n - 0.7, n - 0.9))
    return nb.mul(nb.mul(nb.mapr(d, width, width * 0.4), on), nb.mapr(f['r'], r0, r0 + 0.08))


def wing_finish(nb, f, c, h, rough):
    """Shared wing surface: shingled scale rows broken into single scales, powder dust, a pale fringe on
    the cut edges and rims, crevice soot and a per-part tint."""
    rnd = nb.attr('part_rnd')
    row = nb.add(nb.mul(f['r'], ROWS), nb.mul(nb.noise(f['uv'], 25.0, 2, 0.5), 0.8))
    rowf = nb.frac(row)
    cell = nb.voronoi(vec(nb, row, nb.mul(nb.mul(f['phi'], f['r']), ROWS * 0.75)), 1.0, out='Color')
    shade = nb.mul(nb.mapr(rowf, 0.0, 1.0, 0.7, 1.12), nb.mapr(cell, 0.0, 1.0, 0.8, 1.16))
    c = nb.mixc(1.0, c, shade, 'MULTIPLY')
    c = nb.mixc(nb.mul(nb.mapr(nb.noise(f['uv'], 230.0, 2, 0.5), 0.6, 0.78), 0.3), c, DUST)
    open_ = nb.mapr(nb.ao(0.035), 0.35, 0.95)
    fringe = nb.maxf(K.edge_mask(nb, 0.009, 0.03, 0.3), f['rim'])
    streak = nb.mapr(nb.noise(f['uv'], 140.0, 2, 0.5), 0.3, 0.7, 0.45, 1.0)
    c = nb.mixc(nb.mul(nb.mul(fringe, streak), 0.85), c, FRINGE)
    c = nb.mixc(nb.mul(nb.inv(open_), 0.7), c, SOOT)
    c = nb.mixc(nb.mul(rnd, 0.12), c, ASH)
    h = nb.add(h, nb.add(nb.mul(rowf, 0.22), nb.mul(fringe, 0.2)))
    return nb.finish(c, nb.add(rough, nb.mul(nb.inv(open_), 0.05)), 0.0, nb.bump(h, 0.35, 0.002))


def _fore(nb):
    f = wing_fields(nb, FW_ROOT)
    au, v, r, phi = f['au'], f['v'], f['r'], f['phi']
    mott = nb.noise(nb.coord('Object'), 9.0, 4, 0.6)
    rr = nb.add(r, nb.mul(nb.sub(nb.noise(f['uv'], 6.0, 3, 0.55), 0.5), 0.08))
    c = nb.mixc(nb.mapr(rr, 0.06, 0.3), SOOT, ASH_D)                     # sooty root
    c = nb.mixc(nb.mapr(rr, 0.28, 0.5), c, ASH)                          # powdery ash mid-wing
    c = nb.mixc(nb.mul(nb.mapr(mott, 0.45, 0.75), 0.5), c, ASH_D)        # cloudy mottling
    c = nb.mixc(nb.mul(band(nb, rr, 0.64, 0.8, 0.06), 0.85), c, SOOT)    # dark submarginal band
    c = nb.mixc(nb.mul(nb.mapr(rr, 0.8, 0.95), 0.6), c, DUST)            # dusty outer margin
    for rb, amp, fr in ((0.2, 0.012, 15.0), (0.55, 0.016, 11.0)):         # zigzag cross lines, pale-edged
        d = nb.sub(nb.sub(r, rb), nb.mul(nb.math('SINE', nb.mul(phi, fr)), amp))
        c = nb.mixc(nb.mul(band(nb, d, -0.004, 0.003, 0.002), 0.85), c, SOOT)
        c = nb.mixc(nb.mul(band(nb, d, 0.004, 0.009, 0.002), 0.45), c, DUST)
    # eye-spot: concentric irregular ovals whose centres drift towards the root, hairlines broken
    ex, ez, a, b, tilt = SPOT
    ct, st = math.cos(tilt), math.sin(tilt)
    irr = nb.sub(nb.noise(f['uv'], 11.0, 3, 0.6), 0.5)                   # signed u: the two spots differ

    def ovals(dx, dz, wob):
        sx, sz = nb.sub(au, ex + dx), nb.sub(v, ez + dz)
        q = hyp(nb, nb.mul(nb.add(nb.mul(sx, ct), nb.mul(sz, st)), 1.0 / a),
                nb.mul(nb.sub(nb.mul(sz, ct), nb.mul(sx, st)), 1.0 / b))
        return nb.add(q, nb.mul(irr, nb.mul(q, wob)))
    q, qm, qi = ovals(0.0, 0.0, 0.32), ovals(-0.008, -0.004, 0.26), ovals(-0.014, -0.007, 0.2)
    broken = nb.mapr(nb.noise(f['uv'], 40.0, 2, 0.5), 0.36, 0.5)
    spot = band(nb, q, -1.0, 0.95, 0.03)
    c = nb.mixc(nb.mul(nb.mapr(q, 1.45, 0.95), 0.6), c, SOOT)           # smoky halo
    c = nb.mixc(spot, c, OX_D)
    c = nb.mixc(band(nb, q, 0.86, 0.92, 0.02), c, SOOT)
    c = nb.mixc(nb.mul(nb.mul(band(nb, q, 0.935, 0.96, 0.012), broken), 0.8), c, DUST)
    c = nb.mixc(band(nb, qm, 0.44, 0.66, 0.025), c, OX)
    bone = nb.mul(band(nb, qm, 0.69, 0.73, 0.014), broken)
    c = nb.mixc(bone, c, BONE)
    c = nb.mixc(band(nb, qi, -1.0, 0.4, 0.03), c, SOOT)
    growth = nb.mul(nb.math('SINE', nb.mul(qm, 70.0)), band(nb, qm, 0.44, 0.85, 0.02))
    c = nb.mixc(nb.mul(nb.mapr(growth, 0.3, 1.0), 0.3), c, OX_D)
    vm = nb.mul(veins(nb, f, FW_VEINS), nb.inv(nb.mul(spot, 0.85)))
    c = nb.mixc(nb.mul(vm, 0.75), c, SOOT)
    return wing_finish(nb, f, c, nb.add(nb.mul(vm, 0.6), nb.mul(bone, 0.3)), nb.sub(0.9, nb.mul(spot, 0.15)))


def _hind(nb):
    f = wing_fields(nb, HW_ROOT)
    au, v, r = f['au'], f['v'], f['r']
    rr = nb.add(r, nb.mul(nb.sub(nb.noise(f['uv'], 7.0, 3, 0.55), 0.5), 0.06))
    c = nb.mixc(nb.mapr(rr, 0.03, 0.14), SOOT, OX_D)
    c = nb.mixc(band(nb, rr, 0.12, 0.29, 0.05), c, OX)                   # oxblood flush
    c = nb.mixc(band(nb, rr, 0.31, 0.4, 0.025), c, SOOT)                 # black band
    c = nb.mixc(nb.mapr(rr, 0.4, 0.46), c, ASH_D)                        # ashen margin
    lx, lz = TAIL_SPOT                                                   # pale crescent above the tail
    q = hyp(nb, nb.mul(nb.sub(au, lx), 1 / 0.04), nb.mul(nb.sub(v, lz), 1 / 0.05))
    q = nb.add(q, nb.mul(nb.sub(nb.noise(f['uv'], 14.0, 2, 0.5), 0.5), 0.3))
    c = nb.mixc(nb.mul(nb.mul(band(nb, q, 0.55, 0.85, 0.06), nb.mapr(nb.sub(v, lz), -0.01, 0.02)), 0.8), c, BONE)
    vm = veins(nb, f, HW_VEINS, 0.003, 0.05)
    c = nb.mixc(nb.mul(vm, 0.7), c, SOOT)
    return wing_finish(nb, f, c, nb.mul(vm, 0.6), 0.88)


FUR_D, FUR_L, FUR_P = (0.009, 0.0075, 0.007), (0.07, 0.064, 0.058), (0.2, 0.185, 0.165)


def _fur(nb):
    co = nb.coord('Object')
    x, y, z = nb.xyz(co)
    rnd = nb.attr('part_rnd')
    hair = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 0.3)), 420.0, 3, 0.6)   # fine strands along the body
    tuft = nb.noise(co, 75.0, 4, 0.6, distortion=0.8)
    c = nb.mixc(nb.mapr(nb.add(nb.mul(hair, 0.7), nb.mul(tuft, 0.4)), 0.45, 0.95), FUR_D, FUR_L)
    c = nb.mixc(nb.mul(band(nb, z, 2.722, 2.752, 0.012), 0.55), c, OX_D)  # oxblood collar
    rings = None
    for zg in GROOVES:                                                      # pale rear margin per segment
        bnd = band(nb, nb.sub(z, zg), 0.002, 0.01, 0.003)
        rings = bnd if rings is None else nb.maxf(rings, bnd)
    c = nb.mixc(nb.mul(nb.mul(rings, nb.mapr(z, 2.58, 2.56)), 0.6), c, FUR_P)
    e = nb.math('SQRT', nb.add(nb.add(nb.mul(nb.sub(nb.absf(x), 0.024), nb.sub(nb.absf(x), 0.024)),
                                      nb.mul(nb.sub(y, BODY_Y - 0.014), nb.sub(y, BODY_Y - 0.014))),
                               nb.mul(nb.sub(z, 2.772), nb.sub(z, 2.772))))
    eye = nb.mapr(e, 0.0185, 0.0158)
    open_ = nb.mapr(nb.ao(0.03), 0.35, 0.95)
    c = nb.mixc(nb.mul(nb.inv(open_), 0.75), c, (0.002, 0.002, 0.002))
    c = nb.mixc(nb.mul(rnd, 0.1), c, FUR_L)
    c = nb.mixc(eye, c, (0.004, 0.002, 0.002))
    h = nb.mul(nb.add(nb.mul(hair, 0.55), nb.mul(tuft, 0.45)), nb.inv(eye))
    return nb.finish(c, nb.sub(0.92, nb.mul(eye, 0.7)), 0.0, nb.bump(h, 0.45, 0.003))


def _antenna(nb):
    u, _, _ = nb.xyz(nb.uvmap())
    co = nb.coord('Object')
    n1 = nb.noise(co, 90.0, 3, 0.5)
    ring = band(nb, nb.sub(nb.frac(nb.mul(u, 75.0)), 0.5), -0.12, 0.12, 0.08)   # annuli along the shaft
    c = nb.mixc(nb.mapr(n1, 0.3, 0.7), (0.006, 0.0045, 0.0035), (0.03, 0.021, 0.015))
    c = nb.mixc(nb.mul(ring, 0.5), c, (0.06, 0.045, 0.034))
    open_ = nb.mapr(nb.ao(0.02), 0.35, 0.95)
    c = nb.mixc(nb.mul(nb.inv(open_), 0.7), c, (0.002, 0.0015, 0.001))
    return nb.finish(c, nb.add(0.5, nb.mul(n1, 0.2)), 0.0, nb.bump(nb.add(nb.mul(ring, 0.5), nb.mul(n1, 0.3)),
                                                                   0.3, 0.002))


def _pearl(nb):
    co = nb.coord('Object')
    n1 = nb.noise(co, 55.0, 3, 0.5)
    orient = nb.noise(co, 14.0, 2, 0.5)
    c = nb.mixc(nb.mapr(orient, 0.35, 0.65), (0.56, 0.47, 0.41), (0.45, 0.47, 0.42))   # rosy to green orient
    c = nb.mixc(nb.mul(nb.mapr(n1, 0.5, 0.8), 0.6), c, (0.66, 0.6, 0.5))
    open_ = nb.mapr(nb.ao(0.012), 0.35, 0.95)
    c = nb.mixc(nb.mul(nb.inv(open_), 0.7), c, (0.07, 0.055, 0.045))
    return nb.finish(c, nb.mapr(n1, 0.3, 0.7, 0.16, 0.3), 0.0, nb.bump(n1, 0.08, 0.002))


K.RECIPES.update({
    'moth_fore': _fore, 'moth_hind': _hind, 'moth_fur': _fur, 'moth_antenna': _antenna, 'pearl': _pearl,
    'tarnish': lambda nb: K.metal_recipe(nb, (0.26, 0.26, 0.27), (0.045, 0.04, 0.036), (0.006, 0.005, 0.0045),
                                         (0.6, 0.6, 0.62), 0.3, 0.8, 1.0, bump=0.2, sc=3.0)})


# ------------------------------------------------------------------------------------- fitting
def head_map():
    """Cylindrical radius R(theta, z) of the mannequin head (ray cast), dilated and smoothed so surfaces
    offset from it clear the sculpted eyes and brows without copying their bumps."""
    head = K.load_mannequin(['Head'])[0]
    bvh = BVHTree.FromObject(head, bpy.context.evaluated_depsgraph_get())
    bpy.data.objects.remove(head)
    ths = np.radians(np.arange(-100.0, 100.01, 1.0))
    zs = np.arange(1.84, 3.1601, 0.01)
    R = np.full((len(ths), len(zs)), np.nan)
    for i, t in enumerate(ths):
        d = Vector((-math.sin(t), math.cos(t), 0.0))
        for j, z in enumerate(zs):
            hit = bvh.ray_cast(Vector((2.0 * math.sin(t), -2.0 * math.cos(t), z)), d, 3.0)[0]
            if hit is not None:
                R[i, j] = math.hypot(hit.x, hit.y)
    for i in range(len(ths)):              # above/below the head hold the last radius
        ok = ~np.isnan(R[i])
        R[i] = np.interp(zs, zs[ok], R[i][ok])
    R = gaussian_filter(maximum_filter(R, size=(7, 5)), (2.0, 1.5))
    return RegularGridInterpolator((ths, zs), R, bounds_error=False, fill_value=None)


def smooth(e0, e1, t):
    s = np.clip((np.asarray(t, float) - e0) / (e1 - e0), 0.0, 1.0)
    return s * s * (3 - 2 * s)


def rdist(root, ax, z):
    return np.hypot(np.asarray(ax) - root[0], np.asarray(z) - root[1])


def flare(x, z, clr, cap, curl, x0, zr, lean, n=36):
    """Depth y of a surface `clr` off the head at front-view (x, z). It hugs the head, but its slope away
    from the face is capped (cap, dropping by `curl` per stud beyond |x| = x0), so the outer part flares
    off past the head's silhouette instead of wrapping round it. Outside zr the reference section only
    follows the head's taper by the fraction `lean`, so the wings stand off the forehead."""
    ax = np.abs(np.asarray(x, float)).ravel()
    z = np.asarray(z, float).ravel()
    xs = ax[:, None] * np.linspace(0.0, 1.0, n)[None, :]
    th = np.arcsin(np.clip(xs / 0.64, -1.0, 1.0))
    r_c = HEAD(np.stack([th, np.repeat(np.clip(z, *zr)[:, None], n, 1)], -1))
    r_z = HEAD(np.stack([th, np.repeat(z[:, None], n, 1)], -1))
    rc = r_c + lean * (r_z - r_c) + clr
    yh = -np.sqrt(np.maximum(rc * rc - xs * xs, 1e-4))
    dx = np.maximum(ax / (n - 1), 1e-9)[:, None]
    lim = cap - curl * np.maximum(0.5 * (xs[:, 1:] + xs[:, :-1]) - x0, 0.0)
    return yh[:, 0] + (S.smin(np.diff(yh, axis=1) / dx, lim, 0.25) * dx).sum(1)


class Leaf:
    """A wing's surface over the front view: the face side from flare() pushed forward by `lift`, the
    outer side `thick` out along the normal."""

    def __init__(self, clr, cap, curl, zr, lift, thick, x0=0.45, lean=0.5):
        self.args, self.lift, self.thick = (clr, cap, curl, x0, zr, lean), lift, thick

    def depth(self, x, z):
        return flare(x, z, *self.args) - self.lift(np.abs(x), z)

    def frame(self, x, z):
        """Face-side points, outward normals and outer points at front-view (x, z)."""
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        h = 0.003
        gx = (self.depth(x + h, z) - self.depth(x - h, z)) / (2 * h)
        gz = (self.depth(x, z + h) - self.depth(x, z - h)) / (2 * h)
        n = K._unit(np.c_[gx, -np.ones_like(x), gz])
        inner = np.c_[x, self.depth(x, z), z]
        return inner, n, inner + n * self.thick(np.abs(x), z)[:, None]


# -------------------------------------------------------------------------------------- outlines
def area2(p):
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def curve(ctrl, ds):
    """Open Catmull-Rom through ctrl resampled every ~ds, denser on tight bends; keeps both ends."""
    p = K.spline(ctrl, 24)
    t = K._unit(np.diff(p, axis=0))
    turn = np.r_[0.0, np.arccos(np.clip((t[1:] * t[:-1]).sum(1), -1.0, 1.0))]
    cost = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1) / ds + turn / 0.35)]
    idx = np.interp(np.linspace(0.0, cost[-1], max(2, int(math.ceil(cost[-1]))) + 1), cost, np.arange(len(p)))
    i0 = np.minimum(np.floor(idx).astype(int), len(p) - 2)
    f = (idx - i0)[:, None]
    return p[i0] * (1 - f) + p[i0 + 1] * f


def outline(pieces, jag_x=0.45):
    """Closed outline from chained pieces [(ctrl, ds, jag)]: each an open spline, so the joins stay
    corners; `jag` frays the edge inwards (fading in beyond x = jag_x)."""
    pts, amp = [], []
    for ctrl, ds, jag in pieces:
        c = curve(ctrl, ds)[:-1]
        a = jag * smooth(jag_x, jag_x + 0.15, c[:, 0])
        a[0] = 0.0
        pts.append(c)
        amp.append(a)
    p, amp = np.vstack(pts), np.concatenate(amp)
    t = K._unit(np.roll(p, -1, 0) - np.roll(p, 1, 0))
    inward = np.c_[-t[:, 1], t[:, 0]] * math.copysign(1.0, area2(p))
    return p + inward * (amp * rng.uniform(0.0, 1.0, len(p)) ** 1.5)[:, None]


def tear(pts, at, depth, width, skew=0.0):
    """Cut a torn notch into the closed outline at the point nearest `at`."""
    p = np.roll(np.asarray(pts, float), len(pts) // 2 - int(np.argmin(np.linalg.norm(pts - np.asarray(at), axis=1))), 0)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    s0 = s[len(p) // 2]

    def at_s(q):
        i = min(int(np.searchsorted(s, q, side='right')) - 1, len(p) - 2)
        return p[i] + (p[i + 1] - p[i]) * (q - s[i]) / max(seg[i], 1e-12)
    t = K._unit(at_s(s0 + 0.004) - at_s(s0 - 0.004))
    inward = np.array([-t[1], t[0]]) * math.copysign(1.0, area2(p))
    jit = rng.uniform(-0.15, 0.15, 3) * depth
    notch = [at_s(s0 - width), at_s(s0 - 0.45 * width) + inward * (0.45 * depth + jit[0]),
             at_s(s0 + skew * width * 0.4) + inward * depth, at_s(s0 + 0.5 * width) + inward * (0.35 * depth + jit[1]),
             at_s(s0 + width)]
    before, after = p[s < s0 - width * 1.05], p[s > s0 + width * 1.05]
    return np.vstack([before, notch, after])


def almond(c, a, b_up, b_low, tilt, n=10):
    """Pointed eye opening: two circular arcs meeting at the corners (+-a along the tilted axis)."""
    out = []
    for b, sgn in ((b_up, 1.0), (b_low, -1.0)):
        cy, rad = sgn * (b * b - a * a) / (2 * b), (a * a + b * b) / (2 * b)
        g0, g1 = math.atan2(-cy, sgn * a), math.atan2(-cy, -sgn * a)     # counter-clockwise, corner to corner
        angs = np.linspace(g0, g1 + (2 * np.pi if g1 < g0 else 0.0), n, endpoint=False)
        out += [(rad * math.cos(g), cy + rad * math.sin(g)) for g in angs]
    ct, st = math.cos(tilt), math.sin(tilt)
    return np.array([(c[0] + x * ct - y * st, c[1] + x * st + y * ct) for x, y in out])


def blob(c, r, n=8, jag=0.3):
    g = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.2, 0.2, n)
    rr = r * (1 + jag * rng.uniform(-1, 1, n))
    return np.c_[c[0] + rr * np.cos(g), c[1] + rr * np.sin(g)]


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.einsum('nmk,mk->nm', p[:, None, :] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return np.linalg.norm(p[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=2)


def interior(poly, holes, sp):
    """Hex grid of interior points, kept clear of every boundary."""
    lo, hi = poly.min(0), poly.max(0)
    pts = np.array([(x, z) for j, z in enumerate(np.arange(lo[1] + sp * 0.5, hi[1], sp * 0.866))
                    for x in np.arange(lo[0] + (sp * 0.5 if j % 2 else 0.0), hi[0], sp)])
    pts = pts[[K.point_in_poly(q, poly) and not any(K.point_in_poly(q, h) for h in holes) for q in pts]]
    loops = [poly, *holes]
    a = np.vstack(loops)
    b = np.vstack([np.roll(lp, -1, 0) for lp in loops])
    return pts[seg_dist(pts, a, b).min(1) > 0.55 * sp]


def wing(group, mat, leaf, pieces, side, holes=(), tears=(), sp=0.05, jag_x=0.45):
    pts = outline(pieces, jag_x)
    for t in tears:
        pts = tear(pts, *t)
    m = np.array([side, 1.0])
    pts, holes = pts * m, [np.asarray(h) * m for h in holes]
    v2, tris = K.triangulate_region(pts, holes, interior(pts, holes, sp))
    _, _, outer = leaf.frame(v2[:, 0], v2[:, 1])
    inner = leaf.frame(v2[:, 0], v2[:, 1])[0]
    A.part(group, K.solid_from_surface(outer, tris, inner, uv2d=v2), mat, smooth=55)
    return pts


# ------------------------------------------------------------------------------------ body parts
def body_sdf(P):
    yb = BODY_Y
    q = P.copy()
    q[..., 0] = np.abs(q[..., 0])                  # symmetric fur so decimation can mirror
    thorax = S.ellipsoid(P, (0, yb, 2.662), (0.046, 0.04, 0.088))
    teg = np.minimum(*(S.ellipsoid(P, (s * 0.027, yb - 0.01, 2.672), (0.019, 0.027, 0.058), S.rot_y(s * 0.3))
                       for s in (-1, 1)))
    collar = S.ellipsoid(P, (0, yb - 0.006, 2.738), (0.038, 0.032, 0.02))
    d = S.smin(S.smin(thorax, teg, 0.012), collar, 0.01)
    d = d + 0.0035 * S.waves(q, 150.0, seed=5) * np.clip((P[..., 2] - 2.56) / 0.03, 0.0, 1.0)   # fur lumps
    head = S.ellipsoid(P, (0, yb - 0.008, 2.775), (0.029, 0.027, 0.025))
    eyes = np.minimum(*(S.sphere(P, (s * 0.024, yb - 0.014, 2.772), 0.0145) for s in (-1, 1)))
    d = np.minimum(S.smin(d, head, 0.008), eyes)
    abd = None
    for k, (z, rx, ry, rz) in enumerate(SEGS):
        seg = S.ellipsoid(P, (0, -0.648 - 0.016 * (k / 6) ** 2, z), (rx, ry, rz))
        abd = seg if abd is None else S.smin(abd, seg, 0.004)
    abd = S.smin(abd, S.capsule(P, (0, -0.664, 2.32), (0, -0.674, 2.29), 0.011, 0.003), 0.006)
    return S.smin(d, abd, 0.01)


def surf_point(x, z, off):
    """Point `off` out from the (smoothed) head surface at front-view (x, z)."""
    r = 0.62
    for _ in range(3):
        r = float(HEAD([math.asin(max(-1.0, min(1.0, x / r))), z])[0]) + off
    return np.array([x, -math.sqrt(max(r * r - x * x, 1e-6)), z])


def antenna(side, turns):
    """Bipectinate antenna from the moth's head up over the forehead, ending in a scroll of `turns`."""
    arc = K.spline([(0.012, 2.786), (0.032, 2.84), (0.065, 2.9), (0.11, 2.955), (0.165, 3.0), (0.225, 3.036),
                    (0.29, 3.06), (0.345, 3.066)], 5)
    e, tau = arc[-1], K._unit(arc[-1] - arc[-2])
    rho0 = 0.042
    cen = e + np.array([tau[1], -tau[0]]) * rho0     # right of travel: the scroll turns outwards and down
    psi = np.linspace(0.0, turns * 2 * np.pi, int(turns * 16) + 1)[1:]
    a0 = math.atan2(e[1] - cen[1], e[0] - cen[0])
    scroll = cen + np.c_[np.cos(a0 - psi), np.sin(a0 - psi)] * (rho0 * np.exp(-0.32 * psi))[:, None]
    p2 = np.vstack([arc, scroll])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p2, axis=0), axis=1))]
    t = s / s[-1]
    off = 0.082 - 0.038 * smooth(0.0, 0.4, t) + 0.01 * smooth(0.6, 1.0, t)
    pts = np.array([surf_point(side * x, z, o) for (x, z), o in zip(p2, off)])
    rad = 0.0072 * (1 - 0.62 * t)
    A.part('antennae', K.tube(pts, rad, sides=4, caps=('flat', 'point'), chunk=10.0), 'moth_antenna', smooth=70)
    T = K._unit(np.gradient(pts, axis=0))
    n_arc = len(arc)
    cen3 = np.array([side * cen[0], cen[1]])
    nxt = 0.06
    for i in range(2, len(pts) - 2):
        if s[i] < nxt:
            continue
        in_scroll = i >= n_arc
        nxt = s[i] + (0.03 if in_scroll else 0.022)
        nrm = K._unit(pts[i] - np.array([0.0, 0.05, 2.45]))
        sd = K._unit(np.cross(nrm, T[i]))
        ln = (0.016 if in_scroll else 0.05 * (1 - t[i]) ** 0.6 * smooth(0.02, 0.15, t[i]))
        for sg in (1, -1):
            d = K._unit(math.cos(0.9) * T[i] + sg * math.sin(0.9) * sd + 0.3 * nrm)
            if in_scroll and np.dot(d[[0, 2]], np.array([pts[i][0], pts[i][2]]) - cen3) < 0:
                continue                                 # inside the scroll only the outer barbs fit
            A.part('antennae', K.tube([pts[i] - d * 0.002, pts[i] + d * ln], [0.0032, 0.0], sides=3,
                                      caps=('flat', 'point')), 'moth_antenna', smooth=70)


def jewels():
    """Pearls in bezels at the eye-spot centres and tarnished studs along one vein of the left forewing."""
    for leaf, sx in ((FW_L, 1.0), (FW_R, -1.0)):
        _, n, p = leaf.frame(sx * SPOT[0], SPOT[1])
        n, p = n[0], p[0]
        tg = K._unit(np.cross(n, Z))
        A.part('jewels', K.sphere(0.02, 10, 6), 'pearl', K.mat_trs(p + n * 0.008, (tuple(n), (0, 0, 1)),
                                                                  (1.04, 0.96, 0.95)), smooth=180)
        A.part('jewels', K.ring(0.021, 0.021, 0.0042, segs=10, sides=3), 'tarnish', K.frame(p + n * 0.003, tg, n),
               smooth=70)
    phi0, step, _, bend = FW_VEINS
    ex, ez, a, b, tilt = SPOT
    for r in np.arange(0.2, 0.6, 0.05) + rng.uniform(-0.01, 0.01, 8):
        ph = phi0 + STUD_VEIN * step + bend * r
        x, z = FW_ROOT[0] + r * math.cos(ph), FW_ROOT[1] + r * math.sin(ph)
        dx, dz = x - ex, z - ez
        qq = math.hypot((dx * math.cos(tilt) + dz * math.sin(tilt)) / a, (dz * math.cos(tilt) - dx * math.sin(tilt)) / b)
        if qq < 1.2 or not K.point_in_poly((x, z), FW_OUT_L) or any(K.point_in_poly((x, z), h) for h in FW_HOLES_L):
            continue
        if seg_dist(np.array([[x, z]]), FW_OUT_L, np.roll(FW_OUT_L, -1, 0)).min() < 0.025:
            continue
        _, n, p = FW_L.frame(x, z)
        rs = rng.uniform(0.0065, 0.011)
        stud = K.lathe([(0.0, rs * 0.6), (rs * 0.7, rs * 0.42), (rs, rs * 0.08), (rs * 0.95, -0.006)], segs=6,
                       phase=rng.uniform(0, 1))
        A.part('jewels', stud, 'tarnish', K.mat_trs(p[0] - n[0] * 0.001, (tuple(n[0]), (0, 0, 1))), smooth=50)


# ------------------------------------------------------------------------------------------ build
K.ensure_mannequin()
K.new_scene()
HEAD = head_map()
A = K.Asm()


def fw_lift(amp):
    return lambda ax, z: amp * smooth(0.45, 0.95, rdist(FW_ROOT, ax, z)) ** 2


def fw_thick(ax, z):
    return 0.016 + 0.011 * np.exp(-(rdist(FW_ROOT, ax, z) / 0.13) ** 2)


def hw_lift(ax, z):
    return 0.016 * smooth(0.25, 0.58, rdist(HW_ROOT, ax, z)) ** 2 * smooth(2.52, 2.4, z)


def hw_thick(ax, z):
    return 0.013 + 0.009 * np.exp(-(rdist(HW_ROOT, ax, z) / 0.11) ** 2)


FW_L = Leaf(0.036, 1.0, 0.8, (2.3, 2.84), fw_lift(0.032), fw_thick)
FW_R = Leaf(0.036, 1.0, 0.8, (2.3, 2.84), fw_lift(0.045), fw_thick)       # the right tip curls up further
HW_L = HW_R = Leaf(0.016, 1.7, 0.6, (1.95, 2.84), hw_lift, hw_thick)

EYE = almond(*EYE_HOLE)
FW_HOLES_L = [EYE, blob((0.74, 2.95), 0.018)]
FW_OUT_L = wing('forewings', 'moth_fore', FW_L, FW, 1, FW_HOLES_L, [((0.70, 2.77), 0.03, 0.024, 0.2)])
wing('forewings', 'moth_fore', FW_R, FW, -1, [EYE, blob((0.56, 2.95), 0.011, 7)],
     [((0.805, 2.93), 0.055, 0.034, -0.3), ((0.6, 2.68), 0.02, 0.016, 0.0)])
wing('hindwings', 'moth_hind', HW_L, HW, 1, tears=[((0.505, 2.5), 0.026, 0.02, 0.3)], jag_x=0.3, sp=0.055)
wing('hindwings', 'moth_hind', HW_R, HW, -1, tears=[((0.41, 2.24), 0.016, 0.016, 0.0)], jag_x=0.3, sp=0.055)
A.part('body', S.mesh(body_sdf, (-0.075, -0.75, 2.26), (0.075, -0.6, 2.82), 0.0045, 640, sym='X', name='body'),
       'moth_fur', smooth=180)
antenna(1, 0.8)
antenna(-1, 1.35)
jewels()

body = K.dev_modes(A, NAME, VIEWS)
ob, rep = K.finish_item(A, NAME, 'Face', 'FaceFront', 'Ashen Moth Mask', meta=body)
K.previews(NAME, VIEWS, spp=int(os.environ.get('PREVIEW_SPP', 40)))
print('DONE', NAME)
