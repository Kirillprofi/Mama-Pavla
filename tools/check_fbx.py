"""Re-import exported accessory FBX files and report what Roblox Studio will receive.

    python tools/check_fbx.py assets/*/*.fbx

Prints triangles, size in studs, UV sets, materials and embedded textures for each file, and exits
non-zero when one breaks a Marketplace limit (4k triangles, one mesh, one UV set, textures <= 1024).
"""
import os
import sys

import bpy  # noqa: F401  (must precede bmesh/mathutils)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbxkit as K  # noqa: E402


def check(path):
    K.new_scene()                    # scene scale 0.01 so FBX centimetres read back as studs
    bpy.ops.import_scene.fbx(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    problems = []
    if len(meshes) != 1:
        problems.append(f'{len(meshes)} mesh objects')
    for ob in meshes:
        me = ob.data
        tris = sum(len(p.vertices) - 2 for p in me.polygons)
        v = np.array([ob.matrix_world @ x.co for x in me.vertices])
        size = v.max(0) - v.min(0)
        imgs = sorted({(i.name, tuple(i.size)) for i in bpy.data.images if i.size[0]})
        print(f'{os.path.basename(path)}: {ob.name} tris={tris} size_studs={np.round(size, 3).tolist()} '
              f'centre={np.round((v.max(0) + v.min(0)) / 2, 4).tolist()} uv_layers={len(me.uv_layers)} '
              f'materials={len(me.materials)} textures={imgs}')
        if tris > K.TRI_LIMIT:
            problems.append(f'{tris} triangles')
        if len(me.uv_layers) != 1:
            problems.append(f'{len(me.uv_layers)} UV sets')
        if len(me.materials) != 1:
            problems.append(f'{len(me.materials)} materials')
        if any(max(s) > K.COLOR_SIZE for _, s in imgs):
            problems.append('texture larger than 1024')
    print('  OK' if not problems else '  PROBLEMS: ' + ', '.join(problems))
    return not problems


if __name__ == '__main__':
    files = [a for a in sys.argv[1:] if a.lower().endswith('.fbx')]
    ok = all([check(os.path.abspath(f)) for f in files])
    sys.exit(0 if ok and files else 1)
