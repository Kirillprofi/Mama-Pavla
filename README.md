# Ashen accessories: procedural Roblox rigid accessories

Gothic / visual-kei rigid accessories for the Roblox Marketplace. Every item is built by a Python script
(Blender 5.2 `bpy` module): procedural geometry, procedural materials baked into PBR textures, checked against
Marketplace limits and a fitting mannequin, then exported as FBX with preview renders.

| Path | Contents |
| --- | --- |
| `!free.7z` | Earlier items (FBX, previews, build scripts, original toolkit) |
| `tools/rbxkit.py` | Pipeline: rigs and mannequins, materials, bake, validation, FBX export, previews |
| `tools/rbxgeo.py` | Geometry primitives (tubes, sweeps, lathes, prisms, chains) with world-scale UVs |
| `tools/rbxsdf.py` | Signed-distance modelling for organic parts (marching cubes + decimation) |
| `tools/check_fbx.py` | Re-imports exported FBX files and checks triangles, UV sets, materials, textures |
| `tools/build_<item>.py` | One build script per accessory |
| `assets/<Item>/` | `<Item>.fbx` (mesh + embedded textures), `<Item>.blend`, `<Item>.json` (validation and fit), `textures/`, preview PNGs |

Textures per item: `_ALB` colour 1024 px (sRGB), `_NOR` tangent-space normal, `_RGH` roughness and `_MTL`
metalness at 256 px (linear). Each mesh is one closed object with one material, one UV set and at most 4,000
triangles.

## Running

```sh
uv venv --python 3.13 ~/bpyenv
VIRTUAL_ENV=~/bpyenv uv pip install bpy==5.2.4 numpy pillow scipy scikit-image
~/bpyenv/bin/python tools/build_<item>.py           # full build into assets/<Item>/
DRY=1 ~/bpyenv/bin/python tools/build_<item>.py     # triangles, body clearance and bounds only
LOOK=1 ~/bpyenv/bin/python tools/build_<item>.py    # quick un-baked renders into .cache/
~/bpyenv/bin/python tools/check_fbx.py assets/*/*.fbx
```

`BAKE_SPP`, `PREVIEW_SPP` and `LOOK_SPP` override sample counts. The first run downloads Roblox's
`LC_ClassicMale.fbx` reference mannequin into `.cache/`.

## R6 fitting

New items are fitted on the default R6 avatar (`K.use_rig('R6')`). Frame: 1 unit = 1 stud, the character faces
-Y, +X is the character's left, Z is up and z = 0 is the torso bottom. The torso is a 2 x 1 x 2 block, arms and
legs are 1 x 1 x 2 blocks, and the 1.2-stud classic head sits with its centre 0.5 studs above the torso top.
Attachment points match the R6 rig (`HatAttachment` 0.6 above the head centre, `FaceFrontAttachment` 0.6 in
front of it, `NeckAttachment` and the collar attachments on the torso top, waist attachments on the torso
bottom, shoulder attachments on top of the arms).

Each `<Item>.json` lists the accessory type, the `roblox_attachment` name and `handle_attachment_position`:
the position, in Roblox axes and studs, of that attachment inside the imported (origin-centred) mesh. In Studio,
import the FBX as the accessory's `Handle` and add an `Attachment` with that name and position, or place the
mesh with the Accessory Fitting Tool.
