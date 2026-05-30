"""Load a TurtleBot3 Waffle (with cup), scale it 10×, under /World/TB3.

Extras:
  1) Fix the TurtleBot to the world with a fixed joint (base_footprint stops moving).
  2) Fill the cup with water (PhysX PBD fluid particles) without tunneling.

How the water works: the cup collision is switched to SDF (keeps the cavity hollow), and a
loose water column is placed above the rim. On play it settles/densifies down into the cup,
ending around ~92% below the rim without overflowing or going through the walls.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.physx.scripts import physicsUtils
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
import numpy as np


# ============================================================
# Parameters
# ============================================================

USD_PATH    = "turtlebot3_waffle_cup/turtlebot3_waffle_cup.usd"
TB3_PRIM_PATH = "/World/TB3"
STAGE_UNITS_IN_METERS = 1.0
SCENE_SCALE = 10.0              # xform scale applied to /World/TB3

# cup geometry (measured from the original asset, in world coords at scale=1)
MUG_KEYWORD     = "mug"
MUG_CENTER_X_S1 = -0.065        # cup body center X
MUG_CENTER_Y_S1 = 0.0           # cup body center Y
MUG_OUTER_R_S1  = 0.0612        # cup body outer radius
MUG_INNER_R_S1  = 0.054         # cup inner-wall radius (measured)
MUG_BOTTOM_Z_S1 = 0.1628        # cup outer-bottom Z
MUG_FLOOR_Z_S1  = 0.172         # cup interior floor Z (measured)
MUG_RIM_Z_S1    = 0.3131        # cup rim Z

# water particle settings
PARTICLE_CONTACT_OFFSET = 0.12              # particle contact radius (world units; thin walls -> can't be too small or it tunnels)
FLUID_REST_OFFSET       = PARTICLE_CONTACT_OFFSET * 0.5
FILL_COLUMN_FACTOR      = 1.45              # initial column height = (rim-floor)*factor (~77% full after settling, ~765 particles)
SDF_RESOLUTION          = 320               # cup SDF collision resolution (must be high for thin walls)


# ============================================================
# World & load the USD as a reference
# ============================================================

world = World(
    stage_units_in_meters=STAGE_UNITS_IN_METERS,
    physics_dt=1.0 / 120.0,    # smaller step -> fluid less likely to tunnel
    rendering_dt=1.0 / 60.0,
)
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()

# enable the extension needed for fluid simulation
_ext_manager = omni.kit.app.get_app().get_extension_manager()
for _ext in ["omni.physx.fabric"]:
    try:
        if not _ext_manager.is_extension_enabled(_ext):
            _ext_manager.set_extension_enabled_immediate(_ext, True)
        print(f"✅ extension {_ext} enabled")
    except Exception as e:
        print(f"⚠️ extension {_ext} failed to enable: {e}")

add_reference_to_stage(usd_path=USD_PATH, prim_path=TB3_PRIM_PATH)
print(f"✅ Loaded {USD_PATH} → {TB3_PRIM_PATH}")

# Distant light
light = stage.DefinePrim("/World/DistantLight", "DistantLight")
light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(3000.0)
light.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.9, 0.9))
print("✅ Distant light added")

set_camera_view(
    eye=np.array([15.0, 15.0, 15.0]),
    target=np.array([0.0, 0.0, 1.0]),
    camera_prim_path="/OmniverseKit_Persp",
)


# ============================================================
# Scale /World/TB3 up by 10×
# ============================================================

tb3_prim = stage.GetPrimAtPath(TB3_PRIM_PATH)

if tb3_prim and tb3_prim.IsValid():
    physicsUtils.set_or_add_scale_op(
        UsdGeom.Xformable(tb3_prim),
        Gf.Vec3f(SCENE_SCALE, SCENE_SCALE, SCENE_SCALE),
    )
    print(f"✅ Scaled {TB3_PRIM_PATH} by {SCENE_SCALE}x")
else:
    print(f"⚠️ {TB3_PRIM_PATH} not found, scale not applied")


# ============================================================
# De-instance the "collisions" only (the cup collision mesh sits inside an instanceable
# reference, otherwise it can't be switched to SDF). Don't touch visuals, or the chassis
# and cup geometry stop rendering and "disappear".
# ============================================================

for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if not path_str.startswith(TB3_PRIM_PATH):
        continue
    if not prim.IsInstanceable():
        continue
    if "collision" not in path_str.lower():
        continue
    prim.SetInstanceable(False)
    print(f"✅ de-instance (collision only): {path_str}")


# ============================================================
# GPU physics scene (fluid particles need GPU dynamics)
# ============================================================

_scene_prim = None
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Scene):
        _scene_prim = prim
        break
if _scene_prim is None:
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    _scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    print("✅ PhysicsScene created")

physx_scene = PhysxSchema.PhysxSceneAPI.Apply(_scene_prim)
physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
physx_scene.CreateSolverTypeAttr().Set("TGS")
print(f"✅ GPU dynamics enabled on {_scene_prim.GetPath()}")


# ============================================================
# Fix the TurtleBot in place: add a fixed joint world -> articulation root
# ============================================================

_art_root = None
for prim in stage.Traverse():
    if not str(prim.GetPath()).startswith(TB3_PRIM_PATH):
        continue
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        _art_root = prim
        break

if _art_root is not None:
    xc = UsdGeom.XformCache()
    world_mat = xc.GetLocalToWorldTransform(_art_root)
    w_t = world_mat.ExtractTranslation()
    w_q = world_mat.ExtractRotationQuat()

    fixed_joint_path = f"{_art_root.GetPath()}/FixToWorldJoint"
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)
    # body0 left empty = world; body1 = articulation root link
    fixed_joint.CreateBody1Rel().SetTargets([_art_root.GetPath()])
    # anchor the world side at the root link's current world pose (so it isn't pulled to the origin)
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(w_t[0], w_t[1], w_t[2]))
    fixed_joint.CreateLocalRot0Attr().Set(
        Gf.Quatf(w_q.GetReal(), Gf.Vec3f(*[float(x) for x in w_q.GetImaginary()]))
    )
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    print(f"✅ TurtleBot fixed: {fixed_joint_path}")
else:
    print(f"⚠️ no ArticulationRoot under {TB3_PRIM_PATH}; TurtleBot not fixed")


# ============================================================
# Switch the cup collision to SDF (convexDecomposition fills the cavity solid,
# so water can't get in / it explodes)
# ============================================================

_mug_col_count = 0
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if not path_str.startswith(TB3_PRIM_PATH):
        continue
    if MUG_KEYWORD not in path_str.lower():
        continue
    if not prim.IsA(UsdGeom.Mesh):
        continue
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_col.CreateApproximationAttr().Set("sdf")
    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    sdf_api.CreateSdfResolutionAttr().Set(SDF_RESOLUTION)
    print(f"✅ cup collision switched to SDF: {path_str}")
    _mug_col_count += 1

print(f"✅ set {_mug_col_count} cup collision mesh(es) to SDF")
if _mug_col_count == 0:
    print("⚠️ no cup collision mesh found; water may tunnel or explode")


# ============================================================
# Particle System + PBD water material
# ============================================================

particle_system_path = "/World/WaterParticleSystem"
PhysxSchema.PhysxParticleSystem.Define(stage, particle_system_path)
ps_prim = stage.GetPrimAtPath(particle_system_path)
ps_prim.GetAttribute("particleContactOffset").Set(PARTICLE_CONTACT_OFFSET)
ps_prim.GetAttribute("restOffset").Set(PARTICLE_CONTACT_OFFSET * 0.99)
ps_prim.GetAttribute("fluidRestOffset").Set(FLUID_REST_OFFSET)
# anti-tunneling settings
for _name, _val in [
    ("maxVelocity", 2.0 * SCENE_SCALE),
    ("solverPositionIterationCount", 24),
    ("enableCCD", True),
]:
    _a = ps_prim.GetAttribute(_name)
    if _a and _a.IsValid():
        _a.Set(_val)

water_mat_prim = stage.DefinePrim("/World/WaterMaterial", "Material")
pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(water_mat_prim)
pbd.CreateViscosityAttr().Set(0.091)
pbd.CreateCohesionAttr().Set(0.01)
pbd.CreateSurfaceTensionAttr().Set(0.0074)
pbd.CreateDensityAttr().Set(1.0)
pbd.CreateFrictionAttr().Set(0.1)
UsdShade.MaterialBindingAPI.Apply(ps_prim).Bind(
    UsdShade.Material(water_mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")

# visual material (translucent blue)
water_vis_mat = UsdShade.Material.Define(stage, "/World/WaterVisualMaterial")
shader = UsdShade.Shader.Define(stage, "/World/WaterVisualMaterial/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.4, 0.6))
shader.CreateInput("opacity",      Sdf.ValueTypeNames.Float).Set(0.6)
shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.05)
shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
water_vis_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
print("✅ particle system + water material done")


# ============================================================
# Spawn the in-cup water particles: a loose column above the rim that settles to fill the cup
# ============================================================

cx          = MUG_CENTER_X_S1 * SCENE_SCALE
cy          = MUG_CENTER_Y_S1 * SCENE_SCALE
inner_r     = MUG_INNER_R_S1 * SCENE_SCALE
floor_z     = MUG_FLOOR_Z_S1 * SCENE_SCALE
rim_z       = MUG_RIM_Z_S1 * SCENE_SCALE
fill_top_z  = floor_z + (rim_z - floor_z) * FILL_COLUMN_FACTOR

spacing  = PARTICLE_CONTACT_OFFSET                 # initial grid spacing = 2*fluidRestOffset (no overlap)
r_limit  = inner_r - FLUID_REST_OFFSET * 1.5       # column radius slightly under the inner wall, so it doesn't scrape on the way down

positions = []
n_r = int(np.floor(inner_r / spacing)) + 1
z = floor_z + FLUID_REST_OFFSET
while z <= fill_top_z:
    for ix in range(-n_r, n_r + 1):
        for iy in range(-n_r, n_r + 1):
            x = ix * spacing
            y = iy * spacing
            if x * x + y * y <= r_limit * r_limit:
                positions.append((cx + x, cy + y, z))
    z += spacing

actual_count = len(positions)

points_prim = UsdGeom.Points.Define(stage, "/World/WaterParticles")
points_prim.GetPointsAttr().Set([Gf.Vec3f(*p) for p in positions])
points_prim.GetWidthsAttr().Set([FLUID_REST_OFFSET * 2.0] * actual_count)
particle_set = PhysxSchema.PhysxParticleSetAPI.Apply(points_prim.GetPrim())
particle_set.CreateFluidAttr().Set(True)
points_prim.GetPrim().GetRelationship("physxParticle:particleSystem").AddTarget(
    Sdf.Path(particle_system_path))
print(f"✅ WaterParticles done, {actual_count} particles (initial column z=[{floor_z:.2f},{fill_top_z:.2f}], rim {rim_z:.2f})")

# Isosurface (render the water as a continuous liquid surface)
iso = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(ps_prim)
iso.CreateIsosurfaceEnabledAttr().Set(True)
iso.CreateMaxVerticesAttr().Set(1024 * 1024)
iso.CreateMaxTrianglesAttr().Set(2 * 1024 * 1024)
iso.CreateMaxSubgridsAttr().Set(1024 * 4)
iso.CreateGridSpacingAttr().Set(FLUID_REST_OFFSET * 1.5)
print("✅ Isosurface done")


# ============================================================
# Reset & Main loop
# ============================================================

world.reset()
print("✅ world.reset() done (press play and the column settles to fill the cup)")

_iso_bound = False
_step = 0
while simulation_app.is_running():
    world.step(render=True)
    _step += 1
    # once the isosurface mesh exists, bind the visual material (only once)
    if not _iso_bound and _step > 30:
        iso_prim = stage.GetPrimAtPath(f"{particle_system_path}/Isosurface")
        if iso_prim and iso_prim.IsValid():
            pts = UsdGeom.Mesh(iso_prim).GetPointsAttr().Get()
            if pts and len(pts) > 10:
                UsdShade.MaterialBindingAPI.Apply(iso_prim).Bind(water_vis_mat)
                UsdGeom.Imageable(iso_prim).MakeVisible()
                print(f"✅ Isosurface material bound ({len(pts)} vertices)")
                _iso_bound = True

simulation_app.close()
