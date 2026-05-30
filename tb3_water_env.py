"""Gymnasium env: a TurtleBot3 carries a cup of water and drives to a target point while
spilling as little as possible.

Design notes
------------
- Real PBD fluid runs in the loop (single environment); the scene stays at 10× scale (at real
  scale the thin cup walls let the fluid tunnel through).
- The water is settled only once in __init__ and the settled positions are stored; every episode
  restores them directly (no need to re-settle each episode). A second world.reset() corrupts the
  PBD particles, so the per-episode reset is a live reset (no world.reset).
- The robot starts at a fixed pose; the target is random each episode. action = [v, ω]
  (differential drive, Gazebo cmd_vel style), converted to left/right wheel velocity targets.

Usage (the caller must create the SimulationApp before importing this module):
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})
    from tb3_water_env import Tb3WaterEnv
    env = Tb3WaterEnv(simulation_app)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import omni.usd
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.physx.scripts import physicsUtils
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade


# ============================================================
# Scene constants (measured from the asset)
# ============================================================
USD_PATH      = "turtlebot3_waffle_cup/turtlebot3_waffle_cup.usd"
TB3_PRIM_PATH = "/World/TB3"
S             = 10.0                      # scene scale (required, otherwise the water tunnels)

WHEEL_RADIUS  = 0.033 * S                 # wheel radius
WHEEL_BASE    = 0.288 * S                 # wheel track (left/right wheels at y=±0.144)
WHEEL_DIR     = 1.0                       # set to -1.0 if "forward" command drives backward
WHEEL_DRIVE_KD = 2000.0                   # wheel velocity-drive gain

# Sensible mass/inertia at 10× (key: the URDF's original ~1 kg is too light and gets flung
# by fluid contact -> NaN)
BASE_MASS     = 1000.0
BASE_INERTIA  = (800.0, 800.0, 800.0)
WHEEL_MASS    = 50.0
WHEEL_INERTIA = (10.0, 10.0, 10.0)

# cup interior (already × S: scaled world coords, used to build the water column)
CUP_CX        = -0.065 * S
CUP_CY        = 0.0
CUP_INNER_R   = 0.054 * S
CUP_FLOOR_Z   = 0.172 * S
CUP_RIM_Z     = 0.3131 * S

# cup interior (unscaled base_footprint local coords: to test in-cup after mapping particles back to base)
CUP_CX_L      = -0.065
CUP_CY_L      = 0.0
CUP_INNER_R_L = 0.054
CUP_FLOOR_L   = 0.172
CUP_RIM_L     = 0.3131

# water particles
PCO                = 0.12
FREST              = PCO * 0.5
FREST_L            = FREST / S            # unscaled
FILL_COLUMN_FACTOR = 1.7                  # ~900 particles / ~93% full (near the rim; easier to spill on hard maneuvers)
SDF_RESOLUTION     = 320

# action limits (scaled world units)
V_MAX = 2.6                               # linear velocity (m/s); real TB3 Waffle 0.26 × 10× scale
W_MAX = 1.82                              # angular velocity (rad/s); real TB3 Waffle spec

# task
TARGET_MIN_R   = 15.0                     # min/max target radius from the start (scaled world)
TARGET_MAX_R   = 40.0
REACH_RADIUS   = 3.0                      # within this distance counts as reached
SPILL_FAIL     = 0.5                      # water below this fraction -> failure
DECIMATION     = 4                        # physics steps per RL step
SETTLE_STEPS   = 30                       # settle steps after reset


class Tb3WaterEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, simulation_app, headless=True, max_steps=1000, seed=None):
        super().__init__()
        self.sim_app = simulation_app
        self.headless = headless
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        # action: [v, ω] normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # obs: dx_r, dy_r, dist, cos(he), sin(he), lin_vel, ang_vel, tilt, water_frac
        high = np.array([np.inf] * 9, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._build_scene()
        self._settle_and_capture_water()

        self._step_count = 0
        self._prev_dist = None
        self._prev_action = np.zeros(2, dtype=np.float32)
        self.target = np.array([TARGET_MAX_R * 0.6, 0.0], dtype=np.float32)

    # --------------------------------------------------------
    # build the scene
    # --------------------------------------------------------
    def _build_scene(self):
        self.world = World(stage_units_in_meters=1.0,
                           physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
        self.world.scene.add_default_ground_plane()
        self.stage = omni.usd.get_context().get_stage()

        em = omni.kit.app.get_app().get_extension_manager()
        try:
            if not em.is_extension_enabled("omni.physx.fabric"):
                em.set_extension_enabled_immediate("omni.physx.fabric", True)
        except Exception:
            pass

        add_reference_to_stage(usd_path=USD_PATH, prim_path=TB3_PRIM_PATH)
        physicsUtils.set_or_add_scale_op(
            UsdGeom.Xformable(self.stage.GetPrimAtPath(TB3_PRIM_PATH)),
            Gf.Vec3f(S, S, S))

        # light
        light = self.stage.DefinePrim("/World/DistantLight", "DistantLight")
        light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(3000.0)

        # de-instance collisions only (don't touch visuals, or they stop rendering)
        for prim in self.stage.Traverse():
            sp = str(prim.GetPath())
            if sp.startswith(TB3_PRIM_PATH) and prim.IsInstanceable() and "collision" in sp.lower():
                prim.SetInstanceable(False)

        # GPU physics scene
        scene_prim = None
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene_prim = prim
                break
        if scene_prim is None:
            UsdPhysics.Scene.Define(self.stage, "/World/PhysicsScene")
            scene_prim = self.stage.GetPrimAtPath("/World/PhysicsScene")
        px = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        px.CreateEnableGPUDynamicsAttr().Set(True)
        px.CreateBroadphaseTypeAttr().Set("GPU")
        px.CreateSolverTypeAttr().Set("TGS")

        # articulation root (free base, no fixed joint; reset uses a live teleport)
        self.root_prim = None
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).startswith(TB3_PRIM_PATH) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                self.root_prim = prim
                break

        # cup collision -> SDF
        for prim in self.stage.Traverse():
            sp = str(prim.GetPath())
            if (sp.startswith(TB3_PRIM_PATH) and "mug" in sp.lower()
                    and prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("sdf")
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim).CreateSdfResolutionAttr().Set(SDF_RESOLUTION)

        # mass/inertia fix (key stability fix: at 10× the original mass is too light and gets flung by the fluid)
        for prim in self.stage.Traverse():
            sp = str(prim.GetPath())
            if not (sp.startswith(TB3_PRIM_PATH) and prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                continue
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            if "wheel" in sp.lower():
                mass_api.CreateMassAttr().Set(WHEEL_MASS)
                mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*WHEEL_INERTIA))
            else:
                mass_api.CreateMassAttr().Set(BASE_MASS)
                mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*BASE_INERTIA))

        # particle system + water material
        self.PS = "/World/WaterParticleSystem"
        PhysxSchema.PhysxParticleSystem.Define(self.stage, self.PS)
        psp = self.stage.GetPrimAtPath(self.PS)
        psp.GetAttribute("particleContactOffset").Set(PCO)
        psp.GetAttribute("restOffset").Set(PCO * 0.99)
        psp.GetAttribute("fluidRestOffset").Set(FREST)
        for name, val in [("maxVelocity", 2.0 * S),
                          ("solverPositionIterationCount", 24),
                          ("enableCCD", True)]:
            a = psp.GetAttribute(name)
            if a and a.IsValid():
                a.Set(val)
        mat = self.stage.DefinePrim("/World/WaterMaterial", "Material")
        pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(mat)
        pbd.CreateViscosityAttr().Set(0.091); pbd.CreateCohesionAttr().Set(0.01)
        pbd.CreateSurfaceTensionAttr().Set(0.0074); pbd.CreateDensityAttr().Set(1.0)
        pbd.CreateFrictionAttr().Set(0.1)
        UsdShade.MaterialBindingAPI.Apply(psp).Bind(
            UsdShade.Material(mat), UsdShade.Tokens.weakerThanDescendants, "physics")

        # initial water column
        cx, cy = CUP_CX, CUP_CY
        fill_top = CUP_FLOOR_Z + (CUP_RIM_Z - CUP_FLOOR_Z) * FILL_COLUMN_FACTOR
        spacing = PCO; rlim = CUP_INNER_R - FREST * 1.5
        pos = []; nr = int(np.floor(CUP_INNER_R / spacing)) + 1
        z = CUP_FLOOR_Z + FREST
        while z <= fill_top:
            for ix in range(-nr, nr + 1):
                for iy in range(-nr, nr + 1):
                    x = ix * spacing; y = iy * spacing
                    if x * x + y * y <= rlim * rlim:
                        pos.append((cx + x, cy + y, z))
            z += spacing
        self.n_particles = len(pos)
        self.points = UsdGeom.Points.Define(self.stage, "/World/WaterParticles")
        self.points.GetPointsAttr().Set([Gf.Vec3f(*p) for p in pos])
        self.points.GetWidthsAttr().Set([FREST * 2.0] * self.n_particles)
        pset = PhysxSchema.PhysxParticleSetAPI.Apply(self.points.GetPrim())
        pset.CreateFluidAttr().Set(True)
        self.points.GetPrim().GetRelationship("physxParticle:particleSystem").AddTarget(Sdf.Path(self.PS))

        # target visual marker
        self.target_prim = UsdGeom.Sphere.Define(self.stage, "/World/Target")
        self.target_prim.GetRadiusAttr().Set(REACH_RADIUS)
        self._target_xform = UsdGeom.Xformable(self.target_prim.GetPrim())
        self._target_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

        # robot wrapper
        self.robot = Robot(prim_path=str(self.root_prim.GetPath()), name="tb3")
        self.world.scene.add(self.robot)

        self._points_attr = self.points.GetPointsAttr()
        self._xc = UsdGeom.XformCache()

    # --------------------------------------------------------
    # initial settle + capture settled positions (restored directly on later resets)
    # --------------------------------------------------------
    def _settle_and_capture_water(self):
        # world.reset() is called only once in the whole program (a 2nd call corrupts PBD particles)
        self.world.reset()
        self._setup_wheel_drive()
        self._zero_wheels()
        for _ in range(240):
            self.world.step(render=False)
        # capture settled positions + start pose
        settled = self._points_attr.Get()
        self.settled_positions = [Gf.Vec3f(p2[0], p2[1], p2[2]) for p2 in settled]
        self._zero_vel = [Gf.Vec3f(0, 0, 0)] * self.n_particles
        pos, yaw, m = self._base_pose()
        self.start_pos = np.array([pos[0], pos[1], pos[2]], dtype=np.float32)
        sq = m.ExtractRotationQuat()
        self.start_quat = np.array([sq.GetReal(), *[float(x) for x in sq.GetImaginary()]], dtype=np.float32)
        print(f"[capture] settled water={self._water_fraction_and_tilt(m):.3f} "
              f"start_pos={np.round(self.start_pos,3)} n={self.n_particles}")

    def _zero_wheels(self):
        self._apply_wheel_velocity(0.0, 0.0)

    def _dbg(self):
        p, yaw, m = self._base_pose()
        pts = self._points_attr.Get()
        arr = np.asarray([(q[0], q[1], q[2]) for q in pts])
        wf = self._water_fraction_and_tilt(m)
        bm = np.array(m); z = bm[2, :3] / (np.linalg.norm(bm[2, :3]) + 1e-9)
        tilt = float(np.arccos(np.clip(z[2], -1, 1)))
        return (f"base=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) yaw={yaw:.2f} tilt={tilt:.2f} "
                f"water={wf:.3f} pz=[{arr[:,2].min():.2f},{arr[:,2].max():.2f}]")

    def _setup_wheel_drive(self, kd=WHEEL_DRIVE_KD):
        names = list(self.robot.dof_names)
        self.wheel_idx = [names.index("wheel_left_joint"), names.index("wheel_right_joint")]
        ndof = len(names)
        kps = np.zeros(ndof, dtype=np.float32)
        kds = np.full(ndof, kd, dtype=np.float32)
        self.robot.get_articulation_controller().set_gains(kps=kps, kds=kds)

    # --------------------------------------------------------
    # observation / state
    # --------------------------------------------------------
    def _base_pose(self):
        # read pose from physics (under fabric, UsdGeom.XformCache returns stale values)
        pos, quat = self.robot.get_world_pose()      # quat = (w, x, y, z)
        pos = np.asarray(pos, dtype=np.float64)
        w, x, y, z = [float(v) for v in quat]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        m = Gf.Matrix4d().SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        m.SetTranslateOnly(Gf.Vec3d(pos[0], pos[1], pos[2]))
        return pos, float(yaw), m

    def _water_fraction_and_tilt(self, base_mat):
        pts = self._points_attr.Get()
        if pts is None or len(pts) == 0:
            return 0.0
        arr = np.asarray([(p[0], p[1], p[2]) for p in pts], dtype=np.float64)
        inv = base_mat.GetInverse()
        # base_mat comes from get_world_pose (no scale) -> compare against the scaled cup constants
        ones = np.ones((arr.shape[0], 1))
        h = np.hstack([arr, ones])
        local = (h @ np.array(inv))[:, :3]
        r = np.sqrt((local[:, 0] - CUP_CX) ** 2 + (local[:, 1] - CUP_CY) ** 2)
        inside = (r <= CUP_INNER_R + FREST) & \
                 (local[:, 2] >= CUP_FLOOR_Z - 2 * FREST) & \
                 (local[:, 2] <= CUP_RIM_Z + 4 * FREST)
        return float(inside.mean())

    def _get_obs(self):
        pos, yaw, base_mat = self._base_pose()
        dx = self.target[0] - pos[0]
        dy = self.target[1] - pos[1]
        dist = float(np.hypot(dx, dy))
        # transform into the robot frame
        c, s = np.cos(-yaw), np.sin(-yaw)
        dx_r = c * dx - s * dy
        dy_r = s * dx + c * dy
        heading_err = np.arctan2(dy_r, dx_r)
        lin = self.robot.get_linear_velocity()
        ang = self.robot.get_angular_velocity()
        v_fwd = float(lin[0] * np.cos(yaw) + lin[1] * np.sin(yaw)) if lin is not None else 0.0
        w_z = float(ang[2]) if ang is not None else 0.0
        # cup tilt: angle between the base z-axis and world z
        bm = np.array(base_mat)
        zaxis = bm[2, :3]
        zaxis = zaxis / (np.linalg.norm(zaxis) + 1e-9)
        tilt = float(np.arccos(np.clip(zaxis[2], -1.0, 1.0)))
        water_frac = self._water_fraction_and_tilt(base_mat)
        self._last = dict(pos=pos, yaw=yaw, dist=dist, tilt=tilt, water=water_frac)
        return np.array([dx_r, dy_r, dist, np.cos(heading_err), np.sin(heading_err),
                         v_fwd, w_z, tilt, water_frac], dtype=np.float32)

    # --------------------------------------------------------
    # gym API
    # --------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # random target
        ang = self._rng.uniform(-np.pi, np.pi)
        rad = self._rng.uniform(TARGET_MIN_R, TARGET_MAX_R)
        self.target = np.array([rad * np.cos(ang), rad * np.sin(ang)], dtype=np.float32)
        self._target_xform.GetOrderedXformOps()[0].Set(
            Gf.Vec3d(float(self.target[0]), float(self.target[1]), CUP_FLOOR_Z))

        # ── live reset (no world.reset, to avoid corrupting the particles) ──
        # teleport the robot back to the start + zero its velocities
        self.robot.set_world_pose(position=self.start_pos, orientation=self.start_quat)
        self.robot.set_joint_velocities(np.zeros(len(self.robot.dof_names), dtype=np.float32))
        for setter, val in [("set_linear_velocity", np.zeros(3, np.float32)),
                            ("set_angular_velocity", np.zeros(3, np.float32))]:
            fn = getattr(self.robot, setter, None)
            if callable(fn):
                try: fn(val)
                except Exception: pass
        # restore water to the settled positions + zero velocities
        self._points_attr.Set(self.settled_positions)
        self.points.GetVelocitiesAttr().Set(self._zero_vel)
        self._zero_wheels()
        for _ in range(SETTLE_STEPS):
            self.world.step(render=False)

        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        obs = self._get_obs()
        self._prev_dist = self._last["dist"]
        return obs, {}

    def _apply_wheel_velocity(self, v, w):
        # differential drive: wheel angular velocity = linear velocity / r
        vr = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
        vl = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
        vel = np.zeros(len(self.robot.dof_names), dtype=np.float32)
        vel[self.wheel_idx[0]] = WHEEL_DIR * vl
        vel[self.wheel_idx[1]] = WHEEL_DIR * vr
        self.robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_velocities=vel))

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        v = float(action[0]) * V_MAX
        w = float(action[1]) * W_MAX
        self._apply_wheel_velocity(v, w)
        for _ in range(DECIMATION):
            self.world.step(render=not self._headless() )
        self._step_count += 1

        obs = self._get_obs()
        dist = self._last["dist"]; tilt = self._last["tilt"]; water = self._last["water"]

        # reward
        progress = (self._prev_dist - dist)
        reward = 1.0 * progress
        reward -= 0.01                                   # time penalty
        reward -= 0.02 * float(np.sum((action - self._prev_action) ** 2))  # action smoothness
        reward -= 0.5 * max(0.0, (1.0 - water))          # spill penalty
        reward -= 0.1 * tilt

        terminated = False
        if dist < REACH_RADIUS:
            reward += 100.0 * water                       # reached, and more water is better
            terminated = True
        if water < SPILL_FAIL:
            reward -= 50.0
            terminated = True
        if tilt > 1.0 or self._last["pos"][2] < -1.0:     # tipped over / fell through the ground
            reward -= 50.0
            terminated = True

        truncated = self._step_count >= self.max_steps
        self._prev_dist = dist
        self._prev_action = action
        info = {"dist": dist, "water": water, "tilt": tilt,
                "is_success": bool(dist < REACH_RADIUS)}
        return obs, float(reward), terminated, truncated, info

    def _headless(self):
        return self.headless

    def close(self):
        try:
            self.sim_app.close()
        except Exception:
            pass
