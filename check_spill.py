"""Check: does the water overflow when the TurtleBot drives at full speed (or keeps turning)?

No trained model needed — just issue fixed full-speed commands and watch the water fraction.

Run (opens a GUI by default so you can watch):
    PY=/home/shareduser/anaconda3/envs/env_isaaclab_opt/bin/python
    $PY check_spill.py                 # full-speed straight drive, GUI
    $PY check_spill.py --mode turn     # full-speed turning
    $PY check_spill.py --headless      # data only, no window
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["forward", "turn", "accel_stop"],
                    default="forward", help="forward=full-speed straight, turn=full-speed turn, accel_stop=drive->hard-stop repeated")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--headless", action="store_true", default=False,
                    help="no window, print data only")
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
from tb3_water_env import Tb3WaterEnv

env = Tb3WaterEnv(simulation_app, headless=args.headless, max_steps=args.steps + 10)
obs, _ = env.reset()
print(f"start: n_particles={env.n_particles}  water={env._last['water']:.3f}  mode={args.mode}")

min_water = 1.0
for i in range(args.steps):
    if args.mode == "forward":
        action = [1.0, 0.0]                       # full-speed straight
    elif args.mode == "turn":
        action = [0.3, 1.0]                       # drive while turning at full rate
    else:  # accel_stop: every 40 steps, switch between full forward and hard reverse
        action = [1.0, 0.0] if (i // 40) % 2 == 0 else [-1.0, 0.0]

    obs, r, term, trunc, info = env.step(action)
    min_water = min(min_water, info["water"])

    if (i + 1) % 30 == 0:
        lv = env.robot.get_linear_velocity()
        spd = float(np.hypot(lv[0], lv[1])) if lv is not None else -1.0
        print(f"step{i+1:4d}: speed={spd:4.2f}  water={info['water']:.3f}  "
              f"tilt={info['tilt']:.3f}  min_water={min_water:.3f}")

    if not simulation_app.is_running():
        break

spilled = 1.0 - min_water
print("=" * 50)
print(f"result: lowest water = {min_water:.3f}  ->  spilled about {spilled*100:.1f}%")
if spilled < 0.02:
    print("verdict: barely spills (stable even at full speed)")
elif spilled < 0.2:
    print("verdict: spills somewhat")
else:
    print("verdict: spills noticeably")
env.close()
