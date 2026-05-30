"""Load a trained policy and watch the TurtleBot carry water to the target in a GUI window
(the real water sloshes).

Run:
    /home/shareduser/anaconda3/envs/env_isaaclab_opt/bin/python \
        play_tb3_water.py --model sac_tb3_water --episodes 5
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="sac_tb3_water")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--max-steps", type=int, default=1500)
parser.add_argument("--headless", action="store_true", default=False)
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
from stable_baselines3 import SAC
from tb3_water_env import Tb3WaterEnv

env = Tb3WaterEnv(simulation_app, headless=args.headless, max_steps=args.max_steps)
model = SAC.load(args.model)
print(f"✅ loaded model {args.model}.zip")

for ep in range(args.episodes):
    obs, _ = env.reset()
    done = False
    total_r = 0.0
    steps = 0
    while not done and simulation_app.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        steps += 1
        done = term or trunc
    print(f"ep{ep}: steps={steps} reward={total_r:.1f} dist={info['dist']:.2f} "
          f"water={info['water']:.3f} success={info['is_success']}")

env.close()
