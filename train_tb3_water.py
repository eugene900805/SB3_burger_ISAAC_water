"""Train TurtleBot3 with Stable-Baselines3 (SAC) to carry water to a target.

Single environment + real PBD fluid (in the loop), scene at 10× scale.
Note: real fluid in a single environment, so training is slow — that's expected.

Run:
    /home/shareduser/anaconda3/envs/env_isaaclab_opt/bin/python \
        train_tb3_water.py --timesteps 200000
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", type=int, default=200_000)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--gui", dest="headless", action="store_false",
                    help="show the window (slower)")
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument("--save", type=str, default="sac_tb3_water")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# SimulationApp must be created before importing omni / the env
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from tb3_water_env import Tb3WaterEnv

env = Tb3WaterEnv(simulation_app, headless=args.headless,
                  max_steps=args.max_steps, seed=args.seed)
env = Monitor(env)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    seed=args.seed,
    learning_rate=3e-4,
    buffer_size=300_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    learning_starts=2_000,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log="./tb_tb3_water",
)

ckpt = CheckpointCallback(save_freq=20_000, save_path="./ckpt_tb3_water",
                          name_prefix="sac_tb3_water")

try:
    model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=True)
finally:
    model.save(args.save)
    print(f"✅ model saved: {args.save}.zip")
    env.close()
