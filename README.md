# TurtleBot3 Transport-Water-to-Target — RL

A TurtleBot3 (carrying a cup of water) drives to a random target point while spilling as little
**real water** as possible. Trained with Stable-Baselines3 (SAC); the cup holds real PBD fluid
and the reward uses the fraction of water remaining.

## Files
| File | Purpose |
|---|---|
| `tb3_water_env.py` | Gymnasium environment (real water, spill reward) |
| `train_tb3_water.py` | SAC training |
| `play_tb3_water.py` | Load a trained policy and show it in the GUI |
| `check_spill.py` | No model needed; tests whether aggressive driving spills water |
| `tb3cup.py` | Static demo: TurtleBot with a full cup of real water (no training) |
| `turtlebot3_waffle_cup/` | Robot / cup assets |

---

## 1. Training

```bash
python3 train_tb3_water.py --timesteps 200000
```

When it finishes, this folder will contain:
- `sac_tb3_water.zip` — the trained model
- `ckpt_tb3_water/` — checkpoints every 20000 steps
- `tb_tb3_water/` — TensorBoard logs

**Arguments** (all optional, defaults shown):
| Arg | Default | Meaning |
|---|---|---|
| `--timesteps` | 200000 | Total training steps |
| `--max-steps` | 1000 | Max steps per episode |
| `--save` | sac_tb3_water | Model save name |
| `--seed` | 0 | Random seed |
| `--gui` | (off) | Add it to open a window and watch while training (slower); headless by default |

> Real fluid runs in a single environment, so training is slow — that's expected. Requires GPU
> physics (the `env_isaaclab_opt` env).

### Task / success definition
- **action:** `[v, ω]` linear / angular velocity (differential-drive, Gazebo `cmd_vel` style); limits V=2.6, W=1.82 (real TB3 Waffle × 10× scale)
- **obs (9-d):** target-relative position, heading error, linear/angular velocity, cup tilt, water fraction
- **target:** random each episode, 15–40 (scaled world units) from the start
- **reward:** progress toward target + reach bonus (× water) − spill (real particles) − tilt − action jerk
- **reach:** robot gets within radius 3.0 of the target
- **episode ends:** reached / water < 50% (spill failure) / tipped over / timeout

---

## 2. TensorBoard (training curves)

During or after training, **open a second terminal**:
```bash
tensorboard --logdir tb_tb3_water
```
Open **http://localhost:6006** in a browser (for remote access replace `localhost` with the
machine IP and add `--bind_all`).

**Key metrics:**
- `rollout/ep_rew_mean` — mean episode reward; **the main thing to watch (should trend up)**
- `rollout/ep_len_mean` — mean episode length (usually shortens once it learns to reach)
- `train/critic_loss`, `actor_loss`, `ent_coef` — learning health (just stay stable; no NaN/divergence)
- `time/fps` — training speed

> `rollout/*` only appears after the first episode ends; `train/*` starts after `learning_starts`
> (2000 steps). Empty charts at the start are normal.

---

## 3. Demo / Play (watch the trained policy)

```bash
python3 play_tb3_water.py --model sac_tb3_water --episodes 5
```
Opens a **GUI window** by default; watch the robot carry water toward the target. Per episode it prints:
```
ep0: steps=.. reward=.. dist=.. water=.. success=True/False
```

**Arguments:**
| Arg | Default | Meaning |
|---|---|---|
| `--model` | sac_tb3_water | Which model to load (without `.zip`) |
| `--episodes` | 5 | How many episodes |
| `--max-steps` | 1500 | Max steps per episode |
| `--headless` | (off) | Add it to run without a window (data only) |

---

## 4. Other tools

**Test whether it spills** (no model needed):
```bash
python3 check_spill.py                    # full-speed straight drive (GUI)
python3 check_spill.py --mode accel_stop  # accelerate → hard stop, repeated (most likely to spill)
python3 check_spill.py --headless         # data only
```

**Static water demo** (no training):
```bash
python3 tb3cup.py
```

---

## Notes
- Repeated training runs produce `tb_tb3_water/SAC_1`, `SAC_2`, … ; `--logdir tb_tb3_water`
  shows them all together for comparison.
