# microduck-mjbatch

Train the [Pollen Microduck](https://github.com/pollen-robotics/microduck) — a ~800 g, ~25 cm
biped — with [mjbatch](https://github.com/kevinzakka/mjbatch): thousands of MuJoCo instances
stepped in parallel on the CPU through a C++ thread pool, GIL released. No CUDA, no GPU.

Upstream [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) does the
same task on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp), which requires a CUDA GPU.
Microduck is small enough (21 qpos, 14 actuators, 76 geoms) that CPU batching is competitive with
a single GPU, so this repo ports the training loop to mjbatch.

## Status

Feasibility checked, trainer not written yet.

- `scene_walk.xml` loads in plain CPU MuJoCo — the actuators are stock `<position>` actuators, and
  upstream's BAM actuator models are a Python package, not a MuJoCo plugin, so nothing blocks a
  CPU port
- **54 400 sim-steps/s** for 4096 instances on 8 vCPU (`mjbatch.Batch`, 100 steps each)

For reference, the same robot on the same box through mjlab + RTX 2080 Ti measured 3.1–3.3 s per
PPO iteration at 4096 envs ≈ **30 000 env-steps/s end to end** (physics + rollout + update). The
two numbers are not the same metric: one is bare physics, the other a full training iteration.

## Plan

1. Vendor the Microduck MJCF from `microduck_rl` (Apache-2.0, attribution kept) and load it
   through `mjbatch.Batch`.
2. Write a single-file PPO trainer in the shape of mjbatch's own
   [`examples/go1_joystick.py`](https://github.com/kevinzakka/mjbatch/blob/main/examples/go1_joystick.py):
   velocity-command tracking, upright/pose cost, action-rate and torque penalties, gait-phase
   reward, GAE, clipped PPO.
3. Skip v0: BAM actuator physics, backlash, domain randomization — add once a gait exists.
4. Render the gait offscreen (`mujoco.Renderer` + ffmpeg, no display on this box) and record the
   CPU-vs-GPU comparison in this file.

## Layout

```
microduck/     vendored MJCF + meshes from microduck_rl
docs/          notes on the observation and reward layout being ported
logs/          run logs from the mjlab baseline and, later, this trainer
```

## Environment

Python 3.13, `mujoco==3.11.0` (CPU), `mjbatch`, `numpy`, `torch` (CPU is enough) — see
`pyproject.toml`. Verified on 8 vCPU / RTX 2080 Ti, where the GPU is unused by this repo.
