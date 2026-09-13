# microduck-mjbatch

Train the [Pollen Microduck](https://github.com/pollen-robotics/microduck) — a ~800 g, ~25 cm
biped — with [mjbatch](https://github.com/kevinzakka/mjbatch): 4096 MuJoCo instances stepped in
parallel through a C++ thread pool, GIL released, GPU idle.

```
uv run train_ppo.py --num-envs 4096 --iterations 1500 --timestep 0.005
uv run scripts/render_policy.py --policy microduck_policy.pt --command 0.4,0,0
```

`microduck/` holds the robot MJCF, vendored from
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (Apache-2.0) by
`scripts/vendor_mjcf.py`, which strips the 76 visual mesh geoms (23 MB of STLs, contype=0,
conaffinity=0) and keeps the 4 meshes that actually collide. `scripts/check_vendored.py` asserts
the stripped model takes a bit-identical trajectory over 200 steps of a control sweep.

## Measured on 8 vCPU (RTX 2080 Ti present but unused)

Per PPO iteration at 4096 envs × 24 steps (98 304 env-steps):

| | rollout | GAE | update | total |
|---|---|---|---|---|
| `--timestep 0.002` (500 Hz, 10 substeps/action) | 9.14 s | 0.001 s | 2.26 s | 11.4 s |
| `--timestep 0.005` (200 Hz, 4 substeps/action) | | | | **6.1 s** |

mj_step calls — not simulated seconds — are what the CPU pays for: 110 000 substeps/s either way.
So the physics timestep is a throughput knob, and `0.005` buys 2.5× for a coarser contact
resolution. The policy still runs at 50 Hz in both cases. 1500 iterations at 0.005 ≈ 2.5 h.

## What this does not model (yet)

Upstream trains against a much richer plant. Missing here, in the order that matters:

1. BAM actuator models — real motor dynamics replace the plain `kp=5, kv=0.3` PD, and the torque
   ceiling is the real one, ±0.96 N·m
2. Backlash and the joint encoder bias
3. Domain randomization (friction, mass, latency)
4. The symmetry constraint that upstream uses to prevent a limping gait

The observation is the standard velocity-tracking layout: body-frame linear velocity, body-frame
angular velocity, projected gravity, joint positions and velocities, last action, command, and a
two-phase gait clock (56 dims). Reward: velocity and yaw-rate tracking, foot-lift gait shaping,
posture, uprightness, height, vertical bounce, action rate, torque, joint limits.

## Layout

```
train_ppo.py            model build, batched env, PPO, training loop
microduck/              vendored MJCF (2 XML + 4 STL, 2.9 MB)
scripts/vendor_mjcf.py  re-vendor from a microduck_rl checkout
scripts/check_vendored.py  prove the stripping changed no physics
scripts/render_policy.py   offscreen mp4 of a checkpoint (no display on this box)
```

CPU-only environment: `uv sync` (torch from the PyTorch CPU index).
