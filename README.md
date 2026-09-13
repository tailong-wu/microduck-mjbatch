# microduck-mjbatch

Train the [Pollen Microduck](https://github.com/pollen-robotics/microduck) — a ~800 g, ~25 cm
biped — with [mjbatch](https://github.com/kevinzakka/mjbatch): 4096 MuJoCo instances stepped in
parallel through a C++ thread pool, GIL released, GPU idle.

```
uv run train_ppo.py --num-envs 4096 --iterations 1500 --timestep 0.005
uv run scripts/render_policy.py --policy microduck_policy.pt --command 0.4,0,0 \
    --width 1920 --height 1080 --out media/microduck_walk_1080p.mp4
```

`scripts/vendor_mjcf.py` copies the robot MJCF out of a
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) checkout
(Apache-2.0) into two trees:

| tree | contents | who uses it |
|---|---|---|
| `microduck/` | complete model, 38 meshes, 21 MB | `scripts/render_policy.py` — renders the shell |
| `microduck/lean/` | 75 visual mesh geoms dropped (contype=0), 4 meshes, 2.9 MB | `train_ppo.py` — **2× the substep rate** |

Same physics either way: visual geoms never touch contact. Measured at 4096 envs, `--timestep
0.005`: full tree 62k substeps/s, lean tree 119k. `scripts/check_vendored.py` asserts both trees
take a bit-identical trajectory to upstream over 200 steps of a control sweep.

## Result: 4096 envs, 1500 iterations, 2 h 36 min, on 8 vCPU

Final iteration: reward **4.10**, `track 0.95`, `turn 0.79`, `falls 0.00`.

| command (m/s, m/s, rad/s) | measured |
|---|---|
| `0.3, 0, 0` | 0.21 m/s forward, steady (median \|v\| 0.218, no falls over 5 s) |
| `0, 0, 0` | stands, 0.13 m drift over 8 s |
| `0, 0, 0.8` | turns in place |

`media/microduck_walk_1080p.mp4` (1920×1080, 50 fps) and `media/microduck_walk.mp4` (480p) are the
trained policy at `0.4` m/s, rendered offscreen by
`scripts/render_policy.py`. The checkpoint it renders, `microduck_policy.pt`, is in the repo. (The
policy was trained on the `--strip-visual` tree, which steps identically — visual geoms are
contype=0 — and is re-rendered here on the full model.)
End-to-end throughput of that run: 16k env-steps/s, i.e. 6.26 s per iteration.

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
microduck/              vendored MJCF, full tree (2 XML + 38 STL, 21 MB)
microduck/lean/         same model without the visual meshes (4 STL, 2.9 MB) — training uses this
scripts/vendor_mjcf.py  re-vendor from a microduck_rl checkout
scripts/check_vendored.py  prove the stripping changed no physics
scripts/render_policy.py   offscreen mp4 of a checkpoint (no display on this box)
media/microduck_walk*.mp4  the trained gait, 1080p and 480p
```

CPU-only environment: `uv sync` (torch from the PyTorch CPU index).
