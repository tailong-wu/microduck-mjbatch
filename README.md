# microduck-bench

Training rig notes and results for [Microduck](https://github.com/pollen-robotics/microduck) RL
policies, built on [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
(mjlab / MuJoCo Warp + PPO).

This repo is the measurement layer around that upstream project: what one specific box actually
does, so the next run does not have to rediscover it.

## The box

| | |
|---|---|
| CPU | 8 vCPU (shared with a Celery worker and a Uvicorn app) |
| GPU | NVIDIA GeForce RTX 2080 Ti, 11 GB, sm_75 |
| Stack | Python 3.12, mjlab 1.3.0, warp-lang 1.12.0, torch 2.9.1+cu128 |

## Measured throughput

`Mjlab-Velocity-Flat-MicroDuck`, default PPO config, `num_steps_per_env=24`:

| num-envs | s / iteration | env-steps/s | notes |
|---|---|---|---|
| 4096 | 3.10–3.29 | ~30k | 5.2 GB VRAM, GPU util 73% — **use this** |
| 8192 | 8.17 | ~24k | >11 GB VRAM, spills to host; slower per step |

- 20 000 iterations at 4096 envs ≈ **18.3 h**. The upstream default is 50 000.
- Checkpoints land in `logs/rsl_rl/velocity/*/model_*.pt` every 250 iterations.
- The GPU is the bottleneck, not the CPU: raising env count past what fits in VRAM loses.

## Commands

```bash
git clone https://github.com/pollen-robotics/microduck_rl && cd microduck_rl
UV_HTTP_TIMEOUT=600 uv sync

# smoke test first — catches ~95% of config errors in seconds
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5

# the run this repo records
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 20000

# offscreen video of a checkpoint: mjlab renders with rgb_array, no display needed
uv run play Mjlab-Velocity-Flat-MicroDuck --checkpoint logs/rsl_rl/velocity/<run>/model_<n>.pt --video
```

`WANDB_MODE=offline` keeps runs local; `wandb sync wandb/offline-run-*` uploads later if wanted.
No display is available on this box, so the interactive viewer path is out — use `--video`.

## Status

- 2026-09-13: training started, `max_iterations=20000`. Progress log: `logs/train-microduck-velocity.log`.
- Artifacts (policy checkpoints, gait videos) get added here as they land.

Upstream `microduck_rl` is Apache-2.0; this repo only holds measurements and scripts.
