# SPDX-License-Identifier: Apache-2.0

"""PPO training a Microduck to follow a velocity command on CPU-batched mjbatch envs.

Shape follows mjbatch's own examples/go1_joystick.py: one ArrayLike view per mjData field, the
whole batch stepped with the GIL released, the network on the CPU.

Differences from the upstream mjlab training (pollen-robotics/microduck_rl) are deliberate and
listed in README.md: stock MuJoCo position actuators instead of BAM motor models, no backlash,
no domain randomization, no symmetry constraint.

  uv run train_ppo.py --num-envs 4096 --iterations 4000
"""

import argparse
import os
import pathlib
import time

import mujoco
import numpy as np
import torch
from torch import nn

from mjbatch import Batch

XML = pathlib.Path(__file__).parent / "microduck/scene_walk.xml"

# Control runs at 50 Hz. Upstream simulates at 500 Hz (mjlab's default timestep), which costs 10
# physics substeps per action; mj_step calls, not simulated seconds, are what the CPU pays for, so
# --timestep 0.005 buys 2.5x the throughput at a coarser contact resolution.
CTRL_DT = 0.02
TIMESTEP = 0.002
# Real motor ceiling is +-0.96 N m (see the chosen_actuator class upstream); kp/kv are a plain PD,
# standing in for the BAM motor model the real policy is trained against.
KP, KD, FORCE_LIMIT = 5.0, 0.3, 0.96
ACTION_SCALE = 1.0  # upstream JointPositionActionCfg.scale

NUM_ENVS, HORIZON, ITERS, EPISODE = 4096, 24, 4000, 1000
GAIT_HZ, PHASE = 1.5, np.array([0.0, 0.5])  # radians of thigh phase per leg
COMMAND_RANGE = np.array([0.5, 0.3, 1.2])  # forward m/s, sideways m/s, yaw rad/s
COMMAND_ON = np.array([0.9, 0.25, 0.5])  # chance each axis is nonzero when redrawn
COMMAND_SECONDS = 3.0
SIGMA, FLOOR = 0.25, -10.0
SWING, CEILING, WIDTH = 0.02, 0.04, 0.02  # foot lift targets, m
STAND_HEIGHT = 0.12  # trunk height in the STAND keyframe, m
REWARD = dict(
  track=1.0,
  turn=0.5,
  gait=1.0,
  pose=0.5,
  upright=1.0,
  height=0.5,
  slab=-0.5,
  rate=-0.1,
  torque=-0.005,
  limits=-1.0,
)

OBS_DIM, ACT_DIM = 3 + 3 + 3 + 3 * 14 + 3 + 2, 14
GAMMA, LAMBDA, CLIP, ENT_COEF = 0.99, 0.95, 0.2, 0.005
LR, LR_END, LR_TO = 1e-3, 5e-4, 1000
EPOCHS, MINIBATCHES, LOG_STD, HIDDEN, SEED = 5, 4, np.log(0.5), 128, 0
OUT = pathlib.Path(__file__).parent / "microduck_policy.pt"


def build_model(timestep=TIMESTEP):
  spec = mujoco.MjSpec.from_file(str(XML))
  spec.option.timestep = timestep
  for actuator in spec.actuators:
    actuator.set_to_position(kp=KP, kv=KD)
    actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    actuator.forcerange = [-FORCE_LIMIT, FORCE_LIMIT]
  return spec.compile()


def progress(head, done, elapsed, tail, end=False):
  bar = "━" * round(20 * done) + "─" * (20 - round(20 * done))
  left = f"{int(elapsed / done - elapsed) // 60}:{int(elapsed / done - elapsed) % 60:02d}" if done else "-:--"
  print(
    f"\r{head} {bar} {int(elapsed) // 60}:{int(elapsed) % 60:02d}, {left} left  {tail}\x1b[K",
    end="\n" if end else "",
    flush=True,
  )


class Duck:
  def __init__(self, num_envs, seed=0, timestep=TIMESTEP):
    self.batch = batch = Batch(build_model(timestep), num_envs)
    self.num_envs = num_envs
    self.decimation = round(CTRL_DT / timestep)
    self.model = batch.model
    self.qpos, self.qvel, self.ctrl = (batch.bind(f) for f in ("qpos", "qvel", "ctrl"))
    self.xmat, self.site, self.force = (batch.bind(f) for f in ("xmat", "site_xpos", "actuator_force"))
    self.trunk = self.model.body("trunk_base").id
    self.feet = [self.model.site(s).id for s in ("left_foot", "right_foot")]
    self.rot = self.xmat[:, self.trunk].reshape(-1, 3, 3)  # live view: world <- body
    self.foot = self.site[:, self.feet]  # (N, 2, 3)
    key = self.model.key("STAND")
    self.stand = np.zeros(self.model.nu, np.float32)
    self.stand[:] = key.ctrl
    self.joints = np.arange(7, 7 + ACT_DIM)  # free joint is qpos[0:7]
    self.dofs = np.arange(6, 6 + ACT_DIM)
    self.lower, self.upper = 0.95 * self.model.jnt_range[1:, 0], 0.95 * self.model.jnt_range[1:, 1]
    self.rng = np.random.default_rng(seed)
    self.steps, self.clock = np.zeros(num_envs, np.int64), np.zeros(num_envs)
    self.action = np.zeros((num_envs, ACT_DIM), np.float32)
    self.command, self.until = np.zeros((num_envs, 3)), np.zeros(num_envs)
    self.reset(np.arange(num_envs))

  def reset(self, ids):
    n = ids.size
    self.batch.reset(ids, keyframe=self.model.key("STAND").id)  # writes must follow the reset
    self.qpos[np.ix_(ids, self.joints)] += self.rng.uniform(-0.05, 0.05, (n, ACT_DIM))
    self.qpos[ids, 2] += self.rng.uniform(0.0, 0.01, n)
    self.qvel[np.ix_(ids, np.arange(6, self.model.nv))] = self.rng.uniform(-0.2, 0.2, (n, self.model.nv - 6))
    self.steps[ids], self.action[ids] = 0, 0.0
    self.clock[ids] = self.rng.uniform(0.0, 1.0, n)
    self.resample(ids, keep=0.0)
    self.batch.forward(ids)  # derived fields read by obs()

  def resample(self, ids, keep=0.5):
    n = ids.size
    fresh = self.rng.uniform(-1.0, 1.0, (n, 3)) * COMMAND_RANGE
    fresh *= self.rng.random((n, 3)) < COMMAND_ON
    kept = self.rng.random((n, 3)) < keep
    self.command[ids] = np.where(kept, self.command[ids], fresh)
    self.until[ids] = self.rng.exponential(COMMAND_SECONDS / CTRL_DT, n)

  def moving(self):
    return (np.linalg.norm(self.command, axis=1) > 0.01)[:, None]

  def obs(self):
    rot = self.rot
    vel_body = np.einsum("nji,nj->ni", rot, self.qvel[:, :3])  # world -> body linear velocity
    gyro = self.qvel[:, 3:6]  # MuJoCo's free-joint angular velocity is already body-frame
    up = -rot[:, 2]  # world -z (gravity direction) in the body frame
    angle, moving = 2 * np.pi * self.clock[:, None], self.moving()
    cols = (
      2.0 * vel_body,
      0.25 * gyro,
      up,
      self.qpos[:, self.joints] - self.stand,
      0.05 * self.qvel[:, self.dofs],
      self.action,
      self.command,
      np.sin(angle) * moving,
      np.cos(angle) * moving,
    )
    return np.concatenate(cols, 1, dtype=np.float32)

  def step(self, action):
    self.ctrl[:] = self.stand + ACTION_SCALE * action
    self.batch.step(nstep=self.decimation)
    self.steps += 1
    self.until -= 1
    self.resample(np.flatnonzero(self.until <= 0))
    rot, up, command = self.rot, -self.rot[:, 2], self.command
    q = self.qpos[:, self.joints]
    height, moving = self.foot[:, :, 2], self.moving()
    phase = np.sin(2 * np.pi * (self.clock[:, None] + PHASE)) * moving
    low, high = np.minimum(height - SWING * phase, 0.0), np.maximum(height - CEILING, 0.0)
    swing = np.exp(-(low**2 + high**2) / WIDTH**2)
    blend = np.clip(0.5 + phase, 0.0, 1.0) * moving
    stance = np.exp(-((height / 0.01) ** 2))
    over = np.maximum(self.lower - q, 0.0) + np.maximum(q - self.upper, 0.0)
    terms = dict(
      track=np.exp(
        -np.sum((np.einsum("nji,nj->ni", rot, self.qvel[:, :3])[:, :2] - command[:, :2]) ** 2, 1) / SIGMA
      ),
      turn=np.exp(-((self.qvel[:, 5] - command[:, 2]) ** 2) / SIGMA),
      gait=(blend * swing + (1.0 - blend) * stance).mean(1),
      pose=np.exp(-np.sum((q - self.stand) ** 2, 1)),
      upright=np.exp(-2.0 * (up[:, 0] ** 2 + up[:, 1] ** 2)),
      height=np.exp(-(((self.qpos[:, 2] - STAND_HEIGHT) / 0.02) ** 2)),
      slab=self.qvel[:, 2] ** 2,
      rate=np.sum((action - self.action) ** 2, 1),
      torque=np.mean(self.force**2, 1),
      limits=np.sum(over, 1),
    )
    reward = np.asarray(sum(REWARD[k] * v for k, v in terms.items()), np.float32)
    reward = np.maximum(reward, FLOOR)
    self.clock = (self.clock + GAIT_HZ * CTRL_DT) % 1.0
    self.action = action.astype(np.float32)
    fell = (up[:, 2] > -0.7) | (self.qpos[:, 2] < 0.5 * STAND_HEIGHT)
    return reward, fell | (self.steps >= EPISODE), fell, terms


def log_density(z, log_std):
  return -0.5 * (z * z).sum(-1) - log_std.sum() - 0.5 * ACT_DIM * np.log(2 * np.pi)


def mlp(out_dim):
  hidden = (nn.Linear(OBS_DIM, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, HIDDEN), nn.ELU())
  return nn.Sequential(*hidden, nn.Linear(HIDDEN, out_dim))


class ActorCritic(nn.Module):
  def __init__(self):
    super().__init__()
    self.actor, self.critic = mlp(ACT_DIM), mlp(1)
    self.log_std = nn.Parameter(torch.full((ACT_DIM,), LOG_STD))
    self.register_buffer("mean", torch.zeros(OBS_DIM))
    self.register_buffer("var", torch.ones(OBS_DIM))
    self.register_buffer("count", torch.full((), 1e-4))

  @torch.no_grad()
  def absorb(self, obs):  # parallel running-statistics update over the batch
    n, delta = obs.shape[0], obs.mean(0) - self.mean
    total = self.count + n
    var = self.var * self.count + obs.var(0, correction=0) * n
    self.var.copy_((var + delta**2 * self.count * n / total) / total)
    self.mean.add_(delta * n / total)
    self.count.add_(n)

  def forward(self, obs):
    obs = (obs - self.mean) / (self.var.sqrt() + 1e-5)
    return self.actor(obs), self.critic(obs).squeeze(-1)


@torch.no_grad()
def rollout(actor, env):
  def policy(obs):
    return tuple(x.numpy() for x in actor(torch.as_tensor(obs)))

  shapes = dict(obs=(OBS_DIM,), act=(ACT_DIM,), logp=(), val=(), rew=(), alive=())
  buf = {k: np.empty((HORIZON, env.num_envs, *v), np.float32) for k, v in shapes.items()}
  means, falls, episodes = [], 0, 0
  obs = env.obs()
  for t in range(HORIZON):
    mean, val = policy(obs)
    log_std = actor.log_std.numpy()
    noise = env.rng.standard_normal(mean.shape, np.float32)
    act = mean + np.exp(log_std) * noise
    reward, done, fell, terms = env.step(act)
    next_obs = env.obs()
    timeout = done & ~fell
    if timeout.any():  # bootstrap the truncated episodes
      reward[timeout] += GAMMA * policy(next_obs[timeout])[1]
    for k, v in dict(
      obs=obs, act=act, logp=log_density(noise, log_std), val=val, rew=reward, alive=~done
    ).items():
      buf[k][t] = v
    means.append([terms[k].mean() for k in REWARD])
    ids = np.flatnonzero(done)
    if ids.size:
      episodes, falls = episodes + ids.size, falls + int(fell[ids].sum())
      env.reset(ids)
      next_obs = env.obs()
    obs = next_obs
  batch = {k: torch.as_tensor(v) for k, v in buf.items()}
  batch["last_val"] = torch.as_tensor(policy(obs)[1])
  stats = dict(zip(REWARD, np.mean(means, 0), strict=True))
  stats["falls"] = falls / max(episodes, 1)
  return batch, stats


def gae(batch):
  vals = torch.cat([batch["val"], batch["last_val"][None]])
  adv, carry = torch.zeros_like(batch["rew"]), 0.0
  for t in reversed(range(HORIZON)):
    alive = batch["alive"][t]
    delta = batch["rew"][t] + GAMMA * alive * vals[t + 1] - vals[t]
    adv[t] = carry = delta + GAMMA * LAMBDA * alive * carry
  return adv, adv + batch["val"]


def update(net, opt, batch, adv, ret):
  obs, act = batch["obs"].reshape(-1, OBS_DIM), batch["act"].reshape(-1, ACT_DIM)
  logp_old, adv, ret = batch["logp"].reshape(-1), adv.reshape(-1), ret.reshape(-1)
  adv = (adv - adv.mean()) / (adv.std() + 1e-8)
  for _ in range(EPOCHS):
    for i in torch.randperm(obs.shape[0]).chunk(MINIBATCHES):
      mean, val = net(obs[i])
      logp = log_density((act[i] - mean) / net.log_std.exp(), net.log_std)
      ratio = (logp - logp_old[i]).exp()
      surrogate = torch.min(ratio * adv[i], ratio.clamp(1 - CLIP, 1 + CLIP) * adv[i])
      loss = -surrogate.mean() + 0.5 * (val - ret[i]).pow(2).mean() - ENT_COEF * net.log_std.sum()
      opt.zero_grad(set_to_none=True)
      loss.backward()
      nn.utils.clip_grad_norm_(net.parameters(), 1.0)
      opt.step()
  net.absorb(obs)  # after the epochs: the batch was collected under the old statistics


def train(num_envs=NUM_ENVS, iterations=ITERS, out=OUT, timestep=TIMESTEP):
  torch.manual_seed(SEED)
  env = Duck(num_envs, seed=SEED, timestep=timestep)
  net = ActorCritic()
  opt = torch.optim.Adam(net.parameters(), LR)
  start = time.perf_counter()

  def save():
    torch.save({"model": net.state_dict(), "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "timestep": timestep}, out)

  try:
    for it in range(iterations):
      opt.param_groups[0]["lr"] = float(np.interp(it, (0, LR_TO), (LR, LR_END)))
      batch, stats = rollout(net, env)
      update(net, opt, batch, *gae(batch))
      dt = time.perf_counter() - start
      reward = sum(REWARD[k] * stats[k] for k in REWARD)
      steps_per_s = num_envs * HORIZON * (it + 1) / dt
      tail = f"{steps_per_s / 1e3:4.0f}k steps/s  reward {reward:6.2f}  "
      tail += "  ".join(f"{k} {stats[k]:.2f}" for k in ("track", "turn", "falls"))
      progress(f"{it + 1:5d}/{iterations}", (it + 1) / iterations, dt, tail)
      if (it + 1) % 50 == 0 or it + 1 == iterations:
        print(" " * 12 + "  ".join(f"{k} {REWARD[k] * stats[k]:6.2f}" for k in REWARD))
      if (it + 1) % 100 == 0 or it + 1 == iterations:
        save()
  except KeyboardInterrupt:
    print()
  save()
  print(
    f"saved {out} after {int(time.perf_counter() - start) // 60}m{int(time.perf_counter() - start) % 60:02d}s"
  )


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--num-envs", type=int, default=NUM_ENVS)
  ap.add_argument("--iterations", type=int, default=ITERS)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  ap.add_argument("--timestep", type=float, default=TIMESTEP, help="physics timestep, s")
  args = ap.parse_args()
  train(args.num_envs, args.iterations, args.out, args.timestep)
