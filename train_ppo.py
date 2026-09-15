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
import pathlib
import time

import mujoco
import numpy as np
import torch
from mjbatch import Batch
from torch import nn

# The lean tree drops the visual mesh geoms: same physics, 2x the substep rate. Rendering uses the
# full tree (scripts/render_policy.py --model microduck/scene_walk.xml).
XML = pathlib.Path(__file__).parent / "microduck/lean/scene_walk.xml"

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

# Ball balance — a port of Motphys MotrixLab's microduck-ball-balance task: the duck stands on a
# basketball and holds it under its feet. Ball numbers come from that task's basketball.xml.
BALL_RADIUS, BALL_MASS = 0.14, 0.45
BALL_XML = pathlib.Path(__file__).parent / "microduck/lean/scene_walk.xml"
BALL_XML_FULL = pathlib.Path(__file__).parent / "microduck/scene_walk.xml"
BASE_SPAWN_Z = 0.12 + 2 * BALL_RADIUS  # feet rest at the ball's apex
BALL_SCALE = 0.5  # upstream action_scale for this task
REWARD_BALL = {
  "alive": 1.0,
  "upright": 4.0,
  "height": 1.5,
  "ball": 3.0,
  "pose": 0.5,
  "rate": -0.5,
  "limits": -5.0,
  "contacts": -0.2,
}
OBS_DIM_BALL = 4 * 3 + 3 * 14
GAMMA, LAMBDA, CLIP, ENT_COEF = 0.99, 0.95, 0.2, 0.005
LR, LR_END, LR_TO = 1e-3, 5e-4, 1000
EPOCHS, MINIBATCHES, LOG_STD, HIDDEN, SEED = 5, 4, np.log(0.5), 128, 0
OUT = pathlib.Path(__file__).parent / "microduck_policy.pt"
OUT_BALL = pathlib.Path(__file__).parent / "microduck_ball_policy.pt"


def _set_actuators(spec, timestep):
  spec.option.timestep = timestep
  for actuator in spec.actuators:
    actuator.set_to_position(kp=KP, kv=KD)
    actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    actuator.forcerange = [-FORCE_LIMIT, FORCE_LIMIT]
  return spec


def build_model(timestep=TIMESTEP, path=XML):
  return _set_actuators(mujoco.MjSpec.from_file(str(path)), timestep).compile()


def build_ball_model(timestep=TIMESTEP, path=BALL_XML):
  """Robot, floor and the balance ball. Keyframes go: the ball's freejoint changes nq."""
  spec = _set_actuators(mujoco.MjSpec.from_file(str(path)), timestep)
  for key in list(spec.keys):
    spec.delete(key)
  ball = spec.worldbody.add_body(name="ball", pos=[0.0, 0.0, BALL_RADIUS])
  ball.add_freejoint(name="ball_free")
  ball.add_geom(
    name="ball_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[BALL_RADIUS],
    mass=BALL_MASS,
    friction=[1.0, 0.005, 0.0001],
    condim=3,
    rgba=[1.0, 0.55, 0.0, 1.0],
  )
  return spec.compile()


def stand_pose(timestep=TIMESTEP):
  """The STAND keyframe's joint angles and actuator targets, from the walking model."""
  key = build_model(timestep).key("STAND")
  return key.qpos[7 : 7 + ACT_DIM].copy(), key.ctrl.copy()


def progress(head, done, elapsed, tail, end=False):
  bar = "━" * round(20 * done) + "─" * (20 - round(20 * done))
  left = f"{int(elapsed / done - elapsed) // 60}:{int(elapsed / done - elapsed) % 60:02d}" if done else "-:--"
  print(
    f"\r{head} {bar} {int(elapsed) // 60}:{int(elapsed) % 60:02d}, {left} left  {tail}\x1b[K",
    end="\n" if end else "",
    flush=True,
  )


class Duck:
  obs_dim = OBS_DIM
  rewards = REWARD

  def __init__(self, num_envs, seed=0, timestep=TIMESTEP, model=None, stand=None):
    self.batch = batch = Batch(model if model is not None else build_model(timestep), num_envs)
    self.num_envs = num_envs
    self.decimation = round(CTRL_DT / timestep)
    self.model = batch.model
    self.qpos, self.qvel, self.ctrl = (batch.bind(f) for f in ("qpos", "qvel", "ctrl"))
    self.xmat, self.site, self.force = (batch.bind(f) for f in ("xmat", "site_xpos", "actuator_force"))
    self.trunk = self.model.body("trunk_base").id
    self.feet = [self.model.site(s).id for s in ("left_foot", "right_foot")]
    self.rot = self.xmat[:, self.trunk].reshape(-1, 3, 3)  # live view: world <- body
    self.foot = self.site[:, self.feet]  # (N, 2, 3)
    self.stand_qpos, self.stand = stand if stand is not None else stand_pose(timestep)
    self.keyframe = -1 if stand is not None else self.model.key("STAND").id
    self.joints = np.arange(7, 7 + ACT_DIM)  # free joint is qpos[0:7]
    self.dofs = np.arange(6, 6 + ACT_DIM)
    self.lower, self.upper = (
      0.95 * self.model.jnt_range[1 : 1 + ACT_DIM, 0],
      (0.95 * self.model.jnt_range[1 : 1 + ACT_DIM, 1]),
    )
    self.rng = np.random.default_rng(seed)
    self.steps, self.clock = np.zeros(num_envs, np.int64), np.zeros(num_envs)
    self.action = np.zeros((num_envs, ACT_DIM), np.float32)
    self.command, self.until = np.zeros((num_envs, 3)), np.zeros(num_envs)
    self.reset(np.arange(num_envs))

  def reset(self, ids):
    n = ids.size
    self.batch.reset(ids, keyframe=self.keyframe)  # writes must follow the reset
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


class BallBalance(Duck):
  """The duck stands on a basketball and keeps it under its feet.

  Port of Motphys MotrixLab's microduck-ball-balance reward and termination set: alive, upright,
  base height at the balanced height, ball under the feet, default posture, action rate, joint
  limits, undesired contacts. Episode ends on a low base, a real tilt, the ball rolling out, a
  joint far from its default, or a joint spinning faster than 100 rad/s.
  """

  obs_dim = OBS_DIM_BALL
  rewards = REWARD_BALL
  BALL_QPOS, BALL_QVEL, EPISODE_BALL = 21, 20, 500  # 10 s at 50 Hz

  def __init__(self, num_envs, seed=0, timestep=TIMESTEP, full=False):
    super().__init__(
      num_envs,
      seed,
      timestep,
      model=build_ball_model(timestep, BALL_XML_FULL if full else BALL_XML),
      stand=stand_pose(timestep),
    )
    self.ball = self.model.body("ball").id
    self.cfrc = self.batch.bind("cfrc_ext")
    feet = {self.model.body(f"ankle_{side}").id for side in ("left", "right")}
    self.bad = [i for i in range(1, self.model.nbody) if i != self.ball and i not in feet]

  def reset(self, ids):
    n = ids.size
    self.batch.reset(ids)  # writes must follow the reset
    self.qpos[ids] = 0.0
    self.qpos[ids, 2] = BASE_SPAWN_Z + self.rng.uniform(-0.005, 0.005, n)
    self.qpos[ids, 3] = 1.0
    self.qpos[np.ix_(ids, self.joints)] = self.stand_qpos + self.rng.uniform(-0.05, 0.05, (n, ACT_DIM))
    self.qpos[ids, self.BALL_QPOS : self.BALL_QPOS + 2] = self.rng.uniform(-0.01, 0.01, (n, 2))
    self.qpos[ids, self.BALL_QPOS + 2] = BALL_RADIUS
    self.qpos[ids, self.BALL_QPOS + 3] = 1.0
    self.qvel[ids] = self.rng.uniform(-0.05, 0.05, (n, self.model.nv))
    self.steps[ids], self.action[ids] = 0, 0.0
    self.batch.forward(ids)  # derived fields read by obs()

  def obs(self):
    rot = self.rot
    to_ball = np.einsum(
      "nji,nj->ni", rot, self.qpos[:, self.BALL_QPOS : self.BALL_QPOS + 3] - self.qpos[:, :3]
    )
    ball_vel = np.einsum("nji,nj->ni", rot, self.qvel[:, self.BALL_QVEL : self.BALL_QVEL + 3])
    cols = (
      -rot[:, 2],  # gravity in the base frame
      0.25 * self.qvel[:, 3:6],
      5.0 * to_ball,
      0.5 * ball_vel,
      self.qpos[:, self.joints] - self.stand_qpos,
      0.05 * self.qvel[:, self.dofs],
      self.action,
    )
    return np.concatenate(cols, 1, dtype=np.float32)

  def step(self, action):
    self.ctrl[:] = self.stand + BALL_SCALE * action
    self.batch.step(nstep=self.decimation)
    self.steps += 1
    up, q, base_z = -self.rot[:, 2], self.qpos[:, self.joints], self.qpos[:, 2]
    ball, feet_mid = self.qpos[:, self.BALL_QPOS : self.BALL_QPOS + 3], self.foot[:, :, :2].mean(1)
    d_xy = np.linalg.norm(ball[:, :2] - feet_mid, axis=1)
    tilt_sq = up[:, 0] ** 2 + up[:, 1] ** 2
    centred = np.exp(-(tilt_sq + (up[:, 2] + 1.0) ** 2) / 0.2**2)  # upstream's upright error
    lo, hi = self.model.jnt_range[1 : 1 + ACT_DIM, 0], self.model.jnt_range[1 : 1 + ACT_DIM, 1]
    frac = np.abs(2.0 * (q - lo) / (hi - lo) - 1.0)  # 0 mid-range, 1 at a limit
    terms = dict(
      alive=np.ones(self.num_envs, np.float32),
      upright=centred,
      height=np.exp(-(((base_z - BASE_SPAWN_Z) / 0.05) ** 2)),
      ball=np.exp(-((d_xy**2) / 0.05**2)),
      pose=np.exp(-np.sum((q - self.stand_qpos) ** 2, 1) / ACT_DIM / 0.5**2),
      rate=np.sum((action - self.action) ** 2, 1),
      limits=np.clip((frac - 0.9) / 0.1, 0.0, 5.0).sum(1),
      contacts=(np.linalg.norm(self.cfrc[:, self.bad, :3], axis=2) > 0.5).sum(1),
    )
    reward = np.asarray(sum(self.rewards[k] * v for k, v in terms.items()), np.float32)
    self.action = action.astype(np.float32)
    fell = (
      (base_z < 0.22)
      | (tilt_sq > 0.6**2)
      | (d_xy > 0.20)
      | (np.abs(q - self.stand_qpos).max(1) > 0.5)
      | (np.linalg.norm(self.qvel[:, self.dofs], axis=1) > 100.0)
    )
    return reward, fell | (self.steps >= self.EPISODE_BALL), fell, terms


TASKS = {"velocity": (Duck, OUT), "ball-balance": (BallBalance, OUT_BALL)}


def log_density(z, log_std):
  return -0.5 * (z * z).sum(-1) - log_std.sum() - 0.5 * ACT_DIM * np.log(2 * np.pi)


def mlp(out_dim, obs_dim):
  hidden = (nn.Linear(obs_dim, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, HIDDEN), nn.ELU())
  return nn.Sequential(*hidden, nn.Linear(HIDDEN, out_dim))


class ActorCritic(nn.Module):
  def __init__(self, obs_dim=OBS_DIM):
    super().__init__()
    self.obs_dim = obs_dim
    self.actor, self.critic = mlp(ACT_DIM, obs_dim), mlp(1, obs_dim)
    self.log_std = nn.Parameter(torch.full((ACT_DIM,), LOG_STD))
    self.register_buffer("mean", torch.zeros(obs_dim))
    self.register_buffer("var", torch.ones(obs_dim))
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

  shapes = dict(obs=(env.obs_dim,), act=(ACT_DIM,), logp=(), val=(), rew=(), alive=())
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
    means.append([terms[k].mean() for k in env.rewards])
    ids = np.flatnonzero(done)
    if ids.size:
      episodes, falls = episodes + ids.size, falls + int(fell[ids].sum())
      env.reset(ids)
      next_obs = env.obs()
    obs = next_obs
  batch = {k: torch.as_tensor(v) for k, v in buf.items()}
  batch["last_val"] = torch.as_tensor(policy(obs)[1])
  stats = dict(zip(env.rewards, np.mean(means, 0), strict=True))
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
  obs, act = batch["obs"].reshape(-1, net.obs_dim), batch["act"].reshape(-1, ACT_DIM)
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


def train(task="velocity", num_envs=NUM_ENVS, iterations=ITERS, out=None, timestep=TIMESTEP):
  env = TASKS[task][0](num_envs, seed=SEED, timestep=timestep)
  out = out or TASKS[task][1]
  torch.manual_seed(SEED)
  net = ActorCritic(env.obs_dim)
  opt = torch.optim.Adam(net.parameters(), LR)
  start = time.perf_counter()

  def save():
    torch.save(
      {
        "model": net.state_dict(),
        "obs_dim": env.obs_dim,
        "act_dim": ACT_DIM,
        "timestep": timestep,
        "task": task,
      },
      out,
    )

  try:
    for it in range(iterations):
      opt.param_groups[0]["lr"] = float(np.interp(it, (0, LR_TO), (LR, LR_END)))
      batch, stats = rollout(net, env)
      update(net, opt, batch, *gae(batch))
      dt = time.perf_counter() - start
      reward = sum(env.rewards[k] * stats[k] for k in env.rewards)
      steps_per_s = num_envs * HORIZON * (it + 1) / dt
      tail = f"{steps_per_s / 1e3:4.0f}k steps/s  reward {reward:6.2f}  "
      tail += "  ".join(f"{k} {stats[k]:.2f}" for k in list(env.rewards)[:5])
      tail += f"  falls {stats['falls']:.2f}"
      progress(f"{it + 1:5d}/{iterations}", (it + 1) / iterations, dt, tail)
      if (it + 1) % 50 == 0 or it + 1 == iterations:
        print(" " * 12 + "  ".join(f"{k} {env.rewards[k] * stats[k]:6.2f}" for k in env.rewards))
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
  ap.add_argument("--task", choices=("velocity", "ball-balance"), default="velocity")
  ap.add_argument("--num-envs", type=int, default=NUM_ENVS)
  ap.add_argument("--iterations", type=int, default=ITERS)
  ap.add_argument("--out", type=pathlib.Path, default=None)
  ap.add_argument("--timestep", type=float, default=TIMESTEP, help="physics timestep, s")
  args = ap.parse_args()
  train(args.task, args.num_envs, args.iterations, args.out, args.timestep)
