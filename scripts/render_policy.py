# SPDX-License-Identifier: Apache-2.0

"""Render a trained Microduck policy to an mp4, offscreen — no display needed.

  uv run scripts/render_policy.py --policy microduck_policy.pt --seconds 6 --command 0.4,0,0

Feeds a fixed velocity command, steps the batch, and pipes rgb frames to ffmpeg. `--substeps 4`
renders every physics substep instead of every action, which is what makes a real slow motion:
more frames of the trajectory, not the same frames re-timed.
"""

import argparse
import pathlib
import subprocess
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import train_ppo as T  # the repo's own config and network definition


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", type=pathlib.Path, default=T.OUT)
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("microduck_walk.mp4"))
  ap.add_argument("--seconds", type=float, default=6.0)
  ap.add_argument("--fps", type=int, default=50, help="control rate; the action is applied this often")
  ap.add_argument("--substeps", type=int, default=1, help="frames rendered per control step")
  ap.add_argument("--slowmo", type=float, default=1.0, help="playback slowdown factor")
  ap.add_argument("--command", default="0.4,0,0", help="forward m/s, sideways m/s, yaw rad/s")
  ap.add_argument("--distance", type=float, default=None)
  ap.add_argument(
    "--model",
    type=pathlib.Path,
    default=pathlib.Path("microduck/scene_walk.xml"),
    help="the full tree by default, so the shell renders",
  )
  ap.add_argument("--width", type=int, default=640)
  ap.add_argument("--height", type=int, default=480)
  args = ap.parse_args()

  checkpoint = torch.load(args.policy, map_location="cpu")
  timestep = checkpoint.get("timestep", T.TIMESTEP)
  net = T.ActorCritic(checkpoint.get("obs_dim", T.OBS_DIM))
  net.load_state_dict(checkpoint["model"])
  net.eval()

  task = checkpoint.get("task", "velocity")
  model = (
    T.build_ball_model(timestep, args.model or T.BALL_XML_FULL)
    if task == "ball-balance"
    else T.build_model(timestep, args.model or T.XML)
  )
  # The offscreen framebuffer defaults to 640x480 whatever the model asks for; lift it for HD.
  model.vis.global_.offwidth = max(args.width, model.vis.global_.offwidth)
  model.vis.global_.offheight = max(args.height, model.vis.global_.offheight)
  data = mujoco.MjData(model)

  # obs/step bookkeeping; the rendered state comes from here
  env = (
    T.BallBalance(1, timestep=timestep, full=True) if task == "ball-balance" else T.Duck(1, timestep=timestep)
  )
  if task == "ball-balance":
    data.qpos[:], data.qvel[:] = env.qpos[0], env.qvel[0]
  else:
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
  env.command[:], env.until[:] = [float(v) for v in args.command.split(",")], 1e9

  camera = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, camera)
  args.distance = args.distance if args.distance is not None else (1.0 if task == "ball-balance" else 0.8)
  camera.distance, camera.elevation, camera.azimuth = args.distance, -10, 130
  camera.lookat[:] = (0.0, 0.0, 0.35) if task == "ball-balance" else (0.0, 0.0, 0.12)
  option = mujoco.MjvOption()
  option.geomgroup[:] = [1, 1, 1, 0, 1, 1]  # draw the shell (group 2), hide the green collision geoms (3)
  renderer = mujoco.Renderer(model, height=args.height, width=args.width)

  frames = []

  def shoot():
    data.qpos[:], data.qvel[:] = env.qpos[0], env.qvel[0]
    camera.lookat[:] = data.qpos[:3] + [0, 0, 0.1]  # follow the duck, or it leaves the frame
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera, scene_option=option)
    frames.append(renderer.render())

  def act():
    with torch.no_grad():
      action = net(torch.as_tensor(env.obs()))[0].numpy()[0]
    env.until[:] = 1e9  # hold the command for the whole clip
    env.ctrl[:] = env.stand + T.ACTION_SCALE * action
    return action

  steps = int(args.seconds * args.fps)
  if args.substeps <= 1:
    for _ in range(steps):
      env.step(act()[None])
      shoot()
  else:  # step the batch itself, sub-step by sub-step, so every rendered frame is a real state
    chunk = env.decimation // args.substeps
    assert chunk * args.substeps == env.decimation, "substeps must divide the decimation"
    for _ in range(steps):
      action = act()
      for _ in range(args.substeps):
        env.batch.step(nstep=chunk)
        shoot()
      env.steps += 1
      env.clock = (env.clock + T.GAIT_HZ * T.CTRL_DT) % 1.0
      env.action = action[None].astype(np.float32)

  out_fps = args.fps * max(args.substeps, 1) / args.slowmo
  subprocess.run(
    [
      "ffmpeg",
      "-y",
      "-loglevel",
      "error",
      "-f",
      "rawvideo",
      "-pix_fmt",
      "rgb24",
      "-s",
      f"{args.width}x{args.height}",
      "-r",
      str(out_fps),
      "-i",
      "-",
      "-pix_fmt",
      "yuv420p",
      "-crf",
      "18",
      str(args.out),
    ],
    input=b"".join(f.tobytes() for f in frames),
    check=True,
  )
  print(
    f"wrote {args.out} ({len(frames)} frames at {out_fps:g} fps, "
    f"{len(frames) / out_fps:.1f} s of video for {args.seconds:g} s of simulation)"
  )


if __name__ == "__main__":
  main()
