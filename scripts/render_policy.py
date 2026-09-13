# SPDX-License-Identifier: Apache-2.0

"""Render a trained Microduck policy to an mp4, offscreen — no display needed.

  uv run scripts/render_policy.py --policy microduck_policy.pt --seconds 6 --command 0.4,0,0

Feeds a fixed velocity command, steps the single MuJoCo instance, and pipes rgb frames to ffmpeg.
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
  ap.add_argument("--fps", type=int, default=50)
  ap.add_argument("--command", default="0.4,0,0", help="forward m/s, sideways m/s, yaw rad/s")
  ap.add_argument("--width", type=int, default=960)
  ap.add_argument("--height", type=int, default=540)
  args = ap.parse_args()

  model = T.build_model()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
  net = T.ActorCritic()
  net.load_state_dict(torch.load(args.policy, map_location="cpu")["model"])
  net.eval()

  env = T.Duck(1)  # for obs()/step() bookkeeping only; the frame comes from `data`
  command = np.array([float(v) for v in args.command.split(",")], np.float32)
  env.command[:], env.until[:] = command, 1e9

  camera = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, camera)
  camera.distance, camera.elevation, camera.azimuth, camera.lookat[:] = 1.1, -12, 135, (0, 0, 0.12)
  renderer = mujoco.Renderer(model, height=args.height, width=args.width)

  frames = []
  for _ in range(int(args.seconds * args.fps)):
    with torch.no_grad():
      action = net(torch.as_tensor(env.obs()))[0].numpy()[0]
    env.until[:] = 1e9  # hold the command for the whole clip
    env.step(action[None])
    data.qpos[:], data.qvel[:] = env.qpos[0], env.qvel[0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    frames.append(renderer.render())

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
      str(args.fps),
      "-i",
      "-",
      "-pix_fmt",
      "yuv420p",
      str(args.out),
    ],
    input=b"".join(f.tobytes() for f in frames),
    check=True,
  )
  print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
  main()
