# SPDX-License-Identifier: Apache-2.0

"""Assert the vendored Microduck model steps identically to the upstream one.

Stripping visual meshes must not move a single contact. Steps both models from the STAND keyframe
under the same control sweep and compares qpos/qvel trajectory ends.

  uv run scripts/check_vendored.py --src ~/microduck_rl
"""

import argparse
import pathlib

import mujoco
import numpy as np


def rollout(xml, steps=200):
  model = mujoco.MjModel.from_xml_path(str(xml))
  data = mujoco.MjData(model)
  key = model.key("STAND")
  mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
  t = np.arange(steps) * model.opt.timestep
  for i in range(steps):
    data.ctrl[:] = key.ctrl + 0.15 * np.sin(2.0 * np.pi * 1.5 * t[i]) * np.linspace(-1, 1, model.nu)
    mujoco.mj_step(model, data)
  return model, data


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", type=pathlib.Path, default=pathlib.Path.home() / "microduck_rl")
  ap.add_argument(
    "--vendored",
    type=pathlib.Path,
    nargs="+",
    default=[pathlib.Path("microduck/scene_walk.xml"), pathlib.Path("microduck/lean/scene_walk.xml")],
  )
  args = ap.parse_args()

  upstream = args.src / "src/mjlab_microduck/robot/microduck/scene_walk.xml"
  a, da = rollout(upstream)
  for path in args.vendored:
    b, db = rollout(path)
    assert (a.nq, a.nv, a.nu) == (b.nq, b.nv, b.nu), (
      f"{path}: model sizes differ: {a.nq, a.nv, a.nu} vs {b.nq, b.nv, b.nu}"
    )
    assert a.ngeom >= b.ngeom, f"{path}: more geoms than upstream"
    for field in ("qpos", "qvel", "ctrl"):
      x, y = getattr(da, field), getattr(db, field)
      assert np.allclose(x, y), f"{path}: {field} diverged by {np.abs(x - y).max():.3g}"
    print(
      f"ok {path}: {a.ngeom} geoms upstream -> {b.ngeom}, "
      f"qpos max |diff| {np.abs(da.qpos - db.qpos).max():.3g} over 200 steps"
    )


if __name__ == "__main__":
  main()
