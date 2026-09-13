# SPDX-License-Identifier: Apache-2.0

"""Copy the Microduck MJCF out of a microduck_rl checkout.

Two trees are written: `microduck/` keeps everything (38 meshes, 21 MB) so renders show a complete
robot, and `microduck/lean/` drops the 75 visual mesh geoms (contype=0, conaffinity=0) and keeps the
four that collide. Physics is identical — visual geoms never touch contact — but the lean tree steps
**2x faster** (measured 119k vs 62k substeps/s at 4096 envs), so training uses it and rendering uses
the full one.

  uv run scripts/vendor_mjcf.py --src ~/microduck_rl [--dst microduck]
"""

import argparse
import pathlib
import shutil
import xml.etree.ElementTree as ET

XML = ("robot_walk.xml", "scene_walk.xml")


def strip_visual(root):
  """Drop visual mesh geoms, then return the mesh names still referenced."""
  for parent in root.iter():
    for geom in list(parent.findall("geom")):
      if geom.get("class") == "visual" or (geom.get("type") == "mesh" and geom.get("contype") == "0"):
        parent.remove(geom)
  return {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", type=pathlib.Path, default=pathlib.Path.home() / "microduck_rl")
  ap.add_argument("--dst", type=pathlib.Path, default=pathlib.Path("microduck"))

  args = ap.parse_args()
  src = args.src / "src/mjlab_microduck/robot/microduck"
  dst, assets = args.dst, args.dst / "assets"
  (dst / "assets").mkdir(parents=True, exist_ok=True)
  (dst / "lean" / "assets").mkdir(parents=True, exist_ok=True)

  parsed = ET.parse(src / "robot_walk.xml")
  full = {pathlib.Path(a.get("file")).stem for a in parsed.getroot().find("asset").findall("mesh")}
  parsed.write(dst / "robot_walk.xml", encoding="utf-8", xml_declaration=False)
  shutil.copy(src / "scene_walk.xml", dst / "scene_walk.xml")
  for name in sorted(full):
    shutil.copy(src / "assets" / f"{name}.stl", assets / f"{name}.stl")
  print(f"wrote {dst}/ with {len(XML)} xml files and {len(full)} stl files")

  lean = ET.parse(src / "robot_walk.xml")
  used = strip_visual(lean.getroot())
  for asset in list(lean.getroot().find("asset").findall("mesh")):
    if pathlib.Path(asset.get("file")).stem not in used:
      lean.getroot().find("asset").remove(asset)
  lean.write(dst / "lean" / "robot_walk.xml", encoding="utf-8", xml_declaration=False)
  shutil.copy(src / "scene_walk.xml", dst / "lean" / "scene_walk.xml")
  for name in sorted(used):
    shutil.copy(src / "assets" / f"{name}.stl", dst / "lean" / "assets" / f"{name}.stl")
  print(f"wrote {dst}/lean/ with {len(used)} stl files ({', '.join(sorted(used))})")


if __name__ == "__main__":
  main()
