# SPDX-License-Identifier: Apache-2.0

"""Copy the Microduck MJCF out of a microduck_rl checkout.

By default everything is kept, so renders show a complete robot. `--strip-visual` drops the 75
visual mesh geoms (23 MB of STLs, contype=0, conaffinity=0) and keeps only the four meshes that
collide — a 2.9 MB tree for boxes that only train. Physics is identical either way; renders of the
stripped model show the collision group instead of the shell.

  uv run scripts/vendor_mjcf.py --src ~/microduck_rl [--strip-visual] [--dst microduck]
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
  ap.add_argument("--strip-visual", action="store_true", help="drop the visual mesh geoms (2.9 MB tree)")
  args = ap.parse_args()
  src = args.src / "src/mjlab_microduck/robot/microduck"
  dst, assets = args.dst, args.dst / "assets"
  (dst / "assets").mkdir(parents=True, exist_ok=True)

  parsed = ET.parse(src / "robot_walk.xml")
  if args.strip_visual:
    used = strip_visual(parsed.getroot())
    for asset in list(parsed.getroot().find("asset").findall("mesh")):
      name = pathlib.Path(asset.get("file")).stem
      if name not in used:
        parsed.getroot().find("asset").remove(asset)
  else:
    used = {pathlib.Path(a.get("file")).stem for a in parsed.getroot().find("asset").findall("mesh")}
  parsed.write(dst / "robot_walk.xml", encoding="utf-8", xml_declaration=False)

  scene = ET.parse(src / "scene_walk.xml")
  scene.write(dst / "scene_walk.xml", encoding="utf-8", xml_declaration=False)

  for name in sorted(used):
    shutil.copy(src / "assets" / f"{name}.stl", assets / f"{name}.stl")
  print(f"kept {len(used)} meshes" + (f": {', '.join(sorted(used))}" if args.strip_visual else ""))
  print(f"wrote {dst}/ with {len(XML)} xml files and {len(used)} stl files")


if __name__ == "__main__":
  main()
