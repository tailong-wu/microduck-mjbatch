# SPDX-License-Identifier: Apache-2.0

"""Copy the Microduck MJCF out of a microduck_rl checkout, dropping what only the eye needs.

The upstream model carries 76 visual mesh geoms (23 MB of STLs) that have contype=0 and
conaffinity=0 — they cost compile time and repo weight but touch no contact. The four mesh geoms
that do collide (both soles, the leg shells, the power support) are kept, so the vendored model
steps identically.

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

  parsed = ET.parse(src / "robot_walk.xml")
  used = strip_visual(parsed.getroot())
  for asset in list(parsed.getroot().find("asset").findall("mesh")):
    name = pathlib.Path(asset.get("file")).stem
    if name not in used:
      parsed.getroot().find("asset").remove(asset)
  parsed.write(dst / "robot_walk.xml", encoding="utf-8", xml_declaration=False)

  scene = ET.parse(src / "scene_walk.xml")
  scene.write(dst / "scene_walk.xml", encoding="utf-8", xml_declaration=False)

  for name in sorted(used):
    shutil.copy(src / "assets" / f"{name}.stl", assets / f"{name}.stl")
  print(f"kept {len(used)} meshes: {', '.join(sorted(used))}")
  print(f"wrote {dst}/ with {len(XML)} xml files and {len(used)} stl files")


if __name__ == "__main__":
  main()
