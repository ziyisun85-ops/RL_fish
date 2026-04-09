from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


MODEL_DIR = Path(__file__).resolve().parent
SRC_XML = MODEL_DIR / "fish_2d_stl.xml"
FISH_XML = MODEL_DIR / "fish_robot_model.xml"
POOL_XML = MODEL_DIR / "pool_environment.xml"
SCENE_XML = MODEL_DIR / "fish_pool_scene.xml"

POOL_MATERIALS = {"pool_wall", "water"}
POOL_GEOMS = {
    "water_volume",
    "pool_floor",
    "pool_wall_front",
    "pool_wall_back",
    "pool_wall_left",
    "pool_wall_right",
}
SCENE_VISUALS = {"light", "top", "oblique"}
FISH_ROOT_BODIES = {"front_servo_body", "back_servo_body", "centre_compartment"}
DROP_SECTIONS_IN_POOL = {"default", "tendon", "equality", "actuator", "sensor"}


def indent_xml(elem: ET.Element) -> None:
    ET.indent(elem, space="  ")


def remove_named_children(section: ET.Element, tag: str, names: set[str]) -> None:
    for child in list(section):
        if child.tag == tag and child.get("name") in names:
            section.remove(child)


def remove_sections(root: ET.Element, tags: set[str]) -> None:
    for child in list(root):
        if child.tag in tags:
            root.remove(child)


def strip_option_attrs(root: ET.Element, attrs: set[str]) -> None:
    for option in root.findall("option"):
        for attr in attrs:
            option.attrib.pop(attr, None)


def build_fish_model(root: ET.Element) -> ET.Element:
    fish_root = deepcopy(root)
    fish_root.set("model", "eel_2d_robot")

    strip_option_attrs(fish_root, {"density", "viscosity"})

    for asset in fish_root.findall("asset"):
        remove_named_children(asset, "material", POOL_MATERIALS)

    for worldbody in fish_root.findall("worldbody"):
        remove_named_children(worldbody, "geom", POOL_GEOMS)
        remove_named_children(worldbody, "light", SCENE_VISUALS)
        remove_named_children(worldbody, "camera", SCENE_VISUALS)

    return fish_root


def build_pool_model(root: ET.Element) -> ET.Element:
    pool_root = deepcopy(root)
    pool_root.set("model", "eel_pool_environment")

    remove_sections(pool_root, DROP_SECTIONS_IN_POOL)

    for compiler in pool_root.findall("compiler"):
        pool_root.remove(compiler)

    for asset in pool_root.findall("asset"):
        for child in list(asset):
            if child.tag == "mesh":
                asset.remove(child)
            elif child.tag == "material" and child.get("name") not in POOL_MATERIALS:
                asset.remove(child)

    for worldbody in pool_root.findall("worldbody"):
        for child in list(worldbody):
            child_name = child.get("name")
            if child.tag == "body" and child_name in FISH_ROOT_BODIES:
                worldbody.remove(child)
            elif child.tag == "geom" and child_name not in POOL_GEOMS:
                worldbody.remove(child)
            elif child.tag == "light" and child_name not in SCENE_VISUALS:
                worldbody.remove(child)
            elif child.tag == "camera" and child_name not in SCENE_VISUALS:
                worldbody.remove(child)

    return pool_root


def build_scene_model() -> ET.Element:
    scene_root = ET.Element("mujoco", {"model": "eel_pool_scene"})
    ET.SubElement(scene_root, "include", {"file": FISH_XML.name})
    ET.SubElement(scene_root, "include", {"file": POOL_XML.name})
    return scene_root


def write_xml(path: Path, root: ET.Element) -> None:
    indent_xml(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=False)
    path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")


def main() -> None:
    if not SRC_XML.exists():
        raise FileNotFoundError(f"missing source MJCF: {SRC_XML}")

    root = ET.parse(SRC_XML).getroot()

    fish_root = build_fish_model(root)
    pool_root = build_pool_model(root)
    scene_root = build_scene_model()

    write_xml(FISH_XML, fish_root)
    write_xml(POOL_XML, pool_root)
    write_xml(SCENE_XML, scene_root)

    print(f"Wrote {FISH_XML}")
    print(f"Wrote {POOL_XML}")
    print(f"Wrote {SCENE_XML}")


if __name__ == "__main__":
    main()
