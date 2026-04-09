from __future__ import annotations

from pathlib import Path


OUT_PATH = Path(__file__).with_name("fish_2d_tendon.xml")


ACTIVE_SPACING = 0.0365
ACTIVE_HALF_LENGTH = ACTIVE_SPACING / 2.0
HEAD_HALF_LENGTH = 0.0300
JOINT_TAIL_HALF_LENGTH = 0.0225

WIRE_RADIUS = 0.0375

CENTRE_HALF_LENGTH = 0.1750
CENTRE_HALF_WIDTH = 0.0550
CENTRE_HALF_HEIGHT = 0.0450

ACTIVE_FRONT_RADIUS = 0.0280
ACTIVE_BACK_RADIUS = 0.0260
HEAD_RADIUS = 0.0340
JOINT_TAIL_RADIUS = 0.0220

TAIL_SEGMENTS = [
    ("tail_seg1", 0.0225, 0.0180),
    ("tail_seg2", 0.0210, 0.0160),
    ("tail_seg3", 0.0180, 0.0140),
    ("tail_seg4", 0.0150, 0.0120),
]

ACTIVE_DENSITY = 220.0
TAIL_DENSITY = 140.0
CENTRE_DENSITY = 190.0

ACTIVE_STIFFNESS = 1.041
ACTIVE_DAMPING = 0.05
ACTIVE_RANGE_DEG = 15.0

TAIL_STIFFNESS = 0.18
TAIL_DAMPING = 0.02
TAIL_RANGE_DEG = 25.0

SITE_SIZE = 0.004
SITE_RGBA = "0.90 0.20 0.20 1"

ROOT_Z = 0.0


def f(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def vec3(x: float, y: float, z: float) -> str:
    return f"{f(x)} {f(y)} {f(z)}"


def indent(lines: list[str], level: int) -> list[str]:
    prefix = "  " * level
    return [f"{prefix}{line}" if line else "" for line in lines]


def make_site_pair(prefix: str, x_offset: float = 0.0) -> list[str]:
    return [
        f'<site name="{prefix}_left" pos="{vec3(x_offset, WIRE_RADIUS, 0.0)}"/>',
        f'<site name="{prefix}_right" pos="{vec3(x_offset, -WIRE_RADIUS, 0.0)}"/>',
    ]


def active_body_open(
    name: str,
    body_pos_x: float,
    joint_pos_x: float,
    half_length: float,
    radius: float,
    density: float,
    rgba: str,
    site_x_offset: float = 0.0,
) -> list[str]:
    lines = [
        f'<body name="{name}" pos="{vec3(body_pos_x, 0.0, 0.0)}">',
        (
            f'  <joint name="{name}_yaw" class="active_hinge" '
            f'pos="{vec3(joint_pos_x, 0.0, 0.0)}"/>'
        ),
        (
            f'  <geom name="{name}_geom" type="capsule" '
            f'fromto="{vec3(-half_length, 0.0, 0.0)} {vec3(half_length, 0.0, 0.0)}" '
            f'size="{f(radius)}" density="{f(density)}" rgba="{rgba}"/>'
        ),
    ]
    lines.extend(f"  {line}" for line in make_site_pair(name, site_x_offset))
    return lines


def active_body_close() -> str:
    return "</body>"


def passive_tail_body_open(
    name: str,
    body_pos_x: float,
    joint_pos_x: float,
    half_length: float,
    radius: float,
    density: float,
    rgba: str,
) -> list[str]:
    return [
        f'<body name="{name}" pos="{vec3(body_pos_x, 0.0, 0.0)}">',
        (
            f'  <joint name="{name}_yaw" class="tail_hinge" '
            f'pos="{vec3(joint_pos_x, 0.0, 0.0)}"/>'
        ),
        (
            f'  <geom name="{name}_geom" type="capsule" '
            f'fromto="{vec3(-half_length, 0.0, 0.0)} {vec3(half_length, 0.0, 0.0)}" '
            f'size="{f(radius)}" density="{f(density)}" rgba="{rgba}"/>'
        ),
    ]


def build_front_chain() -> list[str]:
    lines: list[str] = []
    names = [f"front_v{i}" for i in range(1, 10)]
    for idx, name in enumerate(names):
        body_pos_x = CENTRE_HALF_LENGTH + ACTIVE_HALF_LENGTH if idx == 0 else ACTIVE_SPACING
        lines.extend(
            indent(
                active_body_open(
                name=name,
                body_pos_x=body_pos_x,
                joint_pos_x=-ACTIVE_HALF_LENGTH,
                half_length=ACTIVE_HALF_LENGTH,
                radius=ACTIVE_FRONT_RADIUS,
                density=ACTIVE_DENSITY,
                rgba="0.18 0.48 0.78 1",
                site_x_offset=0.0,
                ),
                idx,
            )
        )

    head_pos_x = ACTIVE_HALF_LENGTH + HEAD_HALF_LENGTH
    lines.extend(
        indent(
            active_body_open(
                name="head",
                body_pos_x=head_pos_x,
                joint_pos_x=-HEAD_HALF_LENGTH,
                half_length=HEAD_HALF_LENGTH,
                radius=HEAD_RADIUS,
                density=ACTIVE_DENSITY,
                rgba="0.10 0.35 0.62 1",
                site_x_offset=-0.018,
            ),
            len(names),
        )
    )

    for depth in reversed(range(len(names) + 1)):
        lines.extend(indent([active_body_close()], depth))
    return lines


def build_tail_chain() -> list[str]:
    lines: list[str] = []
    previous_half = JOINT_TAIL_HALF_LENGTH
    for idx, (name, half_length, radius) in enumerate(TAIL_SEGMENTS):
        body_pos_x = -(previous_half + half_length)
        lines.extend(
            indent(
                passive_tail_body_open(
                    name=name,
                    body_pos_x=body_pos_x,
                    joint_pos_x=half_length,
                    half_length=half_length,
                    radius=radius,
                    density=TAIL_DENSITY,
                    rgba="0.88 0.74 0.42 1",
                ),
                idx,
            )
        )
        previous_half = half_length

    for depth in reversed(range(len(TAIL_SEGMENTS))):
        lines.extend(indent(["</body>"], depth))
    return lines


def build_back_chain() -> list[str]:
    lines: list[str] = []
    names = [f"back_v{i}" for i in range(1, 10)]
    for idx, name in enumerate(names):
        body_pos_x = -(CENTRE_HALF_LENGTH + ACTIVE_HALF_LENGTH) if idx == 0 else -ACTIVE_SPACING
        lines.extend(
            indent(
                active_body_open(
                    name=name,
                    body_pos_x=body_pos_x,
                    joint_pos_x=ACTIVE_HALF_LENGTH,
                    half_length=ACTIVE_HALF_LENGTH,
                    radius=ACTIVE_BACK_RADIUS,
                    density=ACTIVE_DENSITY,
                    rgba="0.22 0.62 0.62 1",
                    site_x_offset=0.0,
                ),
                idx,
            )
        )

    joint_tail_pos_x = -(ACTIVE_HALF_LENGTH + JOINT_TAIL_HALF_LENGTH)
    lines.extend(
        indent(
            active_body_open(
                name="joint_tail",
                body_pos_x=joint_tail_pos_x,
                joint_pos_x=JOINT_TAIL_HALF_LENGTH,
                half_length=JOINT_TAIL_HALF_LENGTH,
                radius=JOINT_TAIL_RADIUS,
                density=ACTIVE_DENSITY,
                rgba="0.17 0.53 0.53 1",
                site_x_offset=0.015,
            ),
            len(names),
        )
    )
    lines.extend(indent(build_tail_chain(), len(names) + 1))

    for depth in reversed(range(len(names) + 1)):
        lines.extend(indent(["</body>"], depth))
    return lines


def build_tendons() -> list[str]:
    front_left = ['<spatial name="front_wire_left" width="0.0025" rgba="0.95 0.25 0.25 1">']
    front_left.append('  <site site="centre_front_left"/>')
    for idx in range(1, 10):
        front_left.append(f'  <site site="front_v{idx}_left"/>')
    front_left.append('  <site site="head_left"/>')
    front_left.append("</spatial>")

    front_right = ['<spatial name="front_wire_right" width="0.0025" rgba="0.85 0.18 0.18 1">']
    front_right.append('  <site site="centre_front_right"/>')
    for idx in range(1, 10):
        front_right.append(f'  <site site="front_v{idx}_right"/>')
    front_right.append('  <site site="head_right"/>')
    front_right.append("</spatial>")

    back_left = ['<spatial name="back_wire_left" width="0.0025" rgba="0.25 0.80 0.80 1">']
    back_left.append('  <site site="centre_back_left"/>')
    for idx in range(1, 10):
        back_left.append(f'  <site site="back_v{idx}_left"/>')
    back_left.append('  <site site="joint_tail_left"/>')
    back_left.append("</spatial>")

    back_right = ['<spatial name="back_wire_right" width="0.0025" rgba="0.18 0.68 0.68 1">']
    back_right.append('  <site site="centre_back_right"/>')
    for idx in range(1, 10):
        back_right.append(f'  <site site="back_v{idx}_right"/>')
    back_right.append('  <site site="joint_tail_right"/>')
    back_right.append("</spatial>")

    return ["<tendon>", *indent(front_left, 1), *indent(front_right, 1), *indent(back_left, 1), *indent(back_right, 1), "</tendon>"]


def build_actuators() -> list[str]:
    return [
        "<actuator>",
        '  <motor name="front_left_motor" class="tendon_motor" tendon="front_wire_left"/>',
        '  <motor name="front_right_motor" class="tendon_motor" tendon="front_wire_right"/>',
        '  <motor name="back_left_motor" class="tendon_motor" tendon="back_wire_left"/>',
        '  <motor name="back_right_motor" class="tendon_motor" tendon="back_wire_right"/>',
        "</actuator>",
    ]


def build_sensor_block() -> list[str]:
    return [
        "<sensor>",
        '  <jointpos name="root_x_pos_sensor" joint="root_x"/>',
        '  <jointpos name="root_y_pos_sensor" joint="root_y"/>',
        '  <jointpos name="root_yaw_pos_sensor" joint="root_yaw"/>',
        '  <jointvel name="root_x_vel_sensor" joint="root_x"/>',
        '  <jointvel name="root_y_vel_sensor" joint="root_y"/>',
        '  <jointvel name="root_yaw_vel_sensor" joint="root_yaw"/>',
        '  <framepos name="head_pos" objtype="xbody" objname="head"/>',
        '  <framepos name="joint_tail_pos" objtype="xbody" objname="joint_tail"/>',
        '  <framexaxis name="centre_xaxis" objtype="xbody" objname="centre_compartment"/>',
        "</sensor>",
    ]


def build_xml() -> str:
    lines: list[str] = [
        '<mujoco model="eel_2d_tendon">',
        "  <!-- Topology follows model/prompt_model.md; scale cues are adapted from model/fish.STEP (CAD units are millimeters). -->",
        '  <compiler angle="degree" autolimits="true" inertiafromgeom="true"/>',
        '  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast" cone="elliptic" iterations="100">',
        '    <flag contact="disable"/>',
        "  </option>",
        "  <default>",
        '    <geom type="capsule" contype="0" conaffinity="0" group="1"/>',
        '    <site type="sphere" size="0.004" rgba="0.90 0.20 0.20 1" group="3"/>',
        '    <default class="root_joint">',
        '      <joint limited="false" stiffness="0" armature="0.01" damping="0.4"/>',
        "    </default>",
        '    <default class="active_hinge">',
        (
            f'      <joint type="hinge" axis="0 0 1" limited="true" '
            f'range="-{f(ACTIVE_RANGE_DEG)} {f(ACTIVE_RANGE_DEG)}" '
            f'stiffness="{f(ACTIVE_STIFFNESS)}" damping="{f(ACTIVE_DAMPING)}" armature="0.0005"/>'
        ),
        "    </default>",
        '    <default class="tail_hinge">',
        (
            f'      <joint type="hinge" axis="0 0 1" limited="true" '
            f'range="-{f(TAIL_RANGE_DEG)} {f(TAIL_RANGE_DEG)}" '
            f'stiffness="{f(TAIL_STIFFNESS)}" damping="{f(TAIL_DAMPING)}" armature="0.0002"/>'
        ),
        "    </default>",
        '    <default class="tendon_motor">',
        '      <motor ctrllimited="true" ctrlrange="0 100" gear="1"/>',
        "    </default>",
        "  </default>",
        "  <worldbody>",
        '    <light name="light" directional="true" diffuse="0.8 0.8 0.8" pos="0 0 2.5"/>',
        '    <camera name="top" pos="0 0 2.2" xyaxes="1 0 0 0 1 0"/>',
        '    <camera name="oblique" pos="-1.2 -1.6 0.9" xyaxes="0.8 -0.6 0 0.2 0.3 0.93"/>',
        f'    <body name="centre_compartment" pos="{vec3(0.0, 0.0, ROOT_Z)}">',
        '      <joint name="root_x" class="root_joint" type="slide" axis="1 0 0"/>',
        '      <joint name="root_y" class="root_joint" type="slide" axis="0 1 0"/>',
        '      <joint name="root_yaw" class="root_joint" type="hinge" axis="0 0 1"/>',
        (
            f'      <geom name="centre_compartment_geom" type="box" '
            f'size="{vec3(CENTRE_HALF_LENGTH, CENTRE_HALF_WIDTH, CENTRE_HALF_HEIGHT)}" '
            f'density="{f(CENTRE_DENSITY)}" rgba="0.16 0.22 0.35 1"/>'
        ),
        '      <site name="centre_reference" pos="0 0 0" rgba="0 0 0 0"/>',
        f'      <site name="centre_front_left" pos="{vec3(CENTRE_HALF_LENGTH, WIRE_RADIUS, 0.0)}"/>',
        f'      <site name="centre_front_right" pos="{vec3(CENTRE_HALF_LENGTH, -WIRE_RADIUS, 0.0)}"/>',
        f'      <site name="centre_back_left" pos="{vec3(-CENTRE_HALF_LENGTH, WIRE_RADIUS, 0.0)}"/>',
        f'      <site name="centre_back_right" pos="{vec3(-CENTRE_HALF_LENGTH, -WIRE_RADIUS, 0.0)}"/>',
    ]

    lines.extend(indent(build_front_chain(), 3))
    lines.extend(indent(build_back_chain(), 3))

    lines.extend(
        [
            "    </body>",
            "  </worldbody>",
        ]
    )

    lines.extend(indent(build_tendons(), 1))
    lines.extend(indent(build_actuators(), 1))
    lines.extend(indent(build_sensor_block(), 1))
    lines.append("</mujoco>")
    return "\n".join(lines) + "\n"


def main() -> None:
    xml = build_xml()
    OUT_PATH.write_text(xml, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
