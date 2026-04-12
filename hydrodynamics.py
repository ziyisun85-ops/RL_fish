from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class HydrodynamicsConfig:
    tangential_drag: float = 0.1
    normal_drag: float = 4.0
    head_normal_multiplier: float = 1.5
    tail_normal_multiplier: float = 1.2
    yaw_damping_linear: float = 0.05
    yaw_damping_quadratic: float = 0.01
    lateral_added_mass_fraction: float = 0.5
    yaw_added_inertia_fraction: float = 0.2
    max_segment_force: float = 8.0
    max_yaw_torque: float = 2.0
    water_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class HydrodynamicsDiagnostics:
    segment_count: int
    segment_yaw_moment: float
    damping_yaw_torque: float
    added_yaw_torque: float
    max_segment_force: float


@dataclass(frozen=True)
class _HydrodynamicSegment:
    body_id: int
    name: str
    mass: float


@dataclass
class _HydrodynamicsState:
    time: float
    normal_velocity_by_body: dict[int, float]
    yaw_rate: float


DEFAULT_HYDRODYNAMICS_CONFIG = HydrodynamicsConfig()

_ZERO_TORQUE = np.zeros(3, dtype=np.float64)
_ZERO_FORCE = np.zeros(3, dtype=np.float64)
_MODEL_SEGMENT_CACHE: dict[int, tuple[_HydrodynamicSegment, ...]] = {}
_MODEL_CENTRE_BODY_CACHE: dict[int, int] = {}
_MODEL_YAW_DOF_CACHE: dict[int, int | None] = {}
_STATE_CACHE: dict[int, _HydrodynamicsState] = {}
_LAST_DIAGNOSTICS: dict[int, HydrodynamicsDiagnostics] = {}


def apply_hydrodynamics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: HydrodynamicsConfig = DEFAULT_HYDRODYNAMICS_CONFIG,
) -> None:
    """Apply distributed RFT-style hydrodynamic forces to the current MuJoCo state."""
    segments = _get_hydrodynamic_segments(model)
    centre_body_id = _get_centre_body_id(model, segments)
    reference_position = np.asarray(data.xpos[centre_body_id], dtype=np.float64)
    water_velocity = np.asarray(config.water_velocity, dtype=np.float64)
    current_time = float(data.time)
    previous_state = _STATE_CACHE.get(id(data))
    valid_previous_state = previous_state is not None and current_time > previous_state.time
    dt = current_time - previous_state.time if valid_previous_state else float(model.opt.timestep)
    dt = max(dt, 1e-8)
    yaw_rate = _get_yaw_rate(model, data, centre_body_id)
    yaw_accel = (yaw_rate - previous_state.yaw_rate) / dt if valid_previous_state else 0.0

    data.qfrc_applied[:] = 0.0

    next_normal_velocity_by_body: dict[int, float] = {}
    segment_yaw_moment = 0.0
    max_segment_force = 0.0
    reference_x = float(reference_position[0])
    reference_y = float(reference_position[1])
    water_vx = float(water_velocity[0])
    water_vy = float(water_velocity[1])
    water_vz = float(water_velocity[2])
    velocity = np.empty(6, dtype=np.float64)
    yaw_torque_vector = np.zeros(3, dtype=np.float64)

    for segment in segments:
        body_id = segment.body_id
        position = data.xpos[body_id]
        tangential_axis, normal_axis = _body_planar_axes(data, body_id)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        linear_velocity = velocity[3:]

        relative_vx = float(linear_velocity[0] - water_vx)
        relative_vy = float(linear_velocity[1] - water_vy)
        relative_vz = float(linear_velocity[2] - water_vz)
        tangential_speed = (
            relative_vx * float(tangential_axis[0])
            + relative_vy * float(tangential_axis[1])
            + relative_vz * float(tangential_axis[2])
        )
        normal_speed = (
            relative_vx * float(normal_axis[0])
            + relative_vy * float(normal_axis[1])
            + relative_vz * float(normal_axis[2])
        )
        next_normal_velocity_by_body[body_id] = normal_speed

        tangential_force = -float(config.tangential_drag) * tangential_speed * tangential_axis
        normal_drag = float(config.normal_drag) * _normal_multiplier(segment.name, config)
        normal_force = -normal_drag * abs(normal_speed) * normal_speed * normal_axis

        if valid_previous_state:
            previous_normal_speed = previous_state.normal_velocity_by_body.get(body_id, normal_speed)
            normal_accel = (normal_speed - previous_normal_speed) / dt
        else:
            normal_accel = 0.0
        added_mass = float(config.lateral_added_mass_fraction) * segment.mass
        added_mass_force = -added_mass * normal_accel * normal_axis

        segment_force = tangential_force + normal_force + added_mass_force
        segment_force = _limit_vector(segment_force, float(config.max_segment_force))
        force_norm = _vector_norm(segment_force)
        max_segment_force = max(max_segment_force, force_norm)

        moment_arm_x = float(position[0] - reference_x)
        moment_arm_y = float(position[1] - reference_y)
        segment_yaw_moment += moment_arm_x * float(segment_force[1]) - moment_arm_y * float(segment_force[0])
        mujoco.mj_applyFT(
            model,
            data,
            segment_force,
            _ZERO_TORQUE,
            position,
            body_id,
            data.qfrc_applied,
        )

    damping_yaw_torque = (
        -float(config.yaw_damping_linear) * yaw_rate
        - float(config.yaw_damping_quadratic) * abs(yaw_rate) * yaw_rate
    )
    yaw_added_inertia = float(config.yaw_added_inertia_fraction) * _estimate_planar_yaw_inertia(
        model,
        data,
        segments,
        reference_position,
    )
    added_yaw_torque = -yaw_added_inertia * yaw_accel
    yaw_torque = _limit_scalar(
        damping_yaw_torque + added_yaw_torque,
        float(config.max_yaw_torque),
    )
    if yaw_torque != 0.0:
        yaw_torque_vector[2] = yaw_torque
        mujoco.mj_applyFT(
            model,
            data,
            _ZERO_FORCE,
            yaw_torque_vector,
            reference_position,
            centre_body_id,
            data.qfrc_applied,
        )
        yaw_torque_vector[2] = 0.0

    _STATE_CACHE[id(data)] = _HydrodynamicsState(
        time=current_time,
        normal_velocity_by_body=next_normal_velocity_by_body,
        yaw_rate=yaw_rate,
    )
    _LAST_DIAGNOSTICS[id(data)] = HydrodynamicsDiagnostics(
        segment_count=len(segments),
        segment_yaw_moment=float(segment_yaw_moment),
        damping_yaw_torque=float(damping_yaw_torque),
        added_yaw_torque=float(added_yaw_torque),
        max_segment_force=float(max_segment_force),
    )


def get_last_hydrodynamics_diagnostics(data: mujoco.MjData) -> HydrodynamicsDiagnostics | None:
    return _LAST_DIAGNOSTICS.get(id(data))


def reset_hydrodynamics_state(data: mujoco.MjData) -> None:
    _STATE_CACHE.pop(id(data), None)
    _LAST_DIAGNOSTICS.pop(id(data), None)


def _get_hydrodynamic_segments(
    model: mujoco.MjModel,
) -> tuple[_HydrodynamicSegment, ...]:
    cache_key = id(model)
    cached_segments = _MODEL_SEGMENT_CACHE.get(cache_key)
    if cached_segments is not None:
        return cached_segments

    segments_by_body: dict[int, _HydrodynamicSegment] = {}
    for geom_id in range(int(model.ngeom)):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name is None or not geom_name.endswith("_dyn"):
            continue

        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            continue

        mass = float(model.body_mass[body_id])
        if mass <= 0.0:
            continue

        segments_by_body[body_id] = _HydrodynamicSegment(
            body_id=body_id,
            name=body_name,
            mass=mass,
        )

    segments = tuple(segments_by_body[body_id] for body_id in sorted(segments_by_body))
    if not segments:
        raise ValueError("No fish hydrodynamic segments found. Expected fish collision geoms ending with '_dyn'.")

    _MODEL_SEGMENT_CACHE[cache_key] = segments
    return segments


def _normal_multiplier(body_name: str, config: HydrodynamicsConfig) -> float:
    lower_name = body_name.lower()
    if lower_name == "head":
        return float(config.head_normal_multiplier)
    if lower_name == "joint_tail" or lower_name.startswith("tail_"):
        return float(config.tail_normal_multiplier)
    return 1.0


def _get_centre_body_id(model: mujoco.MjModel, segments: tuple[_HydrodynamicSegment, ...]) -> int:
    cache_key = id(model)
    cached_body_id = _MODEL_CENTRE_BODY_CACHE.get(cache_key)
    if cached_body_id is not None:
        return cached_body_id

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "centre_compartment")
    if body_id < 0:
        body_id = segments[0].body_id

    _MODEL_CENTRE_BODY_CACHE[cache_key] = int(body_id)
    return int(body_id)


def _get_yaw_dof_id(model: mujoco.MjModel) -> int | None:
    cache_key = id(model)
    if cache_key in _MODEL_YAW_DOF_CACHE:
        return _MODEL_YAW_DOF_CACHE[cache_key]

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_yaw")
    yaw_dof_id = int(model.jnt_dofadr[joint_id]) if joint_id >= 0 else None
    _MODEL_YAW_DOF_CACHE[cache_key] = yaw_dof_id
    return yaw_dof_id


def _get_yaw_rate(model: mujoco.MjModel, data: mujoco.MjData, centre_body_id: int) -> float:
    yaw_dof_id = _get_yaw_dof_id(model)
    if yaw_dof_id is not None:
        return float(data.qvel[yaw_dof_id])

    angular_velocity, _ = _body_velocity(model, data, centre_body_id)
    return float(angular_velocity[2])


def _body_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity,
        0,
    )
    return velocity[:3].copy(), velocity[3:].copy()


def _body_planar_axes(data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    tangential_x = float(rotation[0, 0])
    tangential_y = float(rotation[1, 0])
    tangential_norm = math.hypot(tangential_x, tangential_y)
    if tangential_norm <= 1e-12:
        tangential_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        tangential_inv = 1.0 / tangential_norm
        tangential_axis = np.array(
            [tangential_x * tangential_inv, tangential_y * tangential_inv, 0.0],
            dtype=np.float64,
        )

    normal_x = float(rotation[0, 1])
    normal_y = float(rotation[1, 1])
    normal_norm = math.hypot(normal_x, normal_y)
    if normal_norm <= 1e-12:
        normal_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        normal_inv = 1.0 / normal_norm
        normal_axis = np.array([normal_x * normal_inv, normal_y * normal_inv, 0.0], dtype=np.float64)
    return tangential_axis, normal_axis


def _estimate_planar_yaw_inertia(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    segments: tuple[_HydrodynamicSegment, ...],
    reference_position: np.ndarray,
) -> float:
    yaw_inertia = 0.0
    reference_x = float(reference_position[0])
    reference_y = float(reference_position[1])
    for segment in segments:
        body_id = segment.body_id
        position = data.xpos[body_id]
        dx = float(position[0] - reference_x)
        dy = float(position[1] - reference_y)
        yaw_inertia += float(model.body_inertia[body_id][2])
        yaw_inertia += segment.mass * (dx * dx + dy * dy)
    return max(yaw_inertia, 0.0)


def _unit_or_default(vector: np.ndarray, default: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return default.copy()
    return vector / norm


def _limit_vector(vector: np.ndarray, limit: float) -> np.ndarray:
    if limit <= 0.0:
        return vector
    norm = _vector_norm(vector)
    if norm <= limit or norm <= 1e-12:
        return vector
    return vector * (limit / norm)


def _limit_scalar(value: float, limit: float) -> float:
    if limit <= 0.0:
        return float(value)
    return float(np.clip(value, -limit, limit))


def _vector_norm(vector: np.ndarray) -> float:
    return math.sqrt(
        float(vector[0]) * float(vector[0])
        + float(vector[1]) * float(vector[1])
        + float(vector[2]) * float(vector[2])
    )
