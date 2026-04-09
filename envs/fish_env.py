from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from configs.default_config import FishEnvConfig, make_config
from utils.geometry import body_to_world_vector, heading_vector, world_to_body_vector, wrap_to_pi
from utils.mappings import head_angle_to_tail_frequency, servo_angle_to_head_angle
from utils.obstacles import (
    CircularObstacle,
    LocalObstacleObservation,
    get_local_obstacle_observation,
    sample_circular_obstacles,
)
from utils.scenario_io import FixedScenario, load_fixed_scenario


@dataclass
class VisualObstacleObservation:
    detected: float
    pixel_fraction: float
    center_fraction: float
    nearest_depth: float

    @classmethod
    def empty(cls, depth_cap: float) -> "VisualObstacleObservation":
        return cls(
            detected=0.0,
            pixel_fraction=0.0,
            center_fraction=0.0,
            nearest_depth=float(depth_cap),
        )


class FishPathAvoidEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        config: FishEnvConfig | None = None,
        render_mode: str | None = None,
        enable_mujoco_viewer: bool = False,
        realtime_playback: bool = False,
        viewer_slowdown: float = 0.0,
        enable_episode_recording: bool = False,
        recording_camera_name: str = "oblique",
        recording_width: int = 480,
        recording_height: int = 270,
        recording_frame_stride: int = 4,
        scenario_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.config = config or make_config().env
        self.render_mode = render_mode
        self.enable_mujoco_viewer = enable_mujoco_viewer
        self.realtime_playback = realtime_playback
        self.viewer_slowdown = max(0.0, float(viewer_slowdown))
        self.enable_episode_recording = bool(enable_episode_recording)
        self.recording_camera_name = str(recording_camera_name)
        self.recording_width = max(64, int(recording_width))
        self.recording_height = max(64, int(recording_height))
        self.recording_frame_stride = max(1, int(recording_frame_stride))
        self.scenario_path = None if scenario_path is None else Path(scenario_path).resolve()
        self.fixed_scenario = None if self.scenario_path is None else load_fixed_scenario(self.scenario_path)
        self.active_scenario_id = None if self.fixed_scenario is None else self.fixed_scenario.scenario_id

        self.model_path = Path(self.config.model.xml_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.sim_timestep = float(self.model.opt.timestep)
        self.control_timestep = float(self.config.frame_skip) * self.sim_timestep
        self.goal_center = np.asarray(self.config.task.goal_center, dtype=float)
        self.goal_half_extents = np.asarray(self.config.task.goal_half_extents, dtype=float)
        self._obs_renderer = mujoco.Renderer(
            self.model,
            height=int(self.config.camera.height),
            width=int(self.config.camera.width),
        )
        self._video_renderer: mujoco.Renderer | None = None
        if self.enable_episode_recording:
            self._video_renderer = mujoco.Renderer(
                self.model,
                height=self.recording_height,
                width=self.recording_width,
            )

        self.front_servo_act_id = self._actuator_id("front_servo_act")
        self.back_servo_act_id = self._actuator_id("back_servo_act")
        self.root_x_qposadr = self._joint_qposadr("root_x")
        self.root_y_qposadr = self._joint_qposadr("root_y")
        self.root_yaw_qposadr = self._joint_qposadr("root_yaw")
        self.root_x_qveladr = self._joint_qveladr("root_x")
        self.root_y_qveladr = self._joint_qveladr("root_y")
        self.root_yaw_qveladr = self._joint_qveladr("root_yaw")
        self.front_servo_qposadr = self._joint_qposadr("front_servo")
        self.back_servo_qposadr = self._joint_qposadr("back_servo")
        self.goal_region_geom_id = self._geom_id("goal_region_marker")
        self.visual_obstacle_geom_ids = self._visual_obstacle_geom_ids()
        self.visual_obstacle_mocap_ids = self._visual_obstacle_mocap_ids()
        self._visual_obstacle_geom_id_set = set(self.visual_obstacle_geom_ids)
        self.fish_collision_geom_ids = self._fish_collision_geom_ids()
        self.spawn_collision_templates = self._build_spawn_collision_templates()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        int(self.config.camera.height),
                        int(self.config.camera.width),
                        3,
                    ),
                    dtype=np.uint8,
                ),
                "imu": spaces.Box(
                    low=np.array(
                        [
                            -self.config.imu.accel_clip,
                            -self.config.imu.accel_clip,
                            -self.config.imu.gyro_clip,
                        ],
                        dtype=np.float32,
                    ),
                    high=np.array(
                        [
                            self.config.imu.accel_clip,
                            self.config.imu.accel_clip,
                            self.config.imu.gyro_clip,
                        ],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                ),
            }
        )

        self._figure = None
        self._axes = None
        self._plt = None
        self._viewer = None
        self._viewer_last_wall_time: float | None = None
        self._viewer_last_sim_time: float | None = None
        self._episode_video_frames: list[np.ndarray] = []
        self._episode_head_video_frames: list[np.ndarray] = []
        self._completed_episode_video_frames: list[np.ndarray] = []
        self._completed_episode_head_video_frames: list[np.ndarray] = []
        self._completed_episode_index: int | None = None
        self._recorded_frame_counter = 0

        self.elapsed_steps = 0
        self.episode_count = 0
        self.episode_return = 0.0
        self.raw_action = 0.0
        self.filtered_action = 0.0
        self.prev_action = 0.0
        self.tail_phase = 0.0
        self.theta_m_target = 0.0
        self.theta_m = 0.0
        self.theta_m_velocity = 0.0
        self.theta_h = 0.0
        self.tail_freq_target = self.config.mapping.base_tail_freq
        self.tail_freq = self.config.mapping.base_tail_freq
        self.back_servo_command = 0.0
        self.back_servo_velocity = 0.0
        self.head_servo_locked = False
        self.obstacles: list[CircularObstacle] = []
        self.position = np.zeros(2, dtype=float)
        self.yaw = 0.0
        self.forward_speed = 0.0
        self.lateral_speed = 0.0
        self.yaw_rate = 0.0
        self.prev_forward_speed = 0.0
        self.prev_lateral_speed = 0.0
        self.imu_acceleration = np.zeros(2, dtype=float)
        self.goal_target = self.goal_center.copy()
        self.goal_distance = math.inf
        self.goal_progress_ratio = 0.0
        self.initial_goal_distance = 1.0
        self.local_obstacle_obs = LocalObstacleObservation.empty()
        self.visual_obstacle_obs = VisualObstacleObservation.empty(self.config.camera.visibility_distance)
        self.min_obstacle_clearance = math.inf
        self.min_wall_clearance = math.inf
        self.reached_goal = False
        self.goal_hold_complete = False
        self.goal_reached_step: int | None = None
        self.collided = False
        self.wall_collision = False
        self.out_of_bounds = False
        self.timeout = False

    def _is_head_servo_drive_active(self, normalized_action: float) -> bool:
        return abs(normalized_action) > float(self.config.mapping.head_servo_command_deadband)

    def _get_mujoco_viewer(self):
        if self._viewer is not None and not self._viewer.is_running():
            self._viewer = None
        return self._viewer

    def _ensure_mujoco_viewer(self) -> None:
        if not self.enable_mujoco_viewer or self._get_mujoco_viewer() is not None:
            return

        from mujoco import viewer

        self._viewer = viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self._viewer_last_wall_time = time.perf_counter()
        self._viewer_last_sim_time = float(self.data.time)

    def _viewer_lock(self):
        viewer = self._get_mujoco_viewer()
        if viewer is None:
            return nullcontext()
        return viewer.lock()

    def _sync_mujoco_viewer(self, *, state_only: bool = False, reset_timing: bool = False) -> None:
        viewer = self._get_mujoco_viewer()
        if viewer is None:
            return

        current_sim_time = float(self.data.time)
        if reset_timing or self._viewer_last_wall_time is None or self._viewer_last_sim_time is None:
            self._viewer_last_wall_time = time.perf_counter()
            self._viewer_last_sim_time = current_sim_time
        elif self.realtime_playback:
            slowdown = self.viewer_slowdown if self.viewer_slowdown > 0.0 else 1.0
            target_wall_time = self._viewer_last_wall_time + slowdown * max(
                0.0,
                current_sim_time - self._viewer_last_sim_time,
            )
            sleep_time = target_wall_time - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)

        viewer.sync(state_only=state_only)
        self._viewer_last_wall_time = time.perf_counter()
        self._viewer_last_sim_time = current_sim_time

    def _actuator_id(self, actuator_name: str) -> int:
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"Actuator not found in XML: {actuator_name}")
        return actuator_id

    def _geom_id(self, geom_name: str) -> int:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"Geom not found in XML: {geom_name}")
        return geom_id

    def _joint_qposadr(self, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in XML: {joint_name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_qveladr(self, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in XML: {joint_name}")
        return int(self.model.jnt_dofadr[joint_id])

    def _set_pose(self, position: np.ndarray, yaw: float) -> None:
        self.data.qpos[self.root_x_qposadr] = float(position[0])
        self.data.qpos[self.root_y_qposadr] = float(position[1])
        self.data.qpos[self.root_yaw_qposadr] = float(yaw)

    def _visual_obstacle_geom_ids(self) -> list[int]:
        geom_ids: list[int] = []
        index = 0
        while True:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"visual_obstacle_{index}")
            if geom_id < 0:
                break
            geom_ids.append(geom_id)
            index += 1
        return geom_ids

    def _visual_obstacle_mocap_ids(self) -> list[int]:
        mocap_ids: list[int] = []
        index = 0
        while True:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"visual_obstacle_body_{index}")
            if body_id < 0:
                break
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id < 0:
                raise ValueError(f"Obstacle body visual_obstacle_body_{index} is missing a mocap id.")
            mocap_ids.append(mocap_id)
            index += 1
        return mocap_ids

    def _fish_collision_geom_ids(self) -> list[int]:
        geom_ids: list[int] = []
        for geom_id in range(int(self.model.ngeom)):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name is None or not geom_name.endswith("_dyn"):
                continue
            if int(self.model.geom_contype[geom_id]) == 1 and int(self.model.geom_conaffinity[geom_id]) == 2:
                geom_ids.append(geom_id)
        if not geom_ids:
            raise ValueError("No fish collision geoms were found for goal contact detection.")
        return geom_ids

    def _build_spawn_collision_templates(self) -> list[tuple[np.ndarray, float]]:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        root_position = self._get_position()
        templates: list[tuple[np.ndarray, float]] = []
        for geom_id in self.fish_collision_geom_ids:
            geom_center = np.asarray(self.data.geom_xpos[geom_id][:2], dtype=float) - root_position
            geom_type = int(self.model.geom_type[geom_id])
            geom_size = np.asarray(self.model.geom_size[geom_id], dtype=float)
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                planar_radius = float(np.hypot(geom_size[0], geom_size[1]))
            else:
                planar_radius = float(max(geom_size[0], geom_size[1]))
            templates.append((geom_center, planar_radius))
        return templates

    def _sync_visual_obstacles(self) -> None:
        if not self.visual_obstacle_geom_ids:
            return

        hidden_position = np.array([0.0, 0.0, -5.0], dtype=float)
        hidden_size = np.array([0.001, 0.001, 0.0], dtype=float)
        visible_rgba = np.array([0.91, 0.42, 0.32, 0.90], dtype=float)
        pillar_half_height = 0.74

        for slot, (geom_id, mocap_id) in enumerate(zip(self.visual_obstacle_geom_ids, self.visual_obstacle_mocap_ids)):
            if slot < len(self.obstacles):
                obstacle = self.obstacles[slot]
                self.data.mocap_pos[mocap_id] = np.array([obstacle.center[0], obstacle.center[1], 0.0], dtype=float)
                self.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                self.model.geom_size[geom_id] = np.array([obstacle.radius, pillar_half_height, 0.0], dtype=float)
                self.model.geom_rgba[geom_id] = visible_rgba
            else:
                self.data.mocap_pos[mocap_id] = hidden_position
                self.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                self.model.geom_size[geom_id] = hidden_size
                self.model.geom_rgba[geom_id, 3] = 0.0

        mujoco.mj_setConst(self.model, self.data)

    def _sync_goal_region_marker(self) -> None:
        self.model.geom_pos[self.goal_region_geom_id] = np.array(
            [self.goal_center[0], self.goal_center[1], -0.748],
            dtype=float,
        )
        self.model.geom_size[self.goal_region_geom_id] = np.array(
            [self.goal_half_extents[0], self.goal_half_extents[1], 0.002],
            dtype=float,
        )

    def _get_position(self) -> np.ndarray:
        return np.array(
            [
                float(self.data.qpos[self.root_x_qposadr]),
                float(self.data.qpos[self.root_y_qposadr]),
            ],
            dtype=float,
        )

    def _apply_action_filter(self, raw_action: float) -> float:
        tau = float(self.config.action_filter_tau)
        if tau <= 1e-8:
            self.filtered_action = raw_action
            return raw_action

        alpha = 1.0 - math.exp(-self.control_timestep / tau)
        self.filtered_action += alpha * (raw_action - self.filtered_action)
        return float(np.clip(self.filtered_action, -1.0, 1.0))

    def _apply_servo_dynamics(
        self,
        current_value: float,
        current_velocity: float,
        target_value: float,
        speed_sec_per_60deg: float,
        accel_deg_per_s2: float,
        timestep: float,
        angle_limit: float,
    ) -> tuple[float, float]:
        if speed_sec_per_60deg <= 1e-8:
            return float(np.clip(target_value, -angle_limit, angle_limit)), 0.0

        max_speed_rad_per_s = math.radians(60.0) / speed_sec_per_60deg
        desired_velocity = float(
            np.clip(
                (target_value - current_value) / max(timestep, 1e-8),
                -max_speed_rad_per_s,
                max_speed_rad_per_s,
            )
        )
        if accel_deg_per_s2 > 1e-8:
            max_accel_rad_per_s2 = math.radians(accel_deg_per_s2)
            max_dv = max_accel_rad_per_s2 * timestep
            new_velocity = current_velocity + float(np.clip(desired_velocity - current_velocity, -max_dv, max_dv))
        else:
            new_velocity = desired_velocity

        delta = new_velocity * timestep
        remaining_error = target_value - current_value
        if abs(delta) > abs(remaining_error):
            delta = remaining_error
            new_velocity = 0.0

        new_value = float(np.clip(current_value + delta, -angle_limit, angle_limit))
        if abs(new_value) >= angle_limit - 1e-8 and math.copysign(1.0, new_velocity or 1.0) == math.copysign(1.0, new_value):
            new_velocity = 0.0
        return new_value, float(new_velocity)

    def _apply_theta_m_rate_limit(self, target_theta_m: float, timestep: float | None = None) -> float:
        speed_sec_per_60deg = float(self.config.mapping.servo_speed_sec_per_60deg)
        self.theta_m, self.theta_m_velocity = self._apply_servo_dynamics(
            current_value=self.theta_m,
            current_velocity=self.theta_m_velocity,
            target_value=target_theta_m,
            speed_sec_per_60deg=speed_sec_per_60deg,
            accel_deg_per_s2=float(self.config.mapping.servo_accel_deg_per_s2),
            timestep=self.sim_timestep if timestep is None else timestep,
            angle_limit=float(self.config.mapping.theta_m_max),
        )
        return self.theta_m

    def _apply_tail_freq_filter(self, target_tail_freq: float, timestep: float | None = None) -> float:
        tau = float(self.config.mapping.tail_freq_filter_tau)
        if tau <= 1e-8:
            self.tail_freq = target_tail_freq
            return target_tail_freq

        filter_timestep = self.control_timestep if timestep is None else timestep
        alpha = 1.0 - math.exp(-filter_timestep / tau)
        self.tail_freq += alpha * (target_tail_freq - self.tail_freq)
        self.tail_freq = float(
            np.clip(
                self.tail_freq,
                self.config.mapping.tail_freq_min,
                self.config.mapping.tail_freq_max,
            )
        )
        return self.tail_freq

    def _apply_back_servo_rate_limit(self, target_back_servo_command: float, timestep: float | None = None) -> float:
        self.back_servo_command, self.back_servo_velocity = self._apply_servo_dynamics(
            current_value=self.back_servo_command,
            current_velocity=self.back_servo_velocity,
            target_value=target_back_servo_command,
            speed_sec_per_60deg=float(self.config.mapping.tail_servo_speed_sec_per_60deg),
            accel_deg_per_s2=float(self.config.mapping.tail_servo_accel_deg_per_s2),
            timestep=self.sim_timestep if timestep is None else timestep,
            angle_limit=float(self.config.mapping.back_servo_amplitude),
        )
        return self.back_servo_command

    def _get_yaw(self) -> float:
        return float(self.data.qpos[self.root_yaw_qposadr])

    def _get_body_velocities(self) -> tuple[float, float, float]:
        world_velocity = np.array(
            [
                float(self.data.qvel[self.root_x_qveladr]),
                float(self.data.qvel[self.root_y_qveladr]),
            ],
            dtype=float,
        )
        body_velocity = world_to_body_vector(world_velocity, self.yaw)
        yaw_rate = float(self.data.qvel[self.root_yaw_qveladr])
        return float(body_velocity[0]), float(body_velocity[1]), yaw_rate

    def _spawn_pose_is_safe(self, position: np.ndarray, yaw: float, obstacles: list[CircularObstacle]) -> bool:
        wall_margin = float(self.config.task.spawn_wall_margin)
        obstacle_margin = float(self.config.task.spawn_obstacle_margin)
        for local_center, planar_radius in self.spawn_collision_templates:
            world_center = position + body_to_world_vector(local_center, yaw)
            padded_radius = planar_radius + wall_margin
            if (
                abs(world_center[0]) > self.config.pool_half_length - padded_radius
                or abs(world_center[1]) > self.config.pool_half_width - padded_radius
            ):
                return False
            for obstacle in obstacles:
                minimum_distance = planar_radius + obstacle.radius + obstacle_margin
                if float(np.linalg.norm(world_center - obstacle.center)) <= minimum_distance:
                    return False
        return True

    def _sample_spawn_pose(self, obstacles: list[CircularObstacle]) -> tuple[np.ndarray, float]:
        yaw_range = math.radians(self.config.task.spawn_yaw_range_deg)
        for _ in range(max(1, int(self.config.task.spawn_max_attempts))):
            x = float(self.np_random.uniform(*self.config.task.spawn_x_range))
            y = float(self.np_random.uniform(*self.config.task.spawn_y_range))
            position = np.array([x, y], dtype=float)
            goal_heading = math.atan2(self.goal_center[1] - y, self.goal_center[0] - x)
            yaw = wrap_to_pi(goal_heading + float(self.np_random.uniform(-yaw_range, yaw_range)))
            if self._spawn_pose_is_safe(position, yaw, obstacles):
                return position, yaw

        fallback_x = max(self.config.task.spawn_x_range)
        fallback_position = np.array([fallback_x, 0.0], dtype=float)
        fallback_yaw = math.atan2(self.goal_center[1], self.goal_center[0] - fallback_x)
        return fallback_position, fallback_yaw

    def _clone_obstacles(self, obstacles: list[CircularObstacle]) -> list[CircularObstacle]:
        return [
            CircularObstacle(center=np.asarray(obstacle.center, dtype=float).copy(), radius=float(obstacle.radius))
            for obstacle in obstacles
        ]

    def _apply_fixed_scenario(self, scenario: FixedScenario) -> tuple[np.ndarray, float]:
        self.active_scenario_id = scenario.scenario_id
        self.goal_center = np.asarray(scenario.goal_center, dtype=float).copy()
        self.goal_half_extents = np.asarray(scenario.goal_half_extents, dtype=float).copy()
        self.obstacles = self._clone_obstacles(scenario.obstacles)
        return np.asarray(scenario.spawn_position, dtype=float).copy(), float(scenario.spawn_yaw)

    def export_fixed_scenario(
        self,
        scenario_id: str | None = None,
        source_seed: int | None = None,
    ) -> FixedScenario:
        resolved_scenario_id = scenario_id or self.active_scenario_id or f"scenario_{self.episode_count:02d}"
        return FixedScenario(
            scenario_id=resolved_scenario_id,
            source_seed=source_seed,
            spawn_position=np.asarray(self.position, dtype=float).copy(),
            spawn_yaw=float(self.yaw),
            goal_center=np.asarray(self.goal_center, dtype=float).copy(),
            goal_half_extents=np.asarray(self.goal_half_extents, dtype=float).copy(),
            obstacles=self._clone_obstacles(self.obstacles),
        )

    def _get_goal_target(self, position: np.ndarray) -> np.ndarray:
        goal_min = self.goal_center - self.goal_half_extents
        goal_max = self.goal_center + self.goal_half_extents
        return np.clip(position, goal_min, goal_max).astype(float)

    def _is_in_goal_region(self, position: np.ndarray) -> bool:
        goal_min = self.goal_center - self.goal_half_extents
        goal_max = self.goal_center + self.goal_half_extents
        return bool(np.all(position >= goal_min) and np.all(position <= goal_max))

    def _touches_goal_region(self) -> bool:
        goal_min = self.goal_center - self.goal_half_extents
        goal_max = self.goal_center + self.goal_half_extents

        for geom_id in self.fish_collision_geom_ids:
            geom_center = np.asarray(self.data.geom_xpos[geom_id][:2], dtype=float)
            nearest_point = np.clip(geom_center, goal_min, goal_max)
            planar_offset = geom_center - nearest_point
            geom_type = int(self.model.geom_type[geom_id])
            geom_size = np.asarray(self.model.geom_size[geom_id], dtype=float)

            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                planar_radius = float(np.hypot(geom_size[0], geom_size[1]))
            else:
                planar_radius = float(max(geom_size[0], geom_size[1]))

            if float(np.linalg.norm(planar_offset)) <= planar_radius:
                return True

        return False

    def _goal_distance_ratio(self) -> float:
        return float(np.clip(self.goal_distance / max(self.initial_goal_distance, 1e-6), 0.0, 1.0))

    def _box_blur_rgb(self, image: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return image.copy()

        height, width, channels = image.shape
        padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        blurred = np.zeros((height, width, channels), dtype=np.float32)
        kernel_size = float((2 * radius + 1) ** 2)

        for y_offset in range(2 * radius + 1):
            for x_offset in range(2 * radius + 1):
                blurred += padded[y_offset : y_offset + height, x_offset : x_offset + width, :]

        return blurred / kernel_size

    def _apply_underwater_camera_model(self, frame: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        camera_cfg = self.config.camera
        rgb = np.asarray(frame, dtype=np.float32) / 255.0
        depth = np.asarray(depth_map, dtype=np.float32)
        max_depth = max(float(camera_cfg.max_depth_distance), 1e-6)
        visibility_distance = max(float(camera_cfg.visibility_distance), 1e-6)
        visibility_transmittance = float(np.clip(camera_cfg.transmittance_at_visibility, 1e-3, 0.999))
        attenuation_coeff = -math.log(visibility_transmittance) / visibility_distance

        depth = np.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0)
        depth = np.clip(depth, 0.0, max_depth)
        transmittance = np.exp(-attenuation_coeff * depth).astype(np.float32)
        water_color = np.asarray(camera_cfg.water_color_rgb, dtype=np.float32) / 255.0

        degraded = water_color + transmittance[..., None] * (rgb - water_color)

        if camera_cfg.blur_radius > 0 and camera_cfg.max_blur_strength > 0.0:
            blurred = self._box_blur_rgb(degraded, int(camera_cfg.blur_radius))
            blur_mix = float(camera_cfg.max_blur_strength) * np.square(1.0 - transmittance)
            degraded = degraded * (1.0 - blur_mix[..., None]) + blurred * blur_mix[..., None]

        if camera_cfg.far_noise_std > 0.0:
            noise_scale = (float(camera_cfg.far_noise_std) / 255.0) * np.square(1.0 - transmittance)
            degraded += self.np_random.normal(0.0, 1.0, size=degraded.shape).astype(np.float32) * noise_scale[..., None]

        return np.clip(degraded * 255.0, 0.0, 255.0).astype(np.uint8)

    def _render_rgb_frame(self, renderer: mujoco.Renderer, camera_name: str) -> np.ndarray:
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(self.data, camera=camera_name)
        return np.asarray(renderer.render(), dtype=np.uint8)

    def _render_depth_map(self, renderer: mujoco.Renderer, camera_name: str) -> np.ndarray:
        renderer.disable_segmentation_rendering()
        renderer.enable_depth_rendering()
        try:
            renderer.update_scene(self.data, camera=camera_name)
            return np.asarray(renderer.render(), dtype=np.float32)
        finally:
            renderer.disable_depth_rendering()

    def _render_segmentation_frame(self, renderer: mujoco.Renderer, camera_name: str) -> np.ndarray:
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(self.data, camera=camera_name)
            return np.asarray(renderer.render(), dtype=np.int32)
        finally:
            renderer.disable_segmentation_rendering()

    def _begin_episode_recording(self) -> None:
        if not self.enable_episode_recording:
            return
        self._episode_video_frames = []
        self._episode_head_video_frames = []
        self._recorded_frame_counter = 0

    def _stash_completed_episode_video(self) -> None:
        if not self.enable_episode_recording or not self._episode_video_frames:
            self._episode_video_frames = []
            self._episode_head_video_frames = []
            return
        self._completed_episode_video_frames = self._episode_video_frames
        self._completed_episode_head_video_frames = self._episode_head_video_frames
        self._completed_episode_index = self.episode_count
        self._episode_video_frames = []
        self._episode_head_video_frames = []

    def _capture_episode_frame(self, *, force: bool = False) -> None:
        if not self.enable_episode_recording or self._video_renderer is None:
            return
        if not force and (self._recorded_frame_counter % self.recording_frame_stride) != 0:
            self._recorded_frame_counter += 1
            return

        frame = self._render_rgb_frame(self._video_renderer, self.recording_camera_name)
        self._episode_video_frames.append(frame.copy())
        head_frame = self._render_head_camera()
        self._episode_head_video_frames.append(head_frame.copy())
        self._recorded_frame_counter += 1

    def _save_video_frames(self, frames: list[np.ndarray], output_path: str | Path, fps: int) -> Path | None:
        if not frames:
            return None
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "Saving MP4 episode videos requires 'imageio' and 'imageio-ffmpeg' to be installed."
            ) from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            output_path,
            fps=max(1, int(fps)),
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            macro_block_size=1,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        return output_path

    def save_completed_episode_videos(
        self,
        video_path: str | Path,
        head_video_path: str | Path,
        fps: int,
    ) -> tuple[Path | None, Path | None]:
        top_video_path = self._save_video_frames(self._completed_episode_video_frames, video_path, fps)
        head_camera_video_path = self._save_video_frames(self._completed_episode_head_video_frames, head_video_path, fps)
        self._completed_episode_video_frames = []
        self._completed_episode_head_video_frames = []
        self._completed_episode_index = None
        return top_video_path, head_camera_video_path

    def _render_head_camera(self) -> np.ndarray:
        frame = self._render_rgb_frame(self._obs_renderer, self.config.camera.camera_name)
        if not self.config.camera.underwater_effect_enabled:
            return frame

        depth_map = self._render_depth_map(self._obs_renderer, self.config.camera.camera_name)
        return self._apply_underwater_camera_model(frame, depth_map)

    def _get_visual_obstacle_observation(self) -> VisualObstacleObservation:
        camera_cfg = self.config.camera
        visibility_depth_cap = min(float(camera_cfg.visibility_distance), float(camera_cfg.max_depth_distance))
        if not self.config.camera.underwater_effect_enabled:
            return VisualObstacleObservation.empty(visibility_depth_cap)

        segmentation = self._render_segmentation_frame(self._obs_renderer, camera_cfg.camera_name)
        geom_ids = np.asarray(segmentation[..., 0], dtype=np.int32)
        obj_types = np.asarray(segmentation[..., 1], dtype=np.int32)

        depth_map = self._render_depth_map(self._obs_renderer, camera_cfg.camera_name)
        depth_map = np.nan_to_num(depth_map, nan=visibility_depth_cap, posinf=visibility_depth_cap, neginf=0.0)
        depth_map = np.clip(depth_map, 0.0, visibility_depth_cap)

        geom_type_code = int(mujoco.mjtObj.mjOBJ_GEOM)
        obstacle_mask = (obj_types == geom_type_code) & np.isin(geom_ids, list(self._visual_obstacle_geom_id_set))
        obstacle_mask &= depth_map <= visibility_depth_cap
        if not np.any(obstacle_mask):
            return VisualObstacleObservation.empty(visibility_depth_cap)

        total_pixels = float(obstacle_mask.size)
        pixel_fraction = float(np.count_nonzero(obstacle_mask) / max(total_pixels, 1.0))

        width = obstacle_mask.shape[1]
        center_half_width = max(1, int(round(0.18 * width)))
        center_start = max(0, width // 2 - center_half_width)
        center_end = min(width, width // 2 + center_half_width)
        center_mask = obstacle_mask[:, center_start:center_end]
        center_fraction = float(np.count_nonzero(center_mask) / max(center_mask.size, 1))

        nearest_depth = float(np.min(depth_map[obstacle_mask]))
        return VisualObstacleObservation(
            detected=1.0,
            pixel_fraction=pixel_fraction,
            center_fraction=center_fraction,
            nearest_depth=nearest_depth,
        )

    def _update_cached_state(self) -> None:
        self.position = self._get_position()
        self.yaw = self._get_yaw()
        self.forward_speed, self.lateral_speed, self.yaw_rate = self._get_body_velocities()
        self.goal_target = self._get_goal_target(self.position)
        self.goal_distance = float(np.linalg.norm(self.goal_target - self.position))
        self.goal_progress_ratio = float(np.clip(1.0 - self._goal_distance_ratio(), 0.0, 1.0))
        self.local_obstacle_obs = self._get_local_obstacle_observation()
        self.visual_obstacle_obs = self._get_visual_obstacle_observation()
        self.min_obstacle_clearance = self._compute_min_obstacle_clearance()
        self.min_wall_clearance = self._compute_min_wall_clearance()

    def _compute_min_obstacle_clearance(self) -> float:
        if not self.obstacles:
            return math.inf
        distances = [
            np.linalg.norm(self.position - obstacle.center) - (obstacle.radius + self.config.fish_collision_radius)
            for obstacle in self.obstacles
        ]
        return float(min(distances))

    def _compute_min_wall_clearance(self) -> float:
        wall_clearances: list[float] = []
        for geom_id in self.fish_collision_geom_ids:
            geom_center = np.asarray(self.data.geom_xpos[geom_id][:2], dtype=float)
            geom_type = int(self.model.geom_type[geom_id])
            geom_size = np.asarray(self.model.geom_size[geom_id], dtype=float)
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                planar_radius = float(np.hypot(geom_size[0], geom_size[1]))
            else:
                planar_radius = float(max(geom_size[0], geom_size[1]))

            wall_clearances.extend(
                [
                    self.config.pool_half_length - abs(float(geom_center[0])) - planar_radius,
                    self.config.pool_half_width - abs(float(geom_center[1])) - planar_radius,
                ]
            )

        if not wall_clearances:
            return math.inf
        return float(min(wall_clearances))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._stash_completed_episode_video()

        self.elapsed_steps = 0
        self.episode_return = 0.0
        self.raw_action = 0.0
        self.filtered_action = 0.0
        self.prev_action = 0.0
        self.tail_phase = 0.0
        self.theta_m_target = 0.0
        self.theta_m = 0.0
        self.theta_m_velocity = 0.0
        self.theta_h = 0.0
        self.tail_freq_target = self.config.mapping.base_tail_freq
        self.tail_freq = self.config.mapping.base_tail_freq
        self.back_servo_command = 0.0
        self.back_servo_velocity = 0.0
        self.head_servo_locked = False
        self.forward_speed = 0.0
        self.lateral_speed = 0.0
        self.yaw_rate = 0.0
        self.prev_forward_speed = 0.0
        self.prev_lateral_speed = 0.0
        self.imu_acceleration = np.zeros(2, dtype=float)
        self.reached_goal = False
        self.goal_hold_complete = False
        self.goal_reached_step = None
        self.collided = False
        self.wall_collision = False
        self.out_of_bounds = False
        self.timeout = False
        self.initial_goal_distance = 1.0
        if self.fixed_scenario is not None:
            initial_position, initial_heading = self._apply_fixed_scenario(self.fixed_scenario)
        else:
            obstacle_interval = max(1, int(self.config.obstacle.resample_interval_episodes))
            if not self.obstacles or self.episode_count % obstacle_interval == 0:
                self.obstacles = self._sample_obstacles()
            initial_position, initial_heading = self._sample_spawn_pose(self.obstacles)

        with self._viewer_lock():
            mujoco.mj_resetData(self.model, self.data)
            self._sync_goal_region_marker()
            self._sync_visual_obstacles()
            self._set_pose(initial_position, initial_heading)
            self.data.qpos[self.front_servo_qposadr] = 0.0
            self.data.qpos[self.back_servo_qposadr] = 0.0
            self.data.qvel[:] = 0.0
            self.data.ctrl[self.front_servo_act_id] = 0.0
            self.data.ctrl[self.back_servo_act_id] = 0.0
            mujoco.mj_forward(self.model, self.data)

        self._ensure_mujoco_viewer()
        self._update_cached_state()
        self.prev_forward_speed = self.forward_speed
        self.prev_lateral_speed = self.lateral_speed
        self.initial_goal_distance = max(self.goal_distance, 1e-6)
        self.goal_progress_ratio = 0.0
        self._sync_mujoco_viewer(reset_timing=True)
        self._begin_episode_recording()
        self._capture_episode_frame(force=True)
        self.episode_count += 1

        observation = self._get_obs()
        info = self._build_info({})
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        normalized_action = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        prev_goal_distance = self.goal_distance
        prev_visual_obstacle_obs = self.visual_obstacle_obs
        prev_wall_clearance = self.min_wall_clearance
        prev_filtered_action = self.prev_action

        self.raw_action = normalized_action
        if self._is_head_servo_drive_active(normalized_action):
            self.head_servo_locked = False
            action_filter_input = normalized_action
        else:
            self.head_servo_locked = True
            action_filter_input = self.filtered_action

        filtered_action = self._apply_action_filter(action_filter_input)
        self.theta_m_target = filtered_action * self.config.mapping.theta_m_max

        with self._viewer_lock():
            for _ in range(self.config.frame_skip):
                self.theta_m = self._apply_theta_m_rate_limit(self.theta_m_target, timestep=self.sim_timestep)
                self.theta_h = servo_angle_to_head_angle(self.theta_m, self.config.mapping)
                self.tail_freq_target = head_angle_to_tail_frequency(self.theta_h, self.config.mapping)
                self.tail_freq = self._apply_tail_freq_filter(self.tail_freq_target, timestep=self.sim_timestep)
                self.tail_phase += 2.0 * math.pi * self.tail_freq * self.sim_timestep
                back_servo_target = self.config.mapping.back_servo_amplitude * math.sin(self.tail_phase)
                self.back_servo_command = self._apply_back_servo_rate_limit(back_servo_target, timestep=self.sim_timestep)
                self.data.ctrl[self.front_servo_act_id] = self.theta_m
                self.data.ctrl[self.back_servo_act_id] = self.back_servo_command
                mujoco.mj_step(self.model, self.data)

        self.elapsed_steps += 1
        self.prev_action = filtered_action

        self._update_cached_state()
        self.imu_acceleration = np.array(
            [
                (self.forward_speed - self.prev_forward_speed) / max(self.control_timestep, 1e-8),
                (self.lateral_speed - self.prev_lateral_speed) / max(self.control_timestep, 1e-8),
            ],
            dtype=float,
        )
        self.prev_forward_speed = self.forward_speed
        self.prev_lateral_speed = self.lateral_speed
        self._sync_mujoco_viewer()
        self._capture_episode_frame()
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        reward, reward_terms = self._get_reward(
            prev_goal_distance,
            prev_visual_obstacle_obs,
            prev_wall_clearance,
            prev_filtered_action,
            filtered_action,
            terminated,
            truncated,
        )
        self.episode_return += reward

        observation = self._get_obs()
        info = self._build_info(reward_terms)
        return observation, reward, terminated, truncated, info

    def _get_imu_obs(self) -> np.ndarray:
        return np.array(
            [
                np.clip(self.imu_acceleration[0], -self.config.imu.accel_clip, self.config.imu.accel_clip),
                np.clip(self.imu_acceleration[1], -self.config.imu.accel_clip, self.config.imu.accel_clip),
                np.clip(self.yaw_rate, -self.config.imu.gyro_clip, self.config.imu.gyro_clip),
            ],
            dtype=np.float32,
        )

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "image": self._render_head_camera(),
            "imu": self._get_imu_obs(),
        }

    def _normalized_goal_distance(self, goal_distance: float) -> float:
        distance_scale = max(float(self.initial_goal_distance), 1e-6)
        normalized_distance = float(goal_distance) / distance_scale
        return float(np.clip(normalized_distance, 0.0, float(self.config.reward.goal_distance_clip)))

    def _goal_progress_reward(self, prev_goal_distance: float, current_goal_distance: float) -> float:
        prev_distance = self._normalized_goal_distance(prev_goal_distance)
        current_distance = self._normalized_goal_distance(current_goal_distance)
        return float(self.config.reward.target_progress_scale * (prev_distance - current_distance))

    def _current_obstacle_distance(self) -> float:
        candidate_distances = [float(self.min_obstacle_clearance)]
        if self.local_obstacle_obs.detected > 0.0 and math.isfinite(self.local_obstacle_obs.edge_distance):
            candidate_distances.append(float(self.local_obstacle_obs.edge_distance))
        finite_distances = [distance for distance in candidate_distances if math.isfinite(distance)]
        if not finite_distances:
            return math.inf
        return float(min(finite_distances))

    def _obstacle_avoidance_reward(self, obstacle_distance: float) -> float:
        reward_cfg = self.config.reward
        safe_distance = max(float(reward_cfg.obstacle_safe_distance), 1e-6)
        danger_distance = float(
            np.clip(reward_cfg.obstacle_danger_distance, 0.0, safe_distance - 1e-6),
        )

        if not math.isfinite(obstacle_distance) or obstacle_distance >= safe_distance:
            return 0.0
        if obstacle_distance <= danger_distance:
            return -float(reward_cfg.obstacle_collision_penalty)

        normalized_gap = (safe_distance - obstacle_distance) / max(safe_distance - danger_distance, 1e-6)
        return -float(normalized_gap**2)

    def _heading_reward(self) -> float:
        heading_vector_to_goal = self.goal_target - self.position
        if float(np.linalg.norm(heading_vector_to_goal)) <= 1e-8:
            return 0.0

        desired_yaw = math.atan2(float(heading_vector_to_goal[1]), float(heading_vector_to_goal[0]))
        heading_error = wrap_to_pi(desired_yaw - self.yaw)
        normalized_heading_error = abs(heading_error) / math.pi
        return -float(self.config.reward.heading_error_scale * normalized_heading_error)

    def _smoothness_reward(self, prev_action: float, current_action: float) -> float:
        reward_cfg = self.config.reward
        action_magnitude_penalty = float(current_action**2)
        action_delta_penalty = float((current_action - prev_action) ** 2)
        return -float(
            reward_cfg.smooth_action_l2_scale * action_magnitude_penalty
            + reward_cfg.smooth_action_delta_scale * action_delta_penalty
        )

    def _get_reward(
        self,
        prev_goal_distance: float,
        prev_visual_obstacle_obs: VisualObstacleObservation,
        prev_wall_clearance: float,
        prev_action: float,
        current_action: float,
        terminated: bool,
        truncated: bool,
    ) -> tuple[float, dict[str, float]]:
        del prev_visual_obstacle_obs
        del prev_wall_clearance
        del terminated
        del truncated

        reward_cfg = self.config.reward
        obstacle_distance = self._current_obstacle_distance()
        target_reward = self._goal_progress_reward(prev_goal_distance, self.goal_distance)
        obstacle_reward = self._obstacle_avoidance_reward(obstacle_distance)
        heading_reward = self._heading_reward()
        smooth_reward = self._smoothness_reward(prev_action, current_action)

        reward_terms = {
            "target_reward": float(reward_cfg.w_target * target_reward),
            "obstacle_reward": float(reward_cfg.w_obs * obstacle_reward),
            "heading_reward": float(reward_cfg.w_heading * heading_reward),
            "smooth_reward": float(reward_cfg.w_smooth * smooth_reward),
            "step_penalty": -float(reward_cfg.step_penalty),
            "wall_collision_cost": -float(reward_cfg.wall_collision_cost) if self.wall_collision else 0.0,
            "success_reward": float(reward_cfg.success_reward) if self.reached_goal else 0.0,
            "obstacle_distance": float(obstacle_distance) if math.isfinite(obstacle_distance) else math.inf,
            "heading_alignment_error": float(
                abs(wrap_to_pi(math.atan2(
                    float((self.goal_target - self.position)[1]),
                    float((self.goal_target - self.position)[0]),
                ) - self.yaw)) / math.pi
            )
            if float(np.linalg.norm(self.goal_target - self.position)) > 1e-8
            else 0.0,
        }

        reward = float(
            reward_terms["target_reward"]
            + reward_terms["obstacle_reward"]
            + reward_terms["heading_reward"]
            + reward_terms["smooth_reward"]
            + reward_terms["step_penalty"]
            + reward_terms["wall_collision_cost"]
            + reward_terms["success_reward"]
        )
        return reward, reward_terms

    def _check_terminated(self) -> bool:
        goal_contact = self._touches_goal_region()
        if goal_contact and not self.reached_goal:
            self.reached_goal = True
            self.goal_reached_step = int(self.elapsed_steps)

        post_goal_steps = max(
            0,
            int(round(float(self.config.post_goal_duration_sec) / max(self.control_timestep, 1e-8))),
        )
        self.goal_hold_complete = bool(
            self.reached_goal
            and self.goal_reached_step is not None
            and (int(self.elapsed_steps) - int(self.goal_reached_step)) >= post_goal_steps
        )
        self.collided = self.min_obstacle_clearance <= 0.0
        self.wall_collision = self.min_wall_clearance <= 0.0
        self.out_of_bounds = (
            abs(self.position[0]) > self.config.pool_half_length
            or abs(self.position[1]) > self.config.pool_half_width
        )
        return self.goal_hold_complete

    def _check_truncated(self) -> bool:
        if int(self.config.max_episode_steps) <= 0:
            self.timeout = False
            return False
        self.timeout = self.elapsed_steps >= self.config.max_episode_steps
        return self.timeout

    def _termination_reason(self) -> str:
        if self.goal_hold_complete:
            return "goal_reached"
        if self.timeout:
            return "timeout"
        if self.out_of_bounds:
            return "out_of_bounds"
        return "running"

    def _sample_obstacles(self) -> list[CircularObstacle]:
        return sample_circular_obstacles(
            obstacle_config=self.config.obstacle,
            rng=self.np_random,
            pool_half_length=self.config.pool_half_length,
            pool_half_width=self.config.pool_half_width,
            spawn_x_range=self.config.task.spawn_x_range,
            spawn_y_range=self.config.task.spawn_y_range,
            goal_center=self.goal_center,
            goal_half_extents=self.goal_half_extents,
        )

    def _get_local_obstacle_observation(self) -> LocalObstacleObservation:
        sensor_position = self.position + body_to_world_vector(
            np.array([self.config.head_sensor_offset, 0.0], dtype=float),
            self.yaw,
        )
        return get_local_obstacle_observation(
            sensor_position=sensor_position,
            sensor_yaw=self.yaw,
            obstacles=self.obstacles,
            detect_range=self.config.obstacle.obs_detect_range,
            fov_deg=self.config.obstacle.obs_fov_deg,
            safety_margin=self.config.obstacle.safety_margin,
        )

    def _build_info(self, reward_terms: dict[str, float]) -> dict[str, Any]:
        lateral_goal_offset = float(self.position[1] - self.goal_center[1])
        info: dict[str, Any] = {
            "scenario_id": self.active_scenario_id,
            "termination_reason": self._termination_reason(),
            "episode_return": self.episode_return,
            "goal_progress_ratio": self.goal_progress_ratio,
            "path_progress_ratio": self.goal_progress_ratio,
            "distance_to_goal_region": self.goal_distance,
            "goal_target_x": float(self.goal_target[0]),
            "goal_target_y": float(self.goal_target[1]),
            "lateral_goal_offset": lateral_goal_offset,
            "cross_track_error": lateral_goal_offset,
            "imu_ax": float(self.imu_acceleration[0]),
            "imu_ay": float(self.imu_acceleration[1]),
            "imu_yaw_rate": float(self.yaw_rate),
            "raw_action": self.raw_action,
            "filtered_action": self.prev_action,
            "theta_m_target": self.theta_m_target,
            "theta_m": self.theta_m,
            "theta_h": self.theta_h,
            "head_servo_locked": self.head_servo_locked,
            "tail_freq_target": self.tail_freq_target,
            "tail_freq": self.tail_freq,
            "back_servo_command": self.back_servo_command,
            "min_obstacle_clearance": self.min_obstacle_clearance,
            "min_wall_clearance": self.min_wall_clearance,
            "obstacle_detected": bool(self.local_obstacle_obs.detected),
            "visual_obstacle_detected": bool(self.visual_obstacle_obs.detected),
            "visual_obstacle_pixel_fraction": self.visual_obstacle_obs.pixel_fraction,
            "visual_obstacle_center_fraction": self.visual_obstacle_obs.center_fraction,
            "visual_obstacle_nearest_depth": self.visual_obstacle_obs.nearest_depth,
            "success": self.reached_goal,
            "collision": self.collided,
            "wall_collision": self.wall_collision,
            "out_of_bounds": self.out_of_bounds,
            "timeout": self.timeout,
        }
        info.update(reward_terms)
        return info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        if self._plt is None:
            import matplotlib.pyplot as plt

            self._plt = plt

        if self._figure is None or self._axes is None:
            fig_width = self.config.render_size[0] / 100.0
            fig_height = self.config.render_size[1] / 100.0
            self._figure, self._axes = self._plt.subplots(figsize=(fig_width, fig_height))

        from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

        ax = self._axes
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(-self.config.pool_half_length - 0.1, self.config.pool_half_length + 0.1)
        ax.set_ylim(-self.config.pool_half_width - 0.1, self.config.pool_half_width + 0.1)
        ax.set_title("FishPathAvoidEnv Top-Down View")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        pool = Rectangle(
            (-self.config.pool_half_length, -self.config.pool_half_width),
            2.0 * self.config.pool_half_length,
            2.0 * self.config.pool_half_width,
            edgecolor="#335c67",
            facecolor="#d8f3ff",
            linewidth=2.0,
            zorder=0,
        )
        ax.add_patch(pool)

        spawn_x_min, spawn_x_max = self.config.task.spawn_x_range
        spawn_y_min, spawn_y_max = self.config.task.spawn_y_range
        spawn_region = Rectangle(
            (spawn_x_min, spawn_y_min),
            spawn_x_max - spawn_x_min,
            spawn_y_max - spawn_y_min,
            edgecolor="#ee9b00",
            facecolor="#ee9b00",
            linewidth=1.8,
            linestyle="--",
            alpha=0.12,
            zorder=1,
            label="spawn region",
        )
        ax.add_patch(spawn_region)

        goal_region = Rectangle(
            tuple(self.goal_center - self.goal_half_extents),
            2.0 * self.goal_half_extents[0],
            2.0 * self.goal_half_extents[1],
            edgecolor="#2a9d8f",
            facecolor="#2a9d8f",
            linewidth=2.0,
            alpha=0.18,
            zorder=1,
            label="goal region",
        )
        ax.add_patch(goal_region)

        for obstacle in self.obstacles:
            ax.add_patch(
                Circle(
                    obstacle.center,
                    obstacle.radius,
                    facecolor="#e76f51",
                    edgecolor="#8d2e1a",
                    linewidth=1.5,
                    alpha=0.85,
                    zorder=2,
                )
            )

        ax.scatter(self.goal_target[0], self.goal_target[1], color="#2a9d8f", s=70, zorder=4, label="goal target")

        fish_polygon = np.vstack(
            [
                self.position + body_to_world_vector(np.array([0.18, 0.0]), self.yaw),
                self.position + body_to_world_vector(np.array([-0.12, 0.08]), self.yaw),
                self.position + body_to_world_vector(np.array([-0.12, -0.08]), self.yaw),
            ]
        )
        ax.add_patch(Polygon(fish_polygon, closed=True, facecolor="#264653", edgecolor="#1b263b", zorder=5))

        sensor_position = self.position + body_to_world_vector(np.array([self.config.head_sensor_offset, 0.0]), self.yaw)
        ax.add_patch(
            Wedge(
                center=sensor_position,
                r=self.config.obstacle.obs_detect_range,
                theta1=math.degrees(self.yaw - math.radians(self.config.obstacle.obs_fov_deg) / 2.0),
                theta2=math.degrees(self.yaw + math.radians(self.config.obstacle.obs_fov_deg) / 2.0),
                facecolor="#90be6d",
                edgecolor="none",
                alpha=0.15,
                zorder=1,
            )
        )
        ax.scatter(sensor_position[0], sensor_position[1], color="#40916c", s=40, zorder=6)

        velocity_heading = heading_vector(self.yaw)
        ax.arrow(
            self.position[0],
            self.position[1],
            0.2 * velocity_heading[0],
            0.2 * velocity_heading[1],
            width=0.01,
            color="#f4a261",
            zorder=6,
        )
        ax.legend(loc="upper left")
        ax.grid(alpha=0.2)
        self._figure.tight_layout()

        if self.render_mode == "human":
            self._figure.canvas.draw_idle()
            self._plt.pause(1.0 / self.metadata["render_fps"])
            return None

        self._figure.canvas.draw()
        image = np.asarray(self._figure.canvas.buffer_rgba(), dtype=np.uint8)
        return image[..., :3].copy()

    def close(self) -> None:
        viewer = self._get_mujoco_viewer()
        if viewer is not None:
            viewer.close()
        self._viewer = None
        self._viewer_last_wall_time = None
        self._viewer_last_sim_time = None
        self._obs_renderer.close()
        if self._video_renderer is not None:
            self._video_renderer.close()
        self._video_renderer = None
        if self._figure is not None and self._plt is not None:
            self._plt.close(self._figure)
        self._figure = None
        self._axes = None
        self._plt = None
