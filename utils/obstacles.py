from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from configs.default_config import ObstacleConfig
from utils.geometry import wrap_to_pi


@dataclass
class CircularObstacle:
    center: np.ndarray
    radius: float


@dataclass
class LocalObstacleObservation:
    detected: float
    distance_ratio: float
    relative_bearing: float
    safety_margin_ratio: float
    edge_distance: float

    @classmethod
    def empty(cls) -> "LocalObstacleObservation":
        return cls(
            detected=0.0,
            distance_ratio=1.0,
            relative_bearing=0.0,
            safety_margin_ratio=1.0,
            edge_distance=math.inf,
        )


def _interpolate_polyline(path_points: np.ndarray, cumulative_lengths: np.ndarray, arc_length: float) -> np.ndarray:
    if arc_length <= 0.0:
        return path_points[0].copy()
    if arc_length >= cumulative_lengths[-1]:
        return path_points[-1].copy()

    upper_index = int(np.searchsorted(cumulative_lengths, arc_length, side="right"))
    lower_index = max(0, upper_index - 1)
    start_length = cumulative_lengths[lower_index]
    end_length = cumulative_lengths[upper_index]
    segment = path_points[upper_index] - path_points[lower_index]
    if end_length <= start_length:
        return path_points[lower_index].copy()
    alpha = (arc_length - start_length) / (end_length - start_length)
    return path_points[lower_index] + alpha * segment


def _tangent_at_arc_length(path_points: np.ndarray, cumulative_lengths: np.ndarray, arc_length: float) -> np.ndarray:
    if arc_length <= 0.0:
        segment = path_points[1] - path_points[0]
    elif arc_length >= cumulative_lengths[-1]:
        segment = path_points[-1] - path_points[-2]
    else:
        upper_index = int(np.searchsorted(cumulative_lengths, arc_length, side="right"))
        lower_index = max(0, upper_index - 1)
        segment = path_points[upper_index] - path_points[lower_index]
    return segment / (np.linalg.norm(segment) + 1e-8)


def _sample_point_between_spawn_and_goal(
    rng: np.random.Generator,
    spawn_center: np.ndarray,
    goal_center: np.ndarray,
    longitudinal_margin: float,
    lateral_half_width: float,
) -> np.ndarray:
    segment = goal_center - spawn_center
    segment_length = float(np.linalg.norm(segment))
    if segment_length <= 1e-8:
        tangent = np.array([1.0, 0.0], dtype=float)
    else:
        tangent = segment / segment_length
    normal = np.array([-tangent[1], tangent[0]], dtype=float)

    progress_margin = float(np.clip(longitudinal_margin / max(segment_length, 1e-8), 0.0, 0.45))
    progress = float(rng.uniform(progress_margin, 1.0 - progress_margin))
    lateral_offset = float(rng.uniform(-lateral_half_width, lateral_half_width))
    return spawn_center + progress * segment + lateral_offset * normal


def _corridor_frame(spawn_center: np.ndarray, goal_center: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    segment = goal_center - spawn_center
    segment_length = float(np.linalg.norm(segment))
    if segment_length <= 1e-8:
        tangent = np.array([1.0, 0.0], dtype=float)
    else:
        tangent = segment / segment_length
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    return tangent, normal, segment_length


def _sample_random_obstacles(
    obstacle_config: ObstacleConfig,
    rng: np.random.Generator,
    pool_half_length: float,
    pool_half_width: float,
    spawn_min: np.ndarray,
    spawn_max: np.ndarray,
    goal_center: np.ndarray,
    goal_half_extents: np.ndarray,
) -> list[CircularObstacle]:
    goal_min = goal_center - goal_half_extents
    goal_max = goal_center + goal_half_extents
    requested_min = max(0, int(obstacle_config.min_count))
    requested_max = max(requested_min, int(obstacle_config.max_count))
    obstacle_count = int(rng.integers(requested_min, requested_max + 1))
    placed_obstacles: list[CircularObstacle] = []
    max_attempts = max(1, int(obstacle_config.max_sampling_attempts)) * max(obstacle_count, 1) * 10

    attempts = 0
    while attempts < max_attempts and len(placed_obstacles) < obstacle_count:
        attempts += 1
        radius = float(rng.uniform(obstacle_config.radius_min, obstacle_config.radius_max))
        wall_margin = radius + 0.08
        center = np.array(
            [
                float(rng.uniform(-pool_half_length + wall_margin, pool_half_length - wall_margin)),
                float(rng.uniform(-pool_half_width + wall_margin, pool_half_width - wall_margin)),
            ],
            dtype=float,
        )

        spawn_clearance = float(obstacle_config.start_goal_clearance) + radius
        if np.all(center >= spawn_min - spawn_clearance) and np.all(center <= spawn_max + spawn_clearance):
            continue

        goal_clearance = float(obstacle_config.start_goal_clearance) + radius
        if np.all(center >= goal_min - goal_clearance) and np.all(center <= goal_max + goal_clearance):
            continue

        collision_with_existing = False
        for existing in placed_obstacles:
            min_center_distance = float(existing.radius + radius + obstacle_config.obstacle_spacing)
            if float(np.linalg.norm(center - existing.center)) <= min_center_distance:
                collision_with_existing = True
                break
        if collision_with_existing:
            continue

        placed_obstacles.append(CircularObstacle(center=center.astype(float), radius=radius))

    if placed_obstacles:
        return placed_obstacles

    fallback_radius = 0.5 * (obstacle_config.radius_min + obstacle_config.radius_max)
    fallback_center = np.array([0.0, 0.0], dtype=float)
    return [CircularObstacle(center=fallback_center, radius=fallback_radius)]


def sample_circular_obstacles(
    obstacle_config: ObstacleConfig,
    rng: np.random.Generator,
    pool_half_length: float,
    pool_half_width: float,
    spawn_x_range: tuple[float, float],
    spawn_y_range: tuple[float, float],
    goal_center: np.ndarray,
    goal_half_extents: np.ndarray,
) -> list[CircularObstacle]:
    spawn_min = np.array([spawn_x_range[0], spawn_y_range[0]], dtype=float)
    spawn_max = np.array([spawn_x_range[1], spawn_y_range[1]], dtype=float)
    return _sample_random_obstacles(
        obstacle_config=obstacle_config,
        rng=rng,
        pool_half_length=pool_half_length,
        pool_half_width=pool_half_width,
        spawn_min=spawn_min,
        spawn_max=spawn_max,
        goal_center=goal_center,
        goal_half_extents=goal_half_extents,
    )


def get_local_obstacle_observation(
    sensor_position: np.ndarray,
    sensor_yaw: float,
    obstacles: list[CircularObstacle],
    detect_range: float,
    fov_deg: float,
    safety_margin: float,
) -> LocalObstacleObservation:
    if not obstacles:
        return LocalObstacleObservation.empty()

    half_fov = math.radians(fov_deg) / 2.0
    nearest_distance = math.inf
    nearest_bearing = 0.0

    for obstacle in obstacles:
        offset = obstacle.center - sensor_position
        center_distance = float(np.linalg.norm(offset))
        edge_distance = center_distance - obstacle.radius
        bearing = wrap_to_pi(math.atan2(offset[1], offset[0]) - sensor_yaw)
        if edge_distance > detect_range:
            continue
        if abs(bearing) > half_fov:
            continue
        if edge_distance < nearest_distance:
            nearest_distance = edge_distance
            nearest_bearing = bearing

    if not math.isfinite(nearest_distance):
        return LocalObstacleObservation.empty()

    distance_ratio = float(np.clip(nearest_distance / max(detect_range, 1e-6), 0.0, 1.0))
    safety_margin_ratio = float(np.clip(nearest_distance / max(safety_margin, 1e-6), -2.0, 2.0))
    return LocalObstacleObservation(
        detected=1.0,
        distance_ratio=distance_ratio,
        relative_bearing=nearest_bearing,
        safety_margin_ratio=safety_margin_ratio,
        edge_distance=nearest_distance,
    )
