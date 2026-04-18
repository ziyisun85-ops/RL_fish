from __future__ import annotations

from typing import Any

import train as train_module
from envs.fish_env import FishPathAvoidEnv


EPISODE_METRICS_FIELDNAMES = [
    "run_id",
    "scenario_path",
    "scenario_id",
    "episode_index",
    "num_timesteps",
    "train_loss",
    "episode_reward",
    "episode_length",
    "episode_time_sec",
    "episode_train_time_sec",
    "termination_reason",
    "episode_return",
    "goal_progress_ratio",
    "avg_goal_progress_ratio",
    "distance_to_goal_region",
    "visual_obstacle_detected",
    "visual_obstacle_pixel_fraction",
    "visual_obstacle_center_fraction",
    "visual_obstacle_nearest_depth",
    "success",
    "collision",
    "wall_collision",
    "obstacle_collision_count",
    "wall_collision_count",
    "out_of_bounds",
    "timeout",
]


def _convert_legacy_episode_row(
    row: dict[str, str],
    *,
    episode_index: int,
    scenario_path,
) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id", "legacy"),
        "scenario_path": row.get(
            "scenario_path",
            "" if scenario_path is None else (train_module._relative_path_text(scenario_path) or ""),
        ),
        "scenario_id": row.get("scenario_id", ""),
        "episode_index": int(episode_index),
        "num_timesteps": train_module._parse_csv_int(row.get("num_timesteps"), 0),
        "train_loss": train_module._parse_csv_float(row.get("train_loss"), float("nan")),
        "episode_reward": train_module._parse_csv_float(row.get("episode_reward"), 0.0),
        "episode_length": train_module._parse_csv_int(row.get("episode_length"), 0),
        "episode_time_sec": train_module._parse_csv_float(row.get("episode_time_sec"), 0.0),
        "episode_train_time_sec": train_module._parse_csv_float(row.get("episode_train_time_sec"), 0.0),
        "termination_reason": str(row.get("termination_reason", "unknown")),
        "episode_return": train_module._parse_csv_float(row.get("episode_return", row.get("episode_reward", 0.0)), 0.0),
        "goal_progress_ratio": train_module._parse_csv_float(row.get("goal_progress_ratio"), 0.0),
        "avg_goal_progress_ratio": train_module._parse_csv_float(
            row.get("avg_goal_progress_ratio", row.get("goal_progress_ratio")),
            train_module._parse_csv_float(row.get("goal_progress_ratio"), 0.0),
        ),
        "distance_to_goal_region": train_module._parse_csv_float(row.get("distance_to_goal_region"), 0.0),
        "visual_obstacle_detected": train_module._parse_csv_bool(row.get("visual_obstacle_detected")),
        "visual_obstacle_pixel_fraction": train_module._parse_csv_float(row.get("visual_obstacle_pixel_fraction"), 0.0),
        "visual_obstacle_center_fraction": train_module._parse_csv_float(row.get("visual_obstacle_center_fraction"), 0.0),
        "visual_obstacle_nearest_depth": train_module._parse_csv_float(row.get("visual_obstacle_nearest_depth"), 0.0),
        "success": train_module._parse_csv_bool(row.get("success")),
        "collision": train_module._parse_csv_bool(row.get("collision")),
        "wall_collision": train_module._parse_csv_bool(row.get("wall_collision")),
        "obstacle_collision_count": train_module._parse_csv_int(
            row.get("obstacle_collision_count"),
            int(train_module._parse_csv_bool(row.get("collision"))),
        ),
        "wall_collision_count": train_module._parse_csv_int(
            row.get("wall_collision_count"),
            int(train_module._parse_csv_bool(row.get("wall_collision"))),
        ),
        "out_of_bounds": train_module._parse_csv_bool(row.get("out_of_bounds")),
        "timeout": train_module._parse_csv_bool(row.get("timeout")),
    }


def _reset_episode_stats(env: FishPathAvoidEnv) -> None:
    env.avg_goal_progress_ratio = 0.0
    env._goal_progress_ratio_sum = 0.0
    env._goal_progress_ratio_step_count = 0
    env.obstacle_collision_count = 0
    env.wall_collision_count = 0
    env._obstacle_collision_active = False
    env._wall_collision_active = False


def _update_episode_statistics(env: FishPathAvoidEnv) -> None:
    env._goal_progress_ratio_sum += float(getattr(env, "goal_progress_ratio", 0.0))
    env._goal_progress_ratio_step_count += 1
    env.avg_goal_progress_ratio = float(
        env._goal_progress_ratio_sum / max(int(env._goal_progress_ratio_step_count), 1)
    )

    obstacle_collision_active = bool(getattr(env, "collided", False))
    if obstacle_collision_active and not bool(getattr(env, "_obstacle_collision_active", False)):
        env.obstacle_collision_count += 1
    env._obstacle_collision_active = obstacle_collision_active

    wall_collision_active = bool(getattr(env, "wall_collision", False))
    if wall_collision_active and not bool(getattr(env, "_wall_collision_active", False)):
        env.wall_collision_count += 1
    env._wall_collision_active = wall_collision_active


def install_train_and_env_patches() -> None:
    train_module.EPISODE_METRICS_FIELDNAMES = list(EPISODE_METRICS_FIELDNAMES)
    train_module._convert_legacy_episode_row = _convert_legacy_episode_row

    if "_update_episode_statistics" in FishPathAvoidEnv.__dict__:
        return

    original_init = FishPathAvoidEnv.__init__
    original_reset = FishPathAvoidEnv.reset
    original_step = FishPathAvoidEnv.step

    def patched_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        _reset_episode_stats(self)

    def patched_reset(self, *args, **kwargs):
        observation, info = original_reset(self, *args, **kwargs)
        _reset_episode_stats(self)
        info["avg_goal_progress_ratio"] = float(self.avg_goal_progress_ratio)
        info["obstacle_collision_count"] = int(self.obstacle_collision_count)
        info["wall_collision_count"] = int(self.wall_collision_count)
        return observation, info

    def patched_step(self, action):
        observation, reward, terminated, truncated, info = original_step(self, action)
        if not hasattr(self, "_goal_progress_ratio_sum"):
            _reset_episode_stats(self)
        _update_episode_statistics(self)
        info["avg_goal_progress_ratio"] = float(self.avg_goal_progress_ratio)
        info["obstacle_collision_count"] = int(self.obstacle_collision_count)
        info["wall_collision_count"] = int(self.wall_collision_count)
        return observation, reward, terminated, truncated, info

    FishPathAvoidEnv.__init__ = patched_init
    FishPathAvoidEnv.reset = patched_reset
    FishPathAvoidEnv.step = patched_step
    FishPathAvoidEnv._update_episode_statistics = _update_episode_statistics
