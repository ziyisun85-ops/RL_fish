from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from utils.obstacles import CircularObstacle


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATH_FIELD_NAMES = frozenset(
    {
        "output_root",
        "train_manifest",
        "test_manifest",
        "json_path",
        "topdown_path",
        "path",
        "xml_path",
        "log_dir",
    }
)


@dataclass
class FixedScenario:
    scenario_id: str
    spawn_position: np.ndarray
    spawn_yaw: float
    goal_center: np.ndarray
    goal_half_extents: np.ndarray
    obstacles: list[CircularObstacle]
    source_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "source_seed": self.source_seed,
            "spawn_position": [float(self.spawn_position[0]), float(self.spawn_position[1])],
            "spawn_yaw": float(self.spawn_yaw),
            "goal_center": [float(self.goal_center[0]), float(self.goal_center[1])],
            "goal_half_extents": [float(self.goal_half_extents[0]), float(self.goal_half_extents[1])],
            "obstacles": [
                {
                    "center": [float(obstacle.center[0]), float(obstacle.center[1])],
                    "radius": float(obstacle.radius),
                }
                for obstacle in self.obstacles
            ],
        }


def fixed_scenario_from_dict(data: dict[str, Any]) -> FixedScenario:
    obstacles = [
        CircularObstacle(
            center=np.asarray(obstacle["center"], dtype=float),
            radius=float(obstacle["radius"]),
        )
        for obstacle in data.get("obstacles", [])
    ]
    return FixedScenario(
        scenario_id=str(data.get("scenario_id", "scenario")),
        source_seed=None if data.get("source_seed") is None else int(data["source_seed"]),
        spawn_position=np.asarray(data["spawn_position"], dtype=float),
        spawn_yaw=float(data["spawn_yaw"]),
        goal_center=np.asarray(data["goal_center"], dtype=float),
        goal_half_extents=np.asarray(data["goal_half_extents"], dtype=float),
        obstacles=obstacles,
    )


def save_fixed_scenario(scenario: FixedScenario, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(scenario.to_dict(), handle, indent=2, ensure_ascii=False)
    return path


def load_fixed_scenario(input_path: str | Path) -> FixedScenario:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return fixed_scenario_from_dict(data)


def _normalize_path_text(value: str, *, base_dir: Path) -> str:
    text = str(value).strip()
    if not text:
        return text

    path = Path(text)
    resolved = (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        return Path(os.path.relpath(resolved, PROJECT_ROOT)).as_posix()
    except ValueError:
        return resolved.as_posix()


def _normalize_manifest_paths(payload: Any, *, base_dir: Path) -> Any:
    if isinstance(payload, dict):
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            if key in PATH_FIELD_NAMES and isinstance(value, str):
                normalized[key] = _normalize_path_text(value, base_dir=base_dir)
            else:
                normalized[key] = _normalize_manifest_paths(value, base_dir=base_dir)
        return normalized
    if isinstance(payload, list):
        return [_normalize_manifest_paths(item, base_dir=base_dir) for item in payload]
    return payload


@lru_cache(maxsize=None)
def load_dataset_env_config_for_scenario(scenario_path: str | Path) -> dict[str, Any] | None:
    path = Path(scenario_path).resolve()
    for parent in path.parents:
        manifest_path = parent / "dataset_manifest.json"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = _normalize_manifest_paths(json.load(handle), base_dir=manifest_path.parent)
        env_config = manifest.get("config", {}).get("env")
        if isinstance(env_config, dict):
            return env_config
        return None
    return None
