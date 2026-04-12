from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import matplotlib
from matplotlib import image as mpimg

from configs.default_config import PROJECT_ROOT, config_to_dict, make_config
from envs import FishPathAvoidEnv
from utils.scenario_io import save_fixed_scenario

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a large-pool fish scenario dataset with JSON files and top-down previews.",
    )
    parser.add_argument("--train-count", type=int, default=100, help="Number of training scenarios to generate.")
    parser.add_argument("--test-count", type=int, default=100, help="Number of test scenarios to generate.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str((PROJECT_ROOT / "scenarios" / "large_pool_dataset_200").resolve()),
        help="Root directory for the generated dataset.",
    )
    parser.add_argument("--train-base-seed", type=int, default=1000, help="Base RNG seed for training scenarios.")
    parser.add_argument("--test-base-seed", type=int, default=1000000, help="Base RNG seed for test scenarios.")
    parser.add_argument("--pool-half-length", type=float, default=4.8, help="Half pool length in meters.")
    parser.add_argument("--pool-half-width", type=float, default=2.6, help="Half pool width in meters.")
    parser.add_argument("--spawn-x-min", type=float, default=-4.05, help="Spawn x-range minimum.")
    parser.add_argument("--spawn-x-max", type=float, default=-3.55, help="Spawn x-range maximum.")
    parser.add_argument("--spawn-y-min", type=float, default=-1.10, help="Spawn y-range minimum.")
    parser.add_argument("--spawn-y-max", type=float, default=1.10, help="Spawn y-range maximum.")
    parser.add_argument("--goal-center-x", type=float, default=4.0, help="Goal center x position.")
    parser.add_argument("--goal-center-y", type=float, default=0.0, help="Goal center y position.")
    parser.add_argument("--goal-half-x", type=float, default=0.28, help="Goal half extent along x.")
    parser.add_argument("--goal-half-y", type=float, default=0.45, help="Goal half extent along y.")
    parser.add_argument("--min-obstacles", type=int, default=10, help="Minimum obstacle count per scenario.")
    parser.add_argument("--max-obstacles", type=int, default=12, help="Maximum obstacle count per scenario.")
    parser.add_argument("--radius-min", type=float, default=0.10, help="Minimum obstacle radius.")
    parser.add_argument("--radius-max", type=float, default=0.18, help="Maximum obstacle radius.")
    parser.add_argument("--obstacle-spacing", type=float, default=0.16, help="Minimum extra obstacle spacing.")
    parser.add_argument("--start-goal-clearance", type=float, default=0.90, help="Spawn/goal clearance around obstacles.")
    parser.add_argument(
        "--max-resample-attempts",
        type=int,
        default=200,
        help="Maximum scenario resampling attempts before giving up on one scenario.",
    )
    return parser.parse_args()


def build_dataset_config(args: argparse.Namespace):
    config = make_config()
    env_cfg = config.env

    env_cfg.pool_half_length = float(args.pool_half_length)
    env_cfg.pool_half_width = float(args.pool_half_width)
    env_cfg.render_size = (1400, 800)

    env_cfg.task.spawn_x_range = (float(args.spawn_x_min), float(args.spawn_x_max))
    env_cfg.task.spawn_y_range = (float(args.spawn_y_min), float(args.spawn_y_max))
    env_cfg.task.goal_center = (float(args.goal_center_x), float(args.goal_center_y))
    env_cfg.task.goal_half_extents = (float(args.goal_half_x), float(args.goal_half_y))

    env_cfg.obstacle.min_count = int(args.min_obstacles)
    env_cfg.obstacle.max_count = int(args.max_obstacles)
    env_cfg.obstacle.radius_min = float(args.radius_min)
    env_cfg.obstacle.radius_max = float(args.radius_max)
    env_cfg.obstacle.obstacle_spacing = float(args.obstacle_spacing)
    env_cfg.obstacle.start_goal_clearance = float(args.start_goal_clearance)
    env_cfg.obstacle.resample_interval_episodes = 1
    env_cfg.obstacle.max_sampling_attempts = max(200, int(args.max_resample_attempts))

    return config


def ensure_valid_obstacle_count(env: FishPathAvoidEnv, min_count: int, max_count: int) -> bool:
    obstacle_count = len(env.obstacles)
    return int(min_count) <= obstacle_count <= int(max_count)


def generate_split(
    *,
    split_name: str,
    count: int,
    base_seed: int,
    config,
    output_root: Path,
    max_resample_attempts: int,
) -> list[dict[str, object]]:
    split_root = output_root / split_name
    json_dir = split_root / "json"
    preview_dir = split_root / "topdown"
    json_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []

    for index in range(1, int(count) + 1):
        scenario_name = f"{split_name}_env_{index:03d}"
        accepted = False

        for attempt in range(max(1, int(max_resample_attempts))):
            scenario_seed = int(base_seed) + (index - 1) * 1000 + attempt
            env = FishPathAvoidEnv(config=config.env, render_mode="rgb_array")
            try:
                env.reset(seed=scenario_seed)
                if not ensure_valid_obstacle_count(env, config.env.obstacle.min_count, config.env.obstacle.max_count):
                    continue

                scenario = env.export_fixed_scenario(
                    scenario_id=scenario_name,
                    source_seed=scenario_seed,
                )
                preview = env.render()
            finally:
                env.close()

            scenario_path = json_dir / f"{scenario_name}.json"
            preview_path = preview_dir / f"{scenario_name}.png"
            save_fixed_scenario(scenario, scenario_path)
            if preview is None:
                raise RuntimeError(f"Expected rgb_array preview for {scenario_name}, got None.")
            mpimg.imsave(preview_path, preview)

            manifest_entries.append(
                {
                    "scenario_id": scenario_name,
                    "split": split_name,
                    "source_seed": scenario_seed,
                    "obstacle_count": len(scenario.obstacles),
                    "json_path": str(scenario_path),
                    "topdown_path": str(preview_path),
                }
            )
            print(
                f"[{split_name}] Saved {scenario_name}: "
                f"seed={scenario_seed}, obstacles={len(scenario.obstacles)}"
            )
            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                f"Failed to generate a valid {split_name} scenario after {max_resample_attempts} attempts: {scenario_name}",
            )

    manifest_path = split_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": split_name,
                "count": len(manifest_entries),
                "scenarios": manifest_entries,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[{split_name}] Saved manifest to {manifest_path}")
    return manifest_entries


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = build_dataset_config(args)

    train_entries = generate_split(
        split_name="train",
        count=args.train_count,
        base_seed=args.train_base_seed,
        config=deepcopy(config),
        output_root=output_root,
        max_resample_attempts=args.max_resample_attempts,
    )
    test_entries = generate_split(
        split_name="test",
        count=args.test_count,
        base_seed=args.test_base_seed,
        config=deepcopy(config),
        output_root=output_root,
        max_resample_attempts=args.max_resample_attempts,
    )

    dataset_manifest = {
        "train_count": len(train_entries),
        "test_count": len(test_entries),
        "output_root": str(output_root),
        "config": config_to_dict(config),
        "splits": {
            "train_manifest": str((output_root / "train" / "manifest.json").resolve()),
            "test_manifest": str((output_root / "test" / "manifest.json").resolve()),
        },
    }
    dataset_manifest_path = output_root / "dataset_manifest.json"
    with dataset_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_manifest, handle, indent=2, ensure_ascii=False)
    print(f"Saved dataset manifest to {dataset_manifest_path}")


if __name__ == "__main__":
    main()
