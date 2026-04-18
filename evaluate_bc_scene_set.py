from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

from configs.default_config import PROJECT_ROOT, make_config
from evaluate_bc_rl import select_device
from utils.policy_utils import build_ppo_model, build_vec_env, load_actor_state_dict, make_single_env


DEFAULT_WEIGHT_PATHS = [
    PROJECT_ROOT / "runs" / "bc_pretrain" / "large_pool_scene_sets" / "1_20__60_80" / "bc_large_pool_1_20__60_80.zip",
    PROJECT_ROOT / "runs" / "bc_pretrain" / "large_pool_scene_sets" / "1_20__60_100" / "bc_large_pool_1_20__60_100.zip",
    PROJECT_ROOT / "runs" / "bc_pretrain" / "large_pool_scene_sets" / "1_40__60_100" / "bc_large_pool_1_40__60_100.zip",
    PROJECT_ROOT / "runs" / "bc_pretrain" / "large_pool_scene_sets" / "1_100" / "bc_large_pool_1_100.zip",
]
SCENARIO_ID_PATTERN = re.compile(r"train_env_(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BC checkpoints on a fixed scene set and report per-weight success rates.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        nargs="+",
        default=None,
        help="One or more BC checkpoints (.zip full model or .pth/.pt actor checkpoint).",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Optional directory containing BC checkpoints to evaluate recursively.",
    )
    parser.add_argument(
        "--weights-glob",
        type=str,
        default="**/*_actor.pth",
        help="Recursive glob used under --weights-dir. Defaults to actor checkpoints only.",
    )
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=str((PROJECT_ROOT / "scenarios" / "large_pool_dataset_200" / "train" / "json").resolve()),
        help="Directory containing train_env_XXX.json files.",
    )
    parser.add_argument(
        "--scenario-glob",
        type=str,
        default="train_env_*.json",
        help="Glob used to collect scenario files under --scenario-dir.",
    )
    parser.add_argument("--scene-start", type=int, default=1, help="1-based inclusive scene start index.")
    parser.add_argument("--scene-end", type=int, default=100, help="1-based inclusive scene end index.")
    parser.add_argument(
        "--scene-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit 1-based scene indices to evaluate. Overrides --scene-start/--scene-end.",
    )
    parser.add_argument(
        "--uniform-block-size",
        type=int,
        default=0,
        help="Optional block size used to uniformly sample scenes from each block. Use with --uniform-scenes-per-block.",
    )
    parser.add_argument(
        "--uniform-scenes-per-block",
        type=int,
        default=0,
        help="Optional number of uniformly selected scenes from each block. Use with --uniform-block-size.",
    )
    parser.add_argument("--episodes-per-scene", type=int, default=1, help="Evaluation episodes per scene.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device: cuda, cuda:0, cpu, or auto.")
    parser.add_argument(
        "--output",
        type=str,
        default=str((PROJECT_ROOT / "runs" / "eval" / "bc_train_scene_set_ep1.json").resolve()),
        help="JSON output path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to store JSON/CSV outputs. When set, multiple result files are written there.",
    )
    return parser.parse_args()


def _apply_overrides(target: Any, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
            continue
        if isinstance(current, tuple) and isinstance(value, list):
            setattr(target, key, tuple(value))
            continue
        setattr(target, key, value)


def extract_scene_index(scenario_path: Path) -> int:
    match = SCENARIO_ID_PATTERN.search(scenario_path.name)
    if match is None:
        raise ValueError(f"Unsupported train-scene filename: {scenario_path.name}")
    return int(match.group(1))


def build_uniform_scene_indices(
    *,
    scene_start: int,
    scene_end: int,
    block_size: int,
    scenes_per_block: int,
) -> list[int]:
    if block_size <= 0:
        raise ValueError("--uniform-block-size must be positive.")
    if scenes_per_block <= 0:
        raise ValueError("--uniform-scenes-per-block must be positive.")

    selected: list[int] = []
    block_start = int(scene_start)
    while block_start <= int(scene_end):
        block_end = min(block_start + int(block_size) - 1, int(scene_end))
        block_length = block_end - block_start + 1
        if scenes_per_block > block_length:
            raise ValueError(
                "Requested more uniformly sampled scenes than available in a block: "
                f"block {block_start}-{block_end}, requested {scenes_per_block}"
            )

        sampled_positions = np.linspace(0, block_length - 1, num=int(scenes_per_block))
        block_indices: list[int] = []
        seen: set[int] = set()
        for position in sampled_positions.tolist():
            scene_index = block_start + int(round(float(position)))
            scene_index = max(block_start, min(block_end, scene_index))
            if scene_index in seen:
                continue
            seen.add(scene_index)
            block_indices.append(scene_index)

        candidate_index = block_start
        while len(block_indices) < int(scenes_per_block):
            if candidate_index not in seen:
                seen.add(candidate_index)
                block_indices.append(candidate_index)
            candidate_index += 1

        selected.extend(sorted(block_indices))
        block_start += int(block_size)

    return selected


def resolve_selected_scenarios(
    *,
    scenario_paths: list[Path],
    scene_start: int,
    scene_end: int,
    scene_indices: list[int] | None,
    uniform_block_size: int,
    uniform_scenes_per_block: int,
) -> tuple[list[int], list[Path]]:
    if scene_indices is not None and (uniform_block_size > 0 or uniform_scenes_per_block > 0):
        raise ValueError("Use either --scene-indices or the uniform block sampling args, not both.")
    if (uniform_block_size > 0) != (uniform_scenes_per_block > 0):
        raise ValueError("--uniform-block-size and --uniform-scenes-per-block must be used together.")

    scenario_by_index = {extract_scene_index(path): path for path in scenario_paths}
    if scene_indices is not None:
        selected_indices = [int(index) for index in scene_indices]
    elif uniform_block_size > 0:
        selected_indices = build_uniform_scene_indices(
            scene_start=int(scene_start),
            scene_end=int(scene_end),
            block_size=int(uniform_block_size),
            scenes_per_block=int(uniform_scenes_per_block),
        )
    else:
        selected_indices = list(range(int(scene_start), int(scene_end) + 1))

    missing_indices = [index for index in selected_indices if index not in scenario_by_index]
    if missing_indices:
        raise RuntimeError(f"Selected scene indices are missing from {scenario_paths[0].parent}: {missing_indices}")

    selected_paths = [scenario_by_index[index] for index in selected_indices]
    return selected_indices, selected_paths


def _load_metrics_json(weight_path: Path) -> dict[str, Any]:
    if weight_path.name.endswith("_actor.pth") or weight_path.name.endswith("_actor.pt"):
        metrics_candidate = weight_path.with_name(weight_path.name.replace("_actor.pth", "_metrics.json").replace("_actor.pt", "_metrics.json"))
        if metrics_candidate.exists():
            return json.loads(metrics_candidate.read_text(encoding="utf-8"))
    else:
        sibling_metrics = weight_path.with_name(f"{weight_path.stem}_metrics.json")
        if sibling_metrics.exists():
            return json.loads(sibling_metrics.read_text(encoding="utf-8"))

    parent_metrics = weight_path.parent.parent / f"{weight_path.stem.replace('_actor', '')}_metrics.json"
    if parent_metrics.exists():
        return json.loads(parent_metrics.read_text(encoding="utf-8"))
    return {}


def _extract_bc_training_metadata(weight_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "epoch": payload.get("epoch"),
        "train_loss": payload.get("train_loss"),
        "val_loss": payload.get("val_loss"),
        "best_reference_loss": payload.get("best_reference_loss"),
        "dataset_path": payload.get("dataset_path"),
    }

    metrics_payload = _load_metrics_json(weight_path)
    epoch_metrics = metrics_payload.get("epoch_metrics")
    if isinstance(epoch_metrics, list) and epoch_metrics:
        if metadata.get("epoch") is None:
            metadata["epoch"] = epoch_metrics[-1].get("epoch")
        if metadata.get("train_loss") is None:
            metadata["train_loss"] = epoch_metrics[-1].get("train_loss")
        if metadata.get("val_loss") is None:
            metadata["val_loss"] = epoch_metrics[-1].get("val_loss")
    if metadata.get("best_reference_loss") is None:
        metadata["best_reference_loss"] = metrics_payload.get("best_reference_loss")
    if metadata.get("dataset_path") is None:
        metadata["dataset_path"] = metrics_payload.get("dataset_path")
    return metadata


def resolve_weight_paths(args: argparse.Namespace) -> list[Path]:
    if args.weights_dir is not None and args.weights is not None:
        raise ValueError("Use either --weights or --weights-dir, not both.")

    if args.weights_dir is not None:
        weights_dir = Path(args.weights_dir).resolve()
        if not weights_dir.exists():
            raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
        if not weights_dir.is_dir():
            raise NotADirectoryError(f"Weights directory is not a directory: {weights_dir}")
        weight_paths = sorted(path.resolve() for path in weights_dir.glob(str(args.weights_glob)) if path.is_file())
        if not weight_paths:
            raise RuntimeError(
                f"No checkpoints matched '{args.weights_glob}' under {weights_dir}"
            )
        return weight_paths

    if args.weights is not None:
        return [Path(path).resolve() for path in args.weights]

    return [Path(path).resolve() for path in DEFAULT_WEIGHT_PATHS]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_result_bundle(output_dir: Path, results: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    weight_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for weight_result in results.get("weights", []):
        weight_rows.append(
            {
                "weight_name": weight_result.get("weight_name"),
                "weight_path": weight_result.get("weight_path"),
                "load_mode": weight_result.get("load_mode"),
                "epoch": weight_result.get("epoch"),
                "train_loss": weight_result.get("train_loss"),
                "val_loss": weight_result.get("val_loss"),
                "best_reference_loss": weight_result.get("best_reference_loss"),
                "dataset_path": weight_result.get("dataset_path"),
                "scene_count": weight_result.get("scene_count"),
                "episodes_per_scene": weight_result.get("episodes_per_scene"),
                "total_episode_count": weight_result.get("total_episode_count"),
                "overall_success_count": weight_result.get("overall_success_count"),
                "overall_success_rate": weight_result.get("overall_success_rate"),
                "overall_avg_reward": weight_result.get("overall_avg_reward"),
                "overall_avg_steps": weight_result.get("overall_avg_steps"),
                "weight_eval_wall_time_sec": weight_result.get("weight_eval_wall_time_sec"),
                "termination_reason_counts_json": json.dumps(
                    weight_result.get("overall_termination_reason_counts", {}),
                    ensure_ascii=False,
                ),
            }
        )

        for scene_result in weight_result.get("scenes", []):
            scene_rows.append(
                {
                    "weight_name": weight_result.get("weight_name"),
                    "scene_name": scene_result.get("scene_name"),
                    "scenario_path": scene_result.get("scenario_path"),
                    "episode_count": scene_result.get("episode_count"),
                    "success_count": scene_result.get("success_count"),
                    "success_rate": scene_result.get("success_rate"),
                    "avg_reward": scene_result.get("avg_reward"),
                    "avg_steps": scene_result.get("avg_steps"),
                    "scene_eval_wall_time_sec": scene_result.get("scene_eval_wall_time_sec"),
                    "termination_reason_counts_json": json.dumps(
                        scene_result.get("termination_reason_counts", {}),
                        ensure_ascii=False,
                    ),
                }
            )

            for episode in scene_result.get("episodes", []):
                episode_rows.append(
                    {
                        "weight_name": weight_result.get("weight_name"),
                        "scene_name": scene_result.get("scene_name"),
                        "scenario_path": scene_result.get("scenario_path"),
                        "seed": episode.get("seed"),
                        "success": episode.get("success"),
                        "termination_reason": episode.get("termination_reason"),
                        "episode_reward": episode.get("episode_reward"),
                        "episode_steps": episode.get("episode_steps"),
                        "episode_wall_time_sec": episode.get("episode_wall_time_sec"),
                        "collision": episode.get("collision"),
                        "wall_collision": episode.get("wall_collision"),
                        "timeout": episode.get("timeout"),
                    }
                )

    write_csv(
        output_dir / "weight_summary.csv",
        [
            "weight_name",
            "weight_path",
            "load_mode",
            "epoch",
            "train_loss",
            "val_loss",
            "best_reference_loss",
            "dataset_path",
            "scene_count",
            "episodes_per_scene",
            "total_episode_count",
            "overall_success_count",
            "overall_success_rate",
            "overall_avg_reward",
            "overall_avg_steps",
            "weight_eval_wall_time_sec",
            "termination_reason_counts_json",
        ],
        weight_rows,
    )
    write_csv(
        output_dir / "scene_summary.csv",
        [
            "weight_name",
            "scene_name",
            "scenario_path",
            "episode_count",
            "success_count",
            "success_rate",
            "avg_reward",
            "avg_steps",
            "scene_eval_wall_time_sec",
            "termination_reason_counts_json",
        ],
        scene_rows,
    )
    write_csv(
        output_dir / "episode_details.csv",
        [
            "weight_name",
            "scene_name",
            "scenario_path",
            "seed",
            "success",
            "termination_reason",
            "episode_reward",
            "episode_steps",
            "episode_wall_time_sec",
            "collision",
            "wall_collision",
            "timeout",
        ],
        episode_rows,
    )


def build_bc_eval_model(weight_path: Path, device: str) -> tuple[PPO, Any, dict[str, Any]]:
    suffix = weight_path.suffix.lower()
    if suffix == ".zip":
        model = PPO.load(str(weight_path), device=device)
        model.policy.set_training_mode(False)
        metadata = {
            "weight_name": weight_path.name,
            "weight_path": str(weight_path),
            "weight_suffix": suffix,
            "load_mode": "sb3_zip",
        }
        metadata.update(_extract_bc_training_metadata(weight_path, {}))
        return model, make_config(), metadata

    if suffix not in {".pth", ".pt"}:
        raise ValueError(f"Unsupported BC checkpoint suffix: {weight_path.suffix}")

    payload = torch.load(weight_path, map_location="cpu")
    actor_state_dict = payload.get("actor_state_dict")
    if not isinstance(actor_state_dict, dict):
        raise KeyError(f"'actor_state_dict' not found in BC checkpoint: {weight_path}")

    config = make_config()
    config_payload = payload.get("config")
    if isinstance(config_payload, dict):
        if isinstance(config_payload.get("env"), dict):
            _apply_overrides(config.env, config_payload["env"])
        if isinstance(config_payload.get("train"), dict):
            _apply_overrides(config.train, config_payload["train"])
        if isinstance(config_payload.get("eval"), dict):
            _apply_overrides(config.eval, config_payload["eval"])
    config.train.num_envs = 1
    config.train.n_steps = 8
    config.train.batch_size = 8

    vec_env = build_vec_env(config, num_envs=1, seed=config.train.seed)
    model = build_ppo_model(config, vec_env, device=device, seed=config.train.seed, verbose=0)
    loaded_keys = load_actor_state_dict(model.policy, actor_state_dict)
    if not loaded_keys:
        raise RuntimeError(f"No actor parameters were loaded from BC checkpoint: {weight_path}")
    model.policy.set_training_mode(False)
    metadata = {
        "weight_name": weight_path.name,
        "weight_path": str(weight_path),
        "weight_suffix": suffix,
        "load_mode": "actor_state_dict",
        "loaded_tensor_count": int(len(loaded_keys)),
    }
    metadata.update(_extract_bc_training_metadata(weight_path, payload))
    return model, config, metadata


def evaluate_on_scene(
    model: PPO,
    config,
    scenario_path: Path,
    *,
    episodes_per_scene: int,
    base_seed: int,
) -> dict[str, Any]:
    env = make_single_env(config, scenario_path=scenario_path, render_mode=None)
    episodes: list[dict[str, Any]] = []
    scene_start_time = time.perf_counter()
    try:
        for episode_offset in range(int(episodes_per_scene)):
            episode_seed = int(base_seed) + int(episode_offset)
            observation, info = env.reset(seed=episode_seed)
            del info
            done = False
            episode_reward = 0.0
            episode_steps = 0
            final_info: dict[str, Any] = {}
            episode_start_time = time.perf_counter()

            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(action)
                episode_reward += float(reward)
                episode_steps += 1
                done = bool(terminated or truncated)

            episodes.append(
                {
                    "seed": int(episode_seed),
                    "success": bool(final_info.get("success", False)),
                    "termination_reason": str(final_info.get("termination_reason", "unknown")),
                    "episode_reward": float(episode_reward),
                    "episode_steps": int(episode_steps),
                    "episode_wall_time_sec": float(max(0.0, time.perf_counter() - episode_start_time)),
                    "collision": bool(final_info.get("collision", False)),
                    "wall_collision": bool(final_info.get("wall_collision", False)),
                    "timeout": bool(final_info.get("timeout", False)),
                }
            )
    finally:
        env.close()

    success_count = int(sum(1 for item in episodes if item["success"]))
    success_rate = float(success_count / len(episodes)) if episodes else 0.0
    avg_reward = float(np.mean([float(item["episode_reward"]) for item in episodes])) if episodes else 0.0
    avg_steps = float(np.mean([float(item["episode_steps"]) for item in episodes])) if episodes else 0.0
    termination_reason_counts: dict[str, int] = {}
    for item in episodes:
        reason = str(item["termination_reason"])
        termination_reason_counts[reason] = termination_reason_counts.get(reason, 0) + 1
    return {
        "scene_name": scenario_path.name,
        "scenario_path": str(scenario_path),
        "episodes": episodes,
        "episode_count": int(len(episodes)),
        "success_count": int(success_count),
        "success_rate": float(success_rate),
        "avg_reward": float(avg_reward),
        "avg_steps": float(avg_steps),
        "scene_eval_wall_time_sec": float(max(0.0, time.perf_counter() - scene_start_time)),
        "termination_reason_counts": termination_reason_counts,
    }


def evaluate_weight(
    weight_path: Path,
    *,
    scenario_paths: list[Path],
    episodes_per_scene: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    print(f"Loading BC checkpoint: {weight_path.name}", flush=True)
    weight_start_time = time.perf_counter()
    model, config, metadata = build_bc_eval_model(weight_path, device=device)
    scene_results: list[dict[str, Any]] = []
    try:
        running_success_count = 0
        running_episode_count = 0
        for scene_offset, scenario_path in enumerate(scenario_paths):
            scene_seed = int(seed) + scene_offset * 1000
            scene_result = evaluate_on_scene(
                model,
                config,
                scenario_path,
                episodes_per_scene=episodes_per_scene,
                base_seed=scene_seed,
            )
            scene_results.append(scene_result)
            running_success_count += int(scene_result["success_count"])
            running_episode_count += int(scene_result["episode_count"])
            if (scene_offset + 1) % 10 == 0 or scene_offset + 1 == len(scenario_paths):
                running_success_rate = (
                    float(running_success_count / running_episode_count)
                    if running_episode_count > 0
                    else 0.0
                )
                print(
                    f"  scenes {scene_offset + 1:03d}/{len(scenario_paths):03d} | "
                    f"success={running_success_count}/{running_episode_count} ({running_success_rate:.3f})",
                    flush=True,
                )
    finally:
        if model.get_env() is not None:
            model.get_env().close()

    all_episodes = [episode for scene in scene_results for episode in scene["episodes"]]
    overall_success_count = int(sum(1 for item in all_episodes if item["success"]))
    overall_success_rate = float(overall_success_count / len(all_episodes)) if all_episodes else 0.0
    overall_avg_reward = float(np.mean([float(item["episode_reward"]) for item in all_episodes])) if all_episodes else 0.0
    overall_avg_steps = float(np.mean([float(item["episode_steps"]) for item in all_episodes])) if all_episodes else 0.0
    overall_termination_reason_counts: dict[str, int] = {}
    for item in all_episodes:
        reason = str(item["termination_reason"])
        overall_termination_reason_counts[reason] = overall_termination_reason_counts.get(reason, 0) + 1

    print(
        f"Overall for {weight_path.name}: "
        f"success={overall_success_count}/{len(all_episodes)} ({overall_success_rate:.3f})",
        flush=True,
    )
    print("", flush=True)

    return {
        **metadata,
        "scene_count": int(len(scene_results)),
        "episodes_per_scene": int(episodes_per_scene),
        "total_episode_count": int(len(all_episodes)),
        "overall_success_count": int(overall_success_count),
        "overall_success_rate": float(overall_success_rate),
        "overall_avg_reward": float(overall_avg_reward),
        "overall_avg_steps": float(overall_avg_steps),
        "weight_eval_wall_time_sec": float(max(0.0, time.perf_counter() - weight_start_time)),
        "overall_termination_reason_counts": overall_termination_reason_counts,
        "scenes": scene_results,
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    weight_paths = resolve_weight_paths(args)
    for weight_path in weight_paths:
        if not weight_path.exists():
            raise FileNotFoundError(f"BC checkpoint not found: {weight_path}")

    scenario_dir = Path(args.scenario_dir).resolve()
    scenario_paths = sorted(scenario_dir.glob(str(args.scenario_glob)))
    if not scenario_paths:
        raise RuntimeError(f"No scenarios matched '{args.scenario_glob}' in: {scenario_dir}")

    start_index = max(1, int(args.scene_start))
    end_index = int(args.scene_end)
    selected_scene_indices, selected_scenarios = resolve_selected_scenarios(
        scenario_paths=scenario_paths,
        scene_start=int(start_index),
        scene_end=int(end_index),
        scene_indices=args.scene_indices,
        uniform_block_size=int(args.uniform_block_size),
        uniform_scenes_per_block=int(args.uniform_scenes_per_block),
    )
    if not selected_scenarios:
        raise RuntimeError(
            f"Selected scene range is empty: scene_start={start_index}, scene_end={end_index}, dir={scenario_dir}"
        )

    print(f"Using device: {device}", flush=True)
    print(
        f"Testing {len(selected_scenarios)} scenes x {int(args.episodes_per_scene)} episodes",
        flush=True,
    )
    print(f"Selected scene indices: {selected_scene_indices}", flush=True)
    print("", flush=True)

    results = {
        "device": device,
        "weights_dir": None if args.weights_dir is None else str(Path(args.weights_dir).resolve()),
        "weights_glob": str(args.weights_glob),
        "scenario_dir": str(scenario_dir),
        "scenario_glob": str(args.scenario_glob),
        "scene_start": int(start_index),
        "scene_end": int(end_index),
        "selected_scene_indices": [int(index) for index in selected_scene_indices],
        "uniform_block_size": int(args.uniform_block_size),
        "uniform_scenes_per_block": int(args.uniform_scenes_per_block),
        "episodes_per_scene": int(args.episodes_per_scene),
        "weights": [],
    }
    for weight_path in weight_paths:
        weight_result = evaluate_weight(
            weight_path,
            scenario_paths=selected_scenarios,
            episodes_per_scene=int(args.episodes_per_scene),
            seed=int(args.seed),
            device=device,
        )
        results["weights"].append(weight_result)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()
        write_result_bundle(output_dir, results)
        print(f"Saved evaluation bundle to {output_dir}", flush=True)
    else:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved evaluation JSON to {output_path}", flush=True)


if __name__ == "__main__":
    main()
