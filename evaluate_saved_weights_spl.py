from __future__ import annotations

import argparse
import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file

from configs.default_config import PROJECT_ROOT, make_config
from evaluate_bc_rl import select_device
from utils.lora_policy import DEFAULT_LORA_TARGET_MODULES, LoraMultiInputPolicy
from utils.policy_utils import build_vec_env, load_matching_policy_state_dict, make_single_env


DEFAULT_WEIGHT_PATHS = (
    PROJECT_ROOT / "runs" / "ppo_fish_baseline" / "scenario_cycle" / "checkpoints" / "ppo_fish_baseline_update_000025_policy.pth",
    PROJECT_ROOT
    / "runs"
    / "ppo_fish_baseline"
    / "scenario_cycle_rand20_from_update25_20260415"
    / "scenario_cycle"
    / "checkpoints"
    / "ppo_fish_rand20_from_u025_update_000037.zip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved PPO policy checkpoints on test scenes and report success rate and SPL."
    )
    parser.add_argument(
        "--weights",
        type=str,
        nargs="+",
        default=[str(path.resolve()) for path in DEFAULT_WEIGHT_PATHS],
        help="One or more PPO checkpoints (.pth/.pt policy snapshot or .zip SB3 model).",
    )
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=str((PROJECT_ROOT / "scenarios" / "large_pool_dataset_200" / "test" / "json").resolve()),
        help="Directory containing test_env_XXX.json files.",
    )
    parser.add_argument("--scene-start", type=int, default=1, help="1-based inclusive test scene start index.")
    parser.add_argument("--scene-end", type=int, default=20, help="1-based inclusive test scene end index.")
    parser.add_argument("--episodes-per-scene", type=int, default=5, help="Evaluation episodes per scene.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device: cuda, cuda:0, cpu, or auto.")
    parser.add_argument(
        "--output",
        type=str,
        default=str((PROJECT_ROOT / "runs" / "eval" / "saved_weights_test001_020_ep5_spl.json").resolve()),
        help="JSON output path.",
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


def _find_nearest_config_json(weight_path: Path) -> Path | None:
    for parent in [weight_path.parent, *weight_path.parents]:
        candidate = parent / "config.json"
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_run_config(weight_path: Path):
    config = make_config()
    config_path = _find_nearest_config_json(weight_path)
    run_payload: dict[str, Any] = {}
    if config_path is not None:
        run_payload = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(run_payload.get("env"), dict):
            _apply_overrides(config.env, run_payload["env"])
        if isinstance(run_payload.get("train"), dict):
            _apply_overrides(config.train, run_payload["train"])
        if isinstance(run_payload.get("eval"), dict):
            _apply_overrides(config.eval, run_payload["eval"])

    # Keep evaluation lightweight. These values do not change the inference graph.
    config.train.num_envs = 1
    config.train.n_steps = 8
    config.train.batch_size = 8
    return config, config_path, run_payload


def _resolve_policy_spec(run_payload: dict[str, Any]):
    lora_payload = run_payload.get("lora")
    if not isinstance(lora_payload, dict) or not bool(lora_payload.get("enabled", False)):
        return "MultiInputPolicy", {"net_arch": list(make_config().train.policy_hidden_sizes)}, False

    policy_kwargs: dict[str, Any] = {
        "net_arch": list(make_config().train.policy_hidden_sizes),
        "lora_rank": int(lora_payload.get("rank", 4)),
        "lora_alpha": float(lora_payload.get("alpha", 8.0)),
        "lora_dropout": float(lora_payload.get("dropout", 0.0)),
        "lora_target_modules": tuple(
            str(name) for name in lora_payload.get("target_modules", list(DEFAULT_LORA_TARGET_MODULES))
        ),
        "lora_freeze_actor_base": bool(lora_payload.get("freeze_actor_base", True)),
        "lora_train_bias": bool(lora_payload.get("train_bias", False)),
    }
    return LoraMultiInputPolicy, policy_kwargs, True


def _load_policy_state_dict(weight_path: Path) -> dict[str, torch.Tensor]:
    suffix = weight_path.suffix.lower()
    if suffix in {".pth", ".pt"}:
        payload = torch.load(weight_path, map_location="cpu")
        policy_state_dict = payload.get("policy_state_dict")
        if not isinstance(policy_state_dict, dict):
            raise KeyError(f"'policy_state_dict' not found in checkpoint: {weight_path}")
        return policy_state_dict

    if suffix == ".zip":
        _, params, _ = load_from_zip_file(weight_path, device="cpu", custom_objects=None, print_system_info=False)
        policy_state_dict = params.get("policy")
        if not isinstance(policy_state_dict, dict):
            raise KeyError(f"'policy' parameters not found in zip checkpoint: {weight_path}")
        return policy_state_dict

    raise ValueError(f"Unsupported checkpoint suffix: {weight_path.suffix}")


def build_eval_model(weight_path: Path, device: str):
    config, config_path, run_payload = _load_run_config(weight_path)
    policy_spec, policy_kwargs, lora_enabled = _resolve_policy_spec(run_payload)
    if isinstance(run_payload.get("train"), dict) and "policy_hidden_sizes" in run_payload["train"]:
        policy_kwargs["net_arch"] = list(run_payload["train"]["policy_hidden_sizes"])

    vec_env = build_vec_env(config, num_envs=1, seed=config.train.seed)
    model = PPO(
        policy=policy_spec,
        env=vec_env,
        learning_rate=config.train.learning_rate,
        n_steps=config.train.n_steps,
        batch_size=config.train.batch_size,
        gamma=config.train.gamma,
        gae_lambda=config.train.gae_lambda,
        clip_range=config.train.clip_range,
        ent_coef=config.train.ent_coef,
        vf_coef=config.train.vf_coef,
        max_grad_norm=config.train.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=config.train.seed,
        device=device,
    )
    policy_state_dict = _load_policy_state_dict(weight_path)
    loaded_keys, skipped_keys = load_matching_policy_state_dict(model.policy, policy_state_dict)
    if not loaded_keys:
        raise RuntimeError(f"No policy tensors were loaded from checkpoint: {weight_path}")
    model.policy.set_training_mode(False)
    metadata = {
        "config_path": None if config_path is None else str(config_path),
        "run_id": run_payload.get("run_id"),
        "lora_enabled": bool(lora_enabled),
        "loaded_tensor_count": int(len(loaded_keys)),
        "skipped_tensor_count": int(len(set(skipped_keys))),
    }
    return model, config, metadata


def _head_center_xy(env) -> np.ndarray:
    return np.asarray(env.data.geom_xpos[env.head_collision_geom_id][:2], dtype=np.float64).copy()


def _head_planar_radius(env) -> float:
    geom_type = int(env.model.geom_type[env.head_collision_geom_id])
    geom_size = np.asarray(env.model.geom_size[env.head_collision_geom_id], dtype=np.float64)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.hypot(geom_size[0], geom_size[1]))
    return float(max(geom_size[0], geom_size[1]))


def _shortest_touch_distance(env, start_head_xy: np.ndarray) -> float:
    goal_min = np.asarray(env.goal_center - env.goal_half_extents, dtype=np.float64)
    goal_max = np.asarray(env.goal_center + env.goal_half_extents, dtype=np.float64)
    nearest_point = np.clip(start_head_xy, goal_min, goal_max)
    distance_to_goal_box = float(np.linalg.norm(start_head_xy - nearest_point))
    return max(0.0, distance_to_goal_box - _head_planar_radius(env))


def _episode_spl(success: bool, shortest_distance: float, path_length: float) -> float:
    if not success:
        return 0.0
    if shortest_distance <= 1e-8:
        return 1.0
    denominator = max(path_length, shortest_distance, 1e-8)
    return float(shortest_distance / denominator)


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
    try:
        for episode_offset in range(int(episodes_per_scene)):
            episode_seed = int(base_seed) + int(episode_offset)
            observation, info = env.reset(seed=episode_seed)
            del info
            done = False
            episode_reward = 0.0
            episode_steps = 0
            final_info: dict[str, Any] = {}

            previous_head_xy = _head_center_xy(env)
            start_head_xy = previous_head_xy.copy()
            shortest_distance = _shortest_touch_distance(env, start_head_xy)
            path_length = 0.0

            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(action)
                current_head_xy = _head_center_xy(env)
                path_length += float(np.linalg.norm(current_head_xy - previous_head_xy))
                previous_head_xy = current_head_xy
                episode_reward += float(reward)
                episode_steps += 1
                done = bool(terminated or truncated)

            success = bool(final_info.get("success", False))
            spl = _episode_spl(success, shortest_distance, path_length)
            episodes.append(
                {
                    "seed": int(episode_seed),
                    "success": bool(success),
                    "termination_reason": str(final_info.get("termination_reason", "unknown")),
                    "episode_reward": float(episode_reward),
                    "episode_steps": int(episode_steps),
                    "distance_to_goal_region": float(final_info.get("distance_to_goal_region", 0.0)),
                    "goal_progress_ratio_final": float(final_info.get("goal_progress_ratio", 0.0)),
                    "path_length_head_xy": float(path_length),
                    "shortest_touch_distance_head_xy": float(shortest_distance),
                    "spl": float(spl),
                    "collision": bool(final_info.get("collision", False)),
                    "wall_collision": bool(final_info.get("wall_collision", False)),
                    "timeout": bool(final_info.get("timeout", False)),
                }
            )
    finally:
        env.close()

    success_rate = float(np.mean([float(item["success"]) for item in episodes])) if episodes else 0.0
    avg_spl = float(np.mean([float(item["spl"]) for item in episodes])) if episodes else 0.0
    avg_reward = float(np.mean([float(item["episode_reward"]) for item in episodes])) if episodes else 0.0
    avg_steps = float(np.mean([float(item["episode_steps"]) for item in episodes])) if episodes else 0.0
    return {
        "scene_name": scenario_path.name,
        "scenario_path": str(scenario_path),
        "episodes": episodes,
        "episode_count": int(len(episodes)),
        "success_count": int(sum(1 for item in episodes if item["success"])),
        "success_rate": float(success_rate),
        "avg_spl": float(avg_spl),
        "avg_reward": float(avg_reward),
        "avg_steps": float(avg_steps),
    }


def evaluate_weight(
    weight_path: Path,
    *,
    scenario_paths: list[Path],
    episodes_per_scene: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    print(f"Loading checkpoint: {weight_path.name}", flush=True)
    model, config, model_metadata = build_eval_model(weight_path, device=device)
    scene_results: list[dict[str, Any]] = []
    try:
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
            print(
                f"  {scene_result['scene_name']}: "
                f"success={scene_result['success_count']}/{scene_result['episode_count']} "
                f"({scene_result['success_rate']:.3f}) | "
                f"SPL={scene_result['avg_spl']:.3f}",
                flush=True,
            )
    finally:
        if model.get_env() is not None:
            model.get_env().close()

    all_episodes = [episode for scene in scene_results for episode in scene["episodes"]]
    overall_success_rate = float(np.mean([float(item["success"]) for item in all_episodes])) if all_episodes else 0.0
    overall_spl = float(np.mean([float(item["spl"]) for item in all_episodes])) if all_episodes else 0.0
    overall_avg_reward = float(np.mean([float(item["episode_reward"]) for item in all_episodes])) if all_episodes else 0.0
    overall_avg_steps = float(np.mean([float(item["episode_steps"]) for item in all_episodes])) if all_episodes else 0.0

    print(
        f"Overall for {weight_path.name}: "
        f"success={sum(1 for item in all_episodes if item['success'])}/{len(all_episodes)} "
        f"({overall_success_rate:.3f}) | SPL={overall_spl:.3f}",
        flush=True,
    )
    print("", flush=True)

    return {
        "weight_name": weight_path.name,
        "weight_path": str(weight_path),
        "weight_suffix": weight_path.suffix.lower(),
        **model_metadata,
        "scene_count": int(len(scene_results)),
        "episodes_per_scene": int(episodes_per_scene),
        "total_episode_count": int(len(all_episodes)),
        "overall_success_count": int(sum(1 for item in all_episodes if item["success"])),
        "overall_success_rate": float(overall_success_rate),
        "overall_spl": float(overall_spl),
        "overall_avg_reward": float(overall_avg_reward),
        "overall_avg_steps": float(overall_avg_steps),
        "scenes": scene_results,
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    weight_paths = [Path(path).resolve() for path in args.weights]
    for weight_path in weight_paths:
        if not weight_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    scenario_dir = Path(args.scenario_dir).resolve()
    scenario_paths = sorted(scenario_dir.glob("test_env_*.json"))
    if not scenario_paths:
        raise RuntimeError(f"No test scenes found in: {scenario_dir}")

    start_index = max(1, int(args.scene_start))
    end_index = int(args.scene_end)
    selected_scenarios = scenario_paths[start_index - 1 : end_index]
    if not selected_scenarios:
        raise RuntimeError(
            f"Selected test scene range is empty: scene_start={start_index}, scene_end={end_index}, dir={scenario_dir}"
        )

    print(f"Using device: {device}", flush=True)
    print(
        f"Testing scenes: {selected_scenarios[0].name} .. {selected_scenarios[-1].name} "
        f"({len(selected_scenarios)} scenes x {int(args.episodes_per_scene)} episodes)",
        flush=True,
    )
    print("", flush=True)

    results = {
        "device": device,
        "scenario_dir": str(scenario_dir),
        "scene_start": int(start_index),
        "scene_end": int(end_index),
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

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved evaluation JSON to {output_path}", flush=True)


if __name__ == "__main__":
    main()
