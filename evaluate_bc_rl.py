from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from configs.default_config import PROJECT_ROOT, make_config
from utils.policy_utils import build_ppo_model, build_vec_env, load_actor_state_dict, make_single_env, resolve_scenario_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BC and PPO policies on the fish environment.")
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes per policy.")
    parser.add_argument("--seed", type=int, default=7, help="Base episode seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device: cuda, cuda:0, cpu, or auto.")
    parser.add_argument("--xml-path", type=str, default=None, help="Override MuJoCo XML scene path.")
    parser.add_argument("--scenario-path", type=str, default=None, help="Evaluate on one fixed exported environment JSON.")
    parser.add_argument("--scenario-index", type=int, default=None, help="Evaluate on training_env_XX.json.")
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=str((PROJECT_ROOT / "scenarios" / "training_envs").resolve()),
        help="Directory containing exported fixed environment JSON files.",
    )
    parser.add_argument("--render", action="store_true", help="Render top-down trajectories during evaluation.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--bc-weights", type=str, default=None, help="BC actor .pth checkpoint.")
    parser.add_argument("--bc-dataset", type=str, default=None, help="Optional dataset for BC offline MSE evaluation.")
    parser.add_argument("--ppo-scratch", type=str, default=None, help="PPO .zip model trained from scratch.")
    parser.add_argument("--ppo-bc", type=str, default=None, help="PPO .zip model fine-tuned from BC initialization.")
    return parser.parse_args()


def select_device(requested_device: str) -> str:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA device, but CUDA is unavailable. Falling back to CPU.")
        return "cpu"
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def resolve_model_path(path_str: str | None, expected_suffix: str | tuple[str, ...]) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")
    valid_suffixes = (expected_suffix,) if isinstance(expected_suffix, str) else expected_suffix
    if path.suffix.lower() not in valid_suffixes:
        raise ValueError(f"Expected one of {valid_suffixes} for: {path}")
    return path


def evaluate_policy_model(model: PPO, env, episodes: int, seed: int, render: bool) -> dict[str, float]:
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    success_flags: list[bool] = []
    collision_flags: list[bool] = []
    timeout_flags: list[bool] = []
    goal_progress_ratios: list[float] = []
    cross_track_errors: list[float] = []

    for episode_index in range(int(episodes)):
        observation, info = env.reset(seed=int(seed) + episode_index)
        done = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            episode_steps += 1
            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        success_flags.append(bool(info.get("success", False)))
        collision_flags.append(bool(info.get("collision", False)) or bool(info.get("wall_collision", False)))
        timeout_flags.append(bool(info.get("timeout", False)))
        goal_progress_ratios.append(float(info.get("goal_progress_ratio", 0.0)))
        cross_track_errors.append(abs(float(info.get("cross_track_error", 0.0))))

    return {
        "episodes": int(episodes),
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean(success_flags)),
        "collision_rate": float(np.mean(collision_flags)),
        "timeout_rate": float(np.mean(timeout_flags)),
        "avg_goal_progress_ratio": float(np.mean(goal_progress_ratios)),
        "avg_abs_cross_track_error": float(np.mean(cross_track_errors)),
    }


def evaluate_bc_dataset_fit(model: PPO, dataset_path: Path, device: str) -> dict[str, float]:
    payload = np.load(dataset_path, allow_pickle=False)
    obs_image = payload["obs_image"].astype(np.uint8, copy=False)
    obs_imu = payload["obs_imu"].astype(np.float32, copy=False)
    expert_action = payload["action"].astype(np.float32, copy=False)

    batch_size = 128
    mse_terms: list[float] = []
    mae_terms: list[float] = []
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, obs_image.shape[0], batch_size):
            stop = min(start + batch_size, obs_image.shape[0])
            obs = {
                "image": torch.from_numpy(obs_image[start:stop]).permute(0, 3, 1, 2).contiguous().to(
                    device=device,
                    dtype=torch.uint8,
                ),
                "imu": torch.from_numpy(obs_imu[start:stop]).to(device=device, dtype=torch.float32),
            }
            target_action = torch.from_numpy(expert_action[start:stop]).to(device=device, dtype=torch.float32)
            predicted_action = torch.clamp(model.policy.get_distribution(obs).distribution.mean, -1.0, 1.0)
            mse_terms.append(float(torch.mean((predicted_action - target_action) ** 2).cpu()))
            mae_terms.append(float(torch.mean(torch.abs(predicted_action - target_action)).cpu()))

    return {
        "dataset_size": int(obs_image.shape[0]),
        "mse": float(np.mean(mse_terms)),
        "mae": float(np.mean(mae_terms)),
    }


def make_bc_model(config, bc_weights_path: Path, device: str) -> PPO:
    vec_env = build_vec_env(config, num_envs=1, seed=config.train.seed)
    try:
        model = build_ppo_model(config, vec_env, device=device, seed=config.train.seed, verbose=0)
        payload = torch.load(bc_weights_path, map_location="cpu")
        actor_state_dict = payload.get("actor_state_dict")
        if not isinstance(actor_state_dict, dict):
            raise KeyError(f"BC checkpoint does not contain 'actor_state_dict': {bc_weights_path}")
        loaded_keys = load_actor_state_dict(model.policy, actor_state_dict)
        if not loaded_keys:
            raise RuntimeError(f"No actor parameters were loaded from BC checkpoint: {bc_weights_path}")
        return model
    finally:
        vec_env.close()


def main() -> None:
    args = parse_args()
    config = make_config()
    if args.xml_path is not None:
        config.env.model.xml_path = str(Path(args.xml_path).resolve())
    device = select_device(args.device)
    scenario_path = resolve_scenario_path(args.scenario_path, args.scenario_index, args.scenario_dir)

    bc_weights_path = resolve_model_path(args.bc_weights, (".pth", ".pt"))
    ppo_scratch_path = resolve_model_path(args.ppo_scratch, ".zip")
    ppo_bc_path = resolve_model_path(args.ppo_bc, ".zip")
    dataset_path = None if args.bc_dataset is None else Path(args.bc_dataset).resolve()
    if dataset_path is not None and not dataset_path.exists():
        raise FileNotFoundError(f"BC dataset not found: {dataset_path}")

    if bc_weights_path is None and ppo_scratch_path is None and ppo_bc_path is None:
        raise ValueError("Provide at least one of --bc-weights, --ppo-scratch, or --ppo-bc.")

    results: dict[str, object] = {
        "device": device,
        "scenario_path": None if scenario_path is None else str(scenario_path),
        "episodes": int(args.episodes),
        "policies": {},
    }

    policies: list[tuple[str, PPO]] = []
    if bc_weights_path is not None:
        policies.append(("bc_actor", make_bc_model(config, bc_weights_path, device)))
    if ppo_scratch_path is not None:
        policies.append(("ppo_scratch", PPO.load(str(ppo_scratch_path), device=device)))
    if ppo_bc_path is not None:
        policies.append(("ppo_bc_finetuned", PPO.load(str(ppo_bc_path), device=device)))

    try:
        for label, model in policies:
            env = make_single_env(
                config,
                scenario_path=scenario_path,
                render_mode="human" if args.render else None,
            )
            try:
                summary = evaluate_policy_model(model, env, episodes=args.episodes, seed=args.seed, render=args.render)
            finally:
                env.close()
            results["policies"][label] = summary
            print(f"{label}: {json.dumps(summary, ensure_ascii=False)}")

        if bc_weights_path is not None and dataset_path is not None:
            bc_model = next(model for label, model in policies if label == "bc_actor")
            offline_metrics = evaluate_bc_dataset_fit(bc_model, dataset_path, device=device)
            results["bc_dataset_fit"] = offline_metrics
            print(f"bc_dataset_fit: {json.dumps(offline_metrics, ensure_ascii=False)}")
    finally:
        for _, model in policies:
            if model.get_env() is not None:
                model.get_env().close()

    if args.output is not None:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, ensure_ascii=False)
        print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
