from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

from algorithms import EpisodeCyclePPO
from configs.default_config import PROJECT_ROOT, config_to_dict, make_config
from envs import FishPathAvoidEnv
from utils.lora_policy import DEFAULT_LORA_TARGET_MODULES, LoraMultiInputPolicy
from utils.lora_sac_policy import DEFAULT_SAC_LORA_TARGET_MODULES, LoraSACPolicy
from utils.policy_utils import (
    load_actor_state_dict,
    load_bc_actor_state_dict_into_sac_policy,
    load_matching_policy_state_dict,
)


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
    "distance_to_goal_region",
    "visual_obstacle_detected",
    "visual_obstacle_pixel_fraction",
    "visual_obstacle_center_fraction",
    "visual_obstacle_nearest_depth",
    "success",
    "collision",
    "wall_collision",
    "out_of_bounds",
    "timeout",
]


CHECKPOINT_METRICS_FIELDNAMES = [
    "run_id",
    "update_index",
    "num_timesteps",
    "episodes_in_cycle",
    "episodes_completed_total",
    "success_count",
    "success_rate",
    "mean_episode_reward",
    "mean_episode_length",
    "cycle_train_time_sec",
    "model_path",
    "policy_weights_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO or SAC for fish path following with local obstacle avoidance.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total training timesteps.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel environments.")
    parser.add_argument(
        "--algo",
        type=str,
        choices=("ppo", "sac"),
        default=None,
        help="RL algorithm. Defaults to the value in configs/default_config.py.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a previously saved .zip RL model.",
    )
    parser.add_argument(
        "--resume-policy-weights",
        type=str,
        default=None,
        help="Resume training from a saved policy weights snapshot (.pth/.pt).",
    )
    parser.add_argument("--xml-path", type=str, default=None, help="Override MuJoCo XML scene path.")
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show a realtime MuJoCo viewer for one training env.",
    )
    parser.add_argument(
        "--render-env-index",
        type=int,
        default=0,
        help="Index of the vectorized env shown in the MuJoCo viewer when --render is enabled.",
    )
    parser.add_argument(
        "--render-slowdown",
        type=float,
        default=1.0,
        help="Viewer slowdown factor. 0 disables timing slowdown, 1 is realtime, 2 is 2x slower, etc.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device passed to Stable-Baselines3, for example cuda, cuda:0, cpu, or auto.",
    )
    parser.add_argument(
        "--plot-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show a live matplotlib plot of episode rewards during training.",
    )
    parser.add_argument(
        "--reward-plot-window",
        type=int,
        default=20,
        help="Moving-average window size used by the live reward plot.",
    )
    parser.add_argument(
        "--record-videos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable episode video recording.",
    )
    parser.add_argument(
        "--video-interval-episodes",
        type=int,
        default=None,
        help="Override the episode interval for saved videos when recording is enabled.",
    )
    parser.add_argument(
        "--scenario-path",
        type=str,
        default=None,
        help="Train on one fixed exported environment JSON file.",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=None,
        help="Train on training_env_XX.json from the scenario directory.",
    )
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=(Path("scenarios") / "training_envs").as_posix(),
        help="Directory containing exported fixed environment JSON files.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=500,
        help="Stop training after this many completed episodes. Use 0 or a negative value to disable the limit.",
    )
    parser.add_argument(
        "--bc-weights",
        type=str,
        default=None,
        help="Initialize the actor from a BC actor checkpoint saved by train_bc.py.",
    )
    parser.add_argument(
        "--scenario-cycle-dir",
        type=str,
        default=None,
        help="Directory of fixed scenario JSON files used in cyclic multi-scene training.",
    )
    parser.add_argument(
        "--scenario-cycle-glob",
        type=str,
        default="*.json",
        help="Glob pattern used inside --scenario-cycle-dir.",
    )
    parser.add_argument(
        "--scenario-cycle-list",
        type=str,
        default=None,
        help="Optional JSON file defining an explicit per-episode scenario cycle order.",
    )
    parser.add_argument(
        "--scenario-cycle-selection",
        type=str,
        default=None,
        help="Optional scene ids/ranges like '1-20,60-80'. Requires --scenario-cycle-dir and selects train_env_###.json files.",
    )
    parser.add_argument(
        "--scenario-cycle-start-index",
        type=int,
        default=None,
        help="Optional 1-based start offset applied to the selected scenario cycle.",
    )
    parser.add_argument(
        "--scenario-cycle-end-index",
        type=int,
        default=None,
        help="Optional 1-based end offset applied to the selected scenario cycle.",
    )
    parser.add_argument(
        "--scenario-cycle-sample-size",
        type=int,
        default=0,
        help="When positive, draw this many unique scenarios at random for each scenario-cycle batch.",
    )
    parser.add_argument(
        "--rollout-episodes-per-update",
        type=int,
        default=0,
        help="Scenario-cycle cadence. PPO collects this many episodes before each update; SAC saves one cycle checkpoint every this many completed episodes.",
    )
    parser.add_argument(
        "--rollout-step-budget",
        type=int,
        default=0,
        help="Maximum rollout buffer steps reserved for one PPO cycle update in episode-cycle mode. Defaults to max_episode_steps * rollout_episodes_per_update.",
    )
    parser.add_argument(
        "--strict-rollout-step-budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop with an error if episode-cycle rollout hits the step budget before collecting the requested number of episodes.",
    )
    parser.add_argument(
        "--align-rollout-updates-to-episode-count",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Align the first post-resume cycle checkpoint to the next multiple of --rollout-episodes-per-update based on the existing episode_metrics.csv count.",
    )
    parser.add_argument(
        "--sac-buffer-size",
        type=int,
        default=None,
        help="Override SAC replay buffer size.",
    )
    parser.add_argument(
        "--sac-learning-starts",
        type=int,
        default=None,
        help="Override SAC warmup steps before gradient updates start.",
    )
    parser.add_argument(
        "--sac-train-freq-steps",
        type=int,
        default=None,
        help="Override SAC training trigger count. The unit is controlled by --sac-train-freq-unit.",
    )
    parser.add_argument(
        "--sac-train-freq-unit",
        type=str,
        choices=("step", "episode"),
        default=None,
        help="Unit for SAC training triggers: environment steps or completed episodes.",
    )
    parser.add_argument(
        "--sac-gradient-steps",
        type=int,
        default=None,
        help="Override SAC gradient steps per training trigger.",
    )
    parser.add_argument(
        "--sac-tau",
        type=float,
        default=None,
        help="Override SAC target network smoothing coefficient.",
    )
    parser.add_argument(
        "--sac-ent-coef",
        type=str,
        default=None,
        help="Override SAC entropy coefficient, for example auto or 0.05.",
    )
    parser.add_argument(
        "--sac-target-update-interval",
        type=int,
        default=None,
        help="Override SAC target network update interval.",
    )
    parser.add_argument(
        "--sac-target-entropy",
        type=str,
        default=None,
        help="Override SAC target entropy, for example auto or -1.",
    )
    parser.add_argument(
        "--sac-optimize-memory-usage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Stable-Baselines3 replay buffer memory optimization for SAC.",
    )
    parser.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable LoRA-based actor fine-tuning instead of full actor finetuning.",
    )
    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank used when --use-lora is enabled.")
    parser.add_argument("--lora-alpha", type=float, default=8.0, help="LoRA alpha used when --use-lora is enabled.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout used when --use-lora is enabled.")
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        help="Exact linear module names patched with LoRA. Defaults depend on --algo.",
    )
    parser.add_argument(
        "--lora-freeze-actor-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze original actor weights and train only LoRA adapters, actor log_std, and critic parameters.",
    )
    parser.add_argument(
        "--lora-train-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow actor bias terms to remain trainable in LoRA mode.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier used for per-run monitor/config/summary files. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional override for the training log directory.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional override for the saved checkpoint stem.",
    )
    parser.add_argument(
        "--convergence-window",
        type=int,
        default=0,
        help="Enable convergence stopping with a sliding window of this many completed episodes. Set 0 to disable.",
    )
    parser.add_argument(
        "--convergence-min-episodes",
        type=int,
        default=0,
        help="Minimum total completed episodes before convergence can trigger. Defaults to --convergence-window.",
    )
    parser.add_argument(
        "--convergence-min-success-rate",
        type=float,
        default=0.90,
        help="Minimum success rate required over the convergence window.",
    )
    parser.add_argument(
        "--convergence-max-timeout-rate",
        type=float,
        default=0.10,
        help="Maximum timeout rate allowed over the convergence window.",
    )
    parser.add_argument(
        "--convergence-max-failure-rate",
        type=float,
        default=0.10,
        help="Maximum failure rate allowed over the convergence window. Failure means obstacle collision, wall collision, or out-of-bounds.",
    )
    parser.add_argument(
        "--convergence-reward-window",
        type=int,
        default=20,
        help="Window length used for reward stability comparison. Set 0 to disable reward stability checking.",
    )
    parser.add_argument(
        "--convergence-reward-stability-ratio",
        type=float,
        default=0.05,
        help="Maximum relative change allowed between the previous and latest reward windows. Set a negative value to disable.",
    )
    return parser.parse_args()


def resolve_scenario_path(args: argparse.Namespace) -> Path | None:
    if args.scenario_path is not None and args.scenario_index is not None:
        raise ValueError("Use either --scenario-path or --scenario-index, not both.")

    if args.scenario_path is not None:
        scenario_path = Path(args.scenario_path).resolve()
    elif args.scenario_index is not None:
        if args.scenario_index <= 0:
            raise ValueError("--scenario-index must be a positive integer.")
        scenario_dir = Path(args.scenario_dir).resolve()
        scenario_path = scenario_dir / f"training_env_{args.scenario_index:02d}.json"
    else:
        return None

    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario JSON not found: {scenario_path}")
    return scenario_path


def resolve_scenario_cycle_paths(args: argparse.Namespace) -> list[Path]:
    if (
        args.scenario_cycle_dir is None
        and args.scenario_cycle_list is None
        and args.scenario_cycle_selection is None
    ):
        return []

    if args.scenario_cycle_list is not None and args.scenario_cycle_selection is not None:
        raise ValueError("Use either --scenario-cycle-list or --scenario-cycle-selection, not both.")

    if args.scenario_cycle_selection is not None and args.scenario_cycle_dir is None:
        raise ValueError("--scenario-cycle-selection requires --scenario-cycle-dir.")

    if args.scenario_cycle_dir is not None and args.scenario_cycle_list is not None:
        raise ValueError("Use either --scenario-cycle-dir or --scenario-cycle-list, not both.")

    if args.scenario_cycle_selection is not None:
        selected = resolve_selected_scenario_cycle_paths(
            selection_text=args.scenario_cycle_selection,
            scenario_dir_arg=args.scenario_cycle_dir,
        )
        if args.scenario_cycle_start_index is not None or args.scenario_cycle_end_index is not None:
            start_index = 1 if args.scenario_cycle_start_index is None else int(args.scenario_cycle_start_index)
            end_index = len(selected) if args.scenario_cycle_end_index is None else int(args.scenario_cycle_end_index)
            if start_index <= 0 or end_index <= 0:
                raise ValueError("--scenario-cycle-start-index and --scenario-cycle-end-index must be positive.")
            if start_index > end_index:
                raise ValueError("--scenario-cycle-start-index cannot be greater than --scenario-cycle-end-index.")
            selected = selected[start_index - 1 : end_index]
            if not selected:
                raise RuntimeError("The requested scenario cycle selection is empty.")
        return selected

    if args.scenario_cycle_list is not None:
        scenario_list_path = Path(args.scenario_cycle_list).resolve()
        if not scenario_list_path.exists():
            raise FileNotFoundError(f"Scenario cycle list not found: {scenario_list_path}")
        payload = json.loads(scenario_list_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.get("scenarios")
            if items is None:
                raise KeyError(f"Scenario cycle JSON must contain a 'scenarios' key: {scenario_list_path}")
        elif isinstance(payload, list):
            items = payload
        else:
            raise TypeError(f"Unsupported scenario cycle JSON format: {scenario_list_path}")

        base_dir = scenario_list_path.parent
        matched: list[Path] = []
        for item in items:
            if isinstance(item, str):
                raw_path = item
            elif isinstance(item, dict):
                raw_path = item.get("scenario_path") or item.get("path")
                if raw_path is None:
                    raise KeyError(f"Scenario cycle entry is missing scenario_path/path: {item}")
            else:
                raise TypeError(f"Unsupported scenario cycle entry: {item!r}")
            resolved = Path(raw_path)
            if not resolved.is_absolute():
                resolved = (base_dir / resolved).resolve()
            else:
                resolved = resolved.resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Scenario JSON from cycle list does not exist: {resolved}")
            matched.append(resolved)
    else:
        scenario_dir = Path(args.scenario_cycle_dir).resolve()
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Scenario cycle directory not found: {scenario_dir}")
        matched = sorted(path.resolve() for path in scenario_dir.glob(args.scenario_cycle_glob) if path.is_file())
        if not matched:
            raise FileNotFoundError(
                f"No scenarios matched {args.scenario_cycle_glob!r} in {scenario_dir}"
            )

    start_index = 1 if args.scenario_cycle_start_index is None else int(args.scenario_cycle_start_index)
    end_index = len(matched) if args.scenario_cycle_end_index is None else int(args.scenario_cycle_end_index)
    if start_index <= 0 or end_index <= 0:
        raise ValueError("--scenario-cycle-start-index and --scenario-cycle-end-index must be positive.")
    if start_index > end_index:
        raise ValueError("--scenario-cycle-start-index cannot be greater than --scenario-cycle-end-index.")
    selected = matched[start_index - 1 : end_index]
    if not selected:
        raise RuntimeError("The requested scenario cycle selection is empty.")
    return selected


def parse_scenario_cycle_selection(selection_text: str) -> list[int]:
    tokens = [token for token in re.split(r"[\s,]+", str(selection_text).strip()) if token]
    if not tokens:
        raise ValueError("--scenario-cycle-selection cannot be empty.")

    selected_ids: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
        else:
            start = int(token)
            end = start
        if start <= 0 or end <= 0:
            raise ValueError(f"Scenario ids must be positive integers, got: {token!r}")
        if start > end:
            raise ValueError(f"Scenario range start must be <= end, got: {token!r}")
        for scene_index in range(start, end + 1):
            if scene_index not in seen:
                seen.add(scene_index)
                selected_ids.append(scene_index)
    return selected_ids


def resolve_scenario_cycle_json_dir(scenario_dir_arg: str) -> Path:
    scenario_dir = Path(scenario_dir_arg).expanduser().resolve()
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario cycle directory not found: {scenario_dir}")
    if not scenario_dir.is_dir():
        raise NotADirectoryError(f"Scenario cycle directory is not a directory: {scenario_dir}")

    json_dir = scenario_dir / "json" if (scenario_dir / "json").is_dir() else scenario_dir
    if not any(json_dir.glob("train_env_*.json")):
        raise FileNotFoundError(
            "Scenario cycle directory must either contain a 'json' subdirectory or direct "
            f"'train_env_*.json' files: {scenario_dir}"
        )
    return json_dir


def resolve_selected_scenario_cycle_paths(*, selection_text: str, scenario_dir_arg: str) -> list[Path]:
    selected_indexes = parse_scenario_cycle_selection(selection_text)
    scenario_json_dir = resolve_scenario_cycle_json_dir(scenario_dir_arg)

    missing_scenarios: list[str] = []
    scenario_paths: list[Path] = []
    for scene_index in selected_indexes:
        scenario_id = f"train_env_{int(scene_index):03d}"
        candidate_path = (scenario_json_dir / f"{scenario_id}.json").resolve()
        if not candidate_path.exists():
            missing_scenarios.append(scenario_id)
            continue
        scenario_paths.append(candidate_path)

    if missing_scenarios:
        raise FileNotFoundError(
            "The requested scenario cycle selection references missing scenario JSON files under "
            f"{scenario_json_dir}: {', '.join(missing_scenarios)}"
        )
    return scenario_paths


def resolve_resume_path(args: argparse.Namespace) -> Path | None:
    if args.resume_from is None:
        return None

    resume_path = Path(args.resume_from).resolve()
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume model not found: {resume_path}")
    if resume_path.suffix.lower() != ".zip":
        raise ValueError(f"--resume-from must point to a .zip RL model, got: {resume_path}")
    return resume_path


def resolve_resume_policy_weights_path(args: argparse.Namespace) -> Path | None:
    if args.resume_policy_weights is None:
        return None

    weights_path = Path(args.resume_policy_weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Resume policy weights not found: {weights_path}")
    if weights_path.suffix.lower() not in {".pth", ".pt"}:
        raise ValueError(f"--resume-policy-weights must point to a .pth or .pt file, got: {weights_path}")
    return weights_path


def resolve_bc_weights_path(args: argparse.Namespace) -> Path | None:
    if args.bc_weights is None:
        return None

    weights_path = Path(args.bc_weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"BC actor checkpoint not found: {weights_path}")
    if weights_path.suffix.lower() not in {".pth", ".pt"}:
        raise ValueError(f"--bc-weights must point to a .pth or .pt file, got: {weights_path}")
    return weights_path


def _cpu_policy_state_dict(model: BaseAlgorithm) -> dict[str, torch.Tensor]:
    policy_state = model.policy.state_dict()
    return {key: value.detach().cpu() for key, value in policy_state.items()}


def _load_resume_policy_snapshot(
    resume_path: Path,
    *,
    device: str,
    load_candidates: list[type[BaseAlgorithm]],
) -> tuple[dict[str, torch.Tensor], int]:
    load_errors: list[str] = []
    for algorithm_class in load_candidates:
        try:
            loaded_model = algorithm_class.load(str(resume_path), device=device)
        except Exception as exc:
            load_errors.append(f"{algorithm_class.__name__}: {exc}")
            continue
        policy_state = _cpu_policy_state_dict(loaded_model)
        num_timesteps = int(loaded_model.num_timesteps)
        return policy_state, num_timesteps
    raise RuntimeError(
        "Failed to load resume checkpoint with the supported algorithms: "
        + "; ".join(load_errors)
    )


def _load_resume_policy_weights_snapshot(weights_path: Path) -> tuple[dict[str, torch.Tensor], int]:
    payload = torch.load(weights_path, map_location="cpu")
    policy_state = payload.get("policy_state_dict")
    if not isinstance(policy_state, dict):
        raise KeyError(f"Policy weights checkpoint does not contain 'policy_state_dict': {weights_path}")
    num_timesteps = int(payload.get("num_timesteps", 0))
    return policy_state, num_timesteps


def _estimate_rollout_buffer_bytes(max_steps: int, env_config) -> int:
    image_height = int(env_config.camera.height)
    image_width = int(env_config.camera.width)
    image_bytes = int(max_steps) * image_height * image_width * 3
    imu_bytes = int(max_steps) * 5 * 4
    scalar_bytes = int(max_steps) * 6 * 4
    action_bytes = int(max_steps) * 4
    return image_bytes + imu_bytes + scalar_bytes + action_bytes


def _detect_latest_cycle_update_index(checkpoint_dir: Path, model_name: str) -> int:
    if not checkpoint_dir.exists():
        return 0

    pattern = re.compile(rf"^{re.escape(model_name)}_update_(\d{{6}})\.zip$", re.IGNORECASE)
    latest_index = 0
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        latest_index = max(latest_index, int(match.group(1)))
    return int(latest_index)


def _extract_cycle_update_index(path: Path | None) -> int:
    if path is None:
        return 0
    match = re.search(r"_update_(\d{6})(?:_policy)?\.(?:zip|pth|pt)$", path.name, re.IGNORECASE)
    if match is None:
        return 0
    return int(match.group(1))


def _unwrap_vec_env_envs(vec_env) -> list[FishPathAvoidEnv]:
    current = vec_env
    while hasattr(current, "venv"):
        current = current.venv
    envs = getattr(current, "envs", None)
    if envs is None:
        raise TypeError(f"Unsupported VecEnv wrapper chain: {type(current)!r}")
    return list(envs)


def _default_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_p{os.getpid()}"


def _sanitize_run_id(run_id: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in run_id)
    cleaned = cleaned.strip("_")
    return cleaned or _default_run_id()


def _with_run_id(path: Path, run_id: str) -> Path:
    return path.with_name(f"{path.stem}_{run_id}{path.suffix}")


def _parse_csv_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _parse_csv_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_csv_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _parse_auto_or_float(value: Any) -> str | float:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Expected a float or the string 'auto'.")
        if stripped.lower().startswith("auto"):
            return stripped
        return float(stripped)
    return float(value)


def _relative_path_text(path: Path | str | None) -> str | None:
    if path is None:
        return None

    path_obj = Path(path)
    if path_obj.is_absolute():
        try:
            return path_obj.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path_obj.as_posix()
    return path_obj.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _latest_train_loss_from_model(model: BaseAlgorithm | None) -> float | None:
    if model is None:
        return None

    logger = getattr(model, "logger", None)
    values = getattr(logger, "name_to_value", None)
    if not isinstance(values, dict) or not values:
        return None

    direct_loss = values.get("train/loss")
    if direct_loss is not None:
        try:
            return float(direct_loss)
        except (TypeError, ValueError):
            return None

    actor_loss = values.get("train/actor_loss")
    critic_loss = values.get("train/critic_loss")
    ent_coef_loss = values.get("train/ent_coef_loss")
    if actor_loss is None and critic_loss is None and ent_coef_loss is None:
        return None

    total_loss = 0.0
    for item in (actor_loss, critic_loss, ent_coef_loss):
        if item is None:
            continue
        try:
            total_loss += float(item)
        except (TypeError, ValueError):
            continue
    return float(total_loss)


def _convert_legacy_episode_row(
    row: dict[str, str],
    *,
    episode_index: int,
    scenario_path: Path | None,
) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id", "legacy"),
        "scenario_path": row.get("scenario_path", "" if scenario_path is None else (_relative_path_text(scenario_path) or "")),
        "scenario_id": row.get("scenario_id", ""),
        "episode_index": int(episode_index),
        "num_timesteps": _parse_csv_int(row.get("num_timesteps"), 0),
        "train_loss": _parse_csv_float(row.get("train_loss"), float("nan")),
        "episode_reward": _parse_csv_float(row.get("episode_reward"), 0.0),
        "episode_length": _parse_csv_int(row.get("episode_length"), 0),
        "episode_time_sec": _parse_csv_float(row.get("episode_time_sec"), 0.0),
        "episode_train_time_sec": _parse_csv_float(row.get("episode_train_time_sec"), 0.0),
        "termination_reason": str(row.get("termination_reason", "unknown")),
        "episode_return": _parse_csv_float(row.get("episode_return", row.get("episode_reward", 0.0)), 0.0),
        "goal_progress_ratio": _parse_csv_float(row.get("goal_progress_ratio"), 0.0),
        "distance_to_goal_region": _parse_csv_float(row.get("distance_to_goal_region"), 0.0),
        "visual_obstacle_detected": _parse_csv_bool(row.get("visual_obstacle_detected")),
        "visual_obstacle_pixel_fraction": _parse_csv_float(row.get("visual_obstacle_pixel_fraction"), 0.0),
        "visual_obstacle_center_fraction": _parse_csv_float(row.get("visual_obstacle_center_fraction"), 0.0),
        "visual_obstacle_nearest_depth": _parse_csv_float(row.get("visual_obstacle_nearest_depth"), 0.0),
        "success": _parse_csv_bool(row.get("success")),
        "collision": _parse_csv_bool(row.get("collision")),
        "wall_collision": _parse_csv_bool(row.get("wall_collision")),
        "out_of_bounds": _parse_csv_bool(row.get("out_of_bounds")),
        "timeout": _parse_csv_bool(row.get("timeout")),
    }


def prepare_episode_metrics_csv(csv_path: Path, scenario_path: Path | None) -> int:
    if not csv_path.exists():
        return 0

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if fieldnames == EPISODE_METRICS_FIELDNAMES:
        return len(rows)

    backup_path = _with_run_id(csv_path, f"legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    csv_path.replace(backup_path)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EPISODE_METRICS_FIELDNAMES)
        writer.writeheader()
        for episode_index, row in enumerate(rows, start=1):
            writer.writerow(_convert_legacy_episode_row(row, episode_index=episode_index, scenario_path=scenario_path))

    print(f"Upgraded legacy episode metrics file. Backup saved to {backup_path}")
    return len(rows)


def load_recent_episode_history(csv_path: Path, max_rows: int) -> list[dict[str, Any]]:
    if max_rows <= 0 or not csv_path.exists():
        return []

    recent_rows: deque[dict[str, Any]] = deque(maxlen=max_rows)
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            recent_rows.append(
                {
                    "episode_reward": _parse_csv_float(row.get("episode_reward"), 0.0),
                    "success": _parse_csv_bool(row.get("success")),
                    "collision": _parse_csv_bool(row.get("collision")),
                    "wall_collision": _parse_csv_bool(row.get("wall_collision")),
                    "out_of_bounds": _parse_csv_bool(row.get("out_of_bounds")),
                    "timeout": _parse_csv_bool(row.get("timeout")),
                }
            )
    return list(recent_rows)


def load_recent_cycle_episode_rows(csv_path: Path, max_rows: int) -> list[dict[str, Any]]:
    if max_rows <= 0 or not csv_path.exists():
        return []

    recent_rows: deque[dict[str, Any]] = deque(maxlen=max_rows)
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            recent_rows.append(
                {
                    "success": _parse_csv_bool(row.get("success")),
                    "episode_reward": _parse_csv_float(row.get("episode_reward"), 0.0),
                    "episode_length": _parse_csv_int(row.get("episode_length"), 0),
                    "episode_train_time_sec": _parse_csv_float(row.get("episode_train_time_sec"), 0.0),
                }
            )
    return list(recent_rows)


def save_training_artifacts(
    model: BaseAlgorithm,
    save_dir: Path,
    model_name: str,
    save_policy_weights: bool,
    *,
    suffix: str = "",
    save_replay_buffer: bool = False,
) -> tuple[Path, Path | None]:
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}{suffix}"
    model_path = save_dir / f"{stem}.zip"
    model.save(str(model_path))
    if save_replay_buffer and hasattr(model, "save_replay_buffer"):
        replay_buffer_path = save_dir / f"{stem}_replay_buffer.pkl"
        model.save_replay_buffer(str(replay_buffer_path))

    weights_path: Path | None = None
    if save_policy_weights:
        weights_path = save_dir / f"{stem}_policy.pth"
        torch.save(
            {
                "num_timesteps": int(model.num_timesteps),
                "policy_state_dict": _cpu_policy_state_dict(model),
            },
            weights_path,
        )

    return model_path, weights_path


def load_replay_buffer_if_available(model: BaseAlgorithm, model_path: Path) -> Path | None:
    if not hasattr(model, "load_replay_buffer"):
        return None
    replay_buffer_path = model_path.with_name(f"{model_path.stem}_replay_buffer.pkl")
    if not replay_buffer_path.exists():
        return None
    model.load_replay_buffer(str(replay_buffer_path))
    return replay_buffer_path


def _move_optimizer_state_to_device(optimizer: Any, device: torch.device) -> None:
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _move_sac_model_to_device(model: SAC, device: str | torch.device) -> None:
    target_device = torch.device(device)
    model.device = target_device
    model.policy.to(target_device)
    model._create_aliases()

    if getattr(model, "log_ent_coef", None) is not None:
        model.log_ent_coef.data = model.log_ent_coef.data.to(device=target_device)
    if getattr(model, "ent_coef_tensor", None) is not None:
        model.ent_coef_tensor = model.ent_coef_tensor.to(device=target_device)

    _move_optimizer_state_to_device(model.actor.optimizer, target_device)
    _move_optimizer_state_to_device(model.critic.optimizer, target_device)
    _move_optimizer_state_to_device(getattr(model, "ent_coef_optimizer", None), target_device)


class WeightCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_dir: Path,
        model_name: str,
        save_freq: int,
        save_policy_weights: bool,
        save_replay_buffer: bool = False,
    ) -> None:
        super().__init__(verbose=0)
        self.save_dir = save_dir
        self.model_name = model_name
        self.save_freq = max(0, int(save_freq))
        self.save_policy_weights = save_policy_weights
        self.save_replay_buffer = bool(save_replay_buffer)
        self._last_save_timestep = 0

    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True

        if self.num_timesteps < self._last_save_timestep + self.save_freq:
            return True

        step_suffix = f"_step_{int(self.num_timesteps)}"
        model_path, weights_path = save_training_artifacts(
            model=self.model,
            save_dir=self.save_dir,
            model_name=self.model_name,
            save_policy_weights=self.save_policy_weights,
            suffix=step_suffix,
            save_replay_buffer=self.save_replay_buffer,
        )
        weights_message = f", policy weights to {_relative_path_text(weights_path)}" if weights_path is not None else ""
        print(f"Checkpoint saved to {_relative_path_text(model_path)}{weights_message}")
        self._last_save_timestep = int(self.num_timesteps)
        return True


class CycleCheckpointCallback(BaseCallback):
    def __init__(
        self,
        *,
        save_dir: Path,
        metrics_csv_path: Path | None,
        model_name: str,
        run_id: str,
        save_policy_weights: bool,
        save_replay_buffer: bool = False,
        episodes_per_cycle: int,
        initial_episode_count: int = 0,
        initial_cycle_index: int = 0,
        align_to_episode_count: bool = False,
        initial_cycle_episode_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self.save_dir = save_dir
        self.metrics_csv_path = metrics_csv_path
        self.model_name = model_name
        self.run_id = run_id
        self.save_policy_weights = bool(save_policy_weights)
        self.save_replay_buffer = bool(save_replay_buffer)
        self.episodes_per_cycle = max(0, int(episodes_per_cycle))
        self.completed_episodes = max(0, int(initial_episode_count))
        self.next_cycle_index = max(0, int(initial_cycle_index)) + 1
        self.align_to_episode_count = bool(align_to_episode_count)
        initial_rows = list(initial_cycle_episode_rows or [])
        if self.episodes_per_cycle > 0:
            initial_rows = initial_rows[-self.episodes_per_cycle :]
        self._current_cycle_rows = initial_rows
        self._next_checkpoint_episode = self._compute_initial_checkpoint_episode()
        self._metrics_file = None
        self._metrics_writer = None

    def _compute_initial_checkpoint_episode(self) -> int:
        if self.episodes_per_cycle <= 0:
            return 0
        if not self.align_to_episode_count:
            return self.completed_episodes + self.episodes_per_cycle
        remainder = self.completed_episodes % self.episodes_per_cycle
        if remainder == 0:
            return self.completed_episodes + self.episodes_per_cycle
        return self.completed_episodes + (self.episodes_per_cycle - remainder)

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.metrics_csv_path is None:
            return

        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = True
        mode = "w"
        if self.metrics_csv_path.exists():
            mode = "a"
            write_header = self.metrics_csv_path.stat().st_size <= 0

        self._metrics_file = self.metrics_csv_path.open(mode, newline="", encoding="utf-8")
        self._metrics_writer = csv.DictWriter(self._metrics_file, fieldnames=CHECKPOINT_METRICS_FIELDNAMES)
        if write_header:
            self._metrics_writer.writeheader()
            self._metrics_file.flush()

    def _on_training_end(self) -> None:
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
            self._metrics_writer = None

    def _write_cycle_metrics(
        self,
        *,
        update_index: int,
        model_path: Path,
        weights_path: Path | None,
        cycle_rows: list[dict[str, Any]],
    ) -> tuple[float, int, int]:
        episodes_in_cycle = len(cycle_rows)
        success_count = int(sum(1 for row in cycle_rows if bool(row.get("success", False))))
        success_rate = float(success_count / episodes_in_cycle) if episodes_in_cycle > 0 else 0.0
        mean_episode_reward = (
            float(np.mean([float(row.get("episode_reward", 0.0)) for row in cycle_rows]))
            if cycle_rows
            else 0.0
        )
        mean_episode_length = (
            float(np.mean([int(row.get("episode_length", 0)) for row in cycle_rows]))
            if cycle_rows
            else 0.0
        )
        cycle_train_time_sec = float(sum(float(row.get("episode_train_time_sec", 0.0)) for row in cycle_rows))

        if self._metrics_writer is not None and self._metrics_file is not None:
            self._metrics_writer.writerow(
                {
                    "run_id": self.run_id,
                    "update_index": int(update_index),
                    "num_timesteps": int(self.num_timesteps),
                    "episodes_in_cycle": int(episodes_in_cycle),
                    "episodes_completed_total": int(self.completed_episodes),
                    "success_count": int(success_count),
                    "success_rate": float(success_rate),
                    "mean_episode_reward": float(mean_episode_reward),
                    "mean_episode_length": float(mean_episode_length),
                    "cycle_train_time_sec": float(cycle_train_time_sec),
                    "model_path": _relative_path_text(model_path) or "",
                    "policy_weights_path": _relative_path_text(weights_path) or "",
                }
            )
            self._metrics_file.flush()

        return success_rate, success_count, episodes_in_cycle

    def _on_step(self) -> bool:
        if self.episodes_per_cycle <= 0:
            return True

        infos = self.locals.get("infos", [])
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue
            self._current_cycle_rows.append(
                {
                    "success": bool(info.get("success", False)),
                    "episode_reward": float(episode_info.get("r", 0.0)),
                    "episode_length": int(episode_info.get("l", 0)),
                    "episode_train_time_sec": float(info.get("episode_train_time_sec", 0.0)),
                }
            )
            if len(self._current_cycle_rows) > self.episodes_per_cycle:
                self._current_cycle_rows = self._current_cycle_rows[-self.episodes_per_cycle :]
            self.completed_episodes += 1
            while self.completed_episodes >= self._next_checkpoint_episode:
                cycle_rows = list(self._current_cycle_rows)
                update_index = self.next_cycle_index
                cycle_suffix = f"_update_{self.next_cycle_index:06d}"
                model_path, weights_path = save_training_artifacts(
                    model=self.model,
                    save_dir=self.save_dir,
                    model_name=self.model_name,
                    save_policy_weights=self.save_policy_weights,
                    suffix=cycle_suffix,
                    save_replay_buffer=self.save_replay_buffer,
                )
                success_rate, success_count, episodes_in_cycle = self._write_cycle_metrics(
                    update_index=update_index,
                    model_path=model_path,
                    weights_path=weights_path,
                    cycle_rows=cycle_rows,
                )
                weights_message = (
                    f", policy weights to {_relative_path_text(weights_path)}"
                    if weights_path is not None
                    else ""
                )
                print(
                    "Saved cycle checkpoint to "
                    f"{_relative_path_text(model_path)}{weights_message} | "
                    f"cycle_success_rate={success_rate:.3f} ({success_count}/{episodes_in_cycle})"
                )
                self._current_cycle_rows.clear()
                self.next_cycle_index += 1
                self._next_checkpoint_episode += self.episodes_per_cycle
        return True


class EpisodeMetricsCallback(BaseCallback):
    def __init__(
        self,
        csv_path: Path,
        *,
        run_id: str,
        scenario_path: Path | None,
        initial_episode_index: int = 0,
        append: bool = True,
    ) -> None:
        super().__init__(verbose=0)
        self.csv_path = csv_path
        self.run_id = run_id
        self.scenario_path = "" if scenario_path is None else (_relative_path_text(scenario_path) or "")
        self.initial_episode_index = max(0, int(initial_episode_index))
        self.append = bool(append)
        self._file = None
        self._writer = None
        self._episode_counter = self.initial_episode_index
        self.completed_episodes_this_run = 0
        self._training_start_perf = 0.0
        self._last_episode_perf = 0.0
        self._latest_train_loss: float | None = None

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = True
        mode = "w"
        if self.append and self.csv_path.exists():
            mode = "a"
            write_header = self.csv_path.stat().st_size <= 0

        self._file = self.csv_path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=EPISODE_METRICS_FIELDNAMES)
        if write_header:
            self._writer.writeheader()
            self._file.flush()
        self._training_start_perf = time.perf_counter()
        self._last_episode_perf = self._training_start_perf

    def _on_step(self) -> bool:
        if self._writer is None or self._file is None:
            return True

        latest_train_loss = _latest_train_loss_from_model(getattr(self, "model", None))
        if latest_train_loss is not None:
            self._latest_train_loss = float(latest_train_loss)

        infos = self.locals.get("infos", [])
        wrote_row = False
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue

            self._episode_counter += 1
            self.completed_episodes_this_run += 1
            episode_wall_clock_now = time.perf_counter()
            episode_train_time_sec = max(0.0, episode_wall_clock_now - self._last_episode_perf)
            self._last_episode_perf = episode_wall_clock_now
            info["episode_train_time_sec"] = float(episode_train_time_sec)
            row = {
                "run_id": self.run_id,
                "scenario_path": _relative_path_text(info.get("scenario_path", self.scenario_path)) or "",
                "scenario_id": str(info.get("scenario_id", "")),
                "episode_index": int(self._episode_counter),
                "num_timesteps": int(self.num_timesteps),
                "train_loss": float(self._latest_train_loss) if self._latest_train_loss is not None else float("nan"),
                "episode_reward": float(episode_info.get("r", 0.0)),
                "episode_length": int(episode_info.get("l", 0)),
                "episode_time_sec": float(episode_info.get("t", 0.0)),
                "episode_train_time_sec": float(episode_train_time_sec),
                "termination_reason": str(info.get("termination_reason", "unknown")),
                "episode_return": float(info.get("episode_return", episode_info.get("r", 0.0))),
                "goal_progress_ratio": float(info.get("goal_progress_ratio", 0.0)),
                "distance_to_goal_region": float(info.get("distance_to_goal_region", 0.0)),
                "visual_obstacle_detected": bool(info.get("visual_obstacle_detected", False)),
                "visual_obstacle_pixel_fraction": float(info.get("visual_obstacle_pixel_fraction", 0.0)),
                "visual_obstacle_center_fraction": float(info.get("visual_obstacle_center_fraction", 0.0)),
                "visual_obstacle_nearest_depth": float(info.get("visual_obstacle_nearest_depth", 0.0)),
                "success": bool(info.get("success", False)),
                "collision": bool(info.get("collision", False)),
                "wall_collision": bool(info.get("wall_collision", False)),
                "out_of_bounds": bool(info.get("out_of_bounds", False)),
                "timeout": bool(info.get("timeout", False)),
            }
            self._writer.writerow(row)
            wrote_row = True
            print(
                "Episode "
                f"{self._episode_counter} stopped: {row['termination_reason']} | "
                f"reward={row['episode_reward']:.3f} | "
                f"steps={row['episode_length']} | "
                f"train_time={row['episode_train_time_sec']:.2f}s"
            )

        if wrote_row:
            self._file.flush()
        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
        self._file = None
        self._writer = None


class EpisodeArtifactCallback(BaseCallback):
    def __init__(
        self,
        video_dir: Path,
        checkpoint_dir: Path,
        model_name: str,
        save_every_episodes: int,
        fps: int,
        save_policy_weights: bool,
        initial_completed_episodes: int = 0,
        save_checkpoint_artifacts: bool = True,
        save_replay_buffer: bool = False,
    ) -> None:
        super().__init__(verbose=0)
        self.video_dir = video_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.save_every_episodes = max(0, int(save_every_episodes))
        self.fps = max(1, int(fps))
        self.save_policy_weights = save_policy_weights
        self.completed_episodes = max(0, int(initial_completed_episodes))
        self.save_checkpoint_artifacts = bool(save_checkpoint_artifacts)
        self.save_replay_buffer = bool(save_replay_buffer)
        self._envs: list[FishPathAvoidEnv] = []
        self.top_video_dir = self.video_dir / "top_view"
        self.head_video_dir = self.video_dir / "head Cemara"

    def _on_training_start(self) -> None:
        self.top_video_dir.mkdir(parents=True, exist_ok=True)
        self.head_video_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._envs = _unwrap_vec_env_envs(self.training_env)

    def _on_step(self) -> bool:
        if self.save_every_episodes <= 0:
            return True

        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            if info.get("episode") is None:
                continue

            self.completed_episodes += 1
            if self.completed_episodes % self.save_every_episodes != 0:
                continue

            env = self._envs[env_idx]
            episode_suffix = f"_episode_{self.completed_episodes:06d}"
            video_path = self.top_video_dir / f"{self.model_name}{episode_suffix}.mp4"
            head_video_path = self.head_video_dir / f"{self.model_name}{episode_suffix}_head.mp4"
            saved_path, saved_head_path = env.save_completed_episode_videos(
                video_path,
                head_video_path,
                fps=self.fps,
            )
            model_path: Path | None = None
            weights_path: Path | None = None
            if self.save_checkpoint_artifacts:
                model_path, weights_path = save_training_artifacts(
                    model=self.model,
                    save_dir=self.checkpoint_dir,
                    model_name=self.model_name,
                    save_policy_weights=self.save_policy_weights,
                    suffix=episode_suffix,
                    save_replay_buffer=self.save_replay_buffer,
                )
            weights_message = f", weights to {_relative_path_text(weights_path)}" if weights_path is not None else ""
            if saved_path is not None:
                head_message = (
                    f"; head camera video to {_relative_path_text(saved_head_path)}"
                    if saved_head_path is not None
                    else ""
                )
                if model_path is not None:
                    print(
                        "Saved episode video to "
                        f"{_relative_path_text(saved_path)}{head_message}; "
                        f"checkpoint to {_relative_path_text(model_path)}{weights_message}"
                    )
                else:
                    print(f"Saved episode video to {_relative_path_text(saved_path)}{head_message}")
            else:
                if model_path is not None:
                    print(f"Saved episode checkpoint to {_relative_path_text(model_path)}{weights_message}")

        return True


class StopAfterEpisodesCallback(BaseCallback):
    def __init__(self, max_episodes: int) -> None:
        super().__init__(verbose=0)
        self.max_episodes = int(max_episodes)
        self.completed_episodes = 0
        self.stopped_due_to_limit = False

    def _on_step(self) -> bool:
        if self.max_episodes <= 0:
            return True

        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("episode") is None:
                continue
            self.completed_episodes += 1
            if self.completed_episodes >= self.max_episodes:
                self.stopped_due_to_limit = True
                print(f"Reached max episode limit: {self.completed_episodes}. Stopping training.")
                return False
        return True


class ConvergenceStopCallback(BaseCallback):
    def __init__(
        self,
        *,
        window_episodes: int,
        min_episodes: int,
        min_success_rate: float,
        max_timeout_rate: float,
        max_failure_rate: float,
        reward_window: int,
        reward_stability_ratio: float,
        initial_history: list[dict[str, Any]] | None = None,
        initial_episode_count: int = 0,
    ) -> None:
        super().__init__(verbose=0)
        self.window_episodes = max(1, int(window_episodes))
        self.min_episodes = max(self.window_episodes, int(min_episodes))
        self.min_success_rate = float(min_success_rate)
        self.max_timeout_rate = float(max_timeout_rate)
        self.max_failure_rate = float(max_failure_rate)
        self.reward_window = max(0, int(reward_window))
        self.reward_stability_ratio = float(reward_stability_ratio)
        self.completed_episodes = max(0, int(initial_episode_count))
        history_size = max(
            self.window_episodes,
            2 * self.reward_window if self.reward_window > 0 and self.reward_stability_ratio >= 0.0 else 0,
        )
        self._history: deque[dict[str, Any]] = deque(maxlen=max(1, history_size))
        for row in initial_history or []:
            self._history.append(
                {
                    "episode_reward": float(row["episode_reward"]),
                    "success": bool(row["success"]),
                    "collision": bool(row["collision"]),
                    "wall_collision": bool(row["wall_collision"]),
                    "out_of_bounds": bool(row["out_of_bounds"]),
                    "timeout": bool(row["timeout"]),
                }
            )
        self.converged = False
        self.summary: dict[str, Any] | None = None
        self.latest_window_summary: dict[str, Any] | None = None

    def _reward_stability_summary(self) -> tuple[bool, float | None]:
        if self.reward_window <= 0 or self.reward_stability_ratio < 0.0:
            return True, None

        if len(self._history) < 2 * self.reward_window:
            return False, None

        history_rows = list(self._history)
        previous_rewards = [row["episode_reward"] for row in history_rows[-2 * self.reward_window : -self.reward_window]]
        recent_rewards = [row["episode_reward"] for row in history_rows[-self.reward_window :]]
        previous_mean = float(np.mean(previous_rewards)) if previous_rewards else 0.0
        recent_mean = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        denominator = max(abs(previous_mean), 1.0)
        relative_delta = abs(recent_mean - previous_mean) / denominator
        return relative_delta <= self.reward_stability_ratio, float(relative_delta)

    def _window_summary(self) -> dict[str, Any] | None:
        if self.completed_episodes < self.min_episodes or len(self._history) < self.window_episodes:
            return None

        window_rows = list(self._history)[-self.window_episodes :]
        success_rate = float(np.mean([row["success"] for row in window_rows]))
        timeout_rate = float(np.mean([row["timeout"] for row in window_rows]))
        obstacle_collision_rate = float(np.mean([row["collision"] for row in window_rows]))
        wall_collision_rate = float(np.mean([row["wall_collision"] for row in window_rows]))
        out_of_bounds_rate = float(np.mean([row["out_of_bounds"] for row in window_rows]))
        failure_rate = float(
            np.mean(
                [
                    row["collision"] or row["wall_collision"] or row["out_of_bounds"]
                    for row in window_rows
                ]
            )
        )
        mean_reward = float(np.mean([row["episode_reward"] for row in window_rows]))
        reward_stable, reward_relative_delta = self._reward_stability_summary()

        return {
            "episodes_seen_total": int(self.completed_episodes),
            "window_episodes": int(self.window_episodes),
            "mean_reward": mean_reward,
            "success_rate": success_rate,
            "timeout_rate": timeout_rate,
            "obstacle_collision_rate": obstacle_collision_rate,
            "wall_collision_rate": wall_collision_rate,
            "out_of_bounds_rate": out_of_bounds_rate,
            "failure_rate": failure_rate,
            "reward_stable": bool(reward_stable),
            "reward_relative_delta": reward_relative_delta,
        }

    def _criteria_met(self, summary: dict[str, Any]) -> bool:
        if summary["success_rate"] < self.min_success_rate:
            return False
        if summary["timeout_rate"] > self.max_timeout_rate:
            return False
        if summary["failure_rate"] > self.max_failure_rate:
            return False
        if not summary["reward_stable"]:
            return False
        return True

    def _on_step(self) -> bool:
        continue_training = True
        infos = self.locals.get("infos", [])
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue

            self.completed_episodes += 1
            self._history.append(
                {
                    "episode_reward": float(episode_info.get("r", 0.0)),
                    "success": bool(info.get("success", False)),
                    "collision": bool(info.get("collision", False)),
                    "wall_collision": bool(info.get("wall_collision", False)),
                    "out_of_bounds": bool(info.get("out_of_bounds", False)),
                    "timeout": bool(info.get("timeout", False)),
                }
            )
            summary = self._window_summary()
            if summary is None:
                continue

            self.latest_window_summary = summary
            if not self._criteria_met(summary):
                continue

            self.converged = True
            self.summary = summary
            print(
                "Convergence reached: "
                f"success={summary['success_rate']:.3f}, "
                f"timeout={summary['timeout_rate']:.3f}, "
                f"failure={summary['failure_rate']:.3f}, "
                f"mean_reward={summary['mean_reward']:.3f}, "
                f"episodes={summary['episodes_seen_total']}"
            )
            continue_training = False

        return continue_training


class EpisodeRewardPlotCallback(BaseCallback):
    def __init__(
        self,
        plot_path: Path,
        moving_average_window: int = 20,
        save_every_episodes: int = 1,
    ) -> None:
        super().__init__(verbose=0)
        self.plot_path = plot_path
        self.moving_average_window = max(1, int(moving_average_window))
        self.step_average_window = max(5, int(moving_average_window))
        self.save_every_episodes = max(1, int(save_every_episodes))
        self._timesteps: list[int] = []
        self._step_rewards: list[float] = []
        self._step_average_rewards: list[float] = []
        self._episode_indices: list[int] = []
        self._episode_rewards: list[float] = []
        self._moving_average_rewards: list[float] = []
        self._last_saved_episode = 0
        self._last_saved_step = 0
        self._plt = None
        self._figure = None
        self._step_axes = None
        self._episode_axes = None
        self._step_line = None
        self._step_avg_line = None
        self._reward_line = None
        self._avg_line = None

    def _on_training_start(self) -> None:
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Reward plot disabled because matplotlib could not be initialized: {exc}")
            return

        self._plt = plt
        self._plt.ion()
        self._figure, axes = self._plt.subplots(2, 1, figsize=(9.0, 7.0))
        self._step_axes, self._episode_axes = axes
        self._step_line, = self._step_axes.plot([], [], color="tab:green", linewidth=1.0, label="Step reward")
        self._step_avg_line, = self._step_axes.plot(
            [],
            [],
            color="tab:olive",
            linewidth=1.8,
            label=f"{self.step_average_window}-step moving avg",
        )
        self._reward_line, = self._episode_axes.plot([], [], color="tab:blue", linewidth=1.25, label="Episode reward")
        self._avg_line, = self._episode_axes.plot(
            [],
            [],
            color="tab:orange",
            linewidth=2.0,
            label=f"{self.moving_average_window}-episode moving avg",
        )
        self._step_axes.set_xlabel("Timesteps")
        self._step_axes.set_ylabel("Step Reward")
        self._step_axes.set_title("Realtime Step Reward")
        self._step_axes.grid(alpha=0.3)
        self._step_axes.legend(loc="best")
        self._episode_axes.set_xlabel("Episode")
        self._episode_axes.set_ylabel("Episode Reward")
        self._episode_axes.set_title("Episode Reward")
        self._episode_axes.grid(alpha=0.3)
        self._episode_axes.legend(loc="best")
        self._figure.tight_layout()
        self._plt.show(block=False)
        self._plt.pause(0.001)

    def _append_step_reward(self, reward: float) -> None:
        self._timesteps.append(int(self.num_timesteps))
        self._step_rewards.append(reward)
        window_start = max(0, len(self._step_rewards) - self.step_average_window)
        window_rewards = self._step_rewards[window_start:]
        moving_average = sum(window_rewards) / float(len(window_rewards))
        self._step_average_rewards.append(moving_average)

    def _append_episode_reward(self, reward: float) -> None:
        episode_index = len(self._episode_rewards) + 1
        self._episode_indices.append(episode_index)
        self._episode_rewards.append(reward)
        window_start = max(0, len(self._episode_rewards) - self.moving_average_window)
        window_rewards = self._episode_rewards[window_start:]
        moving_average = sum(window_rewards) / float(len(window_rewards))
        self._moving_average_rewards.append(moving_average)

    def _refresh_plot(self) -> None:
        if self._plt is None or self._figure is None or self._step_axes is None or self._episode_axes is None:
            return

        if not self._plt.fignum_exists(self._figure.number):
            return

        if self._timesteps:
            self._step_line.set_data(self._timesteps, self._step_rewards)
            self._step_avg_line.set_data(self._timesteps, self._step_average_rewards)
            self._step_axes.relim()
            self._step_axes.autoscale_view()
            latest_step_reward = self._step_rewards[-1]
            latest_step_average = self._step_average_rewards[-1]
            self._step_axes.set_title(
                f"Realtime Step Reward | latest={latest_step_reward:.3f} | "
                f"avg{self.step_average_window}={latest_step_average:.3f}"
            )

        if self._episode_indices:
            self._reward_line.set_data(self._episode_indices, self._episode_rewards)
            self._avg_line.set_data(self._episode_indices, self._moving_average_rewards)
            self._episode_axes.relim()
            self._episode_axes.autoscale_view()
            latest_reward = self._episode_rewards[-1]
            latest_average = self._moving_average_rewards[-1]
            self._episode_axes.set_title(
                f"Episode Reward | latest={latest_reward:.2f} | "
                f"avg{self.moving_average_window}={latest_average:.2f}"
            )
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()
        self._plt.pause(0.001)

    def _save_plot(self, *, force: bool = False) -> None:
        if self._figure is None:
            return
        step_delta = len(self._step_rewards) - self._last_saved_step
        episode_delta = len(self._episode_rewards) - self._last_saved_episode
        if not force and step_delta < self.step_average_window and episode_delta < self.save_every_episodes:
            return
        self._figure.savefig(self.plot_path, dpi=160)
        self._last_saved_episode = len(self._episode_rewards)
        self._last_saved_step = len(self._step_rewards)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float)
        if rewards.size > 0:
            self._append_step_reward(float(np.mean(rewards)))

        episode_updated = False
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue
            self._append_episode_reward(float(episode_info.get("r", 0.0)))
            episode_updated = True

        if rewards.size > 0 or episode_updated:
            self._refresh_plot()
            self._save_plot()
        return True

    def _on_training_end(self) -> None:
        if self._figure is not None:
            self._save_plot(force=True)
        if self._plt is not None:
            self._plt.ioff()


def main() -> None:
    args = parse_args()
    config = make_config()
    if args.algo is not None:
        config.train.algorithm = str(args.algo).lower()
    if args.log_dir is not None:
        config.train.log_dir = Path(args.log_dir).as_posix()
    if args.model_name is not None:
        config.train.model_name = str(args.model_name)
    if args.sac_buffer_size is not None:
        config.train.sac_buffer_size = int(args.sac_buffer_size)
    if args.sac_learning_starts is not None:
        config.train.sac_learning_starts = int(args.sac_learning_starts)
    if args.sac_train_freq_steps is not None:
        config.train.sac_train_freq_steps = int(args.sac_train_freq_steps)
    if args.sac_train_freq_unit is not None:
        config.train.sac_train_freq_unit = str(args.sac_train_freq_unit)
    if args.sac_gradient_steps is not None:
        config.train.sac_gradient_steps = int(args.sac_gradient_steps)
    if args.sac_tau is not None:
        config.train.sac_tau = float(args.sac_tau)
    if args.sac_ent_coef is not None:
        config.train.sac_ent_coef = str(args.sac_ent_coef)
    if args.sac_target_update_interval is not None:
        config.train.sac_target_update_interval = int(args.sac_target_update_interval)
    if args.sac_target_entropy is not None:
        config.train.sac_target_entropy = str(args.sac_target_entropy)
    if args.sac_optimize_memory_usage is not None:
        config.train.sac_optimize_memory_usage = bool(args.sac_optimize_memory_usage)
    scenario_path = resolve_scenario_path(args)
    scenario_cycle_paths = resolve_scenario_cycle_paths(args)
    resume_path = resolve_resume_path(args)
    resume_policy_weights_path = resolve_resume_policy_weights_path(args)
    bc_weights_path = resolve_bc_weights_path(args)
    run_id = _sanitize_run_id(args.run_id or _default_run_id())
    algo_name = str(config.train.algorithm).strip().lower()
    if algo_name not in {"ppo", "sac"}:
        raise ValueError(f"Unsupported algorithm {config.train.algorithm!r}. Expected 'ppo' or 'sac'.")
    if sum(path is not None for path in (resume_path, resume_policy_weights_path, bc_weights_path)) > 1:
        raise ValueError("Use only one of --resume-from, --resume-policy-weights, or --bc-weights.")
    if scenario_path is not None and scenario_cycle_paths:
        raise ValueError("Use either single-scenario mode or scenario-cycle mode, not both.")
    if args.timesteps is not None:
        config.train.total_timesteps = args.timesteps
    if args.num_envs is not None:
        config.train.num_envs = args.num_envs
    if args.record_videos is not None:
        config.train.save_episode_videos = bool(args.record_videos)
    if args.video_interval_episodes is not None:
        config.train.video_interval_episodes = args.video_interval_episodes
    if args.xml_path is not None:
        config.env.model.xml_path = Path(args.xml_path).as_posix()
    if args.render and not 0 <= args.render_env_index < config.train.num_envs:
        raise ValueError(
            f"--render-env-index must be within [0, {config.train.num_envs - 1}] when --render is enabled."
        )
    if args.render_slowdown < 0.0:
        raise ValueError("--render-slowdown must be non-negative.")
    if args.convergence_window < 0:
        raise ValueError("--convergence-window must be non-negative.")
    if args.convergence_min_episodes < 0:
        raise ValueError("--convergence-min-episodes must be non-negative.")
    if args.rollout_episodes_per_update < 0:
        raise ValueError("--rollout-episodes-per-update must be non-negative.")
    if args.rollout_step_budget < 0:
        raise ValueError("--rollout-step-budget must be non-negative.")
    if args.scenario_cycle_sample_size < 0:
        raise ValueError("--scenario-cycle-sample-size must be non-negative.")
    if args.use_lora and args.lora_rank <= 0:
        raise ValueError("--lora-rank must be positive when --use-lora is enabled.")
    if config.train.sac_buffer_size <= 0:
        raise ValueError("--sac-buffer-size must be positive.")
    if config.train.sac_learning_starts < 0:
        raise ValueError("--sac-learning-starts must be non-negative.")
    if config.train.sac_train_freq_steps <= 0:
        raise ValueError("--sac-train-freq-steps must be positive.")
    if str(config.train.sac_train_freq_unit) not in {"step", "episode"}:
        raise ValueError("--sac-train-freq-unit must be either 'step' or 'episode'.")
    if config.train.sac_gradient_steps == 0:
        raise ValueError("--sac-gradient-steps cannot be 0.")
    if config.train.sac_tau <= 0.0 or config.train.sac_tau > 1.0:
        raise ValueError("--sac-tau must be within (0, 1].")
    if config.train.sac_target_update_interval <= 0:
        raise ValueError("--sac-target-update-interval must be positive.")
    if algo_name == "sac" and config.train.sac_optimize_memory_usage:
        print(
            "SAC optimize_memory_usage is disabled because Stable-Baselines3 DictReplayBuffer "
            "does not support it for dict observations."
        )
        config.train.sac_optimize_memory_usage = False

    convergence_enabled = int(args.convergence_window) > 0
    convergence_min_episodes = int(args.convergence_min_episodes) if int(args.convergence_min_episodes) > 0 else int(args.convergence_window)
    scenario_cycle_sample_size = int(args.scenario_cycle_sample_size)
    if scenario_cycle_sample_size > 0 and not scenario_cycle_paths:
        raise ValueError("--scenario-cycle-sample-size requires scenario-cycle mode.")
    if scenario_cycle_paths and scenario_cycle_sample_size > len(scenario_cycle_paths):
        raise ValueError(
            "--scenario-cycle-sample-size cannot exceed the number of selected scenario-cycle scenes."
        )
    rollout_episodes_per_update = int(args.rollout_episodes_per_update)
    if scenario_cycle_paths and rollout_episodes_per_update <= 0:
        rollout_episodes_per_update = scenario_cycle_sample_size if scenario_cycle_sample_size > 0 else len(scenario_cycle_paths)
    if rollout_episodes_per_update > 0 and not scenario_cycle_paths:
        raise ValueError("--rollout-episodes-per-update requires scenario-cycle mode.")
    if scenario_cycle_sample_size > 0 and rollout_episodes_per_update != scenario_cycle_sample_size:
        raise ValueError(
            "--scenario-cycle-sample-size must equal --rollout-episodes-per-update so each sampled scene runs exactly one episode per cycle."
        )
    episode_cycle_enabled = rollout_episodes_per_update > 0
    if scenario_cycle_paths and config.train.num_envs != 1:
        raise ValueError("Scenario-cycle mode currently requires --num-envs 1.")
    if episode_cycle_enabled and convergence_enabled:
        print("Warning: convergence stop in scenario-cycle mode measures the mixed 100-scene stream, not per-scene convergence.")

    rollout_step_budget = 0
    if episode_cycle_enabled:
        rollout_step_budget = (
            int(args.rollout_step_budget)
            if int(args.rollout_step_budget) > 0
            else int(config.env.max_episode_steps) * int(rollout_episodes_per_update)
        )
        if algo_name == "ppo":
            if rollout_step_budget <= 0:
                raise ValueError("Episode-cycle PPO requires a positive rollout step budget.")
            config.train.n_steps = int(rollout_step_budget)
        if args.video_interval_episodes is None:
            config.train.video_interval_episodes = int(rollout_episodes_per_update)

    if (
        algo_name == "sac"
        and episode_cycle_enabled
        and str(config.train.sac_train_freq_unit) == "episode"
        and args.sac_train_freq_steps is None
    ):
        config.train.sac_train_freq_steps = int(rollout_episodes_per_update)

    if args.lora_target_modules is None:
        resolved_lora_target_modules = (
            tuple(DEFAULT_LORA_TARGET_MODULES)
            if algo_name == "ppo"
            else tuple(DEFAULT_SAC_LORA_TARGET_MODULES)
        )
    else:
        resolved_lora_target_modules = tuple(str(name) for name in args.lora_target_modules)

    log_dir = Path(config.train.log_dir)
    if scenario_path is not None:
        log_dir = log_dir / scenario_path.stem
    elif scenario_cycle_paths:
        log_dir = log_dir / "scenario_cycle"
    checkpoint_dir = log_dir / config.train.checkpoint_dirname
    checkpoint_metrics_path = log_dir / config.train.checkpoint_metrics_filename
    episode_metrics_path = log_dir / config.train.episode_metrics_filename
    video_dir = log_dir / config.train.video_dirname
    monitor_stem = Path(config.train.monitor_filename).stem
    monitor_path = log_dir / f"{monitor_stem}_{run_id}.monitor.csv"
    reward_plot_path = _with_run_id(log_dir / Path("reward_curve.png"), run_id)
    run_config_path = _with_run_id(log_dir / Path("config.json"), run_id)
    latest_summary_path = log_dir / "training_summary.json"
    run_summary_path = _with_run_id(log_dir / Path("training_summary.json"), run_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    existing_cycle_update_index = _detect_latest_cycle_update_index(checkpoint_dir, config.train.model_name)
    resume_source_update_index = max(
        _extract_cycle_update_index(resume_path),
        _extract_cycle_update_index(resume_policy_weights_path),
    )
    if episode_cycle_enabled:
        existing_cycle_update_index = max(existing_cycle_update_index, resume_source_update_index)
    existing_episode_count = prepare_episode_metrics_csv(episode_metrics_path, scenario_path)
    initial_cycle_episode_rows: list[dict[str, Any]] = []
    if (
        episode_cycle_enabled
        and rollout_episodes_per_update > 0
        and bool(args.align_rollout_updates_to_episode_count)
    ):
        pending_cycle_episodes = existing_episode_count % rollout_episodes_per_update
        if pending_cycle_episodes > 0:
            initial_cycle_episode_rows = load_recent_cycle_episode_rows(
                episode_metrics_path,
                pending_cycle_episodes,
            )
    convergence_history_size = max(
        int(args.convergence_window),
        2 * int(args.convergence_reward_window)
        if int(args.convergence_reward_window) > 0 and float(args.convergence_reward_stability_ratio) >= 0.0
        else 0,
    )
    initial_convergence_history = load_recent_episode_history(episode_metrics_path, convergence_history_size)
    config_payload = config_to_dict(config)
    config_payload["run_id"] = run_id
    config_payload["monitor_path"] = _relative_path_text(monitor_path)
    config_payload["episode_metrics_path"] = _relative_path_text(episode_metrics_path)
    config_payload["checkpoint_metrics_path"] = _relative_path_text(checkpoint_metrics_path)
    config_payload["reward_plot_path"] = _relative_path_text(reward_plot_path)
    config_payload["existing_episode_count_at_start"] = int(existing_episode_count)
    config_payload["existing_cycle_update_index_at_start"] = int(existing_cycle_update_index)
    config_payload["lora"] = {
        "enabled": bool(args.use_lora),
        "rank": int(args.lora_rank),
        "alpha": float(args.lora_alpha),
        "dropout": float(args.lora_dropout),
        "target_modules": list(resolved_lora_target_modules),
        "freeze_actor_base": bool(args.lora_freeze_actor_base),
        "train_bias": bool(args.lora_train_bias),
    }
    config_payload["algorithm"] = algo_name
    config_payload["sac"] = {
        "buffer_size": int(config.train.sac_buffer_size),
        "learning_starts": int(config.train.sac_learning_starts),
        "train_freq_steps": int(config.train.sac_train_freq_steps),
        "train_freq_unit": str(config.train.sac_train_freq_unit),
        "gradient_steps": int(config.train.sac_gradient_steps),
        "tau": float(config.train.sac_tau),
        "ent_coef": _parse_auto_or_float(config.train.sac_ent_coef),
        "target_update_interval": int(config.train.sac_target_update_interval),
        "target_entropy": _parse_auto_or_float(config.train.sac_target_entropy),
        "optimize_memory_usage": bool(config.train.sac_optimize_memory_usage),
    }
    config_payload["rollout_schedule"] = {
        "episode_cycle_enabled": episode_cycle_enabled,
        "episodes_per_update": int(rollout_episodes_per_update),
        "scenario_cycle_sample_size": int(scenario_cycle_sample_size),
        "align_updates_to_episode_count": bool(args.align_rollout_updates_to_episode_count),
        "step_budget": int(rollout_step_budget),
        "strict_step_budget": bool(args.strict_rollout_step_budget),
    }
    config_payload["convergence"] = {
        "enabled": convergence_enabled,
        "window_episodes": int(args.convergence_window),
        "min_episodes": convergence_min_episodes,
        "min_success_rate": float(args.convergence_min_success_rate),
        "max_timeout_rate": float(args.convergence_max_timeout_rate),
        "max_failure_rate": float(args.convergence_max_failure_rate),
        "reward_window": int(args.convergence_reward_window),
        "reward_stability_ratio": float(args.convergence_reward_stability_ratio),
    }
    if scenario_path is not None:
        config_payload["selected_scenario_path"] = _relative_path_text(scenario_path)
    if scenario_cycle_paths:
        config_payload["selected_scenario_cycle_paths"] = [
            _relative_path_text(path) or ""
            for path in scenario_cycle_paths
        ]
    if resume_path is not None:
        config_payload["resume_from"] = _relative_path_text(resume_path)
    if resume_policy_weights_path is not None:
        config_payload["resume_policy_weights"] = _relative_path_text(resume_policy_weights_path)
    if bc_weights_path is not None:
        config_payload["bc_weights"] = _relative_path_text(bc_weights_path)
    _write_json(log_dir / "config.json", config_payload)
    _write_json(run_config_path, config_payload)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "Requested CUDA training, but the current PyTorch build has no CUDA support. "
            "Falling back to CPU."
        )
        requested_device = "cpu"
    elif requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Algorithm: {algo_name.upper()}")
    print(f"Training device: {requested_device}")
    print(f"Run ID: {run_id}")
    if scenario_path is not None:
        print(f"Training on fixed scenario: {scenario_path}")
    if scenario_cycle_paths:
        sampling_message = ""
        if scenario_cycle_sample_size > 0:
            sampling_message = f", random {scenario_cycle_sample_size}-scene batch/update"
        alignment_message = ""
        if episode_cycle_enabled and args.align_rollout_updates_to_episode_count:
            next_alignment_episode = existing_episode_count + (
                rollout_episodes_per_update - (existing_episode_count % rollout_episodes_per_update)
                if existing_episode_count % rollout_episodes_per_update != 0
                else rollout_episodes_per_update
            )
            alignment_message = f", first aligned update at episode {next_alignment_episode}"
        print(
            "Training on scenario cycle: "
            f"{len(scenario_cycle_paths)} scenes{sampling_message}{alignment_message}, "
            f"{rollout_episodes_per_update} episodes/cycle, "
            f"step budget {rollout_step_budget}"
        )
    if resume_path is not None:
        print(f"Resuming from model: {_relative_path_text(resume_path)}")
    if resume_policy_weights_path is not None:
        print(f"Resuming from policy weights: {_relative_path_text(resume_policy_weights_path)}")
    if bc_weights_path is not None:
        print(f"Initializing actor from BC weights: {_relative_path_text(bc_weights_path)}")
    if algo_name == "sac":
        print(
            "SAC train frequency: "
            f"{int(config.train.sac_train_freq_steps)} {str(config.train.sac_train_freq_unit)}(s) per update trigger"
        )
    if args.use_lora:
        print(
            "LoRA fine-tuning enabled: "
            f"rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}"
        )
    if existing_episode_count > 0:
        print(f"Appending to existing episode metrics with starting episode index {existing_episode_count}.")
    if episode_cycle_enabled and existing_cycle_update_index > 0:
        print(f"Continuing cycle checkpoint numbering from update index {existing_cycle_update_index}.")
    if config.train.save_episode_videos and config.train.video_interval_episodes > 1 and config.train.num_envs > 1:
        print(
            "Video capture is enabled with multiple environments. "
            "Episodes finish asynchronously, so frames still need to be buffered for every env. "
            "Use --num-envs 1 if you want video capture only on every saved episode and lower RAM usage."
        )
    if convergence_enabled and config.train.num_envs > 1:
        print(
            "Convergence stop is enabled with multiple environments. "
            "Episodes still count correctly, but asynchronous completion makes the stopping point less interpretable. "
            "Use --num-envs 1 for deterministic per-scenario convergence windows."
        )
    if episode_cycle_enabled and algo_name == "ppo":
        estimated_buffer_bytes = _estimate_rollout_buffer_bytes(rollout_step_budget, config.env)
        print(f"Approx rollout observation buffer: {estimated_buffer_bytes / (1024 ** 3):.2f} GiB")

    record_every_n_episodes = 1
    if config.train.save_episode_videos and config.train.video_interval_episodes > 0 and config.train.num_envs == 1:
        record_every_n_episodes = config.train.video_interval_episodes

    def make_env(rank: int):
        def _factory() -> FishPathAvoidEnv:
            env = FishPathAvoidEnv(
                config=config.env,
                enable_mujoco_viewer=args.render and rank == args.render_env_index,
                realtime_playback=args.render and args.render_slowdown > 0.0,
                viewer_slowdown=args.render_slowdown,
                enable_episode_recording=config.train.save_episode_videos,
                recording_camera_name=config.train.video_camera_name,
                recording_width=config.train.video_width,
                recording_height=config.train.video_height,
                recording_frame_stride=config.train.video_frame_stride,
                record_every_n_episodes=record_every_n_episodes,
                scenario_path=scenario_path,
                scenario_cycle_paths=scenario_cycle_paths if scenario_cycle_paths else None,
                scenario_cycle_sample_size=scenario_cycle_sample_size,
            )
            env.reset(seed=config.train.seed + rank)
            return env

        return _factory

    env = DummyVecEnv([make_env(rank) for rank in range(config.train.num_envs)])
    env = VecMonitor(env, filename=str(monitor_path))
    env = VecTransposeImage(env)

    # The policy consumes a dict observation: {"image": head camera RGB, "imu": body-frame IMU vector}.
    algorithm_class: type[BaseAlgorithm]
    if algo_name == "ppo":
        algorithm_class = EpisodeCyclePPO if episode_cycle_enabled else PPO
    else:
        algorithm_class = SAC
    if algo_name == "ppo":
        policy_spec: str | type = LoraMultiInputPolicy if args.use_lora else "MultiInputPolicy"
    else:
        policy_spec = LoraSACPolicy if args.use_lora else "MultiInputPolicy"
    policy_kwargs: dict[str, Any] = {"net_arch": list(config.train.policy_hidden_sizes)}
    if algo_name == "sac" and bc_weights_path is not None:
        # train_bc.py saves a PPO-style actor that uses Tanh MLP activations.
        # Match the SAC actor hidden activations before loading BC weights so the
        # transferred policy stays behaviorally compatible instead of only shape-compatible.
        policy_kwargs["activation_fn"] = nn.Tanh
    if args.use_lora:
        policy_kwargs.update(
            {
                "lora_rank": int(args.lora_rank),
                "lora_alpha": float(args.lora_alpha),
                "lora_dropout": float(args.lora_dropout),
                "lora_target_modules": resolved_lora_target_modules,
                "lora_freeze_actor_base": bool(args.lora_freeze_actor_base),
                "lora_train_bias": bool(args.lora_train_bias),
            }
        )

    direct_load_resume_allowed = resume_path is not None and (
        algo_name == "sac" or (not args.use_lora and not episode_cycle_enabled)
    )
    if direct_load_resume_allowed:
        if algo_name == "sac" and torch.device(requested_device).type == "cuda":
            model = SAC.load(
                str(resume_path),
                env=env,
                device="cpu",
            )
            _move_sac_model_to_device(model, requested_device)
            print(f"Loaded SAC checkpoint on CPU and moved it to {requested_device}.")
        else:
            model = algorithm_class.load(
                str(resume_path),
                env=env,
                device=requested_device,
            )
        print(f"Loaded model with existing num_timesteps={int(model.num_timesteps)}")
        replay_buffer_path = load_replay_buffer_if_available(model, resume_path)
        if replay_buffer_path is not None:
            print(f"Loaded replay buffer from {replay_buffer_path}")
        elif algo_name == "sac":
            print("Replay buffer sidecar not found. SAC resume will continue with a fresh replay buffer.")
    else:
        if algo_name == "ppo":
            model_kwargs: dict[str, Any] = {
                "policy": policy_spec,
                "env": env,
                "learning_rate": config.train.learning_rate,
                "n_steps": config.train.n_steps,
                "batch_size": config.train.batch_size,
                "gamma": config.train.gamma,
                "gae_lambda": config.train.gae_lambda,
                "clip_range": config.train.clip_range,
                "ent_coef": config.train.ent_coef,
                "vf_coef": config.train.vf_coef,
                "max_grad_norm": config.train.max_grad_norm,
                "policy_kwargs": policy_kwargs,
                "verbose": 1,
                "seed": config.train.seed,
                "device": requested_device,
            }
            if episode_cycle_enabled:
                model_kwargs["rollout_episodes_per_update"] = int(rollout_episodes_per_update)
                model_kwargs["align_rollout_updates_to_episode_count"] = bool(args.align_rollout_updates_to_episode_count)
                model_kwargs["rollout_episode_initial_offset"] = int(existing_episode_count)
                model_kwargs["strict_episode_budget"] = bool(args.strict_rollout_step_budget)
                model_kwargs["post_update_save_dir"] = str(checkpoint_dir)
                model_kwargs["post_update_model_name"] = config.train.model_name
                model_kwargs["post_update_save_policy_weights"] = bool(config.train.save_policy_weights)
                model_kwargs["post_update_save_every_iterations"] = 1
                model_kwargs["post_update_initial_iteration"] = int(existing_cycle_update_index)
        else:
            model_kwargs = {
                "policy": policy_spec,
                "env": env,
                "learning_rate": config.train.learning_rate,
                "buffer_size": int(config.train.sac_buffer_size),
                "learning_starts": int(config.train.sac_learning_starts),
                "batch_size": config.train.batch_size,
                "tau": float(config.train.sac_tau),
                "gamma": config.train.gamma,
                "train_freq": (int(config.train.sac_train_freq_steps), str(config.train.sac_train_freq_unit)),
                "gradient_steps": int(config.train.sac_gradient_steps),
                "ent_coef": _parse_auto_or_float(config.train.sac_ent_coef),
                "target_update_interval": int(config.train.sac_target_update_interval),
                "target_entropy": _parse_auto_or_float(config.train.sac_target_entropy),
                "policy_kwargs": dict(policy_kwargs),
                "verbose": 1,
                "seed": config.train.seed,
                "device": requested_device,
                "optimize_memory_usage": bool(config.train.sac_optimize_memory_usage),
            }
        model = algorithm_class(**model_kwargs)

        if resume_path is not None or resume_policy_weights_path is not None:
            if resume_path is not None:
                load_candidates = [algorithm_class]
                if algo_name == "ppo" and algorithm_class is not PPO:
                    load_candidates.append(PPO)
                resume_policy_state, resume_num_timesteps = _load_resume_policy_snapshot(
                    resume_path,
                    device=requested_device,
                    load_candidates=load_candidates,
                )
            else:
                resume_policy_state, resume_num_timesteps = _load_resume_policy_weights_snapshot(
                    resume_policy_weights_path,
                )
            loaded_keys, skipped_keys = load_matching_policy_state_dict(model.policy, resume_policy_state)
            if not loaded_keys:
                resume_source = resume_path if resume_path is not None else resume_policy_weights_path
                raise RuntimeError(f"No policy tensors were loaded from resume checkpoint: {resume_source}")
            model.num_timesteps = int(resume_num_timesteps)
            print(f"Initialized {len(loaded_keys)} policy tensors from resume checkpoint.")
            if skipped_keys:
                print(f"Skipped {len(set(skipped_keys))} unmatched tensors while converting resume checkpoint.")
        elif bc_weights_path is not None:
            bc_payload = torch.load(bc_weights_path, map_location="cpu")
            actor_state_dict = bc_payload.get("actor_state_dict")
            if not isinstance(actor_state_dict, dict):
                raise KeyError(f"BC checkpoint does not contain 'actor_state_dict': {bc_weights_path}")
            if algo_name == "sac":
                loaded_keys, skipped_keys = load_bc_actor_state_dict_into_sac_policy(model.policy, actor_state_dict)
            else:
                loaded_keys = load_actor_state_dict(model.policy, actor_state_dict)
                skipped_keys = []
            if not loaded_keys:
                raise RuntimeError(f"No actor parameters were loaded from BC checkpoint: {bc_weights_path}")
            print(f"Loaded {len(loaded_keys)} BC actor parameter tensors into {algo_name.upper()}.")
            if skipped_keys:
                print(f"Skipped {len(set(skipped_keys))} BC tensors incompatible with the {algo_name.upper()} actor head.")
    checkpoint_callback = WeightCheckpointCallback(
        save_dir=checkpoint_dir,
        model_name=config.train.model_name,
        save_freq=config.train.checkpoint_interval_timesteps,
        save_policy_weights=config.train.save_policy_weights,
        save_replay_buffer=algo_name == "sac",
    )
    episode_metrics_callback = EpisodeMetricsCallback(
        csv_path=episode_metrics_path,
        run_id=run_id,
        scenario_path=scenario_path,
        initial_episode_index=existing_episode_count,
        append=True,
    )
    stop_after_episodes_callback = StopAfterEpisodesCallback(max_episodes=args.max_episodes)
    convergence_callback: ConvergenceStopCallback | None = None
    callback_list: list[BaseCallback] = [
        checkpoint_callback,
        episode_metrics_callback,
    ]
    if algo_name == "sac" and episode_cycle_enabled:
        callback_list.append(
            CycleCheckpointCallback(
                save_dir=checkpoint_dir,
                metrics_csv_path=checkpoint_metrics_path,
                model_name=config.train.model_name,
                run_id=run_id,
                save_policy_weights=config.train.save_policy_weights,
                save_replay_buffer=algo_name == "sac",
                episodes_per_cycle=int(rollout_episodes_per_update),
                initial_episode_count=existing_episode_count,
                initial_cycle_index=existing_cycle_update_index,
                align_to_episode_count=bool(args.align_rollout_updates_to_episode_count),
                initial_cycle_episode_rows=initial_cycle_episode_rows,
            )
        )
    if args.plot_reward:
        callback_list.append(
            EpisodeRewardPlotCallback(
                plot_path=reward_plot_path,
                moving_average_window=args.reward_plot_window,
            )
        )
    if config.train.video_interval_episodes > 0 and (config.train.save_episode_videos or not episode_cycle_enabled):
        callback_list.append(
            EpisodeArtifactCallback(
                video_dir=video_dir,
                checkpoint_dir=checkpoint_dir,
                model_name=config.train.model_name,
                save_every_episodes=config.train.video_interval_episodes,
                fps=config.train.video_fps,
                save_policy_weights=config.train.save_policy_weights,
                initial_completed_episodes=existing_episode_count,
                save_checkpoint_artifacts=not episode_cycle_enabled,
                save_replay_buffer=algo_name == "sac",
            )
        )
    if convergence_enabled:
        convergence_callback = ConvergenceStopCallback(
            window_episodes=int(args.convergence_window),
            min_episodes=convergence_min_episodes,
            min_success_rate=float(args.convergence_min_success_rate),
            max_timeout_rate=float(args.convergence_max_timeout_rate),
            max_failure_rate=float(args.convergence_max_failure_rate),
            reward_window=int(args.convergence_reward_window),
            reward_stability_ratio=float(args.convergence_reward_stability_ratio),
            initial_history=initial_convergence_history,
            initial_episode_count=existing_episode_count,
        )
        callback_list.append(convergence_callback)
    callback_list.append(stop_after_episodes_callback)
    callback = CallbackList(callback_list)

    latest_model_path: Path | None = None
    latest_weights_path: Path | None = None
    archived_model_path: Path | None = None
    archived_weights_path: Path | None = None
    stop_reason = "timesteps_reached"
    try:
        model.learn(
            total_timesteps=config.train.total_timesteps,
            callback=callback,
            reset_num_timesteps=resume_path is None and resume_policy_weights_path is None,
        )
    except KeyboardInterrupt:
        latest_model_path, latest_weights_path = save_training_artifacts(
            model=model,
            save_dir=log_dir,
            model_name=config.train.model_name,
            save_policy_weights=config.train.save_policy_weights,
            suffix="_interrupted",
            save_replay_buffer=algo_name == "sac",
        )
        archived_model_path, archived_weights_path = save_training_artifacts(
            model=model,
            save_dir=log_dir,
            model_name=config.train.model_name,
            save_policy_weights=config.train.save_policy_weights,
            suffix=f"_interrupted_{run_id}",
            save_replay_buffer=algo_name == "sac",
        )
        weights_message = (
            f", interrupted policy weights to {_relative_path_text(latest_weights_path)}"
            if latest_weights_path is not None
            else ""
        )
        stop_reason = "keyboard_interrupt"
        print(
            "Training interrupted. Saved interrupted model to "
            f"{_relative_path_text(latest_model_path)}{weights_message}"
        )
    else:
        latest_model_path, latest_weights_path = save_training_artifacts(
            model=model,
            save_dir=log_dir,
            model_name=config.train.model_name,
            save_policy_weights=config.train.save_policy_weights,
            save_replay_buffer=algo_name == "sac",
        )
        archived_model_path, archived_weights_path = save_training_artifacts(
            model=model,
            save_dir=log_dir,
            model_name=config.train.model_name,
            save_policy_weights=config.train.save_policy_weights,
            suffix=f"_{run_id}",
            save_replay_buffer=algo_name == "sac",
        )
        weights_message = (
            f", policy weights to {_relative_path_text(latest_weights_path)}"
            if latest_weights_path is not None
            else ""
        )
        if convergence_callback is not None and convergence_callback.converged:
            stop_reason = "convergence"
        elif stop_after_episodes_callback.stopped_due_to_limit:
            stop_reason = "max_episodes"
        print(f"Saved model to {_relative_path_text(latest_model_path)}{weights_message}")
    finally:
        env.close()

    training_summary = {
        "run_id": run_id,
        "algorithm": algo_name,
        "scenario_path": _relative_path_text(scenario_path),
        "scenario_cycle_paths": None if not scenario_cycle_paths else [_relative_path_text(path) for path in scenario_cycle_paths],
        "resume_from": _relative_path_text(resume_path),
        "resume_policy_weights": _relative_path_text(resume_policy_weights_path),
        "bc_weights": _relative_path_text(bc_weights_path),
        "device": requested_device,
        "log_dir": _relative_path_text(log_dir),
        "checkpoint_dir": _relative_path_text(checkpoint_dir),
        "video_dir": _relative_path_text(video_dir),
        "monitor_path": _relative_path_text(monitor_path),
        "episode_metrics_path": _relative_path_text(episode_metrics_path),
        "checkpoint_metrics_path": _relative_path_text(checkpoint_metrics_path),
        "existing_episode_count_at_start": int(existing_episode_count),
        "existing_cycle_update_index_at_start": int(existing_cycle_update_index),
        "episodes_completed_this_run": int(episode_metrics_callback.completed_episodes_this_run),
        "episodes_completed_total": int(existing_episode_count + episode_metrics_callback.completed_episodes_this_run),
        "num_timesteps": int(model.num_timesteps),
        "stop_reason": stop_reason,
        "lora_enabled": bool(args.use_lora),
        "rollout_episodes_per_update": int(rollout_episodes_per_update),
        "scenario_cycle_sample_size": int(scenario_cycle_sample_size),
        "align_rollout_updates_to_episode_count": bool(args.align_rollout_updates_to_episode_count),
        "rollout_step_budget": int(rollout_step_budget),
        "sac": {
            "buffer_size": int(config.train.sac_buffer_size),
            "learning_starts": int(config.train.sac_learning_starts),
            "train_freq_steps": int(config.train.sac_train_freq_steps),
            "train_freq_unit": str(config.train.sac_train_freq_unit),
            "gradient_steps": int(config.train.sac_gradient_steps),
            "tau": float(config.train.sac_tau),
            "ent_coef": _parse_auto_or_float(config.train.sac_ent_coef),
            "target_update_interval": int(config.train.sac_target_update_interval),
            "target_entropy": _parse_auto_or_float(config.train.sac_target_entropy),
            "optimize_memory_usage": bool(config.train.sac_optimize_memory_usage),
        },
        "converged": bool(convergence_callback.converged) if convergence_callback is not None else False,
        "convergence_summary": None if convergence_callback is None else convergence_callback.summary,
        "latest_window_summary": None if convergence_callback is None else convergence_callback.latest_window_summary,
        "latest_model_path": _relative_path_text(latest_model_path),
        "latest_policy_weights_path": _relative_path_text(latest_weights_path),
        "archived_model_path": _relative_path_text(archived_model_path),
        "archived_policy_weights_path": _relative_path_text(archived_weights_path),
    }
    _write_json(latest_summary_path, training_summary)
    _write_json(run_summary_path, training_summary)


if __name__ == "__main__":
    main()
