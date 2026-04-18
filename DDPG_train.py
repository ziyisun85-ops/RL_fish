from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

import train as train_module
from configs.default_config import config_to_dict, make_config
from envs import FishPathAvoidEnv
from train_launcher_config import (
    ALIGN_ROLLOUT_UPDATES_TO_EPISODE_COUNT,
    BC_WEIGHTS,
    DEVICE,
    EPISODES_PER_UPDATE,
    MAX_EPISODES,
    NUM_ENVS,
    SCENARIO_CYCLE_DIR,
    SCENARIO_CYCLE_SAMPLE_SIZE,
    USE_LORA,
    install_common_train_patches,
)
from utils.lora_td3_policy import (
    DEFAULT_TD3_LORA_TARGET_MODULES,
    LoraTD3Policy,
    resolve_td3_lora_target_modules,
)


DDPG_LOG_DIR = (Path("runs") / "ddpg_fish_baseline").as_posix()
DDPG_MODEL_NAME = "ddpg_fish_baseline"
DDPG_BUFFER_SIZE = 20_000
DDPG_LEARNING_STARTS = 5_000
DDPG_TRAIN_FREQ_UNIT = "episode"
DDPG_GRADIENT_STEPS = 1
DDPG_TAU = 0.005
DDPG_ACTION_NOISE_STD = 0.10
DDPG_SAVE_REPLAY_BUFFER = False


def build_default_args() -> list[str]:
    if SCENARIO_CYCLE_SAMPLE_SIZE > 0 and SCENARIO_CYCLE_SAMPLE_SIZE != EPISODES_PER_UPDATE:
        raise ValueError(
            "SCENARIO_CYCLE_SAMPLE_SIZE must equal EPISODES_PER_UPDATE so each sampled scene runs exactly one episode."
        )

    args = [
        "--device",
        DEVICE,
        "--num-envs",
        str(NUM_ENVS),
        "--max-episodes",
        str(MAX_EPISODES),
        "--scenario-cycle-dir",
        SCENARIO_CYCLE_DIR,
        "--rollout-episodes-per-update",
        str(EPISODES_PER_UPDATE),
        "--ddpg-buffer-size",
        str(DDPG_BUFFER_SIZE),
        "--ddpg-learning-starts",
        str(DDPG_LEARNING_STARTS),
        "--ddpg-train-freq-unit",
        DDPG_TRAIN_FREQ_UNIT,
        "--ddpg-gradient-steps",
        str(DDPG_GRADIENT_STEPS),
        "--ddpg-tau",
        str(DDPG_TAU),
        "--ddpg-action-noise-std",
        str(DDPG_ACTION_NOISE_STD),
        "--bc-weights",
        BC_WEIGHTS,
        "--log-dir",
        DDPG_LOG_DIR,
        "--model-name",
        DDPG_MODEL_NAME,
    ]
    if USE_LORA:
        args.append("--use-lora")
    if SCENARIO_CYCLE_SAMPLE_SIZE > 0:
        args.extend(
            [
                "--scenario-cycle-sample-size",
                str(SCENARIO_CYCLE_SAMPLE_SIZE),
            ]
        )
    if ALIGN_ROLLOUT_UPDATES_TO_EPISODE_COUNT:
        args.append("--align-rollout-updates-to-episode-count")
    return args


DEFAULT_ARGS = build_default_args()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DDPG for fish path following with local obstacle avoidance."
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Override total training timesteps.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel environments.")
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
        default=0,
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
        help="DDPG collects this many episodes before each episode-triggered update cycle.",
    )
    parser.add_argument(
        "--align-rollout-updates-to-episode-count",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Align the first post-resume cycle checkpoint to the next multiple of --rollout-episodes-per-update.",
    )
    parser.add_argument("--ddpg-buffer-size", type=int, default=None, help="Override DDPG replay buffer size.")
    parser.add_argument(
        "--ddpg-learning-starts",
        type=int,
        default=None,
        help="Override DDPG warmup steps before gradient updates start.",
    )
    parser.add_argument(
        "--ddpg-train-freq-steps",
        type=int,
        default=None,
        help="Override DDPG training trigger count. The unit is controlled by --ddpg-train-freq-unit.",
    )
    parser.add_argument(
        "--ddpg-train-freq-unit",
        type=str,
        choices=("step", "episode"),
        default=None,
        help="Unit for DDPG training triggers: environment steps or completed episodes.",
    )
    parser.add_argument(
        "--ddpg-gradient-steps",
        type=int,
        default=None,
        help="Override DDPG gradient steps per training trigger.",
    )
    parser.add_argument(
        "--ddpg-tau",
        type=float,
        default=None,
        help="Override DDPG target network smoothing coefficient.",
    )
    parser.add_argument(
        "--ddpg-action-noise-std",
        type=float,
        default=None,
        help="Exploration Gaussian noise std applied to DDPG actions during data collection.",
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
        help="Exact linear module names patched with LoRA. Defaults are tuned for TD3/DDPG actor layers.",
    )
    parser.add_argument(
        "--lora-freeze-actor-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze original actor weights and train only LoRA adapters plus critic parameters.",
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
    return parser.parse_args()


def _make_ddpg_action_noise(env, noise_std: float) -> NormalActionNoise:
    action_space = env.action_space
    action_shape = getattr(action_space, "shape", None)
    if not action_shape:
        raise ValueError("DDPG requires a continuous Box action space with a known shape.")
    mean = np.zeros(action_shape, dtype=np.float32)
    sigma = float(noise_std) * np.ones(action_shape, dtype=np.float32)
    return NormalActionNoise(mean=mean, sigma=sigma)


def load_bc_actor_state_dict_into_ddpg_policy(
    policy,
    actor_state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    named_parameters = dict(policy.named_parameters())

    key_mapping = {
        "action_net.weight": "actor.mu.4.weight",
        "action_net.bias": "actor.mu.4.bias",
        "mlp_extractor.policy_net.0.weight": "actor.mu.0.weight",
        "mlp_extractor.policy_net.0.bias": "actor.mu.0.bias",
        "mlp_extractor.policy_net.2.weight": "actor.mu.2.weight",
        "mlp_extractor.policy_net.2.bias": "actor.mu.2.bias",
    }

    for source_key, value in actor_state_dict.items():
        if source_key == "log_std":
            continue
        if source_key.startswith("features_extractor."):
            target_key = f"actor.{source_key}"
        else:
            target_key = key_mapping.get(source_key)

        if target_key is None:
            skipped_keys.append(source_key)
            continue

        parameter = named_parameters.get(target_key)
        if parameter is None or tuple(parameter.shape) != tuple(value.shape):
            skipped_keys.append(source_key)
            continue

        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        loaded_keys.append(target_key)

    policy.actor_target.load_state_dict(policy.actor.state_dict())
    loaded_keys.extend(["actor_target.*"])
    return loaded_keys, skipped_keys


def _install_runtime_patches(args: argparse.Namespace) -> None:
    install_common_train_patches(train_module, args)

    def quiet_cycle_checkpoint_on_step(self) -> bool:
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
                model_path, weights_path = train_module.save_training_artifacts(
                    model=self.model,
                    save_dir=self.save_dir,
                    model_name=self.model_name,
                    save_policy_weights=self.save_policy_weights,
                    suffix=cycle_suffix,
                    save_replay_buffer=self.save_replay_buffer,
                )
                self._write_cycle_metrics(
                    update_index=update_index,
                    model_path=model_path,
                    weights_path=weights_path,
                    cycle_rows=cycle_rows,
                )
                self._current_cycle_rows.clear()
                self.next_cycle_index += 1
                self._next_checkpoint_episode += self.episodes_per_cycle
        return True

    train_module.CycleCheckpointCallback._on_step = quiet_cycle_checkpoint_on_step


def main() -> None:
    args = parse_args()
    _install_runtime_patches(args)

    config = make_config()
    config.train.algorithm = "ddpg"
    if args.log_dir is not None:
        config.train.log_dir = Path(args.log_dir).as_posix()
    if args.model_name is not None:
        config.train.model_name = str(args.model_name)
    if args.timesteps is not None:
        config.train.total_timesteps = int(args.timesteps)
    if args.num_envs is not None:
        config.train.num_envs = int(args.num_envs)
    if args.xml_path is not None:
        config.env.model.xml_path = Path(args.xml_path).as_posix()

    ddpg_buffer_size = DDPG_BUFFER_SIZE if args.ddpg_buffer_size is None else int(args.ddpg_buffer_size)
    ddpg_learning_starts = (
        DDPG_LEARNING_STARTS if args.ddpg_learning_starts is None else int(args.ddpg_learning_starts)
    )
    ddpg_train_freq_steps = 1 if args.ddpg_train_freq_steps is None else int(args.ddpg_train_freq_steps)
    ddpg_train_freq_unit = (
        DDPG_TRAIN_FREQ_UNIT if args.ddpg_train_freq_unit is None else str(args.ddpg_train_freq_unit)
    )
    ddpg_gradient_steps = (
        DDPG_GRADIENT_STEPS if args.ddpg_gradient_steps is None else int(args.ddpg_gradient_steps)
    )
    ddpg_tau = DDPG_TAU if args.ddpg_tau is None else float(args.ddpg_tau)
    ddpg_action_noise_std = (
        DDPG_ACTION_NOISE_STD if args.ddpg_action_noise_std is None else float(args.ddpg_action_noise_std)
    )

    scenario_path = train_module.resolve_scenario_path(args)
    scenario_cycle_paths = train_module.resolve_scenario_cycle_paths(args)
    resume_path = train_module.resolve_resume_path(args)
    resume_policy_weights_path = train_module.resolve_resume_policy_weights_path(args)
    bc_weights_path = train_module.resolve_bc_weights_path(args)
    run_id = train_module._sanitize_run_id(args.run_id or train_module._default_run_id())

    if sum(path is not None for path in (resume_path, resume_policy_weights_path, bc_weights_path)) > 1:
        raise ValueError("Use only one of --resume-from, --resume-policy-weights, or --bc-weights.")
    if scenario_path is not None and scenario_cycle_paths:
        raise ValueError("Use either single-scenario mode or scenario-cycle mode, not both.")
    if args.render and not 0 <= args.render_env_index < config.train.num_envs:
        raise ValueError(
            f"--render-env-index must be within [0, {config.train.num_envs - 1}] when --render is enabled."
        )
    if args.render_slowdown < 0.0:
        raise ValueError("--render-slowdown must be non-negative.")
    if args.rollout_episodes_per_update < 0:
        raise ValueError("--rollout-episodes-per-update must be non-negative.")
    if args.scenario_cycle_sample_size < 0:
        raise ValueError("--scenario-cycle-sample-size must be non-negative.")
    if ddpg_buffer_size <= 0:
        raise ValueError("--ddpg-buffer-size must be positive.")
    if ddpg_learning_starts < 0:
        raise ValueError("--ddpg-learning-starts must be non-negative.")
    if ddpg_train_freq_steps <= 0:
        raise ValueError("--ddpg-train-freq-steps must be positive.")
    if ddpg_train_freq_unit not in {"step", "episode"}:
        raise ValueError("--ddpg-train-freq-unit must be either 'step' or 'episode'.")
    if ddpg_gradient_steps == 0:
        raise ValueError("--ddpg-gradient-steps cannot be 0.")
    if ddpg_tau <= 0.0 or ddpg_tau > 1.0:
        raise ValueError("--ddpg-tau must be within (0, 1].")
    if ddpg_action_noise_std < 0.0:
        raise ValueError("--ddpg-action-noise-std must be non-negative.")
    if args.use_lora and args.lora_rank <= 0:
        raise ValueError("--lora-rank must be positive when --use-lora is enabled.")

    resolved_lora_target_modules = resolve_td3_lora_target_modules(
        tuple(
            str(name) for name in (
                args.lora_target_modules if args.lora_target_modules is not None else DEFAULT_TD3_LORA_TARGET_MODULES
            )
        )
    )

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
    if episode_cycle_enabled and ddpg_train_freq_unit == "episode" and args.ddpg_train_freq_steps is None:
        ddpg_train_freq_steps = int(rollout_episodes_per_update)

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
    reward_plot_path = train_module._with_run_id(log_dir / Path("reward_curve.png"), run_id)
    run_config_path = train_module._with_run_id(log_dir / Path("config.json"), run_id)
    latest_summary_path = log_dir / "training_summary.json"
    run_summary_path = train_module._with_run_id(log_dir / Path("training_summary.json"), run_id)
    log_dir.mkdir(parents=True, exist_ok=True)

    existing_cycle_update_index = train_module._detect_latest_cycle_update_index(checkpoint_dir, config.train.model_name)
    resume_source_update_index = max(
        train_module._extract_cycle_update_index(resume_path),
        train_module._extract_cycle_update_index(resume_policy_weights_path),
    )
    if episode_cycle_enabled:
        existing_cycle_update_index = max(existing_cycle_update_index, resume_source_update_index)
    existing_episode_count = train_module.prepare_episode_metrics_csv(episode_metrics_path, scenario_path)
    initial_cycle_episode_rows: list[dict[str, Any]] = []
    if (
        episode_cycle_enabled
        and rollout_episodes_per_update > 0
        and bool(args.align_rollout_updates_to_episode_count)
    ):
        pending_cycle_episodes = existing_episode_count % rollout_episodes_per_update
        if pending_cycle_episodes > 0:
            initial_cycle_episode_rows = train_module.load_recent_cycle_episode_rows(
                episode_metrics_path,
                pending_cycle_episodes,
            )

    config_payload = config_to_dict(config)
    config_payload["run_id"] = run_id
    config_payload["monitor_path"] = train_module._relative_path_text(monitor_path)
    config_payload["episode_metrics_path"] = train_module._relative_path_text(episode_metrics_path)
    config_payload["checkpoint_metrics_path"] = train_module._relative_path_text(checkpoint_metrics_path)
    config_payload["reward_plot_path"] = train_module._relative_path_text(reward_plot_path)
    config_payload["existing_episode_count_at_start"] = int(existing_episode_count)
    config_payload["existing_cycle_update_index_at_start"] = int(existing_cycle_update_index)
    config_payload["lora"] = {
        "enabled": bool(args.use_lora),
        "rank": int(args.lora_rank) if args.use_lora else 0,
        "alpha": float(args.lora_alpha) if args.use_lora else 0.0,
        "dropout": float(args.lora_dropout) if args.use_lora else 0.0,
        "target_modules": list(resolved_lora_target_modules) if args.use_lora else [],
        "freeze_actor_base": bool(args.lora_freeze_actor_base) if args.use_lora else False,
        "train_bias": bool(args.lora_train_bias) if args.use_lora else False,
    }
    config_payload["algorithm"] = "ddpg"
    config_payload["ddpg"] = {
        "buffer_size": int(ddpg_buffer_size),
        "learning_starts": int(ddpg_learning_starts),
        "train_freq_steps": int(ddpg_train_freq_steps),
        "train_freq_unit": str(ddpg_train_freq_unit),
        "gradient_steps": int(ddpg_gradient_steps),
        "tau": float(ddpg_tau),
        "action_noise_std": float(ddpg_action_noise_std),
        "save_replay_buffer": bool(DDPG_SAVE_REPLAY_BUFFER),
    }
    config_payload["rollout_schedule"] = {
        "episode_cycle_enabled": episode_cycle_enabled,
        "episodes_per_update": int(rollout_episodes_per_update),
        "scenario_cycle_sample_size": int(scenario_cycle_sample_size),
        "align_updates_to_episode_count": bool(args.align_rollout_updates_to_episode_count),
        "step_budget": 0,
        "strict_step_budget": False,
    }
    config_payload["convergence"] = {
        "enabled": False,
        "window_episodes": 0,
        "min_episodes": 0,
        "min_success_rate": 0.0,
        "max_timeout_rate": 0.0,
        "max_failure_rate": 0.0,
        "reward_window": 0,
        "reward_stability_ratio": -1.0,
    }
    if scenario_path is not None:
        config_payload["selected_scenario_path"] = train_module._relative_path_text(scenario_path)
    if scenario_cycle_paths:
        config_payload["selected_scenario_cycle_paths"] = [
            train_module._relative_path_text(path) or ""
            for path in scenario_cycle_paths
        ]
    if resume_path is not None:
        config_payload["resume_from"] = train_module._relative_path_text(resume_path)
    if resume_policy_weights_path is not None:
        config_payload["resume_policy_weights"] = train_module._relative_path_text(resume_policy_weights_path)
    if bc_weights_path is not None:
        config_payload["bc_weights"] = train_module._relative_path_text(bc_weights_path)
    train_module._write_json(log_dir / "config.json", config_payload)
    train_module._write_json(run_config_path, config_payload)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "Requested CUDA training, but the current PyTorch build has no CUDA support. "
            "Falling back to CPU."
        )
        requested_device = "cpu"
    elif requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Algorithm: DDPG")
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
            f"{rollout_episodes_per_update} episodes/cycle"
        )
    if resume_path is not None:
        print(f"Resuming from model: {train_module._relative_path_text(resume_path)}")
    if resume_policy_weights_path is not None:
        print(f"Resuming from policy weights: {train_module._relative_path_text(resume_policy_weights_path)}")
    if bc_weights_path is not None:
        print(f"Initializing actor from BC weights: {train_module._relative_path_text(bc_weights_path)}")
    print(
        "DDPG train frequency: "
        f"{int(ddpg_train_freq_steps)} {str(ddpg_train_freq_unit)}(s) per update trigger"
    )
    print(f"DDPG exploration noise: std={ddpg_action_noise_std:.3f}, tau={ddpg_tau:.3f}")
    if args.use_lora:
        print(
            "LoRA fine-tuning enabled: "
            f"rank={int(args.lora_rank)}, alpha={float(args.lora_alpha):.1f}, "
            f"dropout={float(args.lora_dropout):.1f}"
        )
    if existing_episode_count > 0:
        print(f"Appending to existing episode metrics with starting episode index {existing_episode_count}.")
    if episode_cycle_enabled and existing_cycle_update_index > 0:
        print(f"Continuing cycle checkpoint numbering from update index {existing_cycle_update_index}.")

    def make_env(rank: int):
        def _factory() -> FishPathAvoidEnv:
            env = FishPathAvoidEnv(
                config=config.env,
                enable_mujoco_viewer=args.render and rank == args.render_env_index,
                realtime_playback=args.render and args.render_slowdown > 0.0,
                viewer_slowdown=args.render_slowdown,
                enable_episode_recording=False,
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

    action_noise = _make_ddpg_action_noise(env, ddpg_action_noise_std)
    policy_spec: str | type = LoraTD3Policy if args.use_lora else "MultiInputPolicy"
    policy_kwargs: dict[str, Any] = {"net_arch": list(config.train.policy_hidden_sizes)}
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

    direct_load_resume_allowed = resume_path is not None and not args.use_lora
    if direct_load_resume_allowed:
        model = DDPG.load(
            str(resume_path),
            env=env,
            device=requested_device,
        )
        model.action_noise = action_noise
        print(f"Loaded model with existing num_timesteps={int(model.num_timesteps)}")
        replay_buffer_path = train_module.load_replay_buffer_if_available(model, resume_path)
        if replay_buffer_path is not None:
            print(f"Loaded replay buffer from {replay_buffer_path}")
        else:
            print("Replay buffer sidecar not found. DDPG resume will continue with a fresh replay buffer.")
    else:
        model = DDPG(
            policy=policy_spec,
            env=env,
            learning_rate=config.train.learning_rate,
            buffer_size=int(ddpg_buffer_size),
            learning_starts=int(ddpg_learning_starts),
            batch_size=config.train.batch_size,
            tau=float(ddpg_tau),
            gamma=config.train.gamma,
            train_freq=(int(ddpg_train_freq_steps), str(ddpg_train_freq_unit)),
            gradient_steps=int(ddpg_gradient_steps),
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=config.train.seed,
            device=requested_device,
        )

        if resume_path is not None or resume_policy_weights_path is not None:
            if resume_path is not None:
                resume_policy_state, resume_num_timesteps = train_module._load_resume_policy_snapshot(
                    resume_path,
                    device=requested_device,
                    load_candidates=[DDPG],
                )
            else:
                resume_policy_state, resume_num_timesteps = train_module._load_resume_policy_weights_snapshot(
                    resume_policy_weights_path,
                )
            loaded_keys, skipped_keys = train_module.load_matching_policy_state_dict(model.policy, resume_policy_state)
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
            loaded_keys, skipped_keys = load_bc_actor_state_dict_into_ddpg_policy(model.policy, actor_state_dict)
            if not loaded_keys:
                raise RuntimeError(f"No actor parameters were loaded from BC checkpoint: {bc_weights_path}")
            print(f"Loaded {len(loaded_keys)} BC actor parameter tensors into DDPG.")
            if skipped_keys:
                print(f"Skipped {len(set(skipped_keys))} BC tensors incompatible with the DDPG actor head.")

    checkpoint_callback = train_module.WeightCheckpointCallback(
        save_dir=checkpoint_dir,
        model_name=config.train.model_name,
        save_freq=config.train.checkpoint_interval_timesteps,
        save_policy_weights=config.train.save_policy_weights,
        save_replay_buffer=bool(DDPG_SAVE_REPLAY_BUFFER),
    )
    episode_metrics_callback = train_module.EpisodeMetricsCallback(
        csv_path=episode_metrics_path,
        run_id=run_id,
        scenario_path=scenario_path,
        initial_episode_index=existing_episode_count,
        append=True,
    )
    stop_after_episodes_callback = train_module.StopAfterEpisodesCallback(max_episodes=args.max_episodes)
    callback_list: list[BaseCallback] = [
        checkpoint_callback,
        episode_metrics_callback,
    ]
    if episode_cycle_enabled:
        callback_list.append(
            train_module.CycleCheckpointCallback(
                save_dir=checkpoint_dir,
                metrics_csv_path=checkpoint_metrics_path,
                model_name=config.train.model_name,
                run_id=run_id,
                save_policy_weights=config.train.save_policy_weights,
                save_replay_buffer=bool(DDPG_SAVE_REPLAY_BUFFER),
                episodes_per_cycle=int(rollout_episodes_per_update),
                initial_episode_count=existing_episode_count,
                initial_cycle_index=existing_cycle_update_index,
                align_to_episode_count=bool(args.align_rollout_updates_to_episode_count),
                initial_cycle_episode_rows=initial_cycle_episode_rows,
            )
        )
    if args.plot_reward:
        callback_list.append(
            train_module.EpisodeRewardPlotCallback(
                plot_path=reward_plot_path,
                moving_average_window=args.reward_plot_window,
            )
        )
    callback_list.append(stop_after_episodes_callback)
    callback = train_module.CallbackList(callback_list)

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
        latest_model_path, latest_weights_path = train_module.save_training_artifacts(
            model=model,
            save_dir=log_dir,
            model_name=config.train.model_name,
            save_policy_weights=config.train.save_policy_weights,
            suffix="_interrupted",
            save_replay_buffer=bool(DDPG_SAVE_REPLAY_BUFFER),
        )
        archived_model_path, archived_weights_path = train_module.save_training_artifacts(
            model=model,
            save_dir=log_dir,
            model_name=config.train.model_name,
            save_policy_weights=config.train.save_policy_weights,
            suffix=f"_interrupted_{run_id}",
            save_replay_buffer=bool(DDPG_SAVE_REPLAY_BUFFER),
        )
        weights_message = (
            f", interrupted policy weights to {train_module._relative_path_text(latest_weights_path)}"
            if latest_weights_path is not None
            else ""
        )
        stop_reason = "keyboard_interrupt"
        print(
            "Training interrupted. Saved interrupted model to "
            f"{train_module._relative_path_text(latest_model_path)}{weights_message}"
        )
    else:
        latest_model_path, latest_weights_path = train_module.save_training_artifacts(
            model=model,
            save_dir=log_dir,
            model_name=config.train.model_name,
            save_policy_weights=config.train.save_policy_weights,
            save_replay_buffer=bool(DDPG_SAVE_REPLAY_BUFFER),
        )
        archived_model_path, archived_weights_path = train_module.save_training_artifacts(
            model=model,
            save_dir=log_dir,
            model_name=config.train.model_name,
            save_policy_weights=config.train.save_policy_weights,
            suffix=f"_{run_id}",
            save_replay_buffer=bool(DDPG_SAVE_REPLAY_BUFFER),
        )
        weights_message = (
            f", policy weights to {train_module._relative_path_text(latest_weights_path)}"
            if latest_weights_path is not None
            else ""
        )
        if stop_after_episodes_callback.stopped_due_to_limit:
            stop_reason = "max_episodes"
        print(f"Saved model to {train_module._relative_path_text(latest_model_path)}{weights_message}")
    finally:
        env.close()

    training_summary = {
        "run_id": run_id,
        "algorithm": "ddpg",
        "scenario_path": train_module._relative_path_text(scenario_path),
        "scenario_cycle_paths": None if not scenario_cycle_paths else [train_module._relative_path_text(path) for path in scenario_cycle_paths],
        "resume_from": train_module._relative_path_text(resume_path),
        "resume_policy_weights": train_module._relative_path_text(resume_policy_weights_path),
        "bc_weights": train_module._relative_path_text(bc_weights_path),
        "device": requested_device,
        "log_dir": train_module._relative_path_text(log_dir),
        "checkpoint_dir": train_module._relative_path_text(checkpoint_dir),
        "video_dir": train_module._relative_path_text(video_dir),
        "monitor_path": train_module._relative_path_text(monitor_path),
        "episode_metrics_path": train_module._relative_path_text(episode_metrics_path),
        "checkpoint_metrics_path": train_module._relative_path_text(checkpoint_metrics_path),
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
        "rollout_step_budget": 0,
        "lora": {
            "enabled": bool(args.use_lora),
            "rank": int(args.lora_rank) if args.use_lora else 0,
            "alpha": float(args.lora_alpha) if args.use_lora else 0.0,
            "dropout": float(args.lora_dropout) if args.use_lora else 0.0,
            "target_modules": list(resolved_lora_target_modules) if args.use_lora else [],
            "freeze_actor_base": bool(args.lora_freeze_actor_base) if args.use_lora else False,
            "train_bias": bool(args.lora_train_bias) if args.use_lora else False,
        },
        "ddpg": {
            "buffer_size": int(ddpg_buffer_size),
            "learning_starts": int(ddpg_learning_starts),
            "train_freq_steps": int(ddpg_train_freq_steps),
            "train_freq_unit": str(ddpg_train_freq_unit),
            "gradient_steps": int(ddpg_gradient_steps),
            "tau": float(ddpg_tau),
            "action_noise_std": float(ddpg_action_noise_std),
            "save_replay_buffer": bool(DDPG_SAVE_REPLAY_BUFFER),
        },
        "converged": False,
        "convergence_summary": None,
        "latest_window_summary": None,
        "latest_model_path": train_module._relative_path_text(latest_model_path),
        "latest_policy_weights_path": train_module._relative_path_text(latest_weights_path),
        "archived_model_path": train_module._relative_path_text(archived_model_path),
        "archived_policy_weights_path": train_module._relative_path_text(archived_weights_path),
    }
    train_module._write_json(latest_summary_path, training_summary)
    train_module._write_json(run_summary_path, training_summary)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    main()
