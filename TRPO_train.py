from __future__ import annotations

import argparse
import copy
import csv
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import BaseBuffer, DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage
from torch import nn
from torch.nn import functional as F

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
from utils.lora_policy import DEFAULT_LORA_TARGET_MODULES, LoraMultiInputPolicy
from utils.policy_utils import is_actor_parameter_name, load_actor_state_dict


TRPO_LOG_DIR = (Path("runs") / "trpo_lora_fish_baseline").as_posix()
TRPO_MODEL_NAME = "trpo_lora_fish_baseline"
TRPO_DEFAULT_TOTAL_TIMESTEPS = 1_000_000_000_000
TRPO_CG_MAX_STEPS = 15
TRPO_CG_DAMPING = 0.1
TRPO_LINE_SEARCH_SHRINKING_FACTOR = 0.8
TRPO_LINE_SEARCH_MAX_ITER = 10
TRPO_N_CRITIC_UPDATES = 10
TRPO_TARGET_KL = 0.01
TRPO_SUB_SAMPLING_FACTOR = 1
TRPO_NORMALIZE_ADVANTAGE = True
TRPO_STATS_WINDOW_SIZE = 100


SelfTRPO = TypeVar("SelfTRPO", bound="TRPO")


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
        "--bc-weights",
        BC_WEIGHTS,
        "--log-dir",
        TRPO_LOG_DIR,
        "--model-name",
        TRPO_MODEL_NAME,
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
        description="Train TRPO for fish path following with local obstacle avoidance."
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
        default="auto",
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
        help="TRPO collects this many episodes before each update cycle.",
    )
    parser.add_argument(
        "--rollout-step-budget",
        type=int,
        default=0,
        help="Maximum rollout buffer steps reserved for one TRPO cycle update.",
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
        help="Align the first post-resume cycle checkpoint to the next multiple of --rollout-episodes-per-update.",
    )
    parser.add_argument(
        "--trpo-cg-max-steps",
        type=int,
        default=None,
        help="Override the maximum conjugate-gradient iterations used by TRPO.",
    )
    parser.add_argument(
        "--trpo-cg-damping",
        type=float,
        default=None,
        help="Override the damping term used in the TRPO Hessian-vector product.",
    )
    parser.add_argument(
        "--trpo-line-search-shrinking-factor",
        type=float,
        default=None,
        help="Override the multiplicative backtracking factor used by TRPO line search.",
    )
    parser.add_argument(
        "--trpo-line-search-max-iter",
        type=int,
        default=None,
        help="Override the maximum TRPO line-search backtracking iterations.",
    )
    parser.add_argument(
        "--trpo-n-critic-updates",
        type=int,
        default=None,
        help="Override the number of critic mini-batch updates per TRPO policy update.",
    )
    parser.add_argument(
        "--trpo-target-kl",
        type=float,
        default=None,
        help="Override the target KL divergence constraint used by TRPO.",
    )
    parser.add_argument(
        "--trpo-sub-sampling-factor",
        type=int,
        default=None,
        help="Override the rollout sub-sampling factor used during TRPO policy updates.",
    )
    parser.add_argument(
        "--trpo-normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable advantage normalization in TRPO.",
    )
    parser.add_argument(
        "--trpo-stats-window-size",
        type=int,
        default=None,
        help="Override the rolling stats window size used by TRPO logging.",
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
        help="Exact linear module names patched with LoRA. Defaults match the PPO/TRPO actor layers.",
    )
    parser.add_argument(
        "--lora-freeze-actor-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze original actor weights and train only LoRA adapters.",
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


def _safe_mean(values: list[float], default: float = float("nan")) -> float:
    if not values:
        return float(default)
    return float(np.mean(values))


def _actor_parameter_name(name: str) -> bool:
    return is_actor_parameter_name(name) or name.startswith("pi_features_extractor.")


def conjugate_gradient_solver(
    matrix_vector_dot_fn,
    b: torch.Tensor,
    max_iter: int = 10,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    x = 1e-4 * torch.randn_like(b)
    residual = b - matrix_vector_dot_fn(x)
    residual_squared_norm = torch.matmul(residual, residual)

    if residual_squared_norm < residual_tol:
        return x

    p = residual.clone()

    for iteration in range(int(max_iter)):
        a_dot_p = matrix_vector_dot_fn(p)
        alpha = residual_squared_norm / p.dot(a_dot_p)
        x += alpha * p

        if iteration == int(max_iter) - 1:
            return x

        residual -= alpha * a_dot_p
        new_residual_squared_norm = torch.matmul(residual, residual)
        if new_residual_squared_norm < residual_tol:
            return x

        beta = new_residual_squared_norm / residual_squared_norm
        residual_squared_norm = new_residual_squared_norm
        p = residual + beta * p

    return x


def flat_grad(
    output: torch.Tensor,
    parameters,
    *,
    create_graph: bool = False,
    retain_graph: bool = False,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        output,
        parameters,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return torch.cat([torch.ravel(grad) for grad in grads if grad is not None])


class DynamicResizableDictRolloutBuffer(DictRolloutBuffer):
    """List-backed rollout buffer to avoid preallocating huge arrays for 100-scene TRPO cycles."""

    def __init__(self, *args, **kwargs) -> None:
        if "buffer_size" in kwargs:
            requested_buffer_size = int(kwargs["buffer_size"])
            kwargs = dict(kwargs)
            kwargs["buffer_size"] = 1
            self.max_buffer_size = requested_buffer_size
            super().__init__(*args, **kwargs)
            return

        if not args:
            raise ValueError("DynamicResizableDictRolloutBuffer requires buffer_size.")
        requested_buffer_size = int(args[0])
        self.max_buffer_size = requested_buffer_size
        super().__init__(1, *args[1:], **kwargs)

    def reset(self) -> None:
        self._observations_list = {key: [] for key in self.obs_shape.keys()}
        self._actions_list: list[np.ndarray] = []
        self._rewards_list: list[np.ndarray] = []
        self._returns_list: list[np.ndarray] = []
        self._episode_starts_list: list[np.ndarray] = []
        self._values_list: list[np.ndarray] = []
        self._log_probs_list: list[np.ndarray] = []
        self._advantages_list: list[np.ndarray] = []
        self.generator_ready = False
        BaseBuffer.reset(self)
        self.buffer_size = int(self.max_buffer_size)

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value,
        log_prob,
    ) -> None:
        if self.pos >= self.max_buffer_size:
            raise RuntimeError(
                "DynamicResizableDictRolloutBuffer exceeded the reserved rollout step budget. "
                "Increase --rollout-step-budget or reduce --rollout-episodes-per-update."
            )

        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        for key in self.obs_shape.keys():
            obs_array = np.array(obs[key], copy=True)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_array = obs_array.reshape((self.n_envs,) + self.obs_shape[key])
            self._observations_list[key].append(obs_array)

        action_array = np.array(action.reshape((self.n_envs, self.action_dim)), copy=True)
        self._actions_list.append(action_array)
        self._rewards_list.append(np.array(reward, dtype=np.float32, copy=True))
        self._episode_starts_list.append(np.array(episode_start, dtype=np.float32, copy=True))
        self._values_list.append(value.clone().cpu().numpy().flatten().astype(np.float32, copy=False))
        self._log_probs_list.append(log_prob.clone().cpu().numpy().astype(np.float32, copy=False))
        self.pos += 1
        if self.pos == self.max_buffer_size:
            self.full = True

    def finalize(self, actual_size: int) -> None:
        actual_size = int(actual_size)
        if actual_size <= 0:
            raise ValueError("actual_size must be positive.")
        if actual_size > self.pos:
            raise ValueError(f"actual_size {actual_size} exceeds collected rollout steps {self.pos}.")

        self.observations = {}
        for key, obs_space in self.observation_space.spaces.items():
            self.observations[key] = np.stack(
                self._observations_list[key][:actual_size],
                axis=0,
            ).astype(obs_space.dtype, copy=False)

        self.actions = np.stack(self._actions_list[:actual_size], axis=0).astype(self.action_space.dtype, copy=False)
        self.rewards = np.stack(self._rewards_list[:actual_size], axis=0).astype(np.float32, copy=False)
        self.returns = np.zeros((actual_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.stack(self._episode_starts_list[:actual_size], axis=0).astype(np.float32, copy=False)
        self.values = np.stack(self._values_list[:actual_size], axis=0).astype(np.float32, copy=False)
        self.log_probs = np.stack(self._log_probs_list[:actual_size], axis=0).astype(np.float32, copy=False)
        self.advantages = np.zeros((actual_size, self.n_envs), dtype=np.float32)

        self.buffer_size = actual_size
        self.pos = actual_size
        self.full = True
        self.generator_ready = False

        self._observations_list = {}
        self._actions_list = []
        self._rewards_list = []
        self._returns_list = []
        self._episode_starts_list = []
        self._values_list = []
        self._log_probs_list = []
        self._advantages_list = []


class TRPOCycleMetricsCallback(BaseCallback):
    def __init__(
        self,
        *,
        save_dir: Path,
        metrics_csv_path: Path,
        model_name: str,
        run_id: str,
        save_policy_weights: bool,
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
        self._pending_cycle_metrics: dict[int, dict[str, Any]] = {}

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
        global _ACTIVE_TRPO_CYCLE_METRICS_CALLBACK
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_TRPO_CYCLE_METRICS_CALLBACK = self

        write_header = True
        mode = "w"
        if self.metrics_csv_path.exists():
            mode = "a"
            write_header = self.metrics_csv_path.stat().st_size <= 0

        self._metrics_file = self.metrics_csv_path.open(mode, newline="", encoding="utf-8")
        self._metrics_writer = csv.DictWriter(
            self._metrics_file,
            fieldnames=train_module.CHECKPOINT_METRICS_FIELDNAMES,
        )
        if write_header:
            self._metrics_writer.writeheader()
            self._metrics_file.flush()

    def _on_training_end(self) -> None:
        global _ACTIVE_TRPO_CYCLE_METRICS_CALLBACK
        if _ACTIVE_TRPO_CYCLE_METRICS_CALLBACK is self:
            _ACTIVE_TRPO_CYCLE_METRICS_CALLBACK = None
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
            self._metrics_writer = None

    def _queue_cycle_metrics(
        self,
        *,
        update_index: int,
        cycle_rows: list[dict[str, Any]],
    ) -> None:
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
        self._pending_cycle_metrics[int(update_index)] = {
            "num_timesteps": int(self.num_timesteps),
            "episodes_in_cycle": int(episodes_in_cycle),
            "episodes_completed_total": int(self.completed_episodes),
            "success_count": int(success_count),
            "success_rate": float(success_rate),
            "mean_episode_reward": float(mean_episode_reward),
            "mean_episode_length": float(mean_episode_length),
            "cycle_train_time_sec": float(cycle_train_time_sec),
        }

    def record_saved_cycle_checkpoint(
        self,
        *,
        update_index: int,
        model_path: Path,
        weights_path: Path | None,
    ) -> None:
        metrics = self._pending_cycle_metrics.pop(int(update_index), None)
        if metrics is None:
            return
        if self._metrics_writer is None or self._metrics_file is None:
            return
        self._metrics_writer.writerow(
            {
                "run_id": self.run_id,
                "update_index": int(update_index),
                "num_timesteps": int(metrics["num_timesteps"]),
                "episodes_in_cycle": int(metrics["episodes_in_cycle"]),
                "episodes_completed_total": int(metrics["episodes_completed_total"]),
                "success_count": int(metrics["success_count"]),
                "success_rate": float(metrics["success_rate"]),
                "mean_episode_reward": float(metrics["mean_episode_reward"]),
                "mean_episode_length": float(metrics["mean_episode_length"]),
                "cycle_train_time_sec": float(metrics["cycle_train_time_sec"]),
                "model_path": train_module._relative_path_text(model_path) or "",
                "policy_weights_path": train_module._relative_path_text(weights_path) or "",
            }
        )
        self._metrics_file.flush()

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
                update_index = self.next_cycle_index
                self._queue_cycle_metrics(
                    update_index=update_index,
                    cycle_rows=list(self._current_cycle_rows),
                )
                self._current_cycle_rows.clear()
                self.next_cycle_index += 1
                self._next_checkpoint_episode += self.episodes_per_cycle
        return True


class TRPO(OnPolicyAlgorithm):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 128,
        gamma: float = 0.99,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 10,
        n_critic_updates: int = 10,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        normalize_advantage: bool = True,
        target_kl: float = 0.01,
        sub_sampling_factor: int = 1,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: torch.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=0.0,
            vf_coef=0.0,
            max_grad_norm=0.0,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = bool(normalize_advantage)
        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            if self.normalize_advantage:
                assert buffer_size > 1, (
                    "`n_steps * n_envs` must be greater than 1. "
                    f"Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
                )
            untruncated_batches = buffer_size // int(batch_size)
            if int(batch_size) > 0 and buffer_size % int(batch_size) > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the rollout buffer is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % int(batch_size)}.\n"
                    f"We recommend using a batch_size that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        self.batch_size = int(batch_size)
        self.cg_max_steps = int(cg_max_steps)
        self.cg_damping = float(cg_damping)
        self.line_search_shrinking_factor = float(line_search_shrinking_factor)
        self.line_search_max_iter = int(line_search_max_iter)
        self.target_kl = float(target_kl)
        self.n_critic_updates = int(n_critic_updates)
        self.sub_sampling_factor = int(sub_sampling_factor)

        if _init_setup_model:
            self._setup_model()

    def _compute_actor_grad(
        self,
        kl_div: torch.Tensor,
        policy_objective: torch.Tensor,
    ) -> tuple[list[nn.Parameter], torch.Tensor, torch.Tensor, list[tuple[int, ...]]]:
        policy_objective_gradients_list = []
        grad_kl_list = []
        grad_shape: list[tuple[int, ...]] = []
        actor_params: list[nn.Parameter] = []

        for name, param in self.policy.named_parameters():
            if not param.requires_grad or not _actor_parameter_name(name):
                continue

            kl_param_grad, *_ = torch.autograd.grad(
                kl_div,
                param,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
                only_inputs=True,
            )
            if kl_param_grad is None:
                continue

            policy_objective_grad, *_ = torch.autograd.grad(
                policy_objective,
                param,
                retain_graph=True,
                allow_unused=True,
                only_inputs=True,
            )
            if policy_objective_grad is None:
                continue

            grad_shape.append(tuple(kl_param_grad.shape))
            grad_kl_list.append(kl_param_grad.reshape(-1))
            policy_objective_gradients_list.append(policy_objective_grad.reshape(-1))
            actor_params.append(param)

        if not actor_params:
            raise RuntimeError("TRPO actor parameter set is empty. Check the policy architecture and LoRA settings.")

        policy_objective_gradients = torch.cat(policy_objective_gradients_list)
        grad_kl = torch.cat(grad_kl_list)
        return actor_params, policy_objective_gradients, grad_kl, grad_shape

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        policy_objective_values: list[float] = []
        kl_divergences: list[float] = []
        line_search_results: list[float] = []
        value_losses: list[float] = []
        actor_params: list[nn.Parameter] = []

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            observations = rollout_data.observations
            actions = rollout_data.actions
            old_log_prob = rollout_data.old_log_prob
            advantages = rollout_data.advantages

            if self.sub_sampling_factor > 1:
                if isinstance(observations, dict):
                    observations = {
                        key: value[:: self.sub_sampling_factor]
                        for key, value in observations.items()
                    }
                else:
                    observations = observations[:: self.sub_sampling_factor]
                actions = actions[:: self.sub_sampling_factor]
                old_log_prob = old_log_prob[:: self.sub_sampling_factor]
                advantages = advantages[:: self.sub_sampling_factor]

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            with torch.no_grad():
                old_distribution = copy.copy(self.policy.get_distribution(observations))

            distribution = self.policy.get_distribution(observations)
            log_prob = distribution.log_prob(actions)

            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(log_prob - old_log_prob)
            policy_objective = (advantages * ratio).mean()
            kl_div = kl_divergence(distribution, old_distribution).mean()

            self.policy.optimizer.zero_grad()
            actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(
                kl_div,
                policy_objective,
            )

            hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)
            search_direction = conjugate_gradient_solver(
                hessian_vector_product_fn,
                policy_objective_gradients,
                max_iter=self.cg_max_steps,
            )

            hvp_search_direction = hessian_vector_product_fn(search_direction, retain_graph=False)
            denominator = float(torch.matmul(search_direction, hvp_search_direction).detach().cpu().item())
            if not np.isfinite(denominator) or denominator <= 0.0:
                line_search_results.append(0.0)
                policy_objective_values.append(float(policy_objective.item()))
                kl_divergences.append(0.0)
                continue

            max_step_size = np.sqrt((2.0 * self.target_kl) / denominator)
            if not np.isfinite(max_step_size) or max_step_size <= 0.0:
                line_search_results.append(0.0)
                policy_objective_values.append(float(policy_objective.item()))
                kl_divergences.append(0.0)
                continue

            line_search_backtrack_coeff = 1.0
            original_actor_params = [param.detach().clone() for param in actor_params]

            is_line_search_success = False
            with torch.no_grad():
                for _ in range(self.line_search_max_iter):
                    start_idx = 0
                    for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape, strict=True):
                        n_params = param.numel()
                        param.data = (
                            original_param.data
                            + line_search_backtrack_coeff
                            * max_step_size
                            * search_direction[start_idx : start_idx + n_params].view(shape)
                        )
                        start_idx += n_params

                    distribution = self.policy.get_distribution(observations)
                    log_prob = distribution.log_prob(actions)
                    ratio = torch.exp(log_prob - old_log_prob)
                    new_policy_objective = (advantages * ratio).mean()
                    new_kl_div = kl_divergence(distribution, old_distribution).mean()

                    if new_kl_div < self.target_kl and new_policy_objective > policy_objective:
                        is_line_search_success = True
                        break

                    line_search_backtrack_coeff *= self.line_search_shrinking_factor

                line_search_results.append(1.0 if is_line_search_success else 0.0)
                if not is_line_search_success:
                    for param, original_param in zip(actor_params, original_actor_params, strict=True):
                        param.data = original_param.data.clone()
                    policy_objective_values.append(float(policy_objective.item()))
                    kl_divergences.append(0.0)
                else:
                    policy_objective_values.append(float(new_policy_objective.item()))
                    kl_divergences.append(float(new_kl_div.item()))

        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values_pred = self.policy.predict_values(rollout_data.observations)
                value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
                value_losses.append(float(value_loss.item()))

                self.policy.optimizer.zero_grad()
                value_loss.backward()
                for param in actor_params:
                    param.grad = None
                self.policy.optimizer.step()

        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        policy_objective_mean = _safe_mean(policy_objective_values)
        value_loss_mean = _safe_mean(value_losses)
        kl_div_mean = _safe_mean(kl_divergences, default=0.0)
        train_loss = value_loss_mean - policy_objective_mean + kl_div_mean

        self.logger.record("train/policy_objective", policy_objective_mean)
        self.logger.record("train/value_loss", value_loss_mean)
        self.logger.record("train/kl_divergence_loss", kl_div_mean)
        self.logger.record("train/is_line_search_success", _safe_mean(line_search_results, default=0.0))
        self.logger.record("train/loss", train_loss)
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def hessian_vector_product(
        self,
        params: list[nn.Parameter],
        grad_kl: torch.Tensor,
        vector: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        jacobian_vector_product = (grad_kl * vector).sum()
        return flat_grad(jacobian_vector_product, params, retain_graph=retain_graph) + self.cg_damping * vector

    def learn(
        self: SelfTRPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TRPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTRPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


class EpisodeCycleTRPO(TRPO):
    def __init__(
        self,
        *args,
        rollout_episodes_per_update: int = 0,
        align_rollout_updates_to_episode_count: bool = False,
        rollout_episode_initial_offset: int = 0,
        strict_episode_budget: bool = True,
        post_update_save_dir: str | None = None,
        post_update_model_name: str = "trpo_fish_baseline",
        post_update_save_policy_weights: bool = True,
        post_update_save_every_iterations: int = 0,
        post_update_initial_iteration: int = 0,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("rollout_buffer_class") is None:
            kwargs["rollout_buffer_class"] = DynamicResizableDictRolloutBuffer
        super().__init__(*args, **kwargs)
        self.rollout_episodes_per_update = max(0, int(rollout_episodes_per_update))
        self.align_rollout_updates_to_episode_count = bool(align_rollout_updates_to_episode_count)
        self.rollout_episode_initial_offset = max(0, int(rollout_episode_initial_offset))
        self.rollout_completed_episodes = int(self.rollout_episode_initial_offset)
        self.strict_episode_budget = bool(strict_episode_budget)
        self.post_update_save_dir = None if post_update_save_dir is None else str(Path(post_update_save_dir).resolve())
        self.post_update_model_name = str(post_update_model_name)
        self.post_update_save_policy_weights = bool(post_update_save_policy_weights)
        self.post_update_save_every_iterations = max(0, int(post_update_save_every_iterations))
        self.post_update_initial_iteration = max(0, int(post_update_initial_iteration))
        self._stop_after_current_rollout = False

    def _target_rollout_episode_count(self) -> int:
        if self.rollout_episodes_per_update <= 0:
            return 0
        if not self.align_rollout_updates_to_episode_count:
            return int(self.rollout_episodes_per_update)

        remainder = int(self.rollout_completed_episodes) % int(self.rollout_episodes_per_update)
        if remainder == 0:
            return int(self.rollout_episodes_per_update)
        return int(self.rollout_episodes_per_update) - remainder

    def collect_rollouts(
        self,
        env,
        callback: BaseCallback,
        rollout_buffer: DictRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        if self.rollout_episodes_per_update <= 0:
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        target_rollout_episodes = self._target_rollout_episode_count()
        n_steps = 0
        completed_episodes = 0
        self._stop_after_current_rollout = False
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps and completed_episodes < target_rollout_episodes:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                actions, values, log_probs = self.policy(obs_tensor)
            actions_np = actions.cpu().numpy()

            clipped_actions = actions_np
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions_np, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            step_completed_episodes = int(np.count_nonzero(dones))

            callback.update_locals(locals())
            callback_requested_stop = not callback.on_step()

            self._update_info_buffer(infos, dones)
            n_steps += 1
            completed_episodes += step_completed_episodes

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions_np,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

            if callback_requested_stop:
                if completed_episodes >= target_rollout_episodes:
                    self._stop_after_current_rollout = True
                    break
                return False

        if completed_episodes < target_rollout_episodes and self.strict_episode_budget:
            raise RuntimeError(
                "Episode-cycle rollout hit the step budget before collecting the requested number of episodes. "
                "Increase --rollout-step-budget or reduce --rollout-episodes-per-update."
            )

        if isinstance(rollout_buffer, DynamicResizableDictRolloutBuffer):
            rollout_buffer.finalize(n_steps)
        else:
            rollout_buffer.full = True

        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        self.rollout_completed_episodes += int(completed_episodes)
        return True

    def _save_post_update_checkpoint(self, iteration: int) -> None:
        global _ACTIVE_TRPO_CYCLE_METRICS_CALLBACK
        if self.post_update_save_every_iterations <= 0 or self.post_update_save_dir is None:
            return
        if int(iteration) % int(self.post_update_save_every_iterations) != 0:
            return

        checkpoint_index = self.post_update_initial_iteration + int(iteration)
        suffix = f"_update_{checkpoint_index:06d}"
        model_path, weights_path = train_module.save_training_artifacts(
            model=self,
            save_dir=Path(self.post_update_save_dir),
            model_name=self.post_update_model_name,
            save_policy_weights=self.post_update_save_policy_weights,
            suffix=suffix,
            save_replay_buffer=False,
        )
        cycle_metrics_callback = _ACTIVE_TRPO_CYCLE_METRICS_CALLBACK
        if cycle_metrics_callback is not None:
            cycle_metrics_callback.record_saved_cycle_checkpoint(
                update_index=int(checkpoint_index),
                model_path=model_path,
                weights_path=weights_path,
            )
        weights_message = (
            f", policy weights to {train_module._relative_path_text(weights_path)}"
            if weights_path is not None
            else ""
        )
        print(
            "Saved cycle checkpoint to "
            f"{train_module._relative_path_text(model_path)}{weights_message}"
        )

    def learn(
        self,
        total_timesteps: int,
        callback: BaseCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TRPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env,
                callback,
                self.rollout_buffer,
                n_rollout_steps=self.n_steps,
            )

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.dump_logs(iteration)

            self.train()
            self._save_post_update_checkpoint(iteration)
            if self._stop_after_current_rollout:
                self._stop_after_current_rollout = False
                break

        callback.on_training_end()
        return self


def _install_runtime_patches(args: argparse.Namespace) -> None:
    install_common_train_patches(train_module, args)


_ACTIVE_TRPO_CYCLE_METRICS_CALLBACK: "TRPOCycleMetricsCallback | None" = None


def main() -> None:
    args = parse_args()
    _install_runtime_patches(args)

    config = make_config()
    config.train.algorithm = "trpo"
    config.train.total_timesteps = int(TRPO_DEFAULT_TOTAL_TIMESTEPS)
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
    if args.record_videos is not None:
        config.train.save_episode_videos = bool(args.record_videos)
    if args.video_interval_episodes is not None:
        config.train.video_interval_episodes = int(args.video_interval_episodes)

    trpo_cg_max_steps = TRPO_CG_MAX_STEPS if args.trpo_cg_max_steps is None else int(args.trpo_cg_max_steps)
    trpo_cg_damping = TRPO_CG_DAMPING if args.trpo_cg_damping is None else float(args.trpo_cg_damping)
    trpo_line_search_shrinking_factor = (
        TRPO_LINE_SEARCH_SHRINKING_FACTOR
        if args.trpo_line_search_shrinking_factor is None
        else float(args.trpo_line_search_shrinking_factor)
    )
    trpo_line_search_max_iter = (
        TRPO_LINE_SEARCH_MAX_ITER
        if args.trpo_line_search_max_iter is None
        else int(args.trpo_line_search_max_iter)
    )
    trpo_n_critic_updates = (
        TRPO_N_CRITIC_UPDATES if args.trpo_n_critic_updates is None else int(args.trpo_n_critic_updates)
    )
    trpo_target_kl = TRPO_TARGET_KL if args.trpo_target_kl is None else float(args.trpo_target_kl)
    trpo_sub_sampling_factor = (
        TRPO_SUB_SAMPLING_FACTOR
        if args.trpo_sub_sampling_factor is None
        else int(args.trpo_sub_sampling_factor)
    )
    trpo_normalize_advantage = (
        TRPO_NORMALIZE_ADVANTAGE
        if args.trpo_normalize_advantage is None
        else bool(args.trpo_normalize_advantage)
    )
    trpo_stats_window_size = (
        TRPO_STATS_WINDOW_SIZE
        if args.trpo_stats_window_size is None
        else int(args.trpo_stats_window_size)
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
    if args.rollout_step_budget < 0:
        raise ValueError("--rollout-step-budget must be non-negative.")
    if args.scenario_cycle_sample_size < 0:
        raise ValueError("--scenario-cycle-sample-size must be non-negative.")
    if args.use_lora and args.lora_rank <= 0:
        raise ValueError("--lora-rank must be positive when --use-lora is enabled.")
    if trpo_cg_max_steps <= 0:
        raise ValueError("--trpo-cg-max-steps must be positive.")
    if trpo_cg_damping < 0.0:
        raise ValueError("--trpo-cg-damping must be non-negative.")
    if not 0.0 < trpo_line_search_shrinking_factor < 1.0:
        raise ValueError("--trpo-line-search-shrinking-factor must be within (0, 1).")
    if trpo_line_search_max_iter <= 0:
        raise ValueError("--trpo-line-search-max-iter must be positive.")
    if trpo_n_critic_updates <= 0:
        raise ValueError("--trpo-n-critic-updates must be positive.")
    if trpo_target_kl <= 0.0:
        raise ValueError("--trpo-target-kl must be positive.")
    if trpo_sub_sampling_factor <= 0:
        raise ValueError("--trpo-sub-sampling-factor must be positive.")
    if trpo_stats_window_size <= 0:
        raise ValueError("--trpo-stats-window-size must be positive.")

    if args.lora_target_modules is None:
        resolved_lora_target_modules = tuple(DEFAULT_LORA_TARGET_MODULES)
    else:
        resolved_lora_target_modules = tuple(str(name) for name in args.lora_target_modules)

    scenario_cycle_sample_size = int(args.scenario_cycle_sample_size)
    if scenario_cycle_sample_size > 0 and not scenario_cycle_paths:
        raise ValueError("--scenario-cycle-sample-size requires scenario-cycle mode.")
    if scenario_cycle_paths and scenario_cycle_sample_size > len(scenario_cycle_paths):
        raise ValueError(
            "--scenario-cycle-sample-size cannot exceed the number of selected scenario-cycle scenes."
        )
    rollout_episodes_per_update = int(args.rollout_episodes_per_update)
    if scenario_cycle_paths and rollout_episodes_per_update <= 0:
        rollout_episodes_per_update = (
            scenario_cycle_sample_size if scenario_cycle_sample_size > 0 else len(scenario_cycle_paths)
        )
    if rollout_episodes_per_update > 0 and not scenario_cycle_paths:
        raise ValueError("--rollout-episodes-per-update requires scenario-cycle mode.")
    if scenario_cycle_sample_size > 0 and rollout_episodes_per_update != scenario_cycle_sample_size:
        raise ValueError(
            "--scenario-cycle-sample-size must equal --rollout-episodes-per-update so each sampled scene runs exactly one episode per cycle."
        )
    episode_cycle_enabled = rollout_episodes_per_update > 0
    if scenario_cycle_paths and config.train.num_envs != 1:
        raise ValueError("Scenario-cycle mode currently requires --num-envs 1.")

    rollout_step_budget = 0
    if episode_cycle_enabled:
        rollout_step_budget = (
            int(args.rollout_step_budget)
            if int(args.rollout_step_budget) > 0
            else int(config.env.max_episode_steps) * int(rollout_episodes_per_update)
        )
        if rollout_step_budget <= 0:
            raise ValueError("Episode-cycle TRPO requires a positive rollout step budget.")
        config.train.n_steps = int(rollout_step_budget)
        if args.video_interval_episodes is None:
            config.train.video_interval_episodes = int(rollout_episodes_per_update)

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
    config_payload["algorithm"] = "trpo"
    config_payload["trpo"] = {
        "cg_max_steps": int(trpo_cg_max_steps),
        "cg_damping": float(trpo_cg_damping),
        "line_search_shrinking_factor": float(trpo_line_search_shrinking_factor),
        "line_search_max_iter": int(trpo_line_search_max_iter),
        "n_critic_updates": int(trpo_n_critic_updates),
        "target_kl": float(trpo_target_kl),
        "sub_sampling_factor": int(trpo_sub_sampling_factor),
        "normalize_advantage": bool(trpo_normalize_advantage),
        "stats_window_size": int(trpo_stats_window_size),
        "batch_size": int(config.train.batch_size),
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

    print("Algorithm: TRPO")
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
        print(f"Resuming from model: {train_module._relative_path_text(resume_path)}")
    if resume_policy_weights_path is not None:
        print(f"Resuming from policy weights: {train_module._relative_path_text(resume_policy_weights_path)}")
    if bc_weights_path is not None:
        print(f"Initializing actor from BC weights: {train_module._relative_path_text(bc_weights_path)}")
    print(
        "TRPO setup: "
        f"target_kl={trpo_target_kl:.4f}, cg_steps={trpo_cg_max_steps}, "
        f"critic_updates={trpo_n_critic_updates}, sub_sampling_factor={trpo_sub_sampling_factor}"
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
            "Use --num-envs 1 if you want video capture only on every saved episode and lower RAM usage."
        )
    if episode_cycle_enabled:
        estimated_buffer_bytes = train_module._estimate_rollout_buffer_bytes(rollout_step_budget, config.env)
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

    algorithm_class: type[BaseAlgorithm] = EpisodeCycleTRPO if episode_cycle_enabled else TRPO
    policy_spec: str | type = LoraMultiInputPolicy if args.use_lora else "MultiInputPolicy"
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

    model_kwargs: dict[str, Any] = {
        "policy": policy_spec,
        "env": env,
        "learning_rate": config.train.learning_rate,
        "n_steps": config.train.n_steps,
        "batch_size": config.train.batch_size,
        "gamma": config.train.gamma,
        "gae_lambda": config.train.gae_lambda,
        "cg_max_steps": int(trpo_cg_max_steps),
        "cg_damping": float(trpo_cg_damping),
        "line_search_shrinking_factor": float(trpo_line_search_shrinking_factor),
        "line_search_max_iter": int(trpo_line_search_max_iter),
        "n_critic_updates": int(trpo_n_critic_updates),
        "normalize_advantage": bool(trpo_normalize_advantage),
        "target_kl": float(trpo_target_kl),
        "sub_sampling_factor": int(trpo_sub_sampling_factor),
        "stats_window_size": int(trpo_stats_window_size),
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

    model = algorithm_class(**model_kwargs)

    if resume_path is not None or resume_policy_weights_path is not None:
        if resume_path is not None:
            resume_policy_state, resume_num_timesteps = train_module._load_resume_policy_snapshot(
                resume_path,
                device=requested_device,
                load_candidates=[EpisodeCycleTRPO, TRPO, PPO],
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
        loaded_keys = load_actor_state_dict(model.policy, actor_state_dict)
        if not loaded_keys:
            raise RuntimeError(f"No actor parameters were loaded from BC checkpoint: {bc_weights_path}")
        print(f"Loaded {len(loaded_keys)} BC actor parameter tensors into TRPO.")

    checkpoint_callback = train_module.WeightCheckpointCallback(
        save_dir=checkpoint_dir,
        model_name=config.train.model_name,
        save_freq=config.train.checkpoint_interval_timesteps,
        save_policy_weights=config.train.save_policy_weights,
        save_replay_buffer=False,
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
            TRPOCycleMetricsCallback(
                save_dir=checkpoint_dir,
                metrics_csv_path=checkpoint_metrics_path,
                model_name=config.train.model_name,
                run_id=run_id,
                save_policy_weights=config.train.save_policy_weights,
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
    if config.train.video_interval_episodes > 0 and (config.train.save_episode_videos or not episode_cycle_enabled):
        callback_list.append(
            train_module.EpisodeArtifactCallback(
                video_dir=video_dir,
                checkpoint_dir=checkpoint_dir,
                model_name=config.train.model_name,
                save_every_episodes=config.train.video_interval_episodes,
                fps=config.train.video_fps,
                save_policy_weights=config.train.save_policy_weights,
                initial_completed_episodes=existing_episode_count,
                save_checkpoint_artifacts=not episode_cycle_enabled,
                save_replay_buffer=False,
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
            save_replay_buffer=False,
        )
        archived_model_path, archived_weights_path = train_module.save_training_artifacts(
            model=model,
            save_dir=log_dir,
            model_name=config.train.model_name,
            save_policy_weights=config.train.save_policy_weights,
            suffix=f"_interrupted_{run_id}",
            save_replay_buffer=False,
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
            save_replay_buffer=False,
        )
        archived_model_path, archived_weights_path = train_module.save_training_artifacts(
            model=model,
            save_dir=log_dir,
            model_name=config.train.model_name,
            save_policy_weights=config.train.save_policy_weights,
            suffix=f"_{run_id}",
            save_replay_buffer=False,
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
        "algorithm": "trpo",
        "scenario_path": train_module._relative_path_text(scenario_path),
        "scenario_cycle_paths": None
        if not scenario_cycle_paths
        else [train_module._relative_path_text(path) for path in scenario_cycle_paths],
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
        "rollout_step_budget": int(rollout_step_budget),
        "strict_rollout_step_budget": bool(args.strict_rollout_step_budget),
        "lora": {
            "enabled": bool(args.use_lora),
            "rank": int(args.lora_rank) if args.use_lora else 0,
            "alpha": float(args.lora_alpha) if args.use_lora else 0.0,
            "dropout": float(args.lora_dropout) if args.use_lora else 0.0,
            "target_modules": list(resolved_lora_target_modules) if args.use_lora else [],
            "freeze_actor_base": bool(args.lora_freeze_actor_base) if args.use_lora else False,
            "train_bias": bool(args.lora_train_bias) if args.use_lora else False,
        },
        "trpo": {
            "cg_max_steps": int(trpo_cg_max_steps),
            "cg_damping": float(trpo_cg_damping),
            "line_search_shrinking_factor": float(trpo_line_search_shrinking_factor),
            "line_search_max_iter": int(trpo_line_search_max_iter),
            "n_critic_updates": int(trpo_n_critic_updates),
            "target_kl": float(trpo_target_kl),
            "sub_sampling_factor": int(trpo_sub_sampling_factor),
            "normalize_advantage": bool(trpo_normalize_advantage),
            "stats_window_size": int(trpo_stats_window_size),
            "batch_size": int(config.train.batch_size),
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
