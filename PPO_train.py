from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import BaseBuffer, DictRolloutBuffer

import algorithms.episode_cycle_ppo as episode_cycle_ppo_module
import train as train_module
from train import main
from train_launcher_config import (
    PPO_LOG_DIR,
    PPO_MODEL_NAME,
    PPO_BC_WEIGHTS,
    build_wrapper_default_args,
    install_common_train_patches,
    parse_train_args,
)


DEFAULT_ARGS = build_wrapper_default_args(
    algo="ppo",
    log_dir=PPO_LOG_DIR,
    model_name=PPO_MODEL_NAME,
    bc_weights=PPO_BC_WEIGHTS,
)


class DynamicResizableDictRolloutBuffer(DictRolloutBuffer):
    """List-backed rollout buffer to avoid preallocating ~10 GiB for 100-scene PPO cycles."""

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


class PPOCycleMetricsCallback(BaseCallback):
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
        global _ACTIVE_PPO_CYCLE_METRICS_CALLBACK
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_PPO_CYCLE_METRICS_CALLBACK = self
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
        global _ACTIVE_PPO_CYCLE_METRICS_CALLBACK
        if _ACTIVE_PPO_CYCLE_METRICS_CALLBACK is self:
            _ACTIVE_PPO_CYCLE_METRICS_CALLBACK = None
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


_PATCH_ARGS: Any = None
_ACTIVE_PPO_CYCLE_METRICS_CALLBACK: "PPOCycleMetricsCallback | None" = None


def _maybe_build_ppo_cycle_metrics_callback(callbacks: list[Any]) -> BaseCallback | None:
    global _PATCH_ARGS

    args = _PATCH_ARGS
    if args is None:
        return None
    if str(getattr(args, "algo", "ppo")).strip().lower() != "ppo":
        return None
    episodes_per_cycle = int(getattr(args, "rollout_episodes_per_update", 0))
    if episodes_per_cycle <= 0:
        return None

    weight_callback = next(
        (callback for callback in callbacks if isinstance(callback, train_module.WeightCheckpointCallback)),
        None,
    )
    episode_metrics_callback = next(
        (callback for callback in callbacks if isinstance(callback, train_module.EpisodeMetricsCallback)),
        None,
    )
    if weight_callback is None or episode_metrics_callback is None:
        return None

    checkpoint_dir = Path(weight_callback.save_dir)
    checkpoint_metrics_path = checkpoint_dir.parent / train_module.make_config().train.checkpoint_metrics_filename
    model_name = str(weight_callback.model_name)
    existing_cycle_update_index = max(
        int(train_module._detect_latest_cycle_update_index(checkpoint_dir, model_name)),
        int(train_module._extract_cycle_update_index(train_module.resolve_resume_path(args))),
        int(train_module._extract_cycle_update_index(train_module.resolve_resume_policy_weights_path(args))),
    )
    existing_episode_count = int(episode_metrics_callback.initial_episode_index)
    initial_cycle_episode_rows: list[dict[str, Any]] = []
    align_updates = bool(getattr(args, "align_rollout_updates_to_episode_count", False))
    if align_updates and episodes_per_cycle > 0:
        pending_cycle_episodes = existing_episode_count % episodes_per_cycle
        if pending_cycle_episodes > 0:
            initial_cycle_episode_rows = train_module.load_recent_cycle_episode_rows(
                Path(episode_metrics_callback.csv_path),
                pending_cycle_episodes,
            )

    return PPOCycleMetricsCallback(
        save_dir=checkpoint_dir,
        metrics_csv_path=checkpoint_metrics_path,
        model_name=model_name,
        run_id=str(episode_metrics_callback.run_id),
        save_policy_weights=bool(weight_callback.save_policy_weights),
        episodes_per_cycle=episodes_per_cycle,
        initial_episode_count=existing_episode_count,
        initial_cycle_index=existing_cycle_update_index,
        align_to_episode_count=align_updates,
        initial_cycle_episode_rows=initial_cycle_episode_rows,
    )


def _install_runtime_patches(parsed_args: Any) -> None:
    install_common_train_patches(train_module, parsed_args)
    original_callback_list = train_module.CallbackList

    class PatchedCallbackList(original_callback_list):
        def __init__(self, callbacks):
            callbacks = list(callbacks)
            cycle_metrics_callback = _maybe_build_ppo_cycle_metrics_callback(callbacks)
            if cycle_metrics_callback is not None:
                insert_index = 0
                for index, callback in enumerate(callbacks):
                    insert_index = index + 1
                    if isinstance(callback, train_module.EpisodeMetricsCallback):
                        break
                callbacks.insert(insert_index, cycle_metrics_callback)
            super().__init__(callbacks)

    def patched_save_post_update_checkpoint(self, iteration: int) -> None:
        global _ACTIVE_PPO_CYCLE_METRICS_CALLBACK
        if self.post_update_save_every_iterations <= 0 or self.post_update_save_dir is None:
            return
        if int(iteration) % int(self.post_update_save_every_iterations) != 0:
            return

        checkpoint_index = self.post_update_initial_iteration + int(iteration)
        suffix = f"_update_{checkpoint_index:06d}"
        model_path, weights_path = episode_cycle_ppo_module._save_training_artifacts(
            model=self,
            save_dir=self.post_update_save_dir,
            model_name=self.post_update_model_name,
            save_policy_weights=self.post_update_save_policy_weights,
            suffix=suffix,
        )
        cycle_metrics_callback = _ACTIVE_PPO_CYCLE_METRICS_CALLBACK
        if cycle_metrics_callback is not None:
            cycle_metrics_callback.record_saved_cycle_checkpoint(
                update_index=int(checkpoint_index),
                model_path=model_path,
                weights_path=weights_path,
            )

    global _PATCH_ARGS
    _PATCH_ARGS = parsed_args
    train_module.CallbackList = PatchedCallbackList
    episode_cycle_ppo_module.ResizableDictRolloutBuffer = DynamicResizableDictRolloutBuffer
    episode_cycle_ppo_module.EpisodeCyclePPO._save_post_update_checkpoint = patched_save_post_update_checkpoint


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    _install_runtime_patches(parse_train_args(train_module, sys.argv))
    main()
