from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


def _cpu_policy_state_dict(model: PPO) -> dict[str, th.Tensor]:
    policy_state = model.policy.state_dict()
    return {key: value.detach().cpu() for key, value in policy_state.items()}


def _save_training_artifacts(
    model: PPO,
    save_dir: str | Path,
    model_name: str,
    save_policy_weights: bool,
    *,
    suffix: str = "",
) -> tuple[Path, Path | None]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}{suffix}"
    model_path = save_dir / f"{stem}.zip"
    model.save(str(model_path))

    weights_path: Path | None = None
    if save_policy_weights:
        weights_path = save_dir / f"{stem}_policy.pth"
        th.save(
            {
                "num_timesteps": int(model.num_timesteps),
                "policy_state_dict": _cpu_policy_state_dict(model),
            },
            weights_path,
        )

    return model_path, weights_path


class ResizableDictRolloutBuffer(DictRolloutBuffer):
    def __init__(self, *args, **kwargs) -> None:
        if "buffer_size" in kwargs:
            self.max_buffer_size = int(kwargs["buffer_size"])
        elif args:
            self.max_buffer_size = int(args[0])
        else:
            raise ValueError("ResizableDictRolloutBuffer requires buffer_size.")
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        self.buffer_size = int(self.max_buffer_size)
        super().reset()

    def finalize(self, actual_size: int) -> None:
        actual_size = int(actual_size)
        if actual_size <= 0:
            raise ValueError("actual_size must be positive.")
        if actual_size > self.buffer_size:
            raise ValueError(f"actual_size {actual_size} exceeds buffer_size {self.buffer_size}.")
        if actual_size == self.buffer_size:
            self.full = True
            return
        for key in list(self.observations.keys()):
            self.observations[key] = self.observations[key][:actual_size]
        for name in ["actions", "rewards", "returns", "episode_starts", "values", "log_probs", "advantages"]:
            self.__dict__[name] = self.__dict__[name][:actual_size]
        self.buffer_size = actual_size
        self.pos = actual_size
        self.full = True


class EpisodeCyclePPO(PPO):
    def __init__(
        self,
        *args,
        rollout_episodes_per_update: int = 0,
        strict_episode_budget: bool = True,
        post_update_save_dir: str | None = None,
        post_update_model_name: str = "ppo_fish_baseline",
        post_update_save_policy_weights: bool = True,
        post_update_save_every_iterations: int = 0,
        post_update_initial_iteration: int = 0,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("rollout_buffer_class") is None:
            kwargs["rollout_buffer_class"] = ResizableDictRolloutBuffer
        super().__init__(*args, **kwargs)
        self.rollout_episodes_per_update = max(0, int(rollout_episodes_per_update))
        self.strict_episode_budget = bool(strict_episode_budget)
        self.post_update_save_dir = None if post_update_save_dir is None else str(Path(post_update_save_dir).resolve())
        self.post_update_model_name = str(post_update_model_name)
        self.post_update_save_policy_weights = bool(post_update_save_policy_weights)
        self.post_update_save_every_iterations = max(0, int(post_update_save_every_iterations))
        self.post_update_initial_iteration = max(0, int(post_update_initial_iteration))

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: DictRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        if self.rollout_episodes_per_update <= 0:
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        completed_episodes = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps and completed_episodes < self.rollout_episodes_per_update:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
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

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1
            completed_episodes += int(np.count_nonzero(dones))

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
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

        if completed_episodes < self.rollout_episodes_per_update and self.strict_episode_budget:
            raise RuntimeError(
                "Episode-cycle rollout hit the step budget before collecting the requested number of episodes. "
                "Increase --rollout-step-budget or reduce --rollout-episodes-per-update."
            )

        if isinstance(rollout_buffer, ResizableDictRolloutBuffer):
            rollout_buffer.finalize(n_steps)
        else:
            rollout_buffer.full = True

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def _save_post_update_checkpoint(self, iteration: int) -> None:
        if self.post_update_save_every_iterations <= 0 or self.post_update_save_dir is None:
            return
        if int(iteration) % int(self.post_update_save_every_iterations) != 0:
            return
        checkpoint_index = self.post_update_initial_iteration + int(iteration)
        suffix = f"_update_{checkpoint_index:06d}"
        model_path, weights_path = _save_training_artifacts(
            model=self,
            save_dir=self.post_update_save_dir,
            model_name=self.post_update_model_name,
            save_policy_weights=self.post_update_save_policy_weights,
            suffix=suffix,
        )
        weights_message = f", policy weights to {weights_path}" if weights_path is not None else ""
        print(f"Saved post-update checkpoint to {model_path}{weights_message}")

    def learn(
        self,
        total_timesteps: int,
        callback: BaseCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
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
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.dump_logs(iteration)

            self.train()
            self._save_post_update_checkpoint(iteration)

        callback.on_training_end()
        return self
