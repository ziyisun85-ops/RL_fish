from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


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
        **kwargs: Any,
    ) -> None:
        if kwargs.get("rollout_buffer_class") is None:
            kwargs["rollout_buffer_class"] = ResizableDictRolloutBuffer
        super().__init__(*args, **kwargs)
        self.rollout_episodes_per_update = max(0, int(rollout_episodes_per_update))
        self.strict_episode_budget = bool(strict_episode_budget)

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
