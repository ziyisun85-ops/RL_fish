from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

from configs.default_config import ExperimentConfig
from envs import FishPathAvoidEnv


ACTOR_STATE_PREFIXES = (
    "log_std",
    "features_extractor.",
    "mlp_extractor.policy_net.",
    "action_net.",
)


def resolve_scenario_path(
    scenario_path: str | None,
    scenario_index: int | None,
    scenario_dir: str | Path,
) -> Path | None:
    if scenario_path is not None and scenario_index is not None:
        raise ValueError("Use either scenario_path or scenario_index, not both.")

    if scenario_path is not None:
        resolved_path = Path(scenario_path).resolve()
    elif scenario_index is not None:
        if int(scenario_index) <= 0:
            raise ValueError("scenario_index must be a positive integer.")
        resolved_path = Path(scenario_dir).resolve() / f"training_env_{int(scenario_index):02d}.json"
    else:
        return None

    if not resolved_path.exists():
        raise FileNotFoundError(f"Scenario JSON not found: {resolved_path}")
    return resolved_path


def make_single_env(
    config: ExperimentConfig,
    *,
    scenario_path: str | Path | None = None,
    render_mode: str | None = None,
    enable_mujoco_viewer: bool = False,
    realtime_playback: bool = False,
    viewer_slowdown: float = 0.0,
    enable_episode_recording: bool = False,
    viewer_key_callback: Callable[[int], None] | None = None,
) -> FishPathAvoidEnv:
    return FishPathAvoidEnv(
        config=config.env,
        render_mode=render_mode,
        enable_mujoco_viewer=enable_mujoco_viewer,
        realtime_playback=realtime_playback,
        viewer_slowdown=viewer_slowdown,
        enable_episode_recording=enable_episode_recording,
        viewer_key_callback=viewer_key_callback,
        scenario_path=scenario_path,
    )


def build_vec_env(
    config: ExperimentConfig,
    *,
    scenario_path: str | Path | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    monitor_path: str | Path | None = None,
    enable_mujoco_viewer: bool = False,
    render_env_index: int = 0,
    realtime_playback: bool = False,
    viewer_slowdown: float = 0.0,
    enable_episode_recording: bool = False,
    record_every_n_episodes: int = 1,
):
    env_count = int(config.train.num_envs if num_envs is None else num_envs)
    base_seed = int(config.train.seed if seed is None else seed)

    def make_env(rank: int):
        def _factory() -> FishPathAvoidEnv:
            env = FishPathAvoidEnv(
                config=config.env,
                enable_mujoco_viewer=enable_mujoco_viewer and rank == int(render_env_index),
                realtime_playback=realtime_playback and rank == int(render_env_index),
                viewer_slowdown=viewer_slowdown,
                enable_episode_recording=enable_episode_recording,
                record_every_n_episodes=record_every_n_episodes,
                scenario_path=scenario_path,
            )
            env.reset(seed=base_seed + rank)
            return env

        return _factory

    vec_env = DummyVecEnv([make_env(rank) for rank in range(env_count)])
    if monitor_path is not None:
        vec_env = VecMonitor(vec_env, filename=str(Path(monitor_path)))
    vec_env = VecTransposeImage(vec_env)
    return vec_env


def build_ppo_model(
    config: ExperimentConfig,
    env,
    *,
    device: str = "cpu",
    seed: int | None = None,
    verbose: int = 0,
) -> PPO:
    return PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=config.train.learning_rate,
        n_steps=config.train.n_steps,
        batch_size=config.train.batch_size,
        gamma=config.train.gamma,
        gae_lambda=config.train.gae_lambda,
        clip_range=config.train.clip_range,
        ent_coef=config.train.ent_coef,
        vf_coef=config.train.vf_coef,
        max_grad_norm=config.train.max_grad_norm,
        policy_kwargs={"net_arch": list(config.train.policy_hidden_sizes)},
        verbose=int(verbose),
        seed=int(config.train.seed if seed is None else seed),
        device=device,
    )


def actor_state_dict_from_policy_state(policy_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    actor_state: dict[str, torch.Tensor] = {}
    for key, value in policy_state_dict.items():
        if key == "log_std" or key.startswith(ACTOR_STATE_PREFIXES[1:]):
            actor_state[key] = value.detach().cpu()
    return actor_state


def actor_state_dict_from_policy(policy) -> dict[str, torch.Tensor]:
    actor_state: dict[str, torch.Tensor] = {}
    for name, parameter in policy.named_parameters():
        if not is_actor_parameter_name(name):
            continue
        actor_state[name] = parameter.detach().cpu()
    for name, buffer in policy.named_buffers():
        if not is_actor_parameter_name(name):
            continue
        actor_state[name] = buffer.detach().cpu()
    return actor_state


def is_actor_parameter_name(parameter_name: str) -> bool:
    return parameter_name == "log_std" or parameter_name.startswith(ACTOR_STATE_PREFIXES[1:])


def actor_parameters(policy) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in policy.named_parameters():
        if not is_actor_parameter_name(name):
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        parameters.append(parameter)
    return parameters


def load_actor_state_dict(policy, actor_state_dict: dict[str, torch.Tensor]) -> list[str]:
    loaded_keys: list[str] = []
    named_parameters = dict(policy.named_parameters())
    for key, parameter in named_parameters.items():
        if key not in actor_state_dict:
            continue
        value = actor_state_dict[key].to(device=parameter.device, dtype=parameter.dtype)
        parameter.data.copy_(value)
        loaded_keys.append(key)

    named_buffers = dict(policy.named_buffers())
    for key, buffer in named_buffers.items():
        if key not in actor_state_dict:
            continue
        value = actor_state_dict[key].to(device=buffer.device, dtype=buffer.dtype)
        buffer.data.copy_(value)
        loaded_keys.append(key)
    return loaded_keys


def load_matching_policy_state_dict(policy, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []

    def candidate_keys(key: str) -> list[str]:
        candidates = [key]
        if key.startswith("pi_features_extractor."):
            candidates.append(f"features_extractor.{key.split('.', 1)[1]}")
        if key.startswith("vf_features_extractor."):
            candidates.append(f"features_extractor.{key.split('.', 1)[1]}")
        return candidates

    named_parameters = dict(policy.named_parameters())
    for key, value in state_dict.items():
        parameter = None
        target_key = key
        for candidate in candidate_keys(key):
            parameter = named_parameters.get(candidate)
            if parameter is not None:
                target_key = candidate
                break
        if parameter is None:
            skipped_keys.append(key)
            continue
        if tuple(parameter.shape) != tuple(value.shape):
            skipped_keys.append(key)
            continue
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        loaded_keys.append(target_key)

    named_buffers = dict(policy.named_buffers())
    for key, value in state_dict.items():
        buffer = None
        target_key = key
        for candidate in candidate_keys(key):
            buffer = named_buffers.get(candidate)
            if buffer is not None:
                target_key = candidate
                break
        if buffer is None:
            continue
        if tuple(buffer.shape) != tuple(value.shape):
            skipped_keys.append(key)
            continue
        buffer.data.copy_(value.to(device=buffer.device, dtype=buffer.dtype))
        if target_key not in loaded_keys:
            loaded_keys.append(target_key)

    return loaded_keys, skipped_keys
