from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from configs.default_config import make_config
from envs import FishPathAvoidEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO fish policy.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a .zip model saved by train.py.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Render top-down trajectories during evaluation.")
    parser.add_argument("--xml-path", type=str, default=None, help="Override MuJoCo XML scene path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_config()
    if args.episodes is not None:
        config.eval.episodes = args.episodes
    if args.xml_path is not None:
        config.env.model.xml_path = str(Path(args.xml_path).resolve())

    raw_env: FishPathAvoidEnv | None = None

    def make_env() -> FishPathAvoidEnv:
        nonlocal raw_env
        raw_env = FishPathAvoidEnv(config=config.env, render_mode="human" if args.render else None)
        return raw_env

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    model_path = Path(args.model_path).resolve()
    if model_path.suffix == ".zip":
        model_path = model_path.with_suffix("")
    model = PPO.load(str(model_path))

    episode_rewards: list[float] = []
    goal_progress_ratios: list[float] = []
    collision_flags: list[bool] = []
    success_flags: list[bool] = []

    for episode_index in range(config.eval.episodes):
        if raw_env is None:
            raise RuntimeError("Evaluation env was not created.")
        env.seed(episode_index)
        observation = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(observation, deterministic=config.eval.deterministic)
            observation, rewards, dones, infos = env.step(action)
            done = bool(dones[0])
            info = infos[0]
            total_reward += float(rewards[0])
            if args.render:
                raw_env.render()

        episode_rewards.append(total_reward)
        goal_progress_ratios.append(float(info["goal_progress_ratio"]))
        collision_flags.append(bool(info["collision"]))
        success_flags.append(bool(info["success"]))
        print(
            f"Episode {episode_index + 1}: "
            f"reward={total_reward:.2f}, "
            f"goal_progress={info['goal_progress_ratio']:.3f}, "
            f"collision={info['collision']}, "
            f"success={info['success']}"
        )

    print("\nEvaluation summary")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Average goal completion: {np.mean(goal_progress_ratios):.3f}")
    print(f"Collision rate: {np.mean(collision_flags):.3f}")
    print(f"Success rate: {np.mean(success_flags):.3f}")

    env.close()


if __name__ == "__main__":
    main()
