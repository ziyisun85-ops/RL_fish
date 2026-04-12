from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

from configs.default_config import PROJECT_ROOT, config_to_dict, make_config
from envs import FishPathAvoidEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for fish path following with local obstacle avoidance.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total training timesteps.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel environments.")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a previously saved .zip PPO model.",
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
        default=str((PROJECT_ROOT / "scenarios" / "training_envs").resolve()),
        help="Directory containing exported fixed environment JSON files.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=500,
        help="Stop training after this many completed episodes. Use 0 or a negative value to disable the limit.",
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


def resolve_resume_path(args: argparse.Namespace) -> Path | None:
    if args.resume_from is None:
        return None

    resume_path = Path(args.resume_from).resolve()
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume model not found: {resume_path}")
    if resume_path.suffix.lower() != ".zip":
        raise ValueError(f"--resume-from must point to a .zip PPO model, got: {resume_path}")
    return resume_path


def _cpu_policy_state_dict(model: PPO) -> dict[str, torch.Tensor]:
    policy_state = model.policy.state_dict()
    return {key: value.detach().cpu() for key, value in policy_state.items()}


def _unwrap_vec_env_envs(vec_env) -> list[FishPathAvoidEnv]:
    current = vec_env
    while hasattr(current, "venv"):
        current = current.venv
    envs = getattr(current, "envs", None)
    if envs is None:
        raise TypeError(f"Unsupported VecEnv wrapper chain: {type(current)!r}")
    return list(envs)


def save_training_artifacts(
    model: PPO,
    save_dir: Path,
    model_name: str,
    save_policy_weights: bool,
    *,
    suffix: str = "",
) -> tuple[Path, Path | None]:
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}{suffix}"
    model_path = save_dir / f"{stem}.zip"
    model.save(str(model_path))

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


class WeightCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_dir: Path,
        model_name: str,
        save_freq: int,
        save_policy_weights: bool,
    ) -> None:
        super().__init__(verbose=0)
        self.save_dir = save_dir
        self.model_name = model_name
        self.save_freq = max(0, int(save_freq))
        self.save_policy_weights = save_policy_weights
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
        )
        weights_message = f", policy weights to {weights_path}" if weights_path is not None else ""
        print(f"Checkpoint saved to {model_path}{weights_message}")
        self._last_save_timestep = int(self.num_timesteps)
        return True


class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, csv_path: Path) -> None:
        super().__init__(verbose=0)
        self.csv_path = csv_path
        self._fieldnames = [
            "num_timesteps",
            "episode_reward",
            "episode_length",
            "episode_time_sec",
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
        self._file = None
        self._writer = None
        self._episode_counter = 0

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def _on_step(self) -> bool:
        if self._writer is None or self._file is None:
            return True

        infos = self.locals.get("infos", [])
        wrote_row = False
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue

            row = {
                "num_timesteps": int(self.num_timesteps),
                "episode_reward": float(episode_info.get("r", 0.0)),
                "episode_length": int(episode_info.get("l", 0)),
                "episode_time_sec": float(episode_info.get("t", 0.0)),
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
            self._episode_counter += 1
            print(
                "Episode "
                f"{self._episode_counter} stopped: {row['termination_reason']} | "
                f"reward={row['episode_reward']:.3f} | steps={row['episode_length']}"
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
    ) -> None:
        super().__init__(verbose=0)
        self.video_dir = video_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.save_every_episodes = max(0, int(save_every_episodes))
        self.fps = max(1, int(fps))
        self.save_policy_weights = save_policy_weights
        self.completed_episodes = 0
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
            model_path, weights_path = save_training_artifacts(
                model=self.model,
                save_dir=self.checkpoint_dir,
                model_name=self.model_name,
                save_policy_weights=self.save_policy_weights,
                suffix=episode_suffix,
            )
            if saved_path is not None:
                weights_message = f", weights to {weights_path}" if weights_path is not None else ""
                head_message = f"; head camera video to {saved_head_path}" if saved_head_path is not None else ""
                print(f"Saved episode video to {saved_path}{head_message}; checkpoint to {model_path}{weights_message}")

        return True


class StopAfterEpisodesCallback(BaseCallback):
    def __init__(self, max_episodes: int) -> None:
        super().__init__(verbose=0)
        self.max_episodes = int(max_episodes)
        self.completed_episodes = 0

    def _on_step(self) -> bool:
        if self.max_episodes <= 0:
            return True

        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("episode") is None:
                continue
            self.completed_episodes += 1
            if self.completed_episodes >= self.max_episodes:
                print(f"Reached max episode limit: {self.completed_episodes}. Stopping training.")
                return False
        return True


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
    scenario_path = resolve_scenario_path(args)
    resume_path = resolve_resume_path(args)
    if args.timesteps is not None:
        config.train.total_timesteps = args.timesteps
    if args.num_envs is not None:
        config.train.num_envs = args.num_envs
    if args.record_videos is not None:
        config.train.save_episode_videos = bool(args.record_videos)
    if args.video_interval_episodes is not None:
        config.train.video_interval_episodes = args.video_interval_episodes
    if args.xml_path is not None:
        config.env.model.xml_path = str(Path(args.xml_path).resolve())
    if args.render and not 0 <= args.render_env_index < config.train.num_envs:
        raise ValueError(
            f"--render-env-index must be within [0, {config.train.num_envs - 1}] when --render is enabled."
        )
    if args.render_slowdown < 0.0:
        raise ValueError("--render-slowdown must be non-negative.")

    log_dir = Path(config.train.log_dir)
    if scenario_path is not None:
        log_dir = log_dir / scenario_path.stem
    checkpoint_dir = log_dir / config.train.checkpoint_dirname
    monitor_path = log_dir / config.train.monitor_filename
    episode_metrics_path = log_dir / config.train.episode_metrics_filename
    video_dir = log_dir / config.train.video_dirname
    reward_plot_path = log_dir / "reward_curve.png"
    log_dir.mkdir(parents=True, exist_ok=True)
    config_payload = config_to_dict(config)
    if scenario_path is not None:
        config_payload["selected_scenario_path"] = str(scenario_path)
    if resume_path is not None:
        config_payload["resume_from"] = str(resume_path)
    with (log_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config_payload, config_file, indent=2, ensure_ascii=False)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "Requested CUDA training, but the current PyTorch build has no CUDA support. "
            "Falling back to CPU."
        )
        requested_device = "cpu"
    elif requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training device: {requested_device}")
    if scenario_path is not None:
        print(f"Training on fixed scenario: {scenario_path}")
    if resume_path is not None:
        print(f"Resuming from model: {resume_path}")
    if config.train.save_episode_videos and config.train.video_interval_episodes > 1 and config.train.num_envs > 1:
        print(
            "Video capture is enabled with multiple environments. "
            "Episodes finish asynchronously, so frames still need to be buffered for every env. "
            "Use --num-envs 1 if you want video capture only on every saved episode and lower RAM usage."
        )

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
            )
            env.reset(seed=config.train.seed + rank)
            return env

        return _factory

    env = DummyVecEnv([make_env(rank) for rank in range(config.train.num_envs)])
    env = VecMonitor(env, filename=str(monitor_path))
    env = VecTransposeImage(env)

    # The policy consumes a dict observation: {"image": head camera RGB, "imu": body-frame IMU vector}.
    if resume_path is not None:
        model = PPO.load(
            str(resume_path),
            env=env,
            device=requested_device,
        )
        print(f"Loaded model with existing num_timesteps={int(model.num_timesteps)}")
    else:
        model = PPO(
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
            verbose=1,
            seed=config.train.seed,
            device=requested_device,
        )
    checkpoint_callback = WeightCheckpointCallback(
        save_dir=checkpoint_dir,
        model_name=config.train.model_name,
        save_freq=config.train.checkpoint_interval_timesteps,
        save_policy_weights=config.train.save_policy_weights,
    )
    episode_metrics_callback = EpisodeMetricsCallback(csv_path=episode_metrics_path)
    stop_after_episodes_callback = StopAfterEpisodesCallback(max_episodes=args.max_episodes)
    callback_list: list[BaseCallback] = [
        checkpoint_callback,
        episode_metrics_callback,
        stop_after_episodes_callback,
    ]
    if args.plot_reward:
        callback_list.append(
            EpisodeRewardPlotCallback(
                plot_path=reward_plot_path,
                moving_average_window=args.reward_plot_window,
            )
        )
    if config.train.video_interval_episodes > 0:
        callback_list.append(
            EpisodeArtifactCallback(
                video_dir=video_dir,
                checkpoint_dir=checkpoint_dir,
                model_name=config.train.model_name,
                save_every_episodes=config.train.video_interval_episodes,
                fps=config.train.video_fps,
                save_policy_weights=config.train.save_policy_weights,
            )
        )
    callback = CallbackList(callback_list)

    try:
        model.learn(
            total_timesteps=config.train.total_timesteps,
            callback=callback,
            reset_num_timesteps=resume_path is None,
        )
    except KeyboardInterrupt:
        interrupted_model_path, interrupted_weights_path = save_training_artifacts(
            model=model,
            save_dir=log_dir,
            model_name=config.train.model_name,
            save_policy_weights=config.train.save_policy_weights,
            suffix="_interrupted",
        )
        weights_message = (
            f", interrupted policy weights to {interrupted_weights_path}"
            if interrupted_weights_path is not None
            else ""
        )
        print(f"Training interrupted. Saved interrupted model to {interrupted_model_path}{weights_message}")
    else:
        final_model_path, final_weights_path = save_training_artifacts(
            model=model,
            save_dir=log_dir,
            model_name=config.train.model_name,
            save_policy_weights=config.train.save_policy_weights,
        )
        weights_message = f", policy weights to {final_weights_path}" if final_weights_path is not None else ""
        print(f"Saved model to {final_model_path}{weights_message}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
