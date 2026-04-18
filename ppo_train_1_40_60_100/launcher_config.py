from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent


# User-editable launcher settings.
USE_LORA = True
DEVICE = "auto"
NUM_ENVS = 1
MAX_EPISODES = 0


def _first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


SCENARIO_CYCLE_DIR = _first_existing_path(
    Path("ppo_train_1_40_60_100") / "scenarios" / "train" / "json",
    Path("scenarios") / "large_pool_dataset_200" / "train" / "json",
).as_posix()
PPO_BC_WEIGHTS = _first_existing_path(
    Path("ppo_train_1_40_60_100") / "weights" / "bc_large_pool_1_40__60_100_epoch020_actor.pth",
    Path("runs") / "bc_pretrain" / "bc20_40_60_80_100_ppo" / "bc_large_pool_1_40__60_100_epoch020_actor.pth",
).as_posix()
ALIGN_ROLLOUT_UPDATES_TO_EPISODE_COUNT = True

# If 100-scene updates are too memory-heavy:
# - set EPISODES_PER_UPDATE smaller, for example 20, to process the full scene set in sequential chunks
# - optionally set SCENARIO_CYCLE_SAMPLE_SIZE to the same value if you prefer random scene batches per update
EPISODES_PER_UPDATE = 100
SCENARIO_CYCLE_SAMPLE_SIZE = 0

# Terminal summary cadence. Set to 0 to disable.
SUCCESS_RATE_PRINT_INTERVAL_EPISODES = 100

PPO_LOG_DIR = (Path("runs") / "ppo_train_1_40_60_100").as_posix()
PPO_MODEL_NAME = "ppo_train_1_40_60_100"


def build_wrapper_default_args() -> list[str]:
    if SCENARIO_CYCLE_SAMPLE_SIZE > 0 and SCENARIO_CYCLE_SAMPLE_SIZE != EPISODES_PER_UPDATE:
        raise ValueError(
            "SCENARIO_CYCLE_SAMPLE_SIZE must equal EPISODES_PER_UPDATE so each sampled scene runs exactly one episode."
        )

    args = [
        "--algo",
        "ppo",
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
    args.extend(
        [
            "--bc-weights",
            PPO_BC_WEIGHTS,
            "--log-dir",
            PPO_LOG_DIR,
            "--model-name",
            PPO_MODEL_NAME,
        ]
    )
    return args


def parse_train_args(train_module, argv: list[str]) -> Any:
    original_argv = list(sys.argv)
    try:
        sys.argv = argv
        return train_module.parse_args()
    finally:
        sys.argv = original_argv


class CycleSuccessRatePrinterCallback(BaseCallback):
    def __init__(
        self,
        *,
        print_interval_episodes: int,
        initial_episode_count: int = 0,
        align_to_episode_count: bool = False,
        initial_episode_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self.print_interval_episodes = max(0, int(print_interval_episodes))
        self.completed_episodes = max(0, int(initial_episode_count))
        self.align_to_episode_count = bool(align_to_episode_count)
        initial_rows = list(initial_episode_rows or [])
        if self.print_interval_episodes > 0:
            initial_rows = initial_rows[-self.print_interval_episodes :]
        self._current_rows = initial_rows
        self._next_print_episode = self._compute_initial_print_episode()

    def _compute_initial_print_episode(self) -> int:
        if self.print_interval_episodes <= 0:
            return 0
        if not self.align_to_episode_count:
            return self.completed_episodes + self.print_interval_episodes
        remainder = self.completed_episodes % self.print_interval_episodes
        if remainder == 0:
            return self.completed_episodes + self.print_interval_episodes
        return self.completed_episodes + (self.print_interval_episodes - remainder)

    def _on_step(self) -> bool:
        if self.print_interval_episodes <= 0:
            return True

        infos = self.locals.get("infos", [])
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue

            self._current_rows.append({"success": bool(info.get("success", False))})
            if len(self._current_rows) > self.print_interval_episodes:
                self._current_rows = self._current_rows[-self.print_interval_episodes :]
            self.completed_episodes += 1

            while self.completed_episodes >= self._next_print_episode:
                rows = list(self._current_rows)
                episode_count = len(rows)
                success_count = int(sum(1 for row in rows if bool(row.get("success", False))))
                success_rate = float(success_count / episode_count) if episode_count > 0 else 0.0
                start_episode = max(1, self._next_print_episode - episode_count + 1)
                end_episode = self._next_print_episode
                print(
                    f"Episodes {start_episode:06d}-{end_episode:06d} | "
                    f"success_rate={success_rate:.3f} ({success_count}/{episode_count})"
                )
                self._current_rows.clear()
                self._next_print_episode += self.print_interval_episodes

        return True


def _quiet_episode_metrics_on_step(self) -> bool:
    if self._writer is None or self._file is None:
        return True

    latest_train_loss = self._latest_train_loss_from_model(getattr(self, "model", None))
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
            "scenario_path": self._relative_path_text(info.get("scenario_path", self.scenario_path)) or "",
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
            "avg_goal_progress_ratio": float(
                info.get("avg_goal_progress_ratio", info.get("goal_progress_ratio", 0.0))
            ),
            "distance_to_goal_region": float(info.get("distance_to_goal_region", 0.0)),
            "visual_obstacle_detected": bool(info.get("visual_obstacle_detected", False)),
            "visual_obstacle_pixel_fraction": float(info.get("visual_obstacle_pixel_fraction", 0.0)),
            "visual_obstacle_center_fraction": float(info.get("visual_obstacle_center_fraction", 0.0)),
            "visual_obstacle_nearest_depth": float(info.get("visual_obstacle_nearest_depth", 0.0)),
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "wall_collision": bool(info.get("wall_collision", False)),
            "obstacle_collision_count": int(
                info.get("obstacle_collision_count", int(bool(info.get("collision", False))))
            ),
            "wall_collision_count": int(
                info.get("wall_collision_count", int(bool(info.get("wall_collision", False))))
            ),
            "out_of_bounds": bool(info.get("out_of_bounds", False)),
            "timeout": bool(info.get("timeout", False)),
        }
        self._writer.writerow(row)
        wrote_row = True

    if wrote_row:
        self._file.flush()
    return True


def install_common_train_patches(train_module, parsed_args: Any) -> None:
    train_module.EpisodeMetricsCallback._latest_train_loss_from_model = staticmethod(
        train_module._latest_train_loss_from_model
    )
    train_module.EpisodeMetricsCallback._relative_path_text = staticmethod(train_module._relative_path_text)
    train_module.EpisodeMetricsCallback._on_step = _quiet_episode_metrics_on_step

    original_callback_list = train_module.CallbackList

    def _maybe_build_success_rate_printer(callbacks: list[Any]) -> BaseCallback | None:
        if int(SUCCESS_RATE_PRINT_INTERVAL_EPISODES) <= 0:
            return None

        episode_metrics_callback = next(
            (callback for callback in callbacks if isinstance(callback, train_module.EpisodeMetricsCallback)),
            None,
        )
        if episode_metrics_callback is None:
            return None

        initial_episode_count = int(episode_metrics_callback.initial_episode_index)
        initial_rows: list[dict[str, Any]] = []
        align_prints = bool(getattr(parsed_args, "align_rollout_updates_to_episode_count", False))
        if align_prints and SUCCESS_RATE_PRINT_INTERVAL_EPISODES > 0:
            pending_episodes = initial_episode_count % int(SUCCESS_RATE_PRINT_INTERVAL_EPISODES)
            if pending_episodes > 0:
                initial_rows = train_module.load_recent_cycle_episode_rows(
                    Path(episode_metrics_callback.csv_path),
                    pending_episodes,
                )

        return CycleSuccessRatePrinterCallback(
            print_interval_episodes=int(SUCCESS_RATE_PRINT_INTERVAL_EPISODES),
            initial_episode_count=initial_episode_count,
            align_to_episode_count=align_prints,
            initial_episode_rows=initial_rows,
        )

    class PatchedCallbackList(original_callback_list):
        def __init__(self, callbacks):
            callbacks = list(callbacks)
            printer_callback = _maybe_build_success_rate_printer(callbacks)
            if printer_callback is not None:
                insert_index = 0
                for index, callback in enumerate(callbacks):
                    insert_index = index + 1
                    if isinstance(callback, train_module.EpisodeMetricsCallback):
                        break
                callbacks.insert(insert_index, printer_callback)
            super().__init__(callbacks)

    train_module.CallbackList = PatchedCallbackList
