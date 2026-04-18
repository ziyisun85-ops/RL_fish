from __future__ import annotations

import sys

import train as train_module
from train import main
from train_launcher_config import (
    SAC_LOG_DIR,
    SAC_MODEL_NAME,
    build_wrapper_default_args,
    install_common_train_patches,
    parse_train_args,
)


DEFAULT_ARGS = build_wrapper_default_args(
    algo="sac",
    log_dir=SAC_LOG_DIR,
    model_name=SAC_MODEL_NAME,
)


def _install_runtime_patches(parsed_args) -> None:
    install_common_train_patches(train_module, parsed_args)

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


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    _install_runtime_patches(parse_train_args(train_module, sys.argv))
    main()
