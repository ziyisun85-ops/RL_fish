from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading
from pathlib import Path

import numpy as np

from configs.default_config import PROJECT_ROOT, config_to_dict, make_config
from utils.policy_utils import make_single_env, resolve_scenario_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect mouse-driven demonstrations for BC pretraining.")
    parser.add_argument("--output", type=str, required=True, help="Output .npz dataset path.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of demonstration episodes to collect.")
    parser.add_argument("--max-transitions", type=int, default=0, help="Stop after this many saved transitions.")
    parser.add_argument("--seed", type=int, default=7, help="Base reset seed.")
    parser.add_argument("--xml-path", type=str, default=None, help="Override MuJoCo XML scene path.")
    parser.add_argument("--scenario-path", type=str, default=None, help="Collect on one fixed exported environment JSON.")
    parser.add_argument("--scenario-index", type=int, default=None, help="Collect on training_env_XX.json.")
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=str((PROJECT_ROOT / "scenarios" / "training_envs").resolve()),
        help="Directory containing exported fixed environment JSON files.",
    )
    parser.add_argument("--viewer-slowdown", type=float, default=1.0, help="Passive viewer slowdown factor.")
    parser.add_argument("--left-action", type=float, default=-1.0, help="Action value applied while holding left mouse.")
    parser.add_argument("--right-action", type=float, default=1.0, help="Action value applied while holding right mouse.")
    parser.add_argument("--center-action", type=float, default=0.0, help="Action value applied when no mouse button is held.")
    return parser.parse_args()


class ManualActionController:
    _VK_LBUTTON = 0x01
    _VK_RBUTTON = 0x02

    def __init__(self, left_action: float, right_action: float, center_action: float) -> None:
        if os.name != "nt":
            raise RuntimeError("Mouse-button demonstration control currently requires Windows.")
        self.left_action = float(np.clip(left_action, -1.0, 1.0))
        self.right_action = float(np.clip(right_action, -1.0, 1.0))
        self.center_action = float(np.clip(center_action, -1.0, 1.0))
        self._lock = threading.Lock()
        self._quit_requested = False
        self._user32 = ctypes.windll.user32

    def on_key(self, keycode: int) -> None:
        with self._lock:
            if keycode in (ord("Q"), ord("q"), 256):
                self._quit_requested = True
                print("[keyboard] Quit requested.")

    def _mouse_pressed(self, vk_code: int) -> bool:
        return bool(self._user32.GetAsyncKeyState(int(vk_code)) & 0x8000)

    def current_action(self) -> float:
        left_pressed = self._mouse_pressed(self._VK_LBUTTON)
        right_pressed = self._mouse_pressed(self._VK_RBUTTON)
        if left_pressed and not right_pressed:
            return float(self.left_action)
        if right_pressed and not left_pressed:
            return float(self.right_action)
        return float(self.center_action)

    def quit_requested(self) -> bool:
        with self._lock:
            return bool(self._quit_requested)


class TransitionRecorder:
    def __init__(self) -> None:
        self.obs_images: list[np.ndarray] = []
        self.obs_imus: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.next_obs_images: list[np.ndarray] = []
        self.next_obs_imus: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.successes: list[bool] = []
        self.episode_ids: list[int] = []

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        next_obs: dict[str, np.ndarray],
        reward: float,
        done: bool,
        success: bool,
        episode_id: int,
    ) -> None:
        self.obs_images.append(np.asarray(obs["image"], dtype=np.uint8).copy())
        self.obs_imus.append(np.asarray(obs["imu"], dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(1).copy())
        self.next_obs_images.append(np.asarray(next_obs["image"], dtype=np.uint8).copy())
        self.next_obs_imus.append(np.asarray(next_obs["imu"], dtype=np.float32).copy())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.successes.append(bool(success))
        self.episode_ids.append(int(episode_id))

    def __len__(self) -> int:
        return len(self.rewards)

    def save(self, output_path: Path, metadata: dict[str, object]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            obs_image=np.stack(self.obs_images, axis=0).astype(np.uint8),
            obs_imu=np.stack(self.obs_imus, axis=0).astype(np.float32),
            action=np.stack(self.actions, axis=0).astype(np.float32),
            next_obs_image=np.stack(self.next_obs_images, axis=0).astype(np.uint8),
            next_obs_imu=np.stack(self.next_obs_imus, axis=0).astype(np.float32),
            reward=np.asarray(self.rewards, dtype=np.float32),
            done=np.asarray(self.dones, dtype=bool),
            success=np.asarray(self.successes, dtype=bool),
            episode_id=np.asarray(self.episode_ids, dtype=np.int32),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )


def main() -> None:
    args = parse_args()
    config = make_config()
    if args.xml_path is not None:
        config.env.model.xml_path = str(Path(args.xml_path).resolve())

    scenario_path = resolve_scenario_path(args.scenario_path, args.scenario_index, args.scenario_dir)
    controller = ManualActionController(
        left_action=args.left_action,
        right_action=args.right_action,
        center_action=args.center_action,
    )
    recorder = TransitionRecorder()
    env = make_single_env(
        config,
        scenario_path=scenario_path,
        enable_mujoco_viewer=True,
        realtime_playback=True,
        viewer_slowdown=args.viewer_slowdown,
        viewer_key_callback=controller.on_key,
    )
    output_path = Path(args.output).resolve()

    print("Mouse demo collection started.")
    print("Controls: hold left mouse=left, hold right mouse=right, release=center, Q or ESC=quit.")
    print(f"Dataset path: {output_path}")
    if scenario_path is not None:
        print(f"Fixed scenario: {scenario_path}")

    try:
        for episode_index in range(int(args.episodes)):
            if controller.quit_requested():
                break

            obs, info = env.reset(seed=int(args.seed) + episode_index)
            done = False
            episode_reward = 0.0
            step_count = 0
            success = False

            while not done:
                if controller.quit_requested() or not env.is_viewer_running():
                    done = True
                    break

                action = np.array([controller.current_action()], dtype=np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action)
                success = bool(info.get("success", False))
                done = bool(terminated or truncated)
                recorder.add(obs, action, next_obs, reward, done, success, episode_index)
                obs = next_obs
                episode_reward += float(reward)
                step_count += 1

                if int(args.max_transitions) > 0 and len(recorder) >= int(args.max_transitions):
                    print(f"Reached max transitions: {len(recorder)}")
                    done = True
                    break

            print(
                f"Episode {episode_index + 1}/{int(args.episodes)}: "
                f"steps={step_count}, reward={episode_reward:.3f}, success={success}"
            )

            if int(args.max_transitions) > 0 and len(recorder) >= int(args.max_transitions):
                break
            if not env.is_viewer_running():
                break
    finally:
        env.close()

    if len(recorder) == 0:
        raise RuntimeError("No demonstration transitions were collected.")

    metadata = {
        "schema_version": 1,
        "transition_count": len(recorder),
        "episode_count": int(max(recorder.episode_ids) + 1) if recorder.episode_ids else 0,
        "action_space_range": [-1.0, 1.0],
        "left_action": controller.left_action,
        "right_action": controller.right_action,
        "center_action": controller.center_action,
        "scenario_path": None if scenario_path is None else str(scenario_path),
        "config": config_to_dict(config),
    }
    recorder.save(output_path, metadata=metadata)
    print(f"Saved demonstration dataset to {output_path}")
    print(f"Transitions: {len(recorder)}")


if __name__ == "__main__":
    main()
