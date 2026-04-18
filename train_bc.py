from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from configs.default_config import PROJECT_ROOT, config_to_dict, make_config
from utils.policy_utils import actor_parameters, actor_state_dict_from_policy, build_ppo_model, build_vec_env, load_actor_state_dict


DEFAULT_LARGE_POOL_SCENARIO_DIR = PROJECT_ROOT / "scenarios" / "large_pool_dataset_200" / "train"
SCENARIO_ID_PATTERN = re.compile(r"train_env_(\d{3})")
EPOCH_METRICS_FIELDNAMES = [
    "epoch",
    "train_loss",
    "val_loss",
    "demo_episodes_processed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the PPO actor with behavior cloning.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset source: one .npz file, a directory of .npz files, or a glob pattern.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Directory containing per-scenario BC demo .npz files. Use with --scenario-selection.",
    )
    parser.add_argument(
        "--scenario-selection",
        type=str,
        default=None,
        help="Scene ids/ranges like '1-20,60-80'. Use with --dataset-dir.",
    )
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=str(DEFAULT_LARGE_POOL_SCENARIO_DIR.resolve()),
        help="Scenario split root or json directory used to validate --scenario-selection.",
    )
    parser.add_argument(
        "--allow-missing-demos",
        action="store_true",
        help="Skip selected scenes that do not have a matching per-scene demo file.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for BC checkpoints and metrics.")
    parser.add_argument("--model-name", type=str, default="bc_fish", help="Output checkpoint stem.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of BC epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="BC mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Actor optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optional Adam weight decay.")
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=2,
        help="Save intermediate BC checkpoints every N epochs. Defaults to 2. Set 0 to disable.",
    )
    parser.add_argument(
        "--save-every-demo-episodes",
        type=int,
        default=0,
        help=(
            "Save intermediate BC checkpoints whenever the cumulative number of selected demo episodes "
            "processed reaches another multiple of N. Checkpoints are written at epoch end after the threshold is crossed. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--episodes-per-scene",
        type=int,
        default=0,
        help=(
            "Limit each per-scene demo file to the first N distinct episode_id values in file order. "
            "Set 0 to use all episodes in each file."
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction in [0, 1).")
    parser.add_argument("--seed", type=int, default=7, help="Dataset split and model seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device: cuda, cuda:0, cpu, or auto.")
    parser.add_argument("--xml-path", type=str, default=None, help="Override MuJoCo XML scene path.")
    parser.add_argument(
        "--resume-actor",
        type=str,
        default=None,
        help="Resume BC training from a saved actor checkpoint (.pth/.pt). Epoch numbering continues from it.",
    )
    args = parser.parse_args()
    if args.dataset is not None and (args.dataset_dir is not None or args.scenario_selection is not None):
        parser.error("Use either --dataset, or (--dataset-dir with --scenario-selection), but not both.")
    if args.dataset is None and (args.dataset_dir is None or args.scenario_selection is None):
        parser.error("Provide --dataset, or provide both --dataset-dir and --scenario-selection.")
    if (args.dataset_dir is None) != (args.scenario_selection is None):
        parser.error("--dataset-dir and --scenario-selection must be provided together.")
    if int(args.save_every_epochs) < 0:
        parser.error("--save-every-epochs must be non-negative.")
    if int(args.save_every_demo_episodes) < 0:
        parser.error("--save-every-demo-episodes must be non-negative.")
    if int(args.episodes_per_scene) < 0:
        parser.error("--episodes-per-scene must be non-negative.")
    return args


@dataclass
class DemonstrationFile:
    path: Path
    transition_count: int
    metadata_json: str | None
    selected_transition_indices: np.ndarray
    total_transition_count: int
    total_episode_count: int
    selected_episode_ids: list[int]
    selected_episode_count: int


@dataclass
class ResolvedDatasetSelection:
    dataset_arg: str
    dataset_paths: list[Path]
    dataset_dir: Path | None
    scenario_selection: str | None
    requested_scenario_ids: list[str]
    requested_scenario_paths: list[Path]
    resolved_scenario_ids: list[str]
    resolved_scenario_paths: list[Path]
    missing_demo_scenario_ids: list[str]


def resolve_dataset_paths(dataset_arg: str) -> list[Path]:
    candidate_path = Path(dataset_arg).expanduser()
    if candidate_path.exists():
        if candidate_path.is_dir():
            paths = sorted(path.resolve() for path in candidate_path.glob("*.npz"))
        else:
            paths = [candidate_path.resolve()]
    else:
        paths = sorted(Path(path).resolve() for path in glob.glob(dataset_arg))

    if not paths:
        raise FileNotFoundError(f"No demonstration datasets matched: {dataset_arg}")
    return paths


def scenario_name(scene_index: int) -> str:
    return f"train_env_{int(scene_index):03d}"


def parse_scenario_selection(selection_text: str) -> list[int]:
    tokens = [token for token in re.split(r"[\s,]+", str(selection_text).strip()) if token]
    if not tokens:
        raise ValueError("--scenario-selection cannot be empty.")

    selected_ids: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
        else:
            start = int(token)
            end = start
        if start <= 0 or end <= 0:
            raise ValueError(f"Scenario ids must be positive integers, got: {token!r}")
        if start > end:
            raise ValueError(f"Scenario range start must be <= end, got: {token!r}")
        for scene_index in range(start, end + 1):
            if scene_index not in seen:
                seen.add(scene_index)
                selected_ids.append(scene_index)
    return selected_ids


def resolve_scenario_json_dir(scenario_dir_arg: str) -> Path:
    scenario_dir = Path(scenario_dir_arg).expanduser().resolve()
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
    if not scenario_dir.is_dir():
        raise NotADirectoryError(f"Scenario directory is not a directory: {scenario_dir}")

    json_dir = scenario_dir / "json" if (scenario_dir / "json").is_dir() else scenario_dir
    if not any(json_dir.glob("train_env_*.json")):
        raise FileNotFoundError(
            "Scenario directory must either contain a 'json' subdirectory or direct "
            f"'train_env_*.json' files: {scenario_dir}"
        )
    return json_dir


def resolve_selected_scenario_paths(selection_text: str, scenario_dir_arg: str) -> tuple[list[str], list[Path]]:
    selected_indexes = parse_scenario_selection(selection_text)
    scenario_ids = [scenario_name(scene_index) for scene_index in selected_indexes]
    scenario_json_dir = resolve_scenario_json_dir(scenario_dir_arg)

    missing_scenarios: list[str] = []
    scenario_paths: list[Path] = []
    for scenario_id in scenario_ids:
        candidate_path = (scenario_json_dir / f"{scenario_id}.json").resolve()
        if not candidate_path.exists():
            missing_scenarios.append(scenario_id)
            continue
        scenario_paths.append(candidate_path)

    if missing_scenarios:
        raise FileNotFoundError(
            "The requested scenario selection references missing scenario JSON files under "
            f"{scenario_json_dir}: {', '.join(missing_scenarios)}"
        )
    return scenario_ids, scenario_paths


def extract_scenario_index_from_demo_path(dataset_path: Path) -> int | None:
    match = SCENARIO_ID_PATTERN.search(dataset_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def resolve_dataset_selection(args: argparse.Namespace) -> ResolvedDatasetSelection:
    if args.dataset is not None:
        dataset_paths = resolve_dataset_paths(args.dataset)
        dataset_arg = str(Path(args.dataset).resolve()) if Path(args.dataset).exists() else args.dataset
        return ResolvedDatasetSelection(
            dataset_arg=dataset_arg,
            dataset_paths=dataset_paths,
            dataset_dir=None,
            scenario_selection=None,
            requested_scenario_ids=[],
            requested_scenario_paths=[],
            resolved_scenario_ids=[],
            resolved_scenario_paths=[],
            missing_demo_scenario_ids=[],
        )

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory is not a directory: {dataset_dir}")

    requested_scenario_ids, requested_scenario_paths = resolve_selected_scenario_paths(
        args.scenario_selection,
        args.scenario_dir,
    )

    per_scenario_demo_paths: dict[int, list[Path]] = {}
    for dataset_path in sorted(dataset_dir.glob("*.npz")):
        scene_index = extract_scenario_index_from_demo_path(dataset_path)
        if scene_index is None:
            continue
        per_scenario_demo_paths.setdefault(scene_index, []).append(dataset_path.resolve())

    if not per_scenario_demo_paths:
        raise FileNotFoundError(
            f"No per-scenario demo files named like '*train_env_###*.npz' were found in {dataset_dir}"
        )

    duplicate_matches: dict[str, list[str]] = {}
    missing_demo_scenario_ids: list[str] = []
    resolved_scenario_ids: list[str] = []
    resolved_scenario_paths: list[Path] = []
    dataset_paths: list[Path] = []
    for scenario_id, scenario_path in zip(requested_scenario_ids, requested_scenario_paths):
        scene_index = int(scenario_id.rsplit("_", 1)[-1])
        matches = per_scenario_demo_paths.get(scene_index, [])
        if not matches:
            missing_demo_scenario_ids.append(scenario_id)
            continue
        if len(matches) > 1:
            duplicate_matches[scenario_id] = [str(path) for path in matches]
            continue
        dataset_paths.append(matches[0])
        resolved_scenario_ids.append(scenario_id)
        resolved_scenario_paths.append(scenario_path)

    if duplicate_matches:
        duplicate_summary = "; ".join(
            f"{scenario_id}: {', '.join(paths)}" for scenario_id, paths in sorted(duplicate_matches.items())
        )
        raise RuntimeError(
            "Multiple demo files matched the same requested scene. Keep only one demo file per scene or "
            f"move alternates elsewhere. Conflicts: {duplicate_summary}"
        )

    if missing_demo_scenario_ids and not args.allow_missing_demos:
        raise FileNotFoundError(
            f"Missing demo files for {len(missing_demo_scenario_ids)} requested scenes under {dataset_dir}: "
            f"{', '.join(missing_demo_scenario_ids)}"
        )

    if not dataset_paths:
        raise RuntimeError(
            f"No demo files remain after resolving selection '{args.scenario_selection}' from {dataset_dir}"
        )

    dataset_arg = f"{dataset_dir} [scenario_selection={args.scenario_selection}]"
    return ResolvedDatasetSelection(
        dataset_arg=dataset_arg,
        dataset_paths=dataset_paths,
        dataset_dir=dataset_dir,
        scenario_selection=str(args.scenario_selection),
        requested_scenario_ids=requested_scenario_ids,
        requested_scenario_paths=requested_scenario_paths,
        resolved_scenario_ids=resolved_scenario_ids,
        resolved_scenario_paths=resolved_scenario_paths,
        missing_demo_scenario_ids=missing_demo_scenario_ids,
    )


def _ordered_episode_ids(episode_ids: np.ndarray) -> list[int]:
    ordered_ids: list[int] = []
    seen: set[int] = set()
    for raw_episode_id in episode_ids.tolist():
        episode_id = int(raw_episode_id)
        if episode_id in seen:
            continue
        seen.add(episode_id)
        ordered_ids.append(episode_id)
    return ordered_ids


def inspect_dataset_file(dataset_path: Path, *, episodes_per_scene: int = 0) -> DemonstrationFile:
    with np.load(dataset_path, allow_pickle=False) as payload:
        required_keys = ("obs_image", "obs_imu", "action")
        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            raise KeyError(f"Dataset is missing required keys: {missing_keys} in {dataset_path}")
        total_transition_count = int(payload["action"].shape[0])
        metadata_json = str(payload["metadata_json"].item()) if "metadata_json" in payload else None
        if total_transition_count <= 0:
            raise RuntimeError(f"Dataset contains no transitions: {dataset_path}")

        if "episode_id" in payload:
            episode_ids = np.asarray(payload["episode_id"], dtype=np.int64).reshape(-1)
            if int(episode_ids.shape[0]) != total_transition_count:
                raise ValueError(
                    f"episode_id length mismatch in {dataset_path}: "
                    f"expected {total_transition_count}, got {int(episode_ids.shape[0])}"
                )
            ordered_episode_ids = _ordered_episode_ids(episode_ids)
        else:
            episode_ids = np.zeros((total_transition_count,), dtype=np.int64)
            ordered_episode_ids = [0]

        if episodes_per_scene > 0:
            selected_episode_ids = ordered_episode_ids[: int(episodes_per_scene)]
        else:
            selected_episode_ids = list(ordered_episode_ids)

        if not selected_episode_ids:
            raise RuntimeError(f"No demo episodes remain after filtering {dataset_path}")

        selected_transition_indices = np.flatnonzero(
            np.isin(episode_ids, np.asarray(selected_episode_ids, dtype=np.int64))
        ).astype(np.int64, copy=False)
        transition_count = int(selected_transition_indices.shape[0])
        if transition_count <= 0:
            raise RuntimeError(f"No transitions remain after filtering {dataset_path}")

    return DemonstrationFile(
        path=dataset_path,
        transition_count=transition_count,
        metadata_json=metadata_json,
        selected_transition_indices=selected_transition_indices,
        total_transition_count=total_transition_count,
        total_episode_count=len(ordered_episode_ids),
        selected_episode_ids=selected_episode_ids,
        selected_episode_count=len(selected_episode_ids),
    )


class DemonstrationIterableDataset(IterableDataset):
    def __init__(
        self,
        files: list[DemonstrationFile],
        *,
        val_fraction: float,
        seed: int,
        split: str,
    ) -> None:
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")
        if not 0.0 <= float(val_fraction) < 1.0:
            raise ValueError("--val-fraction must be within [0, 1).")

        self.files = files
        self.split = split
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.epoch = 0
        self._split_indices: list[np.ndarray] = []
        self._length = 0

        rng = np.random.RandomState(self.seed)
        for file_info in self.files:
            count = int(file_info.transition_count)
            if count <= 0:
                selected = np.empty((0,), dtype=np.int64)
            elif self.val_fraction <= 0.0 or count == 1:
                selected = np.arange(count, dtype=np.int64) if self.split == "train" else np.empty((0,), dtype=np.int64)
            else:
                indices = np.arange(count, dtype=np.int64)
                rng.shuffle(indices)
                val_size = max(1, int(round(count * self.val_fraction)))
                if val_size >= count:
                    val_size = count - 1
                selected = indices[:val_size] if self.split == "val" else indices[val_size:]
            self._split_indices.append(selected)
            self._length += int(selected.shape[0])

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        file_order = np.arange(len(self.files), dtype=np.int64)
        epoch_rng = np.random.RandomState(self.seed + max(0, self.epoch) + (0 if self.split == "train" else 10_000))
        if self.split == "train":
            epoch_rng.shuffle(file_order)

        for file_idx in file_order.tolist():
            indices = self._split_indices[file_idx]
            if indices.size == 0:
                continue

            local_indices = indices.copy()
            if self.split == "train":
                epoch_rng.shuffle(local_indices)

            with np.load(self.files[file_idx].path, allow_pickle=False) as payload:
                obs_image = payload["obs_image"]
                obs_imu = payload["obs_imu"]
                action = payload["action"]
                selected_transition_indices = self.files[file_idx].selected_transition_indices
                actual_indices = selected_transition_indices[local_indices]
                for sample_idx in actual_indices.tolist():
                    yield {
                        "image": torch.from_numpy(obs_image[sample_idx]),
                        "imu": torch.from_numpy(obs_imu[sample_idx]),
                        "action": torch.from_numpy(action[sample_idx]),
                    }


def select_device(requested_device: str) -> str:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA device, but CUDA is unavailable. Falling back to CPU.")
        return "cpu"
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def make_loader(dataset: DemonstrationIterableDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=False)


def batch_obs_to_device(batch: dict[str, torch.Tensor], device: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    obs = {
        "image": batch["image"].permute(0, 3, 1, 2).contiguous().to(device=device, dtype=torch.uint8),
        "imu": batch["imu"].to(device=device, dtype=torch.float32),
    }
    action = batch["action"].to(device=device, dtype=torch.float32)
    return obs, action


def evaluate_loader(policy, data_loader: DataLoader, device: str) -> float:
    if len(data_loader.dataset) == 0:
        return 0.0

    policy.set_training_mode(False)
    losses: list[float] = []
    with torch.no_grad():
        for batch in data_loader:
            obs, expert_action = batch_obs_to_device(batch, device)
            predicted_action = policy.get_distribution(obs).distribution.mean
            loss = F.mse_loss(predicted_action, expert_action)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def resolve_optional_checkpoint(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix.lower() not in {".pth", ".pt"}:
        raise ValueError(f"--resume-actor must point to a .pth or .pt file, got: {path}")
    return path


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_bc_checkpoint(
    *,
    model,
    optimizer,
    config,
    output_dir: Path,
    model_name: str,
    epoch: int,
    total_epochs: int,
    dataset_arg: str,
    dataset_files: list[DemonstrationFile],
    best_reference_loss: float,
    train_loss: float,
    val_loss: float,
    selection_metadata: dict[str, object] | None,
    demo_episodes_processed: int,
    checkpoint_triggers: list[str],
) -> tuple[Path, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epoch_width = max(3, len(str(max(1, int(total_epochs)))))
    checkpoint_stem = checkpoint_dir / f"{model_name}_epoch{int(epoch):0{epoch_width}d}"
    actor_path = checkpoint_stem.with_name(f"{checkpoint_stem.name}_actor.pth")
    model_path = checkpoint_stem.with_suffix(".zip")

    torch.save(
        {
            "actor_state_dict": actor_state_dict_from_policy(model.policy),
            "config": config_to_dict(config),
            "dataset_path": dataset_arg,
            "dataset_files": [str(file.path) for file in dataset_files],
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_reference_loss": float(best_reference_loss),
            "demo_episodes_processed": int(demo_episodes_processed),
            "checkpoint_triggers": list(checkpoint_triggers),
            "optimizer_state_dict": optimizer.state_dict(),
            **(selection_metadata or {}),
        },
        actor_path,
    )
    model.save(str(model_path))
    return actor_path, model_path


def main() -> None:
    args = parse_args()
    config = make_config()
    if args.xml_path is not None:
        config.env.model.xml_path = str(Path(args.xml_path).resolve())

    device = select_device(args.device)
    resume_actor_path = resolve_optional_checkpoint(args.resume_actor)
    dataset_selection = resolve_dataset_selection(args)
    dataset_paths = dataset_selection.dataset_paths
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_actor_path = output_dir / f"{args.model_name}_actor.pth"
    bc_model_path = output_dir / f"{args.model_name}.zip"
    metrics_path = output_dir / f"{args.model_name}_metrics.json"
    epoch_metrics_path = output_dir / f"{args.model_name}_epoch_metrics.csv"

    dataset_files = [inspect_dataset_file(path, episodes_per_scene=int(args.episodes_per_scene)) for path in dataset_paths]
    total_transitions = sum(file.transition_count for file in dataset_files)
    if total_transitions <= 0:
        raise RuntimeError(f"Dataset contains no transitions: {dataset_selection.dataset_arg}")
    total_demo_episodes = sum(file.total_episode_count for file in dataset_files)
    selected_demo_episodes = sum(file.selected_episode_count for file in dataset_files)
    if selected_demo_episodes <= 0:
        raise RuntimeError(f"Dataset contains no selected demo episodes: {dataset_selection.dataset_arg}")

    selection_metadata: dict[str, object] | None = None
    if dataset_selection.scenario_selection is not None:
        selection_metadata = {
            "dataset_dir": None if dataset_selection.dataset_dir is None else str(dataset_selection.dataset_dir),
            "scenario_selection": dataset_selection.scenario_selection,
            "episodes_per_scene": int(args.episodes_per_scene),
            "requested_scenario_ids": dataset_selection.requested_scenario_ids,
            "requested_scenario_paths": [str(path) for path in dataset_selection.requested_scenario_paths],
            "resolved_scenario_ids": dataset_selection.resolved_scenario_ids,
            "resolved_scenario_paths": [str(path) for path in dataset_selection.resolved_scenario_paths],
            "missing_demo_scenario_ids": dataset_selection.missing_demo_scenario_ids,
        }
        print(
            "Resolved "
            f"{len(dataset_selection.resolved_scenario_ids)}/{len(dataset_selection.requested_scenario_ids)} "
            f"requested scenes into demo files from {dataset_selection.dataset_dir}"
        )
        if int(args.episodes_per_scene) > 0:
            print(
                "Per-scene demo limit enabled: "
                f"using the first {int(args.episodes_per_scene)} episode(s) from each selected scene"
            )
        if dataset_selection.missing_demo_scenario_ids:
            print(
                "Skipping scenes without matching demos because --allow-missing-demos was set: "
                f"{', '.join(dataset_selection.missing_demo_scenario_ids)}"
            )

    train_dataset = DemonstrationIterableDataset(
        dataset_files,
        val_fraction=args.val_fraction,
        seed=args.seed,
        split="train",
    )
    val_dataset = DemonstrationIterableDataset(
        dataset_files,
        val_fraction=args.val_fraction,
        seed=args.seed,
        split="val",
    )
    train_loader = make_loader(train_dataset, args.batch_size)
    val_loader = make_loader(val_dataset, args.batch_size) if len(val_dataset) > 0 else None

    env = build_vec_env(config, num_envs=1, seed=args.seed)
    model = build_ppo_model(config, env, device=device, seed=args.seed, verbose=0)
    optimizer = torch.optim.Adam(
        actor_parameters(model.policy),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    existing_metrics_payload: dict[str, object] = {}
    if resume_actor_path is not None and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            loaded_metrics = json.load(metrics_file)
        if isinstance(loaded_metrics, dict):
            existing_metrics_payload = loaded_metrics

    metrics: dict[str, object] = {
        "dataset_path": dataset_selection.dataset_arg,
        "dataset_files": [str(file.path) for file in dataset_files],
        "dataset_file_count": len(dataset_files),
        "episodes_per_scene_limit": int(args.episodes_per_scene),
        "save_every_epochs": int(args.save_every_epochs),
        "save_every_demo_episodes": int(args.save_every_demo_episodes),
        "selected_transition_count": int(total_transitions),
        "total_transition_count": int(sum(file.total_transition_count for file in dataset_files)),
        "selected_demo_episode_count": int(selected_demo_episodes),
        "total_demo_episode_count": int(total_demo_episodes),
        "resume_actor_path": None if resume_actor_path is None else str(resume_actor_path),
        "device": device,
        "metrics_path": str(metrics_path),
        "epoch_metrics_path": str(epoch_metrics_path),
        "epoch_metrics": list(existing_metrics_payload.get("epoch_metrics", []))
        if isinstance(existing_metrics_payload.get("epoch_metrics"), list)
        else [],
        "checkpoints": list(existing_metrics_payload.get("checkpoints", []))
        if isinstance(existing_metrics_payload.get("checkpoints"), list)
        else [],
        "train_transition_count": len(train_dataset),
        "val_transition_count": 0 if val_dataset is None else len(val_dataset),
        "dataset_file_summaries": [
            {
                "path": str(file.path),
                "selected_transition_count": int(file.transition_count),
                "total_transition_count": int(file.total_transition_count),
                "selected_episode_count": int(file.selected_episode_count),
                "total_episode_count": int(file.total_episode_count),
                "selected_episode_ids": [int(episode_id) for episode_id in file.selected_episode_ids],
            }
            for file in dataset_files
        ],
    }
    if selection_metadata is not None:
        metrics.update(selection_metadata)
    dataset_metadata = []
    for file_info in dataset_files:
        if file_info.metadata_json is None:
            continue
        dataset_metadata.append(json.loads(file_info.metadata_json))
    if dataset_metadata:
        metrics["dataset_metadata"] = dataset_metadata

    start_epoch = 0
    best_reference_loss = float("inf")
    best_actor_state: dict[str, torch.Tensor] | None = None
    if resume_actor_path is not None:
        resume_payload = torch.load(resume_actor_path, map_location="cpu")
        actor_state_dict = resume_payload.get("actor_state_dict")
        if not isinstance(actor_state_dict, dict):
            raise KeyError(f"Resume checkpoint does not contain 'actor_state_dict': {resume_actor_path}")
        loaded_keys = load_actor_state_dict(model.policy, actor_state_dict)
        if not loaded_keys:
            raise RuntimeError(f"No actor parameters were loaded from checkpoint: {resume_actor_path}")
        start_epoch = int(resume_payload.get("epoch", 0))
        best_reference_loss = float(resume_payload.get("best_reference_loss", float("inf")))
        best_actor_state = actor_state_dict_from_policy(model.policy)
        optimizer_state_dict = resume_payload.get("optimizer_state_dict")
        if isinstance(optimizer_state_dict, dict):
            optimizer.load_state_dict(optimizer_state_dict)
            move_optimizer_state_to_device(optimizer, device)
        metrics["resume_epoch"] = start_epoch
        if int(args.epochs) <= start_epoch:
            raise ValueError(
                f"--epochs must be greater than resume epoch ({start_epoch}); got {int(args.epochs)}.",
            )

    epoch_metrics_resume_mode = resume_actor_path is not None and epoch_metrics_path.exists()
    epoch_metrics_mode = "a" if epoch_metrics_resume_mode else "w"
    epoch_metrics_write_header = True
    if epoch_metrics_resume_mode and epoch_metrics_path.stat().st_size > 0:
        epoch_metrics_write_header = False

    epoch_metrics_file = epoch_metrics_path.open(epoch_metrics_mode, newline="", encoding="utf-8")
    epoch_metrics_writer = csv.DictWriter(epoch_metrics_file, fieldnames=EPOCH_METRICS_FIELDNAMES)
    if epoch_metrics_write_header:
        epoch_metrics_writer.writeheader()
        epoch_metrics_file.flush()

    try:
        for epoch in range(start_epoch + 1, int(args.epochs) + 1):
            train_dataset.set_epoch(epoch - 1)
            if val_dataset is not None:
                val_dataset.set_epoch(epoch - 1)
            model.policy.set_training_mode(True)
            train_losses: list[float] = []
            for batch in train_loader:
                obs, expert_action = batch_obs_to_device(batch, device)
                predicted_action = model.policy.get_distribution(obs).distribution.mean
                loss = F.mse_loss(predicted_action, expert_action)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            val_loss = evaluate_loader(model.policy, val_loader, device) if val_loader is not None else train_loss
            epoch_metric_row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "demo_episodes_processed": int(epoch * selected_demo_episodes),
            }
            metrics["epoch_metrics"].append(epoch_metric_row)
            epoch_metrics_writer.writerow(epoch_metric_row)
            epoch_metrics_file.flush()
            print(f"Epoch {epoch}/{int(args.epochs)}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            reference_loss = val_loss if val_loader is not None else train_loss
            if reference_loss < best_reference_loss:
                best_reference_loss = reference_loss
                best_actor_state = actor_state_dict_from_policy(model.policy)

            checkpoint_triggers: list[str] = []
            save_every_epochs = int(args.save_every_epochs)
            if save_every_epochs > 0 and epoch % save_every_epochs == 0:
                checkpoint_triggers.append(f"epoch_multiple:{save_every_epochs}")

            save_every_demo_episodes = int(args.save_every_demo_episodes)
            if save_every_demo_episodes > 0:
                previous_demo_episodes = int((epoch - 1) * selected_demo_episodes)
                current_demo_episodes = int(epoch * selected_demo_episodes)
                previous_block = previous_demo_episodes // save_every_demo_episodes
                current_block = current_demo_episodes // save_every_demo_episodes
                if current_block > previous_block:
                    checkpoint_triggers.append(f"demo_episode_multiple:{save_every_demo_episodes}")

            if checkpoint_triggers:
                current_demo_episodes = int(epoch * selected_demo_episodes)
                checkpoint_actor_path, checkpoint_model_path = save_bc_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    output_dir=output_dir,
                    model_name=args.model_name,
                    epoch=epoch,
                    total_epochs=int(args.epochs),
                    dataset_arg=dataset_selection.dataset_arg,
                    dataset_files=dataset_files,
                    best_reference_loss=best_reference_loss,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    selection_metadata=selection_metadata,
                    demo_episodes_processed=current_demo_episodes,
                    checkpoint_triggers=checkpoint_triggers,
                )
                metrics["checkpoints"].append(
                    {
                        "epoch": int(epoch),
                        "demo_episodes_processed": int(current_demo_episodes),
                        "checkpoint_triggers": list(checkpoint_triggers),
                        "actor_path": str(checkpoint_actor_path),
                        "model_path": str(checkpoint_model_path),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                    }
                )
                print(
                    f"Saved checkpoint at epoch {epoch}: "
                    f"{checkpoint_model_path.name}, {checkpoint_actor_path.name} "
                    f"(processed_demo_episodes={current_demo_episodes}, triggers={', '.join(checkpoint_triggers)})",
                )
            write_json(metrics_path, metrics)
    finally:
        epoch_metrics_file.close()
        env.close()

    if best_actor_state is None:
        best_actor_state = actor_state_dict_from_policy(model.policy)

    torch.save(
        {
            "actor_state_dict": best_actor_state,
            "config": config_to_dict(config),
            "dataset_path": dataset_selection.dataset_arg,
            "dataset_files": [str(file.path) for file in dataset_files],
            "epoch": int(args.epochs),
            "best_reference_loss": float(best_reference_loss),
            "episodes_per_scene_limit": int(args.episodes_per_scene),
            "selected_demo_episode_count": int(selected_demo_episodes),
            **(selection_metadata or {}),
        },
        best_actor_path,
    )
    load_actor_state_dict(model.policy, best_actor_state)
    model.save(str(bc_model_path))

    metrics["best_reference_loss"] = float(best_reference_loss)
    metrics["demo_episodes_processed_total"] = int(int(args.epochs) * selected_demo_episodes)
    metrics["bc_model_path"] = str(bc_model_path)
    metrics["bc_actor_path"] = str(best_actor_path)
    write_json(metrics_path, metrics)

    print(f"Saved BC model to {bc_model_path}")
    print(f"Saved BC actor weights to {best_actor_path}")
    print(f"Saved BC metrics to {metrics_path}")


if __name__ == "__main__":
    main()
