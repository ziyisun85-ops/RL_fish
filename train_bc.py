from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from configs.default_config import config_to_dict, make_config
from utils.policy_utils import actor_parameters, actor_state_dict_from_policy, build_ppo_model, build_vec_env, load_actor_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the PPO actor with behavior cloning.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset source: one .npz file, a directory of .npz files, or a glob pattern.",
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
        default=0,
        help="Save intermediate BC checkpoints every N epochs. Set 0 to disable.",
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
    return parser.parse_args()


@dataclass
class DemonstrationFile:
    path: Path
    transition_count: int
    metadata_json: str | None


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


def inspect_dataset_file(dataset_path: Path) -> DemonstrationFile:
    with np.load(dataset_path, allow_pickle=False) as payload:
        required_keys = ("obs_image", "obs_imu", "action")
        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            raise KeyError(f"Dataset is missing required keys: {missing_keys} in {dataset_path}")
        transition_count = int(payload["action"].shape[0])
        metadata_json = str(payload["metadata_json"].item()) if "metadata_json" in payload else None
    return DemonstrationFile(
        path=dataset_path,
        transition_count=transition_count,
        metadata_json=metadata_json,
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
                for sample_idx in local_indices.tolist():
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
            "dataset_path": str(Path(dataset_arg).resolve()) if Path(dataset_arg).exists() else dataset_arg,
            "dataset_files": [str(file.path) for file in dataset_files],
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_reference_loss": float(best_reference_loss),
            "optimizer_state_dict": optimizer.state_dict(),
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
    dataset_paths = resolve_dataset_paths(args.dataset)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = [inspect_dataset_file(path) for path in dataset_paths]
    total_transitions = sum(file.transition_count for file in dataset_files)
    if total_transitions <= 0:
        raise RuntimeError(f"Dataset contains no transitions: {args.dataset}")

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

    metrics: dict[str, object] = {
        "dataset_path": str(Path(args.dataset).resolve()) if Path(args.dataset).exists() else args.dataset,
        "dataset_files": [str(file.path) for file in dataset_files],
        "dataset_file_count": len(dataset_files),
        "resume_actor_path": None if resume_actor_path is None else str(resume_actor_path),
        "device": device,
        "epoch_metrics": [],
        "checkpoints": [],
        "train_transition_count": len(train_dataset),
        "val_transition_count": 0 if val_dataset is None else len(val_dataset),
    }
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
            metrics["epoch_metrics"].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )
            print(f"Epoch {epoch}/{int(args.epochs)}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            reference_loss = val_loss if val_loader is not None else train_loss
            if reference_loss < best_reference_loss:
                best_reference_loss = reference_loss
                best_actor_state = actor_state_dict_from_policy(model.policy)

            save_every_epochs = int(args.save_every_epochs)
            if save_every_epochs > 0 and epoch % save_every_epochs == 0:
                checkpoint_actor_path, checkpoint_model_path = save_bc_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    output_dir=output_dir,
                    model_name=args.model_name,
                    epoch=epoch,
                    total_epochs=int(args.epochs),
                    dataset_arg=args.dataset,
                    dataset_files=dataset_files,
                    best_reference_loss=best_reference_loss,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
                metrics["checkpoints"].append(
                    {
                        "epoch": int(epoch),
                        "actor_path": str(checkpoint_actor_path),
                        "model_path": str(checkpoint_model_path),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                    }
                )
                print(
                    f"Saved checkpoint at epoch {epoch}: "
                    f"{checkpoint_model_path.name}, {checkpoint_actor_path.name}",
                )
    finally:
        env.close()

    if best_actor_state is None:
        best_actor_state = actor_state_dict_from_policy(model.policy)

    best_actor_path = output_dir / f"{args.model_name}_actor.pth"
    bc_model_path = output_dir / f"{args.model_name}.zip"
    metrics_path = output_dir / f"{args.model_name}_metrics.json"

    torch.save(
        {
            "actor_state_dict": best_actor_state,
            "config": config_to_dict(config),
            "dataset_path": str(Path(args.dataset).resolve()) if Path(args.dataset).exists() else args.dataset,
            "dataset_files": [str(file.path) for file in dataset_files],
            "epoch": int(args.epochs),
            "best_reference_loss": float(best_reference_loss),
        },
        best_actor_path,
    )
    load_actor_state_dict(model.policy, best_actor_state)
    model.save(str(bc_model_path))

    metrics["best_reference_loss"] = float(best_reference_loss)
    metrics["bc_model_path"] = str(bc_model_path)
    metrics["bc_actor_path"] = str(best_actor_path)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)

    print(f"Saved BC model to {bc_model_path}")
    print(f"Saved BC actor weights to {best_actor_path}")
    print(f"Saved BC metrics to {metrics_path}")


if __name__ == "__main__":
    main()
