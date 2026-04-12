from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from configs.default_config import config_to_dict, make_config
from utils.policy_utils import actor_parameters, actor_state_dict_from_policy, build_ppo_model, build_vec_env, load_actor_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the PPO actor with behavior cloning.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the collected .npz demonstration dataset.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for BC checkpoints and metrics.")
    parser.add_argument("--model-name", type=str, default="bc_fish", help="Output checkpoint stem.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of BC epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="BC mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Actor optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optional Adam weight decay.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction in [0, 1).")
    parser.add_argument("--seed", type=int, default=7, help="Dataset split and model seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device: cuda, cuda:0, cpu, or auto.")
    parser.add_argument("--xml-path", type=str, default=None, help="Override MuJoCo XML scene path.")
    return parser.parse_args()


class DemonstrationDataset(Dataset):
    def __init__(self, dataset_path: Path) -> None:
        payload = np.load(dataset_path, allow_pickle=False)
        required_keys = (
            "obs_image",
            "obs_imu",
            "action",
            "next_obs_image",
            "next_obs_imu",
            "reward",
            "done",
            "success",
            "episode_id",
        )
        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            raise KeyError(f"Dataset is missing required keys: {missing_keys}")

        self.dataset_path = dataset_path
        self.obs_image = payload["obs_image"].astype(np.uint8, copy=False)
        self.obs_imu = payload["obs_imu"].astype(np.float32, copy=False)
        self.action = payload["action"].astype(np.float32, copy=False)
        self.next_obs_image = payload["next_obs_image"].astype(np.uint8, copy=False)
        self.next_obs_imu = payload["next_obs_imu"].astype(np.float32, copy=False)
        self.reward = payload["reward"].astype(np.float32, copy=False)
        self.done = payload["done"].astype(bool, copy=False)
        self.success = payload["success"].astype(bool, copy=False)
        self.episode_id = payload["episode_id"].astype(np.int32, copy=False)
        self.metadata_json = str(payload["metadata_json"].item()) if "metadata_json" in payload else None

    def __len__(self) -> int:
        return int(self.action.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(self.obs_image[index]),
            "imu": torch.from_numpy(self.obs_imu[index]),
            "action": torch.from_numpy(self.action[index]),
        }


def select_device(requested_device: str) -> str:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA device, but CUDA is unavailable. Falling back to CPU.")
        return "cpu"
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def split_dataset(dataset: DemonstrationDataset, val_fraction: float, seed: int) -> tuple[Dataset, Dataset | None]:
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError("--val-fraction must be within [0, 1).")

    total_size = len(dataset)
    if total_size <= 1 or float(val_fraction) <= 0.0:
        return dataset, None

    rng = np.random.RandomState(seed)
    indices = np.arange(total_size, dtype=np.int64)
    rng.shuffle(indices)
    val_size = max(1, int(round(total_size * float(val_fraction))))
    if val_size >= total_size:
        val_size = total_size - 1
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    return Subset(dataset, train_indices.tolist()), Subset(dataset, val_indices.tolist())


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0, pin_memory=False)


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


def main() -> None:
    args = parse_args()
    config = make_config()
    if args.xml_path is not None:
        config.env.model.xml_path = str(Path(args.xml_path).resolve())

    device = select_device(args.device)
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DemonstrationDataset(dataset_path)
    if len(dataset) <= 0:
        raise RuntimeError(f"Dataset contains no transitions: {dataset_path}")
    train_dataset, val_dataset = split_dataset(dataset, args.val_fraction, args.seed)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False) if val_dataset is not None else None

    env = build_vec_env(config, num_envs=1, seed=args.seed)
    model = build_ppo_model(config, env, device=device, seed=args.seed, verbose=0)
    optimizer = torch.optim.Adam(
        actor_parameters(model.policy),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    metrics: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "device": device,
        "epoch_metrics": [],
        "train_transition_count": len(train_dataset),
        "val_transition_count": 0 if val_dataset is None else len(val_dataset),
    }
    if dataset.metadata_json is not None:
        metrics["dataset_metadata"] = json.loads(dataset.metadata_json)

    best_reference_loss = float("inf")
    best_actor_state: dict[str, torch.Tensor] | None = None

    try:
        for epoch in range(1, int(args.epochs) + 1):
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
            "dataset_path": str(dataset_path),
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
