from __future__ import annotations

from pathlib import Path


DEVICE = "auto"
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
VAL_FRACTION = 0.0
SEED = 7

SCENARIO_DIR = (Path("scenarios") / "large_pool_dataset_200" / "train").as_posix()
DATASET_DIR = (Path("runs") / "bc_demos_large_pool_100").as_posix()
OUTPUT_ROOT = (Path("runs") / "bc_pretrain" / "large_pool_scene_sets").as_posix()

EPISODES_PER_SCENE = 1
SAVE_EVERY_EPOCHS = 2
SAVE_EVERY_DEMO_EPISODES = 0
ALLOW_MISSING_DEMOS = False


def build_bc_wrapper_default_args(
    *,
    scenario_selection: str,
    output_dir: str,
    model_name: str,
) -> list[str]:
    args = [
        "--dataset-dir",
        DATASET_DIR,
        "--scenario-selection",
        str(scenario_selection),
        "--scenario-dir",
        SCENARIO_DIR,
        "--output-dir",
        str(output_dir),
        "--model-name",
        str(model_name),
        "--epochs",
        str(int(EPOCHS)),
        "--batch-size",
        str(int(BATCH_SIZE)),
        "--learning-rate",
        str(float(LEARNING_RATE)),
        "--weight-decay",
        str(float(WEIGHT_DECAY)),
        "--episodes-per-scene",
        str(int(EPISODES_PER_SCENE)),
        "--save-every-epochs",
        str(int(SAVE_EVERY_EPOCHS)),
        "--save-every-demo-episodes",
        str(int(SAVE_EVERY_DEMO_EPISODES)),
        "--val-fraction",
        str(float(VAL_FRACTION)),
        "--seed",
        str(int(SEED)),
        "--device",
        str(DEVICE),
    ]
    if ALLOW_MISSING_DEMOS:
        args.append("--allow-missing-demos")
    return args
