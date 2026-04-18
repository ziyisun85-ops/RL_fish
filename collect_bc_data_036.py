from __future__ import annotations

import sys
from pathlib import Path

import collect_bc_data


SCENARIO_PATH = (Path("scenarios") / "large_pool_dataset_200" / "train" / "json" / "train_env_036.json").as_posix()
OUTPUT_PATH = (Path("runs") / "bc_demos_large_pool_100" / "train_env_036_5eps.npz").as_posix()
EPISODES = 5
SEED = 7
VIEWER_SLOWDOWN = 0.1


DEFAULT_ARGS = [
    "--output",
    OUTPUT_PATH,
    "--episodes",
    str(EPISODES),
    "--seed",
    str(SEED),
    "--scenario-path",
    SCENARIO_PATH,
    "--viewer-slowdown",
    str(VIEWER_SLOWDOWN),
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    collect_bc_data.main()
