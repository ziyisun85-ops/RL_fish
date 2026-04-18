from __future__ import annotations

import sys
from pathlib import Path

import train_bc
from train_bc_launcher_config import OUTPUT_ROOT, build_bc_wrapper_default_args


SCENARIO_SELECTION = "1-20,60-80"
OUTPUT_DIR = (Path(OUTPUT_ROOT) / "1_20__60_80").as_posix()
MODEL_NAME = "bc_large_pool_1_20__60_80"


DEFAULT_ARGS = build_bc_wrapper_default_args(
    scenario_selection=SCENARIO_SELECTION,
    output_dir=OUTPUT_DIR,
    model_name=MODEL_NAME,
)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    train_bc.main()
