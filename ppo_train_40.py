from __future__ import annotations

import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNNER_PATH = PROJECT_ROOT / "ppo_train_40" / "run.py"


if __name__ == "__main__":
    runpy.run_path(str(RUNNER_PATH), run_name="__main__")
