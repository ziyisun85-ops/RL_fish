from __future__ import annotations

import os
import sys
import types
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)


def _load_patched_train_module():
    existing_module = sys.modules.get("train")
    if existing_module is not None:
        return existing_module

    train_path = PROJECT_ROOT / "train.py"
    source = train_path.read_text(encoding="utf-8")
    source = source.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"] = ""',
        '# bundle patch: keep CUDA_VISIBLE_DEVICES unchanged so auto can use GPU on the server',
    )

    module = types.ModuleType("train")
    module.__file__ = str(train_path)
    module.__package__ = ""
    sys.modules["train"] = module
    exec(compile(source, str(train_path), "exec"), module.__dict__)
    return module


train_module = _load_patched_train_module()

from launcher_config import parse_train_args
from ppo_runtime import DEFAULT_ARGS, install_runtime_patches
from train import main
from train_env_patch import install_train_and_env_patches


def main_entry() -> None:
    install_train_and_env_patches()
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    install_runtime_patches(parse_train_args(train_module, sys.argv))
    main()


if __name__ == "__main__":
    main_entry()
