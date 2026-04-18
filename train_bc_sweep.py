from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from configs.default_config import PROJECT_ROOT


DEFAULT_SELECTIONS = [
    "1-20,60-80",
    "1-20,60-100",
    "1-40,60-100",
    "1-100",
]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "runs" / "bc_demos_large_pool_100"
DEFAULT_SCENARIO_DIR = PROJECT_ROOT / "scenarios" / "large_pool_dataset_200" / "train"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "bc_pretrain" / "large_pool_range_sweep"
SCENARIO_ID_PATTERN = re.compile(r"train_env_(\d{3})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple BC runs over large-pool scene-range selections.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR.resolve()),
        help="Directory containing per-scene BC demo .npz files.",
    )
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=str(DEFAULT_SCENARIO_DIR.resolve()),
        help="Large-pool train split root or json directory used to validate scene selections.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT.resolve()),
        help="Root directory containing one output subdirectory per BC run.",
    )
    parser.add_argument(
        "--selection",
        action="append",
        default=None,
        help="Repeat to override the default five experiments. Example: --selection 1-20 --selection 1-20,60-80",
    )
    parser.add_argument("--model-prefix", type=str, default="bc_large_pool", help="Model-name prefix per run.")
    parser.add_argument("--epochs", type=int, default=20, help="BC epochs passed through to train_bc.py.")
    parser.add_argument("--batch-size", type=int, default=64, help="BC mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Actor optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument(
        "--episodes-per-scene",
        type=int,
        default=1,
        help="Limit each selected scene to the first N demo episodes. Set 0 to use all per-scene episodes.",
    )
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=2,
        help="Save intermediate checkpoints every N epochs. Defaults to 2. Set 0 to disable.",
    )
    parser.add_argument(
        "--save-every-demo-episodes",
        type=int,
        default=0,
        help="Save intermediate checkpoints whenever cumulative selected demo episodes reaches a multiple of N. Set 0 to disable.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction passed to train_bc.py.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed passed to train_bc.py.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device passed to train_bc.py.")
    parser.add_argument("--xml-path", type=str, default=None, help="Optional MuJoCo XML override.")
    parser.add_argument("--resume-actor", type=str, default=None, help="Optional actor checkpoint used to resume each run.")
    parser.add_argument(
        "--allow-missing-demos",
        action="store_true",
        help="Pass through to train_bc.py so missing selected demos are skipped instead of failing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write the sweep manifest; do not launch training.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep launching later experiments after a precheck or training failure.",
    )
    return parser.parse_args()


def scenario_name(scene_index: int) -> str:
    return f"train_env_{int(scene_index):03d}"


def parse_selection(selection_text: str) -> list[int]:
    tokens = [token for token in re.split(r"[\s,]+", str(selection_text).strip()) if token]
    if not tokens:
        raise ValueError("Selection cannot be empty.")

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
            raise ValueError(f"Scene ids must be positive integers, got: {token!r}")
        if start > end:
            raise ValueError(f"Scene range start must be <= end, got: {token!r}")
        for scene_index in range(start, end + 1):
            if scene_index not in seen:
                seen.add(scene_index)
                selected_ids.append(scene_index)
    return selected_ids


def compact_selection_slug(scene_ids: list[int]) -> str:
    if not scene_ids:
        raise ValueError("Selection slug requires at least one scene id.")

    ranges: list[tuple[int, int]] = []
    start = scene_ids[0]
    end = start
    for scene_index in scene_ids[1:]:
        if scene_index == end + 1:
            end = scene_index
            continue
        ranges.append((start, end))
        start = scene_index
        end = scene_index
    ranges.append((start, end))
    return "__".join(
        f"{range_start:03d}" if range_start == range_end else f"{range_start:03d}-{range_end:03d}"
        for range_start, range_end in ranges
    )


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


def collect_available_scene_ids(path: Path, suffix: str) -> set[int]:
    available_ids: set[int] = set()
    for candidate in sorted(path.glob(f"*{suffix}")):
        match = SCENARIO_ID_PATTERN.search(candidate.stem)
        if match is None:
            continue
        available_ids.add(int(match.group(1)))
    return available_ids


def build_train_command(
    *,
    selection: str,
    output_dir: Path,
    model_name: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str((PROJECT_ROOT / "train_bc.py").resolve()),
        "--dataset-dir",
        str(Path(args.dataset_dir).expanduser().resolve()),
        "--scenario-selection",
        selection,
        "--scenario-dir",
        str(Path(args.scenario_dir).expanduser().resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--model-name",
        model_name,
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--episodes-per-scene",
        str(int(args.episodes_per_scene)),
        "--save-every-epochs",
        str(int(args.save_every_epochs)),
        "--save-every-demo-episodes",
        str(int(args.save_every_demo_episodes)),
        "--val-fraction",
        str(float(args.val_fraction)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
    ]
    if args.xml_path is not None:
        command.extend(["--xml-path", str(Path(args.xml_path).expanduser().resolve())])
    if args.resume_actor is not None:
        command.extend(["--resume-actor", str(Path(args.resume_actor).expanduser().resolve())])
    if args.allow_missing_demos:
        command.append("--allow-missing-demos")
    return command


def write_manifest(manifest_path: Path, payload: dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(payload, manifest_file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory is not a directory: {dataset_dir}")
    scenario_json_dir = resolve_scenario_json_dir(args.scenario_dir)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selections = list(args.selection or DEFAULT_SELECTIONS)
    available_demo_ids = collect_available_scene_ids(dataset_dir, ".npz")
    available_scenario_ids = collect_available_scene_ids(scenario_json_dir, ".json")
    manifest_path = output_root / "sweep_manifest.json"
    manifest: dict[str, object] = {
        "project_root": str(PROJECT_ROOT.resolve()),
        "dataset_dir": str(dataset_dir),
        "scenario_json_dir": str(scenario_json_dir),
        "output_root": str(output_root),
        "allow_missing_demos": bool(args.allow_missing_demos),
        "episodes_per_scene": int(args.episodes_per_scene),
        "save_every_demo_episodes": int(args.save_every_demo_episodes),
        "dry_run": bool(args.dry_run),
        "selections": selections,
        "available_demo_scene_count": len(available_demo_ids),
        "available_scenario_count": len(available_scenario_ids),
        "experiments": [],
    }

    overall_success = True
    for selection in selections:
        scene_ids = parse_selection(selection)
        slug = compact_selection_slug(scene_ids)
        output_dir = output_root / slug
        model_name = f"{args.model_prefix}_{slug.replace('-', '_')}"
        requested_scene_names = [scenario_name(scene_index) for scene_index in scene_ids]
        missing_scenario_names = [
            scenario_name(scene_index) for scene_index in scene_ids if scene_index not in available_scenario_ids
        ]
        missing_demo_names = [scenario_name(scene_index) for scene_index in scene_ids if scene_index not in available_demo_ids]
        command = build_train_command(
            selection=selection,
            output_dir=output_dir,
            model_name=model_name,
            args=args,
        )
        entry: dict[str, object] = {
            "selection": selection,
            "slug": slug,
            "model_name": model_name,
            "output_dir": str(output_dir),
            "requested_scene_count": len(scene_ids),
            "requested_scene_ids": requested_scene_names,
            "missing_scenario_ids": missing_scenario_names,
            "missing_demo_ids": missing_demo_names,
            "command": command,
        }

        precheck_errors: list[str] = []
        if missing_scenario_names:
            precheck_errors.append(f"missing scenarios: {', '.join(missing_scenario_names)}")
        if missing_demo_names and not args.allow_missing_demos:
            precheck_errors.append(f"missing demos: {', '.join(missing_demo_names)}")

        if args.dry_run:
            entry["status"] = "blocked" if precheck_errors else "planned"
            if precheck_errors:
                entry["precheck_errors"] = precheck_errors
            manifest["experiments"].append(entry)
            write_manifest(manifest_path, manifest)
            continue

        if precheck_errors:
            entry["status"] = "blocked"
            entry["precheck_errors"] = precheck_errors
            manifest["experiments"].append(entry)
            write_manifest(manifest_path, manifest)
            overall_success = False
            print(f"[{selection}] precheck failed: {'; '.join(precheck_errors)}")
            if not args.continue_on_error:
                raise SystemExit(1)
            continue

        print(f"[{selection}] launching BC training into {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        entry["status"] = "completed" if completed.returncode == 0 else "failed"
        entry["returncode"] = int(completed.returncode)
        manifest["experiments"].append(entry)
        write_manifest(manifest_path, manifest)
        if completed.returncode != 0:
            overall_success = False
            if not args.continue_on_error:
                raise SystemExit(int(completed.returncode))

    print(f"Sweep manifest saved to {manifest_path}")
    if not overall_success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
