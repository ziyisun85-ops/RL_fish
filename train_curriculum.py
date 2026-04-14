from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from configs.default_config import make_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train scenarios sequentially. Each scenario must converge before the next one starts.",
    )
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=None,
        help="Directory containing exported scenario JSON files. Required unless --scenario-list is provided.",
    )
    parser.add_argument(
        "--scenario-glob",
        type=str,
        default="*.json",
        help="Glob pattern used inside --scenario-dir. The sorted match order defines the curriculum order.",
    )
    parser.add_argument(
        "--scenario-list",
        type=str,
        default=None,
        help="JSON file defining an explicit curriculum order. Supports a list of paths, a list of objects with scenario_path, or an object with a scenarios list.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional 1-based start position after sorting the matched scenarios.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Optional 1-based end position after sorting the matched scenarios.",
    )
    parser.add_argument(
        "--bc-weights",
        type=str,
        default=None,
        help="Initial BC actor checkpoint for the first scenario. Mutually exclusive with --resume-from.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Initial PPO .zip checkpoint for the first scenario. Mutually exclusive with --bc-weights.",
    )
    parser.add_argument("--timesteps", type=int, default=100_000_000, help="Timesteps budget passed to each train.py run.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments per scenario.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device passed through to train.py.")
    parser.add_argument(
        "--record-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable per-episode video capture during curriculum training.",
    )
    parser.add_argument(
        "--video-interval-episodes",
        type=int,
        default=20,
        help="Save videos and episode checkpoints every N completed episodes.",
    )
    parser.add_argument(
        "--max-episodes-per-scenario",
        type=int,
        default=0,
        help="Optional safety cap passed to train.py. Use 0 or a negative value to disable the cap.",
    )
    parser.add_argument(
        "--curriculum-name",
        type=str,
        default=None,
        help="Name used for the curriculum summary files and run IDs. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--convergence-window",
        type=int,
        default=40,
        help="Sliding window size in completed episodes used for per-scenario convergence stopping.",
    )
    parser.add_argument(
        "--convergence-min-episodes",
        type=int,
        default=40,
        help="Minimum total completed episodes before convergence can trigger.",
    )
    parser.add_argument(
        "--convergence-min-success-rate",
        type=float,
        default=0.90,
        help="Minimum success rate required in the convergence window.",
    )
    parser.add_argument(
        "--convergence-max-timeout-rate",
        type=float,
        default=0.10,
        help="Maximum timeout rate allowed in the convergence window.",
    )
    parser.add_argument(
        "--convergence-max-failure-rate",
        type=float,
        default=0.10,
        help="Maximum failure rate allowed in the convergence window. Failure means obstacle collision, wall collision, or out-of-bounds.",
    )
    parser.add_argument(
        "--convergence-reward-window",
        type=int,
        default=20,
        help="Reward comparison window used for reward stability checking. Set 0 to disable.",
    )
    parser.add_argument(
        "--convergence-reward-stability-ratio",
        type=float,
        default=0.05,
        help="Maximum relative change allowed between the previous and latest reward windows. Set a negative value to disable.",
    )
    parser.add_argument(
        "--stop-on-nonconverged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the curriculum immediately if a scenario ends without convergence.",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default=str((Path(__file__).resolve().parent / "train.py").resolve()),
        help="Path to train.py. Defaults to the project-local train.py file.",
    )
    return parser.parse_args()


def _default_curriculum_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_token(text: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in text)
    cleaned = cleaned.strip("_")
    return cleaned or _default_curriculum_name()


def resolve_initial_model(args: argparse.Namespace) -> tuple[str, Path]:
    if args.bc_weights is not None and args.resume_from is not None:
        raise ValueError("Use either --bc-weights or --resume-from, not both.")
    if args.bc_weights is None and args.resume_from is None:
        raise ValueError("One of --bc-weights or --resume-from is required.")

    if args.bc_weights is not None:
        path = Path(args.bc_weights).resolve()
        if not path.exists():
            raise FileNotFoundError(f"BC checkpoint not found: {path}")
        return "bc", path

    path = Path(args.resume_from).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return "ppo", path


def resolve_scenarios(args: argparse.Namespace) -> list[Path]:
    if args.scenario_list is not None:
        scenario_list_path = Path(args.scenario_list).resolve()
        if not scenario_list_path.exists():
            raise FileNotFoundError(f"Scenario list not found: {scenario_list_path}")

        payload = json.loads(scenario_list_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            scenario_items = payload.get("scenarios")
            if scenario_items is None:
                raise KeyError(f"Scenario list JSON must contain a 'scenarios' key: {scenario_list_path}")
        elif isinstance(payload, list):
            scenario_items = payload
        else:
            raise TypeError(f"Unsupported scenario list JSON format: {scenario_list_path}")

        matched: list[Path] = []
        base_dir = scenario_list_path.parent
        for item in scenario_items:
            if isinstance(item, str):
                raw_path = item
            elif isinstance(item, dict):
                raw_path = item.get("scenario_path") or item.get("path")
                if raw_path is None:
                    raise KeyError(f"Scenario list entry is missing scenario_path/path: {item}")
            else:
                raise TypeError(f"Unsupported scenario list entry: {item!r}")

            resolved = Path(raw_path)
            if not resolved.is_absolute():
                resolved = (base_dir / resolved).resolve()
            else:
                resolved = resolved.resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Scenario JSON from list does not exist: {resolved}")
            matched.append(resolved)
    else:
        if args.scenario_dir is None:
            raise ValueError("Provide either --scenario-dir or --scenario-list.")
        scenario_dir = Path(args.scenario_dir).resolve()
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

        matched = sorted(path.resolve() for path in scenario_dir.glob(args.scenario_glob) if path.is_file())
        if not matched:
            raise FileNotFoundError(f"No scenarios matched {args.scenario_glob!r} in {scenario_dir}")

    start_index = 1 if args.start_index is None else int(args.start_index)
    end_index = len(matched) if args.end_index is None else int(args.end_index)
    if start_index <= 0 or end_index <= 0:
        raise ValueError("--start-index and --end-index must be positive when provided.")
    if start_index > end_index:
        raise ValueError("--start-index cannot be greater than --end-index.")

    start_offset = max(0, start_index - 1)
    end_offset = min(len(matched), end_index)
    selected = matched[start_offset:end_offset]
    if not selected:
        raise RuntimeError("The requested scenario slice is empty.")
    return selected


def write_curriculum_summaries(json_path: Path, csv_path: Path, rows: list[dict[str, Any]]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)

    csv_fieldnames = [
        "curriculum_name",
        "scenario_order",
        "scenario_stem",
        "scenario_path",
        "run_id",
        "stop_reason",
        "converged",
        "episodes_completed_this_run",
        "episodes_completed_total",
        "num_timesteps",
        "mean_reward",
        "success_rate",
        "timeout_rate",
        "failure_rate",
        "latest_model_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in rows:
            convergence_summary = row.get("convergence_summary") or {}
            writer.writerow(
                {
                    "curriculum_name": row.get("curriculum_name", ""),
                    "scenario_order": row.get("scenario_order", 0),
                    "scenario_stem": row.get("scenario_stem", ""),
                    "scenario_path": row.get("scenario_path", ""),
                    "run_id": row.get("run_id", ""),
                    "stop_reason": row.get("stop_reason", ""),
                    "converged": row.get("converged", False),
                    "episodes_completed_this_run": row.get("episodes_completed_this_run", 0),
                    "episodes_completed_total": row.get("episodes_completed_total", 0),
                    "num_timesteps": row.get("num_timesteps", 0),
                    "mean_reward": convergence_summary.get("mean_reward", ""),
                    "success_rate": convergence_summary.get("success_rate", ""),
                    "timeout_rate": convergence_summary.get("timeout_rate", ""),
                    "failure_rate": convergence_summary.get("failure_rate", ""),
                    "latest_model_path": row.get("latest_model_path", ""),
                }
            )


def main() -> None:
    args = parse_args()
    initial_mode, current_model = resolve_initial_model(args)
    scenario_paths = resolve_scenarios(args)
    train_script = Path(args.train_script).resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"train.py not found: {train_script}")

    curriculum_name = _sanitize_token(args.curriculum_name or _default_curriculum_name())
    log_root = Path(make_config().train.log_dir).resolve()
    curriculum_json_path = log_root / f"curriculum_{curriculum_name}.json"
    curriculum_csv_path = log_root / f"curriculum_{curriculum_name}.csv"
    curriculum_rows: list[dict[str, Any]] = []

    if args.num_envs != 1:
        print(
            "Warning: curriculum training is cleaner with --num-envs 1. "
            "Multiple environments work, but convergence windows and episode-triggered videos become less interpretable."
        )

    if scenario_paths:
        scenario_roots_text = "\n".join(str(path).lower() for path in scenario_paths)
        if "\\test\\" in scenario_roots_text or scenario_roots_text.endswith("\\test") or "/test/" in scenario_roots_text:
            print("Warning: the selected scenarios look like a test set. Training on them removes their held-out evaluation meaning.")

    for scenario_order, scenario_path in enumerate(scenario_paths, start=1):
        run_id = _sanitize_token(f"{curriculum_name}_{scenario_path.stem}")
        command = [
            sys.executable,
            str(train_script),
            "--scenario-path",
            str(scenario_path),
            "--timesteps",
            str(int(args.timesteps)),
            "--num-envs",
            str(int(args.num_envs)),
            "--device",
            args.device,
            "--run-id",
            run_id,
            "--max-episodes",
            str(int(args.max_episodes_per_scenario)),
            "--convergence-window",
            str(int(args.convergence_window)),
            "--convergence-min-episodes",
            str(int(args.convergence_min_episodes)),
            "--convergence-min-success-rate",
            str(float(args.convergence_min_success_rate)),
            "--convergence-max-timeout-rate",
            str(float(args.convergence_max_timeout_rate)),
            "--convergence-max-failure-rate",
            str(float(args.convergence_max_failure_rate)),
            "--convergence-reward-window",
            str(int(args.convergence_reward_window)),
            "--convergence-reward-stability-ratio",
            str(float(args.convergence_reward_stability_ratio)),
            "--no-render",
            "--no-plot-reward",
        ]
        command.append("--record-videos" if bool(args.record_videos) else "--no-record-videos")
        command.extend(["--video-interval-episodes", str(int(args.video_interval_episodes))])

        if scenario_order == 1 and initial_mode == "bc":
            command.extend(["--bc-weights", str(current_model)])
        else:
            command.extend(["--resume-from", str(current_model)])

        print("")
        print(f"==== Scenario {scenario_order}/{len(scenario_paths)}: {scenario_path.name} ====")
        subprocess.run(command, check=True, cwd=str(Path(__file__).resolve().parent))

        scenario_log_dir = log_root / scenario_path.stem
        run_summary_path = scenario_log_dir / f"training_summary_{run_id}.json"
        if not run_summary_path.exists():
            raise FileNotFoundError(f"Expected training summary not found: {run_summary_path}")

        with run_summary_path.open("r", encoding="utf-8") as file:
            run_summary = json.load(file)

        row = {
            "curriculum_name": curriculum_name,
            "scenario_order": scenario_order,
            "scenario_stem": scenario_path.stem,
            **run_summary,
        }
        curriculum_rows.append(row)
        write_curriculum_summaries(curriculum_json_path, curriculum_csv_path, curriculum_rows)

        if bool(args.stop_on_nonconverged) and not bool(run_summary.get("converged", False)):
            raise RuntimeError(
                f"Scenario {scenario_path.name} stopped without convergence. "
                f"Stop reason: {run_summary.get('stop_reason', 'unknown')}. "
                f"See {run_summary_path} for details."
            )

        latest_model_path = run_summary.get("latest_model_path")
        if not latest_model_path:
            raise RuntimeError(f"Scenario {scenario_path.name} did not produce a latest_model_path in {run_summary_path}")
        current_model = Path(latest_model_path).resolve()

    print("")
    print(f"Curriculum completed. Summary JSON: {curriculum_json_path}")
    print(f"Curriculum completed. Summary CSV: {curriculum_csv_path}")


if __name__ == "__main__":
    main()
