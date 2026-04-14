from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from configs.default_config import make_config
from evaluate_bc_rl import evaluate_policy_model, make_bc_model, select_device
from utils.policy_utils import make_single_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one BC checkpoint on many scenarios and write a curriculum order sorted by BC performance.",
    )
    parser.add_argument(
        "--bc-weights",
        type=str,
        required=True,
        help="BC actor checkpoint (.pth/.pt) used to score the scenarios.",
    )
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=None,
        help="Directory containing scenario JSON files. Required unless --scenario-list is provided.",
    )
    parser.add_argument(
        "--scenario-glob",
        type=str,
        default="*.json",
        help="Glob pattern used inside --scenario-dir.",
    )
    parser.add_argument(
        "--scenario-list",
        type=str,
        default=None,
        help="Optional JSON file containing an explicit list of scenarios to evaluate.",
    )
    parser.add_argument("--start-index", type=int, default=None, help="Optional 1-based start position after ordering.")
    parser.add_argument("--end-index", type=int, default=None, help="Optional 1-based end position after ordering.")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes per scenario.")
    parser.add_argument("--seed", type=int, default=7, help="Base evaluation seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device: cuda, cuda:0, cpu, or auto.")
    parser.add_argument("--xml-path", type=str, default=None, help="Optional MuJoCo XML override.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path. The file contains a 'scenarios' list that train_curriculum.py can consume via --scenario-list.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV summary path for easy spreadsheet inspection.",
    )
    return parser.parse_args()


def resolve_scenarios(args: argparse.Namespace) -> list[Path]:
    if args.scenario_list is not None:
        scenario_list_path = Path(args.scenario_list).resolve()
        if not scenario_list_path.exists():
            raise FileNotFoundError(f"Scenario list not found: {scenario_list_path}")
        payload = json.loads(scenario_list_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            scenario_items = payload.get("scenarios")
            if scenario_items is None:
                raise KeyError(f"Scenario list JSON must contain 'scenarios': {scenario_list_path}")
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


def write_csv_summary(output_csv_path: Path, rows: list[dict[str, Any]]) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "scenario_stem",
        "scenario_path",
        "episodes",
        "avg_reward",
        "avg_episode_length",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "avg_goal_progress_ratio",
        "avg_abs_cross_track_error",
    ]
    with output_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    args = parse_args()
    bc_weights_path = Path(args.bc_weights).resolve()
    if not bc_weights_path.exists():
        raise FileNotFoundError(f"BC checkpoint not found: {bc_weights_path}")

    output_path = Path(args.output).resolve()
    output_csv_path = None if args.output_csv is None else Path(args.output_csv).resolve()
    scenario_paths = resolve_scenarios(args)

    config = make_config()
    if args.xml_path is not None:
        config.env.model.xml_path = str(Path(args.xml_path).resolve())
    device = select_device(args.device)

    ranking_rows: list[dict[str, Any]] = []
    model = make_bc_model(config, bc_weights_path, device)
    try:
        for scenario_path in scenario_paths:
            env = make_single_env(config, scenario_path=scenario_path)
            try:
                summary = evaluate_policy_model(
                    model,
                    env,
                    episodes=int(args.episodes),
                    seed=int(args.seed),
                    render=False,
                )
            finally:
                env.close()

            row = {
                "scenario_path": str(scenario_path),
                "scenario_stem": scenario_path.stem,
                **summary,
            }
            ranking_rows.append(row)
            print(
                f"{scenario_path.name}: "
                f"success={summary['success_rate']:.3f}, "
                f"reward={summary['avg_reward']:.2f}, "
                f"collision={summary['collision_rate']:.3f}, "
                f"timeout={summary['timeout_rate']:.3f}"
            )
    finally:
        if model.get_env() is not None:
            model.get_env().close()

    ranking_rows.sort(
        key=lambda row: (
            -float(row["success_rate"]),
            -float(row["avg_reward"]),
            -float(row["avg_goal_progress_ratio"]),
            float(row["collision_rate"]),
            float(row["timeout_rate"]),
            str(row["scenario_stem"]),
        )
    )
    for rank, row in enumerate(ranking_rows, start=1):
        row["rank"] = int(rank)

    payload = {
        "bc_weights": str(bc_weights_path),
        "device": device,
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "xml_path": config.env.model.xml_path,
        "sort_order": [
            "success_rate desc",
            "avg_reward desc",
            "avg_goal_progress_ratio desc",
            "collision_rate asc",
            "timeout_rate asc",
            "scenario_stem asc",
        ],
        "scenarios": ranking_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(f"Saved ranked scenario list to {output_path}")

    if output_csv_path is not None:
        write_csv_summary(output_csv_path, ranking_rows)
        print(f"Saved ranked CSV summary to {output_csv_path}")


if __name__ == "__main__":
    main()
