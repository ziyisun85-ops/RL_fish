from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot success/failure episodes from episode_metrics.csv as a 0/1 scatter plot.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to episode_metrics.csv.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="latest",
        help="Run filter: latest, all, or one explicit run_id value.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional PNG output path. Defaults next to the CSV.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=24.0,
        help="Scatter point size.",
    )
    return parser.parse_args()


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_int(text: str, default: int) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        return int(default)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"episode_metrics.csv not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def resolve_run_id(rows: list[dict[str, str]], run_id_arg: str) -> tuple[str | None, list[dict[str, str]]]:
    if not rows:
        return None, []

    if run_id_arg == "all":
        return None, rows

    if run_id_arg == "latest":
        latest_run_id = rows[-1].get("run_id", "")
        selected_rows = [row for row in rows if row.get("run_id", "") == latest_run_id]
        return latest_run_id, selected_rows

    selected_rows = [row for row in rows if row.get("run_id", "") == run_id_arg]
    return run_id_arg, selected_rows


def default_output_path(csv_path: Path, run_id: str | None) -> Path:
    suffix = "all_runs" if run_id is None else run_id
    return csv_path.with_name(f"success_scatter_{suffix}.png")


def plot_success_scatter(rows: list[dict[str, str]], output_path: Path, title: str, point_size: float) -> None:
    x_values: list[int] = []
    y_values: list[int] = []
    colors: list[str] = []

    for fallback_index, row in enumerate(rows, start=1):
        episode_index = parse_int(row.get("episode_index", ""), fallback_index)
        success_value = 1 if parse_bool(row.get("success", "false")) else 0
        x_values.append(episode_index)
        y_values.append(success_value)
        colors.append("#2a9d8f" if success_value == 1 else "#d62828")

    figure, axis = plt.subplots(figsize=(11.0, 4.5))
    axis.scatter(x_values, y_values, c=colors, s=float(point_size), alpha=0.9, edgecolors="none")
    axis.set_xlabel("Episode Index")
    axis.set_ylabel("Success")
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Fail", "Success"])
    axis.set_ylim(-0.15, 1.15)
    axis.grid(alpha=0.25, axis="both")
    axis.set_title(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input).resolve()
    all_rows = load_rows(csv_path)
    selected_run_id, rows = resolve_run_id(all_rows, args.run_id)
    if not rows:
        raise RuntimeError(f"No rows matched run selection {args.run_id!r} in {csv_path}")

    output_path = Path(args.output).resolve() if args.output is not None else default_output_path(csv_path, selected_run_id)
    title = args.title
    if title is None:
        if selected_run_id is None:
            title = f"Success Scatter | all runs | n={len(rows)}"
        else:
            title = f"Success Scatter | {selected_run_id} | n={len(rows)}"

    plot_success_scatter(
        rows=rows,
        output_path=output_path,
        title=title,
        point_size=float(args.point_size),
    )

    success_count = sum(1 for row in rows if parse_bool(row.get("success", "false")))
    print(f"Saved scatter plot to {output_path}")
    print(f"Selected rows: {len(rows)} | Successes: {success_count} | Success rate: {success_count / max(len(rows), 1):.4f}")
    if selected_run_id is not None:
        print(f"Run ID: {selected_run_id}")


if __name__ == "__main__":
    main()
