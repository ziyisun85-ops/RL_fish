from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot success rate by saved checkpoint weight name.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to success_by_weight.csv or compatible CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output PNG path. Defaults next to the CSV.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Success Rate by Weight",
        help="Plot title.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        choices=("full", "stem", "update"),
        default="update",
        help="How to label the x-axis: full weight filename, filename stem, or update index.",
    )
    return parser.parse_args()


def default_output_path(csv_path: Path) -> Path:
    return csv_path.with_name("success_rate_by_weight_name.png")


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in: {csv_path}")
    return rows


def label_for_row(row: dict[str, str], mode: str) -> str:
    weight_name = str(row.get("weight_name", "")).strip()
    update_index = str(row.get("update_index", "")).strip()
    if mode == "full":
        return weight_name or update_index
    if mode == "stem":
        return Path(weight_name).stem if weight_name else update_index
    return f"u{int(update_index):03d}" if update_index.isdigit() else (Path(weight_name).stem if weight_name else "?")


def plot_success_by_weight(rows: list[dict[str, str]], output_path: Path, title: str, label_mode: str) -> None:
    ordered_rows = sorted(rows, key=lambda row: int(row.get("update_index", "0")))
    x_positions = list(range(1, len(ordered_rows) + 1))
    x_labels = [label_for_row(row, label_mode) for row in ordered_rows]
    y_success = [float(row["success_rate"]) for row in ordered_rows]
    y_reward = [float(row.get("avg_reward", "0.0")) for row in ordered_rows]

    best_idx = max(range(len(ordered_rows)), key=lambda i: y_success[i])

    figure, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(max(14.0, len(ordered_rows) * 0.35), 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.4]},
    )

    ax1.plot(x_positions, y_success, color="#1d4ed8", linewidth=2.0, marker="o", markersize=4.5)
    ax1.scatter([x_positions[best_idx]], [y_success[best_idx]], color="#dc2626", s=70, zorder=3)
    ax1.annotate(
        f"best={x_labels[best_idx]} ({y_success[best_idx]:.3f})",
        (x_positions[best_idx], y_success[best_idx]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        color="#991b1b",
    )
    ax1.set_ylabel("Success Rate")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.25)
    ax1.set_title(title)

    ax2.plot(x_positions, y_reward, color="#059669", linewidth=1.8, marker="o", markersize=3.8)
    ax2.set_ylabel("Avg Reward")
    ax2.set_xlabel("Checkpoint Weight")
    ax2.grid(True, alpha=0.25)

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, rotation=90, fontsize=8)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output is not None else default_output_path(csv_path)
    rows = load_rows(csv_path)
    plot_success_by_weight(
        rows=rows,
        output_path=output_path,
        title=str(args.title),
        label_mode=str(args.label_mode),
    )
    print(f"Saved plot to {output_path}")
    print(f"Rows plotted: {len(rows)}")


if __name__ == "__main__":
    main()
