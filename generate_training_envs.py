from __future__ import annotations

import argparse
import json
from pathlib import Path

from configs.default_config import PROJECT_ROOT, make_config
from envs import FishPathAvoidEnv
from utils.scenario_io import save_fixed_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed fish-training environments to JSON files.")
    parser.add_argument("--count", type=int, default=20, help="Number of fixed environments to create.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str((PROJECT_ROOT / "scenarios" / "training_envs").resolve()),
        help="Directory where the exported environment JSON files will be written.",
    )
    parser.add_argument("--base-seed", type=int, default=7, help="Base RNG seed used to generate the environments.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_config().env
    manifest: list[dict[str, object]] = []
    for index in range(1, max(1, int(args.count)) + 1):
        scenario_seed = int(args.base_seed) + index - 1
        scenario_name = f"training_env_{index:02d}"
        env = FishPathAvoidEnv(config=config)
        try:
            env.reset(seed=scenario_seed)
            scenario = env.export_fixed_scenario(
                scenario_id=scenario_name,
                source_seed=scenario_seed,
            )
        finally:
            env.close()

        output_path = output_dir / f"{scenario_name}.json"
        save_fixed_scenario(scenario, output_path)
        manifest.append(
            {
                "scenario_id": scenario_name,
                "source_seed": scenario_seed,
                "path": str(output_path),
            }
        )
        print(f"Saved {scenario_name} to {output_path}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"count": len(manifest), "scenarios": manifest}, handle, indent=2, ensure_ascii=False)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
