from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_softsensor_drift.experiments import Reproduction, RunConfig


def parse_seeds(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SCADA state-classifier drift robustness experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/hgsf7bwkrv-1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", type=parse_seeds, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    config = RunConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        device=args.device,
        fast=args.fast,
    )
    Reproduction(config).run()
    print(f"Done. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
