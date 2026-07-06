from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.experiment import make_run_name


def infer_run_dir(cfg) -> Path:
    return Path(cfg["paths"]["output_root"]) / make_run_name(cfg)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print key outputs from a completed or running experiment.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--list-files", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else infer_run_dir(cfg)

    print("Run directory:", run_dir)
    print("Exists:", run_dir.exists())

    if not run_dir.exists():
        return

    if args.list_files:
        print("\nFiles:")
        for p in sorted(run_dir.iterdir()):
            print(" -", p.name)

    summary_path = run_dir / "summary.json"
    complexity_path = run_dir / "complexity.json"
    alpha_mean_path = run_dir / "alpha_mean.csv"
    history_path = run_dir / "history.csv"
    config_used_path = run_dir / "config_used.yaml"

    if summary_path.exists():
        print("\nSummary")
        print(json.dumps(read_json(summary_path), indent=4, ensure_ascii=False))
    else:
        print("\nMissing summary.json")

    if complexity_path.exists():
        print("\nComplexity")
        print(json.dumps(read_json(complexity_path), indent=4, ensure_ascii=False))
    else:
        print("\nMissing complexity.json")

    if alpha_mean_path.exists():
        print("\nMean fusion weights")
        alpha = pd.read_csv(alpha_mean_path, index_col=0)
        print(alpha)
        best_rep = alpha.iloc[:, 0].idxmax()
        best_val = float(alpha.iloc[:, 0].max())
        print(f"Dominant representation: {best_rep} ({best_val:.4f})")
    else:
        print("\nMissing alpha_mean.csv")

    if history_path.exists():
        hist = pd.read_csv(history_path)
        print("\nLast history rows")
        print(hist.tail())
    else:
        print("\nMissing history.csv")

    print("\nReproducibility files:")
    print(" -", config_used_path)
    print(" -", run_dir / "environment.json")

    print("\nUseful figures:")
    print(" -", run_dir / "learning_curve_macro_f1.png")
    print(" -", run_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
