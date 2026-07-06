from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_json(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_one_run(run_dir: Path):
    summary = read_json(run_dir / "summary.json")
    if not summary:
        return None
    complexity = read_json(run_dir / "complexity.json")
    inference = read_json(run_dir / "inference_time.json")

    row = dict(summary)
    row.setdefault("run_dir", str(run_dir))
    row.setdefault("params", complexity.get("params"))
    row.setdefault("flops", complexity.get("flops"))
    row.setdefault("trainable_params", complexity.get("trainable_params"))
    row.setdefault("inference_ms_per_sample", inference.get("ms_per_sample"))

    reps = row.get("representations")
    if isinstance(reps, list):
        row["representations"] = "+".join(reps)

    return row


def collect_results(output_root: Path) -> pd.DataFrame:
    rows = []
    for summary_path in output_root.rglob("summary.json"):
        row = collect_one_run(summary_path.parent)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["dataset", "experiment", "seed"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect all run summaries into one CSV file.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--out-csv", default="outputs/summary_all_runs.csv")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    df = collect_results(output_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Collected {len(df)} runs -> {out_csv}")
    if len(df):
        print(df[[c for c in ["dataset", "experiment", "seed", "test_acc", "test_macro_f1"] if c in df.columns]].head(20))


if __name__ == "__main__":
    main()
