from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def mean_std_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    agg = df.groupby(["dataset", "experiment"])[metric].agg(["mean", "std", "count"]).reset_index()
    agg["mean_std"] = agg.apply(lambda r: f"{r['mean']:.4f} ± {0 if pd.isna(r['std']) else r['std']:.4f}", axis=1)
    return agg.pivot(index="dataset", columns="experiment", values="mean_std").reset_index()


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "experiment",
        "trainable_params",
        "params",
        "flops",
        "inference_ms_per_sample",
        "total_train_time_sec",
    ]
    existing = [c for c in cols if c in df.columns]
    numeric = [c for c in existing if c != "experiment"]
    return df.groupby("experiment", as_index=False)[numeric].mean()


def alpha_table(df: pd.DataFrame) -> pd.DataFrame:
    alpha_cols = [c for c in df.columns if c.startswith("alpha_")]
    if not alpha_cols:
        return pd.DataFrame()
    cols = ["dataset", "experiment", *alpha_cols]
    return df[cols].groupby(["dataset", "experiment"], as_index=False).mean()


def main():
    parser = argparse.ArgumentParser(description="Export CSV tables ready for paper drafting.")
    parser.add_argument("--results-csv", default="outputs/summary_all_runs.csv")
    parser.add_argument("--out-dir", default="outputs/paper_tables")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["test_acc", "test_macro_f1", "test_precision", "test_recall"]:
        if metric in df.columns:
            table = mean_std_table(df, metric)
            table.to_csv(out_dir / f"table_{metric}_mean_std_by_dataset.csv", index=False)

    costs = cost_table(df)
    costs.to_csv(out_dir / "table_average_costs.csv", index=False)

    alphas = alpha_table(df)
    if not alphas.empty:
        alphas.to_csv(out_dir / "table_alpha_by_dataset_method.csv", index=False)

    selected = [
        c
        for c in [
            "dataset",
            "experiment",
            "seed",
            "test_acc",
            "test_macro_f1",
            "trainable_params",
            "flops",
            "inference_ms_per_sample",
        ]
        if c in df.columns
    ]
    df[selected].to_csv(out_dir / "table_all_runs_compact.csv", index=False)
    print(f"Paper tables saved to: {out_dir}")


if __name__ == "__main__":
    main()
