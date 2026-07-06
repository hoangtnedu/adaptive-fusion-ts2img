from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_rank_tables(df: pd.DataFrame, metric: str, higher_is_better: bool = True):
    required = {"dataset", "experiment", "seed", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in results CSV: {missing}")

    # Average across seeds first, then rank methods within each dataset.
    mean_df = (
        df.groupby(["dataset", "experiment"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: f"mean_{metric}"})
    )

    wide = mean_df.pivot(index="dataset", columns="experiment", values=f"mean_{metric}")
    complete = wide.dropna(axis=0, how="any")
    if complete.empty:
        raise ValueError("No complete dataset x method block found for ranking.")

    ranks = complete.rank(axis=1, ascending=not higher_is_better, method="average")
    avg_rank = ranks.mean(axis=0).sort_values().reset_index()
    avg_rank.columns = ["experiment", "average_rank"]

    metric_mean = complete.mean(axis=0).reset_index()
    metric_mean.columns = ["experiment", f"average_{metric}"]
    metric_std = complete.std(axis=0).reset_index()
    metric_std.columns = ["experiment", f"std_{metric}"]

    summary = avg_rank.merge(metric_mean, on="experiment").merge(metric_std, on="experiment")
    return complete, ranks, summary


def main():
    parser = argparse.ArgumentParser(description="Compute average ranks across datasets for paper tables.")
    parser.add_argument("--results-csv", default="outputs/summary_all_runs.csv")
    parser.add_argument("--metric", default="test_macro_f1")
    parser.add_argument("--out-dir", default="outputs/paper_tables")
    parser.add_argument("--lower-is-better", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    wide, ranks, summary = build_rank_tables(df, args.metric, higher_is_better=not args.lower_is_better)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_dir / f"wide_{args.metric}.csv")
    ranks.to_csv(out_dir / f"ranks_{args.metric}.csv")
    summary.to_csv(out_dir / f"average_ranks_{args.metric}.csv", index=False)

    print(summary)


if __name__ == "__main__":
    main()
