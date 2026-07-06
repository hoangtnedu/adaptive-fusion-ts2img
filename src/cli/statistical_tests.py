from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def prepare_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    required = {"dataset", "experiment", "seed", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in results CSV: {missing}")

    mean_df = df.groupby(["dataset", "experiment"], as_index=False)[metric].mean()
    wide = mean_df.pivot(index="dataset", columns="experiment", values=metric)
    return wide.dropna(axis=0, how="any")


def nemenyi_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """Approximate Nemenyi critical difference for average ranks.

    Uses scipy's studentized_range distribution when available. The exact
    Nemenyi q_alpha depends on k and infinite degrees of freedom; this provides
    a reproducible approximation suitable for reporting CD values.
    """
    try:
        from scipy.stats import studentized_range

        q_alpha = float(studentized_range.ppf(1 - alpha, k, math.inf) / math.sqrt(2))
    except Exception:
        # Conservative fallback often used around k=5-10 at alpha=.05.
        q_alpha = 2.85
    return q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))


def run_tests(wide: pd.DataFrame, proposed: str | None, metric: str, alpha: float):
    if wide.shape[1] < 2:
        raise ValueError("At least two methods are required.")
    if wide.shape[0] < 2:
        raise ValueError("At least two complete datasets are required.")

    friedman_stat, friedman_p = friedmanchisquare(*[wide[col].values for col in wide.columns])

    ranks = wide.rank(axis=1, ascending=False, method="average")
    avg_ranks = ranks.mean(axis=0).sort_values()
    cd = nemenyi_critical_difference(k=wide.shape[1], n=wide.shape[0], alpha=alpha)

    pairwise_rows = []
    if proposed is not None:
        if proposed not in wide.columns:
            raise ValueError(f"Proposed method '{proposed}' not found in columns: {list(wide.columns)}")
        for baseline in wide.columns:
            if baseline == proposed:
                continue
            diff = wide[proposed] - wide[baseline]
            try:
                stat, p_value = wilcoxon(wide[proposed], wide[baseline], zero_method="wilcox", alternative="two-sided")
            except ValueError:
                stat, p_value = float("nan"), float("nan")
            pairwise_rows.append(
                {
                    "proposed": proposed,
                    "baseline": baseline,
                    "metric": metric,
                    "mean_diff_proposed_minus_baseline": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "wilcoxon_stat": float(stat),
                    "wilcoxon_p": float(p_value),
                    "significant_at_alpha": bool(p_value < alpha) if p_value == p_value else False,
                }
            )

    friedman_summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "n_datasets_complete": int(wide.shape[0]),
                "n_methods": int(wide.shape[1]),
                "friedman_stat": float(friedman_stat),
                "friedman_p": float(friedman_p),
                "nemenyi_cd_alpha": float(alpha),
                "nemenyi_cd": float(cd),
            }
        ]
    )
    avg_rank_df = avg_ranks.reset_index()
    avg_rank_df.columns = ["experiment", "average_rank"]
    return friedman_summary, avg_rank_df, pd.DataFrame(pairwise_rows)


def main():
    parser = argparse.ArgumentParser(description="Run Friedman and Wilcoxon tests for paper-grade evaluation.")
    parser.add_argument("--results-csv", default="outputs/summary_all_runs.csv")
    parser.add_argument("--metric", default="test_macro_f1")
    parser.add_argument("--proposed", default="adaptive_fusion_full")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-dir", default="outputs/statistical_tests")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    wide = prepare_wide(df, args.metric)
    friedman_summary, avg_rank_df, pairwise_df = run_tests(wide, args.proposed, args.metric, args.alpha)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_dir / f"wide_complete_{args.metric}.csv")
    friedman_summary.to_csv(out_dir / f"friedman_{args.metric}.csv", index=False)
    avg_rank_df.to_csv(out_dir / f"average_ranks_{args.metric}.csv", index=False)
    pairwise_df.to_csv(out_dir / f"wilcoxon_vs_{args.proposed}_{args.metric}.csv", index=False)

    print(friedman_summary)
    print(avg_rank_df)
    if not pairwise_df.empty:
        print(pairwise_df)


if __name__ == "__main__":
    main()
