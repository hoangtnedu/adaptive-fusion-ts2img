from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.cli.collect_results import collect_results
from src.cli.export_paper_tables import alpha_table, cost_table, mean_std_table
from src.cli.rank_results import build_rank_tables
from src.cli.statistical_tests import prepare_wide, run_tests


def ensure_results_csv(output_root: Path, results_csv: Path) -> pd.DataFrame:
    df = collect_results(output_root)
    if df.empty:
        if results_csv.exists():
            df = pd.read_csv(results_csv)
        else:
            raise ValueError(
                f"No run summaries found under {output_root} and results CSV does not exist: {results_csv}"
            )
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv, index=False)
    return df


def save_basic_tables(df: pd.DataFrame, out_dir: Path) -> None:
    for metric in ["test_acc", "test_macro_f1", "test_precision", "test_recall"]:
        if metric in df.columns:
            table = mean_std_table(df, metric)
            table.to_csv(out_dir / f"table_{metric}_mean_std_by_dataset.csv", index=False)

    costs = cost_table(df)
    costs.to_csv(out_dir / "table_average_costs.csv", index=False)

    alphas = alpha_table(df)
    if not alphas.empty:
        alphas.to_csv(out_dir / "table_alpha_by_dataset_method.csv", index=False)

    compact_cols = [
        c
        for c in [
            "dataset",
            "experiment",
            "seed",
            "test_acc",
            "test_macro_f1",
            "test_precision",
            "test_recall",
            "trainable_params",
            "params",
            "flops",
            "inference_ms_per_sample",
            "total_train_time_sec",
            "representations",
        ]
        if c in df.columns
    ]
    df[compact_cols].to_csv(out_dir / "table_all_runs_compact.csv", index=False)


def save_rank_and_stats(df: pd.DataFrame, metric: str, proposed: str, alpha: float, out_dir: Path):
    complete, ranks, rank_summary = build_rank_tables(df, metric, higher_is_better=True)
    complete.to_csv(out_dir / f"wide_{metric}.csv")
    ranks.to_csv(out_dir / f"ranks_{metric}.csv")
    rank_summary.to_csv(out_dir / f"average_ranks_{metric}.csv", index=False)

    stats_dir = out_dir / "statistical_tests"
    stats_dir.mkdir(parents=True, exist_ok=True)
    wide = prepare_wide(df, metric)
    friedman_summary, avg_rank_df, pairwise_df = run_tests(wide, proposed, metric, alpha)
    wide.to_csv(stats_dir / f"wide_complete_{metric}.csv")
    friedman_summary.to_csv(stats_dir / f"friedman_{metric}.csv", index=False)
    avg_rank_df.to_csv(stats_dir / f"average_ranks_{metric}.csv", index=False)
    pairwise_df.to_csv(stats_dir / f"wilcoxon_vs_{proposed}_{metric}.csv", index=False)
    return friedman_summary, avg_rank_df, pairwise_df


def save_cd_diagram(avg_rank_df: pd.DataFrame, friedman_summary: pd.DataFrame, metric: str, out_path: Path) -> None:
    if avg_rank_df.empty:
        return

    df = avg_rank_df.sort_values("average_rank").reset_index(drop=True)
    cd = None
    if not friedman_summary.empty and "nemenyi_cd" in friedman_summary.columns:
        cd = float(friedman_summary["nemenyi_cd"].iloc[0])

    fig_width = max(9, 0.45 * len(df) + 7)
    plt.figure(figsize=(fig_width, 4.5))
    y = list(range(len(df)))
    plt.scatter(df["average_rank"], y)
    for yi, (_, row) in zip(y, df.iterrows()):
        plt.text(row["average_rank"] + 0.03, yi, str(row["experiment"]), va="center", fontsize=9)

    plt.yticks([])
    plt.xlabel(f"Average rank by {metric} (lower is better)")
    plt.title("Average-rank diagram for model comparison")
    plt.grid(axis="x", linestyle="--", linewidth=0.5)

    x_min = max(1.0, float(df["average_rank"].min()) - 0.5)
    x_max = float(df["average_rank"].max()) + 0.8
    if cd is not None and cd == cd:
        top_y = -0.7
        start = x_min
        end = min(x_min + cd, x_max)
        plt.hlines(top_y, start, end, linewidth=2)
        plt.vlines([start, end], top_y - 0.08, top_y + 0.08, linewidth=2)
        plt.text((start + end) / 2, top_y - 0.22, f"CD={cd:.3f}", ha="center", fontsize=9)

    plt.xlim(x_min, x_max)
    plt.ylim(len(df) - 0.5, -1.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_ablation_table(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    if "experiment" not in df.columns or metric not in df.columns:
        return

    mask = df["experiment"].astype(str).eq("adaptive_fusion_full") | df["experiment"].astype(str).str.startswith(
        "ablation_"
    )
    ablation_df = df[mask].copy()
    if ablation_df.empty:
        return

    named_aggs = {
        "mean_metric": (metric, "mean"),
        "std_metric": (metric, "std"),
        "n_runs": (metric, "count"),
    }
    if "test_acc" in ablation_df.columns:
        named_aggs["mean_acc"] = ("test_acc", "mean")

    summary = (
        ablation_df.groupby("experiment", as_index=False)
        .agg(**named_aggs)
        .sort_values("mean_metric", ascending=False)
    )
    summary.to_csv(out_dir / f"table_ablation_summary_{metric}.csv", index=False)

    if {"dataset", "experiment", "seed", metric}.issubset(ablation_df.columns):
        try:
            _, _, ablation_rank_summary = build_rank_tables(ablation_df, metric, higher_is_better=True)
            ablation_rank_summary.to_csv(out_dir / f"table_ablation_average_ranks_{metric}.csv", index=False)
        except Exception as exc:
            (out_dir / "ablation_rank_warning.txt").write_text(str(exc), encoding="utf-8")


def write_readme(
    out_dir: Path,
    output_root: Path,
    results_csv: Path,
    metric: str,
    proposed: str,
    n_runs: int,
    n_datasets: int,
    n_methods: int,
) -> None:
    text = f"""# Paper package

This folder contains aggregated outputs for drafting the Adaptive Fusion TS2Img paper.

## Inputs

- Output root: `{output_root}`
- Results CSV: `{results_csv}`
- Main metric: `{metric}`
- Proposed method: `{proposed}`

## Run coverage

- Runs collected: `{n_runs}`
- Datasets found: `{n_datasets}`
- Methods found: `{n_methods}`

## Main files

- `summary_all_runs.csv`: all collected runs.
- `table_test_acc_mean_std_by_dataset.csv`: accuracy table by dataset and method.
- `table_test_macro_f1_mean_std_by_dataset.csv`: Macro F1 table by dataset and method.
- `table_average_costs.csv`: parameters, FLOPs, inference time, and training time.
- `table_alpha_by_dataset_method.csv`: average adaptive-fusion weights when available.
- `average_ranks_{metric}.csv`: average-rank summary.
- `statistical_tests/friedman_{metric}.csv`: Friedman test summary.
- `statistical_tests/wilcoxon_vs_{proposed}_{metric}.csv`: Wilcoxon tests against the proposed model.
- `critical_difference_{metric}.png`: average-rank diagram with Nemenyi critical difference.
- `table_ablation_summary_{metric}.csv`: ablation summary if ablation runs exist.

## Paper-writing checklist

1. Use Stage 3 results as the minimum paper-grade evidence.
2. Report Accuracy and Macro F1 together.
3. Report Params, FLOPs, training time, and inference time.
4. Use Friedman + Wilcoxon/Nemenyi results to support statistical claims.
5. Use ablation results to justify the contribution of each representation branch.
6. Use alpha-weight tables to discuss which representation contributes more by dataset.
7. Avoid claiming superiority if statistical tests do not support it.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a paper-ready package: collected CSV, tables, ranks, statistics, CD figure, and README."
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--results-csv", default="outputs/summary_all_runs.csv")
    parser.add_argument("--out-dir", default="outputs/paper_package")
    parser.add_argument("--metric", default="test_macro_f1")
    parser.add_argument("--proposed", default="adaptive_fusion_full")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    results_csv = Path(args.results_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_results_csv(output_root, results_csv)
    df.to_csv(out_dir / "summary_all_runs.csv", index=False)

    save_basic_tables(df, out_dir)

    try:
        friedman_summary, avg_rank_df, pairwise_df = save_rank_and_stats(
            df=df,
            metric=args.metric,
            proposed=args.proposed,
            alpha=args.alpha,
            out_dir=out_dir,
        )
        save_cd_diagram(
            avg_rank_df=avg_rank_df,
            friedman_summary=friedman_summary,
            metric=args.metric,
            out_path=out_dir / f"critical_difference_{args.metric}.png",
        )
    except Exception as exc:
        warning = (
            "Rank/statistical analysis was skipped. This usually means the result set "
            "does not yet contain a complete dataset x method block.\n"
            f"Reason: {exc}\n"
        )
        (out_dir / "rank_statistical_warning.txt").write_text(warning, encoding="utf-8")
        print(warning)

    save_ablation_table(df, args.metric, out_dir)

    n_runs = int(len(df))
    n_datasets = int(df["dataset"].nunique()) if "dataset" in df.columns else 0
    n_methods = int(df["experiment"].nunique()) if "experiment" in df.columns else 0

    write_readme(
        out_dir=out_dir,
        output_root=output_root,
        results_csv=results_csv,
        metric=args.metric,
        proposed=args.proposed,
        n_runs=n_runs,
        n_datasets=n_datasets,
        n_methods=n_methods,
    )

    print(f"Paper package saved to: {out_dir}")
    print(f"Runs: {n_runs}; datasets: {n_datasets}; methods: {n_methods}")
    print("Main files:")
    for name in [
        "summary_all_runs.csv",
        f"average_ranks_{args.metric}.csv",
        f"critical_difference_{args.metric}.png",
        f"table_ablation_summary_{args.metric}.csv",
        "README.md",
    ]:
        path = out_dir / name
        if path.exists():
            print(" -", path)


if __name__ == "__main__":
    main()
