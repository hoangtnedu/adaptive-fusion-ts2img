# Experiment stages for paper-grade adaptive fusion research

This project supports staged experimentation. Start small, verify the pipeline, then scale up.

## Stage 1 — Smoke test

Goal: verify data loading, 2D transformations, cache, resume, checkpoint, baseline methods, result collection.

```bash
python -m src.cli.batch_run --suite config/suites/stage1_smoke_5datasets_3seeds_all_methods.yaml
python -m src.cli.collect_results --output-root outputs --out-csv outputs/summary_all_runs.csv
python -m src.cli.export_paper_tables --results-csv outputs/summary_all_runs.csv --out-dir outputs/paper_tables
```

Scale: `5 datasets × 3 seeds × 7 methods = 105 runs`.

## Stage 2 — Pilot experiment

Goal: check whether adaptive fusion is promising before running expensive experiments.

```bash
python -m src.cli.batch_run --suite config/suites/stage2_pilot_12datasets_3seeds_all_methods.yaml
python -m src.cli.collect_results --output-root outputs --out-csv outputs/summary_all_runs.csv
python -m src.cli.rank_results --results-csv outputs/summary_all_runs.csv --metric test_macro_f1
```

Scale: `12 datasets × 3 seeds × 7 methods = 252 runs`.

## Stage 3 — Paper-grade minimum

Goal: meet the minimum dataset scale for a serious Q-level paper.

```bash
python -m src.cli.batch_run --suite config/suites/stage3_paper_20datasets_3seeds_all_methods.yaml
python -m src.cli.collect_results --output-root outputs --out-csv outputs/summary_all_runs.csv
python -m src.cli.rank_results --results-csv outputs/summary_all_runs.csv --metric test_macro_f1
python -m src.cli.statistical_tests --results-csv outputs/summary_all_runs.csv --metric test_macro_f1 --proposed adaptive_fusion_full
python -m src.cli.export_paper_tables --results-csv outputs/summary_all_runs.csv --out-dir outputs/paper_tables
```

Scale: `20 datasets × 3 seeds × 7 methods = 420 runs`.

## Stage 4 — Strong Q1/Q2-scale experiment

Option A: proposed method across 50 datasets:

```bash
python -m src.cli.batch_run --suite config/suites/stage4_strong_50datasets_3seeds_adaptive_only.yaml
```

Option B: all seven methods across 30 datasets and 5 seeds:

```bash
python -m src.cli.batch_run --suite config/suites/stage4_strong_30datasets_5seeds_all_methods.yaml
```

Scale for option B: `30 datasets × 5 seeds × 7 methods = 1050 runs`.

## Methods included

1. `cnn1d`: raw 1D-CNN baseline.
2. `gaf_lightcnn`: GAF + lightweight CNN.
3. `mtf_lightcnn`: MTF + lightweight CNN.
4. `rp_lightcnn`: RP + lightweight CNN.
5. `stft_lightcnn`: STFT + lightweight CNN.
6. `manual_feature_concat`: manual feature concatenation fusion.
7. `adaptive_fusion_full`: proposed adaptive gated fusion model.

## Recommended workflow

Run stages in order. Do not jump directly to Stage 3 or Stage 4 until Stage 1 and Stage 2 complete without errors.
