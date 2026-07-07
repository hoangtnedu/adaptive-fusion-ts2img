# Experiment stages for paper-grade adaptive fusion research

This project supports staged experimentation. Start small, verify the pipeline, then scale up.

## Colab workflow rule

On Google Colab, use `notebooks/01_colab_pipeline_commands_only.ipynb`.

The intended workflow is:

```text
GitHub repository -> clone source code to /content/adaptive-fusion-ts2img
Google Drive      -> store outputs, cache, checkpoints, tables, figures, paper package
```

Do not run the project directly inside Google Drive unless there is a special reason. Running source code from `/content` is cleaner and faster; saving results to Drive preserves outputs after the Colab session ends.

## Stage 1 — Smoke test

Goal: verify data loading, 2D transformations, cache, resume, checkpoint, baseline methods, result collection.

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage1_smoke_5datasets_3seeds_all_methods.yaml

python -m src.cli.make_paper_package \
  --output-root outputs \
  --results-csv outputs/summary_all_runs.csv \
  --out-dir outputs/paper_package \
  --metric test_macro_f1 \
  --proposed adaptive_fusion_full
```

Scale: `5 datasets × 3 seeds × 7 methods = 105 runs`.

## Stage 2 — Pilot experiment

Goal: check whether adaptive fusion is promising before running expensive experiments.

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage2_pilot_12datasets_3seeds_all_methods.yaml

python -m src.cli.make_paper_package \
  --output-root outputs \
  --results-csv outputs/summary_all_runs.csv \
  --out-dir outputs/paper_package \
  --metric test_macro_f1 \
  --proposed adaptive_fusion_full
```

Scale: `12 datasets × 3 seeds × 7 methods = 252 runs`.

## Stage 3 — Paper-grade minimum

Goal: meet the minimum dataset scale for a serious Q-level paper.

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage3_paper_20datasets_3seeds_all_methods.yaml
```

Scale: `20 datasets × 3 seeds × 7 methods = 420 runs`.

## Stage 3B — Ablation study

Goal: prove the contribution of each representation branch.

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage3_ablation_20datasets_3seeds_adaptive.yaml
```

This suite compares:

1. `adaptive_fusion_full`: GAF + MTF + RP + STFT.
2. `ablation_without_gaf`: MTF + RP + STFT.
3. `ablation_without_mtf`: GAF + RP + STFT.
4. `ablation_without_rp`: GAF + MTF + STFT.
5. `ablation_without_stft`: GAF + MTF + RP.

Scale: `20 datasets × 3 seeds × 5 ablation methods = 300 runs`.

## Paper package

After Stage 3 and Stage 3B, create the final paper package:

```bash
python -m src.cli.make_paper_package \
  --output-root outputs \
  --results-csv outputs/summary_all_runs.csv \
  --out-dir outputs/paper_package \
  --metric test_macro_f1 \
  --proposed adaptive_fusion_full
```

The paper package includes:

- collected run summaries;
- mean ± std tables for Accuracy, Macro F1, Precision, Recall;
- parameter/FLOPs/training-time/inference-time cost table;
- average-rank tables;
- Friedman and Wilcoxon statistical tests;
- critical-difference average-rank figure;
- alpha/fusion-weight tables when available;
- ablation summary tables;
- a paper-writing README checklist.

## Stage 4 — Strong Q1/Q2-scale experiment

Option A: all seven methods across 30 datasets and 5 seeds:

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage4_strong_30datasets_5seeds_all_methods.yaml
```

Scale: `30 datasets × 5 seeds × 7 methods = 1050 runs`.

Option B: proposed method across 50 datasets:

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage4_strong_50datasets_3seeds_adaptive_only.yaml
```

This option is useful only after the main baseline comparison is already completed.

## Methods included in the main comparison

1. `cnn1d`: raw 1D-CNN baseline.
2. `gaf_lightcnn`: GAF + lightweight CNN.
3. `mtf_lightcnn`: MTF + lightweight CNN.
4. `rp_lightcnn`: RP + lightweight CNN.
5. `stft_lightcnn`: STFT + lightweight CNN.
6. `manual_feature_concat`: manual feature concatenation fusion.
7. `adaptive_fusion_full`: proposed adaptive gated fusion model.

## Recommended workflow

Run stages in order:

```text
Stage 1 -> paper package check -> Stage 2 -> paper package check -> Stage 3 -> Stage 3B ablation -> final paper package
```

Do not jump directly to Stage 3 or Stage 4 until Stage 1 and Stage 2 complete without errors.
