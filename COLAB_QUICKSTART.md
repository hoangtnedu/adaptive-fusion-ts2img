# Colab Quickstart — GitHub code + Google Drive results

Use this workflow when running the project on Google Colab.

## Rule

```text
GitHub      = source code
Google Drive = outputs, cache, checkpoints, tables, figures, paper package
```

Use notebook:

```text
notebooks/01_colab_pipeline_commands_only.ipynb
```

The notebook clones this repository to:

```text
/content/adaptive-fusion-ts2img
```

and saves all experiment artifacts to:

```text
/content/drive/MyDrive/research_ts2img_adaptive_fusion
```

## Run order

### 1. Mount Drive and clone GitHub

The first notebook cells already do this.

### 2. Install requirements

```bash
pip install -r requirements.txt
python -m compileall src
```

### 3. Download datasets

Run the dataset download cells for Stage 1, Stage 2, and Stage 3.

### 4. Stage 1 — smoke test

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage1_smoke_5datasets_3seeds_all_methods.yaml \
  --set paths.output_root=/content/drive/MyDrive/research_ts2img_adaptive_fusion/outputs paths.cache_root=/content/drive/MyDrive/research_ts2img_adaptive_fusion/cache
```

### 5. Stage 2 — pilot experiment

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage2_pilot_12datasets_3seeds_all_methods.yaml \
  --set paths.output_root=/content/drive/MyDrive/research_ts2img_adaptive_fusion/outputs paths.cache_root=/content/drive/MyDrive/research_ts2img_adaptive_fusion/cache
```

### 6. Stage 3 — paper-grade minimum

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage3_paper_20datasets_3seeds_all_methods.yaml \
  --set paths.output_root=/content/drive/MyDrive/research_ts2img_adaptive_fusion/outputs paths.cache_root=/content/drive/MyDrive/research_ts2img_adaptive_fusion/cache
```

### 7. Stage 3B — ablation study

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage3_ablation_20datasets_3seeds_adaptive.yaml \
  --set paths.output_root=/content/drive/MyDrive/research_ts2img_adaptive_fusion/outputs paths.cache_root=/content/drive/MyDrive/research_ts2img_adaptive_fusion/cache
```

### 8. Generate paper package

```bash
python -m src.cli.make_paper_package \
  --output-root /content/drive/MyDrive/research_ts2img_adaptive_fusion/outputs \
  --results-csv /content/drive/MyDrive/research_ts2img_adaptive_fusion/outputs/summary_all_runs.csv \
  --out-dir /content/drive/MyDrive/research_ts2img_adaptive_fusion/outputs/paper_package \
  --metric test_macro_f1 \
  --proposed adaptive_fusion_full
```

## Paper package outputs

The folder `paper_package` contains:

- `summary_all_runs.csv`
- `table_test_acc_mean_std_by_dataset.csv`
- `table_test_macro_f1_mean_std_by_dataset.csv`
- `table_average_costs.csv`
- `table_alpha_by_dataset_method.csv` when fusion weights exist
- `average_ranks_test_macro_f1.csv`
- `critical_difference_test_macro_f1.png`
- `statistical_tests/friedman_test_macro_f1.csv`
- `statistical_tests/wilcoxon_vs_adaptive_fusion_full_test_macro_f1.csv`
- `table_ablation_summary_test_macro_f1.csv` when ablation runs exist
- `README.md` with a paper-writing checklist
