# Notebooks

Use only these command-only notebooks:

- `01_colab_pipeline_commands_only.ipynb`: run on Google Colab. Source code is cloned from `https://github.com/hoangtnedu/adaptive-fusion-ts2img.git` to `/content/adaptive-fusion-ts2img`. Outputs, cache, checkpoints, tables, figures, and the paper package are saved to Google Drive.
- `02_local_pipeline_commands_only.ipynb`: run locally in Visual Studio/VS Code.
- `03_kaggle_stage2_commands_only.ipynb`: run Stage 2 on Kaggle with GPU and Internet enabled. The notebook supports the full 252-run suite or three resumable chunks of 84 runs, saves outputs/cache/checkpoints under `/kaggle/working`, and can create a resume bundle for the next Kaggle session.

The notebooks should not contain model or data-processing logic. All logic belongs in `src/`; all experiment parameters belong in YAML files under `config/`.

Recommended Colab order:

```text
Mount Drive
-> clone GitHub repo to /content
-> install requirements
-> download datasets
-> Stage 1 smoke test
-> Stage 2 pilot
-> Stage 3 paper-grade minimum
-> Stage 3B ablation
-> make paper package
```

Recommended Kaggle Stage 2 order:

```text
Enable GPU and Internet
-> clone GitHub repo to /kaggle/working
-> install requirements
-> download 12 Stage 2 datasets
-> dry run
-> run chunk 1
-> save/restore resume bundle
-> run chunk 2
-> save/restore resume bundle
-> run chunk 3
-> verify 252 runs
-> make paper package
```
