# Notebooks

Use only these command-only notebooks:

- `01_colab_pipeline_commands_only.ipynb`: run on Google Colab. Source code is cloned from `https://github.com/hoangtnedu/adaptive-fusion-ts2img.git` to `/content/adaptive-fusion-ts2img`. Outputs, cache, checkpoints, tables, figures, and the paper package are saved to Google Drive.
- `02_local_pipeline_commands_only.ipynb`: run locally in Visual Studio/VS Code.

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
