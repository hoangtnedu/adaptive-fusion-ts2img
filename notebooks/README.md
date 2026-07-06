# Notebooks

Use only these command-only notebooks:

- `01_colab_pipeline_commands_only.ipynb`: run on Google Colab. Code is cloned to `/content`; outputs/cache/checkpoints go to Google Drive.
- `02_local_pipeline_commands_only.ipynb`: run locally in Visual Studio/VS Code.

The notebooks should not contain model or data-processing logic. All logic belongs in `src/`; all experiment parameters belong in YAML files under `config/`.
