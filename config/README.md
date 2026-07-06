# Config structure

This project uses paper-grade YAML configuration.

```text
config/
├── default.yaml                         # shared defaults
├── experiments/                         # one adaptive-fusion experiment per dataset
├── baselines/                           # 1D-CNN, single 2D, manual fusion configs
├── ablations/                           # remove one representation branch
└── suites/                              # staged multi-dataset / multi-seed experiments
```

## Single experiment

```bash
python -m src.train --config config/experiments/coffee_adaptive_fusion.yaml
```

## Baselines

```bash
python -m src.train --config config/baselines/cnn1d.yaml
python -m src.train --config config/baselines/single_gaf.yaml
python -m src.train --config config/baselines/manual_feature_concat.yaml
```

## Staged suites

```bash
python -m src.cli.batch_run --suite config/suites/stage1_smoke_5datasets_3seeds_all_methods.yaml
python -m src.cli.batch_run --suite config/suites/stage2_pilot_12datasets_3seeds_all_methods.yaml
python -m src.cli.batch_run --suite config/suites/stage3_paper_20datasets_3seeds_all_methods.yaml
python -m src.cli.batch_run --suite config/suites/stage4_strong_30datasets_5seeds_all_methods.yaml
```

Each run saves its actual resolved configuration to:

```text
outputs/<run_name>/config_used.yaml
```
