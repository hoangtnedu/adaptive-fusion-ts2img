"""CLI for downloading UCR datasets used by the experiment pipeline.

Examples
--------
python -m src.cli.download_data --config config/default.yaml
python -m src.cli.download_data --config config/default.yaml --datasets Coffee GunPoint
python -m src.cli.download_data --config config/suites/stage1_smoke_5datasets_3seeds_all_methods.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from src.data.ucr_downloader import download_ucr_datasets


DEFAULT_BASE_URL = "https://www.timeseriesclassification.com/aeon-toolkit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download UCR datasets required by the adaptive-fusion pipeline."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config path, for example config/default.yaml or config/runtime_colab.yaml.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset override, for example --datasets Coffee GunPoint ECG200.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional data root override. Default is read from config paths.data_root or data/UCR.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional dataset ZIP base URL override.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and overwrite existing TRAIN/TEST TSV files.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Download timeout in seconds. Default: 60.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a dictionary: {path}")
    return cfg


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_datasets(config: dict[str, Any], cli_datasets: list[str] | None) -> list[str]:
    if cli_datasets:
        return _clean_dataset_list(cli_datasets)

    candidates = [
        get_nested(config, "data", "datasets"),
        get_nested(config, "datasets"),
        get_nested(config, "experiment", "datasets"),
        get_nested(config, "experiment", "dataset"),
        get_nested(config, "data", "dataset"),
        get_nested(config, "dataset"),
    ]

    for value in candidates:
        datasets = _coerce_to_dataset_list(value)
        if datasets:
            return datasets

    raise ValueError(
        "No dataset name found. Add experiment.dataset to a run config, "
        "datasets to a suite config, or pass --datasets Coffee GunPoint."
    )


def resolve_data_root(config: dict[str, Any], cli_data_root: str | None) -> str:
    if cli_data_root:
        return cli_data_root

    candidates = [
        get_nested(config, "paths", "data_root"),
        get_nested(config, "data", "root"),
        get_nested(config, "data", "data_root"),
        get_nested(config, "data_root"),
    ]

    for value in candidates:
        if value:
            return str(value)
    return "data/UCR"


def resolve_base_url(config: dict[str, Any], cli_base_url: str | None) -> str:
    if cli_base_url:
        return cli_base_url

    candidates = [
        get_nested(config, "download", "base_url"),
        get_nested(config, "data", "base_url"),
        get_nested(config, "base_url"),
    ]
    for value in candidates:
        if value:
            return str(value)
    return DEFAULT_BASE_URL


def resolve_overwrite(config: dict[str, Any], cli_overwrite: bool) -> bool:
    if cli_overwrite:
        return True
    return bool(get_nested(config, "download", "overwrite", default=False))


def resolve_timeout(config: dict[str, Any], cli_timeout: int | None) -> int:
    if cli_timeout is not None:
        return int(cli_timeout)
    return int(get_nested(config, "download", "timeout", default=60))


def _coerce_to_dataset_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if "," in value:
            return _clean_dataset_list(value.split(","))
        return _clean_dataset_list([value])
    if isinstance(value, (list, tuple)):
        return _clean_dataset_list([str(item) for item in value])
    return []


def _clean_dataset_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        name = str(item).strip()
        if not name or name in seen:
            continue
        cleaned.append(name)
        seen.add(name)
    return cleaned


def main() -> int:
    args = parse_args()

    try:
        cfg = load_yaml_config(args.config)
        datasets = resolve_datasets(cfg, args.datasets)
        data_root = resolve_data_root(cfg, args.data_root)
        base_url = resolve_base_url(cfg, args.base_url)
        overwrite = resolve_overwrite(cfg, args.overwrite)
        timeout = resolve_timeout(cfg, args.timeout)

        print(f"Config    : {args.config}")
        print(f"Data root : {data_root}")
        print(f"Base URL  : {base_url}")
        print(f"Overwrite : {overwrite}")
        print(f"Datasets  : {', '.join(datasets)}")
        print("-" * 72)

        results = download_ucr_datasets(
            datasets=datasets,
            data_root=data_root,
            base_url=base_url,
            overwrite=overwrite,
            timeout=timeout,
        )

        for result in results:
            print(f"[{result.status}] {result.dataset}")
            print(f"  Train: {result.train_file}")
            print(f"  Test : {result.test_file}")

        print("-" * 72)
        print("Dataset download step completed successfully.")
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
