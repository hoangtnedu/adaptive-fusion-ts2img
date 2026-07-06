from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


ConfigDict = Dict[str, Any]


def deep_update(base: ConfigDict, override: Mapping[str, Any]) -> ConfigDict:
    """Recursively merge override into base and return a new dictionary.

    Dictionaries are merged. Lists and scalar values are replaced. This is
    intentional for experiment configs because representations such as
    [MTF, RP, STFT] should replace the default [GAF, MTF, RP, STFT].
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> ConfigDict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_config(config_path: str | Path) -> ConfigDict:
    """Load YAML config with optional inheritance.

    Supported inheritance keys: ``base``, ``defaults`` or ``inherits``.
    The path is resolved relative to the current config file. Example:

    ```yaml
    base: ../default.yaml
    experiment:
      dataset: ECG200
    ```
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _read_yaml(config_path)
    base_key = None
    for candidate in ("base", "defaults", "inherits"):
        if candidate in cfg:
            base_key = candidate
            break

    if base_key is None:
        cfg["_config_path"] = str(config_path)
        return cfg

    base_value = cfg.pop(base_key)
    if isinstance(base_value, (str, Path)):
        base_paths = [base_value]
    elif isinstance(base_value, list):
        base_paths = base_value
    else:
        raise ValueError(f"Invalid base config declaration in {config_path}")

    merged: ConfigDict = {}
    for base_path in base_paths:
        bp = Path(base_path)
        if not bp.is_absolute():
            bp = (config_path.parent / bp).resolve()
        merged = deep_update(merged, load_config(bp))

    merged.pop("_config_path", None)
    merged = deep_update(merged, cfg)
    merged["_config_path"] = str(config_path)
    return merged


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean = copy.deepcopy(dict(cfg))
    clean.pop("_config_path", None)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)


def set_nested(cfg: ConfigDict, dotted_key: str, value: Any) -> ConfigDict:
    """Set a nested key using dot notation, e.g. training.epochs=50."""
    parts = dotted_key.split(".")
    cursor = cfg
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    return cfg


def parse_value(value: str) -> Any:
    """Parse CLI values using YAML so numbers/lists/bools work naturally."""
    return yaml.safe_load(value)
