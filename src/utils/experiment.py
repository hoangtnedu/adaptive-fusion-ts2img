from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


VALID_REPRESENTATIONS = ("GAF", "MTF", "RP", "STFT")


def normalize_representations(representations: Iterable[str]) -> List[str]:
    reps = [str(r).upper() for r in representations]
    unknown = [r for r in reps if r not in VALID_REPRESENTATIONS]
    if unknown:
        raise ValueError(f"Unknown representations: {unknown}. Valid: {VALID_REPRESENTATIONS}")
    if not reps:
        raise ValueError("At least one representation is required.")
    if len(reps) != len(set(reps)):
        raise ValueError(f"Duplicate representations are not allowed: {reps}")
    return reps


def representation_tag(representations: Iterable[str]) -> str:
    return "_".join(normalize_representations(representations))


def short_config_hash(obj: Any, length: int = 8) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def make_run_name(cfg: Dict[str, Any]) -> str:
    experiment = cfg.get("experiment", {})
    project = cfg.get("project", {})

    dataset = experiment.get("dataset", "dataset")
    image_size = int(experiment.get("image_size", 64))
    seed = int(project.get("seed", 42))
    exp_name = experiment.get("name", "experiment")
    reps = representation_tag(experiment.get("representations", ["GAF", "MTF", "RP", "STFT"]))

    template = experiment.get(
        "run_name_template",
        "{dataset}_{experiment}_img{image_size}_seed{seed}",
    )

    return template.format(
        dataset=dataset,
        experiment=exp_name,
        image_size=image_size,
        seed=seed,
        representations=reps,
    )


def summary_path_for_config(cfg: Dict[str, Any]) -> Path:
    output_root = Path(cfg["paths"]["output_root"])
    return output_root / make_run_name(cfg) / "summary.json"


def validate_completed_summary(cfg: Dict[str, Any]) -> Tuple[bool, Path, str]:
    """Check whether the run has a valid final summary matching this config.

    A file merely named ``summary.json`` is not sufficient. Identity fields and
    final test metrics must be present and valid. Corrupt or partial summaries
    are treated as incomplete so checkpoint resume can repair them.
    """

    path = summary_path_for_config(cfg)
    if not path.is_file():
        return False, path, "summary missing"

    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, path, f"summary unreadable: {exc}"

    if not isinstance(summary, dict):
        return False, path, "summary is not a JSON object"

    expected = {
        "dataset": str(cfg.get("experiment", {}).get("dataset", "dataset")),
        "experiment": str(cfg.get("experiment", {}).get("name", "experiment")),
        "run_name": make_run_name(cfg),
        "seed": int(cfg.get("project", {}).get("seed", 42)),
    }

    for key, value in expected.items():
        actual = summary.get(key)
        if key == "seed":
            try:
                actual = int(actual)
            except (TypeError, ValueError):
                return False, path, f"invalid {key}: {summary.get(key)!r}"
        else:
            actual = str(actual)
        if actual != value:
            return False, path, f"{key} mismatch: expected {value!r}, got {actual!r}"

    for key in ("test_acc", "test_macro_f1", "test_precision", "test_recall"):
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, path, f"missing or non-numeric metric: {key}"
        if not math.isfinite(float(value)):
            return False, path, f"non-finite metric: {key}"

    return True, path, "valid completed summary"


def get_git_info(project_root: str | Path = ".") -> Dict[str, Any]:
    root = Path(project_root)

    def run_git(args):
        try:
            out = subprocess.check_output(
                ["git", *args],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return out.strip()
        except Exception:
            return None

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": bool(run_git(["status", "--porcelain"])),
        "remote": run_git(["remote", "get-url", "origin"]),
    }


def collect_environment(project_root: str | Path = ".") -> Dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git": get_git_info(project_root),
    }
