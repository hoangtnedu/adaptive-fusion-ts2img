from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.config import deep_update, load_config, parse_value, save_config, set_nested


def read_suite(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_command(cmd, dry_run: bool):
    print("Running:", " ".join(map(str, cmd)))
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_config(config_path: str | Path, dry_run: bool, no_resume: bool = False):
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    if no_resume:
        cmd.append("--no-resume")
    run_command(cmd, dry_run)


def resolve_path(base_dir: Path, maybe_path: str | Path):
    p = Path(maybe_path)
    if p.is_absolute() or p.exists():
        return p
    return (base_dir / p).resolve()


def apply_cli_set_overrides(cfg: Dict[str, Any], overrides: list[str]):
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set item. Expected key=value, got: {item}")
        key, value = item.split("=", 1)
        set_nested(cfg, key, parse_value(value))
    return cfg


def method_name(method: Dict[str, Any]) -> str:
    name = method.get("name")
    if name:
        return str(name)
    exp_name = method.get("override", {}).get("experiment", {}).get("name")
    if exp_name:
        return str(exp_name)
    return "method"


def generate_and_run(base_cfg, datasets, seeds, suite_overrides, methods, tmp_dir, dry_run, no_resume, cli_sets):
    methods = methods or [{"name": base_cfg.get("experiment", {}).get("name", "adaptive_fusion_full"), "override": {}}]

    for method in methods:
        m_name = method_name(method)
        m_override = method.get("override", {}) or {}

        for dataset in datasets:
            for seed in seeds:
                cfg = deep_update(base_cfg, suite_overrides or {})
                cfg = deep_update(cfg, m_override)
                cfg.setdefault("project", {})["seed"] = int(seed)
                cfg.setdefault("experiment", {})["dataset"] = dataset
                cfg["experiment"].setdefault("name", m_name)
                cfg = apply_cli_set_overrides(cfg, cli_sets)

                exp_name = cfg["experiment"].get("name", m_name)
                tmp_config = tmp_dir / f"{dataset}_seed{seed}_{exp_name}.yaml"
                save_config(cfg, tmp_config)
                run_config(tmp_config, dry_run=dry_run, no_resume=no_resume)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-stage experiment suites for datasets/seeds/baselines.")
    parser.add_argument("--suite", default=None, help="YAML suite file, e.g. config/suites/smoke_5datasets_3seeds.yaml")
    parser.add_argument("--base-config", default="config/default.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--configs", nargs="+", help="Explicit config files to run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], help="Dot overrides for generated configs, e.g. training.epochs=30")
    args = parser.parse_args()

    if args.configs:
        for cfg_path in args.configs:
            run_config(cfg_path, args.dry_run, no_resume=args.no_resume)
        return

    if args.suite:
        suite_path = Path(args.suite)
        suite = read_suite(suite_path)

        if "configs" in suite:
            for cfg_path in suite["configs"]:
                cfg_path = resolve_path(suite_path.parent, cfg_path)
                run_config(cfg_path, args.dry_run, no_resume=args.no_resume)
            return

        base_config = suite.get("base_config", args.base_config)
        base_path = resolve_path(suite_path.parent, base_config)
        datasets = suite.get("datasets", [])
        seeds = suite.get("seeds", [])
        suite_overrides = suite.get("overrides", {}) or {}
        methods = suite.get("methods", []) or None
    else:
        base_path = Path(args.base_config)
        datasets = args.datasets or []
        seeds = args.seeds or []
        suite_overrides = {}
        methods = None

    if not datasets or not seeds:
        raise ValueError("Provide either --configs, a suite file, or both --datasets and --seeds.")

    base_cfg = load_config(base_path)
    tmp_dir = Path(tempfile.gettempdir()) / "adaptive_fusion_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    generate_and_run(
        base_cfg=base_cfg,
        datasets=datasets,
        seeds=seeds,
        suite_overrides=suite_overrides,
        methods=methods,
        tmp_dir=tmp_dir,
        dry_run=args.dry_run,
        no_resume=args.no_resume,
        cli_sets=args.set,
    )


if __name__ == "__main__":
    main()
