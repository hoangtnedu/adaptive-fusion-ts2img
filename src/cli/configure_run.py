from __future__ import annotations

import argparse
from typing import Any, Dict

from src.utils.config import load_config, parse_value, save_config, set_nested
from src.utils.experiment import normalize_representations


def update_if_not_none(container: Dict[str, Any], key: str, value):
    if value is not None:
        container[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a runtime config from an inherited YAML config.")
    parser.add_argument("--base-config", default="config/default.yaml")
    parser.add_argument("--out-config", default="config/runtime.yaml")

    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--representations", nargs="+", help="Example: GAF MTF RP STFT or MTF RP STFT for ablation.")

    parser.add_argument("--data-root", type=str)
    parser.add_argument("--output-root", type=str)
    parser.add_argument("--cache-root", type=str)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--resume", choices=["true", "false"])

    parser.add_argument("--feature-dim", type=int)
    parser.add_argument("--gate-hidden", type=int)
    parser.add_argument("--dropout", type=float)

    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Optional dot overrides, e.g. --set transform.mtf.n_bins=16 training.epochs=50",
    )

    args = parser.parse_args()
    cfg = load_config(args.base_config)

    cfg.setdefault("project", {})
    cfg.setdefault("paths", {})
    cfg.setdefault("experiment", {})
    cfg.setdefault("training", {})
    cfg.setdefault("model", {})
    cfg.setdefault("checkpoint", {})

    update_if_not_none(cfg["project"], "seed", args.seed)

    update_if_not_none(cfg["experiment"], "name", args.experiment_name)
    update_if_not_none(cfg["experiment"], "dataset", args.dataset)
    update_if_not_none(cfg["experiment"], "image_size", args.image_size)
    if args.representations is not None:
        cfg["experiment"]["representations"] = normalize_representations(args.representations)

    update_if_not_none(cfg["paths"], "data_root", args.data_root)
    update_if_not_none(cfg["paths"], "output_root", args.output_root)
    update_if_not_none(cfg["paths"], "cache_root", args.cache_root)

    update_if_not_none(cfg["training"], "epochs", args.epochs)
    update_if_not_none(cfg["training"], "batch_size", args.batch_size)
    update_if_not_none(cfg["training"], "learning_rate", args.learning_rate)
    update_if_not_none(cfg["training"], "weight_decay", args.weight_decay)
    update_if_not_none(cfg["training"], "patience", args.patience)
    update_if_not_none(cfg["training"], "num_workers", args.num_workers)
    update_if_not_none(cfg["training"], "device", args.device)
    if args.resume is not None:
        cfg["checkpoint"]["resume"] = args.resume == "true"

    update_if_not_none(cfg["model"], "feature_dim", args.feature_dim)
    update_if_not_none(cfg["model"], "gate_hidden", args.gate_hidden)
    update_if_not_none(cfg["model"], "dropout", args.dropout)

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item. Expected key=value, got: {item}")
        key, value = item.split("=", 1)
        set_nested(cfg, key, parse_value(value))

    save_config(cfg, args.out_config)

    print("Runtime config saved:", args.out_config)
    print("Experiment:", cfg["experiment"].get("name"))
    print("Dataset:", cfg["experiment"].get("dataset"))
    print("Image size:", cfg["experiment"].get("image_size"))
    print("Representations:", cfg["experiment"].get("representations"))
    print("Seed:", cfg["project"].get("seed"))
    print("Data root:", cfg["paths"].get("data_root"))
    print("Output root:", cfg["paths"].get("output_root"))
    print("Cache root:", cfg["paths"].get("cache_root"))


if __name__ == "__main__":
    main()
