from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.ucr_loader import load_ucr_dataset
from src.datasets import MultiRepresentationDataset, TimeSeriesDataset
from src.models import AdaptiveFusionCNN, CNN1D
from src.train import get_device, benchmark_inference
from src.transforms.ts2image import build_or_load_2d_representations
from src.utils.config import load_config
from src.utils.experiment import make_run_name, normalize_representations
from src.utils.metrics import save_json
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference time for a saved best_model.pt.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["training"].get("device", "auto"))
    dataset_name = cfg["experiment"]["dataset"]
    input_mode = cfg.get("data", {}).get("input_mode", "2d_representations")
    data_root = Path(cfg["paths"]["data_root"])
    cache_root = Path(cfg["paths"]["cache_root"])
    output_root = Path(cfg["paths"]["output_root"])
    run_dir = output_root / make_run_name(cfg)

    X_train, y_train, X_test, y_test, le = load_ucr_dataset(data_root, dataset_name)
    num_classes = len(le.classes_)
    batch_size = args.batch_size or int(cfg["training"].get("batch_size", 32))

    if input_mode == "raw_1d":
        dataset = TimeSeriesDataset(X_test, y_test)
        model = CNN1D(
            num_classes=num_classes,
            in_channels=int(cfg.get("model", {}).get("in_channels", 1)),
            hidden_channels=tuple(cfg.get("model", {}).get("hidden_channels", [32, 64])),
            dropout=float(cfg.get("model", {}).get("dropout", 0.3)),
        )
    else:
        reps = normalize_representations(cfg["experiment"].get("representations", ["GAF", "MTF", "RP", "STFT"]))
        X_train_img, X_test_img = build_or_load_2d_representations(
            X_train,
            X_test,
            dataset_name=dataset_name,
            image_size=int(cfg["experiment"].get("image_size", 64)),
            cache_root=cache_root,
            transform_cfg=cfg.get("transform", {}),
            representations=reps,
        )
        dataset = MultiRepresentationDataset(X_test_img, y_test)
        model = AdaptiveFusionCNN(
            num_classes=num_classes,
            num_representations=len(reps),
            feature_dim=int(cfg.get("model", {}).get("feature_dim", 128)),
            gate_hidden=int(cfg.get("model", {}).get("gate_hidden", 128)),
            dropout=float(cfg.get("model", {}).get("dropout", 0.3)),
            fusion=cfg.get("model", {}).get("fusion", "adaptive_gating"),
        )

    ckpt = torch.load(run_dir / cfg.get("checkpoint", {}).get("best_name", "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    result = benchmark_inference(model, loader, device, repeats=args.repeats)
    save_json(result, run_dir / "inference_time_external.json")
    print(result)


if __name__ == "__main__":
    main()
