from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.ucr_loader import load_ucr_dataset
from src.datasets import MultiRepresentationDataset, TimeSeriesDataset
from src.evaluate import evaluate_model
from src.models import AdaptiveFusionCNN, CNN1D
from src.transforms.ts2image import build_or_load_2d_representations
from src.utils.config import load_config, save_config
from src.utils.experiment import collect_environment, make_run_name, normalize_representations
from src.utils.metrics import (
    save_classification_report,
    save_confusion_matrix,
    save_json,
    save_learning_curve,
)
from src.utils.seed import set_seed


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_f1, bad_epochs, history, cfg):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
            "bad_epochs": bad_epochs,
            "history": history,
            "config": cfg,
        },
        path,
    )


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_train_validation(y_train: np.ndarray, validation_ratio: float, seed: int):
    indices = np.arange(len(y_train))
    try:
        return train_test_split(
            indices,
            test_size=validation_ratio,
            random_state=seed,
            stratify=y_train,
        )
    except ValueError:
        return train_test_split(
            indices,
            test_size=validation_ratio,
            random_state=seed,
            stratify=None,
        )


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_loaders_2d(cfg, X_train_img, y_train, X_test_img, y_test, seed):
    validation_ratio = float(cfg["experiment"].get("validation_ratio", 0.2))
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 0))

    train_idx, val_idx = split_train_validation(y_train, validation_ratio, seed)

    train_ds = MultiRepresentationDataset(X_train_img[train_idx], y_train[train_idx])
    val_ds = MultiRepresentationDataset(X_train_img[val_idx], y_train[val_idx])
    test_ds = MultiRepresentationDataset(X_test_img, y_test)

    return (
        make_loader(train_ds, batch_size, num_workers, shuffle=True),
        make_loader(val_ds, batch_size, num_workers, shuffle=False),
        make_loader(test_ds, batch_size, num_workers, shuffle=False),
    )


def build_loaders_1d(cfg, X_train_raw, y_train, X_test_raw, y_test, seed):
    validation_ratio = float(cfg["experiment"].get("validation_ratio", 0.2))
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 0))

    train_idx, val_idx = split_train_validation(y_train, validation_ratio, seed)

    train_ds = TimeSeriesDataset(X_train_raw[train_idx], y_train[train_idx])
    val_ds = TimeSeriesDataset(X_train_raw[val_idx], y_train[val_idx])
    test_ds = TimeSeriesDataset(X_test_raw, y_test)

    return (
        make_loader(train_ds, batch_size, num_workers, shuffle=True),
        make_loader(val_ds, batch_size, num_workers, shuffle=False),
        make_loader(test_ds, batch_size, num_workers, shuffle=False),
    )


def build_model(cfg: Dict[str, Any], num_classes: int, num_representations: int) -> nn.Module:
    model_cfg = cfg.get("model", {})
    name = str(model_cfg.get("name", "AdaptiveFusionCNN"))

    if name.lower() in {"cnn1d", "1dcnn", "1d-cnn"}:
        return CNN1D(
            num_classes=num_classes,
            in_channels=int(model_cfg.get("in_channels", 1)),
            hidden_channels=tuple(model_cfg.get("hidden_channels", [32, 64])),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )

    if name.lower() in {"adaptivefusioncnn", "adaptive_fusion_cnn", "light2dfusioncnn"}:
        return AdaptiveFusionCNN(
            num_classes=num_classes,
            num_representations=num_representations,
            feature_dim=int(model_cfg.get("feature_dim", 128)),
            gate_hidden=int(model_cfg.get("gate_hidden", 128)),
            dropout=float(model_cfg.get("dropout", 0.3)),
            fusion=str(model_cfg.get("fusion", "adaptive_gating")),
        )

    raise ValueError(f"Unsupported model.name: {name}")


def compute_complexity(model: nn.Module, cfg: Dict[str, Any], device: torch.device, num_representations: int, raw_length: int | None):
    result = {"trainable_params": int(count_trainable_params(model)), "params": None, "flops": None}
    try:
        from thop import profile

        model_cfg = cfg.get("model", {})
        name = str(model_cfg.get("name", "AdaptiveFusionCNN")).lower()
        if name in {"cnn1d", "1dcnn", "1d-cnn"}:
            if raw_length is None:
                raise ValueError("raw_length is required for CNN1D complexity")
            dummy = torch.randn(1, 1, raw_length).to(device)
        else:
            image_size = int(cfg["experiment"].get("image_size", 64))
            dummy = torch.randn(1, num_representations, 1, image_size, image_size).to(device)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        result.update({"params": int(params), "flops": float(flops)})
    except Exception as e:
        result["error"] = str(e)
    return result


def benchmark_inference(model: nn.Module, loader: DataLoader, device: torch.device, repeats: int = 3, warmup_batches: int = 2):
    model.eval()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for batch_idx, (xb, _) in enumerate(loader):
            if batch_idx >= warmup_batches:
                break
            xb = xb.to(device, non_blocking=True)
            _ = model(xb)
        sync()

        total_time = 0.0
        total_samples = 0
        for _ in range(max(1, repeats)):
            sync()
            t0 = time.perf_counter()
            n = 0
            for xb, _ in loader:
                xb = xb.to(device, non_blocking=True)
                _ = model(xb)
                n += xb.size(0)
            sync()
            total_time += time.perf_counter() - t0
            total_samples += n

    ms_per_sample = 1000.0 * total_time / max(total_samples, 1)
    return {
        "device": str(device),
        "repeats": int(repeats),
        "warmup_batches": int(warmup_batches),
        "total_time_sec": float(total_time),
        "total_samples": int(total_samples),
        "ms_per_sample": float(ms_per_sample),
    }


def save_alpha_outputs(cfg, test_metrics, rep_names, run_dir: Path):
    if not cfg.get("evaluation", {}).get("save_alpha", True):
        return None
    if test_metrics.get("alpha") is None:
        return None

    alpha_df = pd.DataFrame(test_metrics["alpha"], columns=[f"alpha_{r}" for r in rep_names])
    alpha_df.insert(0, "y_true", test_metrics["y_true"])
    alpha_df.insert(1, "y_pred", test_metrics["y_pred"])
    alpha_df.to_csv(run_dir / "alpha_test_samples.csv", index=False)

    alpha_mean = alpha_df[[f"alpha_{r}" for r in rep_names]].mean()
    alpha_mean.to_csv(run_dir / "alpha_mean.csv")
    return alpha_mean


def main():
    parser = argparse.ArgumentParser(description="Train paper-grade time-series classification experiments.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing last checkpoint for this run.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    dataset_name = cfg["experiment"]["dataset"]
    image_size = int(cfg["experiment"].get("image_size", 64))
    input_mode = str(cfg.get("data", {}).get("input_mode", "2d_representations"))

    rep_names = []
    if input_mode != "raw_1d":
        rep_names = normalize_representations(cfg["experiment"].get("representations", ["GAF", "MTF", "RP", "STFT"]))

    data_root = Path(cfg["paths"]["data_root"])
    output_root = Path(cfg["paths"]["output_root"])
    cache_root = Path(cfg["paths"]["cache_root"])

    run_name = make_run_name(cfg)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_name}")
    print(f"Experiment: {cfg.get('experiment', {}).get('name', 'experiment')}")
    print(f"Input mode: {input_mode}")
    if rep_names:
        print(f"Representations: {rep_names}")
    print(f"Run directory: {run_dir}")

    if cfg.get("reporting", {}).get("save_config_used", True):
        save_config(cfg, run_dir / "config_used.yaml")

    if cfg.get("reporting", {}).get("save_environment", True):
        save_json(collect_environment(project_root=Path.cwd()), run_dir / "environment.json")

    X_train_raw, y_train, X_test_raw, y_test, label_encoder = load_ucr_dataset(data_root, dataset_name)
    num_classes = len(label_encoder.classes_)
    class_names = [str(c) for c in label_encoder.classes_]
    raw_length = int(X_train_raw.shape[1])

    if input_mode == "raw_1d":
        train_loader, val_loader, test_loader = build_loaders_1d(
            cfg, X_train_raw, y_train, X_test_raw, y_test, seed
        )
        num_representations = 0
    else:
        X_train_img, X_test_img = build_or_load_2d_representations(
            X_train=X_train_raw,
            X_test=X_test_raw,
            dataset_name=dataset_name,
            image_size=image_size,
            cache_root=cache_root,
            transform_cfg=cfg.get("transform", {}),
            representations=rep_names,
        )
        train_loader, val_loader, test_loader = build_loaders_2d(
            cfg, X_train_img, y_train, X_test_img, y_test, seed
        )
        num_representations = len(rep_names)

    device = get_device(cfg["training"].get("device", "auto"))
    print("Device:", device)

    model = build_model(cfg, num_classes=num_classes, num_representations=max(1, num_representations)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg["training"].get("scheduler_factor", 0.5)),
        patience=int(cfg["training"].get("scheduler_patience", 5)),
    )

    checkpoint_cfg = cfg.get("checkpoint", {})
    last_ckpt = run_dir / checkpoint_cfg.get("last_name", "last_checkpoint.pt")
    best_ckpt = run_dir / checkpoint_cfg.get("best_name", "best_model.pt")
    history_csv = run_dir / "history.csv"

    start_epoch = 1
    best_val_f1 = -1.0
    bad_epochs = 0
    history = []

    resume_enabled = bool(checkpoint_cfg.get("resume", True)) and not args.no_resume
    if resume_enabled and last_ckpt.exists():
        print(f"Resume from checkpoint: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        bad_epochs = int(ckpt.get("bad_epochs", 0))
        history = ckpt.get("history", [])
        print(f"Start epoch: {start_epoch}, best val Macro-F1: {best_val_f1:.4f}")

    epochs = int(cfg["training"].get("epochs", 100))
    patience = int(cfg["training"].get("patience", 20))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        t0 = time.time()

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * yb.size(0)
            n_samples += yb.size(0)

        train_loss = epoch_loss / max(n_samples, 1)
        train_metrics = evaluate_model(model, train_loader, device, criterion)
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        scheduler.step(val_metrics["macro_f1"])

        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": elapsed,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_csv, index=False)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        save_checkpoint(last_ckpt, epoch, model, optimizer, scheduler, best_val_f1, bad_epochs, history, cfg)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            bad_epochs = 0
            save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, best_val_f1, bad_epochs, history, cfg)
            print("Saved best model.")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping.")
            break

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint was not created: {best_ckpt}")

    print("Loading best model for final test evaluation...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate_model(model, test_loader, device, criterion)

    complexity = compute_complexity(model, cfg, device, num_representations=max(1, num_representations), raw_length=raw_length)
    save_json(complexity, run_dir / "complexity.json")

    benchmark_cfg = cfg.get("evaluation", {})
    if bool(benchmark_cfg.get("benchmark_inference", True)):
        inference = benchmark_inference(
            model,
            test_loader,
            device,
            repeats=int(benchmark_cfg.get("benchmark_repeats", 3)),
            warmup_batches=int(benchmark_cfg.get("benchmark_warmup_batches", 2)),
        )
        save_json(inference, run_dir / "inference_time.json")
    else:
        inference = None

    summary = {
        "dataset": dataset_name,
        "experiment": cfg.get("experiment", {}).get("name", "experiment"),
        "model": cfg.get("model", {}).get("name", "AdaptiveFusionCNN"),
        "fusion": None if input_mode == "raw_1d" else cfg.get("model", {}).get("fusion", None),
        "input_mode": input_mode,
        "run_name": run_name,
        "image_size": image_size if input_mode != "raw_1d" else None,
        "seed": seed,
        "num_classes": num_classes,
        "classes": class_names,
        "representations": rep_names,
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "best_val_f1": float(ckpt["best_val_f1"]),
        "best_epoch": int(ckpt["epoch"]),
        "trainable_params": int(complexity.get("trainable_params", count_trainable_params(model))),
        "params": complexity.get("params"),
        "flops": complexity.get("flops"),
        "inference_ms_per_sample": None if inference is None else inference["ms_per_sample"],
        "total_train_time_sec": float(sum(float(r.get("time_sec", 0.0)) for r in history)),
    }

    alpha_mean = save_alpha_outputs(cfg, test_metrics, rep_names, run_dir)
    if alpha_mean is not None:
        for key, value in alpha_mean.to_dict().items():
            summary[key] = float(value)

    save_json(summary, run_dir / "summary.json")
    save_classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        class_names,
        run_dir / "classification_report.txt",
    )
    if cfg.get("evaluation", {}).get("save_confusion_matrix", True):
        save_confusion_matrix(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            class_names,
            run_dir / "confusion_matrix.png",
            title=f"Confusion Matrix - {dataset_name}",
        )
    if cfg.get("evaluation", {}).get("save_learning_curve", True):
        save_learning_curve(
            history_csv,
            run_dir / "learning_curve_macro_f1.png",
            title=f"Learning Curve - {dataset_name}",
        )

    print("Final test summary:")
    print(summary)
    if alpha_mean is not None:
        print("Mean fusion weights:")
        print(alpha_mean)

    if cfg.get("reporting", {}).get("save_environment", True):
        env = collect_environment(project_root=Path.cwd())
        env["finished_at_utc"] = env.pop("created_at_utc")
        save_json(env, run_dir / "environment_finished.json")


if __name__ == "__main__":
    main()
