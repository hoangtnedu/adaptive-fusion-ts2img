from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.ucr_loader import load_ucr_dataset
from src.transforms.ts2image import build_or_load_2d_representations
from src.utils.config import load_config
from src.utils.experiment import normalize_representations


def save_preview_grid(X_img: np.ndarray, reps, out_path: Path, dataset: str, sample_index: int = 0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_reps = len(reps)
    fig, axes = plt.subplots(1, n_reps, figsize=(3 * n_reps, 3))
    if n_reps == 1:
        axes = [axes]

    for i, rep in enumerate(reps):
        axes[i].imshow(X_img[sample_index, i], aspect="auto")
        axes[i].set_title(rep)
        axes[i].axis("off")

    fig.suptitle(f"2D representations - {dataset} - sample {sample_index}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save previews of configured 2D representations.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["paths"]["data_root"])
    cache_root = Path(cfg["paths"]["cache_root"])
    output_root = Path(cfg["paths"]["output_root"])
    dataset = cfg["experiment"]["dataset"]
    image_size = int(cfg["experiment"]["image_size"])
    reps = normalize_representations(cfg["experiment"].get("representations", ["GAF", "MTF", "RP", "STFT"]))

    out_dir = Path(args.out_dir) if args.out_dir else output_root / "previews" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, _, X_test, _, _ = load_ucr_dataset(data_root, dataset)

    n_train = min(args.num_samples, len(X_train))
    n_test = min(max(1, args.num_samples // 2), len(X_test))

    preview_dataset_name = f"{dataset}_preview_n{n_train}_img{image_size}"
    preview_cache_root = cache_root / "preview"

    X_train_img, _ = build_or_load_2d_representations(
        X_train=X_train[:n_train],
        X_test=X_test[:n_test],
        dataset_name=preview_dataset_name,
        image_size=image_size,
        cache_root=preview_cache_root,
        transform_cfg=cfg.get("transform", {}),
        representations=reps,
    )

    for idx in range(n_train):
        out_path = out_dir / f"sample_{idx:03d}_{'_'.join(reps)}.png"
        save_preview_grid(X_train_img, reps, out_path, dataset, sample_index=idx)
        print("Saved:", out_path)

    print("Preview completed.")


if __name__ == "__main__":
    main()
