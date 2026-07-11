from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from src.utils.experiment import normalize_representations, representation_tag, short_config_hash


def minmax_per_image(images: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = []
    for img in images:
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        mn = img.min()
        mx = img.max()
        img = (img - mn) / (mx - mn + eps)
        out.append(img.astype(np.float32))
    return np.stack(out, axis=0)


def resample_series_length(X: np.ndarray, target_length: int) -> np.ndarray:
    """Resample each 1D series to a fixed length using linear interpolation.

    This is used before Recurrence Plot creation. A standard RP transformer creates
    an N x T x T tensor, so running it directly on long datasets such as FordA
    (T=500) can exhaust Colab RAM before the image is resized to 64 x 64. Resampling
    first makes RP creation memory-safe and keeps the final RP image size aligned
    with the configured image_size.
    """
    target_length = int(target_length)
    if target_length <= 1:
        raise ValueError("target_length must be > 1")
    if X.shape[1] == target_length:
        return X.astype(np.float32, copy=False)

    old_grid = np.linspace(0.0, 1.0, X.shape[1], dtype=np.float32)
    new_grid = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    out = np.empty((X.shape[0], target_length), dtype=np.float32)
    for i, x in enumerate(X):
        out[i] = np.interp(new_grid, old_grid, x).astype(np.float32)
    return out


def make_gaf(
    X: np.ndarray,
    image_size: int = 64,
    method: str = "summation",
    sample_range=(-1, 1),
) -> np.ndarray:
    """Create GAF images safely for both short and long time series.

    pyts requires an integer image_size to be less than or equal to the number of
    timestamps. For short series, GAF is therefore created at the native temporal
    resolution and then resized to the configured output size.
    """
    from pyts.image import GramianAngularField
    from skimage.transform import resize

    target_size = int(image_size)
    native_size = min(target_size, int(X.shape[1]))

    transformer = GramianAngularField(
        image_size=native_size,
        method=method,
        sample_range=tuple(sample_range) if sample_range is not None else None,
    )
    images = transformer.fit_transform(X)

    if native_size != target_size:
        images = np.stack(
            [
                resize(
                    img,
                    (target_size, target_size),
                    anti_aliasing=True,
                    preserve_range=True,
                ).astype(np.float32)
                for img in images
            ],
            axis=0,
        )

    return minmax_per_image(images)


def make_mtf(
    X: np.ndarray,
    image_size: int = 64,
    n_bins: int = 8,
    strategy: str = "quantile",
) -> np.ndarray:
    """Create MTF images safely for both short and long time series.

    pyts requires an integer image_size to be less than or equal to the number of
    timestamps. For short series, MTF is created at the native temporal resolution
    and then resized to the configured output size.
    """
    from pyts.image import MarkovTransitionField
    from skimage.transform import resize

    target_size = int(image_size)
    native_size = min(target_size, int(X.shape[1]))

    transformer = MarkovTransitionField(
        image_size=native_size,
        n_bins=n_bins,
        strategy=strategy,
    )
    images = transformer.fit_transform(X)

    if native_size != target_size:
        images = np.stack(
            [
                resize(
                    img,
                    (target_size, target_size),
                    anti_aliasing=True,
                    preserve_range=True,
                ).astype(np.float32)
                for img in images
            ],
            axis=0,
        )

    return minmax_per_image(images)


def make_rp(
    X: np.ndarray,
    image_size: int = 64,
    threshold=None,
    percentage: float = 10.0,
    pre_resample: bool = True,
) -> np.ndarray:
    """Create memory-safe Recurrence Plot images.

    If pre_resample=True, each series is first resampled to image_size and then RP
    is computed. This avoids constructing very large N x T x T arrays for long
    UCR datasets and is the recommended Colab-safe setting.
    """
    from pyts.image import RecurrencePlot
    from skimage.transform import resize

    kwargs = {}
    if threshold is not None:
        kwargs["threshold"] = threshold
        kwargs["percentage"] = percentage

    transformer = RecurrencePlot(**kwargs)

    if pre_resample:
        X_rp = resample_series_length(X, target_length=image_size)
        images = transformer.fit_transform(X_rp)
        return minmax_per_image(images.astype(np.float32, copy=False))

    images = transformer.fit_transform(X)
    resized = []
    for img in images:
        resized_img = resize(
            img,
            (image_size, image_size),
            anti_aliasing=True,
            preserve_range=True,
        )
        resized.append(resized_img.astype(np.float32))

    return minmax_per_image(np.stack(resized, axis=0))


def make_stft_images(
    X: np.ndarray,
    image_size: int = 64,
    nperseg_max: int = 32,
    noverlap_ratio: float = 0.5,
    log_scale: bool = True,
) -> np.ndarray:
    from scipy.signal import stft
    from skimage.transform import resize

    images = []

    for x in X:
        T = len(x)
        nperseg = min(nperseg_max, T)
        noverlap = int(max(0, nperseg * noverlap_ratio))
        noverlap = min(noverlap, nperseg - 1) if nperseg > 1 else 0

        _, _, Zxx = stft(x, nperseg=nperseg, noverlap=noverlap, boundary=None)
        S = np.abs(Zxx)
        if log_scale:
            S = np.log1p(S)

        img = resize(
            S,
            (image_size, image_size),
            anti_aliasing=True,
            preserve_range=True,
        )
        images.append(img.astype(np.float32))

    return minmax_per_image(np.stack(images, axis=0))


def _make_one_representation(rep: str, X: np.ndarray, image_size: int, transform_cfg: Dict) -> np.ndarray:
    rep = rep.upper()
    if rep == "GAF":
        cfg = transform_cfg.get("gaf", {})
        return make_gaf(
            X,
            image_size=image_size,
            method=cfg.get("method", "summation"),
            sample_range=cfg.get("sample_range", [-1, 1]),
        )
    if rep == "MTF":
        cfg = transform_cfg.get("mtf", {})
        return make_mtf(
            X,
            image_size=image_size,
            n_bins=int(cfg.get("n_bins", 8)),
            strategy=cfg.get("strategy", "quantile"),
        )
    if rep == "RP":
        cfg = transform_cfg.get("rp", {})
        return make_rp(
            X,
            image_size=image_size,
            threshold=cfg.get("threshold", None),
            percentage=float(cfg.get("percentage", 10.0)),
            pre_resample=bool(cfg.get("pre_resample", True)),
        )
    if rep == "STFT":
        cfg = transform_cfg.get("stft", {})
        return make_stft_images(
            X,
            image_size=image_size,
            nperseg_max=int(cfg.get("nperseg_max", 32)),
            noverlap_ratio=float(cfg.get("noverlap_ratio", 0.5)),
            log_scale=bool(cfg.get("log_scale", True)),
        )
    raise ValueError(f"Unknown representation: {rep}")


def build_or_load_2d_representations(
    X_train: np.ndarray,
    X_test: np.ndarray,
    dataset_name: str,
    image_size: int,
    cache_root: str | Path,
    transform_cfg: Dict,
    representations: Iterable[str] = ("GAF", "MTF", "RP", "STFT"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Create or load cached 2D representations.

    Output shape:
        X_train_img: (N_train, M, H, W)
        X_test_img:  (N_test, M, H, W)

    M is determined by the representation list. This enables ablation configs
    such as [MTF, RP, STFT] without changing source code.
    """
    reps = normalize_representations(representations)

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_descriptor = {
        "dataset": dataset_name,
        "image_size": image_size,
        "representations": reps,
        "transform": transform_cfg,
    }
    cfg_hash = short_config_hash(cache_descriptor)
    rep_tag = representation_tag(reps)
    cache_path = cache_root / f"{dataset_name}_img{image_size}_{rep_tag}_{cfg_hash}.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        return data["X_train_img"], data["X_test_img"]

    train_images = []
    test_images = []

    for rep in reps:
        print(f"Creating {rep} representation...")
        train_images.append(_make_one_representation(rep, X_train, image_size, transform_cfg))
        test_images.append(_make_one_representation(rep, X_test, image_size, transform_cfg))

    X_train_img = np.stack(train_images, axis=1).astype(np.float32)
    X_test_img = np.stack(test_images, axis=1).astype(np.float32)

    np.savez_compressed(
        cache_path,
        X_train_img=X_train_img,
        X_test_img=X_test_img,
        representations=np.array(reps),
        cache_descriptor=np.array([str(cache_descriptor)]),
    )
    return X_train_img, X_test_img
