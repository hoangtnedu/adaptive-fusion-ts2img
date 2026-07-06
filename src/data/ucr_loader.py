from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_ucr_tsv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Đọc dữ liệu UCR/UEA dạng TSV hoặc khoảng trắng.

    File có dạng:
        label, x_1, x_2, ..., x_T
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"UCR file not found: {path}")

    df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def zscore_per_sample(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return ((X - mean) / (std + eps)).astype(np.float32)


def load_ucr_dataset(data_root: str | Path, dataset_name: str):
    data_root = Path(data_root)
    dataset_dir = data_root / dataset_name
    train_file = dataset_dir / f"{dataset_name}_TRAIN.tsv"
    test_file = dataset_dir / f"{dataset_name}_TEST.tsv"

    X_train_raw, y_train_raw = read_ucr_tsv(train_file)
    X_test_raw, y_test_raw = read_ucr_tsv(test_file)

    X_train_raw = zscore_per_sample(X_train_raw)
    X_test_raw = zscore_per_sample(X_test_raw)

    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train_raw, y_test_raw]))

    y_train = label_encoder.transform(y_train_raw).astype(np.int64)
    y_test = label_encoder.transform(y_test_raw).astype(np.int64)

    return X_train_raw, y_train, X_test_raw, y_test, label_encoder
