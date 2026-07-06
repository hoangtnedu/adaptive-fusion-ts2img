from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Dataset for raw 1D time-series baseline.

    Input numpy:
        X: (N, T)
        y: (N,)
    Output torch:
        x: (1, T)
        y: scalar long
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
