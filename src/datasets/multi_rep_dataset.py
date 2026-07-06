import numpy as np
import torch
from torch.utils.data import Dataset


class MultiRepresentationDataset(Dataset):
    """Dataset cho input nhiều biểu diễn ảnh 2D.

    Input numpy:
        X_img: (N, 4, H, W)
        y:     (N,)

    Output torch:
        x:     (4, 1, H, W)
        y:     scalar long
    """

    def __init__(self, X_img: np.ndarray, y: np.ndarray):
        self.X_img = X_img.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X_img[idx]
        x = torch.from_numpy(x).unsqueeze(1)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
