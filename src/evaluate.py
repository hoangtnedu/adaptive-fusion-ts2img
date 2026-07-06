from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _unpack_model_output(output):
    if isinstance(output, tuple):
        if len(output) == 2:
            return output
        if len(output) > 2:
            return output[0], output[1]
    return output, None


def evaluate_model(model, loader, device, criterion=None) -> Dict[str, Any]:
    model.eval()
    all_true = []
    all_pred = []
    all_alpha = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits, alpha = _unpack_model_output(model(xb))

            if criterion is not None:
                loss = criterion(logits, yb)
                total_loss += loss.item() * yb.size(0)
                total_count += yb.size(0)

            pred = torch.argmax(logits, dim=1)
            all_true.extend(yb.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            if alpha is not None:
                all_alpha.append(alpha.detach().cpu().numpy())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_alpha_array = np.concatenate(all_alpha, axis=0) if all_alpha else None

    avg_loss = None
    if criterion is not None and total_count > 0:
        avg_loss = total_loss / total_count

    return {
        "loss": avg_loss,
        "acc": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro")),
        "precision": float(precision_score(all_true, all_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(all_true, all_pred, average="macro", zero_division=0)),
        "y_true": all_true,
        "y_pred": all_pred,
        "alpha": all_alpha_array,
    }
