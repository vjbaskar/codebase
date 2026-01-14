from __future__ import annotations

import numpy as np
import torch


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    with torch.no_grad():
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def pearsonr_per_gene(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Vectorized Pearson correlation per gene (columns). Returns array shape [n_genes]."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Center
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = np.sum(yt * yp, axis=0)
    den = np.sqrt(np.sum(yt ** 2, axis=0) * np.sum(yp ** 2, axis=0)) + 1e-8
    r = num / den
    return r


def overall_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = pearsonr_per_gene(y_true, y_pred)
    # return mean correlation across genes
    return float(np.nanmean(r))
