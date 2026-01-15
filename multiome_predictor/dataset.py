from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List
import sys
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from torch.utils.data import Dataset


def _ensure_csr(mat: sp.spmatrix) -> sp.csr_matrix:
    if sp.isspmatrix_csr(mat):
        return mat
    return mat.tocsr()


def load_sparse_or_dense(path: str) -> np.ndarray | sp.csr_matrix:
    """Load npz (sparse csr) or npy (dense) matrix."""
    if path.endswith(".npz"):
        try:
            m = sp.load_npz(path)
            return _ensure_csr(m)
        except Exception:
            # Could be a dense np.savez file
            data = np.load(path)
            # Heuristics: if contains 'arr_0'
            key = "arr_0" if "arr_0" in data.files else data.files[0]
            return data[key]
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


@dataclass
class LSIParams:
    n_components: int = 128
    use_idf: bool = True
    norm: Optional[str] = None  # L2 norm handled by StandardScaler here


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def train_val_test_split(n: int, val: float = 0.1, test: float = 0.1, seed: int = 42) -> Split:
    assert 0 < val < 1 and 0 < test < 1 and val + test < 1
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test)
    n_val = int(n * val)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return Split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


class ATACLSITransformer:
    """TF-IDF + TruncatedSVD for ATAC peaks.

    Fit on train-only and reuse on val/test. Save components with joblib.
    """

    def __init__(self, params: LSIParams):
        self.params = params
        self.tfidf = TfidfTransformer(use_idf=params.use_idf, norm=None)
        self.svd = TruncatedSVD(n_components=params.n_components, random_state=42)
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.fitted = False

    def fit(self, peaks_csr: sp.csr_matrix) -> "ATACLSITransformer":
        tfidf = self.tfidf.fit_transform(peaks_csr)
        lsi = self.svd.fit_transform(tfidf)
        lsi = self.scaler.fit_transform(lsi)
        self.fitted = True
        return self

    def transform(self, peaks_csr: sp.csr_matrix) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ATACLSITransformer not fitted")
        tfidf = self.tfidf.transform(peaks_csr)
        lsi = self.svd.transform(tfidf)
        lsi = self.scaler.transform(lsi)
        return lsi.astype(np.float32)

    def save(self, outdir: str) -> None:
        os.makedirs(outdir, exist_ok=True)
        joblib.dump(self.tfidf, os.path.join(outdir, "tfidf.joblib"))
        joblib.dump(self.svd, os.path.join(outdir, "svd.joblib"))
        joblib.dump(self.scaler, os.path.join(outdir, "scaler.joblib"))
        with open(os.path.join(outdir, "lsi_params.json"), "w") as f:
            json.dump({"n_components": self.params.n_components, "use_idf": self.params.use_idf}, f)

    @classmethod
    def load(cls, indir: str) -> "ATACLSITransformer":
        with open(os.path.join(indir, "lsi_params.json")) as f:
            p = json.load(f)
        obj = cls(LSIParams(n_components=p["n_components"], use_idf=p["use_idf"]))
        obj.tfidf = joblib.load(os.path.join(indir, "tfidf.joblib"))
        obj.svd = joblib.load(os.path.join(indir, "svd.joblib"))
        obj.scaler = joblib.load(os.path.join(indir, "scaler.joblib"))
        obj.fitted = True
        return obj


def log1p_and_to_dense(mat: np.ndarray | sp.csr_matrix) -> np.ndarray:
    if sp.issparse(mat):
        mat = mat.tocoo()
        mat.data = np.log1p(mat.data)
        mat = mat.tocsr().astype(np.float32)
        return mat.toarray()
    else:
        return np.log1p(mat).astype(np.float32)


def select_top_genes_by_variance(y_dense: np.ndarray, n_genes: Optional[int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (y_selected, gene_indices). If n_genes is None, returns original and None."""
    if n_genes is None or n_genes >= y_dense.shape[1]:
        return y_dense, None
    vars_ = y_dense.var(axis=0)
    top_idx = np.argsort(vars_)[::-1][:n_genes]
    return y_dense[:, top_idx], top_idx


class MultiomeDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, indices: np.ndarray):
        self.X = X
        self.Y = Y
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return self.indices.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[i]
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y[idx])
        return x, y


def build_datasets(
    peaks_path: str,
    rna_path: str,
    lsi_dim: int = 128,
    val: float = 0.1,
    test: float = 0.1,
    n_genes: Optional[int] = None,
    seed: int = 42,
    lsi_dir: Optional[str] = None,
    lsi_pretrained_dir: Optional[str] = None,
) -> Tuple[MultiomeDataset, MultiomeDataset, MultiomeDataset, dict]:
    """
    Load data, fit TF-IDF+LSI on train, select top genes, and prepare datasets.

    Returns: train, val, test datasets and metadata dict.
    """
    peaks = load_sparse_or_dense(peaks_path)
    rna = load_sparse_or_dense(rna_path)

    if not sp.isspmatrix(peaks):
        raise ValueError("peaks must be a sparse CSR matrix stored as .npz")
    peaks = _ensure_csr(peaks)

    y = log1p_and_to_dense(rna)  # predict log1p expression

    n_cells = peaks.shape[0]
    split = train_val_test_split(n_cells, val=val, test=test, seed=seed)

    # Fit LSI on train split only, or load pretrained
    lsi_params = LSIParams(n_components=lsi_dim)
    if lsi_pretrained_dir is not None:
        lsi = ATACLSITransformer.load(lsi_pretrained_dir)
    else:
        lsi = ATACLSITransformer(lsi_params)
        lsi.fit(peaks[split.train_idx])
        if lsi_dir is not None:
            lsi.save(lsi_dir)
    X = lsi.transform(peaks)

    # Gene selection by variance on train only
    y_train = y[split.train_idx]
    y_sel, gene_idx = select_top_genes_by_variance(y, n_genes)

    meta = {
        "n_cells": int(n_cells),
        "n_peaks": int(peaks.shape[1]),
        "n_genes_total": int(y.shape[1]),
        "gene_idx": None if gene_idx is None else gene_idx.tolist(),
        "lsi_dim": int(X.shape[1]),
    }

    train_ds = MultiomeDataset(X, y_sel, split.train_idx)
    val_ds = MultiomeDataset(X, y_sel, split.val_idx)
    test_ds = MultiomeDataset(X, y_sel, split.test_idx)
    return train_ds, val_ds, test_ds, meta
