from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.sparse as sp


def generate_synthetic(n_cells=5000, n_peaks=10000, n_genes=200, sparsity=0.995, seed=123):
    rng = np.random.default_rng(seed)
    # Sparse binary ATAC peaks with TF-like structure
    data = rng.random(int(n_cells * n_peaks * (1 - sparsity)))
    data[:] = 1.0
    rows = rng.integers(0, n_cells, size=data.size)
    cols = rng.integers(0, n_peaks, size=data.size)
    peaks = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_peaks))

    # Random regulatory mapping: peaks -> genes (sparse weights)
    k_peaks_per_gene = max(10, n_peaks // 500)
    rows = []
    cols = []
    vals = []
    for g in range(n_genes):
        p_idx = rng.choice(n_peaks, size=k_peaks_per_gene, replace=False)
        rows.extend([g] * k_peaks_per_gene)
        cols.extend(p_idx.tolist())
        vals.extend(rng.normal(loc=0.5, scale=0.2, size=k_peaks_per_gene))
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n_genes, n_peaks))

    # Latent linear mapping + noise; then log1p target
    signal = peaks @ W.T  # (cells x genes)
    signal = signal.toarray()
    signal += rng.normal(0, 0.5, size=signal.shape)
    signal = np.clip(signal, a_min=0.0, a_max=None)
    # Convert to counts-like by exponential and poisson sampling to be more realistic
    lam = np.exp(signal / signal.std())
    rna = rng.poisson(lam).astype(np.int32)

    return peaks, rna


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic multiome dataset")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cells", type=int, default=4000)
    ap.add_argument("--peaks", type=int, default=8000)
    ap.add_argument("--genes", type=int, default=200)
    ap.add_argument("--sparsity", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    peaks, rna = generate_synthetic(
        n_cells=args.cells,
        n_peaks=args.peaks,
        n_genes=args.genes,
        sparsity=args.sparsity,
        seed=args.seed,
    )

    sp.save_npz(os.path.join(args.outdir, "peaks.npz"), peaks)
    sp.save_npz(os.path.join(args.outdir, "rna.npz"), sp.csr_matrix(rna))
    with open(os.path.join(args.outdir, "genes.txt"), "w") as f:
        for i in range(rna.shape[1]):
            f.write(f"Gene{i}\n")
    with open(os.path.join(args.outdir, "peaks.txt"), "w") as f:
        for i in range(peaks.shape[1]):
            f.write(f"chr1:{i*1000}-{i*1000+500}\n")

    print(f"Wrote synthetic data to {args.outdir}")


if __name__ == "__main__":
    main()
