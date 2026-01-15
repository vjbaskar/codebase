from __future__ import annotations

import argparse
import os
from typing import Optional
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

from .dataset import ATACLSITransformer, LSIParams


def from_h5ad(
    h5ad_path: str,
    atac_key: str = "X_atac",
    rna_key: str = "X",
    outdir: Optional[str] = None,
):
    """Extract peaks and RNA matrices from an .h5ad file.
    - Considers only h5ad. No support for mudata yet.
    - RNA is taken from .X by default (dense or sparse)
    - ATAC is taken from .obsm[atac_key] if present or .layers[atac_key]
    """
    adata = ad.read_h5ad(h5ad_path)
    # RNA
    if rna_key == "X":
        rna = adata.X
    else:
        rna = adata.layers[rna_key]
    if not sp.issparse(rna):
        rna = sp.csr_matrix(rna)

    # ATAC
    if atac_key in adata.obsm:
        atac = adata.obsm[atac_key]
    elif atac_key in adata.layers:
        atac = adata.layers[atac_key]
    else:
        raise KeyError(f"Could not find ATAC in obsm/layers with key: {atac_key}")
    if not sp.issparse(atac):
        atac = sp.csr_matrix(atac)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        sp.save_npz(os.path.join(outdir, "peaks.npz"), atac.tocsr())
        sp.save_npz(os.path.join(outdir, "rna.npz"), rna.tocsr())
    return atac.tocsr(), rna.tocsr()


def precompute_lsi(peaks_npz: str, outdir: str, lsi_dim: int = 128):
    os.makedirs(outdir, exist_ok=True)
    peaks = sp.load_npz(peaks_npz).tocsr()
    # Fit on all data (or you can fit on a subset/train later)
    lsi = ATACLSITransformer(LSIParams(n_components=lsi_dim))
    lsi.fit(peaks)
    lsi.save(outdir)
    print(f"Saved LSI to {outdir}")


def main():
    ap = argparse.ArgumentParser(description="Data preparation utilities for multiome predictor")
    sub = ap.add_subparsers(dest="cmd")

    p1 = sub.add_parser("from-h5ad", help="Extract peaks and RNA from .h5ad")
    p1.add_argument("--h5ad", required=True)
    p1.add_argument("--atac-key", default="X_atac")
    p1.add_argument("--rna-key", default="X")
    p1.add_argument("--outdir", required=True)

    p2 = sub.add_parser("tfidf-lsi", help="Precompute TF-IDF+LSI and save models")
    p2.add_argument("--peaks", required=True, help="peaks CSR .npz")
    p2.add_argument("--outdir", required=True)
    p2.add_argument("--lsi-dim", type=int, default=128)

    args = ap.parse_args()
    if args.cmd == "from-h5ad":
        from_h5ad(args.h5ad, args.atac_key, args.rna_key, args.outdir)
    elif args.cmd == "tfidf-lsi":
        precompute_lsi(args.peaks, args.outdir, args.lsi_dim)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
