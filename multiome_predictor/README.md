# Multiome Predictor: Predict RNA from ATAC (PyTorch)

This module provides a small, runnable baseline that predicts gene expression from ATAC peaks for single-cell multiome data.

Key features:
- Data formats: SciPy CSR `.npz` for peaks (cells x peaks) and RNA (cells x genes). Optional `.npy` dense arrays.
- Preprocessing: ATAC TF-IDF + LSI (TruncatedSVD) to reduce dimensionality.
- Model: Configurable MLP mapping LSI to log1p RNA expression.
- Losses: MSE (default) and Poisson NLL with log link.
- Metrics: RMSE and Pearson correlation (per-gene and overall).
- Tools: Synthetic data generator to verify end-to-end.

## Expected input
- Peaks: `peaks_csr.npz` (CSR, shape [n_cells, n_peaks])
- RNA: `rna_csr_or_dense.npz` (CSR or dense, shape [n_cells, n_genes])
- Optional: `peaks.txt` (peak names), `genes.txt` (gene names). Each line per feature.

If you only have an `.h5ad`, see `data_prep.py --from-h5ad` to extract aligned matrices.

## Quickstart (synthetic data)
1) Install deps (CPU ok):

```
# optional venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r code_bkups/multiome_predictor/requirements.txt
```

2) Generate synthetic data:

```
python code_bkups/multiome_predictor/synthetic.py --outdir code_bkups/multiome_predictor/data
```

3) Train a small model on synthetic data:

```
python code_bkups/multiome_predictor/train.py \
  --peaks code_bkups/multiome_predictor/data/peaks.npz \
  --rna code_bkups/multiome_predictor/data/rna.npz \
  --lsi-dim 64 --epochs 10 --batch-size 256 --loss mse \
  --outdir code_bkups/multiome_predictor/outputs/synth
```

4) Evaluate:

```
python code_bkups/multiome_predictor/evaluate.py \
  --checkpoint code_bkups/multiome_predictor/outputs/synth/model.pt \
  --peaks code_bkups/multiome_predictor/data/peaks.npz \
  --rna code_bkups/multiome_predictor/data/rna.npz \
  --lsi-dim 64
```

## Real data notes
- Ensure peaks and RNA matrices share the same cell order and counts per cell.
- If you have 10x Multiome outputs, you can export both matrices to aligned `.npz` and use `data_prep.py --tfidf-lsi` to precompute features.
- For very large datasets, consider precomputing LSI and saving `lsi.npy` to speed up training.

## Assumptions
- We predict log1p-normalized RNA counts. If using Poisson NLL, inputs are log-mean outputs (exp applied internally during loss).
- Baseline approach (LSI + MLP). You can extend with peak-to-gene aggregation, graph models, or chromatin-aware architectures.

## Files
- `dataset.py`: Data loading, TF-IDF/LSI, split utilities, PyTorch Dataset.
- `model.py`: MLP model and loss selection.
- `metrics.py`: RMSE and Pearson correlation.
- `train.py`: Training loop with early stopping and checkpointing.
- `evaluate.py`: Evaluation script with metrics.
- `data_prep.py`: Helpers for `.h5ad` extraction and LSI precomputation.
- `synthetic.py`: Generate small synthetic sparse multiome dataset.

