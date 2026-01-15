from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import build_datasets
from .model import MLP, MLPConfig
from .metrics import rmse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MLP to predict RNA from ATAC (LSI features)")
    p.add_argument("--peaks", required=True, help="Path to peaks CSR .npz")
    p.add_argument("--rna", required=True, help="Path to RNA .npz/.npy (cells x genes)")
    p.add_argument("--lsi-dim", type=int, default=128)
    p.add_argument("--n-genes", type=int, default=None, help="Select top N variable genes (None=all)")
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--hidden-dims", type=str, default="512,256")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--loss", type=str, choices=["mse", "poisson"], default="mse")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-lsi", action="store_true", help="Save fitted TF-IDF+SVD to outdir/lsi")
    p.add_argument("--load-lsi", type=str, default=None, help="Directory with pretrained LSI (tfidf/svd/scaler)")
    p.add_argument("--device", type=str, default="auto", help="cuda, mps, or cpu (auto picks available)")
    return p.parse_args()


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    lsi_dir = os.path.join(args.outdir, "lsi") if args.save_lsi else None
    train_ds, val_ds, test_ds, meta = build_datasets(
        peaks_path=args.peaks,
        rna_path=args.rna,
        lsi_dim=args.lsi_dim,
        val=args.val,
        test=args.test,
        n_genes=args.n_genes,
        seed=args.seed,
        lsi_dir=lsi_dir,
        lsi_pretrained_dir=args.load_lsi,
    )

    input_dim = train_ds.X.shape[1]
    output_dim = train_ds.Y.shape[1]
    hidden_dims: List[int] = [int(x) for x in args.hidden_dims.split(',') if x]

    cfg = MLPConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        loss=args.loss,
    )
    model = MLP(cfg)

    device = pick_device(args.device)
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2, factor=0.5)

    best_val = float("inf")
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = model.loss(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = model.loss(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            epochs_no_improve = 0
            # save checkpoint
            ckpt = {
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "meta": meta,
            }
            torch.save(ckpt, os.path.join(args.outdir, "model.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    # Save training meta
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump({
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "loss": args.loss,
            "best_val": best_val,
            "n_cells": meta["n_cells"],
            "n_genes_total": meta["n_genes_total"],
            "gene_idx": meta["gene_idx"],
            "lsi_dim": meta["lsi_dim"],
        }, f, indent=2)

    # Quick final RMSE on val set with best checkpoint
    ckpt_path = os.path.join(args.outdir, "model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        ys = []
        ps = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).cpu()
                ys.append(yb)
                ps.append(pred)
        y_true = torch.cat(ys, dim=0)
        y_pred = torch.cat(ps, dim=0)
        print(f"Val RMSE: {rmse(y_true, y_pred):.4f}")


if __name__ == "__main__":
    main()
