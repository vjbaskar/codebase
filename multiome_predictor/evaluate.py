from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import build_datasets, ATACLSITransformer
from .model import MLP, MLPConfig
from .metrics import rmse, overall_pearson, pearsonr_per_gene


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model on RNA prediction from ATAC")
    p.add_argument("--checkpoint", required=True, help="Path to model.pt from training")
    p.add_argument("--peaks", required=True)
    p.add_argument("--rna", required=True)
    p.add_argument("--lsi-dim", type=int, default=None, help="Override if not in meta")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--lsi-dir", type=str, default=None, help="Directory with fitted LSI from training")
    return p.parse_args()


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def main():
    args = parse_args()
    device = pick_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg_dict = ckpt["config"]
    meta = ckpt["meta"]

    # Build datasets with pretrained LSI for consistent transform
    lsi_dir = args.lsi_dir
    if lsi_dir is None:
        # Try infer from checkpoint folder
        candidate = os.path.join(os.path.dirname(args.checkpoint), "lsi")
        if os.path.isdir(candidate):
            lsi_dir = candidate

    train_ds, val_ds, test_ds, _ = build_datasets(
        peaks_path=args.peaks,
        rna_path=args.rna,
        lsi_dim=meta.get("lsi_dim", args.lsi_dim or cfg_dict.get("input_dim", 128)),
        val=0.1,
        test=0.1,
        n_genes=None if meta.get("gene_idx") is None else len(meta["gene_idx"]),
        seed=42,
        lsi_dir=None,
        lsi_pretrained_dir=lsi_dir,
    )

    cfg = MLPConfig(
        input_dim=cfg_dict["input_dim"],
        output_dim=cfg_dict["output_dim"],
        hidden_dims=cfg_dict["hidden_dims"],
        dropout=cfg_dict.get("dropout", 0.1),
        loss=cfg_dict["loss"],
    )
    model = MLP(cfg)
    model.load_state_dict(ckpt["model_state"]) 
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu()
            ys.append(yb)
            ps.append(pred)
    y_true = torch.cat(ys, dim=0).numpy()
    y_pred = torch.cat(ps, dim=0).numpy()

    r = overall_pearson(y_true, y_pred)
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    print(json.dumps({"overall_pearson": r, "rmse": rmse_val}, indent=2))


if __name__ == "__main__":
    main()
