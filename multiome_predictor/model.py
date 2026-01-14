from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    dropout: float = 0.1
    loss: str = "mse"  # or "poisson"


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        layers = []
        prev = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = h
        layers.append(nn.Linear(prev, cfg.output_dim))
        self.net = nn.Sequential(*layers)
        self.loss_name = cfg.loss
        if self.loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_name == "poisson":
            # We'll output log-rate; disable full, use mean
            self.criterion = nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(pred, target)
