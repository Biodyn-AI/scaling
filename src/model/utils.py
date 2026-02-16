"""
Model package utilities for single‑cell transformers (MVP).

This file complements `src/model/model.py` by providing:
  • Size presets/registry and a factory `build_model(...)` for easy scaling sweeps
  • Masked reconstruction losses (MSE/MAE/Huber)
  • Parameter counting utility

Typical usage
-------------
>>> from model import build_model, masked_mse
>>> net = build_model(vocab_size=2000, size="XS")
>>> loss = masked_mse(pred, target, mask)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Literal, Mapping, Optional

import torch
import torch.nn as nn

from .model import ScTransformer, count_parameters as _count_parameters


# ===============================
# Size presets & factory
# ===============================
@dataclass(frozen=True)
class SizePreset:
    d_model: int
    n_layers: int
    n_heads: int
    ff_mult: int = 4
    dropout: float = 0.1


# Roughly aligned with the plan (you can tweak as you go)
PRESETS: Dict[str, SizePreset] = {
    "XXS": SizePreset(d_model=1, n_layers=1, n_heads=1, ff_mult=1, dropout=0.1),
    "TINY": SizePreset(d_model=16, n_layers=1, n_heads=1, ff_mult=1, dropout=0.1),
    "XS": SizePreset(d_model=64, n_layers=2, n_heads=4, ff_mult=4, dropout=0.1),
    "S": SizePreset(d_model=128, n_layers=4, n_heads=8, ff_mult=4, dropout=0.1),
    "M": SizePreset(d_model=512, n_layers=6, n_heads=8, ff_mult=4, dropout=0.1),
    "L": SizePreset(d_model=1020, n_layers=8, n_heads=12, ff_mult=4, dropout=0.1),
    "XL": SizePreset(d_model=1536, n_layers=12, n_heads=16, ff_mult=4, dropout=0.1),
}


def build_model(
        *,
        vocab_size: int,
        size: Literal["XXS", "TINY", "XS", "S", "M", "L", "XL"] | None = None,
        overrides: Optional[Mapping[str, object]] = None,
        **kwargs,
) -> ScTransformer:
    """Instantiate a `ScTransformer` from a named size preset + overrides.

    Parameters
    ----------
    vocab_size : int
        Number of genes (tokens)
    size : {"XS","S","M","L"} or None
        If provided, start from this preset; if None, you must pass explicit kwargs.
    overrides : Mapping[str, object], optional
        Dict of fields to override (e.g., {"dropout": 0.0})
    **kwargs :
        Any `ScTransformer` constructor arg (e.g., d_model=256,...)
    """
    cfg = {}
    if size is not None:
        if size not in PRESETS:
            raise ValueError(f"Unknown size preset: {size}")
        cfg.update(asdict(PRESETS[size]))
    cfg.update(kwargs)
    if overrides:
        cfg.update(dict(overrides))

    # Ensure mandatory fields present
    required = ["d_model", "n_layers", "n_heads", "ff_mult", "dropout"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing config fields: {missing}")

    return ScTransformer(vocab_size=vocab_size, **cfg)  # type: ignore[arg-type]


# ===============================
# Losses (masked reconstruction)
# ===============================

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error on masked positions only.

    pred/target: [B, T] ; mask: Bool/0-1 [B, T]
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, target {target.shape}, mask {mask.shape}")
    m = mask if mask.dtype == torch.bool else (mask != 0)
    diff = (pred - target)[m]
    return (diff * diff).mean()


def _masked_residuals(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, target {target.shape}, mask {mask.shape}")
    m = mask if mask.dtype == torch.bool else (mask != 0)
    return (pred - target)[m]


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask if mask.dtype == torch.bool else (mask != 0)
    return (pred.sub(target).abs()[m]).mean()


def masked_gaussian_nll(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        eps: float = 1e-8,
) -> torch.Tensor:
    """Gaussian negative log-likelihood on masked positions (homoscedastic).

    We use the maximum-likelihood variance estimate on each batch:
      sigma^2 = mean((pred - target)^2)
    and report per-element NLL in nats.
    """
    r = _masked_residuals(pred, target, mask)
    var = (r * r).mean().clamp_min(float(eps))
    return 0.5 * (torch.log(2.0 * torch.pi * var) + 1.0)


def masked_huber(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss (a.k.a. smooth L1) on masked positions.

    For residual r, loss = 0.5*r^2 if |r|<=delta else delta*(|r|-0.5*delta).
    """
    m = mask if mask.dtype == torch.bool else (mask != 0)
    r = (pred - target)[m]
    abs_r = r.abs()
    quad = (0.5 * r * r)
    lin = delta * (abs_r - 0.5 * delta)
    return torch.where(abs_r <= delta, quad, lin).mean()


# ===============================
# Utils
# ===============================
@torch.no_grad()
def count_parameters(module: nn.Module) -> int:
    return _count_parameters(module)
