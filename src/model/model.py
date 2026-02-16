"""
Permutation-invariant transformer models for single-cell expression.

This module provides a small family of transformer encoders that operate on a
set of gene tokens. Each token is formed by (a) a learned gene ID embedding and
(b) a projection of the (possibly masked) expression value. There are no
positional encodings; the model is permutation-invariant over genes.

The main entry point is `ScTransformer`, which supports:
- configurable size: d_model, n_layers, n_heads, ff_mult, dropout
- learned mask token for values
- pooling: mean or CLS token
- flexible prediction head: linear or small MLP

Inputs
------
- gene_ids: LongTensor [T] or [B, T] with values in [0, vocab_size)
- x_in:    FloatTensor [B, T]   (masked positions may be any placeholder)
- mask:    BoolTensor  [B, T]   (True where the value was masked)

Outputs
-------
- y_hat: FloatTensor [B, T]  — per-gene regression (reconstruction)
- z:     FloatTensor [B, d]  — pooled cell embedding (mean or CLS)

Example
-------
>>> import torch
>>> model = ScTransformer(vocab_size=2000, d_model=128, n_layers=2, n_heads=4)
>>> B, T = 8, 2000
>>> gene_ids = torch.arange(T)                 # [T]
>>> x_in = torch.randn(B, T)
>>> mask = torch.rand(B, T) < 0.15
>>> y_hat, z = model(gene_ids, x_in, mask)
>>> y_hat.shape, z.shape
(torch.Size([8, 2000]), torch.Size([8, 128]))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ModelConfig",
    "GeneValueEmbed",
    "ScTransformer",
    "count_parameters",
]


# -------------------------------
# Config
# -------------------------------
@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    norm_first: bool = True           # pre-LN if True
    pool: Literal["mean", "cls"] = "mean"
    head_type: Literal["linear", "mlp"] = "linear"
    head_hidden_mult: int = 2         # only used for mlp head
    use_bias: bool = True
    learned_mask_token: bool = True


# -------------------------------
# Embedding: gene + value
# -------------------------------
class GeneValueEmbed(nn.Module):
    """Embed gene IDs and project values to the model space.

    value -> Linear(1, d_model)
    masked positions -> learned mask token (if enabled) or zeros.
    """

    def __init__(self, vocab_size: int, d_model: int, learned_mask_token: bool = True):
        super().__init__()
        self.gene_emb = nn.Embedding(vocab_size, d_model)
        self.val_proj = nn.Linear(1, d_model)
        if learned_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.register_buffer("mask_token", torch.zeros(1, 1, d_model))

    def forward(
        self, gene_ids: torch.LongTensor, values: torch.Tensor, masked: torch.BoolTensor
    ) -> torch.Tensor:
        """Return token embeddings of shape [B, T, d].

        Parameters
        ----------
        gene_ids : LongTensor [T] or [B, T]
        values   : FloatTensor [B, T]
        masked   : BoolTensor  [B, T]
        """
        if gene_ids.dim() == 1:
            # Broadcast gene IDs across batch
            g = self.gene_emb(gene_ids)            # [T, d]
            g = g.unsqueeze(0).expand(values.size(0), -1, -1)  # [B, T, d]
        else:
            g = self.gene_emb(gene_ids)            # [B, T, d]
        v = self.val_proj(values.unsqueeze(-1))     # [B, T, d]
        # Inject mask token at masked positions
        mtoken = self.mask_token.expand(values.size(0), values.size(1), -1)  # [B,T,d]
        v = torch.where(masked.unsqueeze(-1), mtoken, v)
        return g + v


# -------------------------------
# Transformer encoder
# -------------------------------
class ScTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4, n_heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.1, activation: str = "gelu",
                 norm_first: bool = True, pool: str = "mean", head_type: str = "linear",
                 head_hidden_mult: int = 2, use_bias: bool = True, learned_mask_token: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.config = ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            activation=activation,  # type: ignore
            norm_first=norm_first,
            pool=pool,              # type: ignore
            head_type=head_type,    # type: ignore
            head_hidden_mult=head_hidden_mult,
            use_bias=use_bias,
            learned_mask_token=learned_mask_token,
        )

        self.embed = GeneValueEmbed(vocab_size, d_model, learned_mask_token=learned_mask_token)

        act = {
            "gelu": F.gelu,
            "relu": F.relu,
            "silu": F.silu,
        }[activation]

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Optional CLS token for pooling
        self.use_cls = (pool == "cls")
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Prediction head: per-token regression
        if head_type == "linear":
            self.head = nn.Linear(d_model, 1, bias=use_bias)
        elif head_type == "mlp":
            hidden = d_model * head_hidden_mult
            self.head = nn.Sequential(
                nn.Linear(d_model, hidden, bias=use_bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1, bias=use_bias),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize to stable defaults
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if hasattr(self, "cls_token"):
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        if hasattr(self.embed, "mask_token") and isinstance(self.embed.mask_token, torch.nn.Parameter):
            nn.init.normal_(self.embed.mask_token, mean=0.0, std=0.02)

    @torch.no_grad()
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, gene_ids: torch.LongTensor, x_in: torch.Tensor, mask: torch.BoolTensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        gene_ids : LongTensor [T] or [B, T]
        x_in     : FloatTensor [B, T]
        mask     : BoolTensor  [B, T]

        Returns
        -------
        y_hat : FloatTensor [B, T]
        z     : FloatTensor [B, d_model]
        """
        B, T = x_in.shape
        # Build token embeddings
        H = self.embed(gene_ids, x_in, mask)  # [B, T, d]

        # Optionally prepend CLS for pooling
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # [B,1,d]
            H = torch.cat([cls, H], dim=1)          # [B, 1+T, d]

        # Encode (no attention mask needed; permutation-invariant)
        Z = self.encoder(H)  # [B, T or 1+T, d]

        # Per-token predictions (exclude CLS if present)
        if self.use_cls:
            token_repr = Z[:, 1:, :]  # [B, T, d]
            pooled = Z[:, 0, :]       # [B, d]
        else:
            token_repr = Z
            pooled = Z.mean(dim=1)

        y_hat = self.head(token_repr).squeeze(-1)  # [B, T]
        return y_hat, pooled


# -------------------------------
# Utility
# -------------------------------
@torch.no_grad()
def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
