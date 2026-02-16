#!/usr/bin/env python3
"""
Evaluate trained runs on val/test splits using masked reconstruction metrics.

Reads ckpt_best.pt (or ckpt_final.pt) from each run directory, rebuilds the model,
and computes masked MSE/MAE on val/test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from model.utils import build_model, masked_mse, masked_mae


class MaskedExpressionDataset(Dataset):
    def __init__(self, X, mask_rate: float, seed: int = 0):
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        self.X = X
        self.n_cells, self.n_genes = X.shape
        self.mask_rate = float(mask_rate)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int):
        x = self.X[idx]
        m = self.rng.random(self.n_genes) < self.mask_rate
        x_in = x.copy()
        x_in[m] = 0.0
        return {
            "x_in": torch.from_numpy(x_in),
            "x_tgt": torch.from_numpy(x),
            "mask": torch.from_numpy(m.astype(np.bool_)),
        }


class CollateWithGeneIds:
    def __init__(self, vocab_size: int):
        self.gene_ids = torch.arange(vocab_size, dtype=torch.long)

    def __call__(self, batch):
        out = {k: torch.stack([b[k] for b in batch], dim=0) for k in ["x_in", "x_tgt", "mask"]}
        out["gene_ids"] = self.gene_ids
        return out


@torch.no_grad()
def evaluate(model, loader, device: str) -> Tuple[float, float]:
    model.eval()
    n = 0
    m_mse = 0.0
    m_mae = 0.0
    for batch in loader:
        x_in = batch["x_in"].to(device)
        mask = batch["mask"].to(device)
        tgt = batch["x_tgt"].to(device)
        gene_ids = batch["gene_ids"].to(device)
        pred, _ = model(gene_ids, x_in, mask)
        m_mse += masked_mse(pred, tgt, mask).item()
        m_mae += masked_mae(pred, tgt, mask).item()
        n += 1
    if n == 0:
        return float("nan"), float("nan")
    return m_mse / n, m_mae / n


def pick_device(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch.backends.mps as mps  # type: ignore
        if mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_meta(run_dir: Path) -> Dict:
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def find_ckpt(run_dir: Path, which: str) -> Optional[Path]:
    if which == "best":
        p = run_dir / "ckpt_best.pt"
        return p if p.exists() else None
    if which == "final":
        p = run_dir / "ckpt_final.pt"
        return p if p.exists() else None
    # auto
    p = run_dir / "ckpt_best.pt"
    if p.exists():
        return p
    p = run_dir / "ckpt_final.pt"
    return p if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate runs on val/test splits.")
    ap.add_argument("--runs", default="runs", help="Root runs directory (recurse)")
    ap.add_argument("--out", default="analysis/runs_eval_test.csv", help="Output CSV")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--ckpt", default="best", choices=["best", "final", "auto"])
    ap.add_argument("--mask-rate", type=float, default=None, help="Override mask rate")
    args = ap.parse_args()

    device = pick_device(args.device)
    runs_root = Path(args.runs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    data_cache: Dict[str, sc.AnnData] = {}

    for run_dir in runs_root.rglob("run_meta.json"):
        run_dir = run_dir.parent
        ckpt_path = find_ckpt(run_dir, args.ckpt)
        if not ckpt_path:
            continue

        meta = load_meta(run_dir)
        data_path = meta.get("data")
        if not data_path:
            continue

        if data_path not in data_cache:
            data_cache[data_path] = sc.read_h5ad(data_path)
        adata = data_cache[data_path]

        if "split" in adata.obs:
            ad_val = adata[adata.obs["split"].astype(str) == "val"]
            ad_test = adata[adata.obs["split"].astype(str) == "test"]
        else:
            ad_val = adata
            ad_test = adata

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        vocab_size = int(cfg.get("vocab_size", adata.n_vars))
        size = cfg.get("size")
        overrides = cfg.get("overrides") or {}
        mask_rate = float(args.mask_rate if args.mask_rate is not None else cfg.get("mask_rate", 0.15))

        model = build_model(vocab_size=vocab_size, size=size, overrides=overrides)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        collate = CollateWithGeneIds(vocab_size)
        val_loader = DataLoader(
            MaskedExpressionDataset(ad_val.X, mask_rate=mask_rate, seed=123),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=(device == "cuda"),
        )
        test_loader = DataLoader(
            MaskedExpressionDataset(ad_test.X, mask_rate=mask_rate, seed=456),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=(device == "cuda"),
        )

        val_mse, val_mae = evaluate(model, val_loader, device)
        test_mse, test_mae = evaluate(model, test_loader, device)

        rows.append({
            "run": str(run_dir),
            "ckpt": str(ckpt_path),
            "data": data_path,
            "size": size,
            "vocab_size": vocab_size,
            "mask_rate": mask_rate,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "test_mse": test_mse,
            "test_mae": test_mae,
        })

    # write CSV
    import csv
    with out_path.open("w", newline="") as f:
        fieldnames = ["run", "ckpt", "data", "size", "vocab_size", "mask_rate", "val_mse", "val_mae", "test_mse", "test_mae"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
