#!/usr/bin/env python3
"""
Evaluate a simple baseline: predict global gene-wise mean on masked positions.

Outputs masked MSE/MAE on val/test splits of a processed .h5ad.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp


def _to_dense(x):
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _masked_metrics(X, mean_vec: np.ndarray, mask_rate: float, seed: int, batch_size: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0
    for i in range(0, n, batch_size):
        xb = _to_dense(X[i : i + batch_size]).astype(np.float32, copy=False)
        mask = rng.random(xb.shape) < mask_rate
        if not mask.any():
            continue
        diff = (mean_vec - xb)[mask]
        total_mse += float((diff * diff).sum())
        total_mae += float(np.abs(diff).sum())
        total_count += int(diff.size)
    if total_count == 0:
        return float("nan"), float("nan")
    return total_mse / total_count, total_mae / total_count


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline: predict global gene mean on masked positions.")
    ap.add_argument("--data", required=True, help="Path to processed .h5ad")
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"data not found: {data_path}")

    adata = sc.read_h5ad(data_path)
    if "split" in adata.obs:
        ad_train = adata[adata.obs["split"].astype(str) == "train"]
        ad_val = adata[adata.obs["split"].astype(str) == "val"]
        ad_test = adata[adata.obs["split"].astype(str) == "test"]
    else:
        ad_train = adata
        ad_val = adata
        ad_test = adata

    # Compute global mean over train split
    X_tr = ad_train.X
    mean_vec = np.asarray(X_tr.mean(axis=0)).ravel().astype(np.float32)

    results = []
    for name, split in [("val", ad_val), ("test", ad_test)]:
        if split.n_obs == 0:
            results.append((name, float("nan"), float("nan")))
            continue
        mse, mae = _masked_metrics(split.X, mean_vec, args.mask_rate, args.seed, args.batch_size)
        results.append((name, mse, mae))

    print(f"data: {data_path}")
    print(f"mask_rate: {args.mask_rate}")
    for name, mse, mae in results:
        print(f"{name}: mse={mse:.6f} mae={mae:.6f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            f.write("split,mse,mae,mask_rate,data\n")
            for name, mse, mae in results:
                f.write(f"{name},{mse},{mae},{args.mask_rate},{data_path}\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
