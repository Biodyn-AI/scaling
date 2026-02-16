#!/usr/bin/env python3
"""
Simple checks for the data pipeline in src/data/data.py.

Run from project root, e.g.:

  python test.py \
    --out ./data/hlca_minified.D20k.V2000.log1p_z.h5ad \
    --max-cells 20000 --hvg 2000 --zscore

This script will:
  1) Build and save a processed .h5ad using src/data/data.py
  2) Reload it and run sanity assertions (splits, shapes, metadata, dtypes)
  3) Print a short summary (cells, genes, sparsity, split counts)

Tip: start with 5k–20k cells for a quick run.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import Counter

import numpy as np
import scanpy as sc
import scipy.sparse as sp

# Make local src/ importable without installing a package
ROOT = pathlib.Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.data import (
    LoadConfig,
    PreprocessConfig,
    SplitConfig,
    build_and_save,
)


def _human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def _approx_mem_X(X) -> float:
    """Return an approximate in-memory size of X in bytes."""
    if sp.issparse(X):
        return float(X.data.nbytes + X.indptr.nbytes + X.indices.nbytes)
    return float(X.nbytes)


def run_once(args: argparse.Namespace) -> None:
    # 1) Build & save
    load_cfg = LoadConfig(source=args.source, revision=args.revision, max_cells=args.max_cells)
    prep_cfg = PreprocessConfig(
        hvg=args.hvg,
        normalize_target=args.normalize_target,
        log1p=not args.no_log1p,
        zscore=args.zscore,
        dense=args.dense,
    )
    split_cfg = SplitConfig(val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/3] Building dataset…")
    written = build_and_save(load_cfg, prep_cfg, split_cfg, out_path, compression=args.compression)
    print(f"✓ wrote: {written}")

    # 2) Reload & assertions
    print("[2/3] Reloading and checking…")
    adata = sc.read_h5ad(written, backed=None)

    # Basic shape checks
    assert adata.n_obs > 0 and adata.n_vars > 0, "Empty AnnData!"
    if args.hvg is not None:
        assert adata.n_vars == args.hvg, f"Expected {args.hvg} genes, got {adata.n_vars}"

    # Splits
    assert "split" in adata.obs, "obs['split'] is missing"
    cats = list(adata.obs["split"].cat.categories)
    assert set(["train", "val", "test"]).issubset(set(cats)), f"Unexpected split categories: {cats}"
    split_counts = Counter(adata.obs["split"].to_numpy())
    assert all(c > 0 for c in split_counts.values()), f"Some split has zero rows: {split_counts}"

    # Metadata
    meta = adata.uns.get("scfm_meta", {})
    assert isinstance(meta, dict) and meta.get("source"), "Missing or incomplete uns['scfm_meta']"

    # Z-score stats
    if args.zscore:
        assert "scfm_norm_mean" in adata.uns and "scfm_norm_std" in adata.uns, "Missing z-score stats in uns"
        assert len(adata.uns["scfm_norm_mean"]) == adata.n_vars
        assert len(adata.uns["scfm_norm_std"]) == adata.n_vars

    # Dtypes and sparsity
    if args.dense:
        assert not sp.issparse(adata.X), "Expected dense X but found sparse"
        assert getattr(adata.X, "dtype", np.float32) == np.float32
    else:
        if sp.issparse(adata.X):
            assert adata.X.data.dtype == np.float32
        else:
            assert adata.X.dtype == np.float32

    # 3) Summary
    print("[3/3] Summary")
    is_sparse = sp.issparse(adata.X)
    approx = _human_bytes(_approx_mem_X(adata.X))
    print(
        json.dumps(
            {
                "cells": int(adata.n_obs),
                "genes": int(adata.n_vars),
                "sparse": bool(is_sparse),
                "approx_X_mem": approx,
                "splits": {k: int(v) for k, v in split_counts.items()},
                "source": meta.get("source"),
                "revision": meta.get("revision"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test the data builder and saved .h5ad")
    p.add_argument("--source", default="hlca-minified")
    p.add_argument("--revision", default=None)
    p.add_argument("--max-cells", type=int, default=20000)
    p.add_argument("--hvg", type=int, default=2000)
    p.add_argument("--no-log1p", action="store_true")
    p.add_argument("--normalize-target", type=float, default=1e4)
    p.add_argument("--zscore", action="store_true")
    p.add_argument("--dense", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "data/hlca_minified.D20k.V2000.log1p.h5ad"))
    p.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "zstd"])
    args = p.parse_args()

    run_once(args)
