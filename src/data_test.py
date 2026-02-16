#!/usr/bin/env python3
"""
Python-only smoke test for src.data.data (no shell flags).

It will:
  • Build a dataset with the requested cells/HVGs (using build_and_save)
  • Re-open the written .h5ad
  • Print a compact summary + basic sanity checks
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp

# Make sure we can import "src.data.data" when running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data import (  # noqa: E402
    LoadConfig, PreprocessConfig, SplitConfig, build_and_save
)

def main():
    ap = argparse.ArgumentParser(description="Build+check HLCA minified dataset")
    ap.add_argument("--out", default=str(ROOT / "data/hlca_minified.D20000.V1024.log1p.h5ad"))
    ap.add_argument("--cells", type=int, default=20_000)
    ap.add_argument("--hvg", type=int, default=1024)
    ap.add_argument("--normalize-target", type=float, default=1e4)
    ap.add_argument("--no-log1p", action="store_true")
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # -------- Build via Python API --------
    load_cfg = LoadConfig(source="hlca-minified", revision=None, max_cells=args.cells)
    prep_cfg = PreprocessConfig(
        hvg=args.hvg,
        normalize_target=args.normalize_target,
        log1p=not args.no_log1p,
        zscore=args.zscore,
        dense=args.dense,
    )
    split_cfg = SplitConfig(
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    print(f"[build] writing: {out} (cells={args.cells}, hvg={args.hvg})")
    written = build_and_save(load_cfg, prep_cfg, split_cfg, out, compression="lzf")
    print(f"[build] wrote: {written}")

    # -------- Open and sanity-check --------
    ad = sc.read_h5ad(written)  # loads into memory; OK for ~20k×2k
    X = ad.X
    n_cells, n_genes = ad.n_obs, ad.n_vars

    # dtype + storage
    dtype = X.dtype if not sp.issparse(X) else X.data.dtype
    sparse = sp.issparse(X)

    # row sums (after normalize_total) should be ~target_sum if log1p=False
    # If log1p=True (default), sums will differ; we still check for nonzero rows.
    if sparse:
        lib = np.asarray(X.sum(axis=1)).ravel()
    else:
        lib = X.sum(axis=1)

    n_zero_rows = int((lib <= 0).sum())
    split_counts = {}
    if "split" in ad.obs:
        split_counts = ad.obs["split"].value_counts().to_dict()

    meta = ad.uns.get("scfm_meta", {})
    print(json.dumps({
        "shape": [int(n_cells), int(n_genes)],
        "sparse": bool(sparse),
        "dtype": str(dtype),
        "zero_rows": n_zero_rows,
        "splits": split_counts,
        "meta_source": meta.get("source"),
        "meta_revision": meta.get("revision"),
    }, indent=2))

    # basic assertions that should hold for a healthy file
    assert n_cells > 0, "no cells present"
    assert n_genes > 0, "no genes present"
    assert dtype == np.float32, f"X dtype is not float32: {dtype}"
    assert n_zero_rows == 0, "There are zero-library rows after preprocessing (unexpected)"

    print("\n[data_test] OK.")

if __name__ == "__main__":
    main()
