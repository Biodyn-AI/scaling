#!/usr/bin/env python3
"""
Lightweight sanity checker for large .h5ad datasets.

- Prints shape, sparsity, dtypes
- Shows a small top-left slice and a random slice of X
- Lists basic obs/var columns, tissue counts (if present)
- Approximates library-size stats from a sample (fast)
- Optionally saves a tiny preview .h5ad for manual inspection

Run:
  python -m src.data.check_dataset --file data/census_human.D1000000.V1024.h5ad
  python src/data/check_dataset.py --file data/your_file.h5ad --rows 8 --cols 12 --random 5 --sample 50000 --save-preview
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import pandas as pd


def _fmt_rows(n: int) -> str:
    return f"{n:,}"


def _head_slice(X, rows: int, cols: int) -> np.ndarray:
    """Return a small dense top-left slice [rows, cols] without materializing X."""
    r = min(rows, X.shape[0])
    c = min(cols, X.shape[1])
    if sp.issparse(X):
        return X[:r, :c].toarray()
    return np.asarray(X[:r, :c])


def _random_slice(X, k_rows: int, k_cols: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rows_idx, cols_idx, dense values) for a tiny random slice."""
    rng = np.random.default_rng(seed)
    r_idx = rng.choice(X.shape[0], size=min(k_rows, X.shape[0]), replace=False)
    c_idx = rng.choice(X.shape[1], size=min(k_cols, X.shape[1]), replace=False)
    r_idx.sort(); c_idx.sort()
    if sp.issparse(X):
        vals = X[r_idx[:, None], c_idx].toarray()
    else:
        vals = np.asarray(X)[np.ix_(r_idx, c_idx)]
    return r_idx, c_idx, vals


def _sample_libsize_stats(X, sample_n: int, seed: int = 0) -> dict:
    """Approximate library-size stats from a sample of rows (fast for big X)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if sample_n <= 0:
        return {}
    idx = rng.choice(n, size=min(sample_n, n), replace=False)
    if sp.issparse(X):
        libs = np.asarray(X[idx].sum(axis=1)).ravel()
        nnz = int(X[idx].nnz)
        frac_nz = nnz / (idx.size * X.shape[1])
    else:
        Xi = np.asarray(X)[idx]
        libs = Xi.sum(axis=1)
        frac_nz = float((Xi != 0).sum()) / (Xi.size)
    return {
        "sample_rows": int(idx.size),
        "min": float(libs.min()) if libs.size else float("nan"),
        "p25": float(np.percentile(libs, 25)) if libs.size else float("nan"),
        "median": float(np.median(libs)) if libs.size else float("nan"),
        "p75": float(np.percentile(libs, 75)) if libs.size else float("nan"),
        "max": float(libs.max()) if libs.size else float("nan"),
        "mean": float(libs.mean()) if libs.size else float("nan"),
        "std": float(libs.std()) if libs.size else float("nan"),
        "approx_frac_nonzero": float(frac_nz),
    }


def main():
    ap = argparse.ArgumentParser(description="Sanity check for large .h5ad files")
    ap.add_argument("--file", required=True, help="Path to .h5ad file")
    ap.add_argument("--rows", type=int, default=6, help="Top-left rows to show")
    ap.add_argument("--cols", type=int, default=8, help="Top-left cols to show")
    ap.add_argument("--random", type=int, default=5, help="Random slice rows/cols (square)")
    ap.add_argument("--sample", type=int, default=50000, help="Rows to sample for libsize stats")
    ap.add_argument("--save-preview", action="store_true", help="Save a small preview .h5ad next to the file")
    ap.add_argument("--preview-rows", type=int, default=2000, help="Rows in preview")
    ap.add_argument("--preview-cols", type=int, default=512, help="Cols in preview")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"[error] file not found: {path}")

    print(f"[open] {path}")
    adata = sc.read_h5ad(path)

    X = adata.X
    n_obs, n_vars = adata.n_obs, adata.n_vars
    print(f"[shape] cells × genes = {n_obs:,} × {n_vars:,}")
    print(f"[x] type={type(X).__name__} | sparse={sp.issparse(X)} | dtype={getattr(X, 'dtype', 'n/a')}")

    # Basic metadata
    src = adata.uns.get("source") or adata.uns.get("scfm_meta") or {}
    if src:
        print(f"[meta] source keys: {sorted(src.keys())}")
        print(f"[meta] brief: { {k: src[k] for k in ('dataset','census_version','measurement_name','x_name') if k in src} }")

    # obs & var columns
    print(f"[obs] columns (first 10): {list(map(str, adata.obs.columns[:10]))}")
    print(f"[var] columns (first 10): {list(map(str, adata.var.columns[:10]))}")

    # Tissue counts
    for col in ("tissue", "tissue_general"):
        if col in adata.obs:
            vc = adata.obs[col].astype(str).value_counts().head(20)
            print(f"[obs] top {len(vc)} {col} values:\n{vc.to_string()}")
            break

    # Splits (if present)
    if "split" in adata.obs:
        scounts = adata.obs["split"].astype(str).value_counts()
        print(f"[split] counts:\n{scounts.to_string()}")

    # Gene name column guess
    gene_col = None
    for c in ("feature_name", "gene_symbol", "feature_id", "soma_feature_id"):
        if c in adata.var:
            gene_col = c; break
    if gene_col:
        sample_genes = list(map(str, adata.var[gene_col].astype(str).head(10).to_list()))
        print(f"[var] gene column='{gene_col}' | head(10) genes: {sample_genes}")

    # Top-left slice
    tl = _head_slice(X, args.rows, args.cols)
    tl_df = pd.DataFrame(tl)
    print(f"[slice/top-left] {tl_df.shape[0]}×{tl_df.shape[1]} values:\n{tl_df}")

    # Random small slice
    r = max(1, int(args.random))
    r_idx, c_idx, rs = _random_slice(X, r, r, seed=0)
    rs_df = pd.DataFrame(rs, index=[int(i) for i in r_idx], columns=[int(j) for j in c_idx])
    print(f"[slice/random] rows={list(map(int, r_idx))} cols={list(map(int, c_idx))}\n{rs_df}")

    # Approx lib-size stats on a sample (fast)
    stats = _sample_libsize_stats(X, args.sample, seed=1)
    if stats:
        print(f"[libsize≈] sample_rows={stats['sample_rows']:,} "
              f"min={stats['min']:.3f} p25={stats['p25']:.3f} "
              f"median={stats['median']:.3f} p75={stats['p75']:.3f} "
              f"max={stats['max']:.3f} mean={stats['mean']:.3f} std={stats['std']:.3f} "
              f"frac_nonzero≈{stats['approx_frac_nonzero']:.4f}")

    # Optional preview file
    if args.save_preview:
        pr = min(args.preview_rows, n_obs)
        pc = min(args.preview_cols, n_vars)
        rng = np.random.default_rng(123)
        ridx = np.arange(pr) if pr < n_obs else np.arange(n_obs)
        cidx = np.arange(pc)
        # keep ordering simple & reproducible: first rows/cols
        small = adata[ridx, cidx].copy()
        out_small = path.with_suffix("")  # drop .h5ad
        out_small = out_small.with_name(out_small.name + f".preview_{pr}x{pc}.h5ad")
        small.write_h5ad(out_small, compression="lzf")
        print(f"[preview] wrote {out_small}")

    print("[done] sanity check complete.")


if __name__ == "__main__":
    main()
