#!/usr/bin/env python3
"""
Programmatic sanity check for data loading/preprocessing.
- Builds (if missing) a processed .h5ad
- Prints basic stats and writes a JSON summary
- Optionally draws a UMAP (subsampled) and a libsize histogram

Run:
  python -m src.data_sanity
or:
  python src/data_sanity.py
"""

from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Adjustable parameters (edit these defaults; no CLI is used)
# --------------------------------------------------------------------------------------

# Pick one:
#   "10x-h5"  + TENX_PATH to a local 10x HDF5 file
#   "10x-mtx" + TENX_PATH to a local 10x MTX directory (matrix.mtx[.gz], barcodes.tsv[.gz], features.tsv[.gz]/genes.tsv)
#   "pbmc3k"
#   "pbmc68k"
#   "hlca-minified"   (hub snapshot may lack counts; code will fall back automatically)
SOURCE = "10x-h5"
TENX_PATH = "/absolute/path/to/pbmc68k.h5"   # set when SOURCE is 10x-h5 or 10x-mtx

MAX_CELLS = 60000           # None or int; subsample if dataset is larger
HVG = 1024                  # number of HVGs to keep

MAKE_PLOTS  = True          # draw histogram + UMAP PNGs
MAKE_UMAP   = True          # set False to skip UMAP (faster)
UMAP_MAX_N  = 20000         # subsample for UMAP if too many cells

OUT_DIRNAME  = "data"       # written under project root
# --------------------------------------------------------------------------------------

# Make local packages importable regardless of CWD.
SRC_DIR   = Path(__file__).resolve().parent     # .../src
PROJ_ROOT = SRC_DIR.parent                      # repo root
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.data import (
    LoadConfig, PreprocessConfig, SplitConfig,
    build_and_save, _choose_expression_X, ensure_categorical
)

def _resolve_source_and_path(source: str, tenx_path: str | None) -> tuple[str, str | None]:
    """Validate local 10x paths and auto-fallback if missing."""
    if source in ("10x-h5", "10x-mtx"):
        if not tenx_path or not Path(tenx_path).exists():
            print(f"[sanity] TENX_PATH not found: {tenx_path}")
            print("[sanity] Falling back to SOURCE='pbmc68k' (built-in dataset).")
            return "pbmc68k", None
    return source, tenx_path

def _out_path_for(source: str, max_cells: int | None, hvg: int) -> Path:
    base = f"{source.replace('-', '_')}.D{max_cells if max_cells else 'all'}.V{hvg}.log1p.h5ad"
    return PROJ_ROOT / OUT_DIRNAME / base

def _ensure_dataset(source: str, tenx_path: str | None, out_path: Path) -> Path:
    """
    Build the dataset if missing using the current SOURCE/MAX_CELLS/HVG settings.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[build] found: {out_path}")
        return out_path

    print(f"[build] writing: {out_path} (cells={MAX_CELLS}, hvg={HVG})")
    load_cfg = LoadConfig(source=source, max_cells=MAX_CELLS, path=tenx_path)  # path is used for 10x sources
    prep_cfg = PreprocessConfig(hvg=HVG, normalize_target=1e4, log1p=True, zscore=False, dense=False)
    split_cfg = SplitConfig(val_frac=0.05, test_frac=0.05, seed=42)

    written = build_and_save(load_cfg, prep_cfg, split_cfg, out_path, compression="lzf")
    print(f"[build] wrote: {written}")
    return out_path

def _libsize_stats(X):
    if sp.issparse(X):
        lib = np.asarray(X.sum(axis=1)).ravel()
        nnz = int(X.nnz)
        sparse = True
    else:
        lib = X.sum(axis=1)
        nnz = int((X != 0).sum())
        sparse = False
    return {
        "sparse": sparse,
        "nnz": nnz,
        "frac_nonzero": float(nnz / (X.shape[0] * X.shape[1])),
        "min": float(lib.min()),
        "p25": float(np.percentile(lib, 25)),
        "median": float(np.median(lib)),
        "p75": float(np.percentile(lib, 75)),
        "max": float(lib.max()),
        "mean": float(lib.mean()),
        "std": float(lib.std()),
    }

def _plot_lib_hist(X, out_png: Path):
    if sp.issparse(X):
        lib = np.asarray(X.sum(axis=1)).ravel()
    else:
        lib = X.sum(axis=1)
    plt.figure()
    plt.hist(lib, bins=50)
    plt.xlabel("Library size (sum of counts per cell)")
    plt.ylabel("Cells")
    plt.title("Library size distribution")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def _plot_umap(adata, out_png: Path, max_n: int = 20000):
    # Subsample if needed for speed
    if adata.n_obs > max_n:
        idx = np.random.default_rng(0).choice(adata.n_obs, size=max_n, replace=False)
        ad = adata[idx].copy()
    else:
        ad = adata.copy()
    # PCA/Neighbors/UMAP on already normalized/log1p HVGs
    # Avoid densifying large sparse matrices
    if not sp.issparse(ad.X):
        sc.pp.scale(ad, max_value=10)  # harmless for visualization on dense matrices
    sc.tl.pca(ad, svd_solver="arpack")
    sc.pp.neighbors(ad, n_neighbors=15, n_pcs=min(50, ad.var.shape[0]))
    sc.tl.umap(ad)
    sc.pl.umap(ad, color=None, show=False)
    plt.title("UMAP (unsupervised)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    # Resolve effective source/path with fallback if needed
    eff_source, eff_path = _resolve_source_and_path(SOURCE, TENX_PATH)
    out_path = _out_path_for(eff_source, MAX_CELLS, HVG)
    out_json = out_path.with_suffix(".sanity.json")
    out_hist = out_path.with_suffix(".libhist.png")
    out_umap = out_path.with_suffix(".umap.png")

    _ensure_dataset(eff_source, eff_path, out_path)

    # Read and summarize
    adata = sc.read_h5ad(out_path)
    adata, _ = _choose_expression_X(adata)

    X = adata.X
    n_obs, n_vars = adata.n_obs, adata.n_vars
    zero_rows = int(((np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)) == 0).sum())
    lib_stats = _libsize_stats(X)

    # Splits summary (if present)
    split_counts = {}
    if "split" in adata.obs:
        ensure_categorical(adata, ["split"])
        split_counts = {str(k): int(v) for k, v in adata.obs["split"].value_counts().to_dict().items()}

    summary = {
        "file": str(out_path),
        "matrix": {
            "shape": [int(n_obs), int(n_vars)],
            "sparse": bool(sp.issparse(X)),
            "dtype": str(X.dtype),
            "nnz": lib_stats["nnz"] if sp.issparse(X) else int((X != 0).sum()),
            "frac_nonzero": lib_stats["frac_nonzero"],
            "zero_rows": zero_rows,
        },
        "splits": split_counts or None,
        "libsize": {k: v for k, v in lib_stats.items() if k not in ("sparse", "nnz", "frac_nonzero")},
        "meta_source": adata.uns.get("scfm_meta", {}).get("source"),
        "meta_revision": adata.uns.get("scfm_meta", {}).get("revision"),
        "n_obs": int(n_obs),
        "n_vars": int(n_vars),
    }

    print(json.dumps(summary, indent=2))
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[sanity] wrote {out_json}")

    if MAKE_PLOTS:
        _plot_lib_hist(X, out_hist)
        print(f"[sanity] wrote {out_hist}")
        if MAKE_UMAP:
            _plot_umap(adata, out_umap, max_n=UMAP_MAX_N)
            print(f"[sanity] wrote {out_umap}")

if __name__ == "__main__":
    main()
