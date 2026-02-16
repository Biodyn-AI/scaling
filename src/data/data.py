"""
Data ingestion and preprocessing for scRNA scaling-law experiments.

This module loads the HLCA **minified** reference via scvi-hub when possible.
If the hub snapshot lacks a usable expression matrix, it can fall back to
public Scanpy datasets (pbmc68k_reduced ≈ 68k cells, or pbmc3k ≈ 2.7k cells),
or load a local .h5ad that you provide.

Pipeline (high level)
---------------------
1) Load source AnnData
2) Ensure adata.X contains a non-empty expression matrix (counts or usable expr)
3) (Optional) stratified subsample / (optional) bootstrap to target N
4) HVG selection (Seurat v3 heuristic) on counts-like expression
5) Library-size normalize to target sum and log1p
6) (Optional) z-score per gene; ensure float32; (optional) densify
7) Train/val/test split and metadata write-out

CLI examples
------------
# Fast dev build: 20k cells, 1024 HVGs, fallback to pbmc68k if HLCA hub unusable
python -m src.data.data \
  --source hlca-minified \
  --fallback pbmc68k \
  --out ./data/hlca_minified.D20k.V1024.log1p.h5ad \
  --max-cells 20000 --hvg 1024 --val-frac 0.05 --test-frac 0.05

# Directly use the larger public fallback (≈68k cells)
python -m src.data.data \
  --source pbmc68k \
  --out ./data/pbmc68k.D60000.V1024.log1p.h5ad \
  --max-cells 60000 --hvg 1024

# Use a local .h5ad (counts/log expr) and build 120k cells after HVG selection
python -m src.data.data \
  --source local --local-h5ad /path/to/big_dataset.h5ad \
  --out ./data/local_big.D120k.V2000.log1p.h5ad \
  --max-cells 120000 --hvg 2000

Notes
-----
- HLCA minified is a SCANVI reference with ~585k cells × 2k HVGs in the full release.
  Some hub snapshots may not expose a counts layer; the loader here will fall back.
- pbmc68k_reduced is processed; HVG selection may warn about non-integers (harmless).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

import scvi  # noqa: F401 (ensures scvi is available)
from scvi.hub import HubModel

__all__ = [
    "LoadConfig",
    "PreprocessConfig",
    "SplitConfig",
    "build_and_save",
    "load_source",
]


# ===============================
# Configuration dataclasses
# ===============================
@dataclass
class LoadConfig:
    source: str = "hlca-minified"   # 'hlca-minified', 'pbmc68k', 'pbmc3k', '10x-h5', '10x-mtx'
    revision: Optional[str] = None  # HF snapshot hash/tag (only for HLCA hub)
    max_cells: Optional[int] = None
    path: Optional[str] = None      # local file/dir for 10x sources
    local_h5ad: Optional[str] = None  # local .h5ad path when source="local"
    fallback: str = "pbmc68k"       # fallback dataset if HLCA hub lacks expression
    bootstrap_to_max: bool = False  # sample with replacement if max_cells > n_obs

@dataclass
class PreprocessConfig:
    hvg: Optional[int] = 2000        # None → keep as-is
    normalize_target: float = 1e4
    log1p: bool = True
    zscore: bool = False
    dense: bool = False              # store X as dense float32 in AnnData

@dataclass
class SplitConfig:
    val_frac: float = 0.05
    test_frac: float = 0.05
    seed: int = 42
    # Try these keys for stratification; we will fall back automatically if classes are too small
    stratify_keys: Optional[Tuple[str, ...]] = ("ann_level_1", "study")


# ===============================
# Loading helpers
# ===============================

def _choose_expression_X(adata: ad.AnnData):
    """
    Ensure adata.X holds a non-empty expression matrix.
    Preference order:
      layers['counts'] > layers['raw_counts'] > adata.raw.X > adata.X

    Returns
    -------
    adata, src_name : (AnnData, str)
        adata with .X set to the chosen matrix, and a human-readable source label.
    """
    candidates = []
    if hasattr(adata, "layers"):
        if "counts" in adata.layers:
            candidates.append(("layers['counts']", adata.layers["counts"]))
        if "raw_counts" in adata.layers:
            candidates.append(("layers['raw_counts']", adata.layers["raw_counts"]))
    if getattr(adata, "raw", None) is not None and getattr(adata.raw, "X", None) is not None:
        candidates.append(("raw.X", adata.raw.X))
    if getattr(adata, "X", None) is not None:
        candidates.append(("X", adata.X))

    for name, X in candidates:
        try:
            if X is None:
                continue
            lib = np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)
        except Exception:
            continue
        if np.isfinite(lib).all() and lib.max() > 0:
            if name != "X":
                adata.X = X.copy() if hasattr(X, "copy") else X
            print(f"[data] using expression from {name}")
            return adata, name

    layer_list = list(getattr(adata, "layers", {}).keys()) if hasattr(adata, "layers") else []
    raise RuntimeError(
        "No non-empty expression matrix found in this AnnData. "
        f"Available layers: {layer_list}; has_raw={getattr(adata, 'raw', None) is not None}"
    )


def _load_fallback_dataset(name: str) -> ad.AnnData:
    """
    Return a public dataset by name.
    'pbmc68k' ≈ 68k cells (processed), 'pbmc3k' ≈ 2.7k cells (counts).
    """
    import scanpy as _sc
    name = name.lower()
    if name == "pbmc68k":
        adata = _sc.datasets.pbmc68k_reduced()  # ~68k cells, ~700 genes
        print("[data] fallback: loaded scanpy.datasets.pbmc68k_reduced()")
        return adata
    elif name == "pbmc3k":
        adata = _sc.datasets.pbmc3k()           # ~2.7k cells, ~1.8k genes
        print("[data] fallback: loaded scanpy.datasets.pbmc3k()")
        return adata
    else:
        raise ValueError(f"Unknown fallback dataset: {name}")


def load_source(cfg: LoadConfig) -> ad.AnnData:
    """
    Load a dataset according to `cfg.source`.

    Supported:
      - 'hlca-minified' : scvi-hub HLCA (falls back to PBMC if counts unusable)
      - 'pbmc68k'       : pbmc68k_reduced if present, else pbmc68k, else pbmc3k
      - 'pbmc3k'
      - 'local'         : local .h5ad via cfg.local_h5ad
      - '10x-h5'        : local 10x HDF5 via cfg.path
      - '10x-mtx'       : local 10x Matrix Market dir via cfg.path

    If `cfg.max_cells` < n_obs, a (stratified-if-possible) subsample is applied.
    """
    src = (cfg.source or "hlca-minified").lower()
    fallback = (cfg.fallback or "pbmc68k").lower()

    def _get_pbmc68k_like():
        import scanpy as _sc
        if hasattr(_sc.datasets, "pbmc68k_reduced"):
            print("[data] direct source: scanpy.datasets.pbmc68k_reduced()")
            return _sc.datasets.pbmc68k_reduced()
        if hasattr(_sc.datasets, "pbmc68k"):
            print("[data] direct source: scanpy.datasets.pbmc68k()")
            return _sc.datasets.pbmc68k()
        print("[data] pbmc68k* not available; falling back to scanpy.datasets.pbmc3k()")
        return _sc.datasets.pbmc3k()

    if src in {"10x-h5", "10x_h5", "tenx-h5"}:
        if not cfg.path:
            raise ValueError("For source '10x-h5', set LoadConfig.path to the .h5 file.")
        print(f"[data] local 10x HDF5: {cfg.path}")
        adata = sc.read_10x_h5(cfg.path)
        adata.var_names_make_unique()
        adata, _ = _choose_expression_X(adata)

    elif src in {"10x-mtx", "10x_mtx", "tenx-mtx"}:
        if not cfg.path:
            raise ValueError("For source '10x-mtx', set LoadConfig.path to the directory with matrix.mtx.")
        print(f"[data] local 10x MTX dir: {cfg.path}")
        # var_names='gene_symbols' often gives human-readable names; adjust if needed
        adata = sc.read_10x_mtx(cfg.path, var_names="gene_symbols", cache=True)
        adata.var_names_make_unique()
        adata, _ = _choose_expression_X(adata)

    elif src in {"pbmc68k", "pbmc_68k", "pbmc-68k"}:
        adata = _get_pbmc68k_like()
        adata, _ = _choose_expression_X(adata)

    elif src in {"pbmc3k", "pbmc_3k", "pbmc-3k"}:
        print("[data] direct source: scanpy.datasets.pbmc3k()")
        adata = sc.datasets.pbmc3k()
        adata, _ = _choose_expression_X(adata)

    elif src in {"local"}:
        if not cfg.local_h5ad:
            raise ValueError("For source 'local', set --local-h5ad to a .h5ad file.")
        print(f"[data] local .h5ad: {cfg.local_h5ad}")
        adata = sc.read_h5ad(cfg.local_h5ad)
        adata, _ = _choose_expression_X(adata)

    elif src in {"hlca", "hlca-minified", "human-lung-cell-atlas", "hlca_minified"}:
        try:
            hub = HubModel.pull_from_huggingface_hub(
                "scvi-tools/human-lung-cell-atlas-scanvi",
                revision=cfg.revision,
            )
            adata = hub.adata
            adata, _ = _choose_expression_X(adata)
            print("[data] loaded HLCA minified from Hub (expression OK)")
        except Exception as e:
            print(f"[data] HLCA hub load unusable: {e}")
            print(f"[data] fallback: using {fallback} via Scanpy")
            adata = _load_fallback_dataset(fallback) if fallback in {"pbmc68k", "pbmc3k"} else _get_pbmc68k_like()
            adata, _ = _choose_expression_X(adata)

    else:
        raise ValueError(f"Unknown source {cfg.source!r}.")

    # Optional subsample
    if cfg.max_cells is not None:
        target = int(cfg.max_cells)
        if target < adata.n_obs:
            adata = stratified_subsample(
                adata,
                n_cells=target,
                keys=("ann_level_1", "study"),  # ignored if missing
                seed=42,
            )
            print(f"[data] subsampled to {adata.n_obs} cells")
        elif cfg.bootstrap_to_max and target > adata.n_obs:
            rng = np.random.default_rng(42)
            idx = rng.choice(adata.n_obs, size=target, replace=True)
            adata = adata[idx].copy()
            print(f"[data] bootstrapped to {adata.n_obs} cells")

    return adata

# ===============================
# Preprocessing helpers
# ===============================

def ensure_categorical(adata: ad.AnnData, columns: Iterable[str]) -> None:
    for c in columns:
        if c in adata.obs and adata.obs[c].dtype.name != "category":
            adata.obs[c] = adata.obs[c].astype("category")


def select_hvgs(adata: ad.AnnData, n_top_genes: Optional[int]) -> ad.AnnData:
    if n_top_genes is None or n_top_genes >= adata.n_vars:
        return adata
    tmp = adata.copy()
    # HVGs typically computed on normalized/log1p data
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, flavor="seurat_v3")
    tmp = tmp[:, tmp.var["highly_variable"]].copy()
    return tmp


def normalize_and_log1p(adata: ad.AnnData, target_sum: float = 1e4, do_log1p: bool = True) -> None:
    sc.pp.normalize_total(adata, target_sum=target_sum)
    if do_log1p:
        sc.pp.log1p(adata)


def zscore_inplace(adata: ad.AnnData) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score per gene in-place on adata.X (dense or CSR/CSC sparse).

    Returns
    -------
    mean, std : arrays of shape (n_genes,)
    """
    X = adata.X
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        X2 = X.copy(); X2.data **= 2
        ex2 = np.asarray(X2.mean(axis=0)).ravel()
        var = np.maximum(ex2 - mean ** 2, 1e-12)
        std = np.sqrt(var)
        Xc = X.tocsc(copy=True)
        for j in range(Xc.shape[1]):
            s, e = Xc.indptr[j], Xc.indptr[j + 1]
            if s < e:
                Xc.data[s:e] -= mean[j]
                Xc.data[s:e] /= (std[j] + 1e-6)
        adata.X = Xc.tocsr()
    else:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        X -= mean
        X /= std
        adata.X = X
    return mean.astype(np.float32), std.astype(np.float32)


def ensure_float32_x(adata: ad.AnnData) -> None:
    """Cast X to float32 (dense or sparse). Ensures consistent dtype for training."""
    X = adata.X
    if sp.issparse(X):
        if X.data.dtype != np.float32:
            adata.X = X.astype(np.float32)
    else:
        if X.dtype != np.float32:
            adata.X = X.astype(np.float32)


# ===============================
# Robust stratification utilities
# ===============================

def _make_strat_labels(adata: ad.AnnData, cols: Sequence[str]):
    cols = [c for c in cols if c in adata.obs.columns]
    if not cols:
        return None
    if len(cols) == 1:
        return adata.obs[cols[0]].astype(str)
    return adata.obs[cols].astype(str).agg("|".join, axis=1)


def _labels_ok_for_split(labels, split_frac: float) -> bool:
    """Check if every class can be represented in both subsets for split_frac."""
    vc = labels.value_counts()
    if vc.min() < 2:
        return False
    a = vc * split_frac
    b = vc * (1.0 - split_frac)
    return a.min() >= 1 and b.min() >= 1


def _pick_working_labels(adata: ad.AnnData, keys: Sequence[str] | None, split_frac: float):
    """Try (combo of keys) → (each key alone) → None, returning first viable labels."""
    if not keys:
        return None
    keys = [k for k in keys if k in adata.obs.columns]
    tried: list[list[str]] = []
    if len(keys) >= 2:
        tried.append(keys)
    for k in keys:
        tried.append([k])
    for cols in tried:
        lab = _make_strat_labels(adata, cols)
        if lab is not None and _labels_ok_for_split(lab, split_frac):
            return lab
    return None


# ===============================
# Subsampling and splitting
# ===============================

def stratified_subsample(adata: ad.AnnData, n_cells: int, keys: Tuple[str, ...], seed: int = 42) -> ad.AnnData:
    """Subsample n_cells, attempting stratification; fall back if groups are too small."""
    n = adata.n_obs
    if n_cells >= n:
        return adata
    test_size = 1.0 - (n_cells / n)  # fraction to discard
    labels = _pick_working_labels(adata, keys, split_frac=test_size)
    idx = np.arange(n)
    idx_keep, _ = train_test_split(
        idx, test_size=test_size, random_state=seed, shuffle=True, stratify=labels
    )
    return adata[np.sort(idx_keep)].copy()


def make_splits(adata: ad.AnnData, cfg: SplitConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = adata.n_obs
    idx = np.arange(n)
    holdout_frac = cfg.val_frac + cfg.test_frac

    labels_full = _pick_working_labels(adata, cfg.stratify_keys or tuple(), split_frac=holdout_frac)

    idx_train, idx_tmp = train_test_split(
        idx,
        test_size=holdout_frac,
        random_state=cfg.seed,
        shuffle=True,
        stratify=labels_full,
    )

    rel = cfg.test_frac / (cfg.val_frac + cfg.test_frac) if holdout_frac > 0 else 0.5
    if labels_full is not None:
        lab_tmp = labels_full.iloc[idx_tmp]
        if not _labels_ok_for_split(lab_tmp, split_frac=rel):
            lab_tmp = None
    else:
        lab_tmp = None

    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=rel,
        random_state=cfg.seed,
        shuffle=True,
        stratify=lab_tmp,
    )

    split = np.full(n, "train", dtype=object)
    split[idx_val] = "val"
    split[idx_test] = "test"
    adata.obs["split"] = pd.Categorical(split, categories=["train", "val", "test"], ordered=False)

    return idx_train, idx_val, idx_test


# ===============================
# Orchestration
# ===============================

def build_and_save(
    load_cfg: LoadConfig,
    prep_cfg: PreprocessConfig,
    split_cfg: SplitConfig,
    out_path: Path,
    compression: str = "lzf",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -------- Load source and guarantee a real expression matrix in X --------
    adata = load_source(load_cfg)                 # load
    adata, src_name = _choose_expression_X(adata) # idempotent safety

    # Drop zero-library cells (guard; should be none after _choose_expression_X)
    X = adata.X
    lib = np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)
    nz = lib > 0
    n_drop = int((~nz).sum())
    if n_drop:
        print(f"[data] dropping {n_drop} zero-count cells")
        adata = adata[nz].copy()
    if adata.n_obs == 0:
        raise RuntimeError(
            "All cells became zero-library after layer selection and filtering. "
            "This indicates the upstream loader returned empty counts."
        )

    # -------- HVGs on counts-like expression (Seurat v3), then normalize/log1p --------
    if prep_cfg.hvg is not None and prep_cfg.hvg < adata.n_vars:
        tmp = adata.copy()
        tmp, _ = _choose_expression_X(tmp)
        sc.pp.highly_variable_genes(tmp, n_top_genes=prep_cfg.hvg, flavor="seurat_v3")
        hv_mask = np.array(tmp.var["highly_variable"], dtype=bool)
        adata = adata[:, hv_mask].copy()

    normalize_and_log1p(adata, target_sum=prep_cfg.normalize_target, do_log1p=prep_cfg.log1p)

    mean, std = None, None
    if prep_cfg.zscore:
        mean, std = zscore_inplace(adata)

    if prep_cfg.dense and sp.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)

    ensure_float32_x(adata)

    # -------- Splits --------
    ensure_categorical(adata, ["ann_level_1", "study"])
    idx_tr, idx_va, idx_te = make_splits(adata, split_cfg)

    # -------- Metadata --------
    splits_meta = asdict(split_cfg)
    if isinstance(splits_meta.get("stratify_keys"), tuple):
        splits_meta["stratify_keys"] = list(splits_meta["stratify_keys"])
    meta = {
        "scfm_version": 1,
        "source": load_cfg.source,
        "revision": load_cfg.revision,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "preprocessing": asdict(prep_cfg),
        "splits": splits_meta,
        "expression_source": src_name,
    }
    if mean is not None and std is not None:
        adata.uns["scfm_norm_mean"] = mean
        adata.uns["scfm_norm_std"] = std
    adata.uns["scfm_meta"] = meta

    # -------- Write --------
    adata.write(out_path, compression=compression)
    return out_path


# ===============================
# CLI
# ===============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and save processed scRNA dataset")
    p.add_argument("--source", default="hlca-minified",
                   choices=["hlca-minified", "pbmc68k", "pbmc3k", "local"],
                   help="Data source identifier")
    p.add_argument("--revision", default=None, help="HF revision (snapshot hash/tag) for the hub model")
    p.add_argument("--max-cells", type=int, default=None, help="Subsample to this many cells (approx stratified)")

    p.add_argument("--fallback", default="pbmc68k", choices=["pbmc68k", "pbmc3k"],
                   help="Fallback dataset if HLCA hub lacks expression")
    p.add_argument("--local-h5ad", default=None,
                   help="If set with --source local, read this .h5ad instead of hub/datasets")
    p.add_argument("--bootstrap-to-max", action="store_true",
                   help="If requested max-cells > available, sample with replacement up to max-cells")

    p.add_argument("--hvg", type=int, default=2000, help="Number of HVGs to select (<= current n_vars)")
    p.add_argument("--no-log1p", action="store_true", help="Skip log1p; still normalizes to target sum")
    p.add_argument("--normalize-target", type=float, default=1e4, help="Library size normalization target sum")
    p.add_argument("--zscore", action="store_true", help="Z-score per gene after log1p")
    p.add_argument("--dense", action="store_true", help="Store X as dense float32 in the output file")

    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out", required=True, help="Output .h5ad path")
    p.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "zstd"], help="h5ad compression")
    return p.parse_args()


def main():
    args = parse_args()

    load_cfg = LoadConfig(
        source=args.source,
        revision=args.revision,
        max_cells=args.max_cells,
        local_h5ad=args.local_h5ad,
        fallback=args.fallback,
        bootstrap_to_max=args.bootstrap_to_max,
    )
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

    out = build_and_save(load_cfg, prep_cfg, split_cfg, Path(args.out), compression=args.compression)
    print(f"Wrote {out}")
    try:
        ad_view = sc.read_h5ad(out, backed='r')
        meta = json.dumps({
            "n_cells": int(ad_view.n_obs),
            "n_genes": int(ad_view.n_vars),
            "preprocessing": asdict(prep_cfg),
            "splits": asdict(split_cfg),
        }, indent=2)
        print(meta)
        try:
            # Close on older AnnData where .file exists
            ad_view.file.close()
        except Exception:
            pass
    except Exception as e:
        print(f"[warn] Could not summarize written file: {e}")


if __name__ == "__main__":
    main()
