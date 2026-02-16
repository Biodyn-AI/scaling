#!/usr/bin/env python3
from __future__ import annotations

import os, math, json
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from data.data import (
    ensure_categorical,
    make_splits,
    ensure_float32_x,
    normalize_and_log1p,
    SplitConfig,   # <-- add this
)

# Project root
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# Pipeline API
from pipeline.api import ensure_dataset, train_once, sweep, analyze_runs  # noqa: E402

# Reuse split/normalization helpers from your data module
from data.data import ensure_categorical, make_splits, ensure_float32_x, normalize_and_log1p  # noqa: E402


# ---------- CONFIG ----------
# Your 1M-cell file:
BIG_H5AD = ROOT / "data" / "census_human.auto.D1000000.V1024.20250819_100156.h5ad"

# How many cells you want to train on (random, tissue-aware):
N = 30000

# Training knobs
micro_batch = 4                      # actual per-step batch size
target_batch = 128                    # desired effective batch (via accumulation)
steps = 2 * math.ceil(N / micro_batch)
val_every = 10
log_every = 10
train_metrics_every = 10
num_workers = 0
amp = False
seeds = [1, 2]
sizes = ["XS", "S", "M", "L"]

# ---------- SUBSET HELPER ----------
def make_subset_from_big(
    big_h5ad: str | Path,
    out_path: str | Path,
    *,
    n_cells: int,
    seed: int = 42,
    strat_key: str = "tissue",       # stratify sampling by this obs column if present
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    normalize_target: float = 1e4,
    log1p: bool = True,
    dense: bool = False,             # store X dense in the output (default keeps sparse)
    overwrite: bool = False,
) -> Path:
    """
    Create a random (stratified) subset from a large .h5ad *without* loading the whole matrix.

    Steps:
      1) open big file in backed='r' mode
      2) pick `n_cells` indices (stratified by `strat_key` if present)
      3) slice to memory only those rows
      4) normalize/log1p, cast float32
      5) create train/val/test splits (stratified where possible)
      6) write the subset .h5ad and return its path
    """
    from pathlib import Path
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp

    from data.data import (
        ensure_categorical,
        ensure_float32_x,
        normalize_and_log1p,
        make_splits,
        SplitConfig,
    )

    big_h5ad = Path(big_h5ad)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f"[subset] found: {out_path}")
        return out_path

    # --- 1) Open big file (backed) and read obs only ---
    print(f"[subset] opening (backed) {big_h5ad}")
    ad_b = sc.read_h5ad(big_h5ad, backed="r")  # on-disk; X not loaded
    n_total = int(ad_b.n_obs)
    if n_total == 0:
        raise RuntimeError("Source .h5ad has zero cells.")
    n_take = min(int(n_cells), n_total)

    # We'll read obs as a DataFrame (materializes obs only)
    obs_df = ad_b.obs.copy()
    if strat_key in obs_df.columns:
        labels = obs_df[strat_key].astype(str)
        # proportional allocation across groups
        rng = np.random.default_rng(seed)
        counts = labels.value_counts()
        probs = counts / counts.sum()
        alloc = np.floor(probs * n_take).astype(int)
        # distribute remainder by largest fractional parts
        remainder = n_take - int(alloc.sum())
        if remainder > 0:
            fracs = (probs * n_take) - alloc
            for g in fracs.sort_values(ascending=False).index[:remainder]:
                alloc.loc[g] += 1

        idx_list = []
        for g, k in alloc.items():
            if k <= 0:
                continue
            members = np.where(labels.values == g)[0]
            k = min(int(k), len(members))
            if k > 0:
                idx_list.append(rng.choice(members, size=k, replace=False))
        if not idx_list:
            raise RuntimeError("Stratified sampling produced no indices (unexpected).")
        idx_take = np.concatenate(idx_list)
        rng.shuffle(idx_take)
        print(f"[subset] stratified by '{strat_key}' → {idx_take.size} cells")
    else:
        # uniform sample
        rng = np.random.default_rng(seed)
        idx_take = rng.choice(n_total, size=n_take, replace=False)
        print(f"[subset] '{strat_key}' not found; uniform sample of {n_take} cells")

    # --- 2) Slice rows to memory (only subset rows get materialized) ---
    #     Note: ad_b[idx] with backed='r' returns a view; .to_memory() loads only those rows.
    ad_sub = ad_b[idx_take].to_memory()
    # Close the backed file handle if present
    try:
        if getattr(ad_b, "file", None) is not None:
            ad_b.file.close()
    except Exception:
        pass

    # --- 3) Normalize/log1p and dtype ---
    normalize_and_log1p(ad_sub, target_sum=normalize_target, do_log1p=log1p)
    # keep sparse unless user explicitly wants dense
    if dense and sp.issparse(ad_sub.X):
        ad_sub.X = ad_sub.X.toarray()
    ensure_float32_x(ad_sub)

    # --- 4) Create splits (stratified where viable) ---
    if strat_key in ad_sub.obs.columns:
        ensure_categorical(ad_sub, [strat_key])
        strat_keys = (strat_key,)
    else:
        strat_keys = tuple()
    cfg = SplitConfig(val_frac=val_frac, test_frac=test_frac, seed=seed, stratify_keys=strat_keys)
    _tr, _va, _te = make_splits(ad_sub, cfg)  # writes ad_sub.obs['split']

    # --- 5) Metadata ---
    import json as _json
    ad_sub.uns.setdefault("scfm_meta", {})
    meta = ad_sub.uns["scfm_meta"]
    meta.update({
        "from_big": str(big_h5ad),
        "n_cells": int(ad_sub.n_obs),
        "n_genes": int(ad_sub.n_vars),
        "strat_key": strat_key if strat_keys else None,
        "normalize_target": float(normalize_target),
        "log1p": bool(log1p),
        "dense": bool(dense),
        "splits": {
            "val_frac": float(val_frac),
            "test_frac": float(test_frac),
            "seed": int(seed),
            "stratify_keys": list(strat_keys),
        },
    })
    ad_sub.uns["scfm_meta"] = meta  # ensure writeback

    # --- 6) Save subset ---
    ad_sub.write(out_path, compression="lzf")
    print(f"[subset] wrote {out_path}  (cells={ad_sub.n_obs:,}, genes={ad_sub.n_vars:,})")
    return out_path


def main() -> None:
    # ---------- Choose data ----------
    if BIG_H5AD.exists():
        # Build a stratified random subset from the big file
        subset_path = ROOT / "data" / f"census_subset.D{N}.V1024.log1p.h5ad"
        data = make_subset_from_big(BIG_H5AD, subset_path, n_cells=N, seed=42)
    else:
        # Fallback to builder (pbmc/etc. depending on your env)
        print(f"[warn] Big file not found at {BIG_H5AD}. Using ensure_dataset instead.")
        data = ensure_dataset(cells=N, hvg=1024, out=ROOT / "data" / f"hlca_minified.D{N}.V1024.log1p.h5ad")

    # ---------- Train once ----------
    # Use gradient accumulation to reach target effective batch
    accum = max(1, target_batch // micro_batch)
    extra = ["--accum", str(accum)]
    train_once(
        data=data,
        outdir=ROOT / "runs" / "demo_xs",
        size="XS",
        steps=steps,
        val_every=val_every,
        batch_size=micro_batch,            # actual micro-batch
        log_every=log_every,
        train_metrics_every=train_metrics_every,
        num_workers=num_workers,
        extra_args=extra,                  # pass --accum to train.py
    )

    # ---------- Sweep ----------
    sweep_root = sweep(
        data=data,
        sizes=sizes,
        seeds=seeds,
        steps=steps,
        val_every=val_every,
        log_every=log_every,
        batch_size=micro_batch,
        train_metrics_every=train_metrics_every,
        num_workers=num_workers,
        # If you want AMP: add extra_args=["--amp"] in pipeline.api if you expose it there
    )

    # ---------- Analyze ----------
    df, fits = analyze_runs(runs=sweep_root, out=ROOT / "analysis")
    if df.empty:
        print("[analyze] no completed runs found under:", sweep_root)
    else:
        print(df.sort_values(["params", "seed"]))
        print("Fit keys:", list(fits.keys()))
        print("Params fit summary:", fits.get("params"))


if __name__ == "__main__":
    main()
