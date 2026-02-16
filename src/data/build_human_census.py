#!/usr/bin/env python3
"""
Build a large human scRNA dataset from CELLxGENE Census and write it to .h5ad.

Robustness:
- Handles layouts with measurements under .../homo_sapiens/ms/<MEAS>/ (new) or .../homo_sapiens/<MEAS>/ (old).
- Auto-detects a measurement with both 'var' and 'X'.
- Finds a usable gene column in var (feature_name → feature_id → soma_feature_id).
- Prefers X='raw' if present, else first available.
- Picks cells by tissues (fallback to tissue_general), sharded fetch.
- Uses var_coords (row index) instead of server-side IN filters to avoid freezes.

Usage (sequential):
  python -m src.data.build_human_census \
    --census-version 2025-01-30 --target-cells 200000 --per-tissue-cap 40000 \
    --genes 512 --shard 2000

Threaded (macOS-friendly, no multiprocessing):
  python -m src.data.build_human_census \
    --census-version 2025-01-30 --target-cells 200000 --per-tissue-cap 40000 \
    --genes 512 --shard 2000 --workers 4
"""
from __future__ import annotations

import argparse
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import scipy.sparse as sp
import cellxgene_census as cxg


# -----------------------
# Logging
# -----------------------
def _log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------
# Defaults
# -----------------------
DEFAULT_TISSUES = [
    "blood", "lung", "brain", "kidney", "liver",
    "heart", "intestine", "pancreas", "skin", "spleen",
]


# -----------------------
# Schema helpers
# -----------------------
def _child_names(group) -> List[str]:
    try:
        return list(group.keys())
    except Exception:
        return [k for k in group]


def _experiment(cen):
    return cen["census_data"]["homo_sapiens"]


def _has_ms_container(exp) -> bool:
    return "ms" in _child_names(exp)


def _measurement_group(exp, meas_name: str):
    if _has_ms_container(exp):
        return exp["ms"][meas_name]
    return exp[meas_name]


def _list_measurements(exp) -> List[str]:
    if _has_ms_container(exp):
        return _child_names(exp["ms"])
    cands = []
    for name in _child_names(exp):
        try:
            g = exp[name]
            kids = set(_child_names(g))
            if "var" in kids and "X" in kids:
                cands.append(name)
        except Exception:
            pass
    return cands


def _obs_filter_text(tissue: str, extra: Optional[str], col: str = "tissue") -> str:
    base = f'{col} == "{tissue}"'
    return f"({base}) and ({extra})" if extra else base


def _detect_measurement_with_var_and_x(
    cen,
    preferred_meas: Sequence[str] = ("RNA", "rna"),
    preferred_x: Sequence[str] = ("raw", "Raw", "counts", "Count", "normalized"),
    gene_cols: Sequence[str] = ("feature_name", "feature_id", "soma_feature_id"),
) -> Tuple[str, str, str]:
    exp = _experiment(cen)
    meas_names_all = _list_measurements(exp)
    if not meas_names_all:
        raise RuntimeError(
            f"No usable measurement found. Experiment children: {_child_names(exp)} "
            f"(if 'ms' is present, it may be empty in this snapshot)."
        )

    order = [m for m in preferred_meas if m in meas_names_all] + [m for m in meas_names_all if m not in preferred_meas]
    for meas in order:
        mg = _measurement_group(exp, meas)
        kids = set(_child_names(mg))
        if "var" not in kids or "X" not in kids:
            continue

        # pick X name
        x_children = _child_names(mg["X"])
        if not x_children:
            continue
        x_name = next((xp for xp in preferred_x if xp in x_children), x_children[0])

        # find a gene column in var
        var_df = mg["var"]
        var_cols = None
        try:
            var_cols = [f.name for f in var_df.schema]
        except Exception:
            pass

        gene_col = None
        if var_cols:
            for gc in gene_cols:
                if gc in var_cols:
                    gene_col = gc
                    break
        else:
            for gc in gene_cols:
                try:
                    var_df.read(column_names=[gc])
                    gene_col = gc
                    break
                except Exception:
                    continue

        if gene_col is None:
            continue

        _log(f"[meas] selected measurement '{meas}' | X='{x_name}' | gene_col='{gene_col}'")
        return meas, gene_col, x_name

    raise RuntimeError(f"No usable measurement among: {meas_names_all}")


# -----------------------
# Picking obs joinids
# -----------------------
def pick_joinids_once(
    cen,
    tissues: Sequence[str],
    per_tissue_cap: int,
    extra_filter: Optional[str],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pieces: List[np.ndarray] = []

    for t in tissues:
        df = None
        used_col = "tissue"
        try:
            vf = _obs_filter_text(t, extra_filter, col="tissue")
            df = cxg.get_obs(
                cen, organism="Homo sapiens",
                value_filter=vf,
                column_names=["soma_joinid", "tissue"],
            )
        except Exception as e1:
            if "tissue" in str(e1) and "does not exist" in str(e1):
                vf = _obs_filter_text(t, extra_filter, col="tissue_general")
                df = cxg.get_obs(
                    cen, organism="Homo sapiens",
                    value_filter=vf,
                    column_names=["soma_joinid", "tissue_general"],
                )
                used_col = "tissue_general"
            else:
                raise

        n = len(df) if df is not None else 0
        take = min(per_tissue_cap, n)
        if take <= 0:
            _log(f"[pick] {t:<10}: 0 cells (skipped)")
            continue

        idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        joinids = df.iloc[idx]["soma_joinid"].to_numpy(dtype=np.int64)
        pieces.append(joinids)
        _log(f"[pick] {t:<10}: picked {take:,} joinids (via {used_col})")

    if not pieces:
        raise RuntimeError("No cells matched your tissue filters.")
    all_ids = np.concatenate(pieces, axis=0)
    _log(f"[pick] total selected joinids: {all_ids.size:,}")
    return all_ids


# -----------------------
# Choosing genes (index → var_coords)
# -----------------------
def choose_genes_once(
    cen,
    max_genes: int,
    forced_meas: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str, str, str]:
    """
    Returns:
      gene_vals   : ndarray[str] (e.g., feature_name or feature_id)
      var_coords  : ndarray[int64] row index values for those genes (soma_feature_id or equivalent index)
      gene_col    : which column gene_vals came from
      meas_name   : measurement name (e.g., 'RNA')
      x_name      : X matrix name (e.g., 'raw')
    """
    exp = _experiment(cen)

    if forced_meas is None:
        meas, gene_col, x_name = _detect_measurement_with_var_and_x(cen)
    else:
        mg = _measurement_group(exp, forced_meas)
        kids = set(_child_names(mg))
        if "var" not in kids or "X" not in kids:
            raise RuntimeError(f"Forced measurement '{forced_meas}' lacks 'var' and/or 'X'")
        # pick a reasonable gene_col
        var_df = mg["var"]
        var_cols = None
        try:
            var_cols = [f.name for f in var_df.schema]
        except Exception:
            pass
        gene_col = None
        for gc in ("feature_name", "feature_id", "soma_feature_id"):
            if var_cols and gc in var_cols:
                gene_col = gc
                break
            if not var_cols:
                try:
                    var_df.read(column_names=[gc])
                    gene_col = gc
                    break
                except Exception:
                    pass
        if gene_col is None:
            raise RuntimeError(f"Could not find a gene column in var for measurement '{forced_meas}'")
        # X name
        x_children = _child_names(mg["X"])
        x_name = "raw" if "raw" in x_children else (x_children[0] if x_children else None)
        if x_name is None:
            raise RuntimeError(f"Measurement '{forced_meas}' has no X matrices")
        meas = forced_meas

    # Read only the gene column; take the DataFrame INDEX as var coords.
    var_df = _measurement_group(exp, meas)["var"]
    tbl = var_df.read(column_names=[gene_col]).concat().to_pandas()
    # In SOMADataFrame, the index is the coordinate (e.g., soma_feature_id), even if not a schema column.
    var_coords = tbl.index.to_numpy(dtype=np.int64)
    gene_vals = tbl[gene_col].astype(str).to_numpy()

    if max_genes and max_genes < len(gene_vals):
        gene_vals = gene_vals[: max_genes]
        var_coords = var_coords[: max_genes]

    _log(f"[genes] selected {len(gene_vals):,} genes from '{meas}' via '{gene_col}' (X='{x_name}')")
    return gene_vals, var_coords, gene_col, meas, x_name


# -----------------------
# Shard fetch (single-process; optional threads)
# -----------------------
def _fetch_worker(
    obs_ids: Sequence[int],
    var_coords: Sequence[int],
    gene_col: str,
    meas_name: str,
    x_name: str,
    census_version: Optional[str],
):
    """
    Fetch one shard and return:
      (X.data, X.indices, X.indptr, X.shape, obs_joinid, obs_tissue, var_names)
    """
    import numpy as _np
    import scipy.sparse as _sp
    import cellxgene_census as _cxg

    ctx = _cxg.get_default_soma_context()
    with _cxg.open_soma(census_version=census_version, context=ctx) as cen:
        try:
            adata = _cxg.get_anndata(
                cen,
                organism="Homo sapiens",
                measurement_name=meas_name,
                X_name=x_name,
                obs_coords=_np.asarray(obs_ids, dtype=_np.int64),
                var_coords=_np.asarray(var_coords, dtype=_np.int64),  # key: coords, not IN filter
                obs_column_names=["soma_joinid", "tissue"],
                var_column_names=[gene_col],
            )
            tissue_col = "tissue"
        except Exception as e1:
            if "tissue" in str(e1) and "does not exist" in str(e1):
                adata = _cxg.get_anndata(
                    cen,
                    organism="Homo sapiens",
                    measurement_name=meas_name,
                    X_name=x_name,
                    obs_coords=_np.asarray(obs_ids, dtype=_np.int64),
                    var_coords=_np.asarray(var_coords, dtype=_np.int64),
                    obs_column_names=["soma_joinid", "tissue_general"],
                    var_column_names=[gene_col],
                )
                tissue_col = "tissue_general"
            else:
                raise

    X = adata.X if _sp.issparse(adata.X) else _sp.csr_matrix(adata.X)
    return (
        X.data,
        X.indices,
        X.indptr,
        X.shape,
        adata.obs["soma_joinid"].to_numpy(),
        adata.obs[tissue_col].astype(str).to_numpy(),
        adata.var[gene_col].astype(str).to_numpy(),
    )


# -----------------------
# Orchestration
# -----------------------
@dataclass
class Args:
    out: Path
    census_version: Optional[str]
    tissues: List[str]
    target_cells: int
    per_tissue_cap: int
    genes: int
    shard: int
    seed: int
    compression: str
    obs_extra_filter: Optional[str]
    timeout: Optional[float]   # kept for compat
    no_mp: bool                # kept for compat (we don’t spawn processes)
    workers: int               # threads; 0 => sequential
    measurement_name: Optional[str]


def build(args: Args) -> Path:
    _log("[build] target cells ~{:,} across {} tissues:\n         {}".format(
        args.target_cells, len(args.tissues), ", ".join(args.tissues)
    ))
    _log(f"[build] per-tissue cap: {args.per_tissue_cap:,}; shard size: {args.shard:,}")

    ctx = cxg.get_default_soma_context()
    with cxg.open_soma(census_version=args.census_version, context=ctx) as cen:
        joinids_all = pick_joinids_once(
            cen, tissues=args.tissues, per_tissue_cap=args.per_tissue_cap,
            extra_filter=args.obs_extra_filter, seed=args.seed,
        )

        if joinids_all.size > args.target_cells:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(joinids_all.size, size=args.target_cells, replace=False)
            joinids_all = joinids_all[idx]
            _log(f"[pick] total selected joinids (post-cap): {joinids_all.size:,}")

        gene_vals, var_coords, gene_col, meas_name, x_name = choose_genes_once(
            cen, max_genes=args.genes, forced_meas=args.measurement_name
        )

    # Shard loop
    N = joinids_all.size
    shard = max(1, int(args.shard))
    n_shards = (N + shard - 1) // shard
    _log(f"[fetch] total={N:,} joinids | shards={n_shards} (size ~{shard:,})")

    datas: List[np.ndarray] = []
    indxs: List[np.ndarray] = []
    indptrs: List[np.ndarray] = []
    shapes: List[Tuple[int, int]] = []
    obs_ids_out: List[np.ndarray] = []
    tissues_out: List[np.ndarray] = []
    var_names_ref: Optional[np.ndarray] = None

    slices = [(i, joinids_all[i * shard: min(N, (i + 1) * shard)]) for i in range(n_shards)]

    def _fetch_one(i: int, cur):
        a = i * shard
        b = min(N, (i + 1) * shard)
        _log(f"[fetch] {a:8d} – {b:8d} (n={b - a:,}) …")
        return i, _fetch_worker(cur, var_coords, gene_col, meas_name, x_name, args.census_version)

    if args.workers and args.workers > 0:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        _log(f"[fetch] using ThreadPool with {args.workers} workers")
        results = [None] * n_shards
        errors = {}
        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = {ex.submit(_fetch_one, i, cur): i for i, cur in slices}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    idx, res = fut.result()
                    results[idx] = res
                except Exception as e:
                    errors[i] = str(e)
        if errors:
            first_bad = sorted(errors.items())[0]
            _log(f"[error] {len(errors)} shards failed. First: shard {first_bad[0]+1}/{n_shards}: {first_bad[1]}")
            raise SystemExit(2)

        ordered = [results[i] for i in range(n_shards)]
        for (data, indices, indptr, shape, obs_ids, obs_tissue, var_names) in ordered:
            datas.append(np.asarray(data))
            indxs.append(np.asarray(indices, dtype=np.int32))
            indptrs.append(np.asarray(indptr, dtype=np.int32))
            shapes.append((int(shape[0]), int(shape[1])))
            obs_ids_out.append(np.asarray(obs_ids, dtype=np.int64))
            tissues_out.append(np.asarray(obs_tissue, dtype=object))
            if var_names_ref is None:
                var_names_ref = np.asarray(var_names, dtype=object)
    else:
        for i, cur in slices:
            a = i * shard
            b = min(N, (i + 1) * shard)
            _log(f"[fetch] {a:8d} – {b:8d} (n={b - a:,}) …")
            try:
                data, indices, indptr, shape, obs_ids, obs_tissue, var_names = _fetch_worker(
                    cur, var_coords, gene_col, meas_name, x_name, args.census_version
                )
            except Exception as e:
                _log(f"[error] shard {i+1}/{n_shards}: {e}")
                raise SystemExit(2)

            datas.append(np.asarray(data))
            indxs.append(np.asarray(indices, dtype=np.int32))
            indptrs.append(np.asarray(indptr, dtype=np.int32))
            shapes.append((int(shape[0]), int(shape[1])))
            obs_ids_out.append(np.asarray(obs_ids, dtype=np.int64))
            tissues_out.append(np.asarray(obs_tissue, dtype=object))
            if var_names_ref is None:
                var_names_ref = np.asarray(var_names, dtype=object)

    # Concatenate CSR rows
    assert var_names_ref is not None
    n_rows = sum(s[0] for s in shapes)
    data = np.concatenate(datas, axis=0)
    indices = np.concatenate(indxs, axis=0)
    indptr = np.empty(n_rows + 1, dtype=np.int32)

    cursor = 0
    offset = 0
    for s, ip in zip(shapes, indptrs):
        rows = s[0]
        indptr[offset: offset + rows] = ip[:-1] + cursor
        cursor += int(ip[-1])
        offset += rows
    indptr[-1] = cursor

    X = sp.csr_matrix((data, indices, indptr), shape=(n_rows, shapes[0][1]))
    obs_joinid = np.concatenate(obs_ids_out, axis=0)
    obs_tissue = np.concatenate(tissues_out, axis=0)

    adata = ad.AnnData(X=X)
    adata.obs["soma_joinid"] = obs_joinid
    adata.obs["tissue"] = obs_tissue
    adata.var[gene_col] = var_names_ref
    if gene_col != "feature_name":
        adata.var["gene_symbol"] = var_names_ref
    adata.uns["source"] = {
        "dataset": "cellxgene-census",
        "census_version": args.census_version,
        "tissues": list(args.tissues),
        "target_cells": int(args.target_cells),
        "per_tissue_cap": int(args.per_tissue_cap),
        "genes": int(args.genes),
        "shard": int(args.shard),
        "gene_id_column": gene_col,
        "measurement_name": meas_name,
        "x_name": x_name,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out, compression=args.compression)
    _log(f"[write] {args.out}")
    return args.out


# -----------------------
# CLI
# -----------------------
@dataclass
class ArgsParsed(Args):
    pass


def parse_args() -> ArgsParsed:
    p = argparse.ArgumentParser(description="Build a large human scRNA dataset from CELLxGENE Census.")
    p.add_argument("--out", type=Path, default=None, help="Output .h5ad path (default: auto-named under data/)")
    p.add_argument("--census-version", default="2025-01-30", help="Census version (pin for reproducibility)")
    p.add_argument("--tissues", nargs="+", default=DEFAULT_TISSUES, help="List of tissues")
    p.add_argument("--target-cells", type=int, default=1_000_000)
    p.add_argument("--per-tissue-cap", type=int, default=100_000)
    p.add_argument("--genes", type=int, default=1024)
    p.add_argument("--shard", type=int, default=50_000, help="Shard size (cells per fetch)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "zstd"])
    p.add_argument("--obs-extra-filter", default=None, help='Extra obs filter, e.g. \'sex == "male"\'')
    p.add_argument("--timeout", type=float, default=900.0)  # kept for compat
    p.add_argument("--no-mp", action="store_true")          # kept for compat (no processes used)
    p.add_argument("--workers", type=int, default=0, help="Thread workers for shard fetching (0 = sequential)")
    p.add_argument("--measurement-name", default=None, help="Force measurement (e.g., RNA). If omitted, auto-detect.")
    ns = p.parse_args()

    if ns.out is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        auto = Path("data") / f"census_human.auto.D{ns.target_cells}.V{ns.genes}.{ts}.h5ad"
        _log(f"[info] --out not provided; using default: {auto}")
        out = auto
    else:
        out = Path(ns.out)

    return ArgsParsed(
        out=out,
        census_version=ns.census_version,
        tissues=list(ns.tissues),
        target_cells=int(ns.target_cells),
        per_tissue_cap=int(ns.per_tissue_cap),
        genes=int(ns.genes),
        shard=int(ns.shard),
        seed=int(ns.seed),
        compression=str(ns.compression),
        obs_extra_filter=ns.obs_extra_filter,
        timeout=float(ns.timeout) if ns.timeout is not None else None,
        no_mp=bool(ns.no_mp),
        workers=int(ns.workers),
        measurement_name=ns.measurement_name,
    )


def main():
    args = parse_args()
    build(args)


if __name__ == "__main__":
    main()
