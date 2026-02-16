#!/usr/bin/env python3
"""
Smoke test for the training pipeline (model.train).

This script will:
  1) Ensure a small processed .h5ad exists (build one if missing)
  2) Launch the trainer (python -m model.train) for a short run
  3) Check that checkpoints, logs (and optional plots) were written
  4) Print a compact summary

Run from the project root, e.g.:

  python model_test.py \
    --data ./data/hlca_minified.D20k.V2000.log1p.h5ad \
    --outdir ./runs/smoke_xs_D20k --steps 100 \
    --val-every 5 --plot-smooth 10
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.resolve()
SRC_DIR = ROOT / "src"
DATA_BUILDER_MOD = "src.data.data"  # implicit namespace pkg is fine
TRAIN_MOD = "model.train"


def run(cmd, **kwargs):
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, **kwargs)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def build_dataset(out_path: Path, cells: int, hvg: int = 2000) -> Path:
    """Always (re)build a dataset to out_path with the requested size."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    cmd = [
        sys.executable, "-m", DATA_BUILDER_MOD,
        "--out", str(out_path),
        "--max-cells", str(int(cells)),
        "--hvg", str(int(hvg)),
    ]
    run(cmd, cwd=str(ROOT), env=env)
    return out_path


def ensure_dataset(path: Path, fallback_cells: int = 5000, hvg: int = 2000) -> Path:
    """Build only if missing (legacy behaviour)."""
    if path.exists():
        print(f"[dataset] found: {path}")
        return path
    print(f"[dataset] not found, building a small dataset at {path} …")
    return build_dataset(path, cells=fallback_cells, hvg=hvg)


def main():
    ap = argparse.ArgumentParser(description="Smoke test for model training")
    # Data source / building
    ap.add_argument("--data", default=str(ROOT / "data/hlca_minified.D20k.V2000.log1p.h5ad"))
    ap.add_argument("--cells", type=int, default=None,
                    help="If set, (re)build/use a canonical data file with exactly N cells")
    ap.add_argument("--hvg", type=int, default=2000,
                    help="Number of genes kept when building a dataset")
    # Output & core training knobs
    ap.add_argument("--outdir", default=str(ROOT / "runs/smoke_xs"))
    ap.add_argument("--size", default="L", choices=["XS", "S", "M", "L"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--clean", action="store_true", help="Remove outdir before run")
    # Trainer pass-throughs (optional)
    ap.add_argument("--train-only", action="store_true",
                    help="Use only obs['split']==train if present in the .h5ad")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="Cap number of train cells in-memory (no rebuild)")
    ap.add_argument("--log-every", type=int, default=5,
                    help="Trainer prints progress every N steps")
    ap.add_argument("--eta-window", type=int, default=50,
                    help="Window in steps to average step time for ETA")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="Stop early after this wall-clock time (minutes)")
    # Plotting toggles
    ap.add_argument("--plot-smooth", type=int, default=0,
                    help="Moving-average window for train curves (0 = off)")
    ap.add_argument("--no-plots", action="store_true",
                    help="Disable PNG plot generation")
    ap.add_argument("--train-metrics-every", type=int, default=5,
                    help="Record train metrics to history every N steps")
    args = ap.parse_args()

    # Decide input file
    if args.cells is not None:
        data_path = ROOT / f"data/hlca_minified.D{args.cells}.V{args.hvg}.log1p.h5ad"
        if data_path.exists():
            print(f"[dataset] found: {data_path}")
        else:
            print(f"[dataset] building: {data_path}  (cells={args.cells}, hvg={args.hvg})")
            build_dataset(data_path, args.cells, args.hvg)
    else:
        data_path = ensure_dataset(Path(args.data))

    outdir = Path(args.outdir)
    if args.clean and outdir.exists():
        print(f"[clean] removing existing outdir: {outdir}")
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Train via module invocation; ensure Python can find src/ (for model package)
    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) if not old_pp else old_pp + os.pathsep + str(SRC_DIR)

    cmd = [
        sys.executable, "-m", TRAIN_MOD,
        "--data", str(data_path),
        "--size", args.size,
        "--batch-size", str(args.batch_size),
        "--steps", str(args.steps),
        "--mask-rate", str(args.mask_rate),
        "--val-every", str(args.val_every),
        "--outdir", str(outdir),
    ]

    # Optional trainer flags
    if args.train_only:
        cmd += ["--train-only"]
    if args.limit_train is not None:
        cmd += ["--limit-train", str(int(args.limit_train))]
    if args.log_every is not None:
        cmd += ["--log-every", str(int(args.log_every))]
    if args.eta_window is not None:
        cmd += ["--eta-window", str(int(args.eta_window))]
    if args.max_minutes is not None:
        cmd += ["--max-minutes", str(float(args.max_minutes))]

    # Plotting flags
    if args.plot_smooth:
        cmd += ["--plot-smooth", str(int(args.plot_smooth))]
    if args.no_plots:
        cmd += ["--no-plots"]

    # Run
    run(cmd, cwd=str(ROOT), env=env)

    # Checks
    ck_best = outdir / "ckpt_best.pt"
    ck_final = outdir / "ckpt_final.pt"
    hist = outdir / "history.jsonl"
    png_mse = outdir / "curves_mse.png"
    png_mae = outdir / "curves_mae.png"

    errors = []
    if not ck_best.exists():
        errors.append("missing ckpt_best.pt")
    if not ck_final.exists():
        errors.append("missing ckpt_final.pt")
    if not hist.exists():
        errors.append("missing history.jsonl")

    # Plots are optional
    if not args.no_plots:
        # Don't fail if tiny runs didn't create both, but warn
        if not png_mse.exists() or not png_mae.exists():
            print("[warn] plots not found; set --plot-smooth or increase steps/val cadence")

    if errors:
        print("[ERROR]", "; ".join(errors))
        raise SystemExit(1)

    # Summarize history
    steps_v, val_mse, val_mae = [], [], []
    steps_t, train_mse, train_mae = [], [], []
    with open(hist, "r") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            s = j.get("step")
            if "val_mse" in j:
                steps_v.append(s)
                val_mse.append(float(j["val_mse"]))
                if "val_mae" in j:
                    val_mae.append(float(j["val_mae"]))
            if "train_mse" in j:
                steps_t.append(s)
                train_mse.append(float(j["train_mse"]))
                if "train_mae" in j:
                    train_mae.append(float(j["train_mae"]))

    print("\n[summary]")
    print(json.dumps({
        "data": str(data_path),
        "outdir": str(outdir),
        "n_train_points": len(train_mse),
        "n_val_points": len(val_mse),
        "train_mse_last": None if not train_mse else train_mse[-1],
        "val_mse_first": None if not val_mse else val_mse[0],
        "val_mse_last": None if not val_mse else val_mse[-1],
        "val_mae_last": None if not val_mae else val_mae[-1],
        "plots": None if args.no_plots else {
            "mse_png": str(png_mse) if png_mse.exists() else None,
            "mae_png": str(png_mae) if png_mae.exists() else None,
        }
    }, indent=2))

    # Soft checks
    if val_mse and not (val_mse[-1] == val_mse[-1]):
        print("[WARN] val_mse is NaN at last point")

    print("\nSmoke test OK.")


if __name__ == "__main__":
    main()
