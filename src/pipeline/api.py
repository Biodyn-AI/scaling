# src/pipeline/api.py
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# Paths
SRC_DIR = Path(__file__).resolve().parents[1]   # .../src
PROJ_ROOT = SRC_DIR.parent                      # repo root

DATA_MOD = "src.data.data"      # we call these modules via -m
TRAIN_MOD = "model.train"

# ---------------------------
# small subprocess wrapper
# ---------------------------
def _run(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    import subprocess
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd or PROJ_ROOT), env=env or os.environ.copy())
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def _env_with_src() -> dict:
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) if not old else old + os.pathsep + str(SRC_DIR)
    return env

# ===========================
# DATA
# ===========================
def ensure_dataset(
    *,
    cells: int = 2500,
    hvg: int = 1024,
    source: str = "hlca-minified",
    revision: Optional[str] = None,
    out: Optional[str | Path] = None,
) -> Path:
    """
    Return a processed .h5ad; build it if missing.

    Note: with your current env, HLCA hub may not expose raw counts; the builder
    in src.data.data will fall back to pbmc3k (~2.7k cells). So asking for 20k
    may still yield ~2.7k (file name is cosmetic).
    """
    if out is None:
        out = PROJ_ROOT / f"data/{source.replace('-', '_')}.D{cells}.V{hvg}.log1p.h5ad"
    out = Path(out)
    if out.exists():
        print(f"[dataset] found: {out}")
        return out

    print(f"[dataset] building: {out} (cells={cells}, hvg={hvg})")
    env = _env_with_src()
    cmd = [
        sys.executable, "-m", DATA_MOD,
        "--source", source,
        "--out", str(out),
        "--max-cells", str(int(cells)),
        "--hvg", str(int(hvg)),
    ]
    if revision:
        cmd += ["--revision", revision]
    _run(cmd, env=env)
    return out

# ===========================
# TRAIN ONE RUN
# ===========================
# ===========================
# TRAIN ONE RUN
# ===========================
def train_once(
    *,
    data: str | Path,
    outdir: str | Path,
    size: str = "XS",
    steps: int = 500,
    val_every: int = 50,
    batch_size: int = 32,                 # fallback when no micro_batch given
    micro_batch: int | None = None,       # NEW: per-step batch
    target_batch: int | None = None,      # NEW: desired effective batch (micro * accum)
    accum: int | None = None,             # NEW: override accumulation directly
    mask_rate: float = 0.15,
    log_every: int = 50,
    train_metrics_every: Optional[int] = None,
    plot_smooth: int = 10,
    device: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    num_workers: int = 0,
    amp: bool = False,                    # NEW: enable autocast in train.py
    cap_genes: Optional[int] = None,      # NEW: forward to train.py
    lr: Optional[float] = None,           # NEW: optionally override LR
) -> Path:
    """
    Launch one training job via `python -m model.train`.

    If micro_batch and target_batch are provided, we compute:
        accum = ceil(target_batch / micro_batch)
    and pass:
        --batch-size micro_batch  --accum accum
    Otherwise we pass:
        --batch-size batch_size   (and --accum if given explicitly)
    """
    import math

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Decide batch + accumulation
    if micro_batch is not None:
        bs = int(micro_batch)
        if accum is None and target_batch is not None:
            accum_val = max(1, int(math.ceil(float(target_batch) / float(bs))))
        else:
            accum_val = max(1, int(accum)) if accum is not None else 1
    else:
        bs = int(batch_size)
        accum_val = max(1, int(accum)) if accum is not None else 1

    env = _env_with_src()
    cmd = [
        sys.executable, "-m", TRAIN_MOD,
        "--data", str(Path(data)),
        "--size", size,
        "--batch-size", str(bs),
        "--steps", str(int(steps)),
        "--val-every", str(int(val_every)),
        "--mask-rate", str(mask_rate),
        "--log-every", str(int(log_every)),
        "--outdir", str(outdir),
        "--num-workers", str(int(num_workers)),
        "--plot-smooth", str(int(plot_smooth)),
    ]
    if accum_val > 1:
        cmd += ["--accum", str(accum_val)]
    if device:
        cmd += ["--device", device]
    if train_metrics_every is not None:
        cmd += ["--train-metrics-every", str(int(train_metrics_every))]
    if amp:
        cmd += ["--amp"]
    if cap_genes is not None:
        cmd += ["--cap-genes", str(int(cap_genes))]
    if lr is not None:
        cmd += ["--lr", str(float(lr))]
    if extra_args:
        cmd += list(extra_args)

    _run(cmd, env=env)
    return outdir


# ===========================
# SWEEP
# ===========================
def sweep(
    *,
    data: str | Path,
    sizes: Sequence[str] = ("XS", "S", "M", "L"),
    seeds: Sequence[int] = (7, 13),
    steps: int = 500,
    val_every: int = 50,
    batch_size: int = 32,
    micro_batch: int | None = None,        # NEW
    target_batch: int | None = None,       # NEW
    accum: int | None = None,              # NEW
    log_every: int = 50,
    train_metrics_every: Optional[int] = None,
    plot_smooth: int = 10,
    num_workers: int = 0,
    root: Optional[str | Path] = None,
    amp: bool = False,                     # NEW
    cap_genes: Optional[int] = None,       # NEW
    lr: Optional[float] = None,            # NEW
) -> Path:
    """
    Run size×seed grid and return the sweep root directory.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = Path(root or (PROJ_ROOT / f"runs/sweep_{ts}"))
    root.mkdir(parents=True, exist_ok=True)

    for s in sizes:
        for seed in seeds:
            outdir = root / f"{s}_seed{seed}"
            extra = ["--seed", str(int(seed))]
            print(f"[sweep] {s} seed={seed} -> {outdir}")
            train_once(
                data=data, outdir=outdir, size=s,
                steps=steps, val_every=val_every,
                batch_size=batch_size,
                micro_batch=micro_batch, target_batch=target_batch, accum=accum,
                log_every=log_every, train_metrics_every=train_metrics_every,
                plot_smooth=plot_smooth, num_workers=num_workers,
                extra_args=extra, amp=amp, cap_genes=cap_genes, lr=lr,
            )
    print(f"[sweep] done -> {root}")
    return root

# ===========================
# ANALYZE
# ===========================
def _find_run_dirs(root: Path) -> List[Path]:
    """Recursively find dirs that contain history and a checkpoint."""
    run_dirs: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        files = set(filenames)
        if "history.jsonl" in files and (("ckpt_best.pt" in files) or ("ckpt_final.pt" in files)):
            run_dirs.append(Path(dirpath))
    return sorted(run_dirs)

def _parse_history(run_dir: Path) -> Tuple[List[dict], float, float, int]:
    rows: List[dict] = []
    best_val = float("inf")
    best_mae = float("inf")
    best_step = -1
    fp = run_dir / "history.jsonl"
    if not fp.exists():
        return rows, best_val, best_mae, best_step
    with open(fp, "r") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            rows.append(j)
            if "val_mse" in j:
                vm = float(j["val_mse"])
                if vm < best_val:
                    best_val = vm
                    best_mae = float(j.get("val_mae", np.nan))
                    best_step = int(j.get("step", -1))
    return rows, best_val, best_mae, best_step

def _count_params_from_ckpt(ckpt: Path) -> Tuple[int, Dict]:
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck.get("model_state") or ck.get("state_dict") or {}
    n_params = 0
    for v in sd.values():
        try:
            n_params += int(np.prod(v.shape))
        except Exception:
            pass
    cfg = ck.get("config", {})
    return n_params, cfg

def _fit_power_law(df: pd.DataFrame, y_col: str = "best_val_mse") -> Dict[str, float]:
    """
    Fit y ≈ a * P^{-alpha} + c using a small grid over c and linear fit in log-space.
    """
    sub = df.dropna(subset=["params", y_col]).sort_values("params")
    if len(sub) < 3:
        return {"alpha": np.nan, "a": np.nan, "c": np.nan, "r2": np.nan}
    P = sub["params"].to_numpy(float)
    y = sub[y_col].to_numpy(float)
    cs = np.linspace(0.0, max(1e-9, 0.95 * y.min()), 50)
    best = {"r2": -np.inf}
    X = np.vstack([np.ones_like(P), -np.log(P)]).T  # columns: [1, -log P]
    for c in cs:
        z = y - c
        if np.any(z <= 0):
            continue
        t = np.log(z)
        beta, *_ = np.linalg.lstsq(X, t, rcond=None)
        loga, alpha = beta
        pred = X @ beta
        ss_res = np.sum((t - pred) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if r2 > best["r2"]:
            best = {"alpha": float(alpha), "a": float(np.exp(loga)), "c": float(c), "r2": float(r2)}
    if best["r2"] == -np.inf:
        return {"alpha": np.nan, "a": np.nan, "c": np.nan, "r2": np.nan}
    return best

def analyze_runs(
    *,
    runs: str | Path = PROJ_ROOT / "runs",
    out: str | Path = PROJ_ROOT / "analysis",
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Aggregate completed runs (recursively), write CSV/JSON, and return (df, fits)."""
    runs = Path(runs)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    rdirs = _find_run_dirs(runs)
    if not rdirs:
        print("[analyze] no runs with history.jsonl + checkpoint found under:", runs)
        return pd.DataFrame(), {}

    recs: List[Dict] = []
    for d in rdirs:
        rows, best_val, best_mae, best_step = _parse_history(d)
        ckpt = d / "ckpt_best.pt"
        if not ckpt.exists():
            ckpt = d / "ckpt_final.pt"
        if not (ckpt.exists() and rows):
            continue
        params, cfg = _count_params_from_ckpt(ckpt)
        recs.append({
            "run": str(d),
            "size": cfg.get("size"),
            "seed": cfg.get("seed"),
            "params": int(params),
            "steps": max([int(r.get("step", 0)) for r in rows] or [0]),
            "best_step": int(best_step),
            "best_val_mse": float(best_val) if np.isfinite(best_val) else np.nan,
            "best_val_mae": float(best_mae) if np.isfinite(best_mae) else np.nan,
            "overrides": json.dumps(cfg.get("overrides")) if cfg.get("overrides") else None,
        })

    if not recs:
        print("[analyze] found run folders, but none had usable history + checkpoint.")
        return pd.DataFrame(), {}

    df = pd.DataFrame.from_records(recs).sort_values(["params", "seed"])
    df.to_csv(out / "aggregate.csv", index=False)

    fits = {
        "val_mse": _fit_power_law(df, "best_val_mse"),
        "val_mae": _fit_power_law(df, "best_val_mae"),
        "params": {"min": int(df["params"].min()), "max": int(df["params"].max()), "n": int(df["params"].nunique())},
    }
    with open(out / "fits.json", "w") as f:
        json.dump(fits, f, indent=2)

    print(f"[analyze] wrote {out/'aggregate.csv'} and {out/'fits.json'}")
    return df, fits
