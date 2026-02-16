#!/usr/bin/env python3
"""Minimal training loop for permutation‑invariant scRNA transformers with ETA and plotting.

Reads a processed .h5ad produced by src/data/data.py, builds a ScTransformer
(from presets or explicit sizes), and trains with a masked‑reconstruction loss.

Examples
--------
python -m model.train \
  --data ./data/hlca_minified.D20k.V2000.log1p.h5ad \
  --size XS --batch-size 256 --steps 2000 --val-every 200 --plot-smooth 20 \
  --outdir ./runs/xs_D20k
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import time
from collections import deque
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import nullcontext

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

# -------------------------------------------------
# STRICT: rely on project utilities in model/utils.py
# -------------------------------------------------
try:
    from .utils import build_model, masked_mse, masked_mae, masked_gaussian_nll, count_parameters
except Exception as e:
    raise ImportError(
        "Required utilities not found. Expected 'src/model/utils.py' to define build_model, "
        "masked_mse, masked_mae, masked_gaussian_nll, count_parameters. Run with 'python -m model.train' and ensure "
        "PYTHONPATH includes 'src'."
    ) from e


# -----------------------------
# Dataset
# -----------------------------
class MaskedExpressionDataset(Dataset):
    """Dataset that samples a fresh random mask each __getitem__.

    If X is sparse, it is densified once here for simplicity (OK for ~100k×2k).
    For larger data, switch to a sparse‑aware pipeline.
    """

    def __init__(self, X, mask_rate: float, mask_token_value: float = 0.0, seed: Optional[int] = None):
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        self.X = X
        self.n_cells, self.n_genes = X.shape
        self.mask_rate = float(mask_rate)
        self.mask_token_value = float(mask_token_value)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int):
        x = self.X[idx]
        m = self.rng.random(self.n_genes) < self.mask_rate
        x_in = x.copy()
        x_in[m] = self.mask_token_value
        return {
            "x_in": torch.from_numpy(x_in),       # [T]
            "x_tgt": torch.from_numpy(x),         # [T]
            "mask": torch.from_numpy(m.astype(np.bool_)),
        }


# -----------------------------
# Utilities
# -----------------------------

def pick_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch.backends.mps as mps  # type: ignore
        if mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class CollateWithGeneIds:
    """Top-level, picklable collate that injects shared gene_ids."""

    def __init__(self, vocab_size: int):
        self.gene_ids = torch.arange(vocab_size, dtype=torch.long)

    def __call__(self, batch):
        out = {k: torch.stack([b[k] for b in batch], dim=0) for k in ["x_in", "x_tgt", "mask"]}
        out["gene_ids"] = self.gene_ids
        return out


@torch.no_grad()
def evaluate(model, loader, device: str):
    model.eval()
    n = 0
    m_mse = 0.0
    m_mae = 0.0
    m_nll = 0.0
    for batch in loader:
        x_in = batch["x_in"].to(device)
        mask = batch["mask"].to(device)
        tgt = batch["x_tgt"].to(device)
        gene_ids = batch["gene_ids"].to(device)
        pred, _ = model(gene_ids, x_in, mask)
        m_mse += masked_mse(pred, tgt, mask).item()
        m_mae += masked_mae(pred, tgt, mask).item()
        m_nll += masked_gaussian_nll(pred, tgt, mask).item()
        n += 1
    return {
        "val_mse": m_mse / max(n, 1),
        "val_mae": m_mae / max(n, 1),
        "val_gauss_nll": m_nll / max(n, 1),
    }


def fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1 or k > len(x):
        return x
    w = np.ones(k, dtype=float) / k
    return np.convolve(x, w, mode="valid")


def save_plots(history: List[Dict[str, Any]], outdir: Path, smooth: int = 0) -> None:
    """
    Save MSE/MAE/Gaussian-NLL curves as PNGs.
    - Train curves: optionally smoothed with a moving average of length `smooth`
      (only if len(train_series) >= smooth).
    - Val curves: plotted as-is (usually sparse).
    - If a series has exactly one point, draw a visible dot instead of a line.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"plotting disabled (matplotlib unavailable): {e}")
        return

    # Collect series from history.jsonl-like rows
    steps_tr, tr_mse, tr_mae = [], [], []
    steps_va, va_mse, va_mae = [], [], []
    steps_tr_nll, tr_nll = [], []
    steps_va_nll, va_nll = [], []
    for row in history:
        s = row.get("step")
        if "train_mse" in row:
            steps_tr.append(s)
            tr_mse.append(float(row["train_mse"]))
            if "train_mae" in row:
                tr_mae.append(float(row["train_mae"]))
            if "train_gauss_nll" in row:
                steps_tr_nll.append(s)
                tr_nll.append(float(row["train_gauss_nll"]))
        if "val_mse" in row:
            steps_va.append(s)
            va_mse.append(float(row["val_mse"]))
            if "val_mae" in row:
                va_mae.append(float(row["val_mae"]))
            if "val_gauss_nll" in row:
                steps_va_nll.append(s)
                va_nll.append(float(row["val_gauss_nll"]))

    # Convert to arrays
    steps_tr = np.asarray(steps_tr, dtype=float)
    tr_mse = np.asarray(tr_mse, dtype=float)
    tr_mae = np.asarray(tr_mae, dtype=float)
    tr_nll = np.asarray(tr_nll, dtype=float)
    steps_tr_nll = np.asarray(steps_tr_nll, dtype=float)
    steps_va = np.asarray(steps_va, dtype=float)
    va_mse = np.asarray(va_mse, dtype=float)
    va_mae = np.asarray(va_mae, dtype=float)
    va_nll = np.asarray(va_nll, dtype=float)
    steps_va_nll = np.asarray(steps_va_nll, dtype=float)

    # --- helpers ---
    def maybe_smooth(y: np.ndarray, x: np.ndarray, k: int):
        """Return (y_s, x_s). Only smooth if k>1 and len(y) >= k; else return as-is."""
        if k is None or k <= 1 or len(y) < k:
            return y, x
        y_s = _moving_average(y, k)
        x_s = x[k - 1 :]
        return y_s, x_s

    def _plot_series(ax, x: np.ndarray, y: np.ndarray, label: str):
        """Use a line for >=2 points; a visible dot for a single point."""
        n = len(y)
        if n >= 2:
            ax.plot(x, y, label=label)
        elif n == 1:
            ax.scatter(x, y, label=label)

    k = int(smooth) if smooth is not None else 0
    tr_mse_s, steps_tr_mse = maybe_smooth(tr_mse, steps_tr, k)
    tr_mae_s, steps_tr_mae = maybe_smooth(tr_mae, steps_tr, k)
    tr_nll_s, steps_tr_nll_s = maybe_smooth(tr_nll, steps_tr_nll, k)

    # MSE plot
    try:
        fig, ax = plt.subplots()
        _plot_series(ax, steps_tr_mse, tr_mse_s, "train MSE")
        _plot_series(ax, steps_va, va_mse, "val MSE")
        ax.set_xlabel("step"); ax.set_ylabel("MSE"); ax.legend()
        fig.savefig(outdir / "curves_mse.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[warn] failed to save curves_mse.png: {e}")

    # MAE plot
    try:
        fig, ax = plt.subplots()
        _plot_series(ax, steps_tr_mae, tr_mae_s, "train MAE")
        _plot_series(ax, steps_va, va_mae, "val MAE")
        ax.set_xlabel("step"); ax.set_ylabel("MAE"); ax.legend()
        fig.savefig(outdir / "curves_mae.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[warn] failed to save curves_mae.png: {e}")

    # Gaussian NLL plot
    if len(tr_nll) > 0 or len(va_nll) > 0:
        try:
            fig, ax = plt.subplots()
            _plot_series(ax, steps_tr_nll_s, tr_nll_s, "train Gaussian NLL")
            _plot_series(ax, steps_va_nll, va_nll, "val Gaussian NLL")
            ax.set_xlabel("step"); ax.set_ylabel("Gaussian NLL (nats)"); ax.legend()
            fig.savefig(outdir / "curves_gauss_nll.png", bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[warn] failed to save curves_gauss_nll.png: {e}")


# -----------------------------
# Main training entry
# -----------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train a permutation‑invariant scRNA transformer (MVP)")
    # Data
    p.add_argument("--data", required=True, help="Path to processed .h5ad file")
    p.add_argument("--train-only", action="store_true", help="Use only obs['split']==train for training if present")
    p.add_argument("--val-split", default="val", help="Which split to use for validation")
    p.add_argument("--limit-train", type=int, default=None, help="If set, cap number of training cells to this value")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic random data instead of loading the .h5ad (use --synthetic-cells/genes).")
    p.add_argument("--synthetic-cells", type=int, default=None,
                   help="Number of synthetic training cells (default 10000 if --synthetic).")
    p.add_argument("--synthetic-genes", type=int, default=None,
                   help="Number of genes for synthetic data (default from --cap-genes or 1024).")
    p.add_argument("--mask-rate", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    # Model
    p.add_argument("--size", default="XS", choices=["XXS", "TINY", "XS", "S", "M", "L", "XL"], help="Named size preset")
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--ff-mult", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--pool", default="mean", choices=["mean", "cls"])
    p.add_argument("--head-type", default="linear", choices=["linear", "mlp"])
    # Optim & schedule
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Logging / control
    p.add_argument("--log-every", type=int, default=100, help="Print training status every N steps")
    p.add_argument("--eta-window", type=int, default=50, help="Window (steps) for average step time/ETA")
    p.add_argument("--max-minutes", type=float, default=None, help="If set, stop training after this many minutes")
    p.add_argument("--no-plots", action="store_true", help="Disable PNG plots even if history is recorded")
    p.add_argument("--plot-smooth", type=int, default=0, help="Moving average window for train curves (0=off)")
    p.add_argument("--train-metrics-every", type=int, default=None,
                   help="Record train metrics to history every N steps (defaults to --log-every)")
    p.add_argument("--accum", type=int, default=1,
                   help="Accumulate gradients over N steps before optimizer step")
    p.add_argument("--cap-genes", type=int, default=None,
                   help="If set, slice to first K genes after loading the .h5ad")
    p.add_argument("--amp", action="store_true",
                   help="Use autocast(float16) on mps/cuda for forward+loss")
    p.add_argument("--save-last-every", type=int, default=1000,
                   help="Save ckpt_last.pt every N steps (0 disables periodic last-checkpoints)")
    # Misc
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default=None)
    p.add_argument("--outdir", default="./runs/tmp")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint (.pt) or run dir containing ckpt_best.pt")

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"device: {device}")

    # ---------------- Data ----------------
    if args.synthetic:
        n_cells = int(args.synthetic_cells) if args.synthetic_cells else 10_000
        n_genes = int(args.synthetic_genes) if args.synthetic_genes else int(args.cap_genes or 1024)
        rng = np.random.default_rng(args.seed)
        X_tr = rng.standard_normal((n_cells, n_genes), dtype=np.float32)
        n_val = max(1, n_cells // 10)
        X_va = rng.standard_normal((n_val, n_genes), dtype=np.float32)
        vocab_size = n_genes
        ad_tr_n = n_cells
        print(f"[synthetic] train cells={n_cells} genes={n_genes}; val cells={n_val}")
    else:
        print(f"loading {args.data} …")
        adata = sc.read_h5ad(args.data)
        if args.cap_genes is not None and args.cap_genes < adata.n_vars:
            adata = adata[:, : int(args.cap_genes)].copy()
            print(f"[cap-genes] using first {adata.n_vars} genes")

        def _slice(split_name: Optional[str]):
            if split_name is None:
                return adata
            if "split" in adata.obs:
                return adata[adata.obs["split"].astype(str) == split_name]
            return adata

        ad_tr = _slice("train" if args.train_only else None)
        if args.limit_train is not None:
            n_lim = min(int(args.limit_train), ad_tr.n_obs)
            ad_tr = ad_tr[: n_lim]
        ad_va = _slice(args.val_split)

        X_tr = ad_tr.X
        X_va = ad_va.X if ad_va.n_obs > 0 else ad_tr.X[: min(5000, ad_tr.n_obs)]

        vocab_size = adata.n_vars
        ad_tr_n = ad_tr.n_obs

    ds_tr = MaskedExpressionDataset(X_tr, mask_rate=args.mask_rate, seed=args.seed)
    ds_va = MaskedExpressionDataset(X_va, mask_rate=args.mask_rate, seed=args.seed + 1)

    collate = CollateWithGeneIds(vocab_size)

    tl = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device == "cuda"),
    )
    vl = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        collate_fn=collate,
    )

    # ---------------- Model ----------------
    overrides = {}
    if args.d_model is not None:
        overrides["d_model"] = args.d_model
    if args.n_layers is not None:
        overrides["n_layers"] = args.n_layers
    if args.n_heads is not None:
        overrides["n_heads"] = args.n_heads
    if args.ff_mult is not None:
        overrides["ff_mult"] = args.ff_mult
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    overrides["pool"] = args.pool
    overrides["head_type"] = args.head_type

    model = build_model(vocab_size=vocab_size, size=args.size, overrides=overrides)
    model.to(device)
    print(f"model params: {count_parameters(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------------- Train ----------------
    hist_path = outdir / "history.jsonl"
    history: List[Dict[str, Any]] = load_history(hist_path) if args.resume else []
    if not args.resume:
        # Start a fresh history file for a fresh run in this outdir.
        hist_path.write_text("")
    flushed_rows = len(history)

    def flush_history() -> None:
        nonlocal flushed_rows
        if flushed_rows >= len(history):
            return
        mode = "a" if flushed_rows > 0 else "w"
        with open(hist_path, mode) as f:
            for row in history[flushed_rows:]:
                print(json.dumps(row), file=f)
        flushed_rows = len(history)

    best_val = best_val_from_history(history)
    if best_val is None:
        best_val = math.inf
    step = 0

    steps_per_epoch = max(1, math.ceil(ad_tr_n / args.batch_size))
    print(f"train cells: {ad_tr_n:,} | batch {args.batch_size} | steps/epoch ~ {steps_per_epoch}")

    t0 = time.perf_counter()
    last = t0
    times = deque(maxlen=max(1, args.eta_window))

    # --- 3C: accumulation + AMP + record cadence setup ---
    rec_every = args.train_metrics_every or args.log_every
    use_amp = args.amp and (device in ("mps", "cuda"))
    amp_ctx = torch.autocast(device_type=device, dtype=torch.float16) if use_amp else nullcontext()

    accum = max(1, int(args.accum))

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            candidates = [
                resume_path / "ckpt_last.pt",
                resume_path / "ckpt_best.pt",
                resume_path / "ckpt_final.pt",
            ]
            existing = [p for p in candidates if p.exists()]
            if existing:
                resume_path = max(existing, key=lambda p: p.stat().st_mtime)
        if not resume_path.exists():
            raise SystemExit(f"--resume not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        warn_keys = [
            ("size", args.size),
            ("batch_size", args.batch_size),
            ("lr", args.lr),
            ("weight_decay", args.weight_decay),
            ("accum", accum),
        ]
        for key, cur in warn_keys:
            old = cfg.get(key)
            if old is not None and old != cur:
                print(f"[resume] warning: {key} ckpt={old} args={cur}")
        model.load_state_dict(ckpt["model_state"])
        if "optim_state" in ckpt:
            opt.load_state_dict(ckpt["optim_state"])
        step = int(ckpt.get("step", 0))
        print(f"[resume] loaded {resume_path} (step {step})")
        if best_val == math.inf:
            metrics = evaluate(model, vl, device)
            best_val = float(metrics["val_mse"])
            print(f"[resume] init best_val {best_val:.4f}")

    opt.zero_grad(set_to_none=True)

    # -----------------------------------------------------

    def save_ckpt(tag: str) -> None:
        ck = {
            "model_state": model.state_dict(),
            "optim_state": opt.state_dict(),
            "step": step,
            "config": {
                "vocab_size": vocab_size,
                "size": args.size,
                "overrides": overrides,
                "mask_rate": args.mask_rate,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "accum": accum,  # <--- add this
            },
        }
        path = outdir / f"ckpt_{tag}.pt"
        torch.save(ck, path)
        print(f"saved checkpoint: {path}")

    stop_requested = False
    stop_reason = None

    def _request_stop(reason: str) -> None:
        nonlocal stop_requested, stop_reason
        stop_requested = True
        stop_reason = reason

    def _signal_handler(signum, _frame):
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = f"signal_{signum}"
        print(f"[stop] received {sig_name}; stopping after current step")
        _request_stop(sig_name)

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _signal_handler)
            except Exception:
                pass

    model.train()
    try:
        while step < args.steps:
            for batch in tl:
                step += 1
                x_in = batch["x_in"].to(device)
                mask = batch["mask"].to(device)
                tgt = batch["x_tgt"].to(device)
                gene_ids = batch["gene_ids"].to(device)

                # --- 3C: forward (optional AMP) + scaled backward for accumulation ---
                with amp_ctx:
                    pred, _ = model(gene_ids, x_in, mask)
                    loss = masked_mse(pred, tgt, mask)
                    train_mae = masked_mae(pred, tgt, mask)
                    train_nll = masked_gaussian_nll(pred, tgt, mask)

                # scale loss for gradient accumulation
                (loss / accum).backward()

                # --- optimizer step only every `accum` steps ---
                if (step % accum) == 0:
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                # timing / ETA accounting
                now = time.perf_counter()
                times.append(now - last)
                last = now

                # NEW (record cadence may differ from print cadence)
                if (step % rec_every == 0) or (step == 1):
                    history.append({
                        "step": step,
                        "train_mse": float(loss.item()),
                        "train_mae": float(train_mae.item()),
                        "train_gauss_nll": float(train_nll.item()),
                    })
                    flush_history()
                    avg = sum(times) / len(times) if len(times) else float('nan')
                    remain = max(0, args.steps - step)
                    eta = remain * avg if avg == avg else float('nan')
                    pct = 100.0 * step / max(1, args.steps)
                    epoch = 1 + (step - 1) // steps_per_epoch
                    pos = 1 + (step - 1) % steps_per_epoch
                    print(
                        f"step {step:6d}/{args.steps} ({pct:5.1f}%)  "
                        f"epoch {epoch} [{pos}/{steps_per_epoch}]  "
                        f"loss {loss.item():.4f}  avg_step {avg:.3f}s  ETA ~{fmt_hms(eta)}"
                    )

                if args.save_last_every and args.save_last_every > 0 and (step % args.save_last_every == 0):
                    save_ckpt("last")

                # optional time budget
                if args.max_minutes is not None:
                    if (now - t0) > args.max_minutes * 60.0:
                        print("time budget reached; stopping early")
                        _request_stop("max_minutes")

                if step % args.val_every == 0:
                    metrics = evaluate(model, vl, device)
                    val_mse = metrics["val_mse"]
                    print(
                        f"VAL  step {step:6d}  mse {val_mse:.4f}  "
                        f"mae {metrics['val_mae']:.4f}  nll {metrics['val_gauss_nll']:.4f}"
                    )
                    history.append({
                        "step": step,
                        "val_mse": float(metrics["val_mse"]),
                        "val_mae": float(metrics["val_mae"]),
                        "val_gauss_nll": float(metrics["val_gauss_nll"]),
                        "train_loss": float(loss.item()),
                    })
                    flush_history()
                    if val_mse < best_val:
                        best_val = val_mse
                        save_ckpt("best")

                if stop_requested or step >= args.steps:
                    break
            if stop_requested or step >= args.steps:
                break
    except KeyboardInterrupt:
        print("[stop] KeyboardInterrupt received; stopping after checkpoint save")
        _request_stop("KeyboardInterrupt")

    if stop_requested and step < args.steps:
        save_ckpt("last")
        flush_history()
        total = time.perf_counter() - t0
        print(f"wrote logs: {hist_path} ({len(history)} entries)")
        print(f"total elapsed: {fmt_hms(total)}")
        if not args.no_plots:
            save_plots(history, outdir, smooth=int(args.plot_smooth))
        print(f"stopped early at step {step}/{args.steps} (reason: {stop_reason})")
        return

    # Final validation (always run once at the end)
    metrics = evaluate(model, vl, device)
    val_mse = metrics["val_mse"]
    print(
        f"VAL  final      mse {val_mse:.4f}  "
        f"mae {metrics['val_mae']:.4f}  nll {metrics['val_gauss_nll']:.4f}"
    )
    history.append({"step": step, **{k: float(v) for k, v in metrics.items()}, "train_loss": None})
    if val_mse < best_val:
        best_val = val_mse
        save_ckpt("best")

    save_ckpt("final")
    save_ckpt("last")

    # Save history
    flush_history()
    total = time.perf_counter() - t0
    print(f"wrote logs: {hist_path} ({len(history)} entries)")
    print(f"total elapsed: {fmt_hms(total)}")

    # Save plots (optional)
    if not args.no_plots:
        save_plots(history, outdir, smooth=int(args.plot_smooth))


def load_history(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def best_val_from_history(history: List[Dict[str, Any]]) -> Optional[float]:
    best = None
    for row in history:
        if "val_mse" not in row:
            continue
        try:
            v = float(row["val_mse"])
        except Exception:
            continue
        if best is None or v < best:
            best = v
    return best


if __name__ == "__main__":
    main()
