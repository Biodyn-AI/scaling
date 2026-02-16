#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def fit_power_with_offset(x: np.ndarray, y: np.ndarray, n_cand: int = 200) -> Dict[str, float]:
    """
    Fit y ~= a * x^(-alpha) + c using a grid over c and linear least squares on log(y-c).
    Returns alpha, a, c, r2 (computed on log(y-c)).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("x and y must be positive for log-log fitting.")

    c_min = 0.0
    c_max = 0.99 * float(np.min(y))
    cs = np.linspace(c_min, c_max, n_cand)

    lx = np.log(x)
    best = {"r2": -np.inf}

    for c in cs:
        y_adj = y - c
        if np.any(y_adj <= 0):
            continue
        ly = np.log(y_adj)
        # ly = log(a) - alpha*log(x)
        A = np.vstack([np.ones_like(lx), -lx]).T
        coeff, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
        loga, alpha = coeff[0], coeff[1]
        yhat = A @ coeff
        ss_res = float(np.sum((ly - yhat) ** 2))
        ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if r2 > best["r2"]:
            best = {
                "alpha": float(alpha),
                "a": float(np.exp(loga)),
                "c": float(c),
                "r2": float(r2),
            }
    return best


def load_rows(table_path: Path) -> List[Dict[str, str]]:
    with open(table_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def parse_xy(rows: List[Dict[str, str]], v_value: str) -> Tuple[np.ndarray, np.ndarray]:
    x: List[float] = []
    y: List[float] = []
    for r in rows:
        if str(r.get("V", "")) != v_value:
            continue
        try:
            p = float(r["params"])
            l = float(r["val_mse"])
        except Exception:
            continue
        if p > 0 and l > 0:
            x.append(p)
            y.append(l)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def plot_regime(
    x: np.ndarray,
    y: np.ndarray,
    fit: Dict[str, float],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(x, y, s=36, alpha=0.85, label=f"runs (n={len(x)})")
    xline = np.logspace(np.log10(x.min() * 0.9), np.log10(x.max() * 1.1), 300)
    yline = fit["a"] * xline ** (-fit["alpha"]) + fit["c"]
    ax.plot(
        xline,
        yline,
        linewidth=2.0,
        label=(
            f"fit: L=a*P^-alpha+c\n"
            f"alpha={fit['alpha']:.3f}, c={fit['c']:.4g}, R^2={fit['r2']:.3f}"
        ),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters P")
    ax.set_ylabel("Validation MSE")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit two separate scaling regressions by regime (V=512 and V=1024).")
    ap.add_argument("--table", default="analysis/all_runs_latest/scaling_table.csv", help="Input scaling table CSV.")
    ap.add_argument("--out", default="analysis/all_runs_latest", help="Output directory for plots and fit summary.")
    args = ap.parse_args()

    table_path = Path(args.table)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(table_path)
    regimes = [
        ("512", "Regime A: V=512"),
        ("1024", "Regime B: V=1024"),
    ]

    summary: Dict[str, Dict[str, float]] = {}
    for v_value, title in regimes:
        x, y = parse_xy(rows, v_value)
        if len(x) < 3:
            print(f"[skip] V={v_value}: not enough points ({len(x)})")
            continue
        fit = fit_power_with_offset(x, y)
        fit["n_points"] = int(len(x))
        summary[f"V_{v_value}"] = fit
        out_path = out_dir / f"scaling_loss_vs_params_V{v_value}.png"
        plot_regime(x, y, fit, title, out_path)
        print(
            f"[fit V={v_value}] n={len(x)} alpha={fit['alpha']:.3f} "
            f"a={fit['a']:.4g} c={fit['c']:.4g} R^2={fit['r2']:.4f}"
        )
        print(f"wrote {out_path}")

    summary_path = out_dir / "scaling_two_regimes_fit.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
