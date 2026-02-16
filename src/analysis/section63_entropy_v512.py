#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _load_history(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _as_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _max_step(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return -1
    return max(_as_int(r.get("step"), -1) for r in rows)


def _best_row(rows: List[Dict[str, Any]], metric: str) -> Optional[Dict[str, Any]]:
    best = None
    best_v = float("inf")
    for r in rows:
        if metric not in r:
            continue
        v = _as_float(r.get(metric))
        if not np.isfinite(v):
            continue
        if v < best_v:
            best_v = v
            best = r
    return best


def _fallback_nll_from_mse(mse: float) -> float:
    if not np.isfinite(mse) or mse <= 0:
        return float("nan")
    return 0.5 * (1.0 + math.log(2.0 * math.pi * mse))


def _count_params_from_ckpt(path: Path) -> int:
    if not path.exists():
        return -1
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("model_state") or ck.get("state_dict") or {}
    total = 0
    for t in sd.values():
        try:
            total += int(np.prod(t.shape))
        except Exception:
            try:
                total += int(t.numel())
            except Exception:
                continue
    return total


def _pick_ckpt(run_dir: Path) -> Optional[Path]:
    for ck in ("ckpt_best.pt", "ckpt_final.pt", "ckpt_last.pt"):
        p = run_dir / ck
        if p.exists():
            return p
    return None


def fit_power_with_offset(x: np.ndarray, y: np.ndarray, n_cand: int = 300) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        raise ValueError("need >=3 points")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("x and y must be positive")

    c_max = 0.99 * float(np.min(y))
    c_grid = np.linspace(0.0, max(0.0, c_max), n_cand)
    if len(c_grid) == 0:
        c_grid = np.array([0.0])

    lx = np.log(x)
    best = {"r2": -np.inf}
    for c in c_grid:
        y_adj = y - c
        if np.any(y_adj <= 0):
            continue
        ly = np.log(y_adj)
        A = np.vstack([np.ones_like(lx), -lx]).T
        coeff, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
        loga, alpha = coeff[0], coeff[1]
        yhat = A @ coeff
        ss_res = float(np.sum((ly - yhat) ** 2))
        ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if r2 > best["r2"]:
            best = {"a": float(np.exp(loga)), "alpha": float(alpha), "c": float(c), "r2": float(r2)}
    if best["r2"] == -np.inf:
        raise ValueError("fit failed")
    return best


def fit_power_no_offset(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        raise ValueError("need >=3 points")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("x and y must be positive")
    lx = np.log(x)
    ly = np.log(y)
    A = np.vstack([np.ones_like(lx), lx]).T
    coeff, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    logk, gamma = coeff[0], coeff[1]
    yhat = A @ coeff
    ss_res = float(np.sum((ly - yhat) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
    return {"k": float(np.exp(logk)), "gamma": float(gamma), "r2": float(r2)}


def build_records(manifest: pd.DataFrame) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        run_dir = Path(str(row["outdir"]))
        hist_rows = _load_history(run_dir / "history.jsonl")
        if not hist_rows:
            continue
        max_step = _max_step(hist_rows)
        target_steps = _as_int(row.get("steps"), -1)
        is_complete = (target_steps > 0) and (max_step >= target_steps)

        best_mse = _best_row(hist_rows, "val_mse")
        best_nll = _best_row(hist_rows, "val_gauss_nll")

        best_val_mse = _as_float(best_mse.get("val_mse")) if best_mse else float("nan")
        best_val_nll = _as_float(best_nll.get("val_gauss_nll")) if best_nll else float("nan")
        has_logged_nll = best_nll is not None
        nll_source = "history"
        if not np.isfinite(best_val_nll):
            best_val_nll = _fallback_nll_from_mse(best_val_mse)
            nll_source = "derived_from_best_mse"

        ckpt_path = _pick_ckpt(run_dir)
        params = _count_params_from_ckpt(ckpt_path) if ckpt_path else -1
        V = 512
        eff_batch = _as_int(row.get("eff_batch"), 32)
        tokens_seen = int(max_step * eff_batch * V) if max_step > 0 else -1
        compute_proxy = float(params * tokens_seen) if (params > 0 and tokens_seen > 0) else float("nan")

        recs.append(
            {
                "job_id": row["job_id"],
                "phase": row["phase"],
                "data_cells": _as_int(row["data_cells"], -1),
                "size": row["size"],
                "seed": _as_int(row["seed"], -1),
                "steps_target": target_steps,
                "steps_completed": max_step,
                "is_complete": bool(is_complete),
                "params": params,
                "eff_batch": eff_batch,
                "tokens_seen": tokens_seen,
                "compute_proxy": compute_proxy,
                "best_val_mse": best_val_mse,
                "best_val_gauss_nll": best_val_nll,
                "has_logged_gauss_nll": bool(has_logged_nll),
                "gauss_nll_source": nll_source,
                "run_dir": str(run_dir),
            }
        )
    return pd.DataFrame(recs)


def frontier_from_runs(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["compute_proxy", metric]).copy()
    sub = sub[(sub["compute_proxy"] > 0) & (sub[metric] > 0)].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    grouped = (
        sub.sort_values([metric, "compute_proxy"])
        .groupby("compute_proxy", as_index=False)
        .first()
        .sort_values("compute_proxy")
    )
    keep_rows = []
    best_so_far = float("inf")
    for _, r in grouped.iterrows():
        y = float(r[metric])
        if y < best_so_far:
            best_so_far = y
            keep_rows.append(r.to_dict())
    return pd.DataFrame(keep_rows)


def fit_intersection(
    l_c_fit: Dict[str, float],
    l_d_fit: Dict[str, float],
    d_c_fit: Dict[str, float],
    c_min: float,
    c_max: float,
) -> Dict[str, float]:
    a_c, alpha_c, c_c = l_c_fit["a"], l_c_fit["alpha"], l_c_fit["c"]
    a_d, beta_d, c_d = l_d_fit["a"], l_d_fit["alpha"], l_d_fit["c"]
    k_d, gamma_d = d_c_fit["k"], d_c_fit["gamma"]

    def l_c(c: np.ndarray) -> np.ndarray:
        return a_c * c ** (-alpha_c) + c_c

    def l_d_of_c(c: np.ndarray) -> np.ndarray:
        d_star = k_d * c ** gamma_d
        return a_d * d_star ** (-beta_d) + c_d

    grid = np.logspace(np.log10(c_min), np.log10(c_max), 4000)
    diff = l_c(grid) - l_d_of_c(grid)
    sign = np.sign(diff)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) > 0:
        i = int(idx[0])
        c_star = float(grid[i])
        exact = 1.0
    else:
        i = int(np.argmin(np.abs(diff)))
        c_star = float(grid[i])
        exact = 0.0

    l_star = float(l_c(np.array([c_star]))[0])
    return {"C_star": c_star, "L_star": l_star, "exact_sign_change": exact}


def bits_from_loss(metric: str, loss: float) -> float:
    if not np.isfinite(loss):
        return float("nan")
    if metric == "best_val_gauss_nll":
        return float(loss / math.log(2.0))
    if metric == "best_val_mse":
        if loss <= 0:
            return float("nan")
        return float(0.5 * math.log2(2.0 * math.pi * math.e * loss))
    return float("nan")


def plot_fit(x: np.ndarray, y: np.ndarray, fit: Dict[str, float], title: str, xlabel: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.scatter(x, y, s=40, alpha=0.85)
    xline = np.logspace(np.log10(float(np.min(x)) * 0.9), np.log10(float(np.max(x)) * 1.1), 300)
    yline = fit["a"] * xline ** (-fit["alpha"]) + fit["c"]
    ax.plot(xline, yline, linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_frontier(frontier: pd.DataFrame, metric: str, l_c_fit: Dict[str, float], l_d_of_c_fit: Optional[Dict[str, float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    x = frontier["compute_proxy"].to_numpy(float)
    y = frontier[metric].to_numpy(float)
    ax.scatter(x, y, s=44, alpha=0.85, label="empirical frontier")

    xline = np.logspace(np.log10(float(np.min(x)) * 0.9), np.log10(float(np.max(x)) * 1.3), 300)
    yline = l_c_fit["a"] * xline ** (-l_c_fit["alpha"]) + l_c_fit["c"]
    ax.plot(xline, yline, linewidth=2.0, label="L(C_min) fit")

    if l_d_of_c_fit is not None:
        a_d, beta_d, c_d = l_d_of_c_fit["a_d"], l_d_of_c_fit["beta_d"], l_d_of_c_fit["c_d"]
        k, gamma = l_d_of_c_fit["k"], l_d_of_c_fit["gamma"]
        dline = k * xline ** gamma
        y_d = a_d * dline ** (-beta_d) + c_d
        ax.plot(xline, y_d, linewidth=2.0, label="L(D(C_min))")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute proxy C = params * tokens_seen")
    ax.set_ylabel(metric)
    ax.legend()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Section 6.3-style entropy analysis for V=512 runs.")
    ap.add_argument("--manifest", default="analysis/entropy_v512/job_manifest.csv")
    ap.add_argument("--out", default="analysis/entropy_v512/section63")
    ap.add_argument("--include-partial", action="store_true")
    ap.add_argument("--data-model", default="M", choices=["M", "L"])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    out_root = Path(args.out)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"run_{stamp}{('_' + args.tag) if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)

    runs_df = build_records(manifest)
    if len(runs_df) == 0:
        raise SystemExit("no runs with history found")
    runs_df.to_csv(out_dir / "stage0_runs.csv", index=False)

    fit_df = runs_df.copy()
    if not args.include_partial:
        fit_df = fit_df[fit_df["is_complete"]].copy()
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan)

    status: Dict[str, Any] = {"n_runs_total": int(len(runs_df)), "n_runs_fit": int(len(fit_df)), "metrics": {}}

    for metric in ["best_val_gauss_nll", "best_val_mse"]:
        mstatus: Dict[str, Any] = {"metric": metric}

        fixed = fit_df[(fit_df["phase"] == "fixed") & (fit_df[metric] > 0) & (fit_df["params"] > 0)].copy()
        if len(fixed) >= 3:
            f_fixed = fit_power_with_offset(fixed["params"].to_numpy(float), fixed[metric].to_numpy(float))
            mstatus["fixed_fit"] = f_fixed
            mstatus["fixed_floor_bits"] = bits_from_loss(metric, f_fixed["c"])
            plot_fit(
                fixed["params"].to_numpy(float),
                fixed[metric].to_numpy(float),
                f_fixed,
                title=f"Fixed-regime fit ({metric})",
                xlabel="Parameters P",
                ylabel=metric,
                out=plot_dir / f"fixed_fit_{metric}.png",
            )
        else:
            mstatus["fixed_fit"] = None

        data_raw = fit_df[(fit_df["phase"] == "data") & (fit_df["size"] == args.data_model) & (fit_df[metric] > 0)].copy()
        data_agg = pd.DataFrame()
        if len(data_raw) > 0:
            data_agg = (
                data_raw.groupby("data_cells", as_index=False)
                .agg(loss=(metric, "mean"), n_runs=("job_id", "count"))
                .sort_values("data_cells")
            )
            data_agg.to_csv(out_dir / f"stage1_data_agg_{metric}.csv", index=False)
        if len(data_agg) >= 3:
            f_data = fit_power_with_offset(data_agg["data_cells"].to_numpy(float), data_agg["loss"].to_numpy(float))
            mstatus["data_fit"] = f_data
            plot_fit(
                data_agg["data_cells"].to_numpy(float),
                data_agg["loss"].to_numpy(float),
                f_data,
                title=f"Data-scaling fit ({metric})",
                xlabel="Dataset size D (cells)",
                ylabel=metric,
                out=plot_dir / f"data_fit_{metric}.png",
            )
        else:
            mstatus["data_fit"] = None

        comp = fit_df[
            (fit_df["phase"].isin(["fixed", "compute"])) & (fit_df["data_cells"] == 200000) & (fit_df[metric] > 0)
        ].copy()
        frontier = frontier_from_runs(comp, metric)
        frontier.to_csv(out_dir / f"stage2_frontier_points_{metric}.csv", index=False)
        if len(frontier) >= 3:
            f_frontier = fit_power_with_offset(frontier["compute_proxy"].to_numpy(float), frontier[metric].to_numpy(float))
            mstatus["compute_fit"] = f_frontier
        else:
            mstatus["compute_fit"] = None

        dmap = fit_df[
            (fit_df["phase"].isin(["data", "data_compute"]))
            & (fit_df["size"] == args.data_model)
            & (fit_df[metric] > 0)
            & (fit_df["compute_proxy"] > 0)
            & (fit_df["data_cells"] > 0)
        ].copy()
        if len(dmap) > 0:
            dstar = (
                dmap.sort_values([metric, "compute_proxy"])
                .groupby("compute_proxy", as_index=False)
                .first()[["compute_proxy", "data_cells", metric]]
                .sort_values("compute_proxy")
            )
        else:
            dstar = pd.DataFrame()
        dstar.to_csv(out_dir / f"stage3_dstar_points_{metric}.csv", index=False)
        if len(dstar) >= 3 and dstar["data_cells"].nunique() >= 2:
            f_d_of_c = fit_power_no_offset(dstar["compute_proxy"].to_numpy(float), dstar["data_cells"].to_numpy(float))
            mstatus["d_of_c_fit"] = f_d_of_c
        else:
            mstatus["d_of_c_fit"] = None

        l_d_of_c_args = None
        if mstatus["data_fit"] and mstatus["d_of_c_fit"]:
            l_d_of_c_args = {
                "a_d": mstatus["data_fit"]["a"],
                "beta_d": mstatus["data_fit"]["alpha"],
                "c_d": mstatus["data_fit"]["c"],
                "k": mstatus["d_of_c_fit"]["k"],
                "gamma": mstatus["d_of_c_fit"]["gamma"],
            }

        if mstatus["compute_fit"] and l_d_of_c_args and len(frontier) >= 3:
            c_min = float(min(frontier["compute_proxy"].min(), dstar["compute_proxy"].min())) * 0.5
            c_max = float(max(frontier["compute_proxy"].max(), dstar["compute_proxy"].max())) * 20.0
            inter = fit_intersection(mstatus["compute_fit"], mstatus["data_fit"], mstatus["d_of_c_fit"], c_min, c_max)
            inter["H_bits"] = bits_from_loss(metric, inter["L_star"])
            mstatus["intersection"] = inter
        else:
            mstatus["intersection"] = None

        if mstatus["compute_fit"] and len(frontier) >= 3:
            plot_frontier(
                frontier,
                metric,
                mstatus["compute_fit"],
                l_d_of_c_args,
                out=plot_dir / f"compute_frontier_{metric}.png",
            )

        status["metrics"][metric] = mstatus

    with open(out_dir / "section63_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    lines: List[str] = []
    lines.append("# Section 6.3-style Entropy Analysis (V=512)")
    lines.append("")
    lines.append(f"- Runs seen: {status['n_runs_total']}")
    lines.append(f"- Runs used for fit: {status['n_runs_fit']}")
    lines.append("")
    for metric in ["best_val_gauss_nll", "best_val_mse"]:
        m = status["metrics"][metric]
        lines.append(f"## {metric}")
        ff = m.get("fixed_fit")
        if ff:
            lines.append(
                f"- Fixed-regime floor c: {ff['c']:.6g}; floor bits: {m['fixed_floor_bits']:.4f}; "
                f"alpha={ff['alpha']:.4f}; R2={ff['r2']:.4f}"
            )
        else:
            lines.append("- Fixed-regime fit: insufficient data.")
        df = m.get("data_fit")
        if df:
            lines.append(f"- Data law L(D): alpha={df['alpha']:.4f}, c_D={df['c']:.6g}, R2={df['r2']:.4f}")
        else:
            lines.append("- Data law L(D): insufficient data.")
        cf = m.get("compute_fit")
        if cf:
            lines.append(f"- Compute frontier L(C_min): alpha={cf['alpha']:.4f}, c_C={cf['c']:.6g}, R2={cf['r2']:.4f}")
        else:
            lines.append("- Compute frontier L(C_min): insufficient data.")
        dc = m.get("d_of_c_fit")
        if dc:
            lines.append(f"- D(C_min): gamma={dc['gamma']:.4f}, k={dc['k']:.6g}, R2={dc['r2']:.4f}")
        else:
            lines.append("- D(C_min): insufficient data.")
        inter = m.get("intersection")
        if inter:
            lines.append(
                f"- Intersection: C*={inter['C_star']:.4g}, L*={inter['L_star']:.6g}, "
                f"H_bits={inter['H_bits']:.4f}, exact_sign_change={bool(inter['exact_sign_change'])}"
            )
        else:
            lines.append("- Intersection: not available yet.")
        lines.append("")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    latest = out_root / "latest_section63.txt"
    latest.write_text(str(out_dir), encoding="utf-8")
    print(f"wrote {out_dir}")
    print(f"wrote {latest}")


if __name__ == "__main__":
    main()
