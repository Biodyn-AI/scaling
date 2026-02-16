#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


SIZE_ORDER = ["XXS", "TINY", "XS", "S", "M", "L", "XL"]
RUN_NAME_RE = re.compile(
    r"^(?P<size>[A-Z]+)_V(?P<V>\d+)_B(?P<micro>\d+)x(?P<accum>\d+)_S(?P<steps>\d+)_seed(?P<seed>\d+)$"
)


def _as_int(v: Any, default: int = -1) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


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


def _find_run_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for hist in root.rglob("history.jsonl"):
        run_dir = hist.parent
        if any((run_dir / ck).exists() for ck in ("ckpt_best.pt", "ckpt_final.pt", "ckpt_last.pt")):
            out.append(run_dir)
    return sorted(set(out))


def _parse_run_name(name: str) -> Dict[str, int | str]:
    m = RUN_NAME_RE.match(name)
    if not m:
        return {}
    d = m.groupdict()
    return {
        "size": d["size"],
        "V": int(d["V"]),
        "micro_batch": int(d["micro"]),
        "accum": int(d["accum"]),
        "steps": int(d["steps"]),
        "seed": int(d["seed"]),
    }


def _ckpt_path_for_params(run_dir: Path) -> Optional[Path]:
    for ck in ("ckpt_best.pt", "ckpt_final.pt", "ckpt_last.pt"):
        p = run_dir / ck
        if p.exists():
            return p
    return None


def _count_params(sd: Dict[str, Any]) -> int:
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


def _max_step(rows: Sequence[Dict[str, Any]]) -> int:
    steps = [_as_int(r.get("step"), -1) for r in rows]
    return max(steps) if steps else -1


def _best_row(rows: Sequence[Dict[str, Any]], metric: str) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
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


def _last_val_row(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for r in reversed(rows):
        if any(k in r for k in ("val_mse", "val_mae", "val_gauss_nll")):
            return r
    return None


def _fallback_gauss_nll_from_mse(mse: float) -> float:
    if mse <= 0 or (not np.isfinite(mse)):
        return float("nan")
    return 0.5 * (1.0 + math.log(2.0 * math.pi * mse))


def _infer_v_from_data_path(data_path: str) -> int:
    m = re.search(r"V(\d+)", str(data_path))
    return int(m.group(1)) if m else -1


def _size_sort_key(size: Any) -> int:
    s = str(size)
    if s in SIZE_ORDER:
        return SIZE_ORDER.index(s)
    return len(SIZE_ORDER) + 1


def collect_run_record(run_dir: Path, *, v_filter: Optional[Sequence[int]] = None) -> Optional[Dict[str, Any]]:
    rows = _load_history(run_dir / "history.jsonl")
    if not rows:
        return None

    run_meta = _load_json(run_dir / "run_meta.json")
    sweep_meta = _load_json(run_dir.parent / "sweep_meta.json")
    parsed = _parse_run_name(run_dir.name)
    v_filter_set = set(v_filter) if v_filter is not None else None

    if v_filter_set is not None:
        quick_v = _as_int(parsed.get("V"), -1)
        if quick_v <= 0:
            quick_v = _infer_v_from_data_path(str(run_meta.get("data") or sweep_meta.get("data") or ""))
        if quick_v > 0 and quick_v not in v_filter_set:
            return None

    ckpt_path = _ckpt_path_for_params(run_dir)
    if ckpt_path is None:
        return None

    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck.get("config", {})
    state = ck.get("model_state") or ck.get("state_dict") or {}
    params = _count_params(state)

    data_path = str(run_meta.get("data") or sweep_meta.get("data") or "")
    vocab_size = _as_int(cfg.get("vocab_size"), -1)
    if vocab_size <= 0:
        vocab_size = _as_int(parsed.get("V"), -1)
    if vocab_size <= 0 and data_path:
        vocab_size = _infer_v_from_data_path(data_path)
    if v_filter_set is not None and vocab_size > 0 and vocab_size not in v_filter_set:
        return None

    size = str(cfg.get("size") or run_meta.get("size") or parsed.get("size") or "")
    seed = _as_int(cfg.get("seed"), -1)
    if seed < 0:
        seed = _as_int(run_meta.get("seed"), -1)
    if seed < 0:
        seed = _as_int(parsed.get("seed"), -1)

    batch_size = _as_int(cfg.get("batch_size"), -1)
    if batch_size <= 0:
        batch_size = _as_int(run_meta.get("batch_size"), -1)
    if batch_size <= 0:
        batch_size = _as_int(parsed.get("micro_batch"), -1)

    accum = _as_int(cfg.get("accum"), -1)
    if accum <= 0:
        accum = _as_int(run_meta.get("accum"), -1)
    if accum <= 0:
        accum = _as_int(parsed.get("accum"), 1)
    if accum <= 0:
        accum = 1

    effective_batch = _as_int(run_meta.get("effective_batch"), -1)
    if effective_batch <= 0:
        effective_batch = _as_int(sweep_meta.get("effective_batch"), -1)
    if effective_batch <= 0 and batch_size > 0 and accum > 0:
        effective_batch = batch_size * accum

    steps_target = _as_int(run_meta.get("steps"), -1)
    if steps_target <= 0:
        steps_target = _as_int(sweep_meta.get("steps"), -1)
    if steps_target <= 0:
        steps_target = _as_int(parsed.get("steps"), -1)

    steps_completed = _max_step(rows)

    best_mse = _best_row(rows, "val_mse")
    best_mae = _best_row(rows, "val_mae")
    best_nll = _best_row(rows, "val_gauss_nll")
    final_val = _last_val_row(rows)
    has_logged_nll = any("val_gauss_nll" in r for r in rows)

    best_val_mse = _as_float(best_mse.get("val_mse")) if best_mse else float("nan")
    best_val_mae = _as_float(best_mae.get("val_mae")) if best_mae else float("nan")

    best_val_gauss_nll = _as_float(best_nll.get("val_gauss_nll")) if best_nll else float("nan")
    gauss_nll_source = "history"
    if not np.isfinite(best_val_gauss_nll):
        best_val_gauss_nll = _fallback_gauss_nll_from_mse(best_val_mse)
        gauss_nll_source = "derived_from_best_mse"

    tokens_seen = (
        int(steps_completed * effective_batch * vocab_size)
        if (steps_completed > 0 and effective_batch > 0 and vocab_size > 0)
        else -1
    )
    tokens_target = (
        int(steps_target * effective_batch * vocab_size)
        if (steps_target > 0 and effective_batch > 0 and vocab_size > 0)
        else -1
    )

    rec: Dict[str, Any] = {
        "run": str(run_dir),
        "run_name": run_dir.name,
        "data": data_path,
        "size": size,
        "seed": seed,
        "V": vocab_size,
        "params": params,
        "lr": _as_float(cfg.get("lr"), _as_float(run_meta.get("lr"), float("nan"))),
        "batch_size": batch_size,
        "accum": accum,
        "eff_batch": effective_batch,
        "steps_target": steps_target,
        "steps_completed": steps_completed,
        "tokens_target": tokens_target,
        "tokens_seen": tokens_seen,
        "n_history_rows": len(rows),
        "n_val_points": int(sum(1 for r in rows if "val_mse" in r or "val_gauss_nll" in r)),
        "has_logged_gauss_nll": bool(has_logged_nll),
        "gauss_nll_source": gauss_nll_source,
        "best_step_mse": _as_int(best_mse.get("step"), -1) if best_mse else -1,
        "best_val_mse": best_val_mse,
        "best_step_mae": _as_int(best_mae.get("step"), -1) if best_mae else -1,
        "best_val_mae": best_val_mae,
        "best_step_gauss_nll": _as_int(best_nll.get("step"), -1) if best_nll else -1,
        "best_val_gauss_nll": best_val_gauss_nll,
        "val_mae_at_best_mse": _as_float(best_mse.get("val_mae"), float("nan")) if best_mse else float("nan"),
        "val_gauss_nll_at_best_mse": (
            _as_float(best_mse.get("val_gauss_nll"), _fallback_gauss_nll_from_mse(best_val_mse))
            if best_mse
            else float("nan")
        ),
        "final_step": _as_int(final_val.get("step"), -1) if final_val else -1,
        "final_val_mse": _as_float(final_val.get("val_mse"), float("nan")) if final_val else float("nan"),
        "final_val_mae": _as_float(final_val.get("val_mae"), float("nan")) if final_val else float("nan"),
        "final_val_gauss_nll": (
            _as_float(final_val.get("val_gauss_nll"), _fallback_gauss_nll_from_mse(_as_float(final_val.get("val_mse"), float("nan"))))
            if final_val
            else float("nan")
        ),
    }
    return rec


def fit_power_with_offset(x: np.ndarray, y: np.ndarray, n_cand: int = 300) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        raise ValueError("Need at least 3 points for fit.")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("x and y must be positive for log-log fit.")

    c_max = 0.99 * float(np.min(y))
    c_grid = np.linspace(0.0, max(0.0, c_max), n_cand)
    if len(c_grid) == 0:
        c_grid = np.array([0.0])

    lx = np.log(x)
    best: Dict[str, float] = {"r2": -np.inf}

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
            best = {
                "alpha": float(alpha),
                "a": float(np.exp(loga)),
                "c": float(c),
                "r2": float(r2),
            }
    if best["r2"] == -np.inf:
        raise ValueError("No feasible fit found.")
    return best


def _entropy_bits_from_floor(metric: str, floor: float) -> float:
    if not np.isfinite(floor):
        return float("nan")
    if metric == "best_val_mse":
        if floor <= 0:
            return float("nan")
        return 0.5 * math.log2(2.0 * math.pi * math.e * floor)
    if metric == "best_val_gauss_nll":
        return floor / math.log(2.0)
    return float("nan")


def fit_and_residuals(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    min_points: int,
    n_cand: int,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    fit_rows: List[Dict[str, Any]] = []
    resid_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for v_value, g in df.groupby("V"):
        keep_cols = list(dict.fromkeys([x_col, y_col, "run", "size", "seed", "params", "tokens_seen"]))
        sub = g[keep_cols].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
        sub = sub[(sub[x_col] > 0) & (sub[y_col] > 0)]
        if len(sub) < min_points:
            skipped.append(
                {
                    "V": int(v_value),
                    "x_axis": x_col,
                    "metric": y_col,
                    "reason": "not_enough_points",
                    "n_points": int(len(sub)),
                }
            )
            continue

        try:
            fit = fit_power_with_offset(sub[x_col].to_numpy(float), sub[y_col].to_numpy(float), n_cand=n_cand)
        except Exception as e:
            skipped.append(
                {
                    "V": int(v_value),
                    "x_axis": x_col,
                    "metric": y_col,
                    "reason": f"fit_failed:{e}",
                    "n_points": int(len(sub)),
                }
            )
            continue

        x = sub[x_col].to_numpy(float)
        y = sub[y_col].to_numpy(float)
        y_hat = fit["a"] * x ** (-fit["alpha"]) + fit["c"]
        y_adj = y - fit["c"]
        y_hat_adj = fit["a"] * x ** (-fit["alpha"])
        resid_log = np.log(y_adj) - np.log(y_hat_adj)
        resid_lin = y - y_hat

        med = float(np.median(resid_log))
        hi = resid_log[resid_log >= med]
        lo = resid_log[resid_log < med]
        std = float(np.std(resid_log)) if len(resid_log) > 1 else float("nan")
        band_gap = float(np.mean(hi) - np.mean(lo)) if len(hi) > 0 and len(lo) > 0 else float("nan")
        pos_frac = float(np.mean(resid_log > 0))
        band_gap_over_std = float(band_gap / std) if np.isfinite(std) and std > 0 else float("nan")
        two_band_signal = bool(
            len(resid_log) >= 6
            and np.isfinite(band_gap_over_std)
            and (band_gap_over_std >= 1.0)
            and (abs(pos_frac - 0.5) <= 0.2)
        )

        fit_rows.append(
            {
                "V": int(v_value),
                "x_axis": x_col,
                "metric": y_col,
                "n_points": int(len(sub)),
                "alpha": fit["alpha"],
                "a": fit["a"],
                "c": fit["c"],
                "r2_log": fit["r2"],
                "entropy_floor_bits_per_masked_pos": _entropy_bits_from_floor(y_col, fit["c"]),
                "residual_pos_frac": pos_frac,
                "residual_band_gap_log": band_gap,
                "residual_band_gap_over_std": band_gap_over_std,
                "two_band_signal": two_band_signal,
            }
        )

        for i, row in sub.reset_index(drop=True).iterrows():
            resid_rows.append(
                {
                    "V": int(v_value),
                    "x_axis": x_col,
                    "metric": y_col,
                    "run": row["run"],
                    "size": row["size"],
                    "seed": row["seed"],
                    "x": float(row[x_col]),
                    "y": float(row[y_col]),
                    "y_hat": float(y_hat[i]),
                    "residual_linear": float(resid_lin[i]),
                    "residual_log": float(resid_log[i]),
                }
            )

    return pd.DataFrame(fit_rows), resid_rows, skipped


def summarize_by_size(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    for (v_value, size), g in df.groupby(["V", "size"], dropna=False):
        rec: Dict[str, Any] = {
            "V": int(v_value),
            "size": size,
            "n_runs": int(len(g)),
            "params_median": float(g["params"].median()),
            "tokens_seen_median": float(g["tokens_seen"].replace(-1, np.nan).median()),
            "steps_completed_median": float(g["steps_completed"].median()),
        }
        for col in metric_cols:
            vals = g[col].replace([np.inf, -np.inf], np.nan).dropna()
            rec[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            rec[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            rec[f"{col}_sem"] = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        recs.append(rec)
    out = pd.DataFrame(recs)
    if len(out) == 0:
        return out
    out["size_order"] = out["size"].map(_size_sort_key)
    out = out.sort_values(["V", "size_order", "params_median"]).drop(columns=["size_order"])
    return out


def plot_loss_vs_x(
    df: pd.DataFrame,
    fit_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    plotted_any = False

    for v_value, g in df.groupby("V"):
        sub = g[(g[x_col] > 0) & (g[y_col] > 0)].copy()
        if len(sub) == 0:
            continue
        plotted_any = True
        ax.scatter(sub[x_col], sub[y_col], s=36, alpha=0.8, label=f"V={int(v_value)} runs")

        fit_row = fit_df[(fit_df["V"] == int(v_value)) & (fit_df["x_axis"] == x_col) & (fit_df["metric"] == y_col)]
        if len(fit_row) == 1:
            fr = fit_row.iloc[0]
            x_min = float(sub[x_col].min()) * 0.9
            x_max = float(sub[x_col].max()) * 1.1
            x_line = np.logspace(np.log10(x_min), np.log10(x_max), 250)
            y_line = float(fr["a"]) * x_line ** (-float(fr["alpha"])) + float(fr["c"])
            ax.plot(
                x_line,
                y_line,
                linewidth=2.0,
                label=f"V={int(v_value)} fit: alpha={float(fr['alpha']):.3f}, c={float(fr['c']):.4g}",
            )

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters P" if x_col == "params" else "Tokens seen T")
    ax.set_ylabel(y_col)
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_bars(summary_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    if len(summary_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    plotted_any = False

    for v_value, g in summary_df.groupby("V"):
        sub = g.dropna(subset=[f"{metric}_mean"]).copy()
        if len(sub) == 0:
            continue
        plotted_any = True
        x = sub["params_median"].to_numpy(float)
        y = sub[f"{metric}_mean"].to_numpy(float)
        yerr = sub[f"{metric}_std"].to_numpy(float)
        labels = sub["size"].astype(str).to_list()
        ax.errorbar(
            x,
            y,
            yerr=np.where(np.isfinite(yerr), yerr, 0.0),
            fmt="o",
            capsize=3,
            label=f"V={int(v_value)}",
        )
        for xi, yi, lbl in zip(x, y, labels):
            ax.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters P")
    ax.set_ylabel(metric)
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(residuals_df: pd.DataFrame, *, x_axis: str, metric: str, out_path: Path) -> None:
    sub = residuals_df[(residuals_df["x_axis"] == x_axis) & (residuals_df["metric"] == metric)].copy()
    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for v_value, g in sub.groupby("V"):
        x = g["x"].to_numpy(float)
        y = g["residual_log"].to_numpy(float)
        ax.scatter(x, y, s=30, alpha=0.8, label=f"V={int(v_value)}")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters P" if x_axis == "params" else "Tokens seen T")
    ax.set_ylabel("Log residual")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_contradiction_report(fits: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "material_alpha_rel_diff_threshold": 0.15,
        "two_band_signals": [],
        "comparisons_by_metric_across_V": [],
        "comparisons_by_V_across_metric": [],
    }

    if len(fits) == 0:
        return report

    for _, r in fits.sort_values(["x_axis", "metric", "V"]).iterrows():
        report["two_band_signals"].append(
            {
                "V": int(r["V"]),
                "x_axis": str(r["x_axis"]),
                "metric": str(r["metric"]),
                "two_band_signal": bool(r["two_band_signal"]),
                "residual_pos_frac": float(r["residual_pos_frac"]),
                "residual_band_gap_over_std": float(r["residual_band_gap_over_std"]),
            }
        )

    for (x_axis, metric), g in fits.groupby(["x_axis", "metric"]):
        if g["V"].nunique() < 2:
            continue
        alphas = g["alpha"].to_numpy(float)
        cs = g["c"].to_numpy(float)
        alpha_rel = (float(np.max(alphas)) - float(np.min(alphas))) / max(1e-12, float(np.mean(np.abs(alphas))))
        c_rel = (float(np.max(cs)) - float(np.min(cs))) / max(1e-12, float(np.mean(np.abs(cs))))
        report["comparisons_by_metric_across_V"].append(
            {
                "x_axis": x_axis,
                "metric": metric,
                "V_values": [int(v) for v in sorted(g["V"].unique().tolist())],
                "alpha_rel_diff": alpha_rel,
                "c_rel_diff": c_rel,
                "material_slope_difference": bool(alpha_rel >= 0.15),
            }
        )

    for (x_axis, v_value), g in fits.groupby(["x_axis", "V"]):
        if g["metric"].nunique() < 2:
            continue
        alphas = g["alpha"].to_numpy(float)
        cs = g["c"].to_numpy(float)
        alpha_rel = (float(np.max(alphas)) - float(np.min(alphas))) / max(1e-12, float(np.mean(np.abs(alphas))))
        c_rel = (float(np.max(cs)) - float(np.min(cs))) / max(1e-12, float(np.mean(np.abs(cs))))
        report["comparisons_by_V_across_metric"].append(
            {
                "x_axis": x_axis,
                "V": int(v_value),
                "metrics": sorted(g["metric"].astype(str).unique().tolist()),
                "alpha_rel_diff": alpha_rel,
                "c_rel_diff": c_rel,
                "material_slope_difference": bool(alpha_rel >= 0.15),
            }
        )
    return report


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_v_filter(v_text: Optional[str]) -> Optional[List[int]]:
    if v_text is None or str(v_text).strip() == "":
        return None
    out: List[int] = []
    for x in str(v_text).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except Exception:
            continue
    return sorted(set(out)) if out else None


def save_markdown_summary(path: Path, *, filtered_df: pd.DataFrame, fits: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Scaling Analysis Summary")
    lines.append("")
    lines.append(f"- Runs analyzed: {len(filtered_df)}")
    if len(filtered_df) > 0:
        lines.append(f"- V values: {sorted(filtered_df['V'].dropna().astype(int).unique().tolist())}")
        lines.append(f"- Sizes: {sorted(filtered_df['size'].dropna().astype(str).unique().tolist(), key=_size_sort_key)}")
    lines.append("")
    lines.append("## Fits")
    lines.append("")
    if len(fits) == 0:
        lines.append("- No successful fits.")
    else:
        for _, r in fits.sort_values(["x_axis", "metric", "V"]).iterrows():
            lines.append(
                f"- V={int(r['V'])}, x={r['x_axis']}, metric={r['metric']}: "
                f"alpha={float(r['alpha']):.4f}, c={float(r['c']):.6g}, R2={float(r['r2_log']):.4f}, "
                f"entropy_floor_bits={float(r['entropy_floor_bits_per_masked_pos']):.4f}"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate runs and fit scaling laws with residual diagnostics.")
    ap.add_argument("--runs", type=str, default="runs", help="Root runs directory (recursive).")
    ap.add_argument("--out", type=str, default="analysis", help="Base analysis output directory.")
    ap.add_argument("--v", type=str, default=None, help="Optional comma-separated V filter (e.g., 512 or 512,1024).")
    ap.add_argument("--min-points", type=int, default=3, help="Minimum points per regime to fit.")
    ap.add_argument("--n-cand", type=int, default=300, help="Number of c-grid candidates for offset fit.")
    ap.add_argument("--tag", type=str, default=None, help="Optional tag for the output folder.")
    args = ap.parse_args()

    v_filter = parse_v_filter(args.v)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) / f"scaling_analysis_{timestamp}{('_' + args.tag) if args.tag else ''}"
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    runs_root = Path(args.runs)
    run_dirs = _find_run_dirs(runs_root)
    if not run_dirs:
        print(f"No runs found under {runs_root}")
        return

    records: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        rec = collect_run_record(run_dir, v_filter=v_filter)
        if rec is not None:
            records.append(rec)

    if not records:
        print("No usable runs with checkpoints/history found.")
        return

    inventory_all = pd.DataFrame(records).sort_values(["V", "params", "size", "seed", "run_name"])
    inventory_all.to_csv(out_root / "stage1_inventory_all_runs.csv", index=False)
    print(f"wrote {out_root / 'stage1_inventory_all_runs.csv'}")

    filtered = inventory_all.copy()
    if v_filter is not None:
        filtered = filtered[filtered["V"].isin(v_filter)].copy()
    filtered = filtered.sort_values(["V", "params", "size", "seed", "run_name"])
    filtered.to_csv(out_root / "stage2_filtered_runs.csv", index=False)
    print(f"wrote {out_root / 'stage2_filtered_runs.csv'}")

    if len(filtered) == 0:
        print("No runs left after V filter.")
        return

    # Keep only the most progressed run for each (dataset, V, size, seed, eff_batch).
    # This prevents shorter interrupted runs from double-counting a seed.
    canonical = (
        filtered.sort_values(["steps_completed", "steps_target"], ascending=[False, False])
        .drop_duplicates(subset=["data", "V", "size", "seed", "eff_batch"], keep="first")
        .sort_values(["V", "params", "size", "seed", "run_name"])
        .copy()
    )
    canonical.to_csv(out_root / "stage2b_canonical_runs.csv", index=False)
    print(f"wrote {out_root / 'stage2b_canonical_runs.csv'}")

    metric_cols = ["best_val_mse", "best_val_gauss_nll"]
    summary = summarize_by_size(canonical, metric_cols)
    summary.to_csv(out_root / "stage3_summary_by_size.csv", index=False)
    print(f"wrote {out_root / 'stage3_summary_by_size.csv'}")

    fit_rows: List[pd.DataFrame] = []
    residual_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    for metric in metric_cols:
        f_params, r_params, skipped_params = fit_and_residuals(
            canonical, "params", metric, min_points=args.min_points, n_cand=args.n_cand
        )
        f_tokens, r_tokens, skipped_tokens = fit_and_residuals(
            canonical, "tokens_seen", metric, min_points=args.min_points, n_cand=args.n_cand
        )
        if len(f_params) > 0:
            fit_rows.append(f_params)
        if len(f_tokens) > 0:
            fit_rows.append(f_tokens)
        residual_rows.extend(r_params)
        residual_rows.extend(r_tokens)
        skipped_rows.extend(skipped_params)
        skipped_rows.extend(skipped_tokens)

    fits_df = pd.concat(fit_rows, ignore_index=True) if fit_rows else pd.DataFrame()
    residuals_df = pd.DataFrame(residual_rows)

    if len(fits_df) > 0:
        fits_df = fits_df.sort_values(["x_axis", "metric", "V"])
        fits_df.to_csv(out_root / "stage4_fits.csv", index=False)
        print(f"wrote {out_root / 'stage4_fits.csv'}")
    else:
        print("No successful fits.")

    if len(residuals_df) > 0:
        residuals_df = residuals_df.sort_values(["x_axis", "metric", "V", "size", "seed"])
        residuals_df.to_csv(out_root / "stage5_residuals.csv", index=False)
        print(f"wrote {out_root / 'stage5_residuals.csv'}")

    skipped_df = pd.DataFrame(skipped_rows)
    if len(skipped_df) > 0:
        skipped_df.to_csv(out_root / "stage6_fit_skips.csv", index=False)
        print(f"wrote {out_root / 'stage6_fit_skips.csv'}")

    if len(fits_df) > 0:
        for metric in metric_cols:
            plot_loss_vs_x(
                canonical,
                fits_df,
                x_col="params",
                y_col=metric,
                out_path=plots_dir / f"loss_vs_params_{metric}.png",
            )
            plot_loss_vs_x(
                canonical,
                fits_df,
                x_col="tokens_seen",
                y_col=metric,
                out_path=plots_dir / f"loss_vs_tokens_{metric}.png",
            )
            plot_error_bars(summary, metric, plots_dir / f"errorbars_{metric}.png")
            if len(residuals_df) > 0:
                plot_residuals(
                    residuals_df,
                    x_axis="params",
                    metric=metric,
                    out_path=plots_dir / f"residuals_params_{metric}.png",
                )
                plot_residuals(
                    residuals_df,
                    x_axis="tokens_seen",
                    metric=metric,
                    out_path=plots_dir / f"residuals_tokens_{metric}.png",
                )

    contradiction_report = build_contradiction_report(fits_df if len(fits_df) > 0 else pd.DataFrame())
    write_json(out_root / "stage7_contradiction_report.json", contradiction_report)
    print(f"wrote {out_root / 'stage7_contradiction_report.json'}")

    save_markdown_summary(out_root / "stage8_summary.md", filtered_df=canonical, fits=fits_df)
    print(f"wrote {out_root / 'stage8_summary.md'}")

    latest_ptr = Path(args.out) / "latest_scaling_analysis.txt"
    latest_ptr.write_text(str(out_root), encoding="utf-8")
    print(f"wrote {latest_ptr}")


if __name__ == "__main__":
    main()
