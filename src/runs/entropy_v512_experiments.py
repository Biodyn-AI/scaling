#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc


PHASE_ORDER = {"fixed": 0, "data": 1, "compute": 2, "data_compute": 3}
SIZE_ORDER = {"XXS": 0, "TINY": 1, "XS": 2, "S": 3, "M": 4, "L": 5, "XL": 6}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def cells_label(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def _safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _log_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _max_history_step(run_dir: Path) -> int:
    hist = run_dir / "history.jsonl"
    if not hist.exists():
        return -1
    mx = -1
    with open(hist, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                s = int(j.get("step", -1))
            except Exception:
                continue
            mx = max(mx, s)
    return mx


def _pick_resume(run_dir: Path) -> Path | None:
    cands = [run_dir / "ckpt_last.pt", run_dir / "ckpt_best.pt", run_dir / "ckpt_final.pt"]
    for p in cands:
        if p.exists():
            return p
    return None


def _is_done(run_dir: Path, target_steps: int) -> Tuple[bool, int]:
    mx = _max_history_step(run_dir)
    return (mx >= target_steps), mx


def _build_subset(
    adata,
    n_cells: int,
    *,
    seed: int,
    split_col: str = "split",
):
    if n_cells >= adata.n_obs:
        return adata.copy()
    rng = np.random.default_rng(seed)
    if split_col not in adata.obs.columns:
        idx = rng.choice(adata.n_obs, size=n_cells, replace=False)
        return adata[idx].copy()

    split = adata.obs[split_col].astype(str).to_numpy()
    uniq, counts = np.unique(split, return_counts=True)
    frac = counts / counts.sum()
    raw = frac * n_cells
    take = np.floor(raw).astype(int)
    rem = n_cells - int(take.sum())
    if rem > 0:
        order = np.argsort(raw - take)[::-1]
        for i in order[:rem]:
            take[i] += 1

    idx_parts: List[np.ndarray] = []
    for s, n_take in zip(uniq, take):
        if n_take <= 0:
            continue
        pool = np.where(split == s)[0]
        n_take = min(int(n_take), len(pool))
        idx_parts.append(rng.choice(pool, size=n_take, replace=False))
    idx = np.concatenate(idx_parts, axis=0)
    rng.shuffle(idx)
    return adata[idx].copy()


def ensure_datasets(
    *,
    base_data: Path,
    sizes: Iterable[int],
    out_dir: Path,
    seed: int,
    progress_log: Path,
) -> Dict[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[int, Path] = {}
    sizes = sorted(set(int(x) for x in sizes))
    paths[200_000] = base_data

    to_make = [n for n in sizes if n != 200_000 and not (out_dir / f"census_human.D{cells_label(n)}.V512.log1p.h5ad").exists()]
    if to_make:
        _log_jsonl(progress_log, {"ts": now(), "event": "dataset_prepare_start", "base_data": str(base_data), "targets": to_make})
        ad = sc.read_h5ad(base_data)
        for n in to_make:
            out = out_dir / f"census_human.D{cells_label(n)}.V512.log1p.h5ad"
            sub = _build_subset(ad, n_cells=n, seed=seed + n)
            sub.write_h5ad(out)
            _log_jsonl(
                progress_log,
                {
                    "ts": now(),
                    "event": "dataset_written",
                    "path": str(out),
                    "shape": [int(sub.n_obs), int(sub.n_vars)],
                    "split_counts": (
                        sub.obs["split"].astype(str).value_counts().to_dict()
                        if "split" in sub.obs.columns
                        else {}
                    ),
                },
            )
            paths[n] = out
    for n in sizes:
        if n == 200_000:
            paths[n] = base_data
        else:
            p = out_dir / f"census_human.D{cells_label(n)}.V512.log1p.h5ad"
            if not p.exists():
                raise FileNotFoundError(f"Expected dataset missing: {p}")
            paths[n] = p
    return paths


def build_manifest_rows(
    *,
    dataset_paths: Dict[int, Path],
    run_root: Path,
    phases: List[str],
    batch_size: int,
    accum: int,
    lr: float,
    val_every: int,
    log_every: int,
    train_metrics_every: int,
    save_last_every: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eff_batch = batch_size * accum

    def add_job(phase: str, data_cells: int, size: str, seed: int, steps: int) -> None:
        dlab = cells_label(data_cells)
        jid = f"{phase}_D{dlab}_{size}_seed{seed}_S{steps}"
        outdir = run_root / phase / f"{size}_D{dlab}_seed{seed}_S{steps}"
        rows.append(
            {
                "job_id": jid,
                "phase": phase,
                "data_cells": data_cells,
                "data_path": str(dataset_paths[data_cells]),
                "size": size,
                "seed": seed,
                "steps": steps,
                "batch_size": batch_size,
                "accum": accum,
                "eff_batch": eff_batch,
                "lr": lr,
                "val_every": val_every,
                "log_every": log_every,
                "train_metrics_every": train_metrics_every,
                "save_last_every": save_last_every,
                "amp": True,
                "outdir": str(outdir),
            }
        )

    if "fixed" in phases:
        for size in ["XS", "S", "M", "L"]:
            for seed in [7, 8, 9]:
                add_job("fixed", 200_000, size, seed, 60_000)

    if "data" in phases:
        for data_cells in [25_000, 50_000, 100_000, 200_000]:
            for seed in [7, 8, 9]:
                add_job("data", data_cells, "M", seed, 60_000)

    if "compute" in phases:
        for size in ["XS", "S", "M", "L"]:
            for steps in [10_000, 20_000, 40_000]:
                add_job("compute", 200_000, size, 7, steps)

    if "data_compute" in phases:
        for data_cells in [25_000, 50_000, 100_000, 200_000]:
            for steps in [10_000, 20_000, 40_000]:
                add_job("data_compute", data_cells, "M", 7, steps)

    return rows


def refresh_manifest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    status: List[str] = []
    max_steps: List[int] = []
    for _, r in out.iterrows():
        run_dir = Path(str(r["outdir"]))
        done, mx = _is_done(run_dir, _safe_int(r["steps"], -1))
        status.append("done" if done else "pending")
        max_steps.append(mx)
    out["status"] = status
    out["max_step"] = max_steps
    return out


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(by=["phase", "data_cells", "size", "seed", "steps", "job_id"])
    df.to_csv(path, index=False)


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] CMD: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
        return proc.wait()


def run_intermediate_analyses(python_exe: str, env: Dict[str, str], analysis_root: Path, run_root: Path) -> None:
    a_log = analysis_root / "analysis_runs.log"
    cmd1 = [
        python_exe,
        "src/analysis/analyze_scaling.py",
        "--runs",
        str(run_root),
        "--out",
        str(analysis_root),
        "--v",
        "512",
        "--tag",
        "interim",
    ]
    run_cmd(cmd1, env, a_log)

    cmd2 = [
        python_exe,
        "src/analysis/section63_entropy_v512.py",
        "--manifest",
        str(analysis_root / "job_manifest.csv"),
        "--out",
        str(analysis_root / "section63"),
        "--tag",
        "interim",
    ]
    run_cmd(cmd2, env, a_log)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run V=512 entropy experiment queue with persistent logs.")
    ap.add_argument("--base-data", default="data/census_human.D200000.V512.log1p.h5ad")
    ap.add_argument("--analysis-root", default="analysis/entropy_v512")
    ap.add_argument("--run-root", default="runs/entropy_v512")
    ap.add_argument("--phases", default="fixed,data,compute,data_compute")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--analyze-every", type=int, default=1)
    ap.add_argument("--pause-file", default=None, help="If this file exists, queue stops before starting next job.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    base_data = Path(args.base_data)
    analysis_root = Path(args.analysis_root)
    run_root = Path(args.run_root)
    progress_log = analysis_root / "progress.jsonl"
    manifest_path = analysis_root / "job_manifest.csv"
    snapshots_dir = analysis_root / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    pause_file = Path(args.pause_file) if args.pause_file else (analysis_root / "PAUSE")

    if not base_data.exists():
        raise SystemExit(f"base dataset missing: {base_data}")

    phases = [x.strip() for x in args.phases.split(",") if x.strip()]
    unknown = [x for x in phases if x not in PHASE_ORDER]
    if unknown:
        raise SystemExit(f"unknown phases: {unknown}")

    dataset_paths = ensure_datasets(
        base_data=base_data,
        sizes=[25_000, 50_000, 100_000, 200_000],
        out_dir=base_data.parent,
        seed=args.seed,
        progress_log=progress_log,
    )

    manifest_rows = build_manifest_rows(
        dataset_paths=dataset_paths,
        run_root=run_root,
        phases=phases,
        batch_size=4,
        accum=8,
        lr=3.125e-05,
        val_every=500,
        log_every=100,
        train_metrics_every=50,
        save_last_every=1000,
    )
    manifest = pd.DataFrame(manifest_rows)
    manifest = refresh_manifest(manifest)
    write_manifest(manifest, manifest_path)
    _log_jsonl(progress_log, {"ts": now(), "event": "manifest_prepared", "path": str(manifest_path), "n_jobs": int(len(manifest))})

    if args.prepare_only and not args.run:
        print(f"Prepared manifest: {manifest_path}")
        return

    if not args.run:
        print("Manifest prepared. Use --run to start queue.")
        return

    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src" if not old_pp else f"{old_pp}{os.pathsep}src"
    env["PYTHONUNBUFFERED"] = "1"

    queue = manifest.copy()
    queue["phase_order"] = queue["phase"].map(PHASE_ORDER).fillna(999).astype(int)
    queue["size_order"] = queue["size"].map(SIZE_ORDER).fillna(999).astype(int)
    queue = queue.sort_values(["status", "phase_order", "size_order", "data_cells", "seed", "steps", "job_id"])

    ran = 0
    for _, job in queue.iterrows():
        if pause_file.exists():
            _log_jsonl(progress_log, {"ts": now(), "event": "queue_paused", "pause_file": str(pause_file)})
            print(f"Pause file detected: {pause_file}. Stopping queue before next job.")
            return

        if str(job["status"]) == "done":
            continue
        if args.max_jobs is not None and ran >= int(args.max_jobs):
            break

        job_id = str(job["job_id"])
        run_dir = Path(str(job["outdir"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        train_log = run_dir / "train.out.log"

        done, mx = _is_done(run_dir, _safe_int(job["steps"], -1))
        if done:
            manifest.loc[manifest["job_id"] == job_id, ["status", "max_step"]] = ["done", mx]
            write_manifest(manifest, manifest_path)
            continue

        resume = _pick_resume(run_dir)
        cmd = [
            args.python_exe,
            "-u",
            "-m",
            "model.train",
            "--data",
            str(job["data_path"]),
            "--size",
            str(job["size"]),
            "--batch-size",
            str(_safe_int(job["batch_size"], 4)),
            "--accum",
            str(_safe_int(job["accum"], 8)),
            "--steps",
            str(_safe_int(job["steps"], 0)),
            "--val-every",
            str(_safe_int(job["val_every"], 1000)),
            "--log-every",
            str(_safe_int(job["log_every"], 200)),
            "--train-metrics-every",
            str(_safe_int(job["train_metrics_every"], 200)),
            "--save-last-every",
            str(_safe_int(job["save_last_every"], 1000)),
            "--lr",
            str(float(job["lr"])),
            "--seed",
            str(_safe_int(job["seed"], 7)),
            "--num-workers",
            "0",
            "--device",
            args.device,
            "--outdir",
            str(run_dir),
            "--amp",
        ]
        if resume is not None:
            cmd += ["--resume", str(run_dir)]

        manifest.loc[manifest["job_id"] == job_id, "status"] = "running"
        manifest.loc[manifest["job_id"] == job_id, "last_start"] = now()
        write_manifest(manifest, manifest_path)
        _log_jsonl(progress_log, {"ts": now(), "event": "job_start", "job_id": job_id, "cmd": cmd})

        rc = run_cmd(cmd, env, train_log)
        done, mx = _is_done(run_dir, _safe_int(job["steps"], -1))
        final_status = "done" if done else ("failed" if rc != 0 else "pending")
        manifest.loc[manifest["job_id"] == job_id, ["status", "max_step", "last_rc", "last_end"]] = [
            final_status,
            mx,
            rc,
            now(),
        ]
        write_manifest(manifest, manifest_path)
        snap = snapshots_dir / f"manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        write_manifest(manifest, snap)

        _log_jsonl(
            progress_log,
            {
                "ts": now(),
                "event": "job_end",
                "job_id": job_id,
                "returncode": rc,
                "status": final_status,
                "max_step": mx,
                "snapshot": str(snap),
            },
        )

        ran += 1
        if args.analyze_every > 0 and (ran % int(args.analyze_every) == 0):
            _log_jsonl(progress_log, {"ts": now(), "event": "analysis_start", "after_jobs": ran})
            run_intermediate_analyses(args.python_exe, env, analysis_root, run_root)
            _log_jsonl(progress_log, {"ts": now(), "event": "analysis_end", "after_jobs": ran})

        if rc != 0 and not args.continue_on_error:
            print(f"Stopped on failure for job {job_id}. Check {train_log}")
            return

    _log_jsonl(progress_log, {"ts": now(), "event": "queue_done", "jobs_run_this_session": ran})
    print(f"Queue finished session. Jobs run: {ran}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
