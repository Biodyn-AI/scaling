#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, subprocess, sys, time, re
from datetime import datetime
from pathlib import Path

# project layout
ROOT = Path(__file__).parent.resolve()
SRC_DIR = ROOT / "src"
DATA_BUILDER_MOD = "src.data.data"   # used to (re)build datasets
TRAIN_MOD = "model.train"            # model trainer (python -m model.train)

def run(cmd, **kwargs):
    print("$", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, **kwargs)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

def build_dataset(out_path: Path, cells: int, hvg: int) -> Path:
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

def detect_V_from_filename(path: Path) -> int | None:
    m = re.search(r"V(\d+)", path.name)
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser(description="Launch scaling sweeps for scRNA transformers")
    # Dataset choice
    ap.add_argument("--data", type=str, default=None,
                    help="Path to existing .h5ad. If omitted and --cells is set, a dataset will be built.")
    ap.add_argument("--cells", type=int, default=None, help="If set, build a canonical dataset with N cells")
    ap.add_argument("--hvg", type=int, default=2000, help="Number of genes when building a dataset")
    # Sweep space
    ap.add_argument("--sizes", type=str, default="XS,S,M",
                    help="Comma-separated presets to run (from utils.PRESETS)")
    ap.add_argument("--seeds", type=str, default="7", help="Comma-separated seeds")
    # Training budget / cadence
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--train-metrics-every", type=int, default=None)
    # Batch/accum + LR scaling
    ap.add_argument("--micro-batch", type=int, default=8, help="Per-step micro-batch")
    ap.add_argument("--target-batch", type=int, default=64, help="Desired effective batch (micro*accum)")
    ap.add_argument("--lr-base", type=float, default=2.5e-4, help="Base LR before scaling")
    ap.add_argument("--lr-scale-ref", type=int, default=256,
                    help="Reference batch for linear LR scaling (eff_batch/ref)")
    ap.add_argument("--no-lr-scale", action="store_true", help="Disable LR linear scaling")
    # System
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--amp", action="store_true", help="Use autocast(float16) if trainer supports --amp")
    ap.add_argument("--cap-genes", type=int, default=None, help="Slice to first K genes (no rebuild)")
    # Output
    ap.add_argument("--outdir", type=str, default=None, help="Base output dir (auto-named if not set)")
    ap.add_argument("--tag", type=str, default=None, help="Optional label for the sweep folder name")

    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Decide data path
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            raise SystemExit(f"--data not found: {data_path}")
    elif args.cells is not None:
        data_path = ROOT / f"data/hlca_minified.D{args.cells}.V{args.hvg}.log1p.h5ad"
        if not data_path.exists():
            print(f"[build] creating {data_path}")
            build_dataset(data_path, args.cells, args.hvg)
        else:
            print(f"[build] using existing {data_path}")
    else:
        raise SystemExit("Provide --data or (--cells and --hvg).")

    V_hint = detect_V_from_filename(data_path)  # just for naming

    # Output folder
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.outdir) if args.outdir else (ROOT / "runs" / f"sweep_{stamp}{('_'+args.tag) if args.tag else ''}")
    base.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) if not old_pp else old_pp + os.pathsep + str(SRC_DIR)

    # Compute accumulation + scaled LR
    micro = max(1, int(args.micro_batch))
    eff = int(args.target_batch)
    accum = max(1, (eff + micro - 1) // micro)  # ceil
    eff_actual = micro * accum
    lr = args.lr_base if args.no_lr_scale else args.lr_base * (eff_actual / float(args.lr_scale_ref))

    print(f"[sweep] micro={micro} accum={accum} -> effective batch={eff_actual}; LR={lr:.3g}")

    # Persist overall sweep meta
    with open(base / "sweep_meta.json", "w") as f:
        json.dump({
            "data": str(data_path),
            "sizes": sizes,
            "seeds": seeds,
            "steps": args.steps,
            "val_every": args.val_every,
            "log_every": args.log_every,
            "train_metrics_every": args.train_metrics_every,
            "micro_batch": micro,
            "accum": accum,
            "effective_batch": eff_actual,
            "lr": lr,
            "lr_base": args.lr_base,
            "lr_scale_ref": args.lr_scale_ref,
            "lr_scaled": not args.no_lr_scale,
            "cap_genes": args.cap_genes,
            "V_from_filename": V_hint,
        }, f, indent=2)

    # Launch runs
    for size in sizes:
        for seed in seeds:
            run_name = f"{size}_V{V_hint or 'x'}_B{micro}x{accum}_S{args.steps}_seed{seed}"
            out = base / run_name
            out.mkdir(parents=True, exist_ok=True)

            # per-run meta (useful if you later move folders)
            with open(out / "run_meta.json", "w") as f:
                json.dump({
                    "size": size,
                    "seed": seed,
                    "data": str(data_path),
                    "steps": args.steps,
                    "val_every": args.val_every,
                    "log_every": args.log_every,
                    "train_metrics_every": args.train_metrics_every,
                    "batch_size": micro,
                    "accum": accum,
                    "effective_batch": eff_actual,
                    "lr": lr,
                    "cap_genes": args.cap_genes,
                    "num_workers": args.num_workers,
                }, f, indent=2)

            cmd = [
                sys.executable, "-m", TRAIN_MOD,
                "--data", str(data_path),
                "--size", size,
                "--batch-size", str(micro),
                "--steps", str(args.steps),
                "--val-every", str(args.val_every),
                "--log-every", str(args.log_every),
                "--lr", str(lr),
                "--seed", str(seed),
                "--outdir", str(out),
                "--num-workers", str(int(args.num_workers)),
            ]
            if args.train_metrics_every is not None:
                cmd += ["--train-metrics-every", str(int(args.train_metrics_every))]
            if args.cap_genes is not None:
                cmd += ["--cap-genes", str(int(args.cap_genes))]
            if args.device:
                cmd += ["--device", args.device]
            if args.amp:
                cmd += ["--amp"]
            # gradient accumulation (trainer patch below)
            cmd += ["--accum", str(int(accum))]

            try:
                run(cmd, cwd=str(ROOT), env=env)
            except KeyboardInterrupt:
                print("\n[abort] sweep interrupted by user")
                return

    print(f"\n[sweep] done -> {base}")

if __name__ == "__main__":
    main()
