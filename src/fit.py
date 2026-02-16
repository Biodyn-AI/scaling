# src/fit.py
from pathlib import Path
from pipeline import analyze_runs

# Project root = parent of src/
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"         # <-- real runs dir
OUT  = ROOT / "analysis"     # results will be written here

df, fits = analyze_runs(runs=RUNS, out=OUT)

# Safe print even if some columns are absent
cols_to_show = [c for c in ["run", "size", "params", "seed",
                            "val_mse_first", "val_mse_last", "best_val_mse"]
                if c in df.columns]
print(df.sort_values([c for c in ["params", "seed"] if c in df.columns])[cols_to_show])
print("\nFit keys available:", list(fits.keys()))
