# AGENTS.md

Purpose: Guidance for automated agents and contributors working in this repository.

## Repository Summary

- **Goal**: Study scaling laws for single-cell foundation models (scRNA-seq) using masked-reconstruction transformers.
- **Inputs**: Processed AnnData `.h5ad` files with train/val/test splits.
- **Outputs**: Training runs with checkpoints and metrics, plus aggregate scaling-law fits.

## Layout

- `src/data/` -- Dataset builders and preprocessing helpers.
- `src/model/` -- Model definitions (`model.py`) and training loop (`train.py`).
- `src/analysis/` -- Aggregation and scaling-law fitting scripts.
- `src/pipeline/` -- Python API wrappers (`ensure_dataset`, `train_once`, `sweep`, `analyze_runs`).
- `src/sweep.py` -- Multi-size, multi-seed sweep orchestrator.
- `analysis/` -- Curated aggregate results (CSVs, JSONs, PNGs).

## Conventions

- Run all commands from the repository root.
- `model.*` modules require `PYTHONPATH=src` (namespace packages under `src/`).
- `src/sweep.py` launches training subprocesses; use absolute paths for `--data` and `--outdir`.
- Data artifacts (`data/`, `runs/`) are `.gitignore`d. Rebuild locally as needed.

## Common Workflows

```bash
# Build dataset
python -m src.data.data --source hlca-minified --max-cells 20000 --hvg 1024 \
  --out data/hlca_minified.D20k.V1024.log1p.h5ad

# Train a model
PYTHONPATH=src python -m model.train \
  --data data/hlca_minified.D20k.V1024.log1p.h5ad \
  --size XS --steps 200 --val-every 20 --outdir runs/demo_xs

# Run a sweep
PYTHONPATH=src python src/sweep.py \
  --data data/census_human.D200000.V512.log1p.h5ad \
  --sizes XXS,TINY,XS,S,M,L --seeds 7,8,9 --steps 60000

# Analyze scaling laws
PYTHONPATH=src python -m src.analysis.analyze_scaling --runs runs --out analysis/results
```
