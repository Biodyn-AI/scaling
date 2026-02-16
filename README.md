# Scaling Laws for Masked-Reconstruction Transformers on Single-Cell Transcriptomics

This repository contains the code, analysis pipelines, and experimental results accompanying the paper:

> **Scaling Laws for Masked-Reconstruction Transformers on Single-Cell Transcriptomics**

We study how validation loss scales with model size for masked-reconstruction transformers trained on single-cell RNA-seq (scRNA-seq) data. Across two experimental regimes and seven model sizes (533 to 341M parameters), we fit the parametric scaling law L = aP^{-alpha} + c and find clear power-law scaling in a data-rich regime (alpha ~ 0.27, R^2 = 0.86) but negligible scaling when data are scarce.

## Key Results

| Regime | Genes | Cells | alpha | c (floor) | R^2 | Entropy (bits/pos) |
|--------|-------|-------|-------|-----------|-----|--------------------|
| A (data-rich) | 512 | 200,000 | 0.266 | 1.444 | 0.858 | ~2.30 |
| B (data-limited) | 1,024 | 10,000 | 0.009 | 0.113 | 0.017 | -- |

## Repository Structure

```
src/
  data/              # Dataset loading and preprocessing
    data.py          # Main data builder (CELLxGENE Census, HLCA, PBMC)
    build_human_census.py   # Large-scale Census fetcher
    preprocessing.py # HVG selection, normalization
    check_dataset.py # Quick .h5ad inspection utility
  model/             # Transformer architecture and training
    model.py         # ScTransformer: permutation-invariant encoder
    utils.py         # Size presets (XXS-XL), loss functions, factory
    train.py         # Training loop with checkpointing and plotting
  analysis/          # Scaling-law analysis and fitting
    analyze_scaling.py      # 8-stage analysis pipeline with entropy estimation
    fit_two_regimes.py      # Per-regime power-law fits
    eval_runs_test.py       # Test-set evaluation of trained models
    eval_baseline.py        # Gene-mean baseline predictor
  pipeline/          # High-level Python API
    api.py           # ensure_dataset(), train_once(), sweep(), analyze_runs()
  sweep.py           # Orchestrate multi-size, multi-seed training sweeps
analysis/            # Curated analysis outputs (CSVs, plots, fit summaries)
pyproject.toml       # Python package configuration
```

**Not included in this repository** (due to size):
- `data/` -- Processed `.h5ad` datasets (~700 MB). See [Data](#data) below to rebuild.
- `runs/` -- Training run outputs with checkpoints (~23 GB).
- `paper/` -- LaTeX source and compiled PDF.

## Requirements

- Python >= 3.10
- PyTorch >= 2.2
- scanpy >= 1.9, anndata >= 0.10, scvi-tools >= 1.3
- numpy, pandas, scikit-learn, matplotlib, rich

Install with:
```bash
pip install torch scanpy anndata scvi-tools numpy pandas scikit-learn umap-learn rich matplotlib
```

## Quick Start

### 1. Build a dataset

```bash
# Build from CELLxGENE Census (200k cells, 512 HVGs)
python -m src.data.data \
  --source local --local-h5ad /path/to/census.h5ad \
  --max-cells 200000 --hvg 512 \
  --out data/census_human.D200000.V512.log1p.h5ad

# Or use the HLCA fallback for a quick test
python -m src.data.data \
  --source hlca-minified --max-cells 20000 --hvg 1024 \
  --out data/hlca_minified.D20k.V1024.log1p.h5ad
```

### 2. Train a model

```bash
PYTHONPATH=src python -m model.train \
  --data data/census_human.D200000.V512.log1p.h5ad \
  --size XS --steps 5000 --val-every 500 \
  --batch-size 4 --accum 8 --amp \
  --outdir runs/xs_demo
```

### 3. Run a scaling sweep

```bash
PYTHONPATH=src python src/sweep.py \
  --data data/census_human.D200000.V512.log1p.h5ad \
  --sizes XXS,TINY,XS,S,M,L \
  --seeds 7,8,9 \
  --steps 60000 --val-every 1000
```

### 4. Analyze scaling laws

```bash
PYTHONPATH=src python -m src.analysis.analyze_scaling \
  --runs runs --out analysis/results
```

This produces:
- `scaling_table.csv` -- Per-run metrics
- `stage4_fits.csv` -- Power-law fits with entropy floor estimates
- Scaling plots (loss vs. parameters, loss vs. tokens)

## Model Architecture

The `ScTransformer` is a permutation-invariant transformer encoder for set-structured gene expression data:

- **Input**: Gene ID embeddings + value projections + learned mask token
- **Encoder**: Standard transformer encoder layers (Pre-LN, GELU, no positional encoding)
- **Output**: Per-gene expression reconstruction + pooled cell embedding

Seven size presets span ~5 orders of magnitude:

| Size | d_model | Layers | Heads | Params (V=512) |
|------|---------|--------|-------|----------------|
| XXS  | 1       | 1      | 1     | 534            |
| TINY | 16      | 1      | 1     | 9,937          |
| XS   | 64      | 2      | 4     | 133K           |
| S    | 128     | 4      | 8     | 859K           |
| M    | 512     | 6      | 8     | 19.2M          |
| L    | 1020    | 8      | 12    | 101M           |
| XL   | 1536    | 12     | 16    | 341M           |

## Data

Datasets are built from the [CELLxGENE Census](https://cellxgene.cziscience.com/) using the pipeline in `src/data/data.py`. The preprocessing steps are:

1. Load expression matrix (prefer raw counts)
2. HVG selection (Seurat v3 method)
3. Library-size normalization (target sum = 10^4)
4. log(1+x) transformation
5. Stratified train/val/test split (90/5/5%)

## Analysis Pipeline

The analysis script (`src/analysis/analyze_scaling.py`) implements an 8-stage pipeline:

1. **Inventory**: Discover all runs with checkpoints
2. **Filter**: Select runs by vocabulary size
3. **Canonicalize**: Deduplicate to one run per (dataset, size, seed)
4. **Fit**: Power-law fits L = aP^{-alpha} + c with entropy floor conversion
5. **Residuals**: Per-run residual analysis
6. **Skips**: Log insufficient-data fits
7. **Contradiction**: Cross-metric/axis comparison
8. **Summary**: Human-readable markdown report

## Citation

If you use this code, please cite:

```bibtex
@article{scalinglaws2026scrna,
  title={Scaling Laws for Masked-Reconstruction Transformers on Single-Cell Transcriptomics},
  author={Anonymous},
  year={2026}
}
```

## License

This project is released for academic and research use.
