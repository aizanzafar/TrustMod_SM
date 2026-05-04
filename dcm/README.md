# Dimensional Coupling Matrix (DCM) Analysis

Computes cross-dimensional correlations to assess whether trustworthiness dimensions in content moderation are independently optimisable.

## Overview

The DCM analysis integrates results from all evaluation axes (Trustfulness, Fairness, CI, Safety-Concealment, Safety-Exaggeration) to:
1. Compute the **5×5 Dimensional Coupling Matrix** (Pearson correlation)
2. Measure the **Non-Diagonality Index (NDI)** — overall coupling strength
3. Rank models by **Coupled Trustworthiness Score (CTS)**
4. Identify **Pareto dominance** relations among models

## Scripts

| Script | Description |
|--------|-------------|
| `dcm_analysis.py` | Main DCM computation, CTS ranking, and visualization |

## Input Format

`dcm_scores.csv` — Per-model dimensional scalars in [0,1] where higher = better:

```csv
model,D_Trust,D_Fair,D_CI,D_SafeC,D_SafeE
Qwen-14B,0.720,0.650,0.820,0.100,1.000
...
```

### Scalar Definitions

| Scalar | Formula | Source Dimension |
|--------|---------|-----------------|
| `D_Trust` | Accuracy × (1 − ECE) | Trustfulness |
| `D_Fair` | 1 − mean(DAD) | Fairness |
| `D_CI` | 1 − CI-FPR | Contextual Integrity |
| `D_SafeC` | Abstention rate | Safety (Concealment) |
| `D_SafeE` | 1 − FDS | Safety (Exaggeration) |

## Usage

```bash
# Run DCM analysis with default scores
python dcm_analysis.py --input dcm_scores.csv --output_dir results/

# Custom input
python dcm_analysis.py --input /path/to/your_scores.csv --output_dir my_results/
```

## Output

```
results/
├── dcm_results.json     # Full results (DCM, NDI, CTS, Pareto pairs)
├── dcm_heatmap.pdf      # Annotated DCM heatmap with significance
├── cts_ranking.pdf      # CTS bar chart with limiting dimensions
├── radar_profiles.pdf   # Radar plot of all model profiles
└── cts_table.tex        # LaTeX table for paper
```

## Key Metrics

- **NDI** ∈ [0,1]: 0 = dimensions are independent; 1 = maximally coupled
- **CTS**: Geometric mean of dimension scores with NDI-scaled penalty for uneven profiles
- A CTS of 0 indicates catastrophic failure on at least one dimension

## Reference

- **TrustMod_SM_DCM_Analysis.pdf** — Extended analysis document with full derivations, sensitivity analysis, and discussion
