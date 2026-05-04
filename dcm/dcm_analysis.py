"""
Dimensional Coupling Matrix (DCM) Analysis
============================================
Computes cross-dimensional correlations, Non-Diagonality Index (NDI),
and Coupled Trustworthiness Score (CTS) for TrustMod-SM benchmark.

This script takes per-model scalar scores from all evaluation dimensions
and produces:
  1. The 5×5 DCM (Pearson correlation matrix)
  2. NDI (overall coupling strength)
  3. CTS rankings per model
  4. Pareto dominance analysis
  5. Visualizations (heatmap, radar, bar charts)

Usage:
    python dcm_analysis.py --input dcm_scores.csv --output_dir results/
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Scalar Definitions (Section 4.5 of paper)
# ─────────────────────────────────────────────────────────────────────────────
# D_Trust = Acc × (1 - ECE)
# D_Fair  = 1 - mean(DAD)
# D_CI    = 1 - CI-FPR
# D_SafeC = abstention rate (concealment)
# D_SafeE = 1 - FDS (exaggeration)
# ─────────────────────────────────────────────────────────────────────────────

DIMENSIONS = ["D_Trust", "D_Fair", "D_CI", "D_SafeC", "D_SafeE"]
DIM_LABELS = {
    "D_Trust": "Trustfulness",
    "D_Fair": "Fairness",
    "D_CI": "Context Integrity",
    "D_SafeC": "Safety (Concealment)",
    "D_SafeE": "Safety (Exaggeration)",
}


def load_scores(input_path: str) -> pd.DataFrame:
    """Load per-model dimensional scalars from CSV.

    Expected CSV format:
        model, D_Trust, D_Fair, D_CI, D_SafeC, D_SafeE
        Qwen-14B, 0.72, 0.65, 0.82, 0.10, 1.00
        ...
    """
    df = pd.read_csv(input_path)
    df = df.set_index("model")
    missing = [d for d in DIMENSIONS if d not in df.columns]
    if missing:
        raise ValueError(f"Missing dimension columns: {missing}")
    return df[DIMENSIONS]


def compute_dcm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Dimensional Coupling Matrix (Pearson correlation)."""
    return df.corr(method="pearson")


def compute_dcm_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rank-based DCM (Spearman) for robustness check."""
    return df.corr(method="spearman")


def compute_ndi(dcm: pd.DataFrame) -> float:
    """Non-Diagonality Index: NDI = ||DCM - I||_F / sqrt(n^2 - n)."""
    n = dcm.shape[0]
    identity = np.eye(n)
    frobenius = np.linalg.norm(dcm.values - identity, "fro")
    ndi = frobenius / np.sqrt(n**2 - n)
    return ndi


def compute_cts(df: pd.DataFrame, ndi: float) -> pd.DataFrame:
    """Coupled Trustworthiness Score per model.

    CTS(M) = (∏ D_i)^(1/n) × (1 - min(D_i)/mean(D_i) × NDI)
    """
    n = len(DIMENSIONS)
    results = []
    for model, row in df.iterrows():
        values = row.values.astype(float)
        geo_mean = np.prod(values) ** (1.0 / n)
        min_val = values.min()
        mean_val = values.mean()
        penalty = 1.0 - (min_val / mean_val * ndi) if mean_val > 0 else 1.0
        cts = geo_mean * penalty
        limiting_dim = DIMENSIONS[np.argmin(values)]
        results.append({
            "model": model,
            "CTS": round(cts, 4),
            "geo_mean": round(geo_mean, 4),
            "limiting_dim": DIM_LABELS[limiting_dim],
            "limiting_value": round(min_val, 4),
        })
    results_df = pd.DataFrame(results).sort_values("CTS", ascending=False)
    return results_df.reset_index(drop=True)


def pareto_dominance(df: pd.DataFrame) -> list:
    """Find Pareto dominance relations among models."""
    models = df.index.tolist()
    dominance_pairs = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                continue
            v1 = df.loc[m1].values
            v2 = df.loc[m2].values
            # m1 dominates m2 if m1 >= m2 on all dims and m1 > m2 on at least one
            if np.all(v1 >= v2) and np.any(v1 > v2):
                dominance_pairs.append((m1, m2))
    return dominance_pairs


def significance_test(df: pd.DataFrame) -> pd.DataFrame:
    """Compute p-values for each DCM entry."""
    n_models = len(df)
    dims = df.columns.tolist()
    p_matrix = pd.DataFrame(np.ones((len(dims), len(dims))),
                            index=dims, columns=dims)
    for i, d1 in enumerate(dims):
        for j, d2 in enumerate(dims):
            if i != j:
                _, p = stats.pearsonr(df[d1], df[d2])
                p_matrix.loc[d1, d2] = p
            else:
                p_matrix.loc[d1, d2] = 0.0
    return p_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────


def plot_dcm_heatmap(dcm: pd.DataFrame, p_matrix: pd.DataFrame, output_dir: str):
    """Plot DCM as annotated heatmap with significance markers."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    labels = [DIM_LABELS[d] for d in dcm.columns]

    # Create annotation with significance stars
    annot = dcm.round(3).astype(str)
    for i in range(dcm.shape[0]):
        for j in range(dcm.shape[1]):
            if i != j and p_matrix.iloc[i, j] < 0.05:
                annot.iloc[i, j] += "*"

    sns.heatmap(dcm, annot=annot, fmt="", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax,
                xticklabels=labels, yticklabels=labels,
                square=True, linewidths=0.5)
    ax.set_title("Dimensional Coupling Matrix (DCM)\n* p < 0.05", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dcm_heatmap.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "dcm_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_cts_ranking(cts_df: pd.DataFrame, output_dir: str):
    """Bar chart of CTS rankings."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(cts_df)))
    bars = ax.barh(cts_df["model"], cts_df["CTS"], color=colors)
    ax.set_xlabel("Coupled Trustworthiness Score (CTS)")
    ax.set_title("Model Rankings by CTS")
    ax.invert_yaxis()

    for bar, (_, row) in zip(bars, cts_df.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{row["limiting_dim"]} ({row["limiting_value"]:.2f})',
                va="center", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cts_ranking.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "cts_ranking.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_radar(df: pd.DataFrame, output_dir: str):
    """Radar plot of all models across dimensions."""
    categories = [DIM_LABELS[d] for d in DIMENSIONS]
    n_cats = len(categories)
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for model in df.index:
        values = df.loc[model].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=model)
        ax.fill(angles, values, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Model Profiles Across Dimensions", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_profiles.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "radar_profiles.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DCM Analysis for TrustMod-SM")
    parser.add_argument("--input", type=str, default="dcm_scores.csv",
                        help="Path to CSV with per-model dimensional scalars")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results and plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading scores from: {args.input}")
    df = load_scores(args.input)
    print(f"Models: {df.index.tolist()}")
    print(f"Dimensions: {DIMENSIONS}\n")

    # Compute DCM
    dcm = compute_dcm(df)
    dcm_spearman = compute_dcm_spearman(df)
    print("=" * 60)
    print("DIMENSIONAL COUPLING MATRIX (Pearson)")
    print("=" * 60)
    print(dcm.round(3).to_string())
    print()

    # NDI
    ndi_pearson = compute_ndi(dcm)
    ndi_spearman = compute_ndi(dcm_spearman)
    print(f"NDI (Pearson):  {ndi_pearson:.4f}")
    print(f"NDI (Spearman): {ndi_spearman:.4f}")
    print()

    # Significance test
    p_matrix = significance_test(df)
    print("P-value matrix:")
    print(p_matrix.round(4).to_string())
    print()

    # CTS
    cts_df = compute_cts(df, ndi_pearson)
    print("=" * 60)
    print("COUPLED TRUSTWORTHINESS SCORE (CTS) RANKINGS")
    print("=" * 60)
    print(cts_df.to_string(index=False))
    print()

    # Pareto dominance
    dominance = pareto_dominance(df)
    print(f"Pareto dominance pairs: {len(dominance)} out of {len(df) * (len(df)-1)} possible")
    for m1, m2 in dominance:
        print(f"  {m1} dominates {m2}")
    print()

    # Plots
    plot_dcm_heatmap(dcm, p_matrix, args.output_dir)
    plot_cts_ranking(cts_df, args.output_dir)
    plot_radar(df, args.output_dir)
    print(f"Visualizations saved to: {args.output_dir}/")

    # Save results
    results = {
        "ndi_pearson": round(ndi_pearson, 4),
        "ndi_spearman": round(ndi_spearman, 4),
        "dcm_pearson": dcm.round(4).to_dict(),
        "dcm_spearman": dcm_spearman.round(4).to_dict(),
        "cts_ranking": cts_df.to_dict(orient="records"),
        "pareto_dominance_pairs": dominance,
        "n_models": len(df),
        "n_dimensions": len(DIMENSIONS),
    }
    with open(os.path.join(args.output_dir, "dcm_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save LaTeX table for CTS
    latex_lines = [
        r"\begin{tabular}{lcl}",
        r"\toprule",
        r"\textbf{Model} & \textbf{CTS} & \textbf{Limiting Dim.} \\",
        r"\midrule",
    ]
    for _, row in cts_df.iterrows():
        latex_lines.append(
            f'{row["model"]} & {row["CTS"]:.3f} & {row["limiting_dim"]} ({row["limiting_value"]:.2f}) \\\\'
        )
    latex_lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(args.output_dir, "cts_table.tex"), "w") as f:
        f.write("\n".join(latex_lines))

    print(f"Results saved to: {args.output_dir}/dcm_results.json")
    print(f"LaTeX table saved to: {args.output_dir}/cts_table.tex")


if __name__ == "__main__":
    main()
