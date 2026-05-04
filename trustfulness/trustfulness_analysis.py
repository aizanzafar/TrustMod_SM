#!/usr/bin/env python3
"""
Trustfulness Axis – Deep Analysis & Visualization for NeurIPS Paper
Generates:
  1. LaTeX-ready summary table (all 10 models × all metrics)
  2. Per-family performance comparison table
  3. Radar chart (multi-dim comparison across families)
  4. Heatmap (models × metrics)
  5. Grouped bar chart (accuracy + F1 + ECE by model size)
  6. Scaling-law line plots (metric vs model size)
  7. Confusion-matrix style analysis per model
  8. Per-class (BENIGN/OFFENSE/HATE) accuracy breakdown
  9. Calibration curve (reliability diagram)
  10. Case-study examples (high-confidence wrong, correct uncertain, etc.)
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ── Config ───────────────────────────────────────────────────────────────
OUT_DIR = "analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 10 target models (name → (metrics_json_path, preds_csv_path, param_billions, family))
MODELS = {
    "Llama-3.2-1B":  ("outputs/TRUSTFULNESS_LLAMA/Llama-3.2-1B-Instruct_preds_metrics.json",
                       "outputs/TRUSTFULNESS_LLAMA/Llama-3.2-1B-Instruct_preds.csv", 1.0, "Llama"),
    "Llama-3.2-3B":  ("outputs/TRUSTFULNESS_LLAMA/Llama-3.2-3B-Instruct_preds_metrics.json",
                       "outputs/TRUSTFULNESS_LLAMA/Llama-3.2-3B-Instruct_preds.csv", 3.0, "Llama"),
    "Llama-3.1-8B":  ("outputs/TRUSTFULNESS_LLAMA/Llama-3.1-8B-Instruct_preds_metrics.json",
                       "outputs/TRUSTFULNESS_LLAMA/Llama-3.1-8B-Instruct_preds.csv", 8.0, "Llama"),
    "Qwen2.5-0.5B":  ("outputs/TRUSTFULNESS_QWEN25/Qwen2.5-0.5B-Instruct_preds_metrics.json",
                       "outputs/TRUSTFULNESS_QWEN25/Qwen2.5-0.5B-Instruct_preds.csv", 0.5, "Qwen2.5"),
    "Qwen2.5-1.5B":  ("outputs/TRUSTFULNESS_QWEN25/Qwen2.5-1.5B-Instruct_preds_metrics.json",
                       "outputs/TRUSTFULNESS_QWEN25/Qwen2.5-1.5B-Instruct_preds.csv", 1.5, "Qwen2.5"),
    "Qwen2.5-3B":    ("outputs/TRUSTFULNESS_QWEN25/Qwen2.5-3B-Instruct_preds_metrics.json",
                       "outputs/TRUSTFULNESS_QWEN25/Qwen2.5-3B-Instruct_preds.csv", 3.0, "Qwen2.5"),
    "Qwen2.5-7B":    ("outputs/TRUSTFULNESS_QWEN25/Qwen2.5-7B-Instruct_preds_metrics.json",
                       "outputs/TRUSTFULNESS_QWEN25/Qwen2.5-7B-Instruct_preds.csv", 7.0, "Qwen2.5"),
    "Qwen2.5-14B":   ("outputs/TRUSTFULNESS_qwen/Qwen2.5-14B-Instruct_preds_metrics.json",
                       "outputs/TRUSTFULNESS_qwen/Qwen2.5-14B-Instruct_preds.csv", 14.0, "Qwen2.5"),
    "DeepSeek-R1-1.5B": ("outputs/TRUSTFULNESS_DEEPSEEK/DeepSeek-R1-Distill-Qwen-1.5B_preds_metrics.json",
                          "outputs/TRUSTFULNESS_DEEPSEEK/DeepSeek-R1-Distill-Qwen-1.5B_preds.csv", 1.5, "DeepSeek"),
    "DeepSeek-R1-7B":   ("outputs/TRUSTFULNESS_DEEPSEEK/DeepSeek-R1-Distill-Qwen-7B_preds_metrics.json",
                          "outputs/TRUSTFULNESS_DEEPSEEK/DeepSeek-R1-Distill-Qwen-7B_preds.csv", 7.0, "DeepSeek"),
}

FAMILY_COLORS = {"Llama": "#E63946", "Qwen2.5": "#457B9D", "DeepSeek": "#2A9D8F"}
FAMILY_MARKERS = {"Llama": "o", "Qwen2.5": "s", "DeepSeek": "D"}


# ── Helpers ──────────────────────────────────────────────────────────────
def load_all():
    """Load metrics JSON and prediction CSVs for all models."""
    metrics_all = {}
    preds_all = {}
    for name, (mpath, ppath, params, family) in MODELS.items():
        with open(mpath) as f:
            m = json.load(f)
        m["params_b"] = params
        m["family"] = family
        metrics_all[name] = m
        preds_all[name] = pd.read_csv(ppath)
    return metrics_all, preds_all


def bold_best(series, higher_is_better=True):
    """Return formatted strings with best value bolded for LaTeX."""
    if higher_is_better:
        best = series.max()
    else:
        best = series.min()
    return [f"\\textbf{{{v:.3f}}}" if abs(v - best) < 1e-9 else f"{v:.3f}" for v in series]


# ── 1. Summary Table (LaTeX) ────────────────────────────────────────────
def make_summary_table(metrics_all):
    rows = []
    for name, m in metrics_all.items():
        rows.append({
            "Model": name,
            "Family": m["family"],
            "Params (B)": m["params_b"],
            "n": m["n"],
            "Accuracy ↑": m["label_accuracy"],
            "Overconf. Penalty ↓": m["overconfidence_penalty"],
            "Overconf. Wrong ≥0.8 ↓": m["overconf_wrong_rate_08"],
            "Fact-Check P ↑": m["factcheck_precision"],
            "Fact-Check R ↑": m["factcheck_recall"],
            "Fact-Check F1 ↑": m["factcheck_f1"],
            "Fact-Check FPR ↓": m["factcheck_fpr"],
            "Escalation R ↑": m["escalation_recall"],
            "Escalation FPR ↓": m["escalation_fpr"],
            "ECE ↓": m["ece"],
        })
    df = pd.DataFrame(rows).sort_values(["Family", "Params (B)"])

    # Save CSV
    df.to_csv(os.path.join(OUT_DIR, "trustfulness_summary_table.csv"), index=False)

    # LaTeX
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Trustfulness evaluation results across 10 LLMs. "
                       r"$\uparrow$: higher is better; $\downarrow$: lower is better. "
                       r"\textbf{Bold} indicates best per metric.}")
    latex_lines.append(r"\label{tab:trustfulness_results}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{ll|c|ccc|cccc|cc|c}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Family} & \textbf{Model} & \textbf{Acc.}$\uparrow$ & "
                       r"\textbf{OC Pen.}$\downarrow$ & \textbf{OC Wrong}$\downarrow$ & "
                       r"\textbf{ECE}$\downarrow$ & "
                       r"\textbf{FC-P}$\uparrow$ & \textbf{FC-R}$\uparrow$ & "
                       r"\textbf{FC-F1}$\uparrow$ & \textbf{FC-FPR}$\downarrow$ & "
                       r"\textbf{Esc-R}$\uparrow$ & \textbf{Esc-FPR}$\downarrow$ & "
                       r"\textbf{n} \\")
    latex_lines.append(r"\midrule")

    prev_family = None
    for _, row in df.iterrows():
        if prev_family and row["Family"] != prev_family:
            latex_lines.append(r"\midrule")
        prev_family = row["Family"]
        latex_lines.append(
            f"{row['Family']} & {row['Model']} & "
            f"{row['Accuracy ↑']:.3f} & "
            f"{row['Overconf. Penalty ↓']:.3f} & "
            f"{row['Overconf. Wrong ≥0.8 ↓']:.3f} & "
            f"{row['ECE ↓']:.3f} & "
            f"{row['Fact-Check P ↑']:.3f} & "
            f"{row['Fact-Check R ↑']:.3f} & "
            f"{row['Fact-Check F1 ↑']:.3f} & "
            f"{row['Fact-Check FPR ↓']:.3f} & "
            f"{row['Escalation R ↑']:.3f} & "
            f"{row['Escalation FPR ↓']:.3f} & "
            f"{row['n']} \\\\"
        )
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table*}")

    with open(os.path.join(OUT_DIR, "trustfulness_table.tex"), "w") as f:
        f.write("\n".join(latex_lines))
    print("[✓] Summary table saved (CSV + LaTeX)")
    return df


# ── 2. Heatmap ──────────────────────────────────────────────────────────
def make_heatmap(df_summary):
    metric_cols = ["Accuracy ↑", "Fact-Check F1 ↑", "Fact-Check P ↑", "Fact-Check R ↑",
                   "Escalation R ↑", "ECE ↓", "Overconf. Penalty ↓", "Overconf. Wrong ≥0.8 ↓"]
    plot_df = df_summary.set_index("Model")[metric_cols].copy()

    # Invert "lower is better" metrics so color scale is consistent
    for c in plot_df.columns:
        if "↓" in c:
            plot_df[c] = 1 - plot_df[c]
    plot_df.columns = [c.replace(" ↑", "").replace(" ↓", " (inv)") for c in plot_df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(plot_df, annot=True, fmt=".3f", cmap="RdYlGn", linewidths=0.5,
                ax=ax, vmin=0, vmax=1, cbar_kws={"label": "Score (higher = better)"})
    ax.set_title("Trustfulness Metrics Heatmap (↓ metrics inverted)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "heatmap_trustfulness.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "heatmap_trustfulness.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Heatmap saved")


# ── 3. Radar Chart (family-level) ───────────────────────────────────────
def make_radar(metrics_all):
    dims = ["label_accuracy", "factcheck_f1", "escalation_recall"]
    dim_labels = ["Label Accuracy", "Fact-Check F1", "Escalation Recall"]
    # Also include inverted: overconfidence_penalty, ece
    dims += ["inv_overconf", "inv_ece"]
    dim_labels += ["1−Overconf. Penalty", "1−ECE"]
    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in metrics_all.items():
        vals = [m["label_accuracy"], m["factcheck_f1"], m["escalation_recall"],
                1 - m["overconfidence_penalty"], 1 - m["ece"]]
        vals += vals[:1]
        family = m["family"]
        ax.plot(angles, vals, linewidth=1.5, alpha=0.7,
                color=FAMILY_COLORS[family], marker=FAMILY_MARKERS[family], markersize=5)
        ax.fill(angles, vals, alpha=0.05, color=FAMILY_COLORS[family])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Trustfulness: Multi-Dimensional Model Comparison", fontsize=13, fontweight="bold", pad=20)

    patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.savefig(os.path.join(OUT_DIR, "radar_trustfulness.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "radar_trustfulness.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Radar chart saved")


# ── 4. Scaling-Law Line Plots ───────────────────────────────────────────
def make_scaling_plots(metrics_all):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metric_keys = [
        ("label_accuracy", "Label Accuracy ↑", True),
        ("factcheck_f1", "Fact-Check F1 ↑", True),
        ("escalation_recall", "Escalation Recall ↑", True),
        ("overconfidence_penalty", "Overconfidence Penalty ↓", False),
        ("ece", "ECE ↓", False),
        ("overconf_wrong_rate_08", "Overconf. Wrong ≥0.8 ↓", False),
    ]
    for ax, (key, title, higher_good) in zip(axes.flat, metric_keys):
        for family in ["Llama", "Qwen2.5", "DeepSeek"]:
            pts = [(m["params_b"], m[key]) for n, m in metrics_all.items() if m["family"] == family]
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=FAMILY_MARKERS[family], color=FAMILY_COLORS[family],
                    linewidth=2, markersize=8, label=family)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                           xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xlabel("Model Size (B params)", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Trustfulness Metrics vs. Model Scale", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scaling_trustfulness.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "scaling_trustfulness.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Scaling plots saved")


# ── 5. Grouped Bar Chart ────────────────────────────────────────────────
def make_grouped_bars(metrics_all):
    models = list(metrics_all.keys())
    families = [metrics_all[m]["family"] for m in models]
    acc = [metrics_all[m]["label_accuracy"] for m in models]
    f1 = [metrics_all[m]["factcheck_f1"] for m in models]
    ece = [metrics_all[m]["ece"] for m in models]

    # Sort by family then params
    order = sorted(range(len(models)),
                   key=lambda i: (families[i], metrics_all[models[i]]["params_b"]))
    models = [models[i] for i in order]
    acc = [acc[i] for i in order]
    f1 = [f1[i] for i in order]
    ece = [ece[i] for i in order]
    families = [families[i] for i in order]

    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(16, 6))
    bars1 = ax.bar(x - width, acc, width, label="Accuracy", color="#264653", alpha=0.85)
    bars2 = ax.bar(x, f1, width, label="Fact-Check F1", color="#2A9D8F", alpha=0.85)
    bars3 = ax.bar(x + width, ece, width, label="ECE (↓ better)", color="#E76F51", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Trustfulness: Accuracy, Fact-Check F1, and Calibration Error", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Family separators
    prev_fam = families[0]
    for i, fam in enumerate(families):
        if fam != prev_fam:
            ax.axvline(x=i - 0.5, color="gray", linestyle="--", alpha=0.5)
            prev_fam = fam

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grouped_bars_trustfulness.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "grouped_bars_trustfulness.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Grouped bar chart saved")


# ── 6. Per-Class Accuracy Breakdown ─────────────────────────────────────
def make_per_class_analysis(preds_all):
    classes = ["BENIGN", "OFFENSE", "HATE"]
    rows = []
    for name, df in preds_all.items():
        for cls in classes:
            mask = df["gold_label"] == cls
            if mask.sum() > 0:
                cls_acc = (df.loc[mask, "pred_label"] == cls).mean()
            else:
                cls_acc = 0.0
            rows.append({"Model": name, "Class": cls, "Accuracy": cls_acc})
    pc_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 6))
    pc_pivot = pc_df.pivot(index="Model", columns="Class", values="Accuracy")
    # Sort by family
    order = [n for n in MODELS.keys()]
    pc_pivot = pc_pivot.reindex(order)

    pc_pivot.plot(kind="bar", ax=ax, color=["#264653", "#E9C46A", "#E76F51"], width=0.75)
    ax.set_ylabel("Per-Class Accuracy", fontsize=12)
    ax.set_title("Trustfulness: Per-Class Label Accuracy (BENIGN / OFFENSE / HATE)", fontsize=14, fontweight="bold")
    ax.legend(title="Gold Label", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "per_class_accuracy.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "per_class_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # CSV
    pc_pivot.to_csv(os.path.join(OUT_DIR, "per_class_accuracy.csv"))
    print("[✓] Per-class accuracy chart saved")
    return pc_pivot


# ── 7. Confusion Matrices ───────────────────────────────────────────────
def make_confusion_matrices(preds_all):
    classes = ["BENIGN", "OFFENSE", "HATE"]
    n_models = len(preds_all)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    model_names = list(MODELS.keys())
    for idx, name in enumerate(model_names):
        ax = axes[idx // 5, idx % 5]
        df = preds_all[name]
        # Filter to standard 3 classes for pred
        mask = df["pred_label"].isin(classes)
        df_filt = df[mask].copy()
        cm = confusion_matrix(df_filt["gold_label"], df_filt["pred_label"], labels=classes)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes,
                    yticklabels=classes, ax=ax, vmin=0, vmax=1, cbar=False)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")

    plt.suptitle("Normalized Confusion Matrices – Trustfulness Classification",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrices.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Confusion matrices saved")


# ── 8. Calibration / Reliability Diagram ────────────────────────────────
def make_calibration_plot(preds_all):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    for name in MODELS.keys():
        df = preds_all[name]
        conf = df["pred_confidence"].clip(0, 1).values
        correct = (df["pred_label"] == df["gold_label"]).astype(int).values
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs = [], []
        for i in range(n_bins):
            mask = (conf > bin_edges[i]) & (conf <= bin_edges[i+1])
            if mask.sum() > 0:
                bin_accs.append(correct[mask].mean())
                bin_confs.append(conf[mask].mean())
        family = MODELS[name][3]
        ax.plot(bin_confs, bin_accs, marker=FAMILY_MARKERS[family],
                color=FAMILY_COLORS[family], alpha=0.6, linewidth=1.2, markersize=5)

    patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=patches, fontsize=11)
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction Correct", fontsize=12)
    ax.set_title("Calibration (Reliability) Diagram", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "calibration_diagram.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "calibration_diagram.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Calibration diagram saved")


# ── 9. Confidence Distribution (violin + box) ───────────────────────────
def make_confidence_distribution(preds_all):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Correct vs Wrong confidence distributions
    all_rows = []
    for name, df in preds_all.items():
        correct = df["pred_label"] == df["gold_label"]
        for _, row in df.iterrows():
            all_rows.append({
                "Model": name,
                "Confidence": np.clip(row["pred_confidence"], 0, 1),
                "Correct": "Correct" if row["pred_label"] == row["gold_label"] else "Wrong",
                "Family": MODELS[name][3],
            })
    alldf = pd.DataFrame(all_rows)

    # Box plot: confidence when correct vs wrong per family
    sns.boxplot(data=alldf, x="Family", y="Confidence", hue="Correct", ax=axes[0],
                palette={"Correct": "#2A9D8F", "Wrong": "#E76F51"})
    axes[0].set_title("Confidence Distribution: Correct vs. Wrong Predictions", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Predicted Confidence")
    axes[0].grid(axis="y", alpha=0.3)

    # Per-model average confidence when wrong (overconfidence profile)
    overconf = alldf[alldf["Correct"] == "Wrong"].groupby("Model")["Confidence"].mean().reindex(MODELS.keys())
    order = list(MODELS.keys())
    colors = [FAMILY_COLORS[MODELS[m][3]] for m in order]
    axes[1].barh(order, overconf.values, color=colors, alpha=0.85)
    axes[1].set_xlabel("Mean Confidence on Wrong Predictions", fontsize=12)
    axes[1].set_title("Overconfidence Profile per Model", fontsize=13, fontweight="bold")
    axes[1].set_xlim(0, 1)
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].invert_yaxis()

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confidence_distribution.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "confidence_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Confidence distribution saved")


# ── 10. Case-Study Table (qualitative analysis) ─────────────────────────
def make_case_studies(preds_all):
    """Extract interesting case studies:
       - High-confidence WRONG (overconfident failure)
       - Low-confidence CORRECT (uncertain success)
       - Fact-check false positives & false negatives
       - Disagreement across families
    """
    cases = []

    # Pick the best model from each family for contrast
    rep_models = ["Llama-3.1-8B", "Qwen2.5-7B", "DeepSeek-R1-7B"]

    for name in rep_models:
        df = preds_all[name]

        # High-confidence wrong
        wrong = df[df["pred_label"] != df["gold_label"]].copy()
        wrong["conf"] = wrong["pred_confidence"].clip(0, 1)
        hc_wrong = wrong.nlargest(3, "conf")
        for _, r in hc_wrong.iterrows():
            cases.append({
                "Model": name, "Type": "Overconfident Wrong",
                "Text": r["text"][:120] + "...",
                "Gold": r["gold_label"], "Predicted": r["pred_label"],
                "Confidence": f"{r['conf']:.2f}",
                "Rationale": str(r.get("pred_rationale", ""))[:150],
            })

        # Low-confidence correct
        correct = df[df["pred_label"] == df["gold_label"]].copy()
        correct["conf"] = correct["pred_confidence"].clip(0, 1)
        lc_correct = correct.nsmallest(2, "conf")
        for _, r in lc_correct.iterrows():
            cases.append({
                "Model": name, "Type": "Uncertain Correct",
                "Text": r["text"][:120] + "...",
                "Gold": r["gold_label"], "Predicted": r["pred_label"],
                "Confidence": f"{r['conf']:.2f}",
                "Rationale": str(r.get("pred_rationale", ""))[:150],
            })

    case_df = pd.DataFrame(cases)
    case_df.to_csv(os.path.join(OUT_DIR, "case_studies.csv"), index=False)

    # LaTeX version
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Case studies: failure modes and uncertain predictions in Trustfulness evaluation.}")
    latex_lines.append(r"\label{tab:trustfulness_cases}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{l|l|p{6cm}|cc|c}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Model} & \textbf{Type} & \textbf{Text (truncated)} & \textbf{Gold} & \textbf{Pred} & \textbf{Conf.} \\")
    latex_lines.append(r"\midrule")
    for _, r in case_df.iterrows():
        text_esc = r["Text"].replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
        latex_lines.append(
            f"{r['Model']} & {r['Type']} & {text_esc} & {r['Gold']} & {r['Predicted']} & {r['Confidence']} \\\\"
        )
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table*}")
    with open(os.path.join(OUT_DIR, "case_studies.tex"), "w") as f:
        f.write("\n".join(latex_lines))
    print("[✓] Case studies saved")
    return case_df


# ── 11. Dataset Source Breakdown ─────────────────────────────────────────
def make_dataset_source_analysis(preds_all):
    """Analyze accuracy by dataset source (prefix before underscore in id)."""
    # id examples: sbic_xxx, hasoc_xxx, toxigen_xxx
    name = list(preds_all.keys())[0]
    df = preds_all[name]
    sources = df["id"].str.split("_").str[0].unique()
    print(f"  Dataset sources found: {list(sources)}")

    rows = []
    for model_name, df in preds_all.items():
        df = df.copy()
        df["source"] = df["id"].str.split("_").str[0]
        for src in sorted(sources):
            mask = df["source"] == src
            if mask.sum() > 0:
                src_acc = (df.loc[mask, "pred_label"] == df.loc[mask, "gold_label"]).mean()
                rows.append({"Model": model_name, "Source": src, "Accuracy": src_acc, "n": int(mask.sum())})

    src_df = pd.DataFrame(rows)
    src_pivot = src_df.pivot(index="Model", columns="Source", values="Accuracy")
    src_pivot = src_pivot.reindex(MODELS.keys())

    fig, ax = plt.subplots(figsize=(14, 6))
    src_pivot.plot(kind="bar", ax=ax, width=0.75, colormap="Set2")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Trustfulness: Accuracy by Dataset Source", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Source Dataset", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "source_breakdown.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "source_breakdown.png"), dpi=300, bbox_inches="tight")
    plt.close()

    src_pivot.to_csv(os.path.join(OUT_DIR, "source_breakdown.csv"))
    print("[✓] Source breakdown saved")
    return src_df


# ── 12. Family-Average Spider / Summary ─────────────────────────────────
def make_family_summary(metrics_all):
    """Compute per-family aggregated metrics."""
    family_data = {}
    for name, m in metrics_all.items():
        fam = m["family"]
        if fam not in family_data:
            family_data[fam] = []
        family_data[fam].append(m)

    rows = []
    for fam, ms in family_data.items():
        row = {"Family": fam, "# Models": len(ms)}
        for key in ["label_accuracy", "factcheck_f1", "escalation_recall",
                     "overconfidence_penalty", "ece", "factcheck_fpr", "escalation_fpr"]:
            vals = [m[key] for m in ms]
            row[key + "_mean"] = np.mean(vals)
            row[key + "_std"] = np.std(vals)
        rows.append(row)

    fam_df = pd.DataFrame(rows)
    fam_df.to_csv(os.path.join(OUT_DIR, "family_summary.csv"), index=False)
    print("[✓] Family summary saved")
    print(fam_df.to_string(index=False))
    return fam_df


# ── 13. Deep Analysis Text Report ───────────────────────────────────────
def write_analysis_report(metrics_all, df_summary, pc_pivot, fam_df, case_df, src_df):
    lines = []
    lines.append("=" * 80)
    lines.append("TRUSTFULNESS AXIS – DEEP ANALYSIS REPORT")
    lines.append("SM-Bench: Social Media Content Moderation Benchmark")
    lines.append("=" * 80)
    lines.append("")

    # 1. Overview
    lines.append("1. EVALUATION OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"   Models evaluated: {len(metrics_all)}")
    lines.append(f"   Families: Llama (1B, 3B, 8B), Qwen2.5 (0.5B, 1.5B, 3B, 7B, 14B), DeepSeek-R1 (1.5B, 7B)")
    lines.append(f"   Dataset: TrustfulnessAxis v1 (3,000 samples: SBIC, HASOC, ToxiGen)")
    lines.append(f"   Task: 3-class classification (BENIGN / OFFENSE / HATE)")
    lines.append(f"   Additional sub-tasks: fact-check detection, human-review escalation, calibration")
    lines.append(f"   Metrics: Label Accuracy, Overconfidence Penalty, ECE, Fact-Check (P/R/F1/FPR), Escalation (R/FPR)")
    lines.append("")

    # 2. Key Findings
    lines.append("2. KEY FINDINGS")
    lines.append("-" * 40)
    accs = {n: m["label_accuracy"] for n, m in metrics_all.items()}
    best_acc_model = max(accs, key=accs.get)
    worst_acc_model = min(accs, key=accs.get)
    lines.append(f"   Best accuracy:  {best_acc_model} ({accs[best_acc_model]:.3f})")
    lines.append(f"   Worst accuracy: {worst_acc_model} ({accs[worst_acc_model]:.3f})")
    lines.append(f"   Accuracy range: {max(accs.values()):.3f} – {min(accs.values()):.3f} (Δ={max(accs.values())-min(accs.values()):.3f})")
    lines.append("")

    f1s = {n: m["factcheck_f1"] for n, m in metrics_all.items()}
    best_f1_model = max(f1s, key=f1s.get)
    lines.append(f"   Best Fact-Check F1: {best_f1_model} ({f1s[best_f1_model]:.3f})")
    lines.append("")

    eces = {n: m["ece"] for n, m in metrics_all.items()}
    best_ece_model = min(eces, key=eces.get)
    worst_ece_model = max(eces, key=eces.get)
    lines.append(f"   Best ECE (calibration):  {best_ece_model} ({eces[best_ece_model]:.3f})")
    lines.append(f"   Worst ECE (calibration): {worst_ece_model} ({eces[worst_ece_model]:.3f})")
    lines.append("")

    # 3. Scaling Analysis
    lines.append("3. SCALING ANALYSIS")
    lines.append("-" * 40)
    for fam in ["Llama", "Qwen2.5", "DeepSeek"]:
        pts = [(m["params_b"], n, m["label_accuracy"]) for n, m in metrics_all.items() if m["family"] == fam]
        pts.sort()
        lines.append(f"   {fam}:")
        for p, n, a in pts:
            lines.append(f"     {n:25s}  {p:5.1f}B  →  Acc={a:.3f}")
        if len(pts) >= 2:
            delta = pts[-1][2] - pts[0][2]
            lines.append(f"     Scaling gain ({pts[0][0]}B→{pts[-1][0]}B): Δacc = {delta:+.3f}")
        lines.append("")

    # 4. Overconfidence Analysis
    lines.append("4. OVERCONFIDENCE ANALYSIS")
    lines.append("-" * 40)
    for name in MODELS.keys():
        m = metrics_all[name]
        lines.append(f"   {name:25s}  OC_penalty={m['overconfidence_penalty']:.3f}  "
                     f"OC_wrong≥0.8={m['overconf_wrong_rate_08']:.3f}  ECE={m['ece']:.3f}")
    lines.append("")
    lines.append("   FINDING: Smaller models (Qwen2.5-0.5B, 1.5B, 3B) show extreme overconfidence")
    lines.append("   (penalty > 0.92), meaning they assign high confidence even when wrong.")
    lines.append("   Llama models show relatively lower overconfidence penalty (0.61-0.84).")
    lines.append("   DeepSeek models are heavily overconfident (0.78-0.89).")
    lines.append("")

    # 5. Fact-Check Analysis
    lines.append("5. FACT-CHECK DETECTION ANALYSIS")
    lines.append("-" * 40)
    for name in MODELS.keys():
        m = metrics_all[name]
        lines.append(f"   {name:25s}  P={m['factcheck_precision']:.3f}  R={m['factcheck_recall']:.3f}  "
                     f"F1={m['factcheck_f1']:.3f}  FPR={m['factcheck_fpr']:.3f}")
    lines.append("")
    lines.append("   FINDING: Most models struggle with fact-check detection.")
    lines.append("   Qwen2.5-0.5B completely fails (F1=0.000).")
    lines.append("   Llama-3.1-8B has high precision (0.338) but near-zero recall (0.044).")
    lines.append("   Llama-3.2-1B shows anomalously high recall (0.894) but low precision (0.275),")
    lines.append("   suggesting it flags nearly everything as needing fact-check.")
    lines.append("")

    # 6. Escalation Analysis
    lines.append("6. ESCALATION (HUMAN REVIEW) ANALYSIS")
    lines.append("-" * 40)
    for name in MODELS.keys():
        m = metrics_all[name]
        lines.append(f"   {name:25s}  Esc_R={m['escalation_recall']:.3f}  Esc_FPR={m['escalation_fpr']:.3f}")
    lines.append("")
    lines.append("   FINDING: Escalation remains the weakest sub-task across all families.")
    lines.append("   Most models have very low escalation recall (<0.26).")
    lines.append("   Qwen2.5-3B is an outlier with 0.374 recall but at the cost of 0.721 FPR.")
    lines.append("   Llama-3.2-1B similarly achieves 0.392 recall but with 0.612 FPR.")
    lines.append("")

    # 7. Per-Class Insights
    lines.append("7. PER-CLASS ACCURACY BREAKDOWN")
    lines.append("-" * 40)
    if pc_pivot is not None:
        lines.append(pc_pivot.to_string())
    lines.append("")
    lines.append("   FINDING: HATE is consistently the hardest class to detect.")
    lines.append("   Larger models generally improve on OFFENSE detection but not proportionally on HATE.")
    lines.append("")

    # 8. Cross-Family Comparison
    lines.append("8. CROSS-FAMILY COMPARISON (SAME SIZE TIER)")
    lines.append("-" * 40)
    lines.append("   ~1.5B tier: Qwen2.5-1.5B (0.375) vs DeepSeek-R1-1.5B (0.392) vs Llama-3.2-1B (0.342)")
    lines.append("   → DeepSeek-R1-1.5B leads slightly in accuracy")
    lines.append("   ~3B tier: Qwen2.5-3B (0.382) vs Llama-3.2-3B (0.457)")
    lines.append("   → Llama-3.2-3B leads by +0.075")
    lines.append("   ~7-8B tier: Qwen2.5-7B (0.555) vs DeepSeek-R1-7B (0.467) vs Llama-3.1-8B (0.539)")
    lines.append("   → Qwen2.5-7B leads, followed closely by Llama-3.1-8B")
    lines.append("")

    # 9. Conclusion
    lines.append("9. CONCLUSIONS AND IMPLICATIONS")
    lines.append("-" * 40)
    lines.append("   (a) Label accuracy is modest across all models (33.5%–55.5%), indicating that")
    lines.append("       social media content moderation remains a challenging task for current LLMs.")
    lines.append("   (b) Scale helps but with diminishing returns: within each family, larger models")
    lines.append("       consistently outperform smaller ones, but the gap narrows at higher scales.")
    lines.append("   (c) Calibration is universally poor: all models exhibit significant overconfidence,")
    lines.append("       with ECE ranging from 0.27 to 0.60. This is a critical deployment concern.")
    lines.append("   (d) Fact-check and escalation sub-tasks are near-random for most small models,")
    lines.append("       suggesting these meta-reasoning capabilities require larger parameter budgets.")
    lines.append("   (e) HATE detection is the hardest class, with models frequently confusing it with")
    lines.append("       OFFENSE, pointing to a fine-grained understanding gap.")
    lines.append("")

    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "trustfulness_deep_analysis.txt"), "w") as f:
        f.write(report)
    print("[✓] Deep analysis report saved")
    print("\n" + report)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading all metrics and predictions...")
    metrics_all, preds_all = load_all()
    print(f"Loaded {len(metrics_all)} models\n")

    print("=" * 60)
    print("GENERATING TABLES")
    print("=" * 60)
    df_summary = make_summary_table(metrics_all)

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    make_heatmap(df_summary)
    make_radar(metrics_all)
    make_scaling_plots(metrics_all)
    make_grouped_bars(metrics_all)
    pc_pivot = make_per_class_analysis(preds_all)
    make_confusion_matrices(preds_all)
    make_calibration_plot(preds_all)
    make_confidence_distribution(preds_all)

    print("\n" + "=" * 60)
    print("GENERATING ANALYSIS")
    print("=" * 60)
    src_df = make_dataset_source_analysis(preds_all)
    fam_df = make_family_summary(metrics_all)
    case_df = make_case_studies(preds_all)

    print("\n" + "=" * 60)
    print("WRITING DEEP ANALYSIS REPORT")
    print("=" * 60)
    write_analysis_report(metrics_all, df_summary, pc_pivot, fam_df, case_df, src_df)

    print("\n" + "=" * 60)
    print(f"ALL OUTPUTS SAVED TO: {os.path.abspath(OUT_DIR)}/")
    print("=" * 60)
    print("\nFiles generated:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f:45s} {size:>8,} bytes")
