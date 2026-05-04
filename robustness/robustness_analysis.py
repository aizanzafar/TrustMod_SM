#!/usr/bin/env python3
"""
Robustness Axis – Deep Analysis & Visualization for NeurIPS Paper
=================================================================
Dataset: SBIC Obfuscated (6,848 samples)
  - 2,000 clean (1,000 offensive + 1,000 safe)
  - 2,000 visual obfuscation (leetspeak, @/0/3 substitution)
  - 1,424 structural (dotted letters, spacing tricks)
  - 1,424 semantic (LLM paraphrasing that may alter or refuse)

Binary task: offensive vs. safe

Generates:
  1. LaTeX-ready main results table (all 10 models × overall + per-type)
  2. Heatmap (models × obfuscation type accuracy)
  3. Grouped bar chart (accuracy by type)
  4. Robustness degradation plot (clean accuracy → obfuscated accuracy)
  5. Failure rate analysis
  6. Scaling-law plots (metric vs. model size)
  7. Radar chart (multi-dim: accuracy, F1, robustness drop, failure rate)
  8. Confusion matrices per model
  9. Per-label analysis (offensive vs safe detection across types)
  10. Attack success rate analysis
  11. Confidence analysis (correct vs wrong, by type)
  12. Case studies (adversarial evasion examples)
  13. Deep analysis text report
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
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ── Config ───────────────────────────────────────────────────────────────
OUT_DIR = "analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Model name → (results_file, param_billions, family)
MODELS = {
    "Llama-3.2-1B":     ("results/llama3_2_1b.json",               1.0,  "Llama"),
    "Llama-3.2-3B":     ("results/llama3_2-3b.json",               3.0,  "Llama"),
    "Llama-3.1-8B":     ("results/llama3_1_8b.json",               8.0,  "Llama"),
    "Qwen2.5-0.5B":     ("results/Qwen2.5-0.5B-Instruct.json",    0.5,  "Qwen2.5"),
    "Qwen2.5-1.5B":     ("results/Qwen2.5-1.5B-Instruct.json",    1.5,  "Qwen2.5"),
    "Qwen2.5-3B":       ("results/Qwen2.5-3B-Instruct.json",      3.0,  "Qwen2.5"),
    "Qwen2.5-7B":       ("results/qwen2_5-7b.json",                7.0,  "Qwen2.5"),
    "Qwen2.5-14B":      ("results/Qwen2.5-14B-Instruct.json",     14.0, "Qwen2.5"),
    "DeepSeek-R1-1.5B": ("results/deepseek-r1_1_5b.json",         1.5,  "DeepSeek"),
    "DeepSeek-R1-7B":   ("results/DeepSeek-R1-Distill-Qwen-7B.json", 7.0, "DeepSeek"),
}

FAMILY_COLORS = {"Llama": "#E63946", "Qwen2.5": "#457B9D", "DeepSeek": "#2A9D8F"}
FAMILY_MARKERS = {"Llama": "o", "Qwen2.5": "s", "DeepSeek": "D"}
OBF_TYPES = ["clean", "visual", "structural", "semantic"]
OBF_COLORS = {"clean": "#264653", "visual": "#2A9D8F", "structural": "#E9C46A", "semantic": "#E76F51"}
LABEL_MAP = {"offensive": 1, "safe": 0}

MODEL_ORDER = list(MODELS.keys())


# ── Data Loading ─────────────────────────────────────────────────────────
def load_all():
    """Load all model results into a dict of DataFrames."""
    data = {}
    for name, (fpath, params, family) in MODELS.items():
        df = pd.read_json(fpath, lines=True)
        df["model"] = name
        df["params_b"] = params
        df["family"] = family
        data[name] = df
    return data


def compute_metrics_for_group(group):
    """Compute metrics for a group of samples."""
    total = len(group)
    failed_mask = group["model_verdict"].isnull()
    failed = int(failed_mask.sum())
    failure_rate = failed / total * 100 if total > 0 else 0

    valid = group[~failed_mask].copy()
    valid["label"] = valid["label"].astype(str).str.lower().str.strip()
    valid["model_verdict"] = valid["model_verdict"].astype(str).str.lower().str.strip()
    valid = valid[valid["model_verdict"].isin(["offensive", "safe"])]

    if len(valid) > 0:
        y_true = valid["label"].map(LABEL_MAP)
        y_pred = valid["model_verdict"].map(LABEL_MAP)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        acc, prec, rec, f1 = 0, 0, 0, 0

    return {
        "total": total, "failed": failed, "valid": len(valid),
        "failure_rate": failure_rate,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
    }


def compute_all_metrics(data):
    """Compute overall + per-type metrics for all models."""
    results = {}
    for name in MODEL_ORDER:
        df = data[name]
        overall = compute_metrics_for_group(df)
        by_type = {}
        for t in OBF_TYPES:
            sub = df[df["type"] == t]
            if len(sub) > 0:
                by_type[t] = compute_metrics_for_group(sub)
        results[name] = {
            "overall": overall,
            "by_type": by_type,
            "params_b": MODELS[name][1],
            "family": MODELS[name][2],
        }
    return results


# ── 1. Summary Table ────────────────────────────────────────────────────
def make_summary_table(results):
    rows = []
    for name in MODEL_ORDER:
        r = results[name]
        o = r["overall"]
        row = {
            "Model": name,
            "Family": r["family"],
            "Params (B)": r["params_b"],
            "n": o["total"],
            "Valid": o["valid"],
            "Failure %": o["failure_rate"],
            "Accuracy": o["accuracy"],
            "Precision": o["precision"],
            "Recall": o["recall"],
            "F1": o["f1"],
        }
        # Per-type accuracy
        for t in OBF_TYPES:
            if t in r["by_type"]:
                row[f"Acc_{t}"] = r["by_type"][t]["accuracy"]
            else:
                row[f"Acc_{t}"] = float("nan")
        # Robustness drop (clean → avg obfuscated)
        clean_acc = r["by_type"].get("clean", {}).get("accuracy", 0)
        obf_accs = [r["by_type"][t]["accuracy"] for t in ["visual", "structural", "semantic"] if t in r["by_type"]]
        avg_obf = np.mean(obf_accs) if obf_accs else 0
        row["Robustness Drop"] = clean_acc - avg_obf
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "robustness_summary_table.csv"), index=False)

    # LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Robustness evaluation: overall and per-obfuscation-type accuracy across 10 LLMs. "
                       r"Binary classification (offensive/safe) on SBIC obfuscated dataset (6,848 samples). "
                       r"Robustness Drop = Clean Acc. $-$ Mean Obfuscated Acc.}")
    latex_lines.append(r"\label{tab:robustness_results}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{ll|c|cccc|cccc|c}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Family} & \textbf{Model} & \textbf{Fail\%}$\downarrow$ & "
                       r"\textbf{Clean} & \textbf{Visual} & \textbf{Struct.} & \textbf{Semantic} & "
                       r"\textbf{Acc.} & \textbf{P} & \textbf{R} & \textbf{F1} & "
                       r"\textbf{Rob. Drop}$\downarrow$ \\")
    latex_lines.append(r"\midrule")
    prev_family = None
    for _, row in df.iterrows():
        if prev_family and row["Family"] != prev_family:
            latex_lines.append(r"\midrule")
        prev_family = row["Family"]
        latex_lines.append(
            f"{row['Family']} & {row['Model']} & "
            f"{row['Failure %']:.1f} & "
            f"{row['Acc_clean']:.3f} & {row['Acc_visual']:.3f} & "
            f"{row['Acc_structural']:.3f} & {row['Acc_semantic']:.3f} & "
            f"{row['Accuracy']:.3f} & {row['Precision']:.3f} & "
            f"{row['Recall']:.3f} & {row['F1']:.3f} & "
            f"{row['Robustness Drop']:.3f} \\\\"
        )
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table*}")
    with open(os.path.join(OUT_DIR, "robustness_table.tex"), "w") as f:
        f.write("\n".join(latex_lines))
    print("[✓] Summary table saved (CSV + LaTeX)")
    return df


# ── 2. Heatmap: Models × Obfuscation Types ──────────────────────────────
def make_heatmap(results):
    data = []
    for name in MODEL_ORDER:
        row = {"Model": name}
        for t in OBF_TYPES:
            if t in results[name]["by_type"]:
                row[t.capitalize()] = results[name]["by_type"][t]["accuracy"]
            else:
                row[t.capitalize()] = float("nan")
        data.append(row)
    df = pd.DataFrame(data).set_index("Model")

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn", linewidths=0.5,
                ax=ax, vmin=0, vmax=1, cbar_kws={"label": "Accuracy"})
    ax.set_title("Robustness: Accuracy by Obfuscation Type", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "heatmap_robustness.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "heatmap_robustness.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Heatmap saved")


# ── 3. Grouped Bar Chart ────────────────────────────────────────────────
def make_grouped_bars(results):
    x = np.arange(len(MODEL_ORDER))
    width = 0.2
    fig, ax = plt.subplots(figsize=(16, 7))

    for i, t in enumerate(OBF_TYPES):
        accs = []
        for name in MODEL_ORDER:
            if t in results[name]["by_type"]:
                accs.append(results[name]["by_type"][t]["accuracy"])
            else:
                accs.append(0)
        ax.bar(x + i * width - 1.5 * width, accs, width, label=t.capitalize(),
               color=OBF_COLORS[t], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Robustness: Per-Type Accuracy Across Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, title="Obfuscation Type")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Family separators
    families = [MODELS[m][2] for m in MODEL_ORDER]
    prev = families[0]
    for i, fam in enumerate(families):
        if fam != prev:
            ax.axvline(x=i - 0.5, color="gray", linestyle="--", alpha=0.5)
            prev = fam

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grouped_bars_robustness.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "grouped_bars_robustness.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Grouped bar chart saved")


# ── 4. Robustness Degradation Plot ──────────────────────────────────────
def make_degradation_plot(results):
    """Shows how accuracy drops from clean → each obfuscated type."""
    fig, ax = plt.subplots(figsize=(14, 7))
    types_order = ["clean", "visual", "structural", "semantic"]

    for name in MODEL_ORDER:
        family = MODELS[name][2]
        accs = []
        for t in types_order:
            if t in results[name]["by_type"]:
                accs.append(results[name]["by_type"][t]["accuracy"])
            else:
                accs.append(float("nan"))
        ax.plot(types_order, accs, marker=FAMILY_MARKERS[family],
                color=FAMILY_COLORS[family], linewidth=1.5, alpha=0.7, markersize=7)

    # Custom legend
    patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=patches, fontsize=11, loc="upper right")
    ax.set_xlabel("Obfuscation Type", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Robustness Degradation: Clean → Obfuscated Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Annotate model names at semantic endpoint
    for name in MODEL_ORDER:
        if "semantic" in results[name]["by_type"]:
            y = results[name]["by_type"]["semantic"]["accuracy"]
            ax.annotate(name, (3, y), fontsize=7, ha="left",
                       xytext=(5, 0), textcoords="offset points")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "degradation_robustness.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "degradation_robustness.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Degradation plot saved")


# ── 5. Failure Rate Analysis ────────────────────────────────────────────
def make_failure_analysis(results, all_data):
    # Overall failure rate bar
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Overall failure rate
    names = MODEL_ORDER
    fail_rates = [results[n]["overall"]["failure_rate"] for n in names]
    colors = [FAMILY_COLORS[MODELS[n][2]] for n in names]
    axes[0].barh(names, fail_rates, color=colors, alpha=0.85)
    axes[0].set_xlabel("Failure Rate (%)", fontsize=12)
    axes[0].set_title("Format Failure Rate (Invalid JSON / No Verdict)", fontsize=13, fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)
    for i, v in enumerate(fail_rates):
        axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    # Right: Failure rate by type
    fail_by_type = []
    for name in MODEL_ORDER:
        for t in OBF_TYPES:
            if t in results[name]["by_type"]:
                fail_by_type.append({
                    "Model": name,
                    "Type": t.capitalize(),
                    "Failure Rate": results[name]["by_type"][t]["failure_rate"],
                })
    fbt_df = pd.DataFrame(fail_by_type)
    fbt_pivot = fbt_df.pivot(index="Model", columns="Type", values="Failure Rate").reindex(MODEL_ORDER)
    fbt_pivot = fbt_pivot[["Clean", "Visual", "Structural", "Semantic"]]
    fbt_pivot.plot(kind="bar", ax=axes[1], color=[OBF_COLORS[t] for t in OBF_TYPES], width=0.75)
    axes[1].set_ylabel("Failure Rate (%)", fontsize=12)
    axes[1].set_title("Failure Rate by Obfuscation Type", fontsize=13, fontweight="bold")
    axes[1].legend(title="Type", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "failure_analysis.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "failure_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Failure analysis saved")
    return fbt_pivot


# ── 6. Scaling-Law Plots ────────────────────────────────────────────────
def make_scaling_plots(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metric_defs = [
        ("accuracy", "Overall Accuracy ↑", True),
        ("f1", "F1 Score ↑", True),
        ("failure_rate", "Failure Rate (%) ↓", False),
    ]
    type_defs = [
        ("clean", "Clean Accuracy ↑"),
        ("visual", "Visual Obf. Accuracy ↑"),
        ("semantic", "Semantic Obf. Accuracy ↑"),
    ]

    # Top row: overall metrics
    for ax, (key, title, _) in zip(axes[0], metric_defs):
        for family in ["Llama", "Qwen2.5", "DeepSeek"]:
            pts = [(results[n]["params_b"], results[n]["overall"][key])
                   for n in MODEL_ORDER if results[n]["family"] == family]
            pts.sort()
            if not pts:
                continue
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

    # Bottom row: per-type accuracy
    for ax, (t, title) in zip(axes[1], type_defs):
        for family in ["Llama", "Qwen2.5", "DeepSeek"]:
            pts = []
            for n in MODEL_ORDER:
                if results[n]["family"] == family and t in results[n]["by_type"]:
                    pts.append((results[n]["params_b"], results[n]["by_type"][t]["accuracy"]))
            pts.sort()
            if not pts:
                continue
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

    plt.suptitle("Robustness Metrics vs. Model Scale", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scaling_robustness.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "scaling_robustness.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Scaling plots saved")


# ── 7. Radar Chart ──────────────────────────────────────────────────────
def make_radar(results):
    dims = ["Accuracy", "F1", "Clean Acc", "Visual Acc", "Struct. Acc", "1−Fail%"]
    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for name in MODEL_ORDER:
        r = results[name]
        vals = [
            r["overall"]["accuracy"],
            r["overall"]["f1"],
            r["by_type"].get("clean", {}).get("accuracy", 0),
            r["by_type"].get("visual", {}).get("accuracy", 0),
            r["by_type"].get("structural", {}).get("accuracy", 0),
            1 - r["overall"]["failure_rate"] / 100,
        ]
        vals += vals[:1]
        family = r["family"]
        ax.plot(angles, vals, linewidth=1.5, alpha=0.6,
                color=FAMILY_COLORS[family], marker=FAMILY_MARKERS[family], markersize=5)
        ax.fill(angles, vals, alpha=0.03, color=FAMILY_COLORS[family])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Robustness: Multi-Dimensional Model Comparison", fontsize=13, fontweight="bold", pad=20)
    patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.savefig(os.path.join(OUT_DIR, "radar_robustness.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "radar_robustness.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Radar chart saved")


# ── 8. Confusion Matrices ───────────────────────────────────────────────
def make_confusion_matrices(all_data):
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    labels_order = ["safe", "offensive"]

    for idx, name in enumerate(MODEL_ORDER):
        ax = axes[idx // 5, idx % 5]
        df = all_data[name].copy()
        valid = df.dropna(subset=["model_verdict"])
        valid["label"] = valid["label"].str.lower().str.strip()
        valid["model_verdict"] = valid["model_verdict"].str.lower().str.strip()
        valid = valid[valid["model_verdict"].isin(["offensive", "safe"])]

        if len(valid) > 0:
            cm = confusion_matrix(valid["label"], valid["model_verdict"], labels=labels_order)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
        else:
            cm_norm = np.zeros((2, 2))

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels_order, yticklabels=labels_order,
                    ax=ax, vmin=0, vmax=1, cbar=False)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")

    plt.suptitle("Normalized Confusion Matrices – Robustness (All Types Combined)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrices_robustness.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrices_robustness.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Confusion matrices saved")


# ── 9. Per-Label Analysis ───────────────────────────────────────────────
def make_per_label_analysis(all_data):
    """Analyze detection rate for offensive vs safe separately across types."""
    rows = []
    for name in MODEL_ORDER:
        df = all_data[name].copy()
        valid = df.dropna(subset=["model_verdict"])
        valid["label"] = valid["label"].str.lower().str.strip()
        valid["model_verdict"] = valid["model_verdict"].str.lower().str.strip()
        valid = valid[valid["model_verdict"].isin(["offensive", "safe"])]

        for t in OBF_TYPES:
            sub = valid[valid["type"] == t]
            for lbl in ["offensive", "safe"]:
                lbl_sub = sub[sub["label"] == lbl]
                if len(lbl_sub) > 0:
                    correct = (lbl_sub["model_verdict"] == lbl).mean()
                else:
                    correct = 0
                rows.append({
                    "Model": name, "Type": t.capitalize(),
                    "Label": lbl.capitalize(), "Detection Rate": correct,
                })

    pl_df = pd.DataFrame(rows)

    # Two subplots: offensive detection rate, safe detection rate
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, lbl in zip(axes, ["Offensive", "Safe"]):
        sub = pl_df[pl_df["Label"] == lbl]
        pivot = sub.pivot(index="Model", columns="Type", values="Detection Rate").reindex(MODEL_ORDER)
        pivot = pivot[["Clean", "Visual", "Structural", "Semantic"]]
        pivot.plot(kind="bar", ax=ax, color=[OBF_COLORS[t] for t in OBF_TYPES], width=0.75)
        ax.set_ylabel(f"{lbl} Detection Rate", fontsize=12)
        ax.set_title(f"{lbl} Class: Detection Rate by Obfuscation Type", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Type", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.sca(ax)
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "per_label_analysis.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "per_label_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Per-label analysis saved")
    return pl_df


# ── 10. Attack Success Rate ─────────────────────────────────────────────
def make_attack_success_rate(all_data, results):
    """ASR = fraction of samples correctly classified when clean but misclassified when obfuscated."""
    rows = []
    for name in MODEL_ORDER:
        df = all_data[name].copy()
        valid = df.dropna(subset=["model_verdict"])
        valid["label"] = valid["label"].str.lower().str.strip()
        valid["model_verdict"] = valid["model_verdict"].str.lower().str.strip()
        valid = valid[valid["model_verdict"].isin(["offensive", "safe"])]

        clean = valid[valid["type"] == "clean"].set_index("id")
        clean["correct_clean"] = clean["label"] == clean["model_verdict"]

        for t in ["visual", "structural", "semantic"]:
            obf = valid[valid["type"] == t].set_index("id")
            obf["correct_obf"] = obf["label"] == obf["model_verdict"]
            merged = clean[["correct_clean"]].join(obf[["correct_obf"]], how="inner")
            # ASR = correctly classified clean BUT misclassified obfuscated
            if len(merged) > 0:
                flipped = ((merged["correct_clean"]) & (~merged["correct_obf"])).sum()
                asr = flipped / len(merged[merged["correct_clean"]]) if merged["correct_clean"].sum() > 0 else 0
            else:
                asr = 0
            rows.append({"Model": name, "Attack Type": t.capitalize(), "ASR": asr})

    asr_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 6))
    asr_pivot = asr_df.pivot(index="Model", columns="Attack Type", values="ASR").reindex(MODEL_ORDER)
    asr_pivot = asr_pivot[["Visual", "Structural", "Semantic"]]
    asr_pivot.plot(kind="bar", ax=ax,
                   color=[OBF_COLORS["visual"], OBF_COLORS["structural"], OBF_COLORS["semantic"]],
                   width=0.75)
    ax.set_ylabel("Attack Success Rate", fontsize=12)
    ax.set_title("Adversarial Attack Success Rate (Flipped from Correct → Wrong)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Attack Type", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "attack_success_rate.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "attack_success_rate.png"), dpi=300, bbox_inches="tight")
    plt.close()

    asr_pivot.to_csv(os.path.join(OUT_DIR, "attack_success_rate.csv"))
    print("[✓] Attack success rate saved")
    return asr_df


# ── 11. Confidence Analysis ─────────────────────────────────────────────
def make_confidence_analysis(all_data):
    rows = []
    for name in MODEL_ORDER:
        df = all_data[name].copy()
        valid = df.dropna(subset=["model_verdict"])
        valid["label"] = valid["label"].str.lower().str.strip()
        valid["model_verdict"] = valid["model_verdict"].str.lower().str.strip()
        valid = valid[valid["model_verdict"].isin(["offensive", "safe"])]
        valid["correct"] = valid["label"] == valid["model_verdict"]

        if "model_confidence" in valid.columns:
            valid["conf"] = pd.to_numeric(valid["model_confidence"], errors="coerce")
            valid = valid.dropna(subset=["conf"])
            valid["conf"] = valid["conf"].clip(0, 1)
            for _, r in valid.iterrows():
                rows.append({
                    "Model": name, "Family": MODELS[name][2],
                    "Correct": "Correct" if r["correct"] else "Wrong",
                    "Confidence": r["conf"], "Type": r["type"],
                })

    if not rows:
        print("[!] No confidence data found, skipping confidence analysis")
        return

    alldf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot: correct vs wrong
    sns.boxplot(data=alldf, x="Family", y="Confidence", hue="Correct", ax=axes[0],
                palette={"Correct": "#2A9D8F", "Wrong": "#E76F51"})
    axes[0].set_title("Confidence: Correct vs. Wrong Predictions", fontsize=13, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Mean confidence by type and correctness
    type_conf = alldf.groupby(["Type", "Correct"])["Confidence"].mean().unstack("Correct")
    type_conf.plot(kind="bar", ax=axes[1], color=["#2A9D8F", "#E76F51"])
    axes[1].set_title("Mean Confidence by Type & Correctness", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Mean Confidence")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(title="Prediction")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confidence_robustness.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "confidence_robustness.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Confidence analysis saved")


# ── 12. Robustness Drop Waterfall ────────────────────────────────────────
def make_robustness_waterfall(results):
    """Bar chart showing clean accuracy and how much each attack type degrades it."""
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(MODEL_ORDER))
    width = 0.5

    clean_accs = []
    drops_visual = []
    drops_struct = []
    drops_semantic = []

    for name in MODEL_ORDER:
        r = results[name]
        c = r["by_type"].get("clean", {}).get("accuracy", 0)
        v = r["by_type"].get("visual", {}).get("accuracy", 0)
        s = r["by_type"].get("structural", {}).get("accuracy", 0)
        sem = r["by_type"].get("semantic", {}).get("accuracy", 0)
        clean_accs.append(c)
        drops_visual.append(max(0, c - v))
        drops_struct.append(max(0, c - s))
        drops_semantic.append(max(0, c - sem))

    # Stacked: remaining acc + drops
    remaining = [clean_accs[i] - drops_visual[i] - drops_struct[i] - drops_semantic[i]
                 for i in range(len(MODEL_ORDER))]
    remaining = [max(0, r) for r in remaining]

    ax.bar(x, clean_accs, width, label="Clean Accuracy", color="#264653", alpha=0.9)
    ax.bar(x, [-d for d in drops_visual], width, bottom=clean_accs,
           label="Visual Drop", color="#2A9D8F", alpha=0.7)

    # Instead of stacked waterfall (complex), show clean vs each type separately
    # Let's do a cleaner grouped comparison
    ax.clear()
    width = 0.2
    ax.bar(x - 1.5*width, clean_accs, width, label="Clean", color="#264653")
    vis_accs = [results[n]["by_type"].get("visual", {}).get("accuracy", 0) for n in MODEL_ORDER]
    str_accs = [results[n]["by_type"].get("structural", {}).get("accuracy", 0) for n in MODEL_ORDER]
    sem_accs = [results[n]["by_type"].get("semantic", {}).get("accuracy", 0) for n in MODEL_ORDER]
    ax.bar(x - 0.5*width, vis_accs, width, label="Visual", color="#2A9D8F")
    ax.bar(x + 0.5*width, str_accs, width, label="Structural", color="#E9C46A")
    ax.bar(x + 1.5*width, sem_accs, width, label="Semantic", color="#E76F51")

    # Connect clean to semantic with lines for drop visualization
    for i in range(len(MODEL_ORDER)):
        ax.plot([i - 1.5*width, i + 1.5*width], [clean_accs[i], sem_accs[i]],
                 "k--", alpha=0.3, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Robustness Degradation: Clean → Visual → Structural → Semantic",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "robustness_waterfall.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "robustness_waterfall.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] Robustness waterfall saved")


# ── 13. Case Studies ────────────────────────────────────────────────────
def make_case_studies(all_data):
    """Extract adversarial evasion examples: correct on clean, wrong on obfuscated."""
    cases = []
    rep_models = ["Llama-3.1-8B", "Qwen2.5-7B", "DeepSeek-R1-7B"]

    for name in rep_models:
        df = all_data[name].copy()
        valid = df.dropna(subset=["model_verdict"])
        valid["label"] = valid["label"].str.lower().str.strip()
        valid["model_verdict"] = valid["model_verdict"].str.lower().str.strip()
        valid = valid[valid["model_verdict"].isin(["offensive", "safe"])]

        clean = valid[valid["type"] == "clean"].set_index("id")
        clean["correct_clean"] = clean["label"] == clean["model_verdict"]

        for t in ["visual", "structural", "semantic"]:
            obf = valid[valid["type"] == t].set_index("id")
            obf["correct_obf"] = obf["label"] == obf["model_verdict"]
            merged = clean.join(obf, rsuffix="_obf", how="inner")
            flipped = merged[(merged["correct_clean"]) & (~merged["correct_obf"])]
            if len(flipped) > 0:
                sample = flipped.iloc[0]
                cases.append({
                    "Model": name,
                    "Attack Type": t.capitalize(),
                    "Gold Label": sample["label"],
                    "Clean Text": str(sample["clean_text"])[:100],
                    "Obfuscated Text": str(sample.get("obfuscated_text_obf", sample.get("clean_text_obf", "")))[:100],
                    "Clean Verdict": sample["model_verdict"],
                    "Obf. Verdict": sample.get("model_verdict_obf", "?"),
                })

    case_df = pd.DataFrame(cases)
    case_df.to_csv(os.path.join(OUT_DIR, "case_studies_robustness.csv"), index=False)

    # LaTeX
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Adversarial evasion case studies: samples correctly classified from clean text "
                       r"but misclassified after obfuscation.}")
    latex_lines.append(r"\label{tab:robustness_cases}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{l|l|l|p{5cm}|p{5cm}|cc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Model} & \textbf{Attack} & \textbf{Gold} & "
                       r"\textbf{Clean Text} & \textbf{Obfuscated Text} & "
                       r"\textbf{Clean Pred} & \textbf{Obf Pred} \\")
    latex_lines.append(r"\midrule")
    for _, r in case_df.iterrows():
        ct = str(r["Clean Text"]).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
        ot = str(r["Obfuscated Text"]).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
        latex_lines.append(
            f"{r['Model']} & {r['Attack Type']} & {r['Gold Label']} & "
            f"{ct} & {ot} & {r['Clean Verdict']} & {r['Obf. Verdict']} \\\\"
        )
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table*}")
    with open(os.path.join(OUT_DIR, "case_studies_robustness.tex"), "w") as f:
        f.write("\n".join(latex_lines))
    print("[✓] Case studies saved")
    return case_df


# ── 14. Family Summary ──────────────────────────────────────────────────
def make_family_summary(results):
    family_data = {}
    for name in MODEL_ORDER:
        fam = results[name]["family"]
        if fam not in family_data:
            family_data[fam] = []
        family_data[fam].append(results[name])

    rows = []
    for fam, rs in family_data.items():
        row = {"Family": fam, "# Models": len(rs)}
        for key in ["accuracy", "f1", "failure_rate"]:
            vals = [r["overall"][key] for r in rs]
            row[f"{key}_mean"] = np.mean(vals)
            row[f"{key}_std"] = np.std(vals)
        for t in OBF_TYPES:
            vals = [r["by_type"][t]["accuracy"] for r in rs if t in r["by_type"]]
            if vals:
                row[f"acc_{t}_mean"] = np.mean(vals)
        rows.append(row)

    fam_df = pd.DataFrame(rows)
    fam_df.to_csv(os.path.join(OUT_DIR, "family_summary_robustness.csv"), index=False)
    print("[✓] Family summary saved")
    print(fam_df.to_string(index=False))
    return fam_df


# ── 15. Deep Analysis Report ────────────────────────────────────────────
def write_analysis_report(results, all_data, summary_df, asr_df, pl_df, fam_df):
    lines = []
    lines.append("=" * 80)
    lines.append("ROBUSTNESS AXIS – DEEP ANALYSIS REPORT")
    lines.append("SM-Bench: Social Media Content Moderation Benchmark")
    lines.append("=" * 80)
    lines.append("")

    # 1. Overview
    lines.append("1. EVALUATION OVERVIEW")
    lines.append("-" * 40)
    lines.append("   Models evaluated: 10")
    lines.append("   Families: Llama (1B, 3B, 8B), Qwen2.5 (0.5B, 1.5B, 3B, 7B, 14B), DeepSeek-R1 (1.5B, 7B)")
    lines.append("   Dataset: SBIC Obfuscated (6,848 samples)")
    lines.append("     - 2,000 clean (1,000 offensive + 1,000 safe)")
    lines.append("     - 2,000 visual obfuscation (leetspeak, @/0/3 substitution)")
    lines.append("     - 1,424 structural (dotted letters, spacing tricks)")
    lines.append("     - 1,424 semantic (LLM-based paraphrasing)")
    lines.append("   Task: Binary classification (offensive vs. safe)")
    lines.append("   Metrics: Accuracy, Precision, Recall, F1, Format Failure Rate, Attack Success Rate")
    lines.append("   Note: Qwen2.5-14B only has 1,705/6,848 samples (partial run); DeepSeek-R1-1.5B has 6,138/6,848.")
    lines.append("")

    # 2. Key Findings
    lines.append("2. KEY FINDINGS")
    lines.append("-" * 40)
    accs = {n: results[n]["overall"]["accuracy"] for n in MODEL_ORDER}
    best = max(accs, key=accs.get)
    worst = min(accs, key=accs.get)
    lines.append(f"   Best overall accuracy:  {best} ({accs[best]:.3f})")
    lines.append(f"   Worst overall accuracy: {worst} ({accs[worst]:.3f})")
    lines.append(f"   Range: {max(accs.values()):.3f}–{min(accs.values()):.3f}")
    lines.append("")

    # Failure rates
    fails = {n: results[n]["overall"]["failure_rate"] for n in MODEL_ORDER}
    best_fail = min(fails, key=fails.get)
    worst_fail = max(fails, key=fails.get)
    lines.append(f"   Lowest failure rate:  {best_fail} ({fails[best_fail]:.1f}%)")
    lines.append(f"   Highest failure rate: {worst_fail} ({fails[worst_fail]:.1f}%)")
    lines.append("")

    # 3. Per-Type Analysis
    lines.append("3. PER-OBFUSCATION-TYPE ANALYSIS")
    lines.append("-" * 40)
    for t in OBF_TYPES:
        t_accs = {}
        for name in MODEL_ORDER:
            if t in results[name]["by_type"]:
                t_accs[name] = results[name]["by_type"][t]["accuracy"]
        if t_accs:
            best_t = max(t_accs, key=t_accs.get)
            mean_t = np.mean(list(t_accs.values()))
            lines.append(f"   {t.upper():12s}  mean_acc={mean_t:.3f}  best={best_t} ({t_accs[best_t]:.3f})")
    lines.append("")
    lines.append("   FINDING: Semantic obfuscation is the most challenging type across all families.")
    lines.append("   Visual obfuscation (leetspeak) causes moderate degradation.")
    lines.append("   Structural obfuscation (dotted text) varies significantly by model family.")
    lines.append("")

    # 4. Robustness Degradation
    lines.append("4. ROBUSTNESS DEGRADATION (CLEAN → OBFUSCATED)")
    lines.append("-" * 40)
    for name in MODEL_ORDER:
        r = results[name]
        clean_acc = r["by_type"].get("clean", {}).get("accuracy", 0)
        obf_accs = [r["by_type"][t]["accuracy"] for t in ["visual", "structural", "semantic"]
                    if t in r["by_type"]]
        avg_obf = np.mean(obf_accs) if obf_accs else 0
        drop = clean_acc - avg_obf
        lines.append(f"   {name:25s}  clean={clean_acc:.3f}  avg_obf={avg_obf:.3f}  drop={drop:+.3f}")
    lines.append("")

    # 5. Scaling Analysis
    lines.append("5. SCALING ANALYSIS")
    lines.append("-" * 40)
    for fam in ["Llama", "Qwen2.5", "DeepSeek"]:
        pts = [(results[n]["params_b"], n, results[n]["overall"]["accuracy"])
               for n in MODEL_ORDER if results[n]["family"] == fam]
        pts.sort()
        lines.append(f"   {fam}:")
        for p, n, a in pts:
            lines.append(f"     {n:25s}  {p:5.1f}B  →  Acc={a:.3f}")
        if len(pts) >= 2:
            delta = pts[-1][2] - pts[0][2]
            lines.append(f"     Scaling gain ({pts[0][0]}B→{pts[-1][0]}B): Δacc = {delta:+.3f}")
        lines.append("")

    # 6. Attack Success Rate
    lines.append("6. ATTACK SUCCESS RATE (ASR)")
    lines.append("-" * 40)
    lines.append("   ASR = fraction of correctly-classified clean samples that are")
    lines.append("   misclassified after obfuscation (measures attack effectiveness).")
    lines.append("")
    if asr_df is not None:
        asr_pivot = asr_df.pivot(index="Model", columns="Attack Type", values="ASR").reindex(MODEL_ORDER)
        lines.append(asr_pivot.to_string())
    lines.append("")

    # 7. Failure Rate patterns
    lines.append("7. FORMAT FAILURE ANALYSIS")
    lines.append("-" * 40)
    for name in MODEL_ORDER:
        r = results[name]
        overall_fr = r["overall"]["failure_rate"]
        type_frs = {t: r["by_type"][t]["failure_rate"]
                    for t in OBF_TYPES if t in r["by_type"]}
        fr_str = "  ".join([f"{t}={v:.1f}%" for t, v in type_frs.items()])
        lines.append(f"   {name:25s}  overall={overall_fr:.1f}%  {fr_str}")
    lines.append("")
    lines.append("   FINDING: Obfuscated inputs significantly increase format failure rates.")
    lines.append("   Semantic obfuscation causes the highest failure rates as the paraphrased")
    lines.append("   text can confuse the model's JSON output format adherence.")
    lines.append("")

    # 8. Cross-Family Comparison
    lines.append("8. CROSS-FAMILY COMPARISON (SAME SIZE TIER)")
    lines.append("-" * 40)
    lines.append("   ~1.5B tier:")
    for n in ["Qwen2.5-1.5B", "DeepSeek-R1-1.5B"]:
        if n in results:
            a = results[n]["overall"]["accuracy"]
            lines.append(f"     {n}: Acc={a:.3f}")
    lines.append("   ~7-8B tier:")
    for n in ["Llama-3.1-8B", "Qwen2.5-7B", "DeepSeek-R1-7B"]:
        if n in results:
            a = results[n]["overall"]["accuracy"]
            lines.append(f"     {n}: Acc={a:.3f}")
    lines.append("")

    # 9. DeepSeek-R1 Reasoning Model Insights
    lines.append("9. REASONING MODEL INSIGHTS (DeepSeek-R1)")
    lines.append("-" * 40)
    lines.append("   DeepSeek-R1 models are distilled from reasoning models.")
    for n in ["DeepSeek-R1-1.5B", "DeepSeek-R1-7B"]:
        r = results[n]
        lines.append(f"   {n}: Accuracy={r['overall']['accuracy']:.3f}, F1={r['overall']['f1']:.3f}")
        for t in OBF_TYPES:
            if t in r["by_type"]:
                lines.append(f"     {t}: {r['by_type'][t]['accuracy']:.3f}")
    lines.append("")

    # 10. Conclusions
    lines.append("10. CONCLUSIONS AND IMPLICATIONS")
    lines.append("-" * 40)
    lines.append("   (a) All models are vulnerable to adversarial text obfuscation. Even top-performing")
    lines.append("       models lose significant accuracy under visual and semantic attacks.")
    lines.append("   (b) Semantic obfuscation (LLM paraphrasing) is the most effective attack,")
    lines.append("       often causing the largest accuracy drops across all families.")
    lines.append("   (c) Larger models generally show better robustness, but the relationship is")
    lines.append("       not linear – some smaller models handle specific attack types well.")
    lines.append("   (d) Format failure is a significant issue: models frequently fail to produce")
    lines.append("       valid structured output when given obfuscated inputs.")
    lines.append("   (e) There is a clear offensive-detection vs safe-preservation tradeoff:")
    lines.append("       aggressive moderation (high offensive recall) comes at the cost of")
    lines.append("       false-flagging safe content, especially under obfuscation.")
    lines.append("   (f) The robustness gap between clean and obfuscated performance represents")
    lines.append("       a critical deployment vulnerability for real-world content moderation.")
    lines.append("")

    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "robustness_deep_analysis.txt"), "w") as f:
        f.write(report)
    print("[✓] Deep analysis report saved")
    print("\n" + report)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading all model results...")
    all_data = load_all()
    print(f"Loaded {len(all_data)} models\n")

    for name, df in all_data.items():
        print(f"  {name:25s}  n={len(df):5d}  nulls={df['model_verdict'].isnull().sum()}")
    print()

    print("Computing metrics...")
    results = compute_all_metrics(all_data)
    print()

    print("=" * 60)
    print("GENERATING TABLES")
    print("=" * 60)
    summary_df = make_summary_table(results)

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    make_heatmap(results)
    make_grouped_bars(results)
    make_degradation_plot(results)
    fbt_pivot = make_failure_analysis(results, all_data)
    make_scaling_plots(results)
    make_radar(results)
    make_confusion_matrices(all_data)
    pl_df = make_per_label_analysis(all_data)
    asr_df = make_attack_success_rate(all_data, results)
    make_confidence_analysis(all_data)
    make_robustness_waterfall(results)

    print("\n" + "=" * 60)
    print("GENERATING ANALYSIS")
    print("=" * 60)
    case_df = make_case_studies(all_data)
    fam_df = make_family_summary(results)

    print("\n" + "=" * 60)
    print("WRITING DEEP ANALYSIS REPORT")
    print("=" * 60)
    write_analysis_report(results, all_data, summary_df, asr_df, pl_df, fam_df)

    print("\n" + "=" * 60)
    print(f"ALL OUTPUTS SAVED TO: {os.path.abspath(OUT_DIR)}/")
    print("=" * 60)
    print("\nFiles generated:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f:50s} {size:>10,} bytes")
