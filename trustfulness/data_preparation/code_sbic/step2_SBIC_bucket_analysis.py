# step2_SBIC_bucket_analysis.py

import pandas as pd
import json
from run_trustfulness_model import compute_metrics

# -------------------------------
# CONFIG: set one model at a time
# -------------------------------

# # Qwen
# PRED_FILE = "outputs/SBIC/SBIC_step1_qwen_preds.csv"
# OUT_JSON  = "outputs/SBIC/SBIC_step2_qwen_bucket_metrics.json"

# # Mistral
# PRED_FILE = "outputs/SBIC/SBIC_step1_mistral_preds.csv"
# OUT_JSON  = "outputs/SBIC/SBIC_step2_mistral_bucket_metrics.json"

# LLaMA
PRED_FILE = "outputs/SBIC/SBIC_step1_llama_preds.csv"
OUT_JSON  = "outputs/SBIC/SBIC_step2_llama_bucket_metrics.json"

# This is the Step-1 normalized dataset (contains trustfulness_type)
STEP1_FILE = "Datasets/SBIC/SBIC_step1_sampled_1000.csv"


# -------------------------------
# Load predictions + step1 and MERGE trustfulness_type
# -------------------------------
pred = pd.read_csv(PRED_FILE)
step1 = pd.read_csv(STEP1_FILE)

if "id" not in pred.columns or "id" not in step1.columns:
    raise ValueError("Both prediction file and step1 file must contain 'id' column.")

if "trustfulness_type" not in step1.columns:
    raise ValueError(
        "Missing trustfulness_type in SBIC_step1_sampled_1000.csv. "
        "But it should exist (since SBIC normalization added it)."
    )

# Merge
df = pred.merge(step1[["id", "trustfulness_type"]], on="id", how="left")

# Check merge success
missing_tt = df["trustfulness_type"].isna().sum()
if missing_tt > 0:
    raise ValueError(
        f"Merge failed for {missing_tt} rows (trustfulness_type is NaN). "
        "Likely id mismatch: ensure predictions were run on the same SBIC_step1_sampled_1000.csv."
    )

# Sanity checks for compute_metrics
required_cols = {
    "trustfulness_type",
    "gold_label",
    "pred_label",
    "pred_confidence",
    "needs_fact_check",
    "pred_needs_fact_check",
    "gold_uncertainty",
    "pred_needs_human_review",
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in merged dataframe: {missing}")


# -------------------------------
# Bucket-wise metrics
# -------------------------------
results = {}
for bucket, g in df.groupby("trustfulness_type"):
    metrics = compute_metrics(g)
    results[bucket] = metrics.__dict__
    results[bucket]["n"] = len(g)

# -------------------------------
# Save results
# -------------------------------
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print(f"[DONE] wrote: {OUT_JSON}")
