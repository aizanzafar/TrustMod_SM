import pandas as pd
import os

# PRED_FILE = "outputs/HASOC/HASOC_step1_llama_preds.csv"

PRED_FILE = "outputs/HASOC/HASOC_step1_qwen3_30b_instruct_preds.csv"
BUCKET_FILE = "Datasets/HASOC/HASOC_trustfulness_types.csv"
OUT_DIR = "outputs/HASOC/qwen3_30b_instruct_top_overconfident_errors"

CONF_THRESHOLD = 0.9
TOP_K = 20

os.makedirs(OUT_DIR, exist_ok=True)

df_pred = pd.read_csv(PRED_FILE)
df_bucket = pd.read_csv(BUCKET_FILE)

df = df_pred.merge(df_bucket, on="id", how="inner")

# Define overconfident wrong
df["is_wrong"] = df["pred_label"] != df["gold_label"]
df["is_overconfident"] = df["pred_confidence"] >= CONF_THRESHOLD

df_bad = df[df["is_wrong"] & df["is_overconfident"]]

print(f"Total overconfident wrong cases: {len(df_bad)}")

for bucket, g in df_bad.groupby("trustfulness_type"):
    g_sorted = g.sort_values("pred_confidence", ascending=False).head(TOP_K)

    out_cols = [
        "id",
        "trustfulness_type",
        "text",
        "gold_label",
        "pred_label",
        "pred_confidence",
        "needs_fact_check",
        "pred_needs_fact_check",
        "pred_rationale",
    ]

    out_path = os.path.join(
        OUT_DIR, f"top_{TOP_K}_overconfident_{bucket}.csv"
    )
    g_sorted[out_cols].to_csv(out_path, index=False)

    print(f"[SAVED] {bucket}: {len(g_sorted)} cases → {out_path}")
