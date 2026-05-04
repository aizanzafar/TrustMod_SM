import pandas as pd
import json
from run_trustfulness_model import compute_metrics

# PRED_FILE = "outputs/HASOC/HASOC_step1_mistral_preds.csv"
# PRED_FILE = "outputs/HASOC/HASOC_step1_llama_preds.csv"
PRED_FILE = "outputs/HASOC/HASOC_step1_qwen3_30b_instruct_preds.csv"
BUCKET_FILE = "Datasets/HASOC/HASOC_trustfulness_types.csv"
OUT_JSON = "outputs/HASOC/HASOC_step2_qwen3_30b_instruct_bucket_metrics.json"

df_pred = pd.read_csv(PRED_FILE)
df_bucket = pd.read_csv(BUCKET_FILE)

df = df_pred.merge(df_bucket, on="id", how="inner")

results = {}
for bucket, g in df.groupby("trustfulness_type"):
    metrics = compute_metrics(g)
    results[bucket] = metrics.__dict__
    results[bucket]["n"] = len(g)

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
