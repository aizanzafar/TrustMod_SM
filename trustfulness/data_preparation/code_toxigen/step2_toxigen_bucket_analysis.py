import pandas as pd
import json
from run_trustfulness_model import compute_metrics


# PRED_FILE = "outputs/toxigen/ToxiGen_step1_mistral_preds.csv"
# OUT_JSON  = "outputs/toxigen/ToxiGen_step2_mistral_bucket_metrics.json"

# PRED_FILE = "outputs/toxigen/ToxiGen_step1_llama_preds.csv"
# OUT_JSON  = "outputs/toxigen/ToxiGen_step2_llama_bucket_metrics.json"

PRED_FILE = "outputs/toxigen/ToxiGen_step1_qwen3_30b_instruct_preds.csv"
OUT_JSON  = "outputs/toxigen/ToxiGen_step2_qwen3_30b_instruct_bucket_metrics.json"


df = pd.read_csv(PRED_FILE)

# sanity check
if "trustfulness_type" not in df.columns:
    raise ValueError(
        "Missing trustfulness_type in predictions CSV. "
        "Fix: ensure ToxiGen_step1_sampled_1000.csv includes trustfulness_type "
        "and that run_trustfulness_model.py carries it forward (merge it in)."
    )

results = {}
for bucket, g in df.groupby("trustfulness_type"):
    metrics = compute_metrics(g)
    results[bucket] = metrics.__dict__
    results[bucket]["n"] = len(g)

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print(f"[DONE] wrote: {OUT_JSON}")
