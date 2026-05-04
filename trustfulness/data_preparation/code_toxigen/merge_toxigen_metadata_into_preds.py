import pandas as pd

TOXIGEN_INPUT = "Datasets/toxigen/ToxiGen_step1_sampled_1000.csv"

PRED_FILES = [
    "outputs/toxigen/ToxiGen_step1_mistral_preds.csv",
    "outputs/toxigen/ToxiGen_step1_llama_preds.csv",
    "outputs/toxigen/ToxiGen_step1_qwen3_30b_instruct_preds.csv",
]

def main():
    meta = pd.read_csv(TOXIGEN_INPUT)[["id", "trustfulness_type", "target_group"]].copy()
    meta["id"] = meta["id"].astype(str)

    for pf in PRED_FILES:
        pred = pd.read_csv(pf)
        pred["id"] = pred["id"].astype(str)

        # avoid duplicating columns if you rerun
        for c in ["trustfulness_type", "target_group"]:
            if c in pred.columns:
                pred = pred.drop(columns=[c])

        merged = pred.merge(meta, on="id", how="left")

        missing = merged["trustfulness_type"].isna().sum()
        if missing > 0:
            print(f"[WARN] {pf}: {missing} rows missing trustfulness_type after merge (id mismatch?)")

        merged.to_csv(pf, index=False)
        print(f"[OK] merged -> {pf}")

if __name__ == "__main__":
    main()
