import pandas as pd
import numpy as np

HASOC_CSV = "HASOC/en_Hasoc2021_train.csv"
FACT_CSV  = "HASOC/needs_fact_check.csv"   # your file
OUT_FULL  = "HASOC/HASOC_step1_full.csv"
OUT_SAMP  = "HASOC/HASOC_step1_sampled_1000.csv"

RANDOM_SEED = 42
TARGET_N = 1000  

def map_hasoc_to_gold(task_1: str, task_2: str) -> str:
    t1 = str(task_1).strip().upper()
    t2 = str(task_2).strip().upper()
    if t1 == "NOT":
        return "BENIGN"
    # HOF:
    if t2 == "HATE":
        return "HATE"
    if t2 in {"OFFN", "PRFN"}:
        return "OFFENSE"
    return "OTHER"

def main():
    df = pd.read_csv(HASOC_CSV)

    # Ensure id + text exist
    # HASOC usually uses "_id" or "id". Your file shows "_id".
    if "_id" not in df.columns:
        raise ValueError("Expected HASOC to have column '_id'.")
    if "text" not in df.columns:
        raise ValueError("Expected HASOC to have column 'text'.")
    if "task_1" not in df.columns or "task_2" not in df.columns:
        raise ValueError("Expected HASOC to have columns 'task_1' and 'task_2'.")

    # Gold label mapping
    df["gold_label"] = df.apply(lambda r: map_hasoc_to_gold(r["task_1"], r["task_2"]), axis=1)

    # Load fact-check file and merge
    df_fc = pd.read_csv(FACT_CSV)

    # Try common id column names in needs_fact_check.csv
    id_col = None
    for c in ["_id", "id", "tweet_id"]:
        if c in df_fc.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError("needs_fact_check.csv must contain an id column like _id/id/tweet_id.")

    # The boolean column name could be "requires_fact_check" based on your script
    # (filter_fact_check_tweets.py writes df['requires_fact_check']) — adjust if needed.
    bool_col = None
    for c in ["requires_fact_check", "needs_fact_check"]:
        if c in df_fc.columns:
            bool_col = c
            break
    if bool_col is None:
        raise ValueError("needs_fact_check.csv must contain 'requires_fact_check' or 'needs_fact_check' column.")

    df_fc = df_fc[[id_col, bool_col]].copy()
    df_fc = df_fc.rename(columns={id_col: "_id", bool_col: "needs_fact_check"})
    df_fc["needs_fact_check"] = df_fc["needs_fact_check"].astype(bool)

    # Default needs_fact_check = False unless in df_fc
    df = df.merge(df_fc, on="_id", how="left")
    df["needs_fact_check"] = df["needs_fact_check"].fillna(False).astype(bool)

    # For HASOC Step-1: set uncertainty to CLEAR unless you have ambiguity annotations
    df["gold_uncertainty"] = "CLEAR"
    df["source_dataset"] = "HASOC"
    df = df.rename(columns={"_id": "id"})

    # Keep only fields needed by runner
    out = df[["id", "text", "source_dataset", "gold_label", "needs_fact_check", "gold_uncertainty"]].copy()
    out.to_csv(OUT_FULL, index=False)
    print(f"Saved full normalized HASOC file: {OUT_FULL} ({len(out)} rows)")

    # ---- Sampling: 50/50 fact-check vs non-fact-check, stratified by gold_label ----
    rng = np.random.default_rng(RANDOM_SEED)

    df_true = out[out["needs_fact_check"] == True].copy()
    df_false = out[out["needs_fact_check"] == False].copy()

    half = TARGET_N // 2
    if len(df_true) < half or len(df_false) < half:
        print("[WARN] Not enough samples to do perfect 50/50 split.")
        # fall back to whatever available
        # but still stratify roughly
        half = min(len(df_true), len(df_false), half)

    def stratified_sample(df_part, n):
        # sample within each gold_label proportionally, but ensure at least 1 if possible
        labels = df_part["gold_label"].value_counts()
        total = labels.sum()
        pieces = []
        remaining = n
        for lab, cnt in labels.items():
            k = max(1, int(round(n * (cnt / total))))
            k = min(k, cnt)
            pieces.append(df_part[df_part["gold_label"] == lab].sample(n=k, random_state=RANDOM_SEED))
            remaining -= k

        sampled = pd.concat(pieces, ignore_index=True)
        if len(sampled) > n:
            sampled = sampled.sample(n=n, random_state=RANDOM_SEED)
        elif len(sampled) < n and remaining > 0:
            # top-up uniformly from remaining rows not already in sample
            rest = df_part[~df_part["id"].isin(sampled["id"])]
            if len(rest) > 0:
                top = rest.sample(n=min(remaining, len(rest)), random_state=RANDOM_SEED)
                sampled = pd.concat([sampled, top], ignore_index=True)

        return sampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    samp_true = stratified_sample(df_true, half)
    samp_false = stratified_sample(df_false, half)
    samp = pd.concat([samp_true, samp_false], ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED)

    samp.to_csv(OUT_SAMP, index=False)
    print(f"Saved sampled HASOC file: {OUT_SAMP} ({len(samp)} rows)")

if __name__ == "__main__":
    main()
