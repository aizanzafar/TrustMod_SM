# merge_and_shuffle_trustfulness.py
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42

FILES = [
    r"./Datasets/HASOC/HASOC_step1_sampled_1000_FIXED.csv",
    r"./Datasets/SBIC/SBIC_step1_sampled_1000.csv",
    r"./Datasets/toxigen/ToxiGen_step1_sampled_1000.csv",
]

OUT_FILE = "TrustfulnessAxis_v1_3000.csv"

FINAL_COLS = [
    "id",
    "text",
    "source_dataset",
    "gold_label",
    "needs_fact_check",
    "gold_uncertainty",
    "trustfulness_type",
]

def main():
    dfs = []

    for f in FILES:
        f = Path(f)
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")

        df = pd.read_csv(f)

        # schema check
        missing = set(FINAL_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"{f.name} missing columns: {missing}")

        # keep only final columns (strict)
        df = df[FINAL_COLS].copy()
        dfs.append(df)

        print(f"[OK] Loaded {f.name}: {len(df)} rows")

    # append
    merged = pd.concat(dfs, ignore_index=True)
    print(f"\n[INFO] Appended total rows: {len(merged)}")

    # global shuffle
    merged = merged.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # final save
    merged.to_csv(OUT_FILE, index=False)
    print(f"[DONE] Saved merged + shuffled dataset → {OUT_FILE}")

    # quick sanity summary
    print("\n[FINAL DISTRIBUTIONS]")
    print("\nsource_dataset:")
    print(merged["source_dataset"].value_counts())

    print("\ntrustfulness_type:")
    print(merged["trustfulness_type"].value_counts())

    print("\ngold_label:")
    print(merged["gold_label"].value_counts())


if __name__ == "__main__":
    main()
