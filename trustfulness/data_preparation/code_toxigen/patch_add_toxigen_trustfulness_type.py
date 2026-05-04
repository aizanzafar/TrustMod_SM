import pandas as pd

INP = "Datasets/toxigen/ToxiGen_step1_sampled_1000.csv"
OUT = "Datasets/toxigen/ToxiGen_step1_sampled_1000.csv"  # overwrite in-place

def main():
    df = pd.read_csv(INP)

    if "gold_label" not in df.columns:
        raise ValueError("Missing gold_label in ToxiGen_step1_sampled_1000.csv")

    if "trustfulness_type" not in df.columns:
        df["trustfulness_type"] = df["gold_label"].astype(str).str.upper().map({
            "BENIGN": "benign_minority",
            "HATE": "implicit_toxic",
        }).fillna("implicit_toxic")  # conservative fallback

        print("[OK] Added trustfulness_type column.")

    # target_group should already exist (you printed its distribution),
    # but if not, keep going.
    if "target_group" not in df.columns:
        print("[WARN] target_group missing in sampled csv. Step-2 still works, but group slicing won’t.")

    df.to_csv(OUT, index=False)
    print(f"[DONE] Wrote: {OUT}")
    print(df["trustfulness_type"].value_counts())

if __name__ == "__main__":
    main()
