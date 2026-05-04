import pandas as pd
import numpy as np
import hashlib
import ast

TOXIGEN_TRAIN = "Datasets/toxigen/annotated_train.csv"

OUT_FULL = "Datasets/toxigen/ToxiGen_step1_full.csv"
OUT_SAMP = "Datasets/toxigen/ToxiGen_step1_sampled_1000.csv"

RANDOM_SEED = 42
TARGET_N = 1000


def clean_text(x) -> str:
    """
    Your CSV shows values like b'....' (bytes literal string).
    This converts them into a normal python str.
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()

    # Handle strings that look like python bytes literal: b'...'
    if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
        try:
            b = ast.literal_eval(s)  # -> bytes
            if isinstance(b, (bytes, bytearray)):
                return b.decode("utf-8", errors="ignore").strip()
        except Exception:
            pass

    return s


def make_id(text: str, target_group: str, prefix: str = "toxigen") -> str:
    key = f"{text}||{target_group}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return f"{prefix}_{h}"


def main():
    df = pd.read_csv(TOXIGEN_TRAIN)

    # Required columns in annotated_train.csv
    required = {"text", "target_group", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in annotated_train.csv: {missing}")

    # Clean fields
    df["text"] = df["text"].apply(clean_text)
    df["target_group"] = df["target_group"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Create stable id
    df["id"] = df.apply(lambda r: make_id(r["text"], r["target_group"]), axis=1)

    # Normalize labels
    # label: hate / neutral
    label_map = {"hate": "HATE", "neutral": "BENIGN"}
    df["gold_label"] = df["label"].map(label_map).fillna("OTHER")

    # Trustfulness fields
    df["source_dataset"] = "TOXIGEN"
    df["needs_fact_check"] = False

    # Uncertainty: neutral (benign minority) is where over-moderation happens → AMBIGUOUS
    df["gold_uncertainty"] = np.where(
        df["gold_label"] == "BENIGN",
        "AMBIGUOUS_NEEDS_REVIEW",
        "CLEAR"
    )

    # Keep only Step-1 required fields (+ target_group for later fairness slicing)
    out = df[[
        "id", "text", "source_dataset", "gold_label",
        "needs_fact_check", "gold_uncertainty", "target_group"
    ]].copy()

    out.to_csv(OUT_FULL, index=False)
    print(f"[OK] Saved full normalized ToxiGen: {OUT_FULL} ({len(out)} rows)")

    # -------------------------
    # Sampling (balanced + stratified by target_group)
    # -------------------------
    df_hate = out[out["gold_label"] == "HATE"].copy()
    df_ben  = out[out["gold_label"] == "BENIGN"].copy()

    half = TARGET_N // 2
    half = min(half, len(df_hate), len(df_ben))
    if half < TARGET_N // 2:
        print(f"[WARN] Not enough rows for full 1000. Using {2*half} samples total.")

    def stratified_sample_by_group(df_part: pd.DataFrame, n: int) -> pd.DataFrame:
        # Stratify by target_group
        counts = df_part["target_group"].fillna("UNKNOWN").value_counts()
        total = counts.sum()

        pieces = []
        remaining = n

        for g, cnt in counts.items():
            k = max(1, int(round(n * (cnt / total))))
            k = min(k, cnt)
            take = df_part[df_part["target_group"].fillna("UNKNOWN") == g].sample(
                n=k, random_state=RANDOM_SEED
            )
            pieces.append(take)
            remaining -= k

        sampled = pd.concat(pieces, ignore_index=True)

        # Trim / top-up
        if len(sampled) > n:
            sampled = sampled.sample(n=n, random_state=RANDOM_SEED)
        elif len(sampled) < n and remaining > 0:
            rest = df_part[~df_part["id"].isin(sampled["id"])]
            if len(rest) > 0:
                top = rest.sample(n=min(remaining, len(rest)), random_state=RANDOM_SEED)
                sampled = pd.concat([sampled, top], ignore_index=True)

        return sampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    samp_hate = stratified_sample_by_group(df_hate, half)
    samp_ben  = stratified_sample_by_group(df_ben, half)

    samp = pd.concat([samp_hate, samp_ben], ignore_index=True).sample(
        frac=1.0, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    samp.to_csv(OUT_SAMP, index=False)
    print(f"[OK] Saved sampled ToxiGen: {OUT_SAMP} ({len(samp)} rows)")
    print("\n[INFO] Sample gold_label distribution:")
    print(samp["gold_label"].value_counts())
    print("\n[INFO] Top target_group counts:")
    print(samp["target_group"].value_counts().head(15))


if __name__ == "__main__":
    main()
