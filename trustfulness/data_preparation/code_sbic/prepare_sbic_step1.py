# prepare_sbic_step1.py
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Defaults (adjust to your repo)
# -----------------------------
DEFAULT_IN = r"Datasets/SBIC.v2/SBIC.v2.agg.trn.csv"
DEFAULT_OUT_FULL = r"Datasets/SBIC/SBIC_step1_full.csv"
DEFAULT_OUT_SAMP = r"Datasets/SBIC/SBIC_step1_sampled_1000.csv"

RANDOM_SEED = 42
TARGET_N = 1000

# Thresholds for SBIC aggregated mean scores (0..1)
HIGH = 0.66
MID_LO = 0.33
MID_HI = 0.66


def stable_id(text: str, prefix: str = "sbic") -> str:
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}_{h}"


def clean_post(x) -> str:
    """Normalize SBIC post text and remove empty/garbage posts."""
    if pd.isna(x):
        return ""
    s = str(x)

    # SBIC posts often have lots of newlines and leading quotes
    s = s.replace("\r", "\n")
    s = "\n".join([ln.strip() for ln in s.split("\n")]).strip()

    # Remove wrapping quotes if the whole thing is quoted
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()

    # Collapse repeated whitespace
    s = " ".join(s.split()).strip()

    return s


def safe_json_list(x):
    """
    Aggregated SBIC uses json.dumps for list-like fields.
    We load them with json.loads.
    """
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if s == "" or s == "[]":
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        # Fallback: sometimes values look like python list strings
        # We'll do a conservative parse attempt
        try:
            import ast
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else []
        except Exception:
            return []


def map_trustfulness_type(stereotype_present: bool,
                          offensive: float,
                          intent: float,
                          has_target_minority: bool) -> str:
    """
    SBIC Trustfulness mapping logic.

    Buckets:
      - clear_toxic: high offense+intent (explicit)
      - implicit_toxic: stereotype present but not strongly explicit
      - ambiguous_offense: mid-range offense or intent
      - benign_minority: mentions minority groups but low offense/intent and no stereotypes
      - benign_other: everything else benign (keeps SBIC 'Bible verse' style posts separate)
    """
    offensive = float(offensive) if pd.notna(offensive) else 0.0
    intent = float(intent) if pd.notna(intent) else 0.0

    if stereotype_present:
        # If stereotypes exist, it's toxic by implication; explicitness decided by scores
        if (offensive >= HIGH) or (intent >= HIGH):
            return "clear_toxic"
        return "implicit_toxic"

    if (offensive >= HIGH) and (intent >= HIGH):
        return "clear_toxic"

    if (MID_LO <= offensive < MID_HI) or (MID_LO <= intent < MID_HI):
        return "ambiguous_offense"

    if has_target_minority:
        return "benign_minority"

    return "benign_other"


def map_gold_label(trustfulness_type: str) -> str:
    """
    Keep it compatible with HASOC/ToxiGen-style 'gold_label':
      - HATE: implicit toxicity stereotypes (or severe group harm)
      - OFFENSE: explicit insults/offense without stereotype implication
      - BENIGN: benign buckets
    """
    if trustfulness_type == "implicit_toxic":
        return "HATE"
    if trustfulness_type == "clear_toxic":
        return "OFFENSE"
    if trustfulness_type == "ambiguous_offense":
        return "OFFENSE"
    return "BENIGN"


def map_uncertainty(trustfulness_type: str, offensive: float, intent: float) -> str:
    """
    Uncertainty logic aligned with your trustfulness axis:
      - implicit/ambiguous/benign_minority tend to be uncertainty-heavy
      - clear_toxic is clearer
    """
    offensive = float(offensive) if pd.notna(offensive) else 0.0
    intent = float(intent) if pd.notna(intent) else 0.0

    if trustfulness_type in {"implicit_toxic", "ambiguous_offense", "benign_minority"}:
        return "AMBIGUOUS_NEEDS_REVIEW"

    # Also mark mid-range numeric evidence as ambiguous
    if (MID_LO <= offensive < MID_HI) or (MID_LO <= intent < MID_HI):
        return "AMBIGUOUS_NEEDS_REVIEW"

    return "CLEAR"


def stratified_sample(df: pd.DataFrame, n: int, by_cols: list[str]) -> pd.DataFrame:
    """
    Stratified sampling across multiple columns by grouping.
    If some groups are tiny, it gracefully falls back to random top-up.
    """
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    g = df.groupby(by_cols, dropna=False)
    sizes = g.size().reset_index(name="cnt")
    total = sizes["cnt"].sum()

    pieces = []
    taken_ids = set()

    # proportional allocation with at least 1 per group if possible
    for _, row in sizes.iterrows():
        frac = row["cnt"] / total
        k = int(round(n * frac))
        k = max(1, k)
        key = tuple(row[c] for c in by_cols)

        part = g.get_group(key)
        k = min(k, len(part))

        samp = part.sample(n=k, random_state=RANDOM_SEED)
        pieces.append(samp)
        taken_ids.update(samp["id"].tolist())

    out = pd.concat(pieces, ignore_index=True)

    # trim if overshoot
    if len(out) > n:
        out = out.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
        return out

    # top up if undershoot
    if len(out) < n:
        remaining = n - len(out)
        rest = df[~df["id"].isin(taken_ids)]
        if len(rest) > 0:
            top = rest.sample(n=min(remaining, len(rest)), random_state=RANDOM_SEED)
            out = pd.concat([out, top], ignore_index=True)

    return out.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    ap.add_argument("--out-full", dest="out_full", default=DEFAULT_OUT_FULL)
    ap.add_argument("--out-sampled", dest="out_samp", default=DEFAULT_OUT_SAMP)
    ap.add_argument("--target-n", type=int, default=TARGET_N)
    ap.add_argument("--min-implicit", type=int, default=0)
    ap.add_argument("--min-ambiguous", type=int, default=0)

    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_full = Path(args.out_full)
    out_samp = Path(args.out_samp)
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_samp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Required columns for aggregated SBIC
    required = {
        "post", "targetMinority", "targetCategory", "targetStereotype",
        "whoTarget", "intentYN", "sexYN", "offensiveYN", "dataSource", "hasBiasedImplication"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {in_path.name}: {missing}")

    # Clean post + drop empty
    df["text"] = df["post"].apply(clean_post)
    before = len(df)
    df = df[df["text"].str.len() > 0].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Dropped {dropped} rows with empty/blank post.")

    # Parse list fields (they are json strings in agg files)
    df["targetMinority_list"] = df["targetMinority"].apply(safe_json_list)
    df["targetCategory_list"] = df["targetCategory"].apply(safe_json_list)
    df["targetStereotype_list"] = df["targetStereotype"].apply(safe_json_list)

    # Features
    df["stereotype_present"] = df["targetStereotype_list"].apply(lambda x: len(x) > 0)
    df["has_target_minority"] = df["targetMinority_list"].apply(lambda x: len(x) > 0)

    # Ensure numeric
    df["intentYN"] = pd.to_numeric(df["intentYN"], errors="coerce").fillna(0.0)
    df["offensiveYN"] = pd.to_numeric(df["offensiveYN"], errors="coerce").fillna(0.0)
    df["whoTarget"] = pd.to_numeric(df["whoTarget"], errors="coerce")  # can be NaN in some rows

    # Stable id
    df["id"] = df["text"].apply(lambda t: stable_id(t, prefix="sbic"))

    # Core trustfulness fields (IMPORTANT: trustfulness_type included here)
    df["source_dataset"] = "SBIC"
    df["needs_fact_check"] = False

    df["trustfulness_type"] = df.apply(
        lambda r: map_trustfulness_type(
            stereotype_present=bool(r["stereotype_present"]),
            offensive=float(r["offensiveYN"]),
            intent=float(r["intentYN"]),
            has_target_minority=bool(r["has_target_minority"]),
        ),
        axis=1
    )

    df["gold_label"] = df["trustfulness_type"].apply(map_gold_label)
    df["gold_uncertainty"] = df.apply(
        lambda r: map_uncertainty(
            trustfulness_type=r["trustfulness_type"],
            offensive=float(r["offensiveYN"]),
            intent=float(r["intentYN"]),
        ),
        axis=1
    )

    # Keep your Step-1 normalized schema + SBIC analysis fields
    out = df[[
        "id",
        "text",
        "source_dataset",
        "gold_label",
        "needs_fact_check",
        "gold_uncertainty",
        "trustfulness_type",
        # SBIC-specific analysis/slicing fields (useful later)
        "intentYN",
        "offensiveYN",
        "whoTarget",
        "dataSource",
        "targetMinority",     # keep raw json-string (already in file)
        "targetCategory",
        "targetStereotype",
    ]].copy()

    out.to_csv(out_full, index=False)
    print(f"[OK] Saved full normalized SBIC: {out_full} ({len(out)} rows)")
    print("[INFO] trustfulness_type distribution (full):")
    print(out["trustfulness_type"].value_counts())

    # -------------------------
    # Sampling (balanced across trustfulness_type + gold_label)
    # -------------------------
    n = int(args.target_n)

    # Prefer stratification by trustfulness_type and gold_label
    sampled = stratified_sample(out, n=n, by_cols=["trustfulness_type", "gold_label"])

    # ---- Optional: enforce minimum counts per trustfulness bucket ----
    min_implicit = int(getattr(args, "min_implicit", 0))
    min_ambiguous = int(getattr(args, "min_ambiguous", 0))

    def topup(sampled_df, full_df, bucket, min_k):
        cur = (sampled_df["trustfulness_type"] == bucket).sum()
        if cur >= min_k:
            return sampled_df
        need = min_k - cur
        pool = full_df[(full_df["trustfulness_type"] == bucket) & (~full_df["id"].isin(sampled_df["id"]))].copy()
        if len(pool) == 0:
            return sampled_df
        add = pool.sample(n=min(need, len(pool)), random_state=RANDOM_SEED)
        return pd.concat([sampled_df, add], ignore_index=True)

    sampled = topup(sampled, out, "implicit_toxic", min_implicit)
    sampled = topup(sampled, out, "ambiguous_offense", min_ambiguous)

    # If we overshot target n, trim back down (keep diversity)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        sampled = sampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


    sampled.to_csv(out_samp, index=False)
    print(f"[OK] Saved sampled SBIC: {out_samp} ({len(sampled)} rows)")
    print("[INFO] trustfulness_type distribution (sampled):")
    print(sampled["trustfulness_type"].value_counts())
    print("[INFO] gold_label distribution (sampled):")
    print(sampled["gold_label"].value_counts())


if __name__ == "__main__":
    main()
