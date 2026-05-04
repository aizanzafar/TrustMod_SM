"""
annotate_trustfulness_type_hybrid.py

Hybrid (heuristic + LLM (Mistral)) annotator for trustfulness_type buckets:

Buckets:
- fact_check
- ambiguous_offense
- implicit_toxic
- benign_minority
- clear_toxic

Design:
1) Deterministic: fact_check if needs_fact_check=True
2) Heuristic high-precision:
   - BENIGN + protected group mention -> benign_minority
   - HATE/OFFENSE + explicit profanity/slur -> clear_toxic
3) LLM fallback (few-shot) for remaining:
   - ambiguous_offense vs implicit_toxic vs clear_toxic vs benign_minority
   - returns ONLY one of the 5 bucket strings

Outputs:
- <out_csv>: id, trustfulness_type
- <out_debug_csv>: id, trustfulness_type, method, llm_raw, llm_reason

Env:
- LLM_API_KEY: LLM API key
"""

import argparse
import os
import re
import time
from typing import Dict, Optional, Tuple

import pandas as pd
from openai import OpenAI


# -----------------------------
# Lightweight lexicons (you can expand later)
# -----------------------------
# Keep these minimal + high-precision. Avoid overfitting.
PROFANITY = [
    "fuck", "shit", "bitch", "asshole", "slut", "bastard", "whore",
    "mc", "madarchod", "bhenchod", "chutiya", "harami"
]

# A small protected group list (expand carefully)
PROTECTED_GROUPS = [
    "muslim", "hindu", "christian", "sikh", "jain", "buddhist",
    "dalit", "brahmin", "sc", "st", "obc",
    "woman", "women", "girl", "girls", "man", "men",
    "gay", "lesbian", "trans", "transgender",
    "black", "white",
    "indian", "pakistani", "bangladeshi"
]

# Ambiguity / sarcasm cues (high precision subset)
SARCASM_CUES = ["yeah right", "sure buddy", "as if", "lol", "lmao", "😂", "🤣", "/s"]


def contains_any(text: str, words) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)


def normalize_bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    # tolerate True/False, 0/1, "true"/"false"
    return df[col].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


# -----------------------------
# Few-shot prompt for bucket assignment
# -----------------------------
ALLOWED_BUCKETS = ["fact_check", "ambiguous_offense", "implicit_toxic", "benign_minority", "clear_toxic"]

FEWSHOT_PROMPT = f"""
You are labeling social media posts into ONE trustfulness behavior bucket.

Choose exactly ONE label from:
{", ".join(ALLOWED_BUCKETS)}

Definitions:
- fact_check: post contains a factual claim that should be verified with external evidence/tools (numbers, statistics, "according to", claims about events, etc.).
- benign_minority: benign/non-offensive post that mentions a protected group (e.g., religion, caste, nationality, gender, sexual orientation) and should NOT be over-flagged.
- clear_toxic: explicit hate/offense with clear abusive profanity/slurs/attacks.
- implicit_toxic: toxic or hateful content that is coded, indirect, insinuated, or lacks explicit profanity; still harmful.
- ambiguous_offense: borderline offense where meaning depends on context/sarcasm/uncertainty and should often be escalated.

Return ONLY the label string (no extra text).

Examples:
Post: "As per WHO, the vaccine caused 10,000 deaths."
Label: fact_check

Post: "Muslim doctors are doing great work."
Label: benign_minority

Post: "You are a f***ing idiot."
Label: clear_toxic

Post: "They should go back where they came from."
Label: implicit_toxic

Post: "Sure buddy, you're the smartest person alive 😂"
Label: ambiguous_offense

Now label this post:
Post: {{POST}}
Label:
""".strip()


def llm_label_bucket(client: OpenAI, model: str, post: str, max_retries=3, sleep_sec=1.0) -> Tuple[str, str]:
    """
    Returns: (bucket_label, raw_text)
    """
    prompt = FEWSHOT_PROMPT.replace("{POST}", post)

    last = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            # normalize to first token-ish line
            label = raw.splitlines()[0].strip().lower()
            label = re.sub(r"[^a-z_]", "", label)  # remove punctuation
            if label in ALLOWED_BUCKETS:
                return label, raw
            # if model adds extra words, try to find a valid label substring
            for b in ALLOWED_BUCKETS:
                if b in raw.lower():
                    return b, raw
            return "implicit_toxic", raw  # conservative fallback
        except Exception as e:
            last = e
            print(f"[WARN] LLM call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(sleep_sec)

    return "implicit_toxic", f"__ERROR__ {last}"


def heuristic_bucket(row: pd.Series) -> Optional[str]:
    """
    Returns a bucket if heuristic is confident, else None.
    Priority order:
      fact_check -> benign_minority -> clear_toxic -> (else None)
    """
    text = str(row.get("text", ""))
    gold = str(row.get("gold_label", "")).strip().upper()
    needs_fc = bool(row.get("needs_fact_check", False))

    # 1) Deterministic: fact_check
    if needs_fc:
        return "fact_check"

    # 2) benign_minority: benign + protected group mention
    if gold == "BENIGN" and contains_any(text, PROTECTED_GROUPS):
        return "benign_minority"

    # 3) clear_toxic: explicit profanity and toxic label
    if gold in {"HATE", "OFFENSE"} and contains_any(text, PROFANITY):
        return "clear_toxic"

    # If ambiguous sarcasm cue present and OFFENSE, we can choose to label directly
    # but it's safer to let LLM decide for ambiguity/implicit split.
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Normalized HASOC CSV (id,text,gold_label,needs_fact_check,...)")
    ap.add_argument("--out_csv", required=True, help="Output CSV: id,trustfulness_type")
    ap.add_argument("--out_debug_csv", required=True, help="Debug CSV with method/raw")
    ap.add_argument("--model", default="mistral-7b-instruct", help="Model name")
    ap.add_argument("--base_url", default="https://your-llm-endpoint.example.com/v1")
    ap.add_argument("--api_key_env", default="LLM_API_KEY")
    ap.add_argument("--checkpoint_every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # normalize needs_fact_check column name (support both)
    if "needs_fact_check" not in df.columns and "requires_fact_check" in df.columns:
        df["needs_fact_check"] = normalize_bool_col(df, "requires_fact_check")
    elif "needs_fact_check" in df.columns:
        # keep as bool
        df["needs_fact_check"] = normalize_bool_col(df, "needs_fact_check")
    else:
        raise ValueError("Input must contain 'needs_fact_check' or 'requires_fact_check'.")

    required = {"id", "text", "gold_label", "needs_fact_check"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set.")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # Resume support
    done = {}
    debug_rows = []

    if args.resume and os.path.exists(args.out_debug_csv):
        prev = pd.read_csv(args.out_debug_csv)
        if "id" in prev.columns and "trustfulness_type" in prev.columns:
            for _, r in prev.iterrows():
                done[str(r["id"])] = str(r["trustfulness_type"])
            debug_rows = prev.to_dict(orient="records")
            print(f"[RESUME] Loaded {len(done)} rows from {args.out_debug_csv}")

    total = len(df)
    processed = 0

    for i, row in df.iterrows():
        rid = str(row["id"])
        if rid in done:
            continue

        text = str(row["text"])
        gold = str(row["gold_label"]).strip().upper()

        # 1) Heuristic
        bucket = heuristic_bucket(row)
        method = "heuristic"
        raw = ""
        reason = ""

        # 2) LLM fallback for the rest
        if bucket is None:
            method = "llm"
            bucket, raw = llm_label_bucket(client, args.model, text)
            reason = f"gold_label={gold}, needs_fact_check={bool(row['needs_fact_check'])}"

        done[rid] = bucket
        debug_rows.append({
            "id": rid,
            "trustfulness_type": bucket,
            "method": method,
            "gold_label": gold,
            "needs_fact_check": bool(row["needs_fact_check"]),
            "llm_raw": raw[:1000],
            "llm_reason": reason,
        })

        processed += 1

        # checkpoint
        if processed % args.checkpoint_every == 0:
            df_debug = pd.DataFrame(debug_rows)
            df_debug.to_csv(args.out_debug_csv, index=False)
            df_out = df_debug[["id", "trustfulness_type"]].drop_duplicates(subset=["id"])
            df_out.to_csv(args.out_csv, index=False)
            print(f"[CHECKPOINT] {processed} new rows processed. Saved -> {args.out_csv}")

        if processed % 10 == 0:
            print(f"[PROGRESS] processed {processed} new rows. last id={rid}")

    # final save
    df_debug = pd.DataFrame(debug_rows)
    df_debug.to_csv(args.out_debug_csv, index=False)
    df_out = df_debug[["id", "trustfulness_type"]].drop_duplicates(subset=["id"])
    df_out.to_csv(args.out_csv, index=False)

    print(f"[DONE] Wrote {len(df_out)} trustfulness_type labels to {args.out_csv}")
    print(df_out["trustfulness_type"].value_counts())


if __name__ == "__main__":
    main()
