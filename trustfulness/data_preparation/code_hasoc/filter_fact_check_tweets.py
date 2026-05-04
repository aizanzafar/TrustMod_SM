"""
Filter HASOC tweets that likely need fact-checking using LLM.

Heuristic + LLM:
- Heuristic picks tweets with numbers / factual keywords.
- Those tweets go to the LLM (mistral-7b-instruct) via OpenAI-compatible endpoint.
- LLM answers only "yes"/"no" to: "Does this post require a web search
  to check factual claims?"
- Output: needs_fact_check.csv with a boolean column.
"""

import os
import re
import pandas as pd
from typing import List

# Use the same client style as test_api.py (OpenAI-compatible endpoint)
from openai import OpenAI

###############################################################################
# Configuration
###############################################################################

INPUT_FILE = os.environ.get("HASOC_INPUT_FILE", "en_Hasoc2021_train.csv")
OUTPUT_FILE = os.environ.get("FACT_CHECK_OUTPUT_FILE", "needs_fact_check.csv")

# If True, send every tweet to LLM. If False, use heuristic first.
ALWAYS_USE_LLM = False

FACT_KEYWORDS = [
    "percent", "%", "reported", "according", "study", "survey", "data",
    "increased", "decreased", "million", "billion", "thousand",
    "cases", "deaths", "vaccinated", "doses", "confirmed", "economic",
    "inflation", "gdp", "population", "poll", "rate",
]


def load_dataset(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find the dataset at {csv_path}. "
            "Set HASOC_INPUT_FILE env var if it lives somewhere else."
        )
    return df


def contains_factual_cues(text: str) -> bool:
    text = text.lower() if isinstance(text, str) else ""
    # any digit?
    if re.search(r"\d", text):
        return True
    # any keyword?
    return any(kw in text for kw in FACT_KEYWORDS)


def llm_requires_fact_check(client: OpenAI, text: str) -> bool:
    """
    Ask LLM (Mistral)-7b-instruct if this tweet needs fact checking.
    """
    prompt = (
        "You are an assistant that determines whether a social media post "
        "contains specific factual claims that warrant verification with a web search. "
        "Answer with only 'yes' or 'no'.\n\n"
        f"Post: {text}\n\n"
        "Does this post require a web search to check factual claims?"
    )
    try:
        resp = client.chat.completions.create(
            model="mistral-7b-instruct",   # Model, not gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3,
            temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    except Exception as exc:
        # On error, log and fall back to "no" so script keeps running
        print(f"Warning: LLM call failed for tweet: {exc}")
        return False


def main() -> None:
    # 1) Load data
    df = load_dataset(INPUT_FILE)
    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column with tweets.")

    total = len(df)
    print(f"Starting fact-check filter on {total} tweets from {INPUT_FILE}...")

    # 2) Init OpenAI client
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable LLM_API_KEY is not set.")
    client = OpenAI(
        api_key=api_key,
        base_url="https://your-llm-endpoint.example.com/v1",
    )

    # 3) Run heuristic + LLM with progress + checkpoints
    requires_fact_check: List[bool] = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        text = str(row["text"])
        needs_check = False

        # Heuristic gate
        if ALWAYS_USE_LLM or contains_factual_cues(text):
            needs_check = llm_requires_fact_check(client, text)

        requires_fact_check.append(needs_check)

        # If flagged, print the tweet snippet to CLI
        if needs_check:
            snippet = text.replace("\n", " ")
            if len(snippet) > 140:
                snippet = snippet[:137] + "..."
            print(f"[{i}/{total}] FACT-CHECK ▶ {snippet}")

        # Progress + checkpoint every 50 tweets
        if i % 50 == 0:
            print(f"[Checkpoint] Processed {i}/{total} tweets...")

            # Build a partial dataframe for processed tweets only
            df_partial = df.iloc[:i].copy()
            df_partial["requires_fact_check"] = requires_fact_check

            # Save only flagged tweets so far
            df_partial_flagged = df_partial[df_partial["requires_fact_check"]].copy()
            df_partial_flagged.to_csv(OUTPUT_FILE, index=False)

            print(
                f"[Checkpoint] Saved {len(df_partial_flagged)} flagged tweets "
                f"to {OUTPUT_FILE}"
            )

    # 4) Final save after all tweets processed
    df["requires_fact_check"] = requires_fact_check
    df_filtered = df[df["requires_fact_check"]].copy()
    df_filtered.to_csv(OUTPUT_FILE, index=False)

    print(
        f"✅ Done. Wrote {len(df_filtered)} tweets requiring a fact check to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
