# Contextual Integrity (CI) Axis

Evaluates the consistency of LLM moderation decisions across different demographic contexts, based on the Contextual Integrity framework.

## Dataset

**hatecheck_ci_schema1.csv** — 3,436 test cases designed to probe CI violations across 5 protected demographic groups. Each test case varies the demographic context while keeping the offense structure constant.

## Scripts

| Script | Description |
|--------|-------------|
| `ci.py` | CI evaluation for Qwen 2.5 family (1.5B, 3B, 7B, 14B) |
| `deepseek.py` | CI evaluation for DeepSeek-R1-Distill (1.5B, 7B, 14B) |
| `llamaci.py` | CI evaluation for LLaMA 3.x family (1B, 3B, 8B) |
| `ci_evaluation_deepseek_r1.py` | Extended CI evaluation for DeepSeek with reasoning-token handling |

## Usage

```bash
# Run CI evaluation for Qwen family
python ci.py

# Run CI evaluation for DeepSeek family
python deepseek.py

# Run CI evaluation for LLaMA family
python llamaci.py
```

## CI Metrics

- **CI Score** — Overall contextual integrity score measuring decision consistency
- **FPR (False Positive Rate)** — Per-demographic false positive rates
- **RL-FPR (Relative FPR)** — Relative false positive rate across groups
- **ND-FPR (Normalized Difference FPR)** — Normalized difference in FPR between demographic groups

## Output

Each evaluation produces:
- `ci_results_<model_family>.json` — Aggregate CI metrics per model
- `preds_<model_name>.csv` — Per-sample predictions with demographic labels

## Features

- Crash-safe with resumable checkpointing
- Reasoning token handling (`<think>...</think>` stripping for DeepSeek)
- Auto chat template formatting via tokenizer
- Supports MAX_NEW_TOKENS configuration for reasoning models
