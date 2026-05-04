# Fairness Axis

Evaluates demographic fairness of LLM content moderation decisions across protected groups.

## Dataset

**fairness_merged_dataset_normalized.csv** — 3,298 samples with:
- Social media text content
- Gold labels (offensive/safe)
- Protected demographic attributes (race, gender, religion, etc.)
- Demographic group annotations

## Scripts

| Script | Description |
|--------|-------------|
| `deepseek_14b_cached.py` | Evaluate DeepSeek-R1-Distill (7B, 14B) on fairness dataset |
| `qwen_14b_cached.py` | Evaluate Qwen2.5-14B on fairness dataset |
| `n.py` | Evaluate Qwen2.5-14B with one-shot prompting |
| `m.py` | LLaMA 3.2 11B Vision multimodal fairness evaluation |
| `s.py` | DeepSeek-R1-Distill multimodal evaluation with reasoning tokens |
| `t.py` | LLaMA 3.2 11B Vision multimodal evaluation (Hateful Memes format) |

## Usage

```bash
# Run DeepSeek fairness evaluation
python deepseek_14b_cached.py

# Run Qwen fairness evaluation
python qwen_14b_cached.py

# Run Qwen one-shot evaluation
python n.py
```

## Fairness Metrics

- **Demographic Parity Difference (DPD)** — Difference in positive prediction rates across groups
- **Equalized Odds Difference (EOD)** — Difference in TPR/FPR across groups
- **False Positive Rate per group** — Rate of safe content incorrectly flagged, per demographic
- **Per-group accuracy** — Classification accuracy disaggregated by protected attribute

## Output

Each evaluation produces:
- `fairness_metrics.json` — Aggregate fairness metrics
- `detailed_results.csv` — Per-sample predictions with demographic metadata
- `summary.json` — Overall summary statistics
