# Trustfulness Axis

Evaluates the factual reliability, calibration, and prediction accuracy of LLMs on social media content classification tasks.

## Dataset

**TrustfulnessAxis_v1_3000.csv** — 3,000 samples merged from three source datasets:
- **HASOC** (1,000 samples) — Hate speech and offensive content
- **SBIC** (1,000 samples) — Social bias inference corpus
- **ToxiGen** (1,000 samples) — Machine-generated toxic language

Each sample contains: `id`, `text`, `source_dataset`, `gold_label`, `needs_fact_check`, `gold_uncertainty`, `trustfulness_type`

## Scripts

| Script | Description |
|--------|-------------|
| `run_trustfulness_qwen_family.py` | Evaluate Qwen 2.5 models (1.5B, 3B, 7B, 14B) |
| `run_trustfulness_deepseek_family.py` | Evaluate DeepSeek-R1-Distill models (7B, 14B) |
| `run_trustfulness_llama_family.py` | Evaluate LLaMA 3.x models (1B, 3B, 8B) |
| `trustfulness_analysis.py` | Generate analysis visualizations and LaTeX tables |
| `merge_and_shuffle_trustfulness.py` | Merge source datasets into unified format |

## Data Preparation

The `data_preparation/` directory contains scripts for preparing each source dataset:
- `code_hasoc/` — HASOC dataset preparation and annotation
- `code_sbic/` — SBIC dataset preparation  
- `code_toxigen/` — ToxiGen dataset preparation and metadata merging

## Usage

```bash
# Run evaluation for Qwen family
python run_trustfulness_qwen_family.py

# Run evaluation for DeepSeek family
python run_trustfulness_deepseek_family.py

# Run evaluation for LLaMA family
python run_trustfulness_llama_family.py

# Generate analysis (after all evaluations complete)
python trustfulness_analysis.py
```

## Output

Each evaluation produces:
- `predictions_<model_name>.csv` — Per-sample predictions
- `metrics_<model_name>.json` — Accuracy, precision, recall, F1, calibration metrics

## Features

- Checkpointing with resume support
- Strict JSON parsing with regex fallback
- Supports BF16/FP16 inference
- Batch processing with progress tracking
