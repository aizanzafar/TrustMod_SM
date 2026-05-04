# Multimodal Fairness Evaluation

Extends fairness evaluation to Vision-Language Models (VLMs) by assessing moderation performance on text+image multimodal inputs.

## Dataset

- **trustmod_multimodal_fairness_dataset.csv** — Multimodal dataset with image IDs, captions, labels, and protected attribute annotations
- **trustmod_images/** — 2,500+ PNG images (referenced by ID in the CSV)
- **dataset_card.json** — Dataset metadata card

## Models Evaluated

| Model | Parameters | Type |
|-------|-----------|------|
| Qwen2-VL-2B | 2B | Vision-Language |
| Qwen2-VL-7B | 7B | Vision-Language |
| LLaMA 3.2 11B Vision | 11B | Vision-Language |

## Scripts

| Script | Description |
|--------|-------------|
| `server_vlm_fairness.py` | Main multimodal fairness evaluation with 3 evaluation modes |
| `server_vlm_fairness2.py` | Extended/alternative version |
| `m.py` | Multimodal evaluation utility script |
| `s.py` | Multimodal evaluation utility script |

## Evaluation Modes

1. **Text-only** — Model receives only text caption, no image
2. **Image-only** — Model receives only image, no text
3. **Both (Multimodal)** — Model receives both text and image

This design enables measurement of the **Multimodal Semantic Gap** — differences in fairness when modality inputs are varied.

## Usage

```bash
# Run full evaluation (all modes)
python server_vlm_fairness.py --mode both

# Run text-only evaluation
python server_vlm_fairness.py --mode text

# Run image-only evaluation
python server_vlm_fairness.py --mode image

# Quick sanity check (15 samples)
python server_vlm_fairness.py --test
```

## Output

Results are organized by model and mode:
```
vlm_results/
├── qwen2vl_2b/
│   ├── text/
│   ├── image/
│   └── both/
├── qwen2vl_7b/
│   ├── text/
│   ├── image/
│   └── both/
└── llama_11b_vision/
    ├── text/
    ├── image/
    └── both/
```

Each contains:
- `fairness_metrics.json` — Per-mode fairness metrics
- `detailed_results.csv` — Per-sample predictions
- `predictions.json` — Raw model outputs

## Hardware Requirements

- Minimum: 48GB VRAM (RTX 6000 Ada)
- Recommended: 2× 48GB for parallel evaluation
- Supports FP16/BF16 inference (no 4-bit quantization)
