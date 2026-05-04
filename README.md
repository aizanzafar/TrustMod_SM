# TrustMod-SM: A Multi-Axis Benchmark for Evaluating Trustworthiness of LLMs in Social Media Content Moderation

This repository contains the evaluation codebase for **TrustMod-SM**, a comprehensive multi-axis benchmark designed to evaluate the trustworthiness of Large Language Models (LLMs) in social media content moderation tasks.

## Overview

TrustMod-SM evaluates LLMs across **six dimensions** of trustworthiness:

| Dimension | Description |
|-----------|-------------|
| **Trustfulness** | Evaluates factual reliability and calibration of LLM predictions on social media content |
| **Robustness** | Tests resilience against obfuscation attacks (visual, structural, semantic) |
| **Fairness** | Measures demographic parity and equalized odds across protected groups |
| **Contextual Integrity (CI)** | Assesses consistency of moderation decisions across demographic contexts |
| **Safety** | Evaluates vulnerability to concealment and exaggeration manipulation strategies |
| **Multimodal Fairness** | Extends fairness evaluation to Vision-Language Models (VLMs) with text+image inputs |

Additionally, a **Distributed Capability Model (DCM)** analysis provides a holistic view of model capabilities.

## Models Evaluated

We evaluate three major LLM families across multiple scales:

- **Qwen 2.5**: 0.5B, 1.5B, 3B, 7B, 14B (Instruct)
- **LLaMA 3.x**: 1B, 3B, 8B (Instruct)
- **DeepSeek-R1-Distill**: 1.5B, 7B, 14B

For multimodal evaluation:
- **Qwen2-VL**: 2B, 7B
- **LLaMA 3.2 Vision**: 11B

## Repository Structure

```
├── README.md
├── requirements.txt
├── trustfulness/          # Trustfulness axis evaluation
├── robustness/            # Robustness axis evaluation
├── fairness/              # Fairness axis evaluation
├── contextual_integrity/  # Contextual Integrity evaluation
├── safety/                # Safety axis evaluation
├── multimodal/            # Multimodal fairness evaluation
└── dcm/                   # Distributed Capability Model analysis
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (GPU required)
- Minimum 48GB VRAM (NVIDIA RTX 6000 Ada or equivalent recommended)

### Setup

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/TrustMod-SM
cd TrustMod-SM

# Create conda environment
conda create -n trustmod python=3.10 -y
conda activate trustmod

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Each dimension can be evaluated independently. See individual dimension READMEs for detailed instructions.

### Example: Running Trustfulness Evaluation

```bash
cd trustfulness/
python run_trustfulness_qwen_family.py
```

### Example: Running Robustness Evaluation

```bash
cd robustness/
python run_robustness_hf.py --input <path_to_dataset> --out_dir results --model_list Qwen/Qwen2.5-7B-Instruct --torch_dtype bf16
```

### Example: Running DCM Analysis

```bash
cd dcm/
python dcm_analysis.py --input dcm_scores.csv --output_dir results/
```

## Datasets

Each dimension uses task-specific datasets:

| Dimension | Dataset | Samples | Source |
|-----------|---------|---------|--------|
| Trustfulness | TrustfulnessAxis_v1 | 3,000 | HASOC + SBIC + ToxiGen |
| Robustness | SBIC-Obfuscated | 6,848 | SBIC with 3 obfuscation types |
| Fairness | Fairness Merged | 3,298 | Curated with demographic attributes |
| CI | HateCheck-CI | 3,436 | HateCheck with CI schema |
| Safety (Concealment) | Concealment | 1002 × 4 strategies | Curated toxic samples |
| Safety (Exaggeration) | Exaggeration | 1000 × 4 strategies | COVID tweets |
| Multimodal | TrustMod-Multimodal | 2,500+ | Text + image pairs |

> **Note:** Datasets are available separately. Please refer to the dataset documentation in each dimension's directory.

## Hardware Requirements

| Dimension | Minimum VRAM | Recommended |
|-----------|-------------|-------------|
| Trustfulness | 24GB | 48GB (RTX 6000 Ada) |
| Robustness | 24GB | 48GB |
| Fairness | 48GB | 48GB |
| CI | 24GB | 48GB |
| Safety | 24GB | 48GB |
| Multimodal | 48GB | 2× 48GB |

## Citation

```bibtex
@inproceedings{trustmodsm2026,
  title={TrustMod-SM: A Multi-Axis Benchmark for Evaluating Trustworthiness of LLMs in Social Media Content Moderation},
  author={},
  booktitle={NeurIPS 2026 Datasets and Benchmarks Track},
  year={2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work was conducted as part of the TrustMod-SM project for evaluating LLM trustworthiness in social media content moderation.
