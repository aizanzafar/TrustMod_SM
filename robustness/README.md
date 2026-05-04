# Robustness Axis

Evaluates LLM resilience against adversarial obfuscation attacks on social media content.

## Dataset

**sbic_obfuscated.jsonl** — 6,848 samples derived from SBIC with three obfuscation types:
- **Visual obfuscation** — Character substitutions (e.g., leetspeak, homoglyphs)
- **Structural obfuscation** — Word/sentence reordering, insertion of irrelevant tokens
- **Semantic obfuscation** — Paraphrasing, euphemisms, coded language

## Scripts

| Script | Description |
|--------|-------------|
| `run_robustness_hf.py` | Main evaluation runner (HuggingFace Transformers) |
| `main.py` | Alternative runner using Ollama local server |
| `evaluator.py` | Compute metrics from evaluation results |
| `robustness_analysis.py` | Generate analysis, visualizations, and LaTeX tables |

## Configuration

`config.yaml` contains project paths, model configurations, and evaluation parameters.

## Usage

```bash
# Run evaluation with HuggingFace models
python run_robustness_hf.py \
    --input sbic_obfuscated.jsonl \
    --out_dir results \
    --model_list Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct \
    --resume \
    --torch_dtype bf16

# Alternative: Run with Ollama (requires running Ollama server)
python main.py

# Compute metrics from results
python evaluator.py

# Generate analysis
python robustness_analysis.py
```

## Output

- Per-model JSONL files with predictions
- Metrics: accuracy, precision, recall, F1 per obfuscation type
- Attack success rates per strategy
- Degradation plots (clean vs. obfuscated performance)

## Metrics

- **Overall Accuracy** — Classification performance on obfuscated inputs
- **Attack Success Rate** — Percentage of correctly classified clean samples that become misclassified after obfuscation
- **Robustness Score** — Ratio of obfuscated to clean performance
- **Per-strategy breakdown** — Performance degradation by obfuscation type
