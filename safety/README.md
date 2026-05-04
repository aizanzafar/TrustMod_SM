# Safety Axis

Evaluates LLM vulnerability to adversarial manipulation strategies that aim to either **conceal** toxic content or **exaggerate** benign content.

## Sub-Axes

### Concealment
Tests whether adversarial prompt strategies can cause LLMs to miss genuinely toxic content.

- **Dataset:** 116 toxic social media samples
- **Strategies:** 4 concealment prompt strategies (self-referential framing, etc.)
- **Total evaluations:** 116 × 4 = 464 test cases

### Exaggeration
Tests whether adversarial prompt strategies can cause LLMs to flag benign content as harmful.

- **Dataset:** 100 COVID-related tweets (factual/benign)
- **Strategies:** 4 exaggeration/sensationalization strategies
- **Total evaluations:** 100 × 4 = 400 test cases

## Scripts

| Script | Description |
|--------|-------------|
| `run_safety_hf.py` | Unified HuggingFace runner for both concealment and exaggeration |
| `safety_analysis.py` | Analysis and visualization for NeurIPS paper |
| `concealment/pe-safety-concealment-test-dynamic.ipynb` | Concealment evaluation notebook |
| `exaggeration/pe-safety-exaggeration-test.ipynb` | Exaggeration evaluation notebook |

## Data Files

```
concealment/
├── dataset/
│   └── concealment_116_samples.csv     # Toxic samples for concealment testing
└── all_strategy_prompt_self.json        # Strategy prompt templates

exaggeration/
├── dataset/
│   ├── Constraint_Train_Real_Cleaned.csv  # Real/benign tweets
│   └── Constraint_Train_Fake_Cleaned.csv  # Fake tweets for reference
└── pe-safety-exaggeration-test.ipynb      # Evaluation notebook
```

## Usage

```bash
# Run unified safety evaluation
python run_safety_hf.py \
    --concealment_data concealment/dataset/concealment_116_samples.csv \
    --exaggeration_data exaggeration/dataset/Constraint_Train_Real_Cleaned.csv \
    --concealment_prompts concealment/all_strategy_prompt_self.json \
    --out_dir results \
    --model_list Qwen/Qwen2.5-7B-Instruct \
    --torch_dtype bf16

# Generate analysis
python safety_analysis.py
```

## Safety Metrics

### Concealment Metrics
- **Concealment Score** — Rate at which toxic content is missed under adversarial prompts
- **Toxicity Drop** — Reduction in detected toxicity under concealment strategies
- **Abstention Rate** — Rate of model refusal to classify

### Exaggeration Metrics
- **Factual Drift Score** — Degree to which model labels shift toward "harmful" for benign content
- **Fake Drift Rate** — Proportion of benign content misclassified as harmful
- **CSS (Content Safety Score)** — Combined safety score across strategies
