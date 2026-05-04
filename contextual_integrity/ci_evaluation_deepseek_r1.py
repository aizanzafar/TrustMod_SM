#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   CONTEXTUAL INTEGRITY (CI) EVALUATION – DeepSeek R1 DISTILL FAMILY        ║
║   TrustMod-SM  |  Anonymous Institution                                     ║
║                                                                              ║
║   Server  : <user>@<server> – 2× RTX 6000 Ada (48GB each)                   ║
║   Dataset : ./hatecheck_ci_schema1.csv  (3,436 rows)               ║
║   Models  : DeepSeek-R1-Distill-Qwen-7B                                   ║
║             DeepSeek-R1-Distill-Qwen-14B                                  ║
║                                                                              ║
║   ⚠️  R1-SPECIFIC: These are REASONING models – they generate              ║
║       <think>...</think> blocks before the final answer.                   ║
║       MAX_NEW_TOKENS is set large (512) to accommodate reasoning.          ║
║       parse_output() strips <think> blocks and reads AFTER </think>.       ║
║                                                                              ║
║   Metrics : Accuracy, F1, CS-FPR, RL-FPR, ND-FPR, CI Score               ║
║             + Per-demographic breakdown for all 5 protected groups         ║
║                                                                              ║
║   Features:                                                                  ║
║     ✅ Reasoning token stripping (<think>...</think>)                       ║
║     ✅ Loads from local cache (no re-download needed)                       ║
║     ✅ 15-sample sanity test → y/n prompt → full inference                  ║
║     ✅ Results saved immediately after each model (crash-safe)             ║
║     ✅ DeepSeek-style progress: speed + ETA + GPU memory                  ║
║     ✅ apply_chat_template – auto token format from tokenizer              ║
║     ✅ Resumes from saved results if interrupted                           ║
║                                                                              ║
║   Run:                                                                       ║
║     export HF_TOKEN=hf_xxx  (optional – not needed for cached models)     ║
║     python ci_evaluation_deepseek_r1.py                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, gc, json, sys, time, re, warnings
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_PATH = os.path.expanduser("./hatecheck_ci_schema1.csv")
OUTPUT_DIR   = os.path.expanduser("./ci_results/")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "ci_results_deepseek_r1.json")
CACHE_DIR    = "~/.cache/huggingface/hub/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Inference settings ──────────────────────────────────────────────────────
# ⚠️  R1-CRITICAL: Reasoning models need large MAX_NEW_TOKENS
#     DeepSeek-R1 generates <think>...</think> before answering.
#     The thinking chain can be 100-400 tokens even for simple tasks.
#     Setting this too low (e.g. 15) will TRUNCATE the thinking mid-way,
#     leaving no room for the actual HATE/NOT_HATE label → wrong predictions.
#
#     Recommended values:
#       512  – covers ~95% of cases, fast
#       1024 – safest but ~2× slower
#
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.01   # near-deterministic (same as Llama/Qwen evals for fairness)
TOP_P          = 0.9
TEST_N         = 15

# ── Model configs ────────────────────────────────────────────────────────────
#
# DeepSeek-R1 Distill variants available:
#   1.5B → DeepSeek-R1-Distill-Qwen-1.5B   (distilled into Qwen2.5-1.5B backbone)
#   7B   → DeepSeek-R1-Distill-Qwen-7B     (distilled into Qwen2.5-7B backbone)
#   14B  → DeepSeek-R1-Distill-Qwen-14B    (distilled into Qwen2.5-14B backbone)
#   Also available: Qwen-32B, Llama-8B, Llama-70B
#
# VRAM estimates (FP16):
#   7B   → ~16GB  → runs on single RTX 6000 Ada (48GB)
#   14B  → ~32GB  → runs on single RTX 6000 Ada (48GB)
#
MODEL_CONFIGS = [
    {
        "hf_name":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "short_name": "DeepSeek-R1-7B",
        "quantize":   False,    # FP16, ~16GB VRAM – fits on one 48GB GPU
        "batch_size": 16,       # Smaller batch – 512 output tokens adds memory
        "description": "DeepSeek R1 Distill Qwen 7B FP16 (~16GB VRAM)",
        "is_r1": True,
    },
    {
        "hf_name":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "short_name": "DeepSeek-R1-14B",
        "quantize":   False,    # FP16, ~32GB VRAM – fits on one 48GB GPU
        "batch_size": 8,        # Smaller batch – larger model + 512 output tokens
        "description": "DeepSeek R1 Distill Qwen 14B FP16 (~32GB VRAM)",
        "is_r1": True,
    },
]

# Protected characteristics in hatecheck_ci_schema1.csv
DEMO_GROUPS = ["gender", "lgbtq", "race", "religion", "disability"]


# ═══════════════════════════════════════════════════════════════════════════════
# HF AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════

def setup_hf_auth():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ℹ️  No HF_TOKEN env var found.")
        print("   Cached models don't need it. If model is NOT cached, run:")
        print("   export HF_TOKEN=hf_your_token_here")
        return None
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print(f"✅ HF authenticated: {token[:8]}...{token[-4:]}")
        return token
    except Exception as e:
        print(f"⚠️  HF login failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT – CI-aware binary classification
# ═══════════════════════════════════════════════════════════════════════════════

def create_prompt(tokenizer, text: str) -> str:
    """
    One-shot CI-aware prompt using tokenizer.apply_chat_template().

    Note for R1 models:
      The prompt is the SAME as Llama/Qwen evals (same content, same CI rules).
      The R1 model will first emit <think>...</think> reasoning tokens,
      then produce the HATE/NOT_HATE label. parse_output() handles stripping.

    CI-critical rules explicitly listed:
      • Counterspeech condemning hate  → NOT_HATE
      • Reclaimed language by in-group → NOT_HATE
      • Neutral factual mention        → NOT_HATE
      • Direct attack/derogation       → HATE
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a content moderation classifier.\n"
                "Label the text as HATE or NOT_HATE using these rules:\n\n"
                "HATE:\n"
                "  - Text that directly attacks, demeans, or dehumanises a person/group\n"
                "    based on race, gender, religion, sexual orientation, or disability.\n\n"
                "NOT_HATE (pay special attention to these):\n"
                "  - COUNTERSPEECH: text that discusses or condemns hate/discrimination\n"
                "    (e.g. 'Racism is wrong', 'We must stand against homophobia')\n"
                "  - RECLAIMED LANGUAGE: slurs or sensitive words used respectfully\n"
                "    or reclaimed by the targeted community itself\n"
                "  - NEUTRAL MENTION: factual reference to a demographic group\n"
                "    without attack (e.g. 'Gay people face discrimination')\n"
                "  - GENERAL NON-HATE: any text without group-based attack\n\n"
                "Answer with ONLY the label first, then a brief reason.\n"
                "Do NOT include your thinking process in the final answer."
            )
        },
        # One-shot example – counterspeech (hardest CI case)
        {
            "role": "user",
            "content": 'Text: "We need to stand up against the hatred directed at gay people."'
        },
        {
            "role": "assistant",
            "content": "NOT_HATE. Reason: Counterspeech condemning discrimination, not attacking any group."
        },
        # Actual classification target
        {
            "role": "user",
            "content": f'Text: "{text}"'
        }
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Safety fallback: merge system into first user message
        merged = messages[0]["content"] + "\n\n" + messages[1]["content"]
        fallback_messages = [
            {"role": "user",      "content": merged},
            {"role": "assistant", "content": messages[2]["content"]},
            {"role": "user",      "content": messages[3]["content"]},
        ]
        return tokenizer.apply_chat_template(
            fallback_messages,
            tokenize=False,
            add_generation_prompt=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT PARSING – R1-specific with <think> stripping
# ═══════════════════════════════════════════════════════════════════════════════

def strip_reasoning_tokens(text: str) -> str:
    """
    DeepSeek-R1 outputs reasoning inside <think>...</think> tags
    before the final answer. This function strips those tokens.

    Examples:
      Input:  "<think>\nLet me analyze...\nThe text condemns hate.\n</think>\nNOT_HATE. Reason: ..."
      Output: "NOT_HATE. Reason: ..."

      Input:  "<think>This is hateful content targeting...</think> HATE. Reason: ..."
      Output: "HATE. Reason: ..."

      Input:  "NOT_HATE."  (rare: model skips thinking for obvious cases)
      Output: "NOT_HATE."

    Edge cases handled:
      - Incomplete <think> block (truncated by MAX_NEW_TOKENS) → fallback to full text
      - Multiple <think> blocks → strip all of them
      - <think> with no closing tag → use everything after last </think> if present,
        otherwise use full text (triggers parse fallback to NOT_HATE default)
    """
    if "<think>" not in text:
        return text  # No reasoning block – plain model or obvious case

    # Strip ALL <think>...</think> blocks (handles multiple blocks edge case)
    stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    stripped = stripped.strip()

    if stripped:
        return stripped

    # Fallback: </think> not found (truncated) → try text after last </think>
    # If that fails too, use full text (parse_output will default to NOT_HATE)
    parts = text.split('</think>')
    if len(parts) > 1:
        return parts[-1].strip()

    return text  # truncated reasoning, no answer found → parse_output defaults


def parse_output(text: str) -> int:
    """
    Parse model response → binary label.

    ⚠️  R1-aware: Always call strip_reasoning_tokens() BEFORE this function
        OR pass raw output and let this handle it (it calls strip internally).

    Handles:
      - R1 reasoning tokens via strip_reasoning_tokens()
      - Llama-style '\n\nNOT_HATE' with leading whitespace → strip()
      - Lowercase 'not_hate' → .upper()
      - Model preamble before label → search first 80 chars (R1 final answer
        sometimes has "My answer: NOT_HATE" or similar)

    Returns: 1 = HATE, 0 = NOT_HATE
    """
    # Strip reasoning tokens first (no-op if no <think> block)
    clean = strip_reasoning_tokens(text)

    text_upper = clean.strip().upper()
    # 80 chars for R1 – final answer sometimes has brief preamble
    first_80   = text_upper[:80]

    # Check NOT_HATE before HATE (specific match avoids "NOT_HATE" triggering "HATE")
    not_hate_match = re.search(r'\bNOT[_\s]HATE\b', first_80)
    hate_match     = re.search(r'\bHATE\b', first_80)

    if not_hate_match and hate_match:
        return 0 if not_hate_match.start() < hate_match.start() else 1
    if not_hate_match:
        return 0
    if hate_match:
        return 1

    # Secondary signals (fallback when model doesn't use exact labels)
    for w in ["NOT", "NO ", "NON", "COUNTER", "NEUTRAL", "RECLAIM", "SAFE", "BENIGN"]:
        if w in first_80:
            return 0
    for w in ["ATTACK", "TOXIC", "HARMFUL", "OFFENSIVE", "DEROG"]:
        if w in first_80:
            return 1

    return 0  # default: NOT_HATE (conservative – avoids over-policing CI samples)


# ═══════════════════════════════════════════════════════════════════════════════
# GPU HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    parts = []
    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        parts.append(f"G{i}:{used:.1f}/{total:.0f}GB")
    return " ".join(parts)

def print_gpu_status():
    if not torch.cuda.is_available():
        print("   ❌ No GPU – CPU only"); return
    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        name  = torch.cuda.get_device_name(i)
        print(f"   GPU {i} [{name}]: {used:.1f}GB / {total:.0f}GB")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(cfg: dict, hf_token: Optional[str] = None):
    """
    Load model from local cache (local_files_only=True).
    Falls back to downloading if not in cache.
    """
    model_id   = cfg["hf_name"]
    quantize   = cfg["quantize"]
    short_name = cfg["short_name"]

    print(f"\n{'='*80}")
    print(f"🔄 LOADING: {model_id}")
    print(f"   Mode  : {'4-bit quantized' if quantize else 'FP16'}")
    print(f"   Cache : {CACHE_DIR}")
    print(f"   ⚠️  R1 MODEL: Output includes reasoning tokens (stripped at parse)")
    print(f"   ⚠️  MAX_NEW_TOKENS={MAX_NEW_TOKENS} to accommodate <think> blocks")
    print(f"{'='*80}")

    # ── Check if model is in cache ──────────────────────────────────────────
    cache_name       = "models--" + model_id.replace("/", "--")
    cache_model_path = os.path.join(CACHE_DIR, cache_name)
    use_local        = os.path.exists(cache_model_path)

    if use_local:
        print(f"   ✅ Found in cache → loading locally (no download)")
    else:
        print(f"   ⚠️  Not in cache → will download from HuggingFace")
        if not hf_token:
            print(f"   ⚠️  No HF_TOKEN – set via: export HF_TOKEN=hf_xxx")

    common_kwargs = dict(
        trust_remote_code=True,
        token=hf_token,
        cache_dir=CACHE_DIR,
        local_files_only=use_local,
    )

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batch generation
    print("✅ Tokenizer loaded")

    # ── Quantization config ──────────────────────────────────────────────────
    bnb_cfg = None
    if quantize and torch.cuda.is_available():
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("   📦 4-bit quantization enabled")
    else:
        print("   📦 FP16 (no quantization)")

    # ── Model ────────────────────────────────────────────────────────────────
    print("📥 Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_cfg,
        **common_kwargs
    )
    model.eval()

    allocated = sum(
        torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())
    ) / 1e9
    reserved = sum(
        torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count())
    ) / 1e9
    print(f"\n✅ {short_name} loaded!")
    print(f"💾 VRAM: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    if torch.cuda.device_count() > 1:
        print(f"📊 Multi-GPU distribution:")
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1e9
            print(f"   GPU {i}: {mem:.2f} GB")

    return model, tokenizer


def clear_model(model, tokenizer, short_name: str):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print(f"🧹 {short_name} cleared from memory")


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH INFERENCE – R1-aware with reasoning token handling
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(model, tokenizer, texts: List[str],
                  batch_size: int, label: str = "",
                  debug_n: int = 0) -> List[int]:
    """
    Batched inference with live progress. R1-aware.

    Args:
        debug_n: If > 0, print raw output for first N samples (useful for
                 verifying <think> stripping works correctly during sanity test)

    Returns: list of int predictions (1=HATE, 0=NOT_HATE)
    """
    device     = next(model.parameters()).device
    results    = []
    total      = len(texts)
    bs         = batch_size
    start_time = time.time()
    debug_count = 0

    print(f"\n🚀 STARTING BATCHED EVALUATION: {label}")
    print(f"📊 Total samples: {total} | Batch size: {bs} | MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
    print(f"⚠️  R1 mode: reasoning tokens will be stripped before parsing")

    for i in range(0, total, bs):
        batch   = texts[i : i + bs]
        prompts = [create_prompt(tokenizer, t) for t in batch]

        try:
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,  # input context length limit
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,  # 512 – covers reasoning chain
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=False,                # greedy – consistent with other evals
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_len = input_ids.shape[1]
            for seq in outputs:
                new_toks    = seq[prompt_len:]
                raw_resp    = tokenizer.decode(new_toks, skip_special_tokens=True)
                pred        = parse_output(raw_resp)  # strips <think> internally
                results.append(pred)

                # Debug: print raw + stripped output for first debug_n samples
                if debug_count < debug_n:
                    stripped = strip_reasoning_tokens(raw_resp)
                    think_len = len(raw_resp) - len(stripped)
                    print(f"\n   [DEBUG #{debug_count+1}]")
                    print(f"   Raw output (first 200 chars): {raw_resp[:200]!r}")
                    print(f"   After <think> strip: {stripped[:100]!r}")
                    print(f"   Reasoning chars stripped: {think_len}")
                    print(f"   Final prediction: {'HATE' if pred == 1 else 'NOT_HATE'}")
                    debug_count += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                bs = max(1, bs // 2)
                print(f"\n⚠️  OOM – halving batch to {bs}, retrying...")
                torch.cuda.empty_cache()
                for t in batch:
                    try:
                        enc = tokenizer(
                            [create_prompt(tokenizer, t)],
                            return_tensors="pt", truncation=True, max_length=1024
                        )
                        iids = enc["input_ids"].to(device)
                        with torch.no_grad():
                            out = model.generate(
                                iids, max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id
                            )
                        raw  = tokenizer.decode(out[0][iids.shape[1]:], skip_special_tokens=True)
                        results.append(parse_output(raw))
                    except Exception:
                        results.append(0)
            else:
                print(f"\n⚠️  Batch error at {i}: {e} – defaulting NOT_HATE")
                results.extend([0] * len(batch))

        # ── DeepSeek-style progress ──────────────────────────────────────────
        done    = min(i + batch_size, total)
        elapsed = time.time() - start_time
        speed   = done / elapsed if elapsed > 0 else 0
        eta     = (total - done) / speed if speed > 0 else 0
        pct     = done / total * 100
        print(
            f"\r   [{done:5d}/{total}] {pct:5.1f}% | "
            f"⚡ {speed:5.1f} samp/s | "
            f"⏱  ETA: {eta/60:4.1f}m | "
            f"💾 {gpu_mem_str()}",
            end="", flush=True
        )

        if torch.cuda.is_available() and (i // batch_size) % 20 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    peak = sum(
        torch.cuda.max_memory_allocated(i)
        for i in range(torch.cuda.device_count())
    ) / 1e9
    print(f"\n\n✅ COMPLETE!")
    print(f"⏱  Total: {elapsed/60:.2f} mins | ⚡ Avg Speed: {total/elapsed:.1f} samp/s")
    print(f"💾 Peak Memory: {peak:.2f} GB")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CI METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CIMetrics:
    model_name:        str
    total_samples:     int   = 0
    ci_samples:        int   = 0
    # Overall
    overall_accuracy:  float = 0.0
    overall_f1:        float = 0.0
    overall_precision: float = 0.0
    overall_recall:    float = 0.0
    # CI FPRs – core metric (↓ lower = better = less over-policing)
    cs_fpr:       float = 0.0;  cs_accuracy: float = 0.0;  cs_n: int = 0
    rl_fpr:       float = 0.0;  rl_accuracy: float = 0.0;  rl_n: int = 0
    nd_fpr:       float = 0.0;  nd_accuracy: float = 0.0;  nd_n: int = 0
    ci_fpr_score: float = 0.0   # weighted avg – headline CI metric
    # Confusion matrix on CI-relevant samples
    tp: int = 0;  fp: int = 0;  tn: int = 0;  fn: int = 0
    # Per-demographic
    demo_accuracy: Dict = field(default_factory=dict)
    demo_ci_fpr:   Dict = field(default_factory=dict)
    demo_counts:   Dict = field(default_factory=dict)
    # R1-specific metadata
    max_new_tokens_used: int = MAX_NEW_TOKENS


def compute_ci_metrics(df: pd.DataFrame,
                       preds: List[int],
                       model_name: str) -> CIMetrics:
    df = df.copy()
    df["pred"] = preds
    y_true = df["label"].values
    y_pred = df["pred"].values

    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    def subset_stats(sub_df):
        if len(sub_df) == 0:
            return 0.0, 0.0, 0
        fpr_val = float((sub_df["pred"] == 1).sum() / len(sub_df))
        acc_val = float(accuracy_score(sub_df["label"], sub_df["pred"]))
        return fpr_val, acc_val, len(sub_df)

    cs_fpr, cs_acc, cs_n = subset_stats(df[df["ci_sub_dimension"] == "CS"])
    rl_fpr, rl_acc, rl_n = subset_stats(df[df["ci_sub_dimension"] == "RL"])
    nd_fpr, nd_acc, nd_n = subset_stats(df[df["ci_sub_dimension"] == "ND"])

    total_ci = cs_n + rl_n + nd_n
    ci_score = (
        (cs_fpr * cs_n + rl_fpr * rl_n + nd_fpr * nd_n) / total_ci
        if total_ci > 0 else 0.0
    )

    ci_df = df[df["ci_relevant"] == True]
    tp = int(((ci_df["label"] == 1) & (ci_df["pred"] == 1)).sum())
    fp = int(((ci_df["label"] == 0) & (ci_df["pred"] == 1)).sum())
    tn = int(((ci_df["label"] == 0) & (ci_df["pred"] == 0)).sum())
    fn = int(((ci_df["label"] == 1) & (ci_df["pred"] == 0)).sum())

    demo_acc, demo_fpr, demo_cnt = {}, {}, {}
    for demo in DEMO_GROUPS:
        sub = df[df["demographic_group"] == demo]
        if len(sub) == 0:
            continue
        demo_acc[demo] = float(accuracy_score(sub["label"], sub["pred"]))
        demo_cnt[demo] = len(sub)
        sub_ci = sub[sub["ci_relevant"] == True]
        demo_fpr[demo] = (
            float((sub_ci["pred"] == 1).sum() / len(sub_ci))
            if len(sub_ci) > 0 else None
        )

    return CIMetrics(
        model_name=model_name, total_samples=len(df), ci_samples=total_ci,
        overall_accuracy=float(acc),
        overall_f1=float(f1), overall_precision=float(prec),
        overall_recall=float(rec),
        cs_fpr=cs_fpr, cs_accuracy=cs_acc, cs_n=cs_n,
        rl_fpr=rl_fpr, rl_accuracy=rl_acc, rl_n=rl_n,
        nd_fpr=nd_fpr, nd_accuracy=nd_acc, nd_n=nd_n,
        ci_fpr_score=float(ci_score),
        tp=tp, fp=fp, tn=tn, fn=fn,
        demo_accuracy=demo_acc, demo_ci_fpr=demo_fpr, demo_counts=demo_cnt,
        max_new_tokens_used=MAX_NEW_TOKENS,
    )


def print_ci_metrics(m: CIMetrics):
    W = "=" * 80
    print(f"\n{W}")
    print(f"📊 RESULTS SUMMARY: {m.model_name}  [R1 – MAX_NEW_TOKENS={m.max_new_tokens_used}]")
    print(W)

    print(f"\n🎯 Overall Performance:")
    print(f"   Accuracy  : {m.overall_accuracy*100:.2f}%")
    print(f"   Precision : {m.overall_precision*100:.2f}%")
    print(f"   Recall    : {m.overall_recall*100:.2f}%")
    print(f"   F1 Score  : {m.overall_f1*100:.2f}%")

    print(f"\n⚖️  CONTEXTUAL INTEGRITY METRICS  (↓ FPR = model understands context)")
    print(f"{'Dimension':<28} {'FPR (↓)':>9} {'Accuracy':>10} {'N':>6}")
    print("-" * 58)
    print(f"  Counterspeech     (CS)    {m.cs_fpr*100:>8.2f}% {m.cs_accuracy*100:>9.2f}% {m.cs_n:>6}")
    print(f"  Reclaimed Lang    (RL)    {m.rl_fpr*100:>8.2f}% {m.rl_accuracy*100:>9.2f}% {m.rl_n:>6}")
    print(f"  Neutral Mention   (ND)    {m.nd_fpr*100:>8.2f}% {m.nd_accuracy*100:>9.2f}% {m.nd_n:>6}")
    print(f"  {'CI Score (weighted avg)':<27} {m.ci_fpr_score*100:>8.2f}%")

    print(f"\n📋 Confusion Matrix (CI-relevant samples – all should be NOT_HATE):")
    print(f"                     Predicted:")
    print(f"                   HATE   NOT_HATE")
    print(f"  Actual HATE       {m.tp:4d}     {m.fn:4d}   (rare in CI subset)")
    print(f"  Actual NOT_HATE   {m.fp:4d}     {m.tn:4d}   ← FP = CI failures")

    print(f"\n👥 PER-DEMOGRAPHIC BREAKDOWN:")
    print(f"{'Demographic':<14} {'Accuracy':>10} {'CI FPR (↓)':>12} {'N':>6}")
    print("-" * 47)
    for demo in DEMO_GROUPS:
        if demo in m.demo_accuracy:
            acc_s = f"{m.demo_accuracy[demo]*100:.2f}%"
            fv    = m.demo_ci_fpr.get(demo)
            fpr_s = f"{fv*100:.2f}%" if fv is not None else "   N/A"
            n     = m.demo_counts.get(demo, 0)
            print(f"  {demo:<12}   {acc_s:>9}   {fpr_s:>10}  {n:>5}")
    print(W)


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE / RESUME
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(all_results: Dict, path: str):
    out = {k: asdict(v) for k, v in all_results.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n💾 Results saved → {path}")


def load_existing_results(path: str):
    if not os.path.exists(path):
        return {}, set()
    with open(path) as f:
        raw = json.load(f)
    all_results = {}
    for k, v in raw.items():
        m = CIMetrics(model_name=k)
        for fname, val in v.items():
            setattr(m, fname, val)
        all_results[k] = m
    completed = set(raw.keys())
    print(f"\n⚡ Resuming – already completed: {completed}")
    return all_results, completed


# ═══════════════════════════════════════════════════════════════════════════════
# SANITY TEST (15 stratified samples) – with R1 debug output
# ═══════════════════════════════════════════════════════════════════════════════

def run_sanity_test(model, tokenizer, df: pd.DataFrame,
                   short_name: str, batch_size: int) -> None:
    print(f"\n{'='*60}")
    print(f"🧪 SANITY TEST – {short_name} – {TEST_N} samples")
    print(f"   ⚠️  R1: Will print raw outputs to verify <think> stripping")
    print(f"{'='*60}")

    parts = []
    for ctx in ["counterspeech", "reclaimed", "neutral_mention", "actual_hate"]:
        sub = df[df["context_type"] == ctx]
        n   = min(4, len(sub))
        if n > 0:
            parts.append(sub.sample(n=n, random_state=42))
    test_df = pd.concat(parts).head(TEST_N).reset_index(drop=True)

    print(f"📋 Context type distribution in test:")
    print(test_df["context_type"].value_counts().to_string())
    print()

    t0    = time.time()
    # debug_n=3 → print raw R1 output for first 3 samples to verify stripping
    preds = run_inference(model, tokenizer, test_df["text"].tolist(),
                          batch_size=batch_size,
                          label=f"{short_name} Sanity",
                          debug_n=3)
    elapsed = time.time() - t0

    metrics = compute_ci_metrics(test_df, preds, short_name)
    print_ci_metrics(metrics)

    lmap = {1: "HATE    ", 0: "NOT_HATE"}
    print(f"\n🔍 Row-by-row predictions (all {TEST_N} test samples):")
    print(f"{'#':<3} {'Context':<18} {'Gold':<9} {'Pred':<9} {'OK':<3}  Text")
    print("-" * 82)
    for i, row in test_df.iterrows():
        gold = lmap[row["label"]]
        pred = lmap[preds[i]]
        ok   = "✅" if row["label"] == preds[i] else "❌"
        sub  = row.get("ci_sub_dimension", "  ")
        print(f"{i:<3} {row['context_type']:<18} {gold:<9} {pred:<9} {ok}"
              f"  [{sub}] {str(row['text'])[:55]}")

    print(f"\n⏱  Sanity test time: {elapsed:.1f}s")
    acc = accuracy_score(test_df["label"], preds)
    ci_sub = test_df[test_df["ci_relevant"] == True]
    ci_fpr = (
        (pd.Series(preds)[ci_sub.index] == 1).sum() / len(ci_sub)
        if len(ci_sub) > 0 else 0.0
    )
    print(f"🎯 Test Accuracy : {acc*100:.1f}%  |  CI FPR : {ci_fpr*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

def ask_proceed(model_name: str) -> bool:
    while True:
        try:
            ans = input(
                f"\n▶️  Proceed with FULL inference on {model_name}? [y/n]: "
            ).strip().lower()
        except EOFError:
            print("\n   (non-interactive – defaulting to y)")
            return True
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", "skip", "s"):
            print(f"⏭️  Skipping {model_name} – moving to next model.")
            return False
        print("   Please enter y or n.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  CONTEXTUAL INTEGRITY EVALUATION – DeepSeek R1 DISTILL FAMILY")
    print("  TrustMod-SM  |  Anonymous Institution")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ⚠️  R1 REASONING MODE: MAX_NEW_TOKENS={MAX_NEW_TOKENS} per sample")
    print(f"  ⚠️  <think> blocks are stripped before parsing – see parse_output()")
    print("=" * 80)

    print(f"\n🖥️  System:")
    print(f"   PyTorch : {torch.__version__}")
    print(f"   CUDA    : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("❌ No GPU – exiting"); sys.exit(1)
    print(f"   GPUs    : {torch.cuda.device_count()}")
    print_gpu_status()

    hf_token = setup_hf_auth()

    print(f"\n📥 Loading dataset: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print(f"❌ File not found: {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Loaded {len(df)} rows")
    print(f"\n   context_type distribution:")
    print(df["context_type"].value_counts().to_string())
    ci_df = df[df["ci_relevant"] == True]
    print(f"\n   CI-relevant : {len(ci_df)} samples")
    print(f"\n   demographic × ci_sub_dimension:")
    print(ci_df.groupby(["demographic_group", "ci_sub_dimension"])
          .size().unstack(fill_value=0).to_string())

    # Warn about runtime estimate for R1 models
    print(f"\n⏱️  RUNTIME ESTIMATE (R1 with MAX_NEW_TOKENS={MAX_NEW_TOKENS}):")
    print(f"   DeepSeek-R1-1.5B @ ~30 samp/s → ~{len(df)//30//60+1} min for {len(df)} samples")
    print(f"   DeepSeek-R1-7B   @ ~10 samp/s → ~{len(df)//10//60+1} min for {len(df)} samples")
    print(f"   (Slower than plain models due to reasoning chain generation)")

    all_results, completed = load_existing_results(RESULTS_FILE)

    # ── Evaluate each model ──────────────────────────────────────────────────
    for cfg in MODEL_CONFIGS:
        name = cfg["short_name"]

        print(f"\n{'🔷'*40}")
        print(f"🔷 MODEL: {name}  |  {cfg['description']}")
        print(f"{'🔷'*40}")

        if name in completed:
            print(f"   ✅ Already completed – skipping")
            continue

        try:
            model, tokenizer = load_model(cfg, hf_token)
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            import traceback; traceback.print_exc()
            continue

        try:
            run_sanity_test(model, tokenizer, df, name, cfg["batch_size"])
        except Exception as e:
            print(f"⚠️  Sanity test error: {e}")

        if not ask_proceed(name):
            clear_model(model, tokenizer, name)
            continue

        print(f"\n📥 Starting FULL inference: {len(df)} samples ...")
        try:
            t0    = time.time()
            preds = run_inference(model, tokenizer, df["text"].tolist(),
                                  cfg["batch_size"], label=name)
            t_tot = time.time() - t0
        except Exception as e:
            print(f"❌ Inference failed for {name}: {e}")
            import traceback; traceback.print_exc()
            clear_model(model, tokenizer, name)
            continue

        print(f"\n⚖️  Computing CI metrics...")
        metrics = compute_ci_metrics(df, preds, name)
        print_ci_metrics(metrics)

        out_df = df.copy()
        out_df["prediction"] = preds
        out_df["correct"]    = (out_df["label"] == out_df["prediction"]).astype(int)
        out_df["raw_output"] = ""
        pred_path = os.path.join(OUTPUT_DIR, f"preds_{name.replace('/', '_')}.csv")
        out_df.to_csv(pred_path, index=False)
        print(f"💾 Predictions → {pred_path}")

        all_results[name] = metrics
        save_results(all_results, RESULTS_FILE)

        speed = len(df) / t_tot
        print(f"📝 Speed: {speed:.1f} samp/s | Total: {t_tot/60:.1f} min")

        clear_model(model, tokenizer, name)

    # ── Final Comparison Table ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("✅ ALL EVALUATIONS COMPLETE")
    print(f"{'='*80}")

    if all_results:
        print(f"\n{'Model':<25} {'Acc':>7} {'CI FPR':>8} {'CS-FPR':>8} {'RL-FPR':>8} {'ND-FPR':>8} {'F1':>7}")
        print("-" * 76)
        for nm, m in all_results.items():
            print(
                f"  {nm:<23} "
                f"{m.overall_accuracy*100:>6.2f}% "
                f"{m.ci_fpr_score*100:>7.2f}% "
                f"{m.cs_fpr*100:>7.2f}% "
                f"{m.rl_fpr*100:>7.2f}% "
                f"{m.nd_fpr*100:>7.2f}% "
                f"{m.overall_f1*100:>6.2f}%"
            )
    print(f"\n💾 Results → {RESULTS_FILE}")
    print(f"   End time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
