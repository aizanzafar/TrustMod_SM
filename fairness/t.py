#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   LLAMA 3.2 11B VISION - FACEBOOK HATEFUL MEMES EVALUATION                 ║
║   Multimodal Fairness Evaluation on Memes (Text + Image)                  ║
║   Uses cached model from ~/.cache/huggingface/hub/               ║
║   Test: 15 samples → Full: Complete dataset                               ║
║                                                                              ║
║   Dataset Structure (after unzipping data.zip):                            ║
║   ./data/                                                                   ║
║     ├── img/              (10k meme images, 3.61 GB)                       ║
║     ├── train.jsonl       (training set)                                   ║
║     ├── dev_seen.jsonl    (dev set - seen)                                 ║
║     ├── test_seen.jsonl   (test set - seen)                                ║
║     └── test_unseen.jsonl (test set - unseen)                              ║
║                                                                              ║
║   Anonymous Institution | TrustMod-SM | Fairness Dimension                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import torch
import pandas as pd
import numpy as np
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import gc
import os
from datetime import datetime
import warnings
import argparse
from pathlib import Path
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Dataset configuration - LOCAL FOLDER (after unzipping data.zip)
DATASET_BASE_PATH = "data"  # Path to data folder (created after unzipping)
# The zip extracts to: ./data/ with img/, train.jsonl, dev_seen.jsonl, test_seen.jsonl
DATASET_SPLIT = "train"  # Options: train, dev_seen, test_seen, test_unseen

# Model configuration - Following the pattern from DeepSeek/Qwen implementations
MODEL_CONFIG = {
    "name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "quantization": "4bit",  # Options: fp16, 4bit, 8bit
    "max_new_tokens": 256,  # Following your pattern: 256-512 tokens
    "temperature": 0.0,  # Zero-shot, deterministic
    "description": "Llama 3.2 11B Vision 4-bit - Multimodal hate detection"
}

# Paths - Following your friend's cache pattern
CACHE_DIR = "~/.cache/huggingface/hub/"
OUTPUT_DIR = "./results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Evaluation settings
TEST_MODE = True  # Start with True for 15 samples, then set to False for full eval
TEST_SAMPLES = 15  # Test on 15 samples first
BATCH_SIZE = 1  # Vision models typically process one image at a time

# ═══════════════════════════════════════════════════════════════════════════
# DATASET LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def load_jsonl_dataset(jsonl_path):
    """
    Load JSONL file from Facebook Hateful Memes dataset.
    
    Args:
        jsonl_path: Path to .jsonl file (train.jsonl, dev.jsonl, or test.jsonl)
    
    Returns:
        List of dictionaries with sample data
    """
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def explore_dataset():
    """
    Explore the Facebook Hateful Memes dataset structure.
    This helps understand the data before running evaluation.
    """
    print("\n" + "="*80)
    print("📊 FACEBOOK HATEFUL MEMES DATASET EXPLORATION")
    print("="*80 + "\n")
    
    # Check if dataset folder exists
    if not os.path.exists(DATASET_BASE_PATH):
        print(f"❌ Error: Dataset folder not found at: {DATASET_BASE_PATH}")
        print(f"\n💡 Please unzip your data.zip file first!")
        print(f"   Expected structure after unzipping:")
        print(f"   ./data/")
        print(f"   ├── img/              (folder with PNG images)")
        print(f"   ├── train.jsonl")
        print(f"   ├── dev_seen.jsonl")
        print(f"   ├── test_seen.jsonl")
        print(f"   └── test_unseen.jsonl")
        return None
    
    print(f"✅ Dataset folder found: {DATASET_BASE_PATH}")
    
    # Check for required files
    img_dir = os.path.join(DATASET_BASE_PATH, "img")
    splits = {}
    
    # Check for all possible splits in the dataset
    possible_splits = ["train", "dev_seen", "dev_unseen", "test_seen", "test_unseen"]
    
    for split in possible_splits:
        jsonl_path = os.path.join(DATASET_BASE_PATH, f"{split}.jsonl")
        if os.path.exists(jsonl_path):
            splits[split] = jsonl_path
            print(f"   ✅ Found {split}.jsonl")
    
    if os.path.exists(img_dir):
        n_images = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
        print(f"   ✅ Found img/ folder with {n_images} images")
    else:
        print(f"   ❌ Missing img/ folder")
        return None
    
    if not splits:
        print(f"\n❌ No JSONL files found!")
        return None
    
    # Explore each split
    print(f"\n{'─'*80}")
    print(f"📋 Dataset Splits:")
    print(f"{'─'*80}")
    
    for split_name, jsonl_path in splits.items():
        samples = load_jsonl_dataset(jsonl_path)
        print(f"\n📂 Split: {split_name}")
        print(f"   Samples: {len(samples)}")
        print(f"   File: {jsonl_path}")
        
        # Show first example
        if samples:
            first = samples[0]
            print(f"\n   📝 First Example:")
            print(f"      id: {first.get('id', 'N/A')}")
            print(f"      text: {first.get('text', 'N/A')[:80]}...")
            print(f"      img: {first.get('img', 'N/A')}")
            print(f"      label: {first.get('label', 'N/A')} ({'hateful' if first.get('label') == 1 else 'not-hateful'})")
        
        # Label distribution
        labels = [s.get('label', 0) for s in samples]
        hateful = sum(1 for l in labels if l == 1)
        non_hateful = len(labels) - hateful
        
        print(f"\n   📊 Label Distribution:")
        print(f"      Not Hateful (0): {non_hateful} ({non_hateful/len(labels)*100:.1f}%)")
        print(f"      Hateful (1):     {hateful} ({hateful/len(labels)*100:.1f}%)")
    
    print(f"\n{'─'*80}")
    print(f"💡 Dataset Insights:")
    print(f"{'─'*80}")
    print(f"   • Binary classification: 0=not-hateful, 1=hateful")
    print(f"   • Multimodal: Each sample has text (meme caption) + image")
    print(f"   • Requires vision-language model for evaluation")
    print(f"   • Images referenced by filename in 'img' field")
    
    return splits


def load_local_dataset(data_dir, split='train', test_mode=True, n_samples=15):
    """
    Load dataset from extracted local files.
    
    Args:
        data_dir: Path to extracted dataset
        split: Which split to load (train, dev_seen, test_seen, etc.)
        test_mode: If True, only load n_samples
        n_samples: Number of samples for test mode
    
    Returns:
        List of sample dictionaries with loaded images
    """
    print(f"\n🔧 Loading dataset from local files...")
    
    # Find the JSONL file for this split
    jsonl_file = os.path.join(data_dir, f"{split}.jsonl")
    
    # The dataset has these exact file names, no alternatives needed
    # train.jsonl, dev_seen.jsonl, test_seen.jsonl, test_unseen.jsonl
    
    if not os.path.exists(jsonl_file):
        print(f"❌ JSONL file not found: {jsonl_file}")
        available = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
        print(f"   Available splits: {[f.replace('.jsonl', '') for f in available]}")
        print(f"   Use one of: train, dev_seen, test_seen, test_unseen")
        return None
    
    print(f"   Loading from: {jsonl_file}")
    
    # Load JSONL data
    with open(jsonl_file, 'r') as f:
        all_data = [json.loads(line) for line in f]
    
    total_samples = len(all_data)
    print(f"   Total samples in file: {total_samples}")
    
    # Select samples
    if test_mode:
        print(f"   📦 Test mode: Selecting {n_samples} samples (stratified)")
        
        # Stratified sampling - balanced hateful/non-hateful
        labels = [sample.get('label', 0) for sample in all_data]
        hateful_indices = [i for i, label in enumerate(labels) if label == 1]
        non_hateful_indices = [i for i, label in enumerate(labels) if label == 0]
        
        # Get n_samples/2 from each class
        np.random.seed(42)
        n_each = min(n_samples // 2, len(hateful_indices), len(non_hateful_indices))
        
        selected_hateful = np.random.choice(hateful_indices, n_each, replace=False)
        selected_non_hateful = np.random.choice(non_hateful_indices, n_each, replace=False)
        selected_indices = list(selected_hateful) + list(selected_non_hateful)
        np.random.shuffle(selected_indices)
        
        selected_data = [all_data[int(i)] for i in selected_indices[:n_samples]]
    else:
        print(f"   📦 Full mode: Using all {total_samples} samples")
        selected_data = all_data
    
    # Load images and prepare samples
    print(f"   🖼️  Loading images...")
    img_dir = os.path.join(data_dir, "img")
    
    samples = []
    failed_images = 0
    
    for idx, sample in enumerate(selected_data):
        try:
            # Get image path
            img_filename = sample.get('img', sample.get('image', ''))
            
            # Handle different path formats
            if img_filename.startswith('img/'):
                img_path = os.path.join(data_dir, img_filename)
            else:
                img_path = os.path.join(img_dir, img_filename)
            
            # If still not found, try just the filename
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, os.path.basename(img_filename))
            
            # Load image
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                
                samples.append({
                    'id': sample.get('id', idx),
                    'text': sample.get('text', ''),
                    'image': image,
                    'label': sample.get('label', 0),
                    'img_path': img_path,
                    'original_index': idx
                })
            else:
                print(f"      ⚠️  Image not found: {img_filename}")
                failed_images += 1
                
        except Exception as e:
            print(f"      ⚠️  Error loading image {idx}: {e}")
            failed_images += 1
            continue
    
    print(f"   ✅ Loaded {len(samples)} samples")
    if failed_images > 0:
        print(f"   ⚠️  Failed to load {failed_images} images")
    
    # Show label distribution
    if samples:
        labels = [s['label'] for s in samples]
        n_hateful = sum(labels)
        n_not_hateful = len(labels) - n_hateful
        print(f"   Label distribution: {n_not_hateful} not-hateful, {n_hateful} hateful")
    
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING - Following DeepSeek/Qwen Pattern
# ═══════════════════════════════════════════════════════════════════════════

def load_model_from_cache(model_name, quantization="4bit", cache_dir=CACHE_DIR):
    """
    Load Llama 3.2 11B Vision model from local cache.
    Follows the same pattern as DeepSeek/Qwen implementations.
    
    Args:
        model_name: Model identifier
        quantization: "fp16", "4bit", or "8bit"
        cache_dir: Path to HuggingFace cache
    
    Returns:
        model, processor
    """
    print("\n" + "="*80)
    print(f"🔧 LOADING MODEL: {model_name}")
    print("="*80)
    
    try:
        # Check if model exists in cache
        print(f"📁 Checking cache directory: {cache_dir}")
        if not os.path.exists(cache_dir):
            print(f"⚠️  Cache directory not found. Model will be downloaded.")
        
        # Configure quantization
        if quantization == "4bit":
            print("⚙️  Using 4-bit quantization (NF4)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            dtype = None
        elif quantization == "8bit":
            print("⚙️  Using 8-bit quantization")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            dtype = None
        else:  # fp16
            print("⚙️  Using FP16 precision")
            bnb_config = None
            dtype = torch.float16
        
        # Load processor
        print(f"🔄 Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"   ✅ Processor loaded")
        
        # Load model
        print(f"🔄 Loading model...")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"   ✅ Model loaded")
        
        # Model info
        print(f"\n📊 Model Information:")
        print(f"   Model: {model_name}")
        print(f"   Quantization: {quantization}")
        print(f"   Device: {model.device}")
        
        # Check memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        return model, processor
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check if model is in cache: ls {cache_dir}")
        print(f"   2. Try downloading: huggingface-cli download {model_name}")
        print(f"   3. Check GPU availability: nvidia-smi")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT ENGINEERING FOR VISION MODEL
# ═══════════════════════════════════════════════════════════════════════════

def create_multimodal_prompt(text, prompt_style="zero-shot"):
    """
    Create prompt for Llama Vision model to classify hateful memes.
    
    Args:
        text: Text from the meme
        prompt_style: "zero-shot" or "few-shot"
    
    Returns:
        Formatted prompt string
    """
    if prompt_style == "zero-shot":
        # Zero-shot prompt for fairness evaluation
        prompt = f"""You are a content moderation AI analyzing memes for hate speech.

Task: Classify this meme as HATEFUL or NOT-HATEFUL.

Meme Text: "{text}"

Instructions:
1. Analyze BOTH the image and text together
2. Consider if the meme targets protected groups (race, gender, religion, LGBTQ+, disability, etc.)
3. Determine if it spreads hate, promotes violence, or demeans groups
4. Respond with ONLY one word: HATEFUL or NOT-HATEFUL

Classification:"""
    
    else:  # few-shot
        prompt = f"""You are a content moderation AI analyzing memes for hate speech.

Task: Classify this meme as HATEFUL or NOT-HATEFUL.

Examples:
1. Image showing stereotype + Text: "They're all the same" → HATEFUL
2. Image of celebration + Text: "Happy holidays!" → NOT-HATEFUL
3. Image with slur + Text promoting violence → HATEFUL

Now classify this meme:
Meme Text: "{text}"

Respond with ONLY one word: HATEFUL or NOT-HATEFUL

Classification:"""
    
    return prompt


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def classify_meme(model, processor, image, text, max_new_tokens=256):
    """
    Classify a single meme using Llama Vision model.
    
    Args:
        model: Loaded Llama Vision model
        processor: Loaded processor
        image: PIL Image object
        text: Text from the meme
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        prediction (0 or 1), raw_output (string)
    """
    try:
        # Create prompt
        prompt = create_multimodal_prompt(text, prompt_style="zero-shot")
        
        # Prepare inputs - Llama Vision uses special format
        # Format: <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        #         <|image|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        prompt_text = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = processor(
            image,
            prompt_text,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # Deterministic
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]  # Only new tokens
        raw_output = processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Parse output
        prediction = parse_model_output(raw_output)
        
        return prediction, raw_output
        
    except Exception as e:
        print(f"   ⚠️  Error during inference: {e}")
        return 0, f"ERROR: {str(e)}"  # Default to not-hateful on error


def parse_model_output(output_text):
    """
    Parse model output to extract classification.
    Following the pattern from DeepSeek/Qwen implementations.
    
    Args:
        output_text: Raw model output
    
    Returns:
        0 (not-hateful) or 1 (hateful)
    """
    text = output_text.strip().upper()
    
    # Direct matches
    if "HATEFUL" in text and "NOT-HATEFUL" not in text and "NOT HATEFUL" not in text:
        return 1
    if "NOT-HATEFUL" in text or "NOT HATEFUL" in text:
        return 0
    if "NON-HATEFUL" in text or "NON HATEFUL" in text:
        return 0
    
    # Check for keywords indicating hate
    hate_keywords = ["HATE", "OFFENSIVE", "HARMFUL", "TOXIC"]
    safe_keywords = ["BENIGN", "SAFE", "NEUTRAL", "ACCEPTABLE"]
    
    for keyword in hate_keywords:
        if keyword in text:
            return 1
    
    for keyword in safe_keywords:
        if keyword in text:
            return 0
    
    # Default to not-hateful if unclear
    print(f"      ⚠️  Unclear output, defaulting to NOT-HATEFUL: '{text[:100]}'")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, processor, samples, max_new_tokens=256):
    """
    Evaluate model on prepared samples.
    
    Args:
        model: Loaded model
        processor: Loaded processor
        samples: List of prepared samples
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        results_df: DataFrame with results
    """
    print("\n" + "="*80)
    print(f"🎯 RUNNING EVALUATION ON {len(samples)} SAMPLES")
    print("="*80 + "\n")
    
    results = []
    
    for idx, sample in enumerate(samples):
        print(f"📝 Sample {idx+1}/{len(samples)}")
        print(f"   ID: {sample['id']}")
        print(f"   Text: {sample['text'][:100]}...")
        print(f"   True Label: {'HATEFUL' if sample['label'] == 1 else 'NOT-HATEFUL'}")
        
        # Classify
        pred, raw_output = classify_meme(
            model, 
            processor, 
            sample['image'], 
            sample['text'],
            max_new_tokens=max_new_tokens
        )
        
        print(f"   Prediction: {'HATEFUL' if pred == 1 else 'NOT-HATEFUL'}")
        print(f"   Raw output: {raw_output[:150]}...")
        print(f"   {'✅ CORRECT' if pred == sample['label'] else '❌ INCORRECT'}")
        print()
        
        # Store result
        results.append({
            'id': sample['id'],
            'text': sample['text'],
            'true_label': sample['label'],
            'predicted_label': pred,
            'raw_output': raw_output,
            'correct': pred == sample['label']
        })
        
        # Memory cleanup
        if idx % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    results_df = pd.DataFrame(results)
    return results_df


# ═══════════════════════════════════════════════════════════════════════════
# METRICS CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

def calculate_metrics(results_df):
    """
    Calculate evaluation metrics.
    Following the pattern from previous implementations.
    """
    print("\n" + "="*80)
    print("📊 EVALUATION METRICS")
    print("="*80 + "\n")
    
    y_true = results_df['true_label'].values
    y_pred = results_df['predicted_label'].values
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Print results
    print(f"Overall Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision:         {precision*100:.2f}%")
    print(f"Recall:            {recall*100:.2f}%")
    print(f"F1 Score:          {f1*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    # Per-class metrics
    print(f"\n{'─'*80}")
    print(f"Per-Class Performance:")
    print(f"{'─'*80}")
    for label in [0, 1]:
        label_name = "NOT-HATEFUL" if label == 0 else "HATEFUL"
        mask = y_true == label
        if mask.sum() > 0:
            class_acc = (y_true[mask] == y_pred[mask]).mean()
            print(f"  {label_name:15} Accuracy: {class_acc*100:.2f}% (n={mask.sum()})")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("🚀 LLAMA 3.2 11B VISION - FACEBOOK HATEFUL MEMES EVALUATION")
    print("="*80)
    print(f"\n⚙️  Configuration:")
    print(f"   Dataset path: {DATASET_BASE_PATH}")
    print(f"   Model: {MODEL_CONFIG['name']}")
    print(f"   Quantization: {MODEL_CONFIG['quantization']}")
    print(f"   Max tokens: {MODEL_CONFIG['max_new_tokens']}")
    print(f"   Test mode: {TEST_MODE}")
    print(f"   Samples: {TEST_SAMPLES if TEST_MODE else 'ALL'}")
    print(f"   Split: {DATASET_SPLIT}")
    
    # Step 1: Explore dataset
    splits_info = explore_dataset()
    if splits_info is None:
        print("\n❌ Failed to load dataset. Exiting.")
        return
    
    input("\n⏸️  Press Enter to continue to model loading...")
    
    # Step 2: Load model
    model, processor = load_model_from_cache(
        MODEL_CONFIG['name'],
        quantization=MODEL_CONFIG['quantization'],
        cache_dir=CACHE_DIR
    )
    
    input("\n⏸️  Press Enter to start evaluation...")
    
    # Step 3: Load samples
    samples = load_local_dataset(
        DATASET_BASE_PATH,
        split=DATASET_SPLIT,
        test_mode=TEST_MODE,
        n_samples=TEST_SAMPLES
    )
    
    if not samples:
        print("\n❌ Failed to load samples. Exiting.")
        return
    
    # Step 4: Run evaluation
    results_df = evaluate_model(
        model,
        processor,
        samples,
        max_new_tokens=MODEL_CONFIG['max_new_tokens']
    )
    
    # Step 5: Calculate metrics
    metrics = calculate_metrics(results_df)
    
    # Step 6: Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "test" if TEST_MODE else "full"
    
    # Save detailed results
    results_file = f"{OUTPUT_DIR}llama_vision_11b_hateful_memes_{mode}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Save metrics
    metrics_file = f"{OUTPUT_DIR}llama_vision_11b_hateful_memes_{mode}_metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'model': MODEL_CONFIG['name'],
            'quantization': MODEL_CONFIG['quantization'],
            'test_mode': TEST_MODE,
            'n_samples': len(samples),
            'dataset_split': DATASET_SPLIT,
            'metrics': metrics,
            'timestamp': timestamp
        }, f, indent=2)
    print(f"💾 Metrics saved to: {metrics_file}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    
    if TEST_MODE:
        print(f"\n💡 Next steps:")
        print(f"   1. Review the results above")
        print(f"   2. If satisfied, set TEST_MODE = False in the script")
        print(f"   3. Run again for full dataset evaluation")


if __name__ == "__main__":
    # Handle command line arguments
    parser = argparse.ArgumentParser(description='Llama Vision 11B Hateful Memes Evaluation')
    parser.add_argument('--test', action='store_true', help='Run in test mode (15 samples)')
    parser.add_argument('--full', action='store_true', help='Run full evaluation')
    parser.add_argument('--samples', type=int, default=15, help='Number of test samples')
    
    args = parser.parse_args()
    
    # Override TEST_MODE based on arguments
    if args.full:
        TEST_MODE = False
    if args.test:
        TEST_MODE = True
    if args.samples:
        TEST_SAMPLES = args.samples
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
