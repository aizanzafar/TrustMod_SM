import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. Configuration and Data Loading
# ==========================================
RESULTS_DIR = "results"

def load_data(folder_path):
    """Loads all JSONL files from the specified folder."""
    all_data = []
    # Find all .json files in the directory
    for file_path in glob.glob(os.path.join(folder_path, "*.json")):
        model_name = os.path.basename(file_path).replace(".json", "")
        try:
            # The snippet indicates JSON Lines format (one object per line)
            df = pd.read_json(file_path, lines=True)
            df['model'] = model_name
            all_data.append(df)
            print(f"Loaded {len(df)} rows for model: {model_name}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    if not all_data:
        raise ValueError(f"No JSON files found in '{folder_path}'. Please check the directory.")
        
    return pd.concat(all_data, ignore_index=True)

# ==========================================
# 2. Metric Calculation
# ==========================================
def calculate_metrics(df):
    """Calculates failure rates and performance metrics per model."""
    metrics_list = []
    
    # We map labels to binary for sklearn metric calculations
    # Assuming 'offensive' is the positive class (1) and 'safe' is negative (0)
    label_map = {'offensive': 1, 'safe': 0}
    
    for model, group in df.groupby('model'):
        total_requests = len(group)
        
        # 1. Calculate Format Failure Rate (where model_verdict is null)
        failed_mask = group['model_verdict'].isnull()
        failed_count = failed_mask.sum()
        failure_rate = (failed_count / total_requests) * 100
        
        # 2. Filter to only valid responses for performance metrics
        valid_group = group[~failed_mask].copy()
        
        # Clean up text just in case (lowercase, strip whitespace)
        valid_group['label'] = valid_group['label'].astype(str).str.lower().str.strip()
        valid_group['model_verdict'] = valid_group['model_verdict'].astype(str).str.lower().str.strip()
        
        # Drop rows where the model hallucinated a verdict outside our binary scope
        valid_group = valid_group[valid_group['model_verdict'].isin(['offensive', 'safe'])]
        
        if len(valid_group) > 0:
            y_true = valid_group['label'].map(label_map)
            y_pred = valid_group['model_verdict'].map(label_map)
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            acc, prec, rec, f1 = 0, 0, 0, 0
            
        metrics_list.append({
            'Model': model,
            'Total Prompts': total_requests,
            'Failed Formats': failed_count,
            'Failure Rate (%)': failure_rate,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
    return pd.DataFrame(metrics_list)

def calculate_metrics_by_type(df):
    """Calculates accuracy broken down by the obfuscation 'type' field."""
    type_metrics = []
    
    # Filter out nulls first
    valid_df = df.dropna(subset=['model_verdict']).copy()
    valid_df['label'] = valid_df['label'].astype(str).str.lower().str.strip()
    valid_df['model_verdict'] = valid_df['model_verdict'].astype(str).str.lower().str.strip()
    valid_df = valid_df[valid_df['model_verdict'].isin(['offensive', 'safe'])]
    
    for (model, obf_type), group in valid_df.groupby(['model', 'type']):
        correct = (group['label'] == group['model_verdict']).sum()
        total = len(group)
        accuracy = correct / total if total > 0 else 0
        
        type_metrics.append({
            'Model': model,
            'Obfuscation Type': obf_type,
            'Accuracy': accuracy
        })
        
    return pd.DataFrame(type_metrics)

# ==========================================
# 3. Plotting
# ==========================================
def plot_metrics(summary_df, type_df):
    """Generates visualizations for the calculated metrics."""
    sns.set_theme(style="whitegrid")
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Format Failure Rate
    sns.barplot(data=summary_df, x='Model', y='Failure Rate (%)', ax=axes[0], palette="Reds_d")
    axes[0].set_title('Format Failure Rate by Model', fontsize=14)
    axes[0].set_ylabel('Failure Rate (%)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Overall F1 Score
    sns.barplot(data=summary_df, x='Model', y='F1 Score', ax=axes[1], palette="Blues_d")
    axes[1].set_title('Overall F1 Score (Valid Responses)', fontsize=14)
    axes[1].set_ylabel('F1 Score')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Accuracy by Obfuscation Type
    sns.barplot(data=type_df, x='Obfuscation Type', y='Accuracy', hue='Model', ax=axes[2], palette="viridis")
    axes[2].set_title('Accuracy by Obfuscation Type', fontsize=14)
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('moderation_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("Loading data...")
    df = load_data(RESULTS_DIR)
    
    print("\nCalculating metrics...")
    summary_metrics = calculate_metrics(df)
    type_metrics = calculate_metrics_by_type(df)
    
    print("\n--- Overall Performance Summary ---")
    print(summary_metrics.to_string(index=False))
    
    print("\nGenerating plots...")
    plot_metrics(summary_metrics, type_metrics)
    print("Done! Plots saved as 'moderation_evaluation_results.png'.")