"""
Analysis: Statistical Tests and Training Convergence Plots
Generates statistical analysis and model comparison visualizations.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys


class Config:
    pass

sys.modules['__main__'].Config = Config

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_results(results_path):
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def load_checkpoint_history(checkpoint_path):
    """Load training history from checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return checkpoint.get('history', None)
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def extract_mse_values(data, model_name):
    """Extract MSE values from individual runs."""
    mse_values = []
    individual_runs = data['individual_runs'][model_name]

    for run_key in sorted(individual_runs.keys()):
        mse = individual_runs[run_key]['mse']
        mse_values.append(mse)

    return np.array(mse_values)


def calculate_statistical_tests(results_path):
    """Calculate p-values comparing different models."""

    data = load_results(results_path)

    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)
    print("Using paired t-tests (same random seeds across models)")
    print("Null hypothesis: No difference in MSE between models")
    print("=" * 80)
    print()

    models = {
        'baseline': 'sequence',
        'patchtst_seq': 'patchtst_sequence',
        'patchtst_patch': 'patchtst_patchwise',
        'gpt2': 'allm4ts_official'
    }

    mse_data = {}
    for short_name, full_name in models.items():
        mse_data[short_name] = extract_mse_values(data, full_name)
        print(f"{full_name}:")
        print(f"  MSE values: {mse_data[short_name]}")
        print(f"  Mean: {np.mean(mse_data[short_name]):.4f}")
        print(f"  Std: {np.std(mse_data[short_name]):.4f}")
        print()

    print("=" * 80)
    print("PAIRWISE COMPARISONS")
    print("=" * 80)
    print()

    comparisons = [
        ('patchtst_patch', 'baseline',
         'PatchTST Patch-wise vs Baseline (No Patches)'),
        ('patchtst_patch', 'patchtst_seq',
         'PatchTST Patch-wise vs Sequence-wise'),
        ('patchtst_patch', 'gpt2',
         'PatchTST Patch-wise vs GPT-2'),
        ('patchtst_seq', 'baseline',
         'PatchTST Sequence-wise vs Baseline'),
    ]

    results_table = []

    for model1, model2, description in comparisons:
        t_stat, p_value = stats.ttest_rel(mse_data[model1], mse_data[model2])

        mean_diff = np.mean(mse_data[model1]) - np.mean(mse_data[model2])
        percent_improvement = -100 * mean_diff / np.mean(mse_data[model2])

        if p_value < 0.001:
            sig = "***"
            sig_text = "Highly significant"
        elif p_value < 0.01:
            sig = "**"
            sig_text = "Very significant"
        elif p_value < 0.05:
            sig = "*"
            sig_text = "Significant"
        else:
            sig = "ns"
            sig_text = "Not significant"

        print(f"{description}")
        print(f"  Model 1 MSE: {np.mean(mse_data[model1]):.4f} +- {np.std(mse_data[model1]):.4f}")
        print(f"  Model 2 MSE: {np.mean(mse_data[model2]):.4f} +- {np.std(mse_data[model2]):.4f}")
        print(f"  Difference: {mean_diff:.4f} ({percent_improvement:+.2f}%)")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f} {sig}")
        print(f"  Interpretation: {sig_text}")
        print()

        results_table.append({
            'comparison': description,
            'model1_mse': np.mean(mse_data[model1]),
            'model2_mse': np.mean(mse_data[model2]),
            'difference': mean_diff,
            'improvement_pct': percent_improvement,
            'p_value': p_value,
            'significance': sig_text
        })

    output_path = Path("results/final_evaluation/statistical_tests.txt")
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL SIGNIFICANCE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Significance levels:\n")
        f.write("  * p < 0.05 (significant)\n")
        f.write("  ** p < 0.01 (very significant)\n")
        f.write("  *** p < 0.001 (highly significant)\n")
        f.write("  ns = not significant (p >= 0.05)\n\n")
        f.write("=" * 80 + "\n\n")

        for result in results_table:
            f.write(f"{result['comparison']}\n")
            f.write(f"  Improvement: {result['improvement_pct']:+.2f}%\n")
            f.write(f"  p-value: {result['p_value']:.6f}\n")
            f.write(f"  Result: {result['significance']}\n\n")

    print("=" * 80)
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    return results_table


def plot_model_comparison(checkpoint_dir, model_names, num_runs=5):
    """Compare validation MSE across different models."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Model Comparison: Validation MSE', fontweight='bold', fontsize=20)

    colors = {
        'sequence': '#e74c3c',
        'patchtst_sequence': '#3498db',
        'patchtst_patchwise': '#2ecc71',
        'allm4ts_official': '#9b59b6'
    }

    labels = {
        'sequence': 'Baseline (No Patches)',
        'patchtst_sequence': 'PatchTST Sequence-wise',
        'patchtst_patchwise': 'PatchTST Patch-wise',
        'allm4ts_official': 'aLLM4TS (GPT-2)'
    }

    models_with_data = 0

    for model_name in model_names:
        all_val_mse = []

        for run_idx in range(1, num_runs + 1):
            checkpoint_path = checkpoint_dir / f"{model_name}_run{run_idx}.pt"

            if not checkpoint_path.exists():
                continue

            history = load_checkpoint_history(checkpoint_path)
            if history is None or 'val_mse' not in history:
                continue

            all_val_mse.append(history['val_mse'])

        if not all_val_mse:
            continue

        models_with_data += 1

        max_epochs = max(len(mse) for mse in all_val_mse)

        padded_mse = []
        for mse in all_val_mse:
            padded = mse + [np.nan] * (max_epochs - len(mse))
            padded_mse.append(padded)

        padded_mse = np.array(padded_mse)
        mean_mse = np.nanmean(padded_mse, axis=0)
        std_mse = np.nanstd(padded_mse, axis=0)

        epochs = range(1, len(mean_mse) + 1)

        color = colors.get(model_name, '#95a5a6')
        label = labels.get(model_name, model_name)

        ax.plot(epochs, mean_mse, label=label, color=color, linewidth=2.5)
        ax.fill_between(epochs,
                        mean_mse - std_mse,
                        mean_mse + std_mse,
                        color=color, alpha=0.2)

    ax.set_xlabel('Epoch', fontweight='bold', fontsize=16)
    ax.set_ylabel('Validation MSE', fontweight='bold', fontsize=16)
    ax.set_title('Validation MSE (Mean +- Std)', fontweight='bold', fontsize=16)

    handles, labels_list = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize=14)

    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig, models_with_data


def generate_training_curves(checkpoint_dir, output_dir):
    """Generate model comparison training curve visualization."""

    model_names = ['sequence', 'patchtst_sequence', 'patchtst_patchwise', 'allm4ts_official']

    print()
    print("=" * 80)
    print("TRAINING CONVERGENCE VISUALIZATION")
    print("=" * 80)
    print()

    fig, models_with_data = plot_model_comparison(checkpoint_dir, model_names, num_runs=5)

    if models_with_data > 0:
        output_path = output_dir / "model_comparison_convergence.pdf"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        print(f"Models plotted: {models_with_data}")
    else:
        print("No data found for comparison")

    plt.close(fig)

    print()
    print("=" * 80)


def main():
    """Run complete analysis."""

    results_path = Path("results/final_evaluation/results.json")
    checkpoint_dir = Path("checkpoints/final_evaluation")
    output_dir = Path("results/final_evaluation/training_curves")
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("ANALYSIS SUITE")
    print("=" * 80)
    print()

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        print("Run final_evaluation.py first")
        return

    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        print("Run final_evaluation.py first")
        return

    calculate_statistical_tests(results_path)
    generate_training_curves(checkpoint_dir, output_dir)

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  results/final_evaluation/statistical_tests.txt")
    print("  results/final_evaluation/training_curves/model_comparison_convergence.pdf")
    print()


if __name__ == "__main__":
    main()