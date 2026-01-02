"""
Final Evaluation with Optimal Configuration

Uses best settings from experiments 1-3:
- PATCH_LEN from Exp 1
- SEQ_LEN from Exp 2
- PRED_LEN from Exp 3
Runs: 4 models x 5 seeds = 20 total
"""

import torch
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class Config:
    """Final evaluation configuration with optimal settings."""
    FEATURES = ["temperature", "wind_speed", "humidity", "pressure"]

    # UPDATE THESE after running experiments 1-3
    SEQ_LEN = 672     # Best from Exp 2
    PRED_LEN = 24      # Best from Exp 3
    PATCH_LEN = 32     # Best from Exp 1
    STRIDE = 16        # PATCH_LEN // 2

    DROPOUT = 0.1
    USE_REVIN = True
    RANDOM_SEED = 42

    D_MODEL_BASE = 128
    N_HEADS_BASE = 16
    E_LAYERS_BASE = 3
    D_MODEL_LLM = 768
    N_HEADS_LLM = 12
    E_LAYERS_LLM = 12

    DATA_SOURCE = "knmi"
    DATA_DIR = "./data"
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 50
    PATIENCE = 10

    DEVICE = "mps"
    CHECKPOINT_DIR = "./checkpoints/final_evaluation"
    RESULTS_DIR = "./results/final_evaluation"


from data_loader import prepare_data
from model_baseline_sequential import SequenceWiseModelNoPatch
from model_patchtst_sequential import OfficialPatchTSTModel
from model_patchtst_patchwise import PatchTSTPatchWise
from model_allm4ts_gpt2 import OfficialALLM4TSModel
from train_and_evaluate import train_model, test_model_denorm


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_baseline_comparison(averaged_results):
    """Generate comparison plots for final evaluation."""
    results_dir = Path(Config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_order = ["sequence", "patchtst_sequence", "patchtst_patchwise", "allm4ts_official"]
    model_labels = {
        "sequence": "Baseline\n(No Patches)",
        "patchtst_patchwise": "PatchTST\n(Patch-wise)",
        "patchtst_sequence": "PatchTST\n(Sequence-wise)",
        "allm4ts_official": "aLLM4TS\n(GPT-2)"
    }
    model_colors = {
        "sequence": "#e74c3c",
        "patchtst_patchwise": "#2ecc71",
        "patchtst_sequence": "#3498db",
        "allm4ts_official": "#9b59b6"
    }

    # Per-station plots
    stations = set()
    for model_results in averaged_results.values():
        if "per_station" in model_results:
            stations.update(model_results["per_station"].keys())

    for station in sorted(stations):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x = np.arange(len(model_order))
        mse_means, mse_stds, mae_means, mae_stds = [], [], [], []

        for m in model_order:
            if station in averaged_results[m].get("per_station", {}):
                mse_means.append(averaged_results[m]["per_station"][station]["mse_mean"])
                mse_stds.append(averaged_results[m]["per_station"][station]["mse_std"])
                mae_means.append(averaged_results[m]["per_station"][station]["mae_mean"])
                mae_stds.append(averaged_results[m]["per_station"][station]["mae_std"])
            else:
                mse_means.append(0)
                mse_stds.append(0)
                mae_means.append(0)
                mae_stds.append(0)

        colors = [model_colors[m] for m in model_order]

        bars1 = ax1.bar(x, mse_means, 0.6, yerr=mse_stds, capsize=5,
                        color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)
        ax1.set_xlabel("Model", fontweight="bold", fontsize=12)
        ax1.set_ylabel("MSE (Mean ± Std)", fontweight="bold", fontsize=12)
        ax1.set_title(f"MSE - {station.upper()}", fontweight="bold", fontsize=13, pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels([model_labels[m] for m in model_order])
        ax1.grid(axis="y", alpha=0.3, linestyle="--")
        ax1.set_ylim(0, max(mse_means) * 1.25)

        for bar, mean, std in zip(bars1, mse_means, mse_stds):
            if mean > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.5,
                         f"{mean:.2f}\n±{std:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        bars2 = ax2.bar(x, mae_means, 0.6, yerr=mae_stds, capsize=5,
                        color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)
        ax2.set_xlabel("Model", fontweight="bold", fontsize=12)
        ax2.set_ylabel("MAE (Mean ± Std)", fontweight="bold", fontsize=12)
        ax2.set_title(f"MAE - {station.upper()}", fontweight="bold", fontsize=13, pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels([model_labels[m] for m in model_order])
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        ax2.set_ylim(0, max(mae_means) * 1.25)

        for bar, mean, std in zip(bars2, mae_means, mae_stds):
            if mean > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.05,
                         f"{mean:.2f}\n±{std:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(results_dir / f"{station}_baseline_comparison.png", bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Saved {station} plot")

    # Overall plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(model_order))
    mse_means = [averaged_results[m]['mse_mean'] for m in model_order]
    mse_stds = [averaged_results[m]['mse_std'] for m in model_order]
    mae_means = [averaged_results[m]['mae_mean'] for m in model_order]
    mae_stds = [averaged_results[m]['mae_std'] for m in model_order]

    colors = [model_colors[m] for m in model_order]

    bars1 = ax1.bar(x, mse_means, 0.6, yerr=mse_stds, capsize=5,
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax1.set_ylabel('MSE (Mean ± Std)', fontweight='bold', fontsize=12)
    ax1.set_title('MSE Comparison (Overall)', fontweight='bold', fontsize=13, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels([model_labels[m] for m in model_order])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(mse_means) * 1.25)

    for bar, mean, std in zip(bars1, mse_means, mse_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + std + 0.5,
                 f'{mean:.2f}\n±{std:.2f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    bars2 = ax2.bar(x, mae_means, 0.6, yerr=mae_stds, capsize=5,
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax2.set_ylabel('MAE (Mean ± Std)', fontweight='bold', fontsize=12)
    ax2.set_title('MAE Comparison (Overall)', fontweight='bold', fontsize=13, pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels([model_labels[m] for m in model_order])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(mae_means) * 1.25)

    for bar, mean, std in zip(bars2, mae_means, mae_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + std + 0.05,
                 f'{mean:.2f}\n±{std:.2f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / 'baseline_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("Saved overall plot")


def run_final_evaluation():
    """Main function for final evaluation with optimal settings."""
    config = Config()

    print("=" * 70)
    print("FINAL EVALUATION WITH OPTIMAL CONFIGURATION")
    print("=" * 70)
    print(f"SEQ_LEN={config.SEQ_LEN}, PRED_LEN={config.PRED_LEN}, PATCH_LEN={config.PATCH_LEN}")
    print("Total: 4 models x 5 runs = 20 training runs")
    print("=" * 70)


    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    NUM_RUNS = 5
    SEEDS = [42, 123, 456, 789, 101112]

    model_list = [
        ('sequence', SequenceWiseModelNoPatch),
        ('patchtst_patchwise', PatchTSTPatchWise),
        ('patchtst_sequence', OfficialPatchTSTModel),
        ('allm4ts_official', OfficialALLM4TSModel),
    ]

    all_runs_results = {model_name: [] for model_name, _ in model_list}
    all_runs_logs = {model_name: [] for model_name, _ in model_list}

    for run_idx, seed in enumerate(SEEDS):
        print(f"\nRun {run_idx + 1}/{NUM_RUNS} (Seed: {seed})")

        set_seed(seed)
        train_loader, val_loader, test_loader, scaler = prepare_data(config)

        run_results = {}
        run_logs = {}

        for model_name, model_class in model_list:
            checkpoint = os.path.join(config.CHECKPOINT_DIR, f'{model_name}_run{run_idx + 1}.pt')
            model = model_class(config)
            params = model.count_parameters()

            if os.path.exists(checkpoint):
                if config.DEVICE == 'mps':
                    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                else:
                    state = torch.load(checkpoint, weights_only=False)
                model.load_state_dict(state['model_state_dict'])
                history = state.get('history', {})
            else:
                history = train_model(model, train_loader, val_loader, config,
                                      f'{model_name}_run{run_idx + 1}')

            model.to(config.DEVICE)
            metrics, _, _ = test_model_denorm(model, test_loader, config, scaler)

            run_results[model_name] = {
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'per_station': metrics.get('per_station', {}),
                'params': params
            }
            run_logs[model_name] = history

        for model_name in run_results:
            all_runs_results[model_name].append(run_results[model_name])
            all_runs_logs[model_name].append(run_logs[model_name])

        # Save results after each run
        results_data = {
            'config': {
                'seq_len': config.SEQ_LEN,
                'pred_len': config.PRED_LEN,
                'patch_len': config.PATCH_LEN,
                'stride': config.STRIDE,
                'num_runs': NUM_RUNS,
                'seeds': SEEDS,
                'learning_rate': config.LEARNING_RATE
            },
            'individual_runs': {},
            'averaged_results': {}
        }

        for model_name in all_runs_results:
            results_data['individual_runs'][model_name] = {}
            for i, run in enumerate(all_runs_results[model_name]):
                results_data['individual_runs'][model_name][f'run_{i + 1}'] = {
                    'seed': SEEDS[i],
                    'mse': float(run['mse']),
                    'mae': float(run['mae']),
                    'per_station': run.get('per_station', {}),
                    'total_params': int(run['params']['total']),
                    'head_params': int(run['params'].get('patch_wise_head', 0)),
                    'epochs_trained': all_runs_logs[model_name][i].get('epochs_trained', 0),
                    'early_stopped': all_runs_logs[model_name][i].get('early_stopped', False)
                }

        # Compute averaged results
        for model_name in all_runs_results:
            if len(all_runs_results[model_name]) > 0:
                mse_values = [run['mse'] for run in all_runs_results[model_name]]
                mae_values = [run['mae'] for run in all_runs_results[model_name]]

                results_data['averaged_results'][model_name] = {
                    'mse_mean': float(np.mean(mse_values)),
                    'mse_std': float(np.std(mse_values)),
                    'mae_mean': float(np.mean(mae_values)),
                    'mae_std': float(np.std(mae_values))
                }

                # Per-station averages
                per_station_avg = {}
                stations = set()
                for run in all_runs_results[model_name]:
                    if 'per_station' in run:
                        stations.update(run['per_station'].keys())

                for station in stations:
                    station_mses = [run['per_station'][station]['mse']
                                    for run in all_runs_results[model_name]
                                    if station in run.get('per_station', {})]
                    station_maes = [run['per_station'][station]['mae']
                                    for run in all_runs_results[model_name]
                                    if station in run.get('per_station', {})]

                    if station_mses:
                        per_station_avg[station] = {
                            'mse_mean': float(np.mean(station_mses)),
                            'mse_std': float(np.std(station_mses)),
                            'mae_mean': float(np.mean(station_maes)),
                            'mae_std': float(np.std(station_maes))
                        }

                results_data['averaged_results'][model_name]['per_station'] = per_station_avg

        with open(os.path.join(config.RESULTS_DIR, 'results.json'), 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"Run {run_idx + 1} saved")

    averaged_results = results_data['averaged_results']

    # Save summary
    with open(os.path.join(config.RESULTS_DIR, 'summary.txt'), 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("FINAL EVALUATION - SUMMARY\n")
        f.write("=" * 90 + "\n")
        f.write(f"Configuration: SEQ_LEN={config.SEQ_LEN}, PRED_LEN={config.PRED_LEN}, PATCH_LEN={config.PATCH_LEN}\n")
        f.write(f"Runs: {NUM_RUNS} (seeds: {SEEDS})\n")
        f.write("=" * 90 + "\n\n")

        f.write("OVERALL RESULTS\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Model':<30} {'MSE (mean±std)':<25} {'MAE (mean±std)':<25}\n")
        f.write("-" * 90 + "\n")

        for model in ['sequence', 'patchtst_patchwise', 'patchtst_sequence', 'allm4ts_official']:
            res = averaged_results[model]
            f.write(f"{model:<30} {res['mse_mean']:.4f}±{res['mse_std']:.4f}           ")
            f.write(f"{res['mae_mean']:.4f}±{res['mae_std']:.4f}\n")

        # Per-station breakdown
        all_stations = set()
        for model_results in averaged_results.values():
            if "per_station" in model_results:
                all_stations.update(model_results["per_station"].keys())

        if all_stations:
            f.write("\n" + "=" * 90 + "\n")
            f.write("PER-STATION RESULTS\n")
            f.write("=" * 90 + "\n\n")

            for station in sorted(all_stations):
                f.write(f"\n{station.upper()}:\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Model':<30} {'MSE (mean±std)':<25} {'MAE (mean±std)':<25}\n")
                f.write("-" * 90 + "\n")

                for model in ["sequence", "patchtst_patchwise", "patchtst_sequence", "allm4ts_official"]:
                    if station in averaged_results[model].get("per_station", {}):
                        res = averaged_results[model]["per_station"][station]
                        f.write(f"{model:<30} {res['mse_mean']:.4f}±{res['mse_std']:.4f}           ")
                        f.write(f"{res['mae_mean']:.4f}±{res['mae_std']:.4f}\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 90 + "\n")

        best_model = min(['sequence', 'patchtst_patchwise', 'patchtst_sequence', 'allm4ts_official'],
                         key=lambda m: averaged_results[m]['mse_mean'])
        best_mse = averaged_results[best_model]['mse_mean']
        best_mae = averaged_results[best_model]['mae_mean']

        f.write(f"Best model: {best_model}\n")
        f.write(f"MSE: {best_mse:.4f}, MAE: {best_mae:.4f}\n")
        f.write("=" * 90 + "\n")

    # Print summary
    print("\n" + "=" * 90)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Model':<30} {'MSE (mean±std)':<25} {'MAE (mean±std)':<25}")
    print("-" * 90)

    for model in ['sequence', 'patchtst_patchwise', 'patchtst_sequence', 'allm4ts_official']:
        res = averaged_results[model]
        print(f"{model:<30} {res['mse_mean']:.4f}±{res['mse_std']:.4f}           "
              f"{res['mae_mean']:.4f}±{res['mae_std']:.4f}")

    print("=" * 90)

    # Per-station console output
    all_stations = set()
    for model_results in averaged_results.values():
        if "per_station" in model_results:
            all_stations.update(model_results["per_station"].keys())

    if all_stations:
        print("\nPER-STATION BREAKDOWN")
        print("=" * 90)

        for station in sorted(all_stations):
            print(f"\n{station.upper()}:")
            print("-" * 90)
            for model in ["sequence", "patchtst_sequence", "patchtst_patchwise", "allm4ts_official"]:
                if station in averaged_results[model].get("per_station", {}):
                    res = averaged_results[model]["per_station"][station]
                    print(f"  {model:<25} MSE: {res['mse_mean']:.4f}±{res['mse_std']:.4f}  "
                          f"MAE: {res['mae_mean']:.4f}±{res['mae_std']:.4f}")

    best_model = min(['sequence', 'patchtst_patchwise', 'patchtst_sequence', 'allm4ts_official'],
                     key=lambda m: averaged_results[m]['mse_mean'])
    best_mse = averaged_results[best_model]['mse_mean']

    print(f"\nBest model: {best_model} (MSE: {best_mse:.4f})")

    plot_baseline_comparison(averaged_results)

    print(f"\nResults saved to {config.RESULTS_DIR}/")


if __name__ == "__main__":
    run_final_evaluation()