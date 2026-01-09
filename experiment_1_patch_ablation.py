"""
Experiment 1: Patch Length Ablation (PatchTST Patchwise Only)

Tests patch lengths [8, 16, 32] on PatchTST Patchwise model only.
Fixed: SEQ_LEN=336, PRED_LEN=96
Runs: 3 patch configs x 1 model x 3 seeds = 9 total
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
    """Configuration for patch length ablation."""
    FEATURES = ["temperature", "wind_speed", "humidity", "pressure"]
    SEQ_LEN = 336
    PRED_LEN = 96
    PATCH_LEN = 16  # Will be varied in this experiment
    STRIDE = 8
    DROPOUT = 0.1
    USE_REVIN = True
    RANDOM_SEED = 42

    D_MODEL_BASE = 128
    N_HEADS_BASE = 16
    E_LAYERS_BASE = 3

    DATA_SOURCE = "knmi"
    DATA_DIR = "./data"
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 20
    PATIENCE = 5

    DEVICE = "mps"  # Options: "cuda", "mps", or "cpu"
    CHECKPOINT_DIR = "./checkpoints"
    RESULTS_DIR = "./results"


from data_loader import prepare_data
from model_patchtst_patchwise import PatchTSTPatchWise
from train_and_evaluate import train_model, test_model_denorm


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_patch_config(config, patch_len):
    """Run PatchTST Patchwise for one patch length configuration."""
    config.PATCH_LEN = patch_len
    config.STRIDE = patch_len // 2
    config.CHECKPOINT_DIR = f"./checkpoints/experiment1_patch/patch_{patch_len}"
    config.RESULTS_DIR = f"./results/experiment1_patch/patch_{patch_len}"

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    NUM_RUNS = 3
    SEEDS = [42, 123, 456]

    model_list = [('patchtst_patchwise', PatchTSTPatchWise)]
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

        # Save results
        results_data = {
            'config': {
                'patch_length': patch_len,
                'stride': patch_len // 2,
                'seq_len': 336,
                'pred_len': 96,
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

        # Compute averages
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

    return results_data['averaged_results']


def plot_patch_ablation(all_patch_results, patch_lengths):
    """Generate combined plot with overall and per-station results."""
    summary_dir = Path("./results/experiment1_patch/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    model = 'patchtst_patchwise'

    # Get station names
    stations = set()
    for patch_results in all_patch_results.values():
        if 'per_station' in patch_results[model]:
            stations.update(patch_results[model]['per_station'].keys())
    stations = sorted(stations)

    # Define colors
    colors = {
        'overall': '#2ecc71',
        'schiphol': '#3498db',
        'maastricht': '#e74c3c'
    }

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    # Plot overall MSE (solid line)
    overall_mse_means = [all_patch_results[p][model]['mse_mean'] for p in patch_lengths]
    overall_mse_stds = [all_patch_results[p][model]['mse_std'] for p in patch_lengths]

    ax1.plot(patch_lengths, overall_mse_means, marker='o', linewidth=2.5, linestyle='-',
             label='Overall MSE (solid)', color=colors['overall'])
    ax1.fill_between(patch_lengths,
                     np.array(overall_mse_means) - np.array(overall_mse_stds),
                     np.array(overall_mse_means) + np.array(overall_mse_stds),
                     alpha=0.15, color=colors['overall'])

    # Plot per-station MSE (solid lines)
    for station in stations:
        means, stds = [], []
        for p in patch_lengths:
            if station in all_patch_results[p][model].get('per_station', {}):
                means.append(all_patch_results[p][model]['per_station'][station]['mse_mean'])
                stds.append(all_patch_results[p][model]['per_station'][station]['mse_std'])

        if means:
            ax1.plot(patch_lengths, means, marker='o', linewidth=2, linestyle='-',
                     label=f'{station.capitalize()} MSE (solid)', color=colors.get(station, '#95a5a6'), alpha=0.7)

    # Plot overall MAE (dashed line)
    overall_mae_means = [all_patch_results[p][model]['mae_mean'] for p in patch_lengths]

    ax2.plot(patch_lengths, overall_mae_means, marker='s', linewidth=2.5, linestyle='--',
             label='Overall MAE (dashed)', color=colors['overall'], alpha=0.7)

    # Plot per-station MAE (dashed lines)
    for station in stations:
        mae_means = []
        for p in patch_lengths:
            if station in all_patch_results[p][model].get('per_station', {}):
                mae_means.append(all_patch_results[p][model]['per_station'][station]['mae_mean'])

        if mae_means:
            ax2.plot(patch_lengths, mae_means, marker='s', linewidth=2, linestyle='--',
                     label=f'{station.capitalize()} MAE (dashed)', color=colors.get(station, '#95a5a6'), alpha=0.5)

    ax1.set_xlabel('Patch Length', fontweight='bold', fontsize=12)
    ax1.set_ylabel('MSE (Mean ± Std)', fontweight='bold', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(patch_lengths)

    ax2.set_ylabel('MAE (Mean)', fontweight='bold', fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    ax1.set_title('Patch Length Ablation: PatchTST Patchwise',
                  fontweight='bold', fontsize=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(summary_dir / 'patch_ablation_combined.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print("Saved combined plot")


def run_patch_ablation():
    """Main function for patch length ablation study."""
    config = Config()

    print("=" * 70)
    print("EXPERIMENT 1: PATCH LENGTH ABLATION (PATCHTST PATCHWISE)")
    print("=" * 70)
    print("Testing patch lengths: [8, 16, 32]")
    print("Fixed: SEQ_LEN=336, PRED_LEN=96")
    print("Total: 3 configs x 1 model x 3 runs = 9 training runs")
    print("=" * 70)

    PATCH_LENGTHS = [8, 16, 32]
    all_patch_results = {}

    for patch_len in PATCH_LENGTHS:
        print(f"\n{'=' * 70}")
        print(f"PATCH LENGTH: {patch_len}")
        print(f"{'=' * 70}")

        results = run_single_patch_config(config, patch_len)
        all_patch_results[patch_len] = results

        print(f"\nPatch {patch_len} complete")
        model = 'patchtst_patchwise'
        print(f"  MSE: {results[model]['mse_mean']:.4f}±{results[model]['mse_std']:.4f}")

    # Save summary
    summary_dir = Path("./results/experiment1_patch/summary")
    summary_dir.mkdir(exist_ok=True)

    with open(summary_dir / 'summary.txt', 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("PATCH LENGTH ABLATION - PATCHTST PATCHWISE ONLY\n")
        f.write("=" * 90 + "\n")
        f.write(f"Tested: PATCH_LEN = {PATCH_LENGTHS}\n")
        f.write(f"Fixed: SEQ_LEN = 336, PRED_LEN = 96\n")
        f.write(f"Runs per config: 3 (seeds: [42, 123, 456])\n")
        f.write("=" * 90 + "\n\n")

        f.write("OVERALL RESULTS (Combined performance across both stations)\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Patch':<10} {'MSE (mean±std)':<25} {'MAE (mean±std)':<25}\n")
        f.write("-" * 90 + "\n")

        model = 'patchtst_patchwise'
        for p in PATCH_LENGTHS:
            res = all_patch_results[p][model]
            f.write(f"{p:<10} {res['mse_mean']:.4f}±{res['mse_std']:.4f}           ")
            f.write(f"{res['mae_mean']:.4f}±{res['mae_std']:.4f}\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("PER-STATION RESULTS\n")
        f.write("=" * 90 + "\n\n")

        for p in PATCH_LENGTHS:
            all_stations = set()
            if 'per_station' in all_patch_results[p][model]:
                all_stations.update(all_patch_results[p][model]['per_station'].keys())

            if all_stations:
                f.write(f"\nPatch {p}:\n")
                for station in sorted(all_stations):
                    f.write(f"  {station.upper()}:\n")
                    f.write("-" * 90 + "\n")
                    f.write(f"{'MSE (mean±std)':<25} {'MAE (mean±std)':<25}\n")
                    f.write("-" * 90 + "\n")

                    if station in all_patch_results[p][model].get('per_station', {}):
                        res = all_patch_results[p][model]['per_station'][station]
                        f.write(f"{res['mse_mean']:.4f}±{res['mse_std']:.4f}           ")
                        f.write(f"{res['mae_mean']:.4f}±{res['mae_std']:.4f}\n")
                    f.write("\n")

        best_patch = min(PATCH_LENGTHS, key=lambda p: all_patch_results[p][model]['mse_mean'])
        best_mse = all_patch_results[best_patch][model]['mse_mean']
        f.write("=" * 90 + "\n")
        f.write(f"BEST PATCH LENGTH: {best_patch} (Based on lowest overall MSE: {best_mse:.4f})\n")
        f.write("=" * 90 + "\n")

    # Print summary
    print("\n" + "=" * 90)
    print("EXPERIMENT 1 SUMMARY")
    print("=" * 90)
    print(f"{'Patch':<10} {'MSE (mean±std)':<25}")
    print("-" * 90)

    model = 'patchtst_patchwise'
    for p in PATCH_LENGTHS:
        res = all_patch_results[p][model]
        print(f"{p:<10} {res['mse_mean']:.4f}±{res['mse_std']:.4f}")

    print("=" * 90)

    # Per-station console output
    for p in PATCH_LENGTHS:
        all_stations = set()
        if 'per_station' in all_patch_results[p][model]:
            all_stations.update(all_patch_results[p][model]['per_station'].keys())

        if all_stations:
            print(f"\nPatch {p}:")
            for station in sorted(all_stations):
                if station in all_patch_results[p][model].get('per_station', {}):
                    res = all_patch_results[p][model]['per_station'][station]
                    print(f"  {station.upper()}: MSE: {res['mse_mean']:.4f}±{res['mse_std']:.4f}")

    best_patch = min(PATCH_LENGTHS, key=lambda p: all_patch_results[p][model]['mse_mean'])
    best_mse = all_patch_results[best_patch][model]['mse_mean']

    print(f"\nBest patch length: {best_patch} (MSE: {best_mse:.4f})")

    plot_patch_ablation(all_patch_results, PATCH_LENGTHS)

    print(f"\nResults saved to {summary_dir}/")
    print(f"Recommendation: Use PATCH_LEN={best_patch}")


if __name__ == "__main__":
    run_patch_ablation()