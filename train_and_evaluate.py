"""
Training and Evaluation for Time Series Forecasting

Provides training pipeline with early stopping, metric tracking, and per-station evaluation.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from tqdm import tqdm


class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    Saves the best model automatically.
    """

    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Val loss improved: {self.best_loss:.6f} -> {val_loss:.6f}')
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0

        return self.early_stop


def compute_metrics(predictions, targets):
    """Calculate MSE and MAE."""
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    return {'mse': mse, 'mae': mae}


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_x, batch_y, stations in pbar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': total_loss / n_batches})

    return total_loss / n_batches


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on validation or test set.

    Returns:
        metrics, predictions, targets, stations
    """
    model.eval()
    total_loss = 0
    n_batches = 0
    all_predictions = []
    all_targets = []
    all_stations = []

    with torch.no_grad():
        for batch_x, batch_y, stations in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            output = model(batch_x)
            loss = criterion(output, batch_y)

            total_loss += loss.item()
            n_batches += 1
            all_predictions.append(output.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            all_stations.extend(stations)

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    metrics = compute_metrics(predictions, targets)
    metrics['loss'] = total_loss / n_batches

    return metrics, predictions, targets, all_stations


def train_model(model, train_loader, val_loader, config, save_name):
    """
    Train model with early stopping and save best checkpoint.

    Returns:
        history: Dictionary with training metrics
    """
    device = torch.device(config.DEVICE)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=config.PATIENCE, verbose=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': [],
        'learning_rate': [],
        'epochs_trained': 0,
        'early_stopped': False,
        'epoch_details': []
    }

    print(f"\nTraining {save_name}")
    print(f"Parameters: {model.count_parameters()}")

    for epoch in range(config.NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{config.NUM_EPOCHS}')

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mse'].append(val_metrics['mse'])
        history['val_mae'].append(val_metrics['mae'])
        history['learning_rate'].append(current_lr)

        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_metrics['loss']),
            'val_mse': float(val_metrics['mse']),
            'val_mae': float(val_metrics['mae']),
            'learning_rate': float(current_lr)
        }
        history['epoch_details'].append(epoch_info)

        scheduler.step(val_metrics['loss'])

        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_metrics["loss"]:.6f} | MSE: {val_metrics["mse"]:.6f} | MAE: {val_metrics["mae"]:.6f}')
        print(f'LR: {current_lr:.6f}')

        # Early stopping check
        if early_stopping(val_metrics['loss'], model):
            print('Early stopping triggered')
            history['early_stopped'] = True
            history['epochs_trained'] = epoch + 1
            break

    if not history['early_stopped']:
        history['epochs_trained'] = config.NUM_EPOCHS

    # Restore best model
    model.load_state_dict(early_stopping.best_model)

    # Save checkpoint
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{save_name}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, checkpoint_path)

    print(f'\nSaved to {checkpoint_path}')
    print(f'Best val loss: {early_stopping.best_loss:.6f}')
    print(f'Epochs trained: {history["epochs_trained"]}')

    return history


def test_model_denorm(model, test_loader, config, scaler):
    """
    Evaluate on test set with metrics in original scale (denormalized).
    Calculates overall and per-station performance.

    Returns:
        metrics: Dict with overall MSE/MAE and per-station breakdown
        predictions: Normalized predictions
        targets: Normalized targets
    """
    device = torch.device(config.DEVICE)
    model = model.to(device)
    model.eval()

    criterion = nn.MSELoss()

    print("\nEvaluating on test set...")
    test_metrics_norm, predictions, targets, stations = evaluate(model, test_loader, criterion, device)

    # Denormalize to original scale
    n_samples, pred_len, n_features = predictions.shape
    predictions_flat = predictions.reshape(-1, n_features)
    targets_flat = targets.reshape(-1, n_features)

    predictions_denorm = scaler.inverse_transform(predictions_flat)
    targets_denorm = scaler.inverse_transform(targets_flat)

    predictions_denorm = predictions_denorm.reshape(n_samples, pred_len, n_features)
    targets_denorm = targets_denorm.reshape(n_samples, pred_len, n_features)

    # Overall metrics
    mse_overall = np.mean((predictions_denorm - targets_denorm) ** 2)
    mae_overall = np.mean(np.abs(predictions_denorm - targets_denorm))

    print(f'Overall - MSE: {mse_overall:.6f} | MAE: {mae_overall:.6f}')

    # Per-station metrics
    unique_stations = sorted(set(stations))
    per_station_metrics = {}

    for station in unique_stations:
        station_indices = [i for i, s in enumerate(stations) if s == station]

        if len(station_indices) > 0:
            station_preds = predictions_denorm[station_indices]
            station_targets = targets_denorm[station_indices]

            mse_station = np.mean((station_preds - station_targets) ** 2)
            mae_station = np.mean(np.abs(station_preds - station_targets))

            per_station_metrics[station] = {
                'mse': float(mse_station),
                'mae': float(mae_station),
                'n_samples': len(station_indices)
            }

            print(
                f'{station.capitalize()} - MSE: {mse_station:.6f} | MAE: {mae_station:.6f} ({len(station_indices)} samples)')

    return {
        'mse': mse_overall,
        'mae': mae_overall,
        'per_station': per_station_metrics,
        'mse_normalized': test_metrics_norm['mse'],
        'mae_normalized': test_metrics_norm['mae']
    }, predictions, targets