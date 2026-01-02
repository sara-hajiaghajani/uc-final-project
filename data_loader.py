"""
Data Loading and Preprocessing for Weather Forecasting

Handles KNMI hourly weather data with normalization and sliding window creation.
Supports multi-station datasets for generalization analysis.
"""

import pandas as pd
import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch


class WeatherDataset(Dataset):
    """
    PyTorch Dataset for multivariate time series forecasting with station tracking.

    Creates sliding windows from weather data:
    - Input: seq_len consecutive timesteps
    - Target: pred_len timesteps following the input
    - Station: identifier for per-station evaluation
    """

    def __init__(self, data, station_ids, seq_len, pred_len, features):
        self.data = data
        self.station_ids = station_ids
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        station = self.station_ids[idx + self.seq_len - 1]
        return torch.FloatTensor(x), torch.FloatTensor(y), station


def seed_worker(worker_id):
    """Set random seed for each worker to ensure consistent results."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_knmi_data(data_path, features):
    """
    Load KNMI hourly weather data with station information.

    Handles missing values using time-based interpolation per station.

    Args:
        data_path: path to CSV file
        features: list of feature column names

    Returns:
        DataFrame with datetime index, station column, and features
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data not found at {data_path}\n"
            "Run data_preparation.py first to process raw KNMI data"
        )

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    df = df[['station'] + features]

    # Fill missing values per station
    stations = df['station'].unique()
    dfs = []
    for station in stations:
        station_df = df[df['station'] == station].copy()
        # Drop station column before interpolation (object type causes warning)
        station_name = station
        station_df = station_df.drop(columns=['station'])
        station_df = station_df.interpolate(method='time').bfill().ffill()
        # Add station back
        station_df['station'] = station_name
        dfs.append(station_df)

    df = pd.concat(dfs).sort_index()

    return df


def prepare_data(config):
    """
    Complete data preparation pipeline for time series forecasting.

    Process:
    1. Load raw data with station identifiers
    2. Normalize features using StandardScaler
    3. Split data by time (train/val/test)
    4. Create PyTorch datasets with sliding windows
    5. Create DataLoaders with consistent random seeds

    Args:
        config: configuration object with hyperparameters

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    data_path = os.path.join(config.DATA_DIR, f"{config.DATA_SOURCE}_hourly.csv")
    df = load_knmi_data(data_path, config.FEATURES)

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Stations: {', '.join(df['station'].unique())}")

    # Separate station IDs and features
    station_ids = df['station'].values
    data = df[config.FEATURES].values

    # Normalize features (zero mean, unit variance)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Split data by time
    n = len(data_scaled)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = int(n * (config.TRAIN_SPLIT + config.VAL_SPLIT))

    train_data = data_scaled[:train_end]
    train_stations = station_ids[:train_end]
    val_data = data_scaled[train_end:val_end]
    val_stations = station_ids[train_end:val_end]
    test_data = data_scaled[val_end:]
    test_stations = station_ids[val_end:]

    print(f"Split: Train={len(train_data):,} | Val={len(val_data):,} | Test={len(test_data):,}")

    # Create datasets
    train_dataset = WeatherDataset(train_data, train_stations, config.SEQ_LEN, config.PRED_LEN, config.FEATURES)
    val_dataset = WeatherDataset(val_data, val_stations, config.SEQ_LEN, config.PRED_LEN, config.FEATURES)
    test_dataset = WeatherDataset(test_data, test_stations, config.SEQ_LEN, config.PRED_LEN, config.FEATURES)

    # Random seed for shuffling
    g = torch.Generator()
    g.manual_seed(config.RANDOM_SEED)

    def collate_fn(batch):
        """Combine batch items including station information."""
        x_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.stack([item[1] for item in batch])
        stations = [item[2] for item in batch]
        return x_batch, y_batch, stations

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, scaler