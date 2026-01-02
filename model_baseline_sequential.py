"""
Baseline Sequential Model (No Patching)

Standard transformer with timestep-based tokenization.
Processes one timestep at a time without overlapping patches.

Architecture: Timestep Projection -> Positional Encoding -> Transformer -> Flatten -> Project
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SequenceWiseModelNoPatch(nn.Module):
    """
    Baseline transformer with timestep projection (no patching).
    Uses sequence-wise decoding (flatten all tokens and project).
    """

    def __init__(self, config):
        super().__init__()

        D_MODEL = config.D_MODEL_BASE
        N_HEADS = config.N_HEADS_BASE
        E_LAYERS = config.E_LAYERS_BASE
        N_FEATURES = len(config.FEATURES)
        SEQ_LEN = config.SEQ_LEN

        # Project each timestep to D_MODEL
        self.feature_projection = nn.Linear(N_FEATURES, D_MODEL)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(D_MODEL)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=E_LAYERS)

        # Sequence-wise decoder
        self.flatten_proj = nn.Linear(SEQ_LEN * D_MODEL, config.PRED_LEN * N_FEATURES)

        self.dropout = nn.Dropout(config.DROPOUT)
        self.config = config
        self.n_features = N_FEATURES

    def forward(self, x):
        # Project timesteps
        x = self.feature_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Transformer processing
        x = self.transformer_encoder(x)

        # Flatten and project
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        x = self.flatten_proj(x)

        # Reshape output
        x = x.reshape(batch_size, self.config.PRED_LEN, self.n_features)

        return x

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        head_params = sum(p.numel() for p in self.flatten_proj.parameters())

        return {
            'total': total_params,
            'encoder': total_params - head_params,
            'head': head_params,
            'patch_wise_head': 0,
            'head_percentage': 100 * head_params / total_params
        }