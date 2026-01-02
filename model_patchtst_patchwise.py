"""
PatchTST with Patch-wise Decoder

Patch-wise decoding: each patch is independently decoded to time series values,
preserving patch structure throughout the forecasting pipeline.

Architecture: Patching -> Transformer Encoder -> Patch Selection -> Patch-wise Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RevIN(nn.Module):
    """Reversible Instance Normalization for time series."""

    def __init__(self, num_features, affine=True, eps=1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def _get_statistics(self, x):
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._get_statistics(x)
            x = (x - self.mean) / self.stdev
            if self.affine: x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine: x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            x = x * self.stdev + self.mean
        return x


class SelfAttention(nn.Module):
    """Multi-head self-attention with residual attention."""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=True):
        super().__init__()
        d_k = d_v = d_model // n_heads
        self.n_heads = n_heads
        self.scale = d_k ** -0.5
        self.res_attention = res_attention

        self.W_QKV = nn.Linear(d_model, (d_k * n_heads * 3), bias=True)
        self.to_out = nn.Sequential(nn.Linear(d_v * n_heads, d_model, bias=True), nn.Dropout(attn_dropout))
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, Q, prev=None):
        bs = Q.size(0)
        qkv = self.W_QKV(Q).view(bs, -1, self.n_heads, 3 * (Q.size(-1) // self.n_heads))
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2).transpose(-2, -1)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k) * self.scale
        if prev is not None and self.res_attention: scores = scores + prev

        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v)

        output = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * context.size(-1))
        output = self.to_out(output)

        return output, scores if self.res_attention else None


class TSTEncoderLayer(nn.Module):
    """Transformer encoder layer with batch normalization."""

    def __init__(self, d_model, n_heads, dropout=0.):
        super().__init__()
        d_ff = d_model * 4

        self.self_attn = SelfAttention(d_model, n_heads, attn_dropout=dropout, res_attention=True)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.BatchNorm1d(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.BatchNorm1d(d_model)

    def _norm(self, norm_layer, x):
        return norm_layer(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, src, prev=None):
        src2, scores = self.self_attn(src, prev=prev)
        src = src + self.dropout_attn(src2)
        src = self._norm(self.norm_attn, src)

        src2 = self.ffn(src)
        src = self._norm(self.norm_ffn, src + src2)

        return src, scores


class TSTiEncoder(nn.Module):
    """Channel-independent transformer encoder for patches."""

    def __init__(self, n_vars, patch_num, patch_len, d_model, n_heads, n_layers, dropout=0.):
        super().__init__()
        self.n_vars = n_vars
        self.patch_num = patch_num

        # Patch embedding
        self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.W_pos = nn.Parameter(torch.zeros(patch_num, d_model), requires_grad=True)

        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.encoder_layers = nn.ModuleList([
            TSTEncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, x):
        # Embed patches
        x = self.W_P(x)

        # Reshape for channel-independent processing
        bs, _, _, d_model = x.shape
        x = x.reshape(-1, self.patch_num, d_model)

        # Add positional encoding
        x = self.dropout(x + self.W_pos)

        # Process through transformer
        scores = None
        for layer in self.encoder_layers:
            x, scores = layer(x, prev=scores)

        # Restore batch and channel dimensions
        x = x.reshape(bs, self.n_vars, self.patch_num, d_model)

        return x


class PatchWiseHead(nn.Module):
    """Patch-wise decoder that independently decodes each patch."""

    def __init__(self, individual, n_vars, d_model, patch_len, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        # Channel-specific or shared decoders
        if individual:
            self.decoders = nn.ModuleList([
                nn.Linear(d_model, patch_len) for _ in range(n_vars)
            ])
        else:
            self.decoder = nn.Linear(d_model, patch_len)

        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        batch_size, n_vars, n_patches, d_model = x.shape

        if self.individual:
            x_out = []
            for i in range(n_vars):
                decoded = self.decoders[i](x[:, i, :, :])
                x_out.append(decoded)
            x = torch.stack(x_out, dim=1)
        else:
            x = x.reshape(-1, d_model)
            x = self.decoder(x)
            x = x.reshape(batch_size, n_vars, n_patches, -1)

        x = self.dropout(x)

        # Flatten patches into continuous sequence
        x = x.reshape(batch_size, n_vars, -1)

        return x


class PatchTSTPatchWise(nn.Module):
    """
    PatchTST with patch-wise decoding.
    Preserves patch structure throughout the forecasting pipeline.
    """

    def __init__(self, config, use_revin=True, individual=False):
        super().__init__()

        D_MODEL = config.D_MODEL_BASE
        N_HEADS = config.N_HEADS_BASE
        E_LAYERS = config.E_LAYERS_BASE

        self.patch_len = config.PATCH_LEN
        self.stride = config.STRIDE
        self.use_revin = config.USE_REVIN
        self.n_vars = len(config.FEATURES)
        self.individual = individual
        self.pred_len = config.PRED_LEN

        # Calculate patches
        n_patches = (config.SEQ_LEN - self.patch_len) // config.STRIDE + 1
        self.n_pred_patches = (config.PRED_LEN + self.patch_len - 1) // self.patch_len

        # RevIN normalization
        self.revin = RevIN(self.n_vars) if self.use_revin else nn.Identity()

        # Transformer encoder
        self.backbone = TSTiEncoder(
            n_vars=self.n_vars,
            patch_num=n_patches,
            patch_len=self.patch_len,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=E_LAYERS,
            dropout=config.DROPOUT
        )

        # Patch-wise decoder
        self.head = PatchWiseHead(
            individual=individual,
            n_vars=self.n_vars,
            d_model=D_MODEL,
            patch_len=self.patch_len,
            head_dropout=0.0
        )

    def forward(self, x):
        # Instance normalization
        if self.use_revin: x = self.revin(x, mode='norm')

        # Create patches
        x = x.transpose(1, 2)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Process through encoder
        x = self.backbone(x)

        # Select patches for prediction
        pred_patch_reprs = x[:, :, -self.n_pred_patches:, :]

        # Decode patches
        x = self.head(pred_patch_reprs)

        # Trim and transpose
        x = x[:, :, :self.pred_len]
        x = x.transpose(1, 2)

        # Denormalization
        if self.use_revin: x = self.revin(x, mode='denorm')

        return x

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())

        if self.individual:
            head_params = sum(p.numel() for decoder in self.head.decoders for p in decoder.parameters())
        else:
            head_params = sum(p.numel() for p in self.head.decoder.parameters())

        return {
            'total': total_params,
            'encoder': total_params - head_params,
            'head': head_params,
            'patch_wise_head': head_params,
            'head_percentage': 100 * head_params / total_params if total_params > 0 else 0
        }