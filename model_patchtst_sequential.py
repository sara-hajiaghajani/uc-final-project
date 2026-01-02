"""
PatchTST with Sequence-wise Decoder

Standard PatchTST with flatten-and-project decoder.
Flattens all patch representations before projecting to target sequence.

Architecture: Patching -> Transformer Encoder -> Flatten -> Linear Projection
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

        # Restore shape
        x = x.reshape(bs, self.n_vars, self.patch_num, d_model)

        # Permute for FlattenHead
        return x.permute(0, 1, 3, 2)


class FlattenHead(nn.Module):
    """Sequence-wise decoder that flattens all patch representations."""

    def __init__(self, n_vars, head_nf, target_window, individual=False, head_dropout=0.):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        # Channel-specific or shared projection
        if individual:
            self.linears = nn.ModuleList([nn.Linear(head_nf, target_window) for _ in range(n_vars)])
        else:
            self.linear = nn.Linear(head_nf, target_window)

        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # Flatten patch and model dimensions
        x = x.flatten(start_dim=-2)

        if self.individual:
            x_out = [self.dropout(self.linears[i](x[:, i, :])) for i in range(self.n_vars)]
            x = torch.stack(x_out, dim=1)
        else:
            x = self.dropout(self.linear(x))

        return x


class OfficialPatchTSTModel(nn.Module):
    """
    Standard PatchTST with sequence-wise decoding.
    Flattens all patches before projection.
    """

    def __init__(self, config, use_revin=True, individual=False):
        super().__init__()

        self.patch_len = config.PATCH_LEN
        self.stride = config.STRIDE
        self.use_revin = config.USE_REVIN
        self.n_vars = len(config.FEATURES)
        self.individual = individual

        # Calculate dimensions
        self.patch_num = (config.SEQ_LEN - self.patch_len) // self.stride + 1
        self.head_nf = config.D_MODEL_BASE * self.patch_num

        # RevIN normalization
        self.revin = RevIN(self.n_vars) if self.use_revin else nn.Identity()

        # Transformer encoder
        self.backbone = TSTiEncoder(
            n_vars=self.n_vars,
            patch_num=self.patch_num,
            patch_len=self.patch_len,
            d_model=config.D_MODEL_BASE,
            n_heads=config.N_HEADS_BASE,
            n_layers=config.E_LAYERS_BASE,
            dropout=config.DROPOUT
        )

        # Sequence-wise decoder
        self.head = FlattenHead(
            n_vars=self.n_vars,
            head_nf=self.head_nf,
            target_window=config.PRED_LEN,
            individual=self.individual
        )

    def forward(self, x):
        # Instance normalization
        if self.use_revin: x = self.revin(x, mode='norm')

        # Create patches
        x = x.transpose(1, 2)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Process through encoder
        x = self.backbone(x)

        # Flatten and project
        x = self.head(x)

        # Transpose and denormalize
        x = x.transpose(1, 2)
        if self.use_revin: x = self.revin(x, mode='denorm')

        return x

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())

        if self.individual:
            head_params = sum(p.numel() for linear in self.head.linears for p in linear.parameters())
        else:
            head_params = sum(p.numel() for p in self.head.linear.parameters())

        return {
            'total': total_params,
            'encoder': total_params - head_params,
            'head': head_params,
            'patch_wise_head': 0,
            'head_percentage': 100 * head_params / total_params if total_params > 0 else 0
        }