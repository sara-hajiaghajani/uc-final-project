"""
aLLM4TS - Large Language Model for Time Series

Adapts pre-trained GPT-2 for time series forecasting.
Uses channel-dependent patching and fine-tunes with frozen transformer blocks.

Architecture: CD Patching -> GPT-2 Backbone -> Patch-wise Decoder
"""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
import warnings


class PatchEmbedding(nn.Module):
    """Channel-dependent patch embedding for multivariate time series."""

    def __init__(self, patch_len, stride, n_features, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        self.value_embedding = nn.Linear(patch_len * n_features, d_model, bias=False)

    def forward(self, x):
        batch_size = x.shape[0]

        # Create overlapping patches
        x = x.transpose(1, 2)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Flatten channels within each patch
        batch_size, n_features, n_patches, patch_len = patches.shape
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.reshape(batch_size, n_patches, -1)

        # Project to d_model
        patch_embeddings = self.value_embedding(patches)
        return patch_embeddings


class OfficialALLM4TSModel(nn.Module):
    """
    GPT-2 adapted for time series forecasting.
    Freezes transformer blocks to preserve pre-trained knowledge.
    """

    def __init__(self, config, freeze_backbone=True):
        super().__init__()

        D_MODEL = config.D_MODEL_LLM
        N_HEADS = config.N_HEADS_LLM
        E_LAYERS = config.E_LAYERS_LLM

        self.config = config
        self.patch_len = config.PATCH_LEN
        self.n_features = len(config.FEATURES)
        self.pred_len = config.PRED_LEN
        self.d_model = D_MODEL

        # Calculate patch dimensions
        self.n_input_patches = (config.SEQ_LEN - self.patch_len) // config.STRIDE + 1
        self.n_pred_patches = (config.PRED_LEN + self.patch_len - 1) // config.PATCH_LEN

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_len=config.PATCH_LEN,
            stride=config.STRIDE,
            n_features=self.n_features,
            d_model=D_MODEL
        )

        # Load GPT-2
        try:
            gpt2_config = GPT2Config(
                n_embd=D_MODEL,
                n_layer=E_LAYERS,
                n_head=N_HEADS,
                n_positions=self.n_input_patches + self.n_pred_patches + 10,
                resid_pdrop=config.DROPOUT,
                embd_pdrop=config.DROPOUT,
            )

            # Suppress shape mismatch warnings (expected for positional embeddings)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Some weights of GPT2Model")
                self.gpt2 = GPT2Model.from_pretrained('gpt2', config=gpt2_config, ignore_mismatched_sizes=True)

            print(f"Loaded GPT-2 with {E_LAYERS} layers (positional embeddings resized)")

            # Freeze transformer blocks
            if freeze_backbone:
                for param in self.gpt2.h.parameters():
                    param.requires_grad = False
                for param in self.gpt2.ln_f.parameters():
                    param.requires_grad = True

        except Exception as e:
            warnings.warn(f"GPT-2 loading failed: {e}. Using random weights.")
            gpt2_config = GPT2Config(n_embd=D_MODEL, n_layer=E_LAYERS, n_head=N_HEADS, n_positions=1024)
            self.gpt2 = GPT2Model(gpt2_config)

        # Learnable positional embeddings
        max_patches = self.n_input_patches + self.n_pred_patches
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_patches, D_MODEL))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        # Patch-wise decoder
        self.patch_decoder = nn.Linear(D_MODEL, config.PATCH_LEN * self.n_features)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding
        tokens = self.patch_embedding(x)
        n_patches = tokens.shape[1]

        # Add positional encoding
        tokens = tokens + self.pos_embedding[:, :n_patches, :]
        tokens = self.dropout(tokens)

        # GPT-2 processing
        outputs = self.gpt2(inputs_embeds=tokens, use_cache=False)
        hidden_states = outputs.last_hidden_state

        # Select patches for prediction
        if n_patches >= self.n_pred_patches:
            pred_patch_reprs = hidden_states[:, -self.n_pred_patches:, :]
        else:
            last_repr = hidden_states[:, -1:, :]
            pred_patch_reprs = last_repr.repeat(1, self.n_pred_patches, 1)

        # Decode patches
        decoded_patches = self.patch_decoder(pred_patch_reprs)

        # Reshape and trim
        output = decoded_patches.reshape(batch_size, -1, self.n_features)
        output = output[:, :self.pred_len, :]

        return output

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.patch_decoder.parameters())
        gpt2_params = sum(p.numel() for p in self.gpt2.parameters())
        non_gpt2_params = total_params - gpt2_params

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'gpt2_backbone': gpt2_params,
            'adaptation_layers': non_gpt2_params,
            'patch_wise_head': decoder_params,
            'head': decoder_params,
            'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0
        }