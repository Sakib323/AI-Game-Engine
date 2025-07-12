# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
# The user will now import and use the tokenizer in their training script, not in the model file.
# from transformers import AutoTokenizer, AutoModel 

import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

# Assuming these are custom modules from your project structure.
# If these are not found, you might need to adjust the import paths.
try:
    from mmfreelm.layers.hgrn_bit import HGRNBitAttention
    from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
    from mmfreelm.models.utils import RecurrentCache
    from mmfreelm.modules import FusedCrossEntropyLoss, RMSNorm, LayerNorm
    from mmfreelm.modules.activations import swiglu_linear, swiglu, ACT2FN
    from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear
    from mmfreelm.models.hgrn_bit.hgrn_bit_moe import HGRNBitMoE
except ImportError:
    print("Warning: Custom mmfreelm modules not found. Using standard LayerNorm and a placeholder for BitLinear/HGRNBitAttention.")
    # Define placeholder modules if the custom ones are not available
    class BitLinear(nn.Linear):
        pass
    class HGRNBitAttention(nn.Module):
        def __init__(self, mode, hidden_size, num_heads, **kwargs):
            super().__init__()
            self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        def forward(self, x, **kwargs):
            return self.attn(x,x,x)[0], None, None # Return tuple similar to original
    from torch.nn import LayerNorm
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


logger = logging.get_logger(__name__)


def modulate(x, shift, scale):
    # Modulates input features x with learned shift and scale parameters.
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class HGRNBitMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) block using BitLinear layers.
    """
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ) -> HGRNBitMLP:
        super().__init__()

        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = BitLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        # Assuming swiglu is defined elsewhere, if not, use a standard implementation
        try:
            from mmfreelm.modules.activations import swiglu
            z = self.down_proj(swiglu(gate, y))
        except ImportError:
            z = self.down_proj(self.act_fn(gate) * y)
        return z

#################################################################################
#               Embedding Layers for Timesteps and Text                         #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # Creates sinusoidal timestep embeddings.
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TextEmbedder(nn.Module):
    """
    Embeds tokenized text into vector representations.
    This module now expects pre-tokenized input_ids and attention_mask.
    """
    def __init__(self, vocab_size, hidden_size, dropout_prob):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        # We add +1 for a learnable null embedding for classifier-free guidance.
        self.embedding = nn.Embedding(self.vocab_size + 1, hidden_size)
        self.null_idx = self.vocab_size  # Index for the null embedding

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def token_drop(self, input_ids, force_drop_ids=None):
        # Drops tokens by replacing them with the null token index for CFG.
        if force_drop_ids is None:
            drop_ids = torch.rand(input_ids.shape[0], device=input_ids.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        
        input_ids = torch.where(drop_ids.unsqueeze(-1), self.null_idx, input_ids)
        return input_ids

    def forward(self, input_ids, attention_mask, train=True, force_drop_ids=None):
        # Apply token dropout for classifier-free guidance
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            input_ids = self.token_drop(input_ids, force_drop_ids)

        embeddings = self.embedding(input_ids)  # [batch, seq_len, hidden_size]
        attention_mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        pooled_embeddings = (embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        final_embeddings = self.mlp(pooled_embeddings) # [batch, hidden_size]
        return final_embeddings


#################################################################################
#                                 Core DiT Block                                #
#################################################################################

class AdaLNConditioning(nn.Module):
    """
    Adaptive LayerNorm conditioning module. It generates shift, scale, and gate
    parameters from a conditioning signal.
    """
    def __init__(self, hidden_size: int, output_dim_mult: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            BitLinear(hidden_size, int(hidden_size * mlp_ratio * output_dim_mult), bias=True)
        )
        self.output_dim_mult = output_dim_mult
        
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        # The output dimension is 6 * hidden_size for DiTBlock (shift/scale/gate for attn/mlp)
        # and 2 * hidden_size for FinalLayer (shift/scale).
        return self.mlp(condition)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = HGRNBitAttention(
            mode='fused_recurrent', # Or use a standard nn.MultiheadAttention
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = HGRNBitMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim, hidden_act='swish')
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=1)
        
        # Attention block
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _, _ = self.attn(modulated_x)
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        # MLP block
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of the DiT, which projects features back to image space.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        params = self.adaLN_modulation(c)
        shift, scale = params.chunk(2, dim=1)
        
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

    
#################################################################################
#                   Original Single-View DiT Model (For Reference)              #
#################################################################################
        
class DiT(nn.Module):
    """
    Original single-view Diffusion model with a Transformer backbone.
    """ 
    def __init__(
        self,
        vocab_size: int = 2048,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.1,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(vocab_size, hidden_size, dropout_prob)
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        
    def initialize_weights(self):
        # Standard weight initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize positional embedding with sincos
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize specific layers
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def unpatchify(self, x):
        """Converts token sequences back into images."""
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        y: dict with "input_ids" and "attention_mask" for text conditioning.
        """
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y_emb = self.y_embedder(y["input_ids"], y["attention_mask"], train=self.training)
        c = t + y_emb
        
        for block in self.blocks:
            x = block(x, c)
        
        x = self.final_layer(x, c)
        return self.unpatchify(x)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """Forward pass with Classifier-Free Guidance."""
        half = x.shape[0] // 2
        x_combined = torch.cat([x[:half], x[:half]], dim=0)
        model_out = self.forward(x_combined, t, y)
        
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0)
        
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   NEW Multi-View DiT Model                                    #
#################################################################################

class MultiViewDiT(nn.Module):
    """
    Multi-View Diffusion model with a Transformer backbone.
    Processes multiple views of an object simultaneously to generate consistent outputs.
    """
    def __init__(
        self,
        vocab_size: int = 2048,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.1,
        learn_sigma: bool = True,
        num_views: int = 6, # New parameter for number of views
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_views = num_views

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(vocab_size, hidden_size, dropout_prob)
        
        num_patches = self.x_embedder.num_patches
        # Standard positional embedding for patches within a single view
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        # New learnable embedding to distinguish between different views
        self.view_embed = nn.Parameter(torch.zeros(1, self.num_views, 1, hidden_size))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Standard weight initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize positional embedding with sincos
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize the new view embeddings
        nn.init.normal_(self.view_embed, std=0.02)
        
        # Initialize specific layers
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, num_views: int):
        """
        Converts token sequences from multiple views back into image tensors.
        x: (N * V, T, P*P*C_out) tensor of tokens
        Returns: (N, V, C_out, H, W) tensor of images
        """
        N_times_V, T, _ = x.shape
        N = N_times_V // num_views
        C_out = self.out_channels
        P = self.patch_size
        H_patches = W_patches = int(T ** 0.5)
        assert H_patches * W_patches == T, "Number of patches must be a perfect square."

        # Reshape tokens into patches
        x = x.reshape(N_times_V, H_patches, W_patches, P, P, C_out)
        # Reorder dimensions to form image: (N*V, C, H, W)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(N_times_V, C_out, H_patches * P, W_patches * P)
        
        # Reshape to separate the view dimension: (N, V, C, H, W)
        _, C, H, W = imgs.shape
        return imgs.view(N, num_views, C, H, W)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: dict) -> torch.Tensor:
        """
        Forward pass of MultiViewDiT.
        x: (N, V, C, H, W) tensor of multi-view spatial inputs.
        t: (N,) tensor of diffusion timesteps.
        y: dict with "input_ids" and "attention_mask" for text conditioning.
        """
        N, V, C, H, W = x.shape
        assert V == self.num_views, f"Input has {V} views, but model is configured for {self.num_views}."

        # Reshape for patch embedding: (N, V, C, H, W) -> (N*V, C, H, W)
        x = x.reshape(N * V, C, H, W)
        x = self.x_embedder(x)  # (N*V, T, D), where T=num_patches, D=hidden_size

        # Reshape to add positional and view embeddings
        # (N*V, T, D) -> (N, V, T, D)
        x = x.view(N, V, -1, x.shape[-1])
        
        # Add positional embeddings (broadcasts across batch and views)
        # Add view embeddings (broadcasts across batch and patches)
        x = x + self.pos_embed.unsqueeze(1) + self.view_embed

        # Concatenate all tokens into a single sequence for the transformer
        # (N, V, T, D) -> (N, V*T, D)
        x = x.view(N, V * self.x_embedder.num_patches, x.shape[-1])

        # Prepare conditioning vector (same for all views of an object)
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y["input_ids"], y["attention_mask"], train=self.training)
        c = t_emb + y_emb  # (N, D)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Apply final layer
        x = self.final_layer(x, c)  # Output: (N, V*T, P*P*C_out)
        
        # Reshape and unpatchify to get final multi-view images
        return self.unpatchify(x, self.num_views)

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: dict, cfg_scale: float) -> torch.Tensor:
        """Forward pass with Classifier-Free Guidance for Multi-View."""
        N_half = x.shape[0] // 2
        # Duplicate the noisy latents for unconditional pass
        x_combined = torch.cat([x[:N_half], x[:N_half]], dim=0) # Shape: (N, V, C, H, W)
        
        # The 'y' dict must be prepared by the caller to have conditional and unconditional prompts
        model_out = self.forward(x_combined, t, y) # Output: (N, V, C_out, H, W)

        # Split predictions into epsilon and the rest (e.g., learned sigma)
        eps, rest = model_out[..., :self.in_channels, :, :], model_out[..., self.in_channels:, :, :]
        
        # Split into conditional and unconditional epsilon predictions
        cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0)
        
        # Combine using CFG scale
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        # Concatenate final epsilon with the rest of the predictions
        return torch.cat([eps, rest], dim=2) # Concatenate along the channel dimension


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# From https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


#################################################################################
#                           Multi-View DiT Configs                              #
#################################################################################
# You must provide `vocab_size` and can optionally provide `num_views`.
# e.g., MultiViewDiT_XL_2(vocab_size=tokenizer.vocab_size, num_views=6)

def MultiViewDiT_XL_2(**kwargs):
    return MultiViewDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def MultiViewDiT_L_2(**kwargs):
    return MultiViewDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def MultiViewDiT_B_2(**kwargs):
    return MultiViewDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def MultiViewDiT_S_2(**kwargs):
    return MultiViewDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


MultiViewDiT_models = {
    'MultiViewDiT-XL/2': MultiViewDiT_XL_2,
    'MultiViewDiT-L/2':  MultiViewDiT_L_2,
    'MultiViewDiT-B/2':  MultiViewDiT_B_2,
    'MultiViewDiT-S/2':  MultiViewDiT_S_2,
}
