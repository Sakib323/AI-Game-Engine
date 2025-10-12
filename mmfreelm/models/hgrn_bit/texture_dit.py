# texture_dit.py

# -*- coding: utf-8 -*-
"""
Ternary MV-Adapter: A Matmul-Free Multi-View Diffusion Adapter with Ternary Weights.

This script implements a novel multi-view adapter that leverages the HGRN-BitNet
architecture to create a parameter-efficient, matmul-free model. It follows the
conceptual design of the original MV-Adapter by separating geometric conditioning
from appearance/consistency enforcement, but replaces the core components with
ternary-weight layers.

Key Architectural Features:
1.  **Ternary Operations**: Replaces all nn.Linear layers with `FusedBitLinear` from
    the provided scripts, which uses 1.58-bit ternary weight quantization and
    8-bit activation quantization. This drastically reduces the model size and
    computational cost.
2.  **Matmul-Free Attention**: The standard multi-head self-attention is replaced
    with `HGRNBitAttention`. This recurrent-based mechanism avoids large matrix
    multiplications, making it highly efficient.
3.  **Decoupled Multi-View Consistency**: A new `DecoupledMVHGRNAttention` module
    is introduced. It wraps the base `HGRNBitAttention` to intelligently reshape
    and process feature maps, allowing it to enforce consistency across multiple
    views in a way analogous to the original full-precision model.
4.  **Spatial Conditioning**: A lightweight spatial adapter (`SpatialCondAdapter`)
    processes geometric inputs (like position maps) using the same ternary building
    blocks, injecting spatial guidance into the main model.
5.  **DiT-Style Backbone**: The overall structure is inspired by Diffusion Transformers
    (DiT), using a series of `TernaryMVAdapterBlock` modules to process the latent
    representations, conditioned on timestep and text embeddings.
UPDATED: Added full_precision and optimized_bitlinear arguments for consistency with mesh_dit.py.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed

from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.modules import LayerNorm
from mmfreelm.modules.activations import ACT2FN
from mmfreelm.ops.bitnet import BitLinear as StandardBitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear as FusedBitLinear

def modulate(x, shift, scale):
    """Applies affine modulation to the input tensor."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class HGRNBitMLP(nn.Module):
    """
    A standard MLP block using FusedBitLinear layers and Swish activation.
    """
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, intermediate_size: Optional[int] = None, hidden_act: str = 'swish', optimized_bitlinear: bool = True, full_precision: bool = False):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * mlp_ratio)
        
        if full_precision:
            LinearCls = nn.Linear
        else:
            LinearCls = FusedBitLinear if optimized_bitlinear else StandardBitLinear

        self.gate_proj = LinearCls(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = LinearCls(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        z = self.down_proj(self.act_fn(gate) * y)
        return z

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, optimized_bitlinear: bool = True, full_precision: bool = False):
        super().__init__()
        if full_precision:
            LinearCls = nn.Linear
        else:
            LinearCls = FusedBitLinear if optimized_bitlinear else StandardBitLinear
        
        self.mlp = nn.Sequential(
            LinearCls(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            LinearCls(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
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

# --- Core Ternary MV-Adapter Modules ---

class DecoupledMVHGRNAttention(nn.Module):
    """
    A wrapper around HGRNBitAttention to handle decoupled multi-view and
    reference image conditioning.
    """
    def __init__(self, hidden_size: int, num_heads: int, optimized_bitlinear: bool = True, full_precision: bool = False, **hgrn_kwargs):
        super().__init__()
        self.self_attn = HGRNBitAttention(hidden_size=hidden_size, num_heads=num_heads, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision, **hgrn_kwargs)
        self.mv_attn = HGRNBitAttention(hidden_size=hidden_size, num_heads=num_heads, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision, **hgrn_kwargs)
        self.ref_attn = HGRNBitAttention(hidden_size=hidden_size, num_heads=num_heads, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision, **hgrn_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_views: int,
        ref_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0] // num_views
        seq_len = hidden_states.shape[1]

        self_out, _, _ = self.self_attn(hidden_states)

        mv_in = rearrange(hidden_states, '(b n) l d -> (b l) n d', b=batch_size, n=num_views)
        mv_out, _, _ = self.mv_attn(mv_in)
        mv_out = rearrange(mv_out, '(b l) n d -> (b n) l d', b=batch_size, l=seq_len)

        if ref_tokens is not None:
            ref_out, _, _ = self.ref_attn(ref_tokens.repeat_interleave(num_views, dim=0))
            final_out = self_out + mv_out + ref_out
        else:
            final_out = self_out + mv_out
        return final_out

class TernaryMVAdapterBlock(nn.Module):
    """
    A single block of the Ternary MV-Adapter.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, optimized_bitlinear: bool = True, full_precision: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DecoupledMVHGRNAttention(hidden_size, num_heads, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = HGRNBitMLP(hidden_size, mlp_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        
        if full_precision:
            LinearCls = nn.Linear
        else:
            LinearCls = FusedBitLinear if optimized_bitlinear else StandardBitLinear

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            LinearCls(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        num_views: int,
        ref_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        x_modulated = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(x_modulated, num_views, ref_tokens)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_modulated = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_modulated)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x

class SpatialCondAdapter(nn.Module):
    """
    A lightweight adapter for spatial conditioning.
    """
    def __init__(self, in_channels: int, hidden_size: int, patch_size=16, depth=3, optimized_bitlinear: bool = True, full_precision: bool = False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=None, patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size)
        self.blocks = nn.ModuleList([
            nn.Sequential(LayerNorm(hidden_size), HGRNBitMLP(hidden_size, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)) for _ in range(depth)
        ])
        self.downsamplers = nn.ModuleList([
             nn.Conv2d(hidden_size, hidden_size, kernel_size=2, stride=2) for _ in range(depth)
        ])

    def forward(self, x_cond):
        x = self.patch_embed(x_cond)
        features = []
        num_feature_blocks = min(len(self.blocks), len(self.downsamplers))

        for i in range(num_feature_blocks):
            x = self.blocks[i](x)
            B, L, D = x.shape
            H = W = int(math.sqrt(L))
            assert H * W == L, f"The number of patches ({L}) is not a perfect square."

            x_reshaped = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
            x_down = self.downsamplers[i](x_reshaped)
            features.append(rearrange(x_down, 'b d h w -> b (h w) d'))
            x = rearrange(x_down, 'b d h w -> b (h w) d')
        return features


class FinalLayer(nn.Module):
    """The final layer of the adapter."""
    def __init__(self, hidden_size, patch_size, out_channels, optimized_bitlinear: bool = True, full_precision: bool = False):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        if full_precision:
            LinearCls = nn.Linear
        else:
            LinearCls = FusedBitLinear if optimized_bitlinear else StandardBitLinear

        self.linear = LinearCls(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            LinearCls(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# --- Main Model ---

class TernaryMVAdapter(nn.Module):
    """
    The main Ternary Multi-View Adapter model.
    """
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        cond_channels: int = 6,
        text_embed_dim: int = 768,
        learn_sigma: bool = True,
        optimized_bitlinear: bool = True,
        full_precision: bool = False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size

        if full_precision:
            LinearCls = nn.Linear
        else:
            LinearCls = FusedBitLinear if optimized_bitlinear else StandardBitLinear

        self.x_embedder = PatchEmbed(img_size=input_size, patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size))
        self.prompt_proj = LinearCls(text_embed_dim, hidden_size, bias=True)
        self.spatial_adapter = SpatialCondAdapter(cond_channels, hidden_size, patch_size=16, depth=3, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.blocks = nn.ModuleList([
            TernaryMVAdapterBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, StandardBitLinear, FusedBitLinear)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.pos_embed, std=0.02)

    def unpatchify(self, x):
        """Converts tokens back to image shape."""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1], "Cannot unpatchify if not a square number of tokens."
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        num_views: int,
        encoder_hidden_states: torch.Tensor,
        control_image_feature: Optional[torch.Tensor] = None,
        ref_hidden_states: Optional[torch.Tensor] = None # This is the raw VAE latent
    ):
        # 1. Process the main noisy latent
        x_embed = self.x_embedder(x) + self.pos_embed
        
        # 2. Process the reference image latent (THE FIX IS HERE)
        ref_tokens = None
        if ref_hidden_states is not None:
            # Use the same embedder to patchify and project the reference latent
            ref_tokens = self.x_embedder(ref_hidden_states) + self.pos_embed

        # 3. Process time and text embeddings
        t_emb = self.t_embedder(t)
        raw_prompt_emb = encoder_hidden_states.mean(dim=1).repeat_interleave(num_views, dim=0)
        prompt_emb = self.prompt_proj(raw_prompt_emb)
        c = t_emb + prompt_emb

        # 4. Process spatial control features
        spatial_features = self.spatial_adapter(control_image_feature) if control_image_feature is not None else [0] * 3

        # 5. Run through the main blocks
        for i, block in enumerate(self.blocks):
            block_c = c
            if control_image_feature is not None and i < len(spatial_features):
                spatial_feature_pooled = spatial_features[i].mean(dim=1)
                block_c = block_c + spatial_feature_pooled

            # Pass the correctly processed ref_tokens to the block
            x_embed = block(x_embed, block_c, num_views, ref_tokens)

        output = self.final_layer(x_embed, c)
        output = self.unpatchify(output)
        return output

#################################################################################
#              Scalable Model Variants (S, B, L, XL)                          #
#################################################################################

def TernaryMVAdapter_XL(**kwargs):
    """Extra-Large model variant."""
    return TernaryMVAdapter(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def TernaryMVAdapter_L(**kwargs):
    """Large model variant."""
    return TernaryMVAdapter(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def TernaryMVAdapter_B(**kwargs):
    """Base model variant."""
    return TernaryMVAdapter(depth=12, hidden_size=768, num_heads=12, **kwargs)

def TernaryMVAdapter_S(**kwargs):
    """Small model variant."""
    return TernaryMVAdapter(depth=12, hidden_size=384, num_heads=6, **kwargs)

# Dictionary to easily access the models by name
TernaryMVAdapter_models = {
    'TernaryMVAdapter-XL': TernaryMVAdapter_XL,
    'TernaryMVAdapter-L': TernaryMVAdapter_L,
    'TernaryMVAdapter-B': TernaryMVAdapter_B,
    'TernaryMVAdapter-S': TernaryMVAdapter_S,
} 