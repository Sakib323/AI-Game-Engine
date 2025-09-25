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
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed

from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.modules import RMSNorm, LayerNorm
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear
from mmfreelm.modules.activations import ACT2FN
from mmfreelm.modules.layernorm import LayerNorm

# --- Self-Contained Helper Functions & Classes ---

def modulate(x, shift, scale):
    """Applies affine modulation to the input tensor."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class HGRNBitMLP(nn.Module):
    """
    A standard MLP block using FusedBitLinear layers and Swish activation.
    This is a core component for feed-forward networks in the architecture.
    """
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, intermediate_size: Optional[int] = None, hidden_act: str = 'swish'):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * mlp_ratio)
        self.gate_proj = BitLinear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        # Applying swish activation
        z = self.down_proj(self.act_fn(gate) * y)
        return z

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal embeddings
    followed by an HGRNBitMLP.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            BitLinear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            BitLinear(hidden_size, hidden_size, bias=True),
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
    reference image conditioning in a matmul-free way.
    """
    def __init__(self, hidden_size: int, num_heads: int, **hgrn_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.self_attn = HGRNBitAttention(hidden_size=hidden_size, num_heads=num_heads, **hgrn_kwargs)
        self.mv_attn = HGRNBitAttention(hidden_size=hidden_size, num_heads=num_heads, **hgrn_kwargs)
        self.ref_attn = HGRNBitAttention(hidden_size=hidden_size, num_heads=num_heads, **hgrn_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_views: int,
        ref_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0] // num_views
        seq_len = hidden_states.shape[1]

        # 1. Standard Self-Attention (within each view)
        self_out, _, _ = self.self_attn(hidden_states)

        # 2. Multi-View Consistency Attention (across views)
        mv_in = rearrange(hidden_states, '(b n) l d -> (b l) n d', b=batch_size, n=num_views)
        mv_out, _, _ = self.mv_attn(mv_in)
        mv_out = rearrange(mv_out, '(b l) n d -> (b n) l d', b=batch_size, l=seq_len)

        # 3. Reference Image Cross-Attention (if provided)
        if ref_hidden_states is not None:
            ref_out, _, _ = self.ref_attn(ref_hidden_states.repeat_interleave(num_views, dim=0))
            final_out = self_out + mv_out + ref_out
        else:
            final_out = self_out + mv_out

        return final_out

class TernaryMVAdapterBlock(nn.Module):
    """
    A single block of the Ternary MV-Adapter, analogous to a Transformer block.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DecoupledMVHGRNAttention(hidden_size, num_heads)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = HGRNBitMLP(hidden_size, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            BitLinear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        num_views: int,
        ref_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        x_modulated = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(x_modulated, num_views, ref_hidden_states)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_modulated = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_modulated)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class SpatialCondAdapter(nn.Module):
    """
    A lightweight adapter to process spatial conditioning signals like position maps.
    """
    def __init__(self, in_channels, hidden_size, depth=3):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=None, patch_size=1, in_chans=in_channels, embed_dim=hidden_size)
        self.blocks = nn.ModuleList([
            nn.Sequential(LayerNorm(hidden_size), HGRNBitMLP(hidden_size)) for _ in range(depth)
        ])
        self.downsamplers = nn.ModuleList([
             nn.Conv2d(hidden_size, hidden_size, kernel_size=2, stride=2) for _ in range(depth)
        ])

    def forward(self, x_cond):
        x = self.patch_embed(x_cond)
        features = []
        for block, downsampler in zip(self.blocks, self.downsamplers):
            x = block(x)
            B, L, D = x.shape
            H = W = int(math.sqrt(L))
            x_reshaped = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
            x_down = downsampler(x_reshaped)
            features.append(rearrange(x_down, 'b d h w -> b (h w) d'))
            x = rearrange(x_down, 'b d h w -> b (h w) d')
        return features

class FinalLayer(nn.Module):
    """The final layer of the adapter."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = BitLinear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            BitLinear(hidden_size, 2 * hidden_size, bias=True)
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
        cond_channels: int = 3,
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
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size))
        self.spatial_adapter = SpatialCondAdapter(cond_channels, hidden_size, depth=3)
        self.blocks = nn.ModuleList([
            TernaryMVAdapterBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, BitLinear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.pos_embed, std=0.02)

    def unpatchify(self, x):
        """Converts tokens back to image shape."""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
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
        ref_hidden_states: Optional[torch.Tensor] = None
    ):
        x_embed = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        prompt_emb = encoder_hidden_states.mean(dim=1).repeat_interleave(num_views, dim=0)
        c = t_emb + prompt_emb

        spatial_features = self.spatial_adapter(control_image_feature) if control_image_feature is not None else [0] * 3

        for i, block in enumerate(self.blocks):
            if control_image_feature is not None and i < len(spatial_features) and x_embed.shape[1] == spatial_features[i].shape[1]:
                x_embed = x_embed + spatial_features[i]
            x_embed = block(x_embed, c, num_views, ref_hidden_states)
        
        output = self.final_layer(x_embed, c)
        output = self.unpatchify(output)
        return output

# --- Example Usage ---
if __name__ == '__main__':
    # Configuration
    BATCH_SIZE = 2
    NUM_VIEWS = 4
    IMG_SIZE = 32
    PATCH_SIZE = 2
    IN_CHANNELS = 4
    HIDDEN_SIZE = 512
    DEPTH = 12
    NUM_HEADS = 8
    COND_CHANNELS = 3
    TEXT_SEQ_LEN = 77
    TEXT_DIM = 768

    model = TernaryMVAdapter(
        input_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS,
        hidden_size=HIDDEN_SIZE, depth=DEPTH, num_heads=NUM_HEADS,
        cond_channels=COND_CHANNELS
    ).cuda()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Create dummy inputs
    dummy_latents = torch.randn(BATCH_SIZE * NUM_VIEWS, IN_CHANNELS, IMG_SIZE, IMG_SIZE).cuda()
    dummy_timesteps = torch.randint(0, 1000, (BATCH_SIZE * NUM_VIEWS,)).cuda()
    dummy_text_embeds = torch.randn(BATCH_SIZE, TEXT_SEQ_LEN, TEXT_DIM).cuda()
    dummy_control_image = torch.randn(BATCH_SIZE * NUM_VIEWS, COND_CHANNELS, IMG_SIZE, IMG_SIZE).cuda()
    dummy_ref_feats = torch.randn(BATCH_SIZE, (IMG_SIZE // PATCH_SIZE)**2, HIDDEN_SIZE).cuda()

    try:
        output = model(
            x=dummy_latents, t=dummy_timesteps, num_views=NUM_VIEWS,
            encoder_hidden_states=dummy_text_embeds,
            control_image_feature=dummy_control_image,
            ref_hidden_states=dummy_ref_feats
        )
        print("Forward pass successful!")
        print("Input shape:", dummy_latents.shape)
        print("Output shape:", output.shape)
        expected_out_channels = IN_CHANNELS * 2 if model.learn_sigma else IN_CHANNELS
        assert output.shape == (BATCH_SIZE * NUM_VIEWS, expected_out_channels, IMG_SIZE, IMG_SIZE)
        print("Output shape is correct.")
    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")

