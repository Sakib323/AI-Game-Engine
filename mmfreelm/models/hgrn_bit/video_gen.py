# video_dit.py

# -*- coding: utf-8 -*-
# Adapted for Text-to-Video Generation using Ternary Weights.
# This architecture treats Video as a 3D volume (Time, Height, Width).
# It uses "Tubelet" embedding (3D Patches) to flatten the video into a sequence,
# leveraging the linear complexity of HGRNBitAttention to handle long video sequences efficiently.

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.utils import logging
from einops import rearrange

# Core dependencies from your provided modules
from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.modules import RMSNorm, LayerNorm
from mmfreelm.ops.bitnet import BitLinear as StandardBitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear as FusedBitLinear

logger = logging.get_logger(__name__)


def modulate(x, shift, scale):
    """
    Helper function to apply adaptive layer normalization.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class HGRNBitMLP(nn.Module):
    """
    A custom MLP block used within the transformer.
    """
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
        optimized_bitlinear: bool = True,
        full_precision: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)

        if full_precision:
            LinearCls = nn.Linear
        else:
            LinearCls = FusedBitLinear if optimized_bitlinear else StandardBitLinear

        self.gate_proj = LinearCls(self.hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = LinearCls(intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        z = self.down_proj(self.act_fn(gate) * y)
        return z


class FullPrecisionMLP(nn.Module):
    """
    A standard MLP block using full-precision linear layers for conditioning.
    """
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ):
        super().__init__()

        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)

        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        z = self.down_proj(self.act_fn(gate) * y)
        return z


#################################################################################
#               Embedding Layers: Time, Text, and Video Tubelets                #
#################################################################################

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
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
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class TextEmbedder(nn.Module):
    """Embeds tokenized text into vector representations."""
    def __init__(self, vocab_size, hidden_size, dropout_prob):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, hidden_size)
        self.null_idx = vocab_size
        self.dropout_prob = dropout_prob
        self.mlp = FullPrecisionMLP(hidden_size, 4, hidden_act='swish')

    def token_drop(self, input_ids, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(input_ids.shape[0], device=input_ids.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids.unsqueeze(-1), self.null_idx, input_ids)

    def forward(self, input_ids, attention_mask, train=True, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            input_ids = self.token_drop(input_ids, force_drop_ids)

        embeddings = self.embedding(input_ids)
        attention_mask = attention_mask.unsqueeze(-1)
        pooled_embeddings = (embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        return self.mlp(pooled_embeddings).unsqueeze(1)


class VideoPatchEmbedder(nn.Module):
    """
    3D Tubelet Embedding.
    Embeds video latents [B, C, T, H, W] into a sequence of vectors.
    """
    def __init__(
        self, 
        input_channels: int, 
        hidden_size: int, 
        patch_size: Tuple[int, int, int], # (t, h, w)
        optimized_bitlinear: bool = True, 
        full_precision: bool = False
    ):
        super().__init__()
        self.patch_size = patch_size
        
        # We use a Conv3d to perform the tubelet embedding
        # Using standard Conv3d for input projection is standard even in BitNet papers to preserve input fidelity
        self.proj = nn.Conv3d(
            input_channels, 
            hidden_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.proj(x) # [B, Hidden, T_p, H_p, W_p]
        # Flatten spatial and temporal dimensions into a single sequence
        x = rearrange(x, 'b d t h w -> b (t h w) d')
        return x


class ImageConditionEmbedder(nn.Module):
    """
    Embeds a single image to use as a condition (e.g., first frame for Image-to-Video).
    """
    def __init__(self, input_channels, hidden_size, patch_size, dropout_prob):
        super().__init__()
        self.patch_embed = nn.Conv2d(input_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.null_embedding = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.dropout_prob = dropout_prob
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, image, train=True, force_drop_mask=None):
        # image: [B, C, H, W]
        x = self.patch_embed(image) # [B, D, H_p, W_p]
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.mlp(x)

        if force_drop_mask is not None:
            mask = force_drop_mask.float().unsqueeze(-1).unsqueeze(-1)
        elif train and self.dropout_prob > 0:
            mask = (torch.rand(image.shape[0], device=image.device) < self.dropout_prob).float().unsqueeze(-1).unsqueeze(-1)
        else:
            mask = torch.zeros(image.shape[0], 1, 1, device=image.device)
            
        # Broadcast null embedding across sequence length
        return mask * self.null_embedding + (1 - mask) * x


#################################################################################
#                          Transformer Blocks (Dual & Single)                   #
#################################################################################

class DualCrossAttention(nn.Module):
    """
    DoRA-inspired DUAL cross-attention (Bidirectional).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        rank: int = 16,
        optimized_bitlinear: bool = True,
        full_precision: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.scale = self.head_dim ** -0.5

        if full_precision:
            LinearCls = nn.Linear
        else:
            LinearCls = FusedBitLinear if optimized_bitlinear else StandardBitLinear

        # X -> C
        self.to_q_x = LinearCls(dim, dim, bias=False)
        self.to_kv_c = LinearCls(dim, dim * 2, bias=False)
        self.magnitude_x = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.lora_A_x = nn.Linear(dim, rank, bias=False)
        self.lora_B_x = nn.Linear(rank, dim, bias=False)
        self.proj_x = LinearCls(dim, dim, bias=False)

        # C -> X
        self.to_q_c = LinearCls(dim, dim, bias=False)
        self.to_kv_x = LinearCls(dim, dim * 2, bias=False)
        self.magnitude_c = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.lora_A_c = nn.Linear(dim, rank, bias=False)
        self.lora_B_c = nn.Linear(rank, dim, bias=False)
        self.proj_c = LinearCls(dim, dim, bias=False)

    def _attention_pass(self, query, context, to_q, to_kv, magnitude, lora_A, lora_B, proj):
        B, N_q, C = query.shape
        _, N_c, _ = context.shape

        q_proj = to_q(query)
        delta = lora_B(lora_A(query))
        
        q_reshaped = q_proj.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        delta_reshaped = delta.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q_final = q_reshaped + magnitude * delta_reshaped

        k, v = to_kv(context).chunk(2, dim=-1)
        k = k.reshape(B, N_c, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N_c, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q_final, k, v)
        out = attn.transpose(1, 2).reshape(B, N_q, C)
        return proj(out)

    def forward(self, x, c):
        # X attends to C
        x_out = x + self._attention_pass(
            query=x, context=c, to_q=self.to_q_x, to_kv=self.to_kv_c,
            magnitude=self.magnitude_x, lora_A=self.lora_A_x, lora_B=self.lora_B_x,
            proj=self.proj_x
        )
        # C attends to X
        c_out = c + self._attention_pass(
            query=c, context=x, to_q=self.to_q_c, to_kv=self.to_kv_x,
            magnitude=self.magnitude_c, lora_A=self.lora_A_c, lora_B=self.lora_B_c,
            proj=self.proj_c
        )
        return x_out, c_out


class FullPrecisionAdaLNConditioning(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, eps: float = 1e-6, hidden_ratio: Optional[int] = None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size, bias=False)
        self.mlp = FullPrecisionMLP(hidden_size=hidden_size, hidden_ratio=hidden_ratio, hidden_act='swish')
        self.output_proj = nn.Linear(hidden_size, output_dim, bias=True)
        self.norm = RMSNorm(output_dim, eps=eps)
        self.out_proj = nn.Linear(output_dim, output_dim, bias=True)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(condition)
        x = self.mlp(x)
        x = self.output_proj(x)
        x = self.norm(x)
        return self.out_proj(x)


class DualStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, use_rope: bool = False, use_ternary_rope: bool = False, optimized_bitlinear: bool = True, full_precision: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # HGRNBitAttention handles long sequence lengths (Video) efficiently
        self.attn_x = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, expand_ratio=1, use_short_conv=False, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.attn_c = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, expand_ratio=1, use_short_conv=False, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.dual_cross_attn = DualCrossAttention(hidden_size, num_heads, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.mlp = HGRNBitMLP(hidden_size, hidden_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.mlp_c = HGRNBitMLP(hidden_size, hidden_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.adaLN_modulation = FullPrecisionAdaLNConditioning(hidden_size, hidden_size, 12 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio)

    def forward(self, x, c, t):
        modulated_t = ACT2FN['silu'](t)
        params = self.adaLN_modulation(modulated_t)
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x, \
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = params.chunk(12, dim=1)

        # Video Stream (X)
        modulated_x = modulate(self.norm1(x), shift_msa_x, scale_msa_x)
        attn_x, _, _ = self.attn_x(modulated_x)
        x = x + gate_msa_x.unsqueeze(1) * attn_x

        modulated_x = modulate(self.norm2(x), shift_mlp_x, scale_mlp_x)
        mlp_x = self.mlp(modulated_x)
        x = x + gate_mlp_x.unsqueeze(1) * mlp_x

        # Prompt Stream (C)
        modulated_c = modulate(self.norm3(c), shift_msa_c, scale_msa_c)
        attn_c, _, _ = self.attn_c(modulated_c)
        c = c + gate_msa_c.unsqueeze(1) * attn_c

        modulated_c = modulate(self.norm4(c), shift_mlp_c, scale_mlp_c)
        mlp_c = self.mlp_c(modulated_c)
        c = c + gate_mlp_c.unsqueeze(1) * mlp_c

        # Fusion
        x, c = self.dual_cross_attn(x, c)
        return x, c


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, use_rope: bool = False, use_ternary_rope: bool = False, optimized_bitlinear: bool = True, full_precision: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, expand_ratio=1, use_short_conv=False, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.mlp = HGRNBitMLP(hidden_size, hidden_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.adaLN_modulation = FullPrecisionAdaLNConditioning(hidden_size, hidden_size, 6 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio)

    def forward(self, x, c):
        modulated_c = ACT2FN['silu'](c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(modulated_c).chunk(6, dim=1)

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _, _ = self.attn(modulated_x)
        x = x + gate_msa.unsqueeze(1) * attn_output

        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(mlp_input)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_dim, mlp_ratio=4.0, optimized_bitlinear: bool = True, full_precision: bool = False):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if full_precision:
            LinearCls = nn.Linear
        else:
            LinearCls = FusedBitLinear if optimized_bitlinear else StandardBitLinear
        self.linear = LinearCls(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = FullPrecisionAdaLNConditioning(hidden_size, hidden_size, 2 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio)

    def forward(self, x, c):
        modulated_c = ACT2FN['silu'](c)
        shift, scale = self.adaLN_modulation(modulated_c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


#################################################################################
#                          VideoDiT Main Class                                  #
#################################################################################

class VideoDiT(nn.Module):
    """
    Ternary Diffusion Transformer for Text-to-Video Generation.
    Uses 3D Tubelet embeddings and Dual-Stream architecture.
    """
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (16, 64, 64), # (Frames, Height, Width)
        patch_size: Tuple[int, int, int] = (1, 2, 2),    # (t, h, w) patch dimensions
        in_channels: int = 4,                            # VAE Latent channels
        vocab_size: int = 49408,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.1,
        num_dual_stream_blocks: int = 14,
        use_rope: bool = False,
        use_ternary_rope: bool = False,
        first_frame_condition: bool = False, # For Image-to-Video
        optimized_bitlinear: bool = True,
        full_precision: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2  # Learn sigma
        self.input_size = input_size # (T, H, W)
        self.patch_size = patch_size
        self.first_frame_condition = first_frame_condition

        # 1. Calculate sequence length
        self.t_patches = input_size[0] // patch_size[0]
        self.h_patches = input_size[1] // patch_size[1]
        self.w_patches = input_size[2] // patch_size[2]
        self.num_patches = self.t_patches * self.h_patches * self.w_patches
        
        # Output dim per patch = patch_volume * out_channels
        self.patch_dim = (patch_size[0] * patch_size[1] * patch_size[2]) * self.out_channels

        if full_precision:
            LinearCls = nn.Linear
        else:
            LinearCls = FusedBitLinear if optimized_bitlinear else StandardBitLinear

        # 2. Embedders
        self.x_embedder = VideoPatchEmbedder(in_channels, hidden_size, patch_size, optimized_bitlinear, full_precision)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(vocab_size, hidden_size, dropout_prob)
        
        if first_frame_condition:
            # Embeds a 2D image (first frame) to condition the generation
            self.img_embedder = ImageConditionEmbedder(in_channels, hidden_size, patch_size[1], dropout_prob)

        # No absolute positional embeddings (using RoPE in HGRN or implicit)
        logger.info(f"Initialized VideoDiT with {self.num_patches} tokens per video.")

        # 3. Blocks
        self.dual_stream_blocks = nn.ModuleList([
            DualStreamBlock(hidden_size, num_heads, mlp_ratio, use_rope, use_ternary_rope, optimized_bitlinear, full_precision)
            for _ in range(num_dual_stream_blocks)
        ])
        num_single_stream_blocks = depth - num_dual_stream_blocks
        self.single_stream_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio, use_rope, use_ternary_rope, optimized_bitlinear, full_precision)
            for _ in range(num_single_stream_blocks)
        ])

        self.final_layer = FinalLayer(hidden_size, self.patch_dim, mlp_ratio, optimized_bitlinear, full_precision)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, StandardBitLinear, FusedBitLinear, nn.Conv3d, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
        
        # Zero-out output layers for better convergence at start
        for block in self.dual_stream_blocks:
            nn.init.constant_(block.adaLN_modulation.output_proj.weight, 0)
            nn.init.constant_(block.adaLN_modulation.output_proj.bias, 0)
        for block in self.single_stream_blocks:
            nn.init.constant_(block.adaLN_modulation.output_proj.weight, 0)
            nn.init.constant_(block.adaLN_modulation.output_proj.bias, 0)
            
        nn.init.constant_(self.final_layer.adaLN_modulation.output_proj.weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation.output_proj.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Reconstructs the 3D video volume from the token sequence.
        x: [B, L, patch_dim]
        Returns: [B, out_channels, T, H, W]
        """
        b, l, d = x.shape
        c_out = self.out_channels
        pt, ph, pw = self.patch_size
        nt, nh, nw = self.t_patches, self.h_patches, self.w_patches
        
        assert l == nt * nh * nw, "Token count mismatch during unpatchify"

        # 1. Reshape into grid of patches
        x = x.reshape(b, nt, nh, nw, pt, ph, pw, c_out)
        
        # 2. Permute to gather spatial/temporal dims: [B, C, nt, pt, nh, ph, nw, pw]
        x = torch.einsum('bthwzyxc->bctzhwyx', x)
        
        # 3. Combine patch dims with grid dims
        x = x.reshape(b, c_out, nt * pt, nh * ph, nw * pw)
        return x

    def forward(self, x, t, y, first_frame=None):
        """
        x: [B, C, T, H, W] - Noisy Video Latents
        t: [B] - Timesteps
        y: Dict with "input_ids", "attention_mask"
        first_frame: [B, C, H, W] - Optional conditioning (e.g., first frame latent)
        """
        num_x_tokens = self.num_patches
        
        # Embeddings
        x_tokens = self.x_embedder(x) # [B, L, D]
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y["input_ids"], y["attention_mask"], train=self.training)

        # Handle Image Conditioning
        if self.first_frame_condition:
            assert first_frame is not None, "first_frame required when first_frame_condition=True"
            img_tokens = self.img_embedder(first_frame, train=self.training)
        else:
            # Create a minimal null embedding to match structure if needed, or just skip
            # Here we assume we append *something* if the architecture expects consistency
            # But for variable length sequences (HGRN), we can just append nothing if disabled.
            # However, to keep the y-stream logic simple, we usually only append if active.
            img_tokens = torch.tensor([], device=x.device) 
            # If you want strict structure like MeshDiT, use a null param:
            # img_tokens = self.img_embedder.null_embedding.expand(...)
            pass

        if self.first_frame_condition:
             y_tokens = torch.cat([y_emb, img_tokens], dim=1)
        else:
             y_tokens = y_emb

        # Dual Stream Blocks (Separate processing)
        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb)

        # Concatenate for Single Stream
        combined_tokens = torch.cat([x_tokens, y_tokens], dim=1)

        # Single Stream Blocks (Joint processing)
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)

        # Output Projection
        processed_x_tokens = combined_tokens[:, :num_x_tokens]
        output = self.final_layer(processed_x_tokens, t_emb)
        
        return self.unpatchify(output)

    def forward_with_cfg(self, x, t, y, cfg_scale_text, first_frame=None, cfg_scale_img=1.0):
        """
        Classifier-Free Guidance Forward Pass.
        """
        half = x.shape[0] // 2
        
        # Duplicate input for Conditional + Unconditional pass
        x_tokens = self.x_embedder(x)
        x_tokens = torch.cat([x_tokens[:half]] * 4, dim=0) if self.first_frame_condition else torch.cat([x_tokens[:half]] * 2, dim=0)
        
        t_in = torch.cat([t[:half]] * 4, dim=0) if self.first_frame_condition else torch.cat([t[:half]] * 2, dim=0)
        t_emb = self.t_embedder(t_in)

        # Prepare Text Condition (with dropout)
        y_ids = y["input_ids"][:half]
        y_mask = y["attention_mask"][:half]
        
        if self.first_frame_condition:
            # Batch structure: [Uncond, Text_Cond, Img_Cond, Both_Cond]
            text_drop_mask = torch.tensor([1, 0, 1, 0], device=x.device).repeat_interleave(half)
            y_emb = self.y_embedder(y_ids.repeat(4, 1), y_mask.repeat(4, 1), force_drop_ids=text_drop_mask)
            
            img_in = first_frame[:half]
            img_drop_mask = torch.tensor([1, 1, 0, 0], device=x.device).repeat_interleave(half)
            img_tokens = self.img_embedder(img_in.repeat(4, 1, 1, 1), force_drop_mask=img_drop_mask)
            y_tokens = torch.cat([y_emb, img_tokens], dim=1)
        else:
            # Batch structure: [Uncond, Text_Cond]
            text_drop_mask = torch.tensor([1, 0], device=x.device).repeat_interleave(half)
            y_emb = self.y_embedder(y_ids.repeat(2, 1), y_mask.repeat(2, 1), force_drop_ids=text_drop_mask)
            y_tokens = y_emb

        # Process
        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb)
        
        combined_tokens = torch.cat([x_tokens, y_tokens], dim=1)
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)
            
        processed_x_tokens = combined_tokens[:, :self.num_patches]
        model_out = self.final_layer(processed_x_tokens, t_emb)

        if self.first_frame_condition:
            # Split: Uncond, TextOnly, ImgOnly, Both
            e_uncond, e_text, e_img, e_both = torch.chunk(model_out, 4, dim=0)
            # Compositional CFG
            noise_pred = e_uncond + cfg_scale_text * (e_text - e_uncond) + cfg_scale_img * (e_img - e_uncond)
        else:
            e_uncond, e_text = torch.chunk(model_out, 2, dim=0)
            noise_pred = e_uncond + cfg_scale_text * (e_text - e_uncond)

        return noise_pred


#################################################################################
#                                VideoDiT Configs                               #
#################################################################################

def VideoDiT_XL(**kwargs):
    return VideoDiT(depth=28, hidden_size=1152, num_heads=16, num_dual_stream_blocks=14, **kwargs)

def VideoDiT_L(**kwargs):
    return VideoDiT(depth=24, hidden_size=1024, num_heads=16, num_dual_stream_blocks=12, **kwargs)

def VideoDiT_B(**kwargs):
    return VideoDiT(depth=12, hidden_size=768, num_heads=12, num_dual_stream_blocks=6, **kwargs)

def VideoDiT_S(**kwargs):
    return VideoDiT(depth=12, hidden_size=384, num_heads=6, num_dual_stream_blocks=6, **kwargs)


VideoDiT_models = {
    'VideoDiT-XL': VideoDiT_XL,
    'VideoDiT-L':  VideoDiT_L,
    'VideoDiT-B':  VideoDiT_B,
    'VideoDiT-S':  VideoDiT_S,
}