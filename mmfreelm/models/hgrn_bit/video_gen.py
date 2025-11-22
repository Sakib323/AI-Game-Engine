# video_gen.py
# -*- coding: utf-8 -*-
# Unified Video Generation Model: Flow Matching + Decoupled Spatial/Temporal Attention
# Merges concepts from VideoDiT (Flow) and TextureDiT (Decoupled Attention/Refinement)

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.utils import logging
from einops import rearrange

# Core dependencies
from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.modules import RMSNorm, LayerNorm
from mmfreelm.ops.bitnet import BitLinear as StandardBitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear as FusedBitLinear

logger = logging.get_logger(__name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def modulate(x, shift, scale):
    """Helper function to apply adaptive layer normalization."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# -----------------------------------------------------------------------------
# MLPs and Basic Blocks
# -----------------------------------------------------------------------------

class HGRNBitMLP(nn.Module):
    """A custom MLP block used within the transformer."""
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
    """Standard MLP for conditioning (always full precision/high fidelity)."""
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

# -----------------------------------------------------------------------------
# Embedders
# -----------------------------------------------------------------------------

class FlowTimestepEmbedder(nn.Module):
    """Embeds continuous timesteps t in [0, 1] using Fourier Features."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t[:, None].float() * self.freqs[None] * 1000.0
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            t_freq = torch.cat([t_freq, torch.zeros_like(t_freq[:, :1])], dim=-1)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb

class FramePosEmbedder(FlowTimestepEmbedder):
    """Embeds frame indices (Temporal Position) for the temporal attention stream."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size, frequency_embedding_size)

    def forward(self, f_indices):
        return super().forward(f_indices)

class TextEmbedder(nn.Module):
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
    def __init__(self, input_channels, hidden_size, patch_size, optimized_bitlinear=True, full_precision=False):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(input_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b d t h w -> b (t h w) d')
        return x

class ImageConditionEmbedder(nn.Module):
    def __init__(self, input_channels, hidden_size, patch_size, dropout_prob):
        super().__init__()
        self.patch_embed = nn.Conv2d(input_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.null_embedding = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.dropout_prob = dropout_prob
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, image, train=True, force_drop_mask=None):
        x = self.patch_embed(image)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.mlp(x)
        if force_drop_mask is not None:
            mask = force_drop_mask.float().unsqueeze(-1).unsqueeze(-1)
        elif train and self.dropout_prob > 0:
            mask = (torch.rand(image.shape[0], device=image.device) < self.dropout_prob).float().unsqueeze(-1).unsqueeze(-1)
        else:
            mask = torch.zeros(image.shape[0], 1, 1, device=image.device)
        return mask * self.null_embedding + (1 - mask) * x

# -----------------------------------------------------------------------------
# Advanced Attention Mechanics (Decoupled, Grid, Resampling)
# -----------------------------------------------------------------------------

class DecoupledVideoAttention(nn.Module):
    """
    Combines Spatial Attention (Intra-Frame), Temporal Attention (Inter-Frame),
    Grid Attention, and Resampling.
    
    NOTE: Masking is applied ONLY to the resampling branch to preserve global context.
    """
    def __init__(
        self, 
        hidden_size, 
        num_heads, 
        use_temporal=True, 
        use_grid=True, 
        use_resampling=True, 
        optimized_bitlinear=True, 
        full_precision=False, 
        **kwargs
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_grid = use_grid
        self.use_resampling = use_resampling

        # 1. Base Spatial Attention (or Global if use_temporal=False)
        self.spatial_attn = HGRNBitAttention(
            hidden_size=hidden_size, num_heads=num_heads, 
            optimized_bitlinear=optimized_bitlinear, full_precision=full_precision, **kwargs
        )

        # 2. Temporal Attention
        if self.use_temporal:
            self.temporal_attn = HGRNBitAttention(
                hidden_size=hidden_size, num_heads=num_heads, 
                optimized_bitlinear=optimized_bitlinear, full_precision=full_precision, **kwargs
            )
            self.frame_pos_embedder = FramePosEmbedder(hidden_size)

        # 3. Grid Attention (Axis-based decomposition)
        if self.use_grid:
            self.grid_attn = HGRNBitAttention(
                hidden_size=hidden_size, num_heads=num_heads, 
                optimized_bitlinear=optimized_bitlinear, full_precision=full_precision, **kwargs
            )

        # 4. Resampling Attention (Refinement)
        if self.use_resampling:
            self.resampling_attn = HGRNBitAttention(
                hidden_size=hidden_size, num_heads=num_heads, 
                optimized_bitlinear=optimized_bitlinear, full_precision=full_precision, **kwargs
            )

    def forward(self, x, num_frames: int, mask: Optional[torch.Tensor] = None):
        """
        x: [B, L, D] where L = T * H * W
        mask: [B, L, 1] Optional weighting for resampling.
        """
        B, L, D = x.shape
        T = num_frames
        
        # Determine spatial dimensions
        HW = L // T
        # Safety check, only run if cleanly divisible
        if L % T != 0:
             # Fallback for non-standard shapes
             out_spatial, _, _ = self.spatial_attn(x)
             return out_spatial

        # --- 1. Base Pass (Spatial or Global) ---
        if self.use_temporal:
            # Reshape to run attention ONLY within each frame independently
            # [B, T*HW, D] -> [B*T, HW, D]
            x_spatial = rearrange(x, 'b (t hw) d -> (b t) hw d', t=T, hw=HW)
            out_spatial, _, _ = self.spatial_attn(x_spatial)
            out_spatial = rearrange(out_spatial, '(b t) hw d -> b (t hw) d', t=T)
        else:
            # Run on the whole flattened sequence (Legacy Mode)
            out_spatial, _, _ = self.spatial_attn(x)

        # --- 2. Temporal Pass (Decoupled) ---
        out_temporal = 0
        if self.use_temporal:
            # Reshape to run attention ONLY across time for each pixel independently
            # [B, T*HW, D] -> [B*HW, T, D]
            x_temporal = rearrange(x, 'b (t hw) d -> (b hw) t d', t=T, hw=HW)
            
            # Add Frame Position Embeddings
            frame_indices = torch.arange(T, device=x.device, dtype=torch.float32)
            # Normalize indices to 0-1 range for consistency
            frame_indices = frame_indices / max(T, 1.0)
            t_pos = self.frame_pos_embedder(frame_indices) # [T, D]
            x_temporal = x_temporal + t_pos.unsqueeze(0)

            t_out, _, _ = self.temporal_attn(x_temporal)
            out_temporal = rearrange(t_out, '(b hw) t d -> b (t hw) d', hw=HW)

        # --- 3. Grid Pass ---
        out_grid = 0
        if self.use_grid and self.use_temporal:
            # Apply attention on factorized spatial axes (Width vs Height)
            H = int(math.sqrt(HW))
            if H * H == HW:
                W = H
                # Column Attention: [B*T, H, W, D] -> [B*T*W, H, D]
                x_grid_col = rearrange(x, 'b (t h w) d -> (b t w) h d', t=T, h=H, w=W)
                g_col, _, _ = self.grid_attn(x_grid_col)
                g_col = rearrange(g_col, '(b t w) h d -> b (t h w) d', t=T, w=W)
                
                # Row Attention: [B*T, H, W, D] -> [B*T*H, W, D]
                x_grid_row = rearrange(x, 'b (t h w) d -> (b t h) w d', t=T, h=H, w=W)
                g_row, _, _ = self.grid_attn(x_grid_row)
                g_row = rearrange(g_row, '(b t h) w d -> b (t h w) d', t=T, h=H)
                
                out_grid = g_col + g_row

        # Combine Signals (Additive)
        final_out = out_spatial + out_temporal + out_grid

        # --- 4. Resampling (Refinement) ---
        # Applied ONLY to the specific branch input using the mask
        if self.use_resampling:
            if mask is not None:
                res_in = x * mask
                res_out, _, _ = self.resampling_attn(res_in)
            else:
                res_out, _, _ = self.resampling_attn(x)
            
            final_out = final_out + res_out

        return final_out

# -----------------------------------------------------------------------------
# Transformer Blocks
# -----------------------------------------------------------------------------

class DualCrossAttention(nn.Module):
    """DoRA-inspired DUAL cross-attention (Text condition)."""
    def __init__(self, dim, num_heads, head_dim=None, rank=16, optimized_bitlinear=True, full_precision=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.scale = self.head_dim ** -0.5
        LinearCls = nn.Linear if full_precision else (FusedBitLinear if optimized_bitlinear else StandardBitLinear)

        self.to_q_x = LinearCls(dim, dim, bias=False)
        self.to_kv_c = LinearCls(dim, dim * 2, bias=False)
        self.magnitude_x = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.lora_A_x = nn.Linear(dim, rank, bias=False)
        self.lora_B_x = nn.Linear(rank, dim, bias=False)
        self.proj_x = LinearCls(dim, dim, bias=False)

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
        x_out = x + self._attention_pass(x, c, self.to_q_x, self.to_kv_c, self.magnitude_x, self.lora_A_x, self.lora_B_x, self.proj_x)
        c_out = c + self._attention_pass(c, x, self.to_q_c, self.to_kv_x, self.magnitude_c, self.lora_A_c, self.lora_B_c, self.proj_c)
        return x_out, c_out

class FullPrecisionAdaLNConditioning(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, eps=1e-6, hidden_ratio=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size, bias=False)
        self.mlp = FullPrecisionMLP(hidden_size=hidden_size, hidden_ratio=hidden_ratio, hidden_act='swish')
        self.output_proj = nn.Linear(hidden_size, output_dim, bias=True)
        self.norm = RMSNorm(output_dim, eps=eps)
        self.out_proj = nn.Linear(output_dim, output_dim, bias=True)

    def forward(self, condition):
        x = self.input_proj(condition)
        x = self.mlp(x)
        x = self.output_proj(x)
        x = self.norm(x)
        return self.out_proj(x)

class DualStreamBlock(nn.Module):
    """
    Advanced Block supporting Spatial/Temporal Decoupling, Grid, and Resampling.
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        use_rope: bool = False, 
        use_ternary_rope: bool = False, 
        optimized_bitlinear: bool = True, 
        full_precision: bool = False,
        use_temporal: bool = True,
        use_grid: bool = True,
        use_resampling: bool = True
    ):
        super().__init__()
        self.use_resampling = use_resampling
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # REPLACEMENT: Use DecoupledVideoAttention instead of standard HGRN for 'x'
        self.attn_x = DecoupledVideoAttention(
            hidden_size=hidden_size, num_heads=num_heads, 
            use_temporal=use_temporal, use_grid=use_grid, use_resampling=use_resampling,
            rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, 
            optimized_bitlinear=optimized_bitlinear, full_precision=full_precision
        )

        # Text stream (c) remains standard
        self.attn_c = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, expand_ratio=1, use_short_conv=False, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.dual_cross_attn = DualCrossAttention(hidden_size, num_heads, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.mlp = HGRNBitMLP(hidden_size, hidden_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.mlp_c = HGRNBitMLP(hidden_size, hidden_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.adaLN_modulation = FullPrecisionAdaLNConditioning(hidden_size, hidden_size, 12 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio)

        # Resampling Mask Generator (from TextureDiT)
        if self.use_resampling:
            LinearCls = nn.Linear if full_precision else (FusedBitLinear if optimized_bitlinear else StandardBitLinear)
            self.mask_gen = nn.Sequential(
                LayerNorm(hidden_size),
                LinearCls(hidden_size, hidden_size),
                nn.ReLU(),
                LinearCls(hidden_size, 1),
                nn.Sigmoid()
            )

    def forward(self, x, c, t, num_frames: int):
        modulated_t = ACT2FN['silu'](t)
        params = self.adaLN_modulation(modulated_t)
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x, \
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = params.chunk(12, dim=1)

        # --- Stream X (Video) ---
        modulated_x = modulate(self.norm1(x), shift_msa_x, scale_msa_x)
        
        # FIX: Generate Mask, but pass it separately to attention
        mask = None
        if self.use_resampling:
            mask = self.mask_gen(modulated_x) # [B, L, 1]

        # Advanced Attention: Pass mask to control resampling branch
        attn_x = self.attn_x(modulated_x, num_frames=num_frames, mask=mask)
        x = x + gate_msa_x.unsqueeze(1) * attn_x

        modulated_x = modulate(self.norm2(x), shift_mlp_x, scale_mlp_x)
        mlp_x = self.mlp(modulated_x)
        x = x + gate_mlp_x.unsqueeze(1) * mlp_x

        # --- Stream C (Text) ---
        modulated_c = modulate(self.norm3(c), shift_msa_c, scale_msa_c)
        attn_c, _, _ = self.attn_c(modulated_c)
        c = c + gate_msa_c.unsqueeze(1) * attn_c

        modulated_c = modulate(self.norm4(c), shift_mlp_c, scale_mlp_c)
        mlp_c = self.mlp_c(modulated_c)
        c = c + gate_mlp_c.unsqueeze(1) * mlp_c

        # --- Cross ---
        x, c = self.dual_cross_attn(x, c)
        return x, c

class SingleStreamBlock(nn.Module):
    """
    Single stream block. Uses standard attention for simplicity in deep layers.
    """
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
    def __init__(self, hidden_size, output_dim, mlp_ratio=4.0, optimized_bitlinear=True, full_precision=False):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        LinearCls = nn.Linear if full_precision else (FusedBitLinear if optimized_bitlinear else StandardBitLinear)
        self.linear = LinearCls(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = FullPrecisionAdaLNConditioning(hidden_size, hidden_size, 2 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio)

    def forward(self, x, c):
        modulated_c = ACT2FN['silu'](c)
        shift, scale = self.adaLN_modulation(modulated_c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# -----------------------------------------------------------------------------
# Main VideoDiT Class
# -----------------------------------------------------------------------------

class VideoDiT(nn.Module):
    """
    Ternary Flow Matching Transformer for Text-to-Video.
    Supports advanced Decoupled Spatial/Temporal/Grid Attention via feature flags.
    """
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (16, 64, 64),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 4,
        vocab_size: int = 49408,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.1,
        num_dual_stream_blocks: int = 14,
        use_rope: bool = False,
        use_ternary_rope: bool = False,
        first_frame_condition: bool = False, 
        optimized_bitlinear: bool = True,
        full_precision: bool = False,
        # --- NEW ARGS ---
        use_temporal: bool = False,
        use_grid: bool = False,
        use_resampling: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size 
        self.patch_size = patch_size
        self.first_frame_condition = first_frame_condition
        
        # Advanced Feature Flags
        self.use_temporal = use_temporal
        self.use_grid = use_grid
        self.use_resampling = use_resampling

        self.t_patches = input_size[0] // patch_size[0]
        self.h_patches = input_size[1] // patch_size[1]
        self.w_patches = input_size[2] // patch_size[2]
        self.num_patches = self.t_patches * self.h_patches * self.w_patches
        self.patch_dim = (patch_size[0] * patch_size[1] * patch_size[2]) * self.out_channels

        # Embedders
        self.x_embedder = VideoPatchEmbedder(in_channels, hidden_size, patch_size, optimized_bitlinear, full_precision)
        self.t_embedder = FlowTimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(vocab_size, hidden_size, dropout_prob)
        if first_frame_condition:
            self.img_embedder = ImageConditionEmbedder(in_channels, hidden_size, patch_size[1], dropout_prob)

        logger.info(f"Initialized VideoDiT (Temporal={use_temporal}, Grid={use_grid}, Resample={use_resampling})")

        # Blocks
        self.dual_stream_blocks = nn.ModuleList([
            DualStreamBlock(
                hidden_size, num_heads, mlp_ratio, use_rope, use_ternary_rope, 
                optimized_bitlinear, full_precision,
                use_temporal=use_temporal, use_grid=use_grid, use_resampling=use_resampling
            )
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
        
        # Zero-out for Flow Matching
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
        b, l, d = x.shape
        c_out = self.out_channels
        pt, ph, pw = self.patch_size
        nt, nh, nw = self.t_patches, self.h_patches, self.w_patches
        x = x.reshape(b, nt, nh, nw, pt, ph, pw, c_out)
        x = torch.einsum('bthwzyxc->bctzhwyx', x)
        x = x.reshape(b, c_out, nt * pt, nh * ph, nw * pw)
        return x

    def forward(self, x, t, y, first_frame=None):
        num_x_tokens = self.num_patches
        x_tokens = self.x_embedder(x)
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y["input_ids"], y["attention_mask"], train=self.training)

        if self.first_frame_condition:
            assert first_frame is not None
            img_tokens = self.img_embedder(first_frame, train=self.training)
            y_tokens = torch.cat([y_emb, img_tokens], dim=1)
        else:
            y_tokens = y_emb

        # Dual Stream (Uses new Toggles)
        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb, num_frames=self.t_patches)

        # Single Stream (Standard)
        combined_tokens = torch.cat([x_tokens, y_tokens], dim=1)
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)

        processed_x_tokens = combined_tokens[:, :num_x_tokens]
        output = self.final_layer(processed_x_tokens, t_emb)
        return self.unpatchify(output)

    def forward_with_cfg(self, x, t, y, cfg_scale_text, first_frame=None, cfg_scale_img=1.0):
        """CFG Forward Pass for Flow Matching."""
        half = x.shape[0] // 2
        x_tokens = self.x_embedder(x)
        x_tokens = torch.cat([x_tokens[:half]] * 4, dim=0) if self.first_frame_condition else torch.cat([x_tokens[:half]] * 2, dim=0)
        t_in = torch.cat([t[:half]] * 4, dim=0) if self.first_frame_condition else torch.cat([t[:half]] * 2, dim=0)
        t_emb = self.t_embedder(t_in)

        y_ids = y["input_ids"][:half]
        y_mask = y["attention_mask"][:half]
        
        if self.first_frame_condition:
            text_drop_mask = torch.tensor([1, 0, 1, 0], device=x.device).repeat_interleave(half)
            y_emb = self.y_embedder(y_ids.repeat(4, 1), y_mask.repeat(4, 1), force_drop_ids=text_drop_mask)
            img_in = first_frame[:half]
            img_drop_mask = torch.tensor([1, 1, 0, 0], device=x.device).repeat_interleave(half)
            img_tokens = self.img_embedder(img_in.repeat(4, 1, 1, 1), force_drop_mask=img_drop_mask)
            y_tokens = torch.cat([y_emb, img_tokens], dim=1)
        else:
            text_drop_mask = torch.tensor([1, 0], device=x.device).repeat_interleave(half)
            y_emb = self.y_embedder(y_ids.repeat(2, 1), y_mask.repeat(2, 1), force_drop_ids=text_drop_mask)
            y_tokens = y_emb

        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb, num_frames=self.t_patches)
        
        combined_tokens = torch.cat([x_tokens, y_tokens], dim=1)
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)
            
        processed_x_tokens = combined_tokens[:, :self.num_patches]
        model_out = self.final_layer(processed_x_tokens, t_emb)
        model_out = self.unpatchify(model_out)

        if self.first_frame_condition:
            v_uncond, v_text, v_img, _ = torch.chunk(model_out, 4, dim=0)
            v_pred = v_uncond + cfg_scale_text * (v_text - v_uncond) + cfg_scale_img * (v_img - v_uncond)
        else:
            v_uncond, v_text = torch.chunk(model_out, 2, dim=0)
            v_pred = v_uncond + cfg_scale_text * (v_text - v_uncond)

        return v_pred

    @torch.no_grad()
    def sample(self, z, y, steps=50, cfg_scale=7.0, first_frame=None):
        b = z.shape[0]
        dt = 1.0 / steps
        current_x = z
        for i in range(steps):
            t_curr = i / steps
            t_tensor = torch.full((b * 2,), t_curr, device=z.device, dtype=torch.float32)
            z_in = torch.cat([current_x, current_x], dim=0)
            y_in = {k: torch.cat([v, v], dim=0) for k, v in y.items()}
            img_in = torch.cat([first_frame, first_frame], dim=0) if first_frame is not None else None
            v_pred = self.forward_with_cfg(z_in, t_tensor, y_in, cfg_scale, first_frame=img_in)
            current_x = current_x + v_pred * dt
        return current_x

# -----------------------------------------------------------------------------
# Loss & Factories
# -----------------------------------------------------------------------------

def flow_matching_loss(model, x_1, y, first_frame=None):
    b, c, t, h, w = x_1.shape
    device = x_1.device
    x_0 = torch.randn_like(x_1)
    t_step = torch.rand(b, device=device)
    t_expand = t_step.view(b, 1, 1, 1, 1)
    x_t = t_expand * x_1 + (1 - t_expand) * x_0
    v_target = x_1 - x_0
    v_pred = model(x_t, t_step, y, first_frame=first_frame)
    loss = F.mse_loss(v_pred, v_target)
    return loss

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