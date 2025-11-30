# mesh_dit.py

# -*- coding: utf-8 -*-
# Adapted for 3D Mesh Latent Generation.
# This version implements a dual-stream architecture inspired by MMDiT.
# It processes shape and conditioning latents in parallel before fusion.
# UPDATED: Replaced the cross-attention mechanism with a bidirectional,
# DoRA-powered DualCrossAttention module for more efficient and powerful fusion.

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.utils import logging
# The HGRNBitAttention class handles the RoPE implementation internally.
from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.modules import RMSNorm, LayerNorm
from mmfreelm.ops.bitnet import BitLinear as StandardBitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear as FusedBitLinear

logger = logging.get_logger(__name__)


def modulate(x, shift, scale):
    """
    Helper function to apply adaptive layer normalization.
    """
    # The unsqueeze is for sequence broadcasting
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

        # Use nn.Linear for full precision
        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        z = self.down_proj(self.act_fn(gate) * y)
        return z


#################################################################################
#               Embedding Layers for Timesteps, Text, and Image                 #
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
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class TextEmbedder(nn.Module):
    """
    Embeds tokenized text into vector representations using pooling.
    Outputs a sequence of length 1.
    """
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
        
        if train and use_dropout and torch.rand(1).item() < 0.01: 
            print(f"DEBUG: Conditioning Dropout is ACTIVE. Prob: {self.dropout_prob}")
            
        if (train and use_dropout) or (force_drop_ids is not None):
            input_ids = self.token_drop(input_ids, force_drop_ids)

        embeddings = self.embedding(input_ids)
        attention_mask = attention_mask.unsqueeze(-1)
        pooled_embeddings = (embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        return self.mlp(pooled_embeddings).unsqueeze(1)


class ImageLatentEmbedder(nn.Module):
    """
    Embeds 4D image latents [N, C, H, W] into vector representations.
    Outputs a sequence of length 1.
    """
    def __init__(self, input_dim, hidden_size, dropout_prob):
        super().__init__()
        self.null_embedding = nn.Parameter(torch.randn(1, 1, hidden_size)) # Shape [1, 1, D]
        self.dropout_prob = dropout_prob
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, image_latent, train=True, force_drop_mask=None):
        flattened_latent = torch.flatten(image_latent, start_dim=1)
        projected_latent = self.mlp(flattened_latent).unsqueeze(1)

        if force_drop_mask is not None:
            mask = force_drop_mask.float().unsqueeze(-1).unsqueeze(-1)
        elif train and self.dropout_prob > 0:
            mask = (torch.rand(image_latent.shape[0], device=image_latent.device) < self.dropout_prob).float().unsqueeze(-1).unsqueeze(-1)
        else:
            mask = torch.zeros(image_latent.shape[0], 1, 1, device=image_latent.device)
        
        return mask * self.null_embedding + (1 - mask) * projected_latent


#################################################################################
#               Core MMDiT-style Model Components (UPDATED)                     #
#################################################################################

class DualCrossAttention(nn.Module):
    """
    DoRA-inspired DUAL cross-attention for better fusion of shape and conditioning.
    This implementation is bidirectional, with DoRA applied to each attention pass.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        rank: int = 16,
        optimized_bitlinear: bool = False,
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

        # --- Components for Attention from X to C ---
        self.to_q_x = LinearCls(dim, dim, bias=False)
        self.to_kv_c = LinearCls(dim, dim * 2, bias=False)
        self.magnitude_x = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.lora_A_x = nn.Linear(dim, rank, bias=False)
        self.lora_B_x = nn.Linear(rank, dim, bias=False)
        self.proj_x = LinearCls(dim, dim, bias=False)

        # --- Components for Attention from C to X ---
        self.to_q_c = LinearCls(dim, dim, bias=False)
        self.to_kv_x = LinearCls(dim, dim * 2, bias=False)
        self.magnitude_c = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.lora_A_c = nn.Linear(dim, rank, bias=False)
        self.lora_B_c = nn.Linear(rank, dim, bias=False)
        self.proj_c = LinearCls(dim, dim, bias=False)

    def _attention_pass(self, query, context, to_q, to_kv, magnitude, lora_A, lora_B, proj):
        B, N_q, C = query.shape
        _, N_c, _ = context.shape

        # Project query and apply DoRA adjustment
        q_proj = to_q(query)
        delta = lora_B(lora_A(query))
        
        # Reshape for multi-head attention and apply magnitude scaling
        q_reshaped = q_proj.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        delta_reshaped = delta.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q_final = q_reshaped + magnitude * delta_reshaped

        # Project key and value from context
        k, v = to_kv(context).chunk(2, dim=-1)
        k = k.reshape(B, N_c, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N_c, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q_final, k, v)
        
        # Reshape and project output
        out = attn.transpose(1, 2).reshape(B, N_q, C)
        return proj(out)

    def forward(self, x, c):
        # Bidirectional flow:
        # 1. Shape stream (x) attends to conditioning stream (c)
        x_out = x + self._attention_pass(
            query=x, context=c, to_q=self.to_q_x, to_kv=self.to_kv_c,
            magnitude=self.magnitude_x, lora_A=self.lora_A_x, lora_B=self.lora_B_x,
            proj=self.proj_x
        )

        # 2. Conditioning stream (c) attends to shape stream (x)
        c_out = c + self._attention_pass(
            query=c, context=x, to_q=self.to_q_c, to_kv=self.to_kv_x,
            magnitude=self.magnitude_c, lora_A=self.lora_A_c, lora_B=self.lora_B_c,
            proj=self.proj_c
        )

        return x_out, c_out

class FullPrecisionAdaLNConditioning(nn.Module):
    """
    Generates adaptive layer norm modulation parameters from a conditioning signal
    using full-precision layers.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        eps: float = 1e-6,
        hidden_ratio: Optional[int] = None
    ):
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
    """
    A dual-stream block with separate attention for shape and conditioning,
    and bidirectional DoRA cross-attention for fusion.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, use_rope: bool = False, use_ternary_rope: bool = False, optimized_bitlinear: bool = True, full_precision: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn_x = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, expand_ratio=1, use_short_conv=False, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.attn_c = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, expand_ratio=1, use_short_conv=False, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.dual_cross_attn = DualCrossAttention(hidden_size, num_heads, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.mlp = HGRNBitMLP(hidden_size, hidden_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.mlp_c = HGRNBitMLP(hidden_size, hidden_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.adaLN_modulation = FullPrecisionAdaLNConditioning(hidden_size, hidden_size, 12 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio)

    def forward(self, x, c, t):
        modulated_t = ACT2FN['silu'](t)
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x, shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.adaLN_modulation(modulated_t).chunk(12, dim=1)

        # Shape stream
        modulated_x = modulate(self.norm1(x), shift_msa_x, scale_msa_x)
        attn_x, _, _ = self.attn_x(modulated_x)
        x = x + gate_msa_x.unsqueeze(1) * attn_x

        modulated_x = modulate(self.norm2(x), shift_mlp_x, scale_mlp_x)
        mlp_x = self.mlp(modulated_x)
        x = x + gate_mlp_x.unsqueeze(1) * mlp_x

        # Conditioning stream
        modulated_c = modulate(self.norm3(c), shift_msa_c, scale_msa_c)
        attn_c, _, _ = self.attn_c(modulated_c)
        c = c + gate_msa_c.unsqueeze(1) * attn_c

        modulated_c = modulate(self.norm4(c), shift_mlp_c, scale_mlp_c)
        mlp_c = self.mlp_c(modulated_c)
        c = c + gate_mlp_c.unsqueeze(1) * mlp_c

        # Dual cross-attention fusion
        x, c = self.dual_cross_attn(x, c)

        return x, c


class SingleStreamBlock(nn.Module):
    """
    A single-stream block for combined tokens.
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
    """
    The final layer of the MeshDiT.
    """
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
#                          MeshDiT Main Class                                   #
#################################################################################

class MeshDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone for 3D Mesh Generation.
    This version uses a dual-stream architecture with DoRA-powered attention.
    """
    def __init__(
        self,
        input_tokens: int = 2048,
        input_dim: int = 64,
        vocab_size: int = 49408,
        image_latent_channels: int = 4,
        image_latent_height: int = 64,
        image_latent_width: int = 64,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.1,
        num_dual_stream_blocks: int = 14,
        use_rope: bool = False,
        use_ternary_rope: bool = False,
        image_condition: bool = False,
        optimized_bitlinear: bool = True,
        full_precision: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.num_heads = num_heads
        self.image_condition = image_condition

        if full_precision:
            LinearCls = nn.Linear
        else:
            LinearCls = FusedBitLinear if optimized_bitlinear else StandardBitLinear

        self.x_embedder = LinearCls(input_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(vocab_size, hidden_size, dropout_prob)
        image_latent_dim = image_latent_channels * image_latent_height * image_latent_width
        self.image_embedder = ImageLatentEmbedder(image_latent_dim, hidden_size, dropout_prob)

        self.pos_embed = None
        logger.info("Absolute positional embeddings are disabled for this model.")

        self.dual_stream_blocks = nn.ModuleList([
            DualStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_rope=use_rope, use_ternary_rope=use_ternary_rope, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
            for _ in range(num_dual_stream_blocks)
        ])
        num_single_stream_blocks = depth - num_dual_stream_blocks
        self.single_stream_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_rope=use_rope, use_ternary_rope=use_ternary_rope, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
            for _ in range(num_single_stream_blocks)
        ])

        self.final_layer = FinalLayer(hidden_size, self.output_dim, mlp_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, StandardBitLinear, FusedBitLinear)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
        
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

    def forward(self, x, t, y):
        num_x_tokens = x.shape[1]

        x_tokens = self.x_embedder(x)
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y["input_ids"], y["attention_mask"], train=self.training)
        
        if self.image_condition:
            img_emb = self.image_embedder(y["image_latent"], train=self.training)
        else:
            batch_size = x_tokens.shape[0]
            img_emb = self.image_embedder.null_embedding.expand(batch_size, -1, -1)

        y_tokens = torch.cat([y_emb, img_emb], dim=1)

        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb)

        combined_tokens = torch.cat([x_tokens, y_tokens], dim=1)
        
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)
            
        processed_x_tokens = combined_tokens[:, :num_x_tokens]
        output = self.final_layer(processed_x_tokens, t_emb)

        assert output.shape == x.shape, f"Final shape mismatch! Output: {output.shape}, Input: {x.shape}"
        return output

    def forward_with_cfg(self, x, t, y, cfg_scale_text, cfg_scale_image):
        num_x_tokens = x.shape[1]
        half = x.shape[0] // 2
        
        x_tokens = self.x_embedder(x)
        x_tokens = torch.cat([x_tokens[:half]] * 4, dim=0)
        t_emb = self.t_embedder(torch.cat([t[:half]] * 4, dim=0))

        y_ids = y["input_ids"][:half]
        y_mask = y["attention_mask"][:half]

        text_drop_mask = torch.tensor([1, 1, 0, 0], device=x.device).repeat_interleave(half)
        y_emb = self.y_embedder(y_ids.repeat(4, 1), y_mask.repeat(4, 1), force_drop_ids=text_drop_mask)
        
        if self.image_condition:
            y_img = y["image_latent"][:half]
            img_drop_mask = torch.tensor([1, 0, 1, 0], device=x.device).repeat_interleave(half)
            img_emb = self.image_embedder(y_img.repeat(4, 1, 1, 1), force_drop_mask=img_drop_mask)
        else:
            cfg_batch_size = x_tokens.shape[0]
            img_emb = self.image_embedder.null_embedding.expand(cfg_batch_size, -1, -1)
        
        y_tokens = torch.cat([y_emb, img_emb], dim=1)

        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb)
        
        combined_tokens = torch.cat([x_tokens, y_tokens], dim=1)
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)
            
        processed_x_tokens = combined_tokens[:, :num_x_tokens]
        model_out = self.final_layer(processed_x_tokens, t_emb)

        e_uncond, e_img, e_text, e_full = torch.chunk(model_out, 4, dim=0)
        
        if self.image_condition:
            noise_pred = e_uncond + \
                         cfg_scale_text * (e_text - e_uncond) + \
                         cfg_scale_image * (e_img - e_uncond)
        else:
            noise_pred = e_uncond + cfg_scale_text * (e_text - e_uncond)
        
        return noise_pred


#################################################################################
#                                  MeshDiT Configs                              #
#################################################################################

def MeshDiT_XL(**kwargs):
    return MeshDiT(depth=28, hidden_size=1152, num_heads=16, num_dual_stream_blocks=14, **kwargs)

def MeshDiT_L(**kwargs):
    return MeshDiT(depth=24, hidden_size=1024, num_heads=16, num_dual_stream_blocks=12, **kwargs)

def MeshDiT_B(**kwargs):
    return MeshDiT(depth=12, hidden_size=768, num_heads=12, num_dual_stream_blocks=6, **kwargs)

def MeshDiT_S(**kwargs):
    return MeshDiT(depth=12, hidden_size=384, num_heads=6, num_dual_stream_blocks=6, **kwargs)


MeshDiT_models = {
    'MeshDiT-XL': MeshDiT_XL,
    'MeshDiT-L':  MeshDiT_L,
    'MeshDiT-B':  MeshDiT_B,
    'MeshDiT-S':  MeshDiT_S,
}

