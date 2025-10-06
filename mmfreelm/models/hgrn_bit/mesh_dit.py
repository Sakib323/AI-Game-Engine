# mesh_dit.py

# -*- coding: utf-8 -*-
# Adapted for 3D Mesh Latent Generation.
# This version implements a dual-stream architecture inspired by MMDiT.
# It processes shape and conditioning latents in parallel before fusion.
# FIXED: Made token slicing dynamic to prevent shape assertion errors.
# FIXED: Removed redundant normalization layer in SingleStreamBlock.
# ADDED: image_condition flag to control image conditioning.
# FIXED: Adjusted x_embedder to handle correct input feature dimension.

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
        optimized_bitlinear: bool = False,
        full_precision: bool = False
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
        if (train and use_dropout) or (force_drop_ids is not None):
            input_ids = self.token_drop(input_ids, force_drop_ids)

        embeddings = self.embedding(input_ids)
        attention_mask = attention_mask.unsqueeze(-1)
        pooled_embeddings = (embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        pooled_embeddings = self.mlp(pooled_embeddings)
        return pooled_embeddings


class ImageLatentEmbedder(nn.Module):
    """
    Embeds image latents into vector representations using pooling.
    Outputs a sequence of length 1.
    """
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.mlp = FullPrecisionMLP(hidden_size, 4, hidden_act='swish')
        self.null_embedding = nn.Parameter(torch.zeros(1, hidden_size))

    def latent_drop(self, latents, force_drop_mask=None):
        if force_drop_mask is None:
            drop_mask = torch.rand(latents.shape[0], device=latents.device) < self.dropout_prob
        else:
            drop_mask = force_drop_mask == 1
        return torch.where(drop_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0, latents)

    def forward(self, latents, train=True, force_drop_mask=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_mask is not None):
            latents = self.latent_drop(latents, force_drop_mask)
        pooled_latents = latents.mean(dim=[2, 3])
        pooled_latents = self.mlp(pooled_latents)
        return pooled_latents


#################################################################################
#                             DoRA Cross-Attention                              #
#################################################################################

class DoRACrossAttention(nn.Module):
    """
    DoRA-inspired dual cross-attention for better fusion of shape and conditioning.
    """
    def __init__(
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        rank: int = 16,  # Low-rank for DoRA
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

        # Standard QKV
        self.qkv = LinearCls(dim, dim * 3, bias=False)

        # DoRA: Decompose into magnitude (scalar) and direction (LoRA)
        self.magnitude = nn.Parameter(torch.ones(1, num_heads, 1, 1))  # Learnable magnitude
        self.lora_A = nn.Linear(dim, rank, bias=False)  # Low-rank A
        self.lora_B = nn.Linear(rank, dim, bias=False)  # Low-rank B

        self.proj = LinearCls(dim, dim, bias=False)

    def forward(self, query, key, value, mask=None):
        B, N, C = query.shape
        qkv = self.qkv(query).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # DoRA adjustment: Direction via LoRA, scaled by magnitude
        delta = self.lora_B(self.lora_A(query))  # LoRA delta
        q = q + self.magnitude * delta.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


#################################################################################
#                                  Attention Blocks                             #
#################################################################################

class AdaLNModulation(nn.Module):
    """
    Adaptive layer norm modulation for conditioning.
    """
    def __init__(
        hidden_size: int,
        optimized_bitlinear: bool = False,
        full_precision: bool = False
    ):
        super().__init__()
        if full_precision:
            LinearCls = nn.Linear
        else:
            LinearCls = FusedBitLinear if optimized_bitlinear else StandardBitLinear
        self.mlp = nn.Sequential(
            nn.SiLU(),
            LinearCls(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.mlp(c).chunk(6, dim=1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class DualStreamBlock(nn.Module):
    """
    Dual-stream block for parallel processing of shape and conditioning.
    """
    def __init__(
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rope: bool = False,
        use_ternary_rope: bool = False,
        optimized_bitlinear: bool = False,
        full_precision: bool = False
    ):
        super().__init__()
        self.norm_x = LayerNorm(hidden_size)
        self.norm_y = LayerNorm(hidden_size)
        self.adaLN_modulation = AdaLNModulation(hidden_size, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        # Use DoRACrossAttention for dual cross-attention
        self.cross_attn_x_to_y = DoRACrossAttention(hidden_size, num_heads, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.cross_attn_y_to_x = DoRACrossAttention(hidden_size, num_heads, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.attn_x = HGRNBitAttention(
            mode='fused_recurrent',
            hidden_size=hidden_size,
            num_heads=num_heads,
            rotary_embeddings=use_rope,
            use_ternary_rope=use_ternary_rope,
            optimized_bitlinear=optimized_bitlinear,
            full_precision=full_precision
        )
        self.attn_y = HGRNBitAttention(
            mode='fused_recurrent',
            hidden_size=hidden_size,
            num_heads=num_heads,
            rotary_embeddings=use_rope,
            use_ternary_rope=use_ternary_rope,
            optimized_bitlinear=optimized_bitlinear,
            full_precision=full_precision
        )

        self.mlp_x = HGRNBitMLP(hidden_size, mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.mlp_y = HGRNBitMLP(hidden_size, mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

    def forward(self, x, y, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)

        # Normalize
        x_norm = self.norm_x(x)
        y_norm = self.norm_y(y)

        # Dual cross-attention for fusion
        x = x + gate_msa.unsqueeze(1) * self.cross_attn_x_to_y(x_norm, y_norm, y_norm)
        y = y + gate_msa.unsqueeze(1) * self.cross_attn_y_to_x(y_norm, x_norm, x_norm)

        # Self-attention
        x = x + gate_msa.unsqueeze(1) * self.attn_x(modulate(x_norm, shift_msa, scale_msa))
        y = y + gate_msa.unsqueeze(1) * self.attn_y(modulate(y_norm, shift_msa, scale_msa))

        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp_x(modulate(self.norm_x(x), shift_mlp, scale_mlp))
        y = y + gate_mlp.unsqueeze(1) * self.mlp_y(modulate(self.norm_y(y), shift_mlp, scale_mlp))

        return x, y


class SingleStreamBlock(nn.Module):
    """
    Single-stream block for fused processing after dual-stream.
    """
    def __init__(
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rope: bool = False,
        use_ternary_rope: bool = False,
        optimized_bitlinear: bool = False,
        full_precision: bool = False
    ):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.adaLN_modulation = AdaLNModulation(hidden_size, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.attn = HGRNBitAttention(
            mode='fused_recurrent',
            hidden_size=hidden_size,
            num_heads=num_heads,
            rotary_embeddings=use_rope,
            use_ternary_rope=use_ternary_rope,
            optimized_bitlinear=optimized_bitlinear,
            full_precision=full_precision
        )
        self.mlp = HGRNBitMLP(hidden_size, mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    Final layer to map back to output dimension.
    """
    def __init__(
        hidden_size: int,
        out_dim: int,
        mlp_ratio: float = 4.0,
        optimized_bitlinear: bool = False,
        full_precision: bool = False
    ):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.adaLN_modulation = AdaLNModulation(hidden_size, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        if full_precision:
            LinearCls = nn.Linear
        else:
            LinearCls = FusedBitLinear if optimized_bitlinear else StandardBitLinear
        self.linear = LinearCls(hidden_size, out_dim, bias=True)

    def forward(self, x, c):
        shift, scale, gate = self.adaLN_modulation(c)[:3]
        x = self.linear(modulate(self.norm(x), shift, scale))
        return x


#################################################################################
#                                MeshDiT Model                                  #
#################################################################################

class MeshDiT(nn.Module):
    """
    The main MeshDiT model class.
    """
    def __init__(
        input_tokens: int = 4096,  # Updated to match observed input
        depth: int = 12,
        hidden_size: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_dual_stream_blocks: Optional[int] = None,
        vocab_size: int = 32000,
        dropout_prob: float = 0.1,
        image_condition: bool = False,
        use_rope: bool = False,
        use_ternary_rope: bool = False,
        optimized_bitlinear: bool = False,
        full_precision: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_tokens = input_tokens
        self.output_dim = input_tokens * 64  # Updated to 64 channels to match VAE output
        self.image_condition = image_condition
        self.num_dual_stream_blocks = num_dual_stream_blocks or depth // 2
        num_single_stream_blocks = depth - self.num_dual_stream_blocks

        # Embedders
        self.x_embedder = nn.Linear(64, hidden_size, bias=False)  # Match input feature dim (64)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(vocab_size, hidden_size, dropout_prob)
        if image_condition:
            self.image_embedder = ImageLatentEmbedder(hidden_size, dropout_prob)
        else:
            self.image_embedder = ImageLatentEmbedder(hidden_size, 0.0)  # Null embedder

        # Dual-Stream Blocks
        self.dual_stream_blocks = nn.ModuleList([
            DualStreamBlock(
                hidden_size,
                num_heads,
                mlp_ratio,
                use_rope=use_rope,
                use_ternary_rope=use_ternary_rope,
                optimized_bitlinear=optimized_bitlinear,
                full_precision=full_precision
            ) for _ in range(self.num_dual_stream_blocks)
        ])

        # Single-Stream Blocks
        self.single_stream_blocks = nn.ModuleList([
            SingleStreamBlock(
                hidden_size,
                num_heads,
                mlp_ratio,
                use_rope=use_rope,
                use_ternary_rope=use_ternary_rope,
                optimized_bitlinear=optimized_bitlinear,
                full_precision=full_precision
            ) for _ in range(num_single_stream_blocks)
        ])
        
        self.final_layer = FinalLayer(hidden_size, 64, mlp_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)  # Output 64 channels
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initializes weights for all sub-modules."""
        def _basic_init(module):
            if isinstance(module, (nn.Linear, StandardBitLinear, FusedBitLinear)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
        
        for block in self.dual_stream_blocks:
            nn.init.constant_(block.adaLN_modulation.mlp[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation.mlp[-1].bias, 0)
        for block in self.single_stream_blocks:
            nn.init.constant_(block.adaLN_modulation.mlp[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation.mlp[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation.mlp[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation.mlp[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x, t, y):
        """
        Forward pass for the MeshDiT model.
        """
        num_x_tokens = x.shape[1]

        # 1. Embed all inputs
        x_tokens = self.x_embedder(x)  # Now handles (batch, tokens, 64) -> (batch, tokens, hidden_size)
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y["input_ids"], y["attention_mask"], train=self.training)
        # 2. Handle image embedding based on condition flag
        if self.image_condition:
            img_emb = self.image_embedder(y["image_latent"], train=self.training)
        else:
            # If conditioning is off, create a null embedding matching the batch size of x_tokens.
            batch_size = x_tokens.shape[0]
            img_emb = self.image_embedder.null_embedding.expand(batch_size, -1, -1)

        # 3. Create conditioning stream (y_tokens)
        y_tokens = torch.cat([y_emb, img_emb], dim=1)

        # 4. Process through Dual-Stream blocks
        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb)

        # 5. Concatenate for Single-Stream blocks
        combined_tokens = torch.cat([x_tokens, y_tokens], dim=1)
        
        # 6. Process through Single-Stream blocks
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)
            
        # 7. Isolate the shape tokens and process through final layer
        processed_x_tokens = combined_tokens[:, :num_x_tokens]
        output = self.final_layer(processed_x_tokens, t_emb)

        assert output.shape == (x.shape[0], x.shape[1], 64), f"Final shape mismatch! Output: {output.shape}, Expected: {(x.shape[0], x.shape[1], 64)}"
        return output

    def forward_with_cfg(self, x, t, y, cfg_scale_text, cfg_scale_image):
        """
        Forward pass with Classifier-Free Guidance.
        """
        num_x_tokens = x.shape[1]
        half = x.shape[0] // 2
        
        x_tokens = self.x_embedder(x)
        x_tokens = torch.cat([x_tokens[:half]] * 4, dim=0)
        t_emb = self.t_embedder(torch.cat([t[:half]] * 4, dim=0))

        y_ids = y["input_ids"][:half]
        y_mask = y["attention_mask"][:half]
        y_img = y["image_latent"][:half]

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
            # When image condition is off, guidance is only on text.
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