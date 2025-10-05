# animation.py

# -*- coding: utf-8 -*-
# Adapted for 3D Animation Generation.
# This version extends the MeshDiT architecture to generate sequences of animated 3D latents.
# It treats the animation as a concatenated sequence of frame latents, conditioned on a static 3D latent and text prompt.
# The static latent is incorporated as multi-token conditioning in the y stream.
# FIXED: Adjusted embedders to handle multi-token static latents without flattening.
# ADDED: num_frames parameter to define the animation sequence length.
# ADDED: static_condition flag to control static latent conditioning (replaces image_condition).

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
        full_precision: bool = False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)

        BitLinearCls = StandardBitLinear if full_precision else FusedBitLinear

        self.gate_proj = BitLinearCls(self.hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = BitLinearCls(intermediate_size, self.hidden_size, bias=False)
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
#               Embedding Layers for Timesteps, Text, and Static Latent         #
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
        # Output a sequence of length 1 for concatenation into the conditioning stream
        return self.mlp(pooled_embeddings).unsqueeze(1)


class StaticLatentEmbedder(nn.Module):
    """
    Embeds static 3D latents [N, L, C] into vector representations.
    Projects per token, keeps multi-token sequence.
    """
    def __init__(self, input_dim, hidden_size, dropout_prob, num_tokens, full_precision: bool = False):
        super().__init__()
        BitLinearCls = StandardBitLinear if full_precision else FusedBitLinear
        self.proj = BitLinearCls(input_dim, hidden_size, bias=False)
        self.null_embedding = nn.Parameter(torch.randn(1, num_tokens, hidden_size))  # Learnable null for multi-token
        self.dropout_prob = dropout_prob

    def forward(self, static_latent, train=True, force_drop_mask=None):
        # static_latent: [N, L, input_dim]
        projected_latent = self.proj(static_latent)  # [N, L, hidden_size]

        if force_drop_mask is not None:
            mask = force_drop_mask.float().unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
        elif train and self.dropout_prob > 0:
            mask = (torch.rand(static_latent.shape[0], device=static_latent.device) < self.dropout_prob).float().unsqueeze(-1).unsqueeze(-1)
        else:
            mask = torch.zeros(static_latent.shape[0], 1, 1, device=static_latent.device)
        
        # Use broadcasting for null embedding
        return mask * self.null_embedding + (1 - mask) * projected_latent


#################################################################################
#               Core MMDiT-style Model Components (NEW)                         #
#################################################################################

class CrossAttention(nn.Module):
    """
    A standard cross-attention layer, built with BitLinear for quantization.
    """
    def __init__(self, dim, num_heads, head_dim, full_precision: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        BitLinearCls = StandardBitLinear if full_precision else FusedBitLinear

        self.to_q = BitLinearCls(dim, num_heads * head_dim, bias=False)
        self.to_k = BitLinearCls(dim, num_heads * head_dim, bias=False)
        self.to_v = BitLinearCls(dim, num_heads * head_dim, bias=False)
        self.to_out = BitLinearCls(num_heads * head_dim, dim, bias=False)

    def forward(self, x, context):
        b, n, _, h = *x.shape, self.num_heads
        
        q = self.to_q(x).view(b, n, h, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(b, context.shape[1], h, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(b, context.shape[1], h, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


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
    A dual-stream block with separate attention for shape and conditioning.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, use_rope: bool = False, use_ternary_rope: bool = False, full_precision: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.attn_x = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, expand_ratio=1, use_short_conv=False, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, full_precision=full_precision)
        self.attn_c = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, expand_ratio=1, use_short_conv=False, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, full_precision=full_precision)
        
        self.cross_attn = CrossAttention(hidden_size, num_heads, hidden_size // num_heads, full_precision=full_precision)
        
        self.mlp = HGRNBitMLP(hidden_size, hidden_ratio=mlp_ratio, full_precision=full_precision)
        self.mlp_c = HGRNBitMLP(hidden_size, hidden_ratio=mlp_ratio, full_precision=full_precision)
        
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
        
        # Cross-attention fusion
        x = x + self.cross_attn(x, c)
        
        return x, c


class SingleStreamBlock(nn.Module):
    """
    A single-stream block for combined tokens.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, use_rope: bool = False, use_ternary_rope: bool = False, full_precision: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.attn = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, expand_ratio=1, use_short_conv=False, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, full_precision=full_precision)
        self.mlp = HGRNBitMLP(hidden_size, hidden_ratio=mlp_ratio, full_precision=full_precision)
        
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
    The final layer of the AnimDiT.
    """
    def __init__(self, hidden_size, output_dim, mlp_ratio=4.0, full_precision: bool = False):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        BitLinearCls = StandardBitLinear if full_precision else FusedBitLinear
        self.linear = BitLinearCls(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = FullPrecisionAdaLNConditioning(hidden_size, hidden_size, 2 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio)

    def forward(self, x, c):
        modulated_c = ACT2FN['silu'](c)
        shift, scale = self.adaLN_modulation(modulated_c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


#################################################################################
#                          AnimDiT Main Class                                   #
#################################################################################
        
class AnimDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone for 3D Animation Generation.
    This version uses a dual-stream architecture.
    """ 
    def __init__(
        self,
        input_tokens: int = 2048,  # Tokens per frame
        num_frames: int = 16,     # Number of frames in animation
        input_dim: int = 64,
        vocab_size: int = 49408,
        static_latent_tokens: int = 2048,  # Tokens in static latent (e.g., same as input_tokens)
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.1,
        num_dual_stream_blocks: int = 14,
        use_rope: bool = False,
        use_ternary_rope: bool = False,
        static_condition: bool = False, # Default to False
        full_precision: bool = False,  # New parameter
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.num_heads = num_heads
        self.static_condition = static_condition
        self.tokens_per_frame = input_tokens
        self.num_frames = num_frames
        
        # Select the appropriate BitLinear class based on full_precision
        BitLinearCls = StandardBitLinear if full_precision else FusedBitLinear
        
        # Input Embedders
        self.x_embedder = BitLinearCls(input_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(vocab_size, hidden_size, dropout_prob)
        self.static_embedder = StaticLatentEmbedder(input_dim, hidden_size, dropout_prob, static_latent_tokens, full_precision=full_precision)
        
        # Positional embeddings are disabled
        self.pos_embed = None
        logger.info("Absolute positional embeddings are disabled for this model.")
        
        # Architecture blocks
        self.dual_stream_blocks = nn.ModuleList([
            DualStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_rope=use_rope, use_ternary_rope=use_ternary_rope, full_precision=full_precision)
            for _ in range(num_dual_stream_blocks)
        ])
        num_single_stream_blocks = depth - num_dual_stream_blocks
        self.single_stream_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_rope=use_rope, use_ternary_rope=use_ternary_rope, full_precision=full_precision)
            for _ in range(num_single_stream_blocks)
        ])
        
        self.final_layer = FinalLayer(hidden_size, self.output_dim, mlp_ratio=mlp_ratio, full_precision=full_precision)
        
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
            nn.init.constant_(block.adaLN_modulation.output_proj.weight, 0)
            nn.init.constant_(block.adaLN_modulation.output_proj.bias, 0)
        for block in self.single_stream_blocks:
            nn.init.constant_(block.adaLN_modulation.output_proj.weight, 0)
            nn.init.constant_(block.adaLN_modulation.output_proj.bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation.output_proj.weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation.output_proj.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x, t, y, static_latent=None):
        """
        Forward pass for the AnimDiT model.
        x: concatenated noisy latents for all frames [B, num_frames * tokens_per_frame, input_dim]
        """
        total_seq_tokens = x.shape[1]
        assert total_seq_tokens == self.num_frames * self.tokens_per_frame, f"Input sequence length mismatch: {total_seq_tokens} vs expected {self.num_frames * self.tokens_per_frame}"

        # 1. Embed all inputs
        x_tokens = self.x_embedder(x)
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y["input_ids"], y["attention_mask"], train=self.training)
        # 2. Handle static embedding based on condition flag
        if self.static_condition:
            assert static_latent is not None, "Static latent required when static_condition=True"
            static_emb = self.static_embedder(static_latent, train=self.training)
        else:
            # Create a null embedding matching the batch size
            batch_size = x_tokens.shape[0]
            static_emb = self.static_embedder.null_embedding.expand(batch_size, -1, -1)

        # 3. Create conditioning stream (y_tokens)
        y_tokens = torch.cat([y_emb, static_emb], dim=1)

        # 4. Process through Dual-Stream blocks
        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb)

        # 5. Concatenate for Single-Stream blocks
        combined_tokens = torch.cat([x_tokens, y_tokens], dim=1)
        
        # 6. Process through Single-Stream blocks
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)
            
        # 7. Isolate the animation sequence tokens and process through final layer
        processed_x_tokens = combined_tokens[:, :total_seq_tokens]
        output = self.final_layer(processed_x_tokens, t_emb)

        assert output.shape == x.shape, f"Final shape mismatch! Output: {output.shape}, Input: {x.shape}"
        return output

    def forward_with_cfg(self, x, t, y, cfg_scale_text, cfg_scale_static, static_latent=None):
        """
        Forward pass with Classifier-Free Guidance.
        """
        total_seq_tokens = x.shape[1]
        half = x.shape[0] // 2
        
        x_tokens = self.x_embedder(x)
        x_tokens = torch.cat([x_tokens[:half]] * 4, dim=0)
        t_emb = self.t_embedder(torch.cat([t[:half]] * 4, dim=0))

        y_ids = y["input_ids"][:half]
        y_mask = y["attention_mask"][:half]

        text_drop_mask = torch.tensor([1, 1, 0, 0], device=x.device).repeat_interleave(half)
        y_emb = self.y_embedder(y_ids.repeat(4, 1), y_mask.repeat(4, 1), force_drop_ids=text_drop_mask)
        
        if self.static_condition:
            assert static_latent is not None, "Static latent required when static_condition=True"
            static_latent_input = static_latent[:half]
            static_drop_mask = torch.tensor([1, 0, 1, 0], device=x.device).repeat_interleave(half)
            static_emb = self.static_embedder(static_latent_input.repeat(4, 1, 1), force_drop_mask=static_drop_mask)
        else:
            cfg_batch_size = x_tokens.shape[0]
            static_emb = self.static_embedder.null_embedding.expand(cfg_batch_size, -1, -1)
        
        y_tokens = torch.cat([y_emb, static_emb], dim=1)

        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb)
        
        combined_tokens = torch.cat([x_tokens, y_tokens], dim=1)
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)
            
        processed_x_tokens = combined_tokens[:, :total_seq_tokens]
        model_out = self.final_layer(processed_x_tokens, t_emb)

        e_uncond, e_static, e_text, e_full = torch.chunk(model_out, 4, dim=0)
        
        if self.static_condition:
            noise_pred = e_uncond + \
                         cfg_scale_text * (e_text - e_uncond) + \
                         cfg_scale_static * (e_static - e_uncond)
        else:
            # When static condition is off, guidance is only on text.
            noise_pred = e_uncond + cfg_scale_text * (e_text - e_uncond)
        
        return noise_pred


#################################################################################
#                                  AnimDiT Configs                              #
#################################################################################

def AnimDiT_XL(**kwargs):
    return AnimDiT(depth=28, hidden_size=1152, num_heads=16, num_dual_stream_blocks=14, **kwargs)

def AnimDiT_L(**kwargs):
    return AnimDiT(depth=24, hidden_size=1024, num_heads=16, num_dual_stream_blocks=12, **kwargs)

def AnimDiT_B(**kwargs):
    return AnimDiT(depth=12, hidden_size=768, num_heads=12, num_dual_stream_blocks=6, **kwargs)

def AnimDiT_S(**kwargs):
    return AnimDiT(depth=12, hidden_size=384, num_heads=6, num_dual_stream_blocks=6, **kwargs)


AnimDiT_models = {
    'AnimDiT-XL': AnimDiT_XL,
    'AnimDiT-L':  AnimDiT_L,
    'AnimDiT-B':  AnimDiT_B,
    'AnimDiT-S':  AnimDiT_S,
}