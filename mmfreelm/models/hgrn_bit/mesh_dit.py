# -*- coding: utf-8 -*-
# Adapted for 3D Mesh Latent Generation.
# This version implements a dual-stream architecture inspired by MMDiT.
# It processes shape and conditioning latents in parallel before fusion.

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
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear

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
        hidden_act: str = 'swish'
    ):
        super().__init__()

        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)

        self.gate_proj = BitLinear(self.hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = BitLinear(intermediate_size, self.hidden_size, bias=False)
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
        # Output a sequence of length 1 for concatenation into the conditioning stream
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
        # Project and reshape to [N, 1, D]
        projected_latent = self.mlp(flattened_latent).unsqueeze(1)

        if force_drop_mask is not None:
            mask = force_drop_mask.float().unsqueeze(-1).unsqueeze(-1) # Shape [N, 1, 1]
        elif train and self.dropout_prob > 0:
            mask = (torch.rand(image_latent.shape[0], device=image_latent.device) < self.dropout_prob).float().unsqueeze(-1).unsqueeze(-1)
        else:
            mask = torch.zeros(image_latent.shape[0], 1, 1, device=image_latent.device)
        
        # Use broadcasting for null embedding
        return mask * self.null_embedding + (1 - mask) * projected_latent


#################################################################################
#               Core MMDiT-style Model Components (NEW)                         #
#################################################################################

class CrossAttention(nn.Module):
    """
    A standard cross-attention layer, built with BitLinear for quantization.
    """
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_q = BitLinear(dim, num_heads * head_dim, bias=False)
        self.to_k = BitLinear(dim, num_heads * head_dim, bias=False)
        self.to_v = BitLinear(dim, num_heads * head_dim, bias=False)
        self.to_out = BitLinear(num_heads * head_dim, dim, bias=False)

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
    A Dual-Stream block inspired by MMDiT. It processes two separate streams
    (e.g., shape `x` and condition `y`) with self-attention and fuses them
    via cross-attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_rope=False, use_ternary_rope=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # --- FIXED: Changed output dimension from 10*hidden_size to 15*hidden_size ---
        # We need 15 sets of modulation parameters (shift, scale, gate for 5 operations)
        # 3 for x_attn, 3 for x_cross_attn, 3 for x_mlp -> 9 for stream x
        # 3 for y_attn, 3 for y_mlp -> 6 for stream y
        # Total needed: (3+3+3+3+3) = 15 * hidden_size
        self.adaLN_modulation = FullPrecisionAdaLNConditioning(hidden_size, hidden_size, 15 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio)
        # --- END FIX ---

        # Stream X (Shape Latents)
        self.norm_x1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_x = HGRNBitAttention(hidden_size=hidden_size, num_heads=num_heads, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, **block_kwargs)
        self.norm_x2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads, self.head_dim)
        self.norm_x3 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_x = HGRNBitMLP(hidden_size=hidden_size, intermediate_size=int(hidden_size * mlp_ratio), hidden_act='swish')

        # Stream Y (Conditioning Latents)
        self.norm_y1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_y = HGRNBitAttention(hidden_size=hidden_size, num_heads=num_heads, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, **block_kwargs)
        self.norm_y2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_y = HGRNBitMLP(hidden_size=hidden_size, intermediate_size=int(hidden_size * mlp_ratio), hidden_act='swish')

    def forward(self, x, y, c):
        # Generate modulation parameters from timestep conditioning `c`
        modulated_c = ACT2FN['silu'](c)
        (shift_x_attn, scale_x_attn, gate_x_attn,
         shift_x_cross, scale_x_cross, gate_x_cross,
         shift_x_mlp, scale_x_mlp, gate_x_mlp,
         shift_y_attn, scale_y_attn, gate_y_attn,
         shift_y_mlp, scale_y_mlp, gate_y_mlp) = self.adaLN_modulation(modulated_c).chunk(15, dim=1)

        # --- Stream X Processing ---
        # 1. Self-Attention
        attn_x_output, _, _ = self.attn_x(modulate(self.norm_x1(x), shift_x_attn, scale_x_attn))
        x = x + gate_x_attn.unsqueeze(1) * attn_x_output
        # 2. Cross-Attention (X queries Y)
        cross_attn_output = self.cross_attn(modulate(self.norm_x2(x), shift_x_cross, scale_x_cross), context=y)
        x = x + gate_x_cross.unsqueeze(1) * cross_attn_output
        # 3. MLP
        mlp_x_output = self.mlp_x(modulate(self.norm_x3(x), shift_x_mlp, scale_x_mlp))
        x = x + gate_x_mlp.unsqueeze(1) * mlp_x_output

        # --- Stream Y Processing ---
        # 1. Self-Attention
        attn_y_output, _, _ = self.attn_y(modulate(self.norm_y1(y), shift_y_attn, scale_y_attn))
        y = y + gate_y_attn.unsqueeze(1) * attn_y_output
        # 2. MLP
        mlp_y_output = self.mlp_y(modulate(self.norm_y2(y), shift_y_mlp, scale_y_mlp))
        y = y + gate_y_mlp.unsqueeze(1) * mlp_y_output

        return x, y

class SingleStreamBlock(nn.Module):
    """
    A single-stream block, formerly MeshDiTBlock. Processes concatenated tokens.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_rope=False, use_ternary_rope=False, **block_kwargs):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, **block_kwargs)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = HGRNBitMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim, hidden_act='swish')
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        self.adaLN_modulation = FullPrecisionAdaLNConditioning(hidden_size, hidden_size, 6 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio)

    def forward(self, x, c):
        modulated_c = ACT2FN['silu'](c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(modulated_c).chunk(6, dim=1)
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _, _ = self.attn(modulated_x)
        x = x + gate_msa.unsqueeze(1) * attn_output
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output = self.norm3(self.mlp(mlp_input))
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        return x


class FinalLayer(nn.Module):
    """
    The final layer of the MeshDiT.
    """
    def __init__(self, hidden_size, output_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = BitLinear(hidden_size, output_dim, bias=True)
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
    This version uses a dual-stream architecture.
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
        num_dual_stream_blocks: int = 14, # Number of initial dual-stream blocks
        use_rope: bool = False,
        use_ternary_rope: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.num_heads = num_heads
        self.num_x_tokens = input_tokens

        # Input Embedders
        self.x_embedder = BitLinear(input_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(vocab_size, hidden_size, dropout_prob)
        image_latent_dim = image_latent_channels * image_latent_height * image_latent_width
        self.image_embedder = ImageLatentEmbedder(image_latent_dim, hidden_size, dropout_prob)
        
        # Positional embeddings are disabled
        self.pos_embed = None
        logger.info("Absolute positional embeddings are disabled for this model.")
        
        # Architecture blocks
        self.dual_stream_blocks = nn.ModuleList([
            DualStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_rope=use_rope, use_ternary_rope=use_ternary_rope)
            for _ in range(num_dual_stream_blocks)
        ])
        num_single_stream_blocks = depth - num_dual_stream_blocks
        self.single_stream_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_rope=use_rope, use_ternary_rope=use_ternary_rope)
            for _ in range(num_single_stream_blocks)
        ])
        
        self.final_layer = FinalLayer(hidden_size, self.output_dim, mlp_ratio=mlp_ratio)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initializes weights for all sub-modules."""
        def _basic_init(module):
            if isinstance(module, (nn.Linear, BitLinear)):
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
        """
        Forward pass for the MeshDiT model.
        """
        # 1. Embed all inputs
        x_tokens = self.x_embedder(x) # [B, N, D]
        t_emb = self.t_embedder(t) # [B, D]
        y_emb = self.y_embedder(y["input_ids"], y["attention_mask"], train=self.training) # [B, 1, D]
        img_emb = self.image_embedder(y["image_latent"], train=self.training) # [B, 1, D]
        
        # 2. Create conditioning stream (y_tokens)
        y_tokens = torch.cat([y_emb, img_emb], dim=1) # [B, 2, D]

        # 3. Process through Dual-Stream blocks
        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb)

        # 4. Concatenate for Single-Stream blocks
        combined_tokens = torch.cat([x_tokens, y_tokens], dim=1)
        
        # 5. Process through Single-Stream blocks
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)
            
        # 6. Isolate the shape tokens and process through final layer
        processed_x_tokens = combined_tokens[:, :self.num_x_tokens]
        output = self.final_layer(processed_x_tokens, t_emb)
        return output

    def forward_with_cfg(self, x, t, y, cfg_scale_text, cfg_scale_image):
        """
        Forward pass with Classifier-Free Guidance.
        """
        half = x.shape[0] // 2
        
        # Prepare inputs for 4 CFG branches
        x_tokens = self.x_embedder(x)
        x_tokens = torch.cat([x_tokens[:half]] * 4, dim=0)
        t_emb = self.t_embedder(torch.cat([t[:half]] * 4, dim=0))

        # Prepare conditioning stream (y_tokens) for 4 CFG branches
        y_ids = y["input_ids"][:half]
        y_mask = y["attention_mask"][:half]
        y_img = y["image_latent"][:half]

        text_drop_mask = torch.tensor([1, 1, 0, 0], device=x.device).repeat_interleave(half)
        y_emb = self.y_embedder(y_ids.repeat(4, 1), y_mask.repeat(4, 1), force_drop_ids=text_drop_mask)
        
        img_drop_mask = torch.tensor([1, 0, 1, 0], device=x.device).repeat_interleave(half)
        img_emb = self.image_embedder(y_img.repeat(4, 1, 1, 1), force_drop_mask=img_drop_mask)
        
        y_tokens = torch.cat([y_emb, img_emb], dim=1) # [B*4, 2, D]

        # Dual-stream processing
        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb)
        
        # Concatenate and single-stream processing
        combined_tokens = torch.cat([x_tokens, y_tokens], dim=1)
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)
            
        # Final layer and CFG combination
        processed_x_tokens = combined_tokens[:, :self.num_x_tokens]
        model_out = self.final_layer(processed_x_tokens, t_emb)

        e_uncond, e_img, e_text, e_full = torch.chunk(model_out, 4, dim=0)
        
        # Note: The original paper uses a more complex guidance formula. This is a standard implementation.
        noise_pred = e_uncond + \
                     cfg_scale_text * (e_text - e_uncond) + \
                     cfg_scale_image * (e_img - e_uncond)
        
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
