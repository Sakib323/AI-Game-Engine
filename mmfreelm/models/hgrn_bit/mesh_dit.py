# -*- coding: utf-8 -*-
# FIXED: The logic within FluxBitBlock is corrected to concatenate hidden states
# before the attention call, resolving the RuntimeError.
# ADDED: forward_with_cfg is properly implemented for inference-time guidance.

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.utils import logging

# Your custom ternary layers are preserved
from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear

logger = logging.get_logger(__name__)

#################################################################################
#               AdaLayerNorm Implementations (from Diffusers)                   #
#################################################################################

class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Normalization Zero, used in Flux."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, 6 * embedding_dim, bias=True),
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb_out = self.emb(emb).unsqueeze(1)
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = emb_out.chunk(6, dim=2)
        x = self.norm(x)
        x = x * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class AdaLayerNormZeroSingle(nn.Module):
    """AdaLayerNormZero for single stream blocks in Flux."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, 2 * embedding_dim, bias=True),
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb_out = self.emb(emb).unsqueeze(1)
        scale, gate = emb_out.chunk(2, dim=2)
        x = self.norm(x)
        x = x * (1 + scale)
        return x, gate

#################################################################################
#               Core MMDiT-style Model Components (Refactored & Fixed)          #
#################################################################################

class HGRNBitMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        intermediate_size = int(hidden_size * mlp_ratio)
        self.gate_proj = BitLinear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        z = self.down_proj(self.act_fn(gate) * y)
        return z

class FluxBitBlock(nn.Module):
    """
    FIXED: A Dual-Stream block that correctly implements the Flux architecture.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_rope=False, use_ternary_rope=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = AdaLayerNormZero(hidden_size)
        self.norm1_context = AdaLayerNormZero(hidden_size)
        # The single attention block will process the concatenated streams
        self.attn = HGRNBitAttention(hidden_size=hidden_size, num_heads=num_heads, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = HGRNBitMLP(hidden_size=hidden_size, mlp_ratio=mlp_ratio)
        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = HGRNBitMLP(hidden_size=hidden_size, mlp_ratio=mlp_ratio)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor):
        # 1. Apply separate AdaLN to each stream
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

        # 2. Concatenate before attention
        len_c = norm_encoder_hidden_states.shape[1]
        joint_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)

        # 3. Single attention call on the joint stream
        attn_output, _, _ = self.attn(joint_hidden_states)

        # 4. Split after attention
        context_attn_output, attn_output = attn_output.split([len_c, norm_hidden_states.shape[1]], dim=1)

        # 5. Apply residuals and MLPs to each stream independently
        hidden_states = hidden_states + gate_msa * attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output

        return hidden_states, encoder_hidden_states


class FluxBitSingleBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_rope=False, use_ternary_rope=False, **block_kwargs):
        super().__init__()
        self.norm1 = AdaLayerNormZeroSingle(hidden_size)
        self.attn = HGRNBitAttention(hidden_size=hidden_size, num_heads=num_heads, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, **block_kwargs)
        self.mlp = HGRNBitMLP(hidden_size=hidden_size, mlp_ratio=mlp_ratio)
        self.proj_out = BitLinear(hidden_size * 2, hidden_size, bias=True)

    def forward(self, x, c):
        residual = x
        norm_x, gate = self.norm1(x, emb=c)
        attn_output, _, _ = self.attn(norm_x)
        mlp_output = self.mlp(norm_x)
        
        hidden_states = self.proj_out(torch.cat([attn_output, mlp_output], dim=2))
        x = residual + gate * hidden_states
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = BitLinear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


#################################################################################
#                          MeshDiT Main Class (Refactored)                      #
#################################################################################

class MeshDiT(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_dual_stream_blocks: int = 14,
        use_rope: bool = False,
        use_ternary_rope: bool = False,
    ):
        super().__init__()
        self.output_dim = input_dim
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        self.x_embedder = BitLinear(input_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.dual_stream_blocks = nn.ModuleList([
            FluxBitBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_rope=use_rope, use_ternary_rope=use_ternary_rope)
            for _ in range(num_dual_stream_blocks)
        ])
        num_single_stream_blocks = depth - num_dual_stream_blocks
        self.single_stream_blocks = nn.ModuleList([
            FluxBitSingleBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_rope=use_rope, use_ternary_rope=use_ternary_rope)
            for _ in range(num_single_stream_blocks)
        ])
        
        self.final_layer = FinalLayer(hidden_size, self.output_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, BitLinear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x, t, encoder_hidden_states):
        num_x_tokens = x.shape[1]
        x_tokens = self.x_embedder(x)
        t_emb = self.t_embedder(t)
        y_tokens = encoder_hidden_states

        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb)

        combined_tokens = torch.cat([y_tokens, x_tokens], dim=1)
        
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)
            
        processed_x_tokens = combined_tokens[:, -num_x_tokens:]
        output = self.final_layer(processed_x_tokens, t_emb)

        return output
    
    def forward_with_cfg(self, x, t, encoder_hidden_states, cfg_scale):
        """
        Forward pass with Classifier-Free Guidance.
        """
        model_out = self.forward(x, t, encoder_hidden_states)
        e_uncond, e_cond = model_out.chunk(2, dim=0)
        noise_pred = e_uncond + cfg_scale * (e_cond - e_uncond)
        
        return noise_pred

#################################################################################
#               Embedding Layers                                                #
#################################################################################

class TimestepEmbedder(nn.Module):
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
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))

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

