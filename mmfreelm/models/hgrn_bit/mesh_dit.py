# -*- coding: utf-8 -*-
# MODIFIED: Refactored to align with the FluxDenoiser/MMDiT architecture.
# This version implements a true two-way dual-stream architecture where both
# shape and conditioning streams are updated jointly. It now accepts a single
# pre-combined conditioning tensor (encoder_hidden_states).

from __future__ import annotations

import math
from typing import Optional, Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.utils import logging

# Your custom ternary layers are preserved
from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear

logger = logging.get_logger(__name__)


#################################################################################
#               AdaLayerNorm Implementations (from Diffusers)                   #
#               MODIFIED: Added directly to this script as requested.           #
#################################################################################

class AdaLayerNorm(nn.Module):
    """
    Norm layer adaptive to embedding inputs.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, emb):
        emb_out = self.linear(self.silu(self.emb(emb))).unsqueeze(1)
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        x = self.norm(x) * (1 + scale) + shift
        return x

class AdaLayerNormZero(nn.Module):
    """
    Adaptive Layer Normalization Zero, used in Flux.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, 6 * embedding_dim, bias=True),
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb_out = self.emb(emb).unsqueeze(1)
        # Theunsqueeze is for sequence broadcasting
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = emb_out.chunk(6, dim=2)
        x = self.norm(x)
        x = x * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class AdaLayerNormZeroSingle(nn.Module):
    """
    AdaLayerNormZero for single stream blocks in Flux.
    """
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
#               Core MMDiT-style Model Components (Refactored)                  #
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

## MODIFIED: This is the new Joint Attention module that enables two-way communication.
class JointStreamAttention(nn.Module):
    """
    Custom attention module to process two streams jointly, inspired by FluxAttnProcessor2_0.
    It uses your HGRNBitAttention as the core attention mechanism.
    """
    def __init__(self, query_dim: int, num_heads: int, use_rope: bool, use_ternary_rope: bool, **block_kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        # Projections for the 'shape' stream (hidden_states)
        self.to_q = BitLinear(query_dim, query_dim, bias=False)
        self.to_k = BitLinear(query_dim, query_dim, bias=False)
        self.to_v = BitLinear(query_dim, query_dim, bias=False)
        self.to_out = BitLinear(query_dim, query_dim, bias=False)

        # Projections for the 'conditioning' stream (encoder_hidden_states)
        self.add_q_proj = BitLinear(query_dim, query_dim, bias=False)
        self.add_k_proj = BitLinear(query_dim, query_dim, bias=False)
        self.add_v_proj = BitLinear(query_dim, query_dim, bias=False)
        self.to_add_out = BitLinear(query_dim, query_dim, bias=False)

        # Core ternary attention mechanism
        self.attn = HGRNBitAttention(
            hidden_size=query_dim, num_heads=num_heads, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, **block_kwargs
        )

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        # Project both streams
        q_s, k_s, v_s = self.to_q(hidden_states), self.to_k(hidden_states), self.to_v(hidden_states)
        q_c, k_c, v_c = self.add_q_proj(encoder_hidden_states), self.add_k_proj(encoder_hidden_states), self.add_v_proj(encoder_hidden_states)

        # Concatenate along the sequence dimension for joint processing
        q_joint = torch.cat([q_c, q_s], dim=1)
        k_joint = torch.cat([k_c, k_s], dim=1)
        v_joint = torch.cat([v_c, v_s], dim=1)

        # Pass through the core attention mechanism
        attn_output, _, _ = self.attn(q_joint, k_joint, v_joint) # Using a simplified call for HGRNBitAttention

        # Split the output back into two streams
        len_c = encoder_hidden_states.shape[1]
        attn_output_c, attn_output_s = attn_output[:, :len_c, :], attn_output[:, len_c:, :]

        # Final projection for each stream
        hidden_states = self.to_out(attn_output_s)
        encoder_hidden_states = self.to_add_out(attn_output_c)

        return hidden_states, encoder_hidden_states


## MODIFIED: Renamed from DualStreamBlock to FluxBitBlock and completely refactored.
class FluxBitBlock(nn.Module):
    """
    A Dual-Stream block that aligns with the Flux architecture, using ternary layers.
    It implements two-way, deeply integrated communication between streams.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_rope=False, use_ternary_rope=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size

        # Use AdaLayerNormZero for adaptive normalization, as in Flux
        self.norm1 = AdaLayerNormZero(hidden_size)
        self.norm1_context = AdaLayerNormZero(hidden_size)

        self.attn = JointStreamAttention(hidden_size, num_heads, use_rope, use_ternary_rope, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = HGRNBitMLP(hidden_size=hidden_size, mlp_ratio=mlp_ratio)
        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = HGRNBitMLP(hidden_size=hidden_size, mlp_ratio=mlp_ratio)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor):
        # Normalize shape stream
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # Normalize condition stream
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

        # Joint Attention (Two-way communication)
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        # --- Update Shape Stream (hidden_states) ---
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        # --- Update Conditioning Stream (encoder_hidden_states) ---
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states

## MODIFIED: Renamed from SingleStreamBlock to FluxBitSingleBlock for clarity.
class FluxBitSingleBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_rope=False, use_ternary_rope=False, **block_kwargs):
        super().__init__()
        self.norm1 = AdaLayerNormZeroSingle(hidden_size)
        self.attn = HGRNBitAttention(hidden_size=hidden_size, num_heads=num_heads, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, **block_kwargs)
        self.mlp = HGRNBitMLP(hidden_size=hidden_size, mlp_ratio=mlp_ratio)
        self.proj_out = BitLinear(hidden_size * 2, hidden_size, bias=True)


    def forward(self, x, c):
        # The single block in Flux processes the concatenated streams
        residual = x
        norm_x, gate = self.norm1(x, emb=c)
        
        attn_output, _, _ = self.attn(norm_x)
        mlp_output = self.mlp(norm_x)

        # Concatenate and project out
        hidden_states = torch.cat([attn_output, mlp_output], dim=2)
        
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        x = residual + hidden_states
        
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
        
        # Input Projections
        self.x_embedder = BitLinear(input_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # MODIFIED: Conditioning embedders removed, as we now expect a single pre-combined tensor.
        
        # Architecture blocks
        ## MODIFIED: Using the new FluxBitBlock
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
        """
        Forward pass for the refactored MeshDiT model.
        Args:
            x (torch.Tensor): Shape latents, e.g., [B, N_shape, D].
            t (torch.Tensor): Timesteps, e.g., [B].
            encoder_hidden_states (torch.Tensor): Pre-combined conditioning latents, e.g., [B, N_cond, D].
        """
        num_x_tokens = x.shape[1]
        
        x_tokens = self.x_embedder(x)
        t_emb = self.t_embedder(t)
        # MODIFIED: No internal embedding for y. Use encoder_hidden_states directly.
        y_tokens = encoder_hidden_states

        # Process through Dual-Stream blocks
        for block in self.dual_stream_blocks:
            x_tokens, y_tokens = block(x_tokens, y_tokens, t_emb)

        # Concatenate for Single-Stream blocks
        combined_tokens = torch.cat([y_tokens, x_tokens], dim=1)
        
        # Process through Single-Stream blocks
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)
            
        # Isolate the shape tokens and process through final layer
        # MODIFIED: Slicing must account for the order of concatenation.
        processed_x_tokens = combined_tokens[:, -num_x_tokens:]
        output = self.final_layer(processed_x_tokens, t_emb)

        assert output.shape == x.shape, f"Final shape mismatch! Output: {output.shape}, Input: {x.shape}"
        return output
    
    # The forward_with_cfg is complex and highly specific to the training loop.
    # It would need to be updated to handle the single `encoder_hidden_states` tensor.
    # For now, I have commented it out as it requires a corresponding change in the training script.
    # def forward_with_cfg(...)


#################################################################################
#               Embedding Layers (Moved here for better organization)           #
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
