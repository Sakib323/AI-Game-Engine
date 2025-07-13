# -*- coding: utf-8 -*-
# Adapted for 3D Mesh Latent Generation with optional Rotary Embeddings

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.utils import logging

# The HGRNBitAttention class handles the RoPE implementation internally.
# We don't need to import or use RotaryEmbedding directly in this file.
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
    """
    def __init__(self, vocab_size, hidden_size, dropout_prob):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, hidden_size)
        self.null_idx = vocab_size
        self.dropout_prob = dropout_prob
        self.mlp = HGRNBitMLP(hidden_size, 4, hidden_act='swish')

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
        return self.mlp(pooled_embeddings)


class ImageLatentEmbedder(nn.Module):
    """
    Embeds 4D image latents [N, C, H, W] into vector representations.
    """
    def __init__(self, input_dim, hidden_size, dropout_prob):
        super().__init__()
        self.null_embedding = nn.Parameter(torch.randn(hidden_size))
        self.dropout_prob = dropout_prob
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, image_latent, train=True, force_drop_mask=None):
        flattened_latent = torch.flatten(image_latent, start_dim=1)
        projected_latent = self.mlp(flattened_latent)

        if force_drop_mask is not None:
            mask = force_drop_mask.float().unsqueeze(1)
        elif train and self.dropout_prob > 0:
            mask = (torch.rand(image_latent.shape[0], device=image_latent.device) < self.dropout_prob).float().unsqueeze(1)
        else:
            mask = torch.zeros(image_latent.shape[0], 1, device=image_latent.device)
        return mask * self.null_embedding + (1 - mask) * projected_latent


#################################################################################
#                          Core MeshDiT Model Components                        #
#################################################################################

class AdaLNConditioning(nn.Module):
    """
    Generates adaptive layer norm modulation parameters from a conditioning signal.
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
        self.hgrn_mlp = HGRNBitMLP(hidden_size=hidden_size, hidden_ratio=hidden_ratio, hidden_act='swish')
        self.output_proj = nn.Linear(hidden_size, output_dim, bias=True)
        self.norm = RMSNorm(output_dim, eps=eps)
        self.out_proj = nn.Linear(output_dim, output_dim, bias=True)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(condition)
        x = self.hgrn_mlp(x)
        x = self.output_proj(x)
        x = self.norm(x)
        return self.out_proj(x)


class MeshDiTBlock(nn.Module):
    """
    A single block of the Diffusion Transformer. It delegates RoPE logic
    to the HGRNBitAttention layer.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_rope=False, use_ternary_rope=False, **block_kwargs):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # We pass the RoPE flags here. HGRNBitAttention will handle the rest.
        self.attn = HGRNBitAttention(
            mode='fused_recurrent',
            hidden_size=hidden_size,
            num_heads=num_heads,
            rotary_embeddings=use_rope,
            use_ternary_rope=use_ternary_rope,
            **block_kwargs
        )
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = HGRNBitMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim, hidden_act='swish')
        
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        
        self.adaLN_modulation = AdaLNConditioning(hidden_size, hidden_size, 6 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio)

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
        self.adaLN_modulation = AdaLNConditioning(hidden_size, hidden_size, 2 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio)

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
    Diffusion model with a Transformer backbone for 3D Mesh Generation, 
    configurable to use either absolute positional embeddings or RoPE.
    """ 
    def __init__(
        self,
        # Mesh Latent parameters
        input_tokens: int = 2048,
        input_dim: int = 64,
        # Text conditioning parameters
        vocab_size: int = 49408,
        # Image conditioning parameters
        image_latent_channels: int = 4,
        image_latent_height: int = 64,
        image_latent_width: int = 64,
        # Model architecture parameters
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.1,
        # --- RoPE configuration flags ---
        use_rope: bool = True,
        use_ternary_rope: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.num_heads = num_heads
        self.use_rope = use_rope

        self.x_embedder = BitLinear(input_dim, hidden_size, bias=True)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(vocab_size, hidden_size, dropout_prob)
        image_latent_dim = image_latent_channels * image_latent_height * image_latent_width
        self.image_embedder = ImageLatentEmbedder(image_latent_dim, hidden_size, dropout_prob)
        
        # --- Conditional Positional Embedding Strategy ---
        if not self.use_rope:
            # If RoPE is off, use standard learnable absolute positional embeddings.
            self.pos_embed = nn.Parameter(torch.zeros(1, input_tokens, hidden_size))
            logger.info("Using learnable absolute positional embeddings.")
        else:
            # If RoPE is on, we don't need absolute embeddings.
            # The HGRNBitAttention layer will handle rotary embeddings internally.
            self.pos_embed = None
            logger.info("Using Rotary Positional Embeddings (RoPE) within attention blocks.")

        # Pass RoPE configuration down to each transformer block.
        self.blocks = nn.ModuleList([
            MeshDiTBlock(
                hidden_size, 
                num_heads, 
                mlp_ratio=mlp_ratio,
                use_rope=use_rope,
                use_ternary_rope=use_ternary_rope
            ) for _ in range(depth)
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
        
        # Initialize absolute positional embeddings only if they exist.
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)

        nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
        
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation.output_proj.weight, 0)
            nn.init.constant_(block.adaLN_modulation.output_proj.bias, 0)
            nn.init.constant_(block.adaLN_modulation.out_proj.weight, 0)
            nn.init.constant_(block.adaLN_modulation.out_proj.bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation.output_proj.weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation.output_proj.bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation.out_proj.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x, t, y):
        """
        Forward pass for the MeshDiT model.
        """
        x = self.x_embedder(x)
        # Add absolute positional embeddings only if RoPE is not being used.
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y["input_ids"], y["attention_mask"], train=self.training)
        img_emb = self.image_embedder(y["image_latent"], train=self.training)
        
        c = t_emb + y_emb + img_emb
        
        for block in self.blocks:
            x = block(x, c)
        
        x = self.final_layer(x, c)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale_text, cfg_scale_image):
        """
        Forward pass with Classifier-Free Guidance.
        """
        half = x.shape[0] // 2
        combined_x = torch.cat([x[:half], x[:half], x[:half], x[:half]], dim=0)
        combined_t = torch.cat([t[:half], t[:half], t[:half], t[:half]], dim=0)

        y_ids = y["input_ids"][:half]
        y_mask = y["attention_mask"][:half]
        y_img = y["image_latent"][:half]

        t_emb = self.t_embedder(combined_t)
        
        text_drop_mask = torch.tensor([1, 1, 0, 0], device=x.device).repeat_interleave(half)
        y_emb = self.y_embedder(
            y_ids.repeat(4, 1), 
            y_mask.repeat(4, 1), 
            force_drop_ids=text_drop_mask
        )
        
        img_drop_mask = torch.tensor([1, 0, 1, 0], device=x.device).repeat_interleave(half)
        img_emb = self.image_embedder(
            y_img.repeat(4, 1, 1, 1),
            force_drop_mask=img_drop_mask
        )
        
        c_combined = t_emb + y_emb + img_emb
        
        x_combined = self.x_embedder(combined_x)
        # Apply absolute positional embeddings conditionally for CFG.
        if self.pos_embed is not None:
            x_combined = x_combined + self.pos_embed.repeat(4, 1, 1)

        for block in self.blocks:
            x_combined = block(x_combined, c_combined)
        model_out = self.final_layer(x_combined, c_combined)

        e_uncond, e_img, e_text, e_full = torch.chunk(model_out, 4, dim=0)
        
        noise_pred = e_uncond + \
                     cfg_scale_text * (e_text - e_uncond) + \
                     cfg_scale_image * (e_img - e_uncond)
        
        return noise_pred


#################################################################################
#                                  MeshDiT Configs                              #
#################################################################################

def MeshDiT_XL(**kwargs):
    return MeshDiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def MeshDiT_L(**kwargs):
    return MeshDiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def MeshDiT_B(**kwargs):
    return MeshDiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def MeshDiT_S(**kwargs):
    return MeshDiT(depth=12, hidden_size=384, num_heads=6, **kwargs)


MeshDiT_models = {
    'MeshDiT-XL': MeshDiT_XL,
    'MeshDiT-L':  MeshDiT_L,
    'MeshDiT-B':  MeshDiT_B,
    'MeshDiT-S':  MeshDiT_S,
}
