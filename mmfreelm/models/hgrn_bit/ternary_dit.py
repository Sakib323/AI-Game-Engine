# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModel

import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.utils import RecurrentCache
from mmfreelm.modules import FusedCrossEntropyLoss, RMSNorm, LayerNorm
from mmfreelm.modules.activations import swiglu_linear, swiglu, ACT2FN
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear
from mmfreelm.models.hgrn_bit.hgrn_bit_moe import HGRNBitMoE
from mmfreelm.layers.hgrn_bit import HGRNBitAttention


logger = logging.get_logger(__name__)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class HGRNBitMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ) -> HGRNBitMLP:
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = BitLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        z = self.down_proj(swiglu(gate, y))
        return z

#################################################################################
#               Embedding Layers for Timesteps and Text                         #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using HGRNBitMLP.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()

        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

        # Project from [batch, frequency_embedding_size] -> [batch, hidden_size]
        self.input_proj = BitLinear(frequency_embedding_size, hidden_size, bias=False)

        self.mlp = HGRNBitMLP(
            hidden_size=hidden_size,
            hidden_ratio=4,
            intermediate_size=None,
            hidden_act='swish'
        )

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
        t_freq = self.input_proj(t_freq)
        t_emb = self.mlp(t_freq)
        return t_emb




class TextEmbedder(nn.Module):
    def __init__(self, tokenizer_name, hidden_size, dropout_prob, max_length=128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.max_length = max_length

        # CORRECTED: Use standard name 'embedding'
        self.embedding = nn.Embedding(self.vocab_size + 1, hidden_size)  # +1 for null embedding
        self.null_idx = self.vocab_size  # Index for null embedding

        self.mlp = HGRNBitMLP(
            hidden_size=hidden_size,
            hidden_ratio=4,
            intermediate_size=None,
            hidden_act='swish'
        )

    # ADD NULL TOKEN HANDLING
    def token_drop(self, input_ids, attention_mask, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(input_ids.shape[0], device=input_ids.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        
        # Ensure we have same dimensions
        drop_ids = drop_ids.unsqueeze(-1).expand_as(input_ids)
        return torch.where(drop_ids, self.null_idx, input_ids)
    def forward(self, texts, train=True, force_drop_ids=None):
        """
        Args:
            texts: List of strings or batched tensor of token IDs.
            train: Whether to apply dropout (True during training).
            force_drop_ids: Optional tensor to force dropout for specific samples.
        Returns:
            embeddings: Tensor of shape [batch, hidden_size].
        """
        # Tokenize texts if input is strings
        if isinstance(texts, (list, tuple)):
            
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = tokenized["input_ids"].to(self.embedding.weight.device)
            attention_mask = tokenized["attention_mask"].to(self.embedding.weight.device)
        else:
            input_ids = texts
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Apply dropout for classifier-free guidance
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            input_ids = self.token_drop(input_ids, attention_mask, force_drop_ids)

        # Embed tokens
        embeddings = self.embedding(input_ids)  # [batch, seq_len, hidden_size]

        # Mean pooling over non-padded tokens
        attention_mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        sum_embeddings = (embeddings * attention_mask).sum(dim=1)  # [batch, hidden_size]
        sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
        pooled_embeddings = sum_embeddings / sum_mask  # [batch, hidden_size]

        # Project to final hidden_size
        final_embeddings = self.mlp(pooled_embeddings)  # [batch, hidden_size]

        return final_embeddings




#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class AdaLNConditioning(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_size: int, 
        output_dim: int,  # New output dimension parameter
        eps: float = 1e-6,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ):
        super().__init__()
        # Project input to hidden_size
        self.input_proj = BitLinear(input_dim, hidden_size, bias=False)
        
        # HGRN-based MLP processing
        self.hgrn_mlp = HGRNBitMLP(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act
        )
        
        # Final output projection
        self.output_proj = BitLinear(hidden_size, output_dim, bias=True)  # Maintain original bias
        
        # Normalization and output layers
        self.norm = RMSNorm(output_dim, eps=eps)
        self.out_proj = BitLinear(output_dim, output_dim, bias=True)  # Maintain original bias

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(condition)
        x = self.hgrn_mlp(x)
        x = self.output_proj(x)
        x = self.norm(x)
        return self.out_proj(x)


class DiTBlock(nn.Module):
    """
    A DiT block with enhanced adaptive layer norm conditioning using HGRN-based modulation
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = HGRNBitAttention(
            mode='fused_recurrent',
            hidden_size=hidden_size,
            num_heads=num_heads,
            expand_ratio=1,
            use_short_conv=False,
            conv_size=4,
            conv_bias=False,
            share_conv_kernel=True,
            layernorm_eps=1e-5,
            layer_idx=None,
            rotary_embeddings=False,
            rope_theta=10000.0,
            use_ternary_rope=False
        )
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = HGRNBitMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim, hidden_act='swish')
        
        # Replace simple linear with enhanced conditioning network
        self.adaLN_modulation = AdaLNConditioning(
            input_dim=hidden_size,
            hidden_size=hidden_size,      # Same hidden size as block
            output_dim=6 * hidden_size,   # Produces 6 modulation parameters
            eps=1e-6,
            hidden_ratio=mlp_ratio,       # Use same ratio as main MLP
            hidden_act='swish'
        )

    def forward(self, x, c):
        # Process conditioning vector
        modulated_c = ACT2FN['silu'](c)
        # Get 6 modulation parameters
        params = self.adaLN_modulation(modulated_c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=1)
        
        # Attention path
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _, _ = self.attn(
            modulated_x,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            lower_bound=None
        )
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        # MLP path
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT with enhanced conditioning.
    """
    def __init__(self, hidden_size, patch_size, out_channels, mlp_ratio=4.0):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = BitLinear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
        self.adaLN_modulation = AdaLNConditioning(
            input_dim=hidden_size,
            hidden_size=hidden_size,
            output_dim=2 * hidden_size,   # Produces scale and shift
            eps=1e-6,
            hidden_ratio=mlp_ratio,       # Use same ratio as blocks
            hidden_act='swish'
        )

    def forward(self, x, c):
        # Apply SiLU to conditioning input (consistent with DiTBlock)
        modulated_c = ACT2FN['silu'](c)
        
        # Get modulation parameters
        params = self.adaLN_modulation(modulated_c)
        shift, scale = params.chunk(2, dim=1)
        
        # Apply modulation and output
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

    
        
    
    
#################################################################################
#                                 TerDiT                                        #
#################################################################################
        
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """ 
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        dropout_prob=0.1,
        tokenizer_name = "Sakib323/MMfreeLM-370M",
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(tokenizer_name, hidden_size, dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        
    def initialize_weights(self):
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
            
            grid_size = int(self.x_embedder.num_patches ** 0.5)
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp.gate_proj.weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp.down_proj.weight, std=0.02)
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
    

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        num_patches = x.shape[1]
        h = w = int(math.isqrt(num_patches))
        assert h * w == num_patches, "Number of patches must be a perfect square"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x, t, y):
            x = self.x_embedder(x) + self.pos_embed
            t = self.t_embedder(t)
            y_emb = self.y_embedder(y, train=self.training)
            c = t + y_emb
            for block in self.blocks:
                x = block(x, c)
                x = self.final_layer(x, c)
                return self.unpatchify(x)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                 #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}