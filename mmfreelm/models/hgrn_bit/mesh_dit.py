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
        t_emb = self.mlp(t_freq)
        return t_emb


class TextEmbedder(nn.Module):
    """
    Embeds tokenized text into vector representations.
    """
    def __init__(self, vocab_size, hidden_size, dropout_prob):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(self.vocab_size + 1, hidden_size)
        self.null_idx = self.vocab_size

        self.mlp = HGRNBitMLP(hidden_size, 4, hidden_act='swish')

    def token_drop(self, input_ids, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(input_ids.shape[0], device=input_ids.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        
        drop_ids = drop_ids.unsqueeze(-1)
        return torch.where(drop_ids, self.null_idx, input_ids)

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
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        
        self.null_embedding = nn.Parameter(torch.randn(hidden_size))
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, image_latent, train=True, force_drop_mask=None):
        """
        image_latent: A tensor of shape [N, C, H, W]
        force_drop_mask: A tensor of shape [N,] with 1s for dropping and 0s for keeping.
        """
        flattened_latent = torch.flatten(image_latent, start_dim=1)
        projected_latent = self.mlp(flattened_latent)

        if force_drop_mask is not None:
            mask = force_drop_mask.float().unsqueeze(1)
        elif train and self.dropout_prob > 0:
            mask = (torch.rand(image_latent.shape[0], device=image_latent.device) < self.dropout_prob).float().unsqueeze(1)
        else:
            mask = torch.zeros(image_latent.shape[0], 1, device=image_latent.device)

        final_embedding = mask * self.null_embedding + (1 - mask) * projected_latent
        return final_embedding


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class AdaLNConditioning(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_size: int, 
        output_dim: int,
        eps: float = 1e-6,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ):
        super().__init__()
        self.input_proj = BitLinear(input_dim, hidden_size, bias=False)
        self.hgrn_mlp = HGRNBitMLP(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act
        )
        self.output_proj = BitLinear(hidden_size, output_dim, bias=True)
        self.norm = RMSNorm(output_dim, eps=eps)
        self.out_proj = BitLinear(output_dim, output_dim, bias=True)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(condition)
        x = self.hgrn_mlp(x)
        x = self.output_proj(x)
        x = self.norm(x)
        return self.out_proj(x)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, **block_kwargs)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = HGRNBitMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim, hidden_act='swish')
        self.adaLN_modulation = AdaLNConditioning(hidden_size, hidden_size, 6 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio, hidden_act='swish')

    def forward(self, x, c):
        modulated_c = ACT2FN['silu'](c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(modulated_c).chunk(6, dim=1)
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _, _ = self.attn(modulated_x)
        x = x + gate_msa.unsqueeze(1) * attn_output
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, mlp_ratio=4.0):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = BitLinear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = AdaLNConditioning(hidden_size, hidden_size, 2 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio, hidden_act='swish')

    def forward(self, x, c):
        modulated_c = ACT2FN['silu'](c)
        shift, scale = self.adaLN_modulation(modulated_c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

#################################################################################
#                                 DiT Main Class                                #
#################################################################################
        
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone, conditioned on text and image latents.
    """ 
    def __init__(
        self,
        vocab_size: int,
        image_latent_channels: int,
        image_latent_height: int,
        image_latent_width: int,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.1,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(vocab_size, hidden_size, dropout_prob)
        
        image_latent_dim = image_latent_channels * image_latent_height * image_latent_width
        self.image_embedder = ImageLatentEmbedder(image_latent_dim, hidden_size, dropout_prob)
        
        num_patches = self.x_embedder.num_patches
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
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x, t, y):
        """
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        y: dict containing conditioning information:
           "input_ids": (N, seq_len)
           "attention_mask": (N, seq_len)
           "image_latent": (N, C_latent, H_latent, W_latent)
        """
        x = self.x_embedder(x) + self.pos_embed
        
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y["input_ids"], y["attention_mask"], train=self.training)
        img_emb = self.image_embedder(y["image_latent"], train=self.training)
        
        c = t_emb + y_emb + img_emb
        
        for block in self.blocks:
            x = block(x, c)
        
        x = self.final_layer(x, c)
        return self.unpatchify(x)

    def forward_with_cfg(self, x, t, y, cfg_scale_text, cfg_scale_image):
        """
        Forward pass with Classifier-Free Guidance for both text and image.
        """
        half = x.shape[0] // 2
        combined_x = torch.cat([x[:half], x[:half], x[:half], x[:half]], dim=0)
        combined_t = torch.cat([t[:half], t[:half], t[:half], t[:half]], dim=0)

        # Prepare conditioning inputs for all 4 cases
        y_ids = y["input_ids"][:half]
        y_mask = y["attention_mask"][:half]
        y_img = y["image_latent"][:half]

        # Get timestep embedding
        t_emb = self.t_embedder(combined_t)
        
        # Get text embeddings (uncond, uncond, cond, cond)
        text_drop_mask = torch.tensor([1, 1, 0, 0], device=x.device).repeat_interleave(half)
        y_emb = self.y_embedder(
            y_ids.repeat(4, 1), 
            y_mask.repeat(4, 1), 
            force_drop_ids=text_drop_mask
        )
        
        # Get image embeddings (uncond, cond, uncond, cond)
        img_drop_mask = torch.tensor([1, 0, 1, 0], device=x.device).repeat_interleave(half)
        img_emb = self.image_embedder(
            y_img.repeat(4, 1, 1, 1),
            force_drop_mask=img_drop_mask
        )
        
        # Combine conditioning signals for all 4 cases
        c_combined = t_emb + y_emb + img_emb
        
        # Process through the model
        x_combined = self.x_embedder(combined_x) + self.pos_embed.repeat(4, 1, 1)
        for block in self.blocks:
            x_combined = block(x_combined, c_combined)
        model_out = self.unpatchify(self.final_layer(x_combined, c_combined))

        # Separate outputs and perform guidance
        e_uncond, e_img, e_text, e_full = torch.chunk(model_out, 4, dim=0)
        
        noise_pred = e_uncond + \
                     cfg_scale_text * (e_text - e_uncond) + \
                     cfg_scale_image * (e_img - e_uncond)
        
        # The output should be duplicated to match the original input batch size
        return noise_pred.repeat(2, 1, 1, 1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


#################################################################################
#                                   DiT Configs                                 #
#################################################################################
def DiT_XXL_2(**kwargs):
    return DiT(depth=32, hidden_size=1536, patch_size=2, num_heads=24, **kwargs)

def DiT_XXL_4(**kwargs):
    return DiT(depth=32, hidden_size=1536, patch_size=4, num_heads=24, **kwargs)

def DiT_XXL_8(**kwargs):
    return DiT(depth=32, hidden_size=1536, patch_size=8, num_heads=24, **kwargs)
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
    'DiT-XXL/2': DiT_XXL_2,'DiT-XXL/4': DiT_XXL_4,'DiT-XXL/8': DiT_XXL_8,
}
