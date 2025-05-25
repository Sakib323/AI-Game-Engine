# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.utils import RecurrentCache
from mmfreelm.modules import FusedCrossEntropyLoss, RMSNorm
from mmfreelm.modules.activations import swiglu_linear, swiglu
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear
from mmfreelm.models.hgrn_bit.hgrn_bit_moe import HGRNBitMoE

logger = logging.get_logger(__name__)

# Base class for inheritance
class HGRNBitPreTrainedModel(PreTrainedModel):
    config_class = HGRNBitConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, BitLinear)):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

# AdaLNConditioning (already defined, kept as is)
class AdaLNConditioning(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = HGRNBitMLP(
            hidden_size=input_dim,
            intermediate_size=hidden_size * 2,
            hidden_act='swish',
            output_dim=hidden_size * 2
        )
        self.norm = RMSNorm(hidden_size * 2, eps=eps)
        self.out_proj = BitLinear(hidden_size * 2, hidden_size * 2, bias=False)

    def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        print(f"AdaLNConditioning: condition shape before mlp = {condition.shape}")
        x = self.mlp(condition)
        print(f"AdaLNConditioning: x shape after mlp = {x.shape}")
        x = self.norm(x)
        params = self.out_proj(x)
        scale, shift = params.chunk(2, dim=-1)
        return scale, shift

class HGRNBitMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
        output_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.output_dim = output_dim if output_dim is not None else hidden_size

        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = BitLinear(self.intermediate_size, self.output_dim, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        z = self.down_proj(swiglu(gate, y))
        return z
    
    
class HGRNBitBlock(nn.Module):
    def __init__(self, config: HGRNBitConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.attn = HGRNBitAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_ratio=config.expand_ratio,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            share_conv_kernel=config.share_conv_kernel,
            layernorm_eps=config.rms_norm_eps,
            layer_idx=layer_idx,
            rotary_embeddings=config.rotary_embeddings,
            rope_theta=config.rope_theta,
            use_ternary_rope=config.use_ternary_rope
        )
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

        # Add adaLN modules for attention and MLP outputs
        self.attn_adaln = AdaLNConditioning(input_dim=config.hidden_size, hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_adaln = AdaLNConditioning(input_dim=config.hidden_size, hidden_size=config.hidden_size, eps=config.rms_norm_eps)

        # Embeddings for conditioning inputs (timestep t and label y)
        self.timestep_embedding = nn.Embedding(config.diffusion_timesteps, config.hidden_size)
        self.label_embedding = nn.Embedding(10, config.hidden_size)

        # MLP initialization
        if config.moe:
            self.mlp = HGRNBitMoE(config)
        else:
            self.mlp = HGRNBitMLP(
                hidden_size=config.hidden_size,
                hidden_ratio=config.hidden_ratio,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = False,
        timestep: Optional[torch.LongTensor] = None,
        label: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        print(f"Input hidden_states shape: {hidden_states.shape}")
        print(f"attn_norm weight shape: {self.attn_norm.weight.shape}")
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Prepare conditioning input for adaLN
        condition = torch.zeros(hidden_states.shape[0], self.hidden_size, device=hidden_states.device)
        if timestep is not None:
            timestep_embed = self.timestep_embedding(timestep)
            condition = condition + timestep_embed
        if label is not None:
            label_embed = self.label_embedding(label)
            condition = condition + label_embed

        # Attention layer with adaLN
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            lower_bound=lower_bound
        )
        scale, shift = self.attn_adaln(condition)
        hidden_states = hidden_states * (1 + scale) + shift

        # MLP layer with adaLN
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        scale, shift = self.mlp_adaln(condition)
        hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)
        return outputs
# Restored HGRNBitModel class for compatibility with other files
class HGRNBitModel(HGRNBitPreTrainedModel):
    def __init__(self, config: HGRNBitConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))

        self.layers = nn.ModuleList([HGRNBitBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            hidden_states = self.embeddings(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
            batch_size, seq_length, _ = hidden_states.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache:
            if past_key_values is None:
                past_key_values = [layer.attn.init_state(batch_size) for layer in self.layers]
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden_states.device)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        if self.config.use_lower_bound:
            lower_bounds = self.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            lower_bound = lower_bounds[i] if self.config.use_lower_bound else None
            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    lower_bound,
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    lower_bound=lower_bound,
                )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(x for x in [hidden_states, next_cache, all_hidden_states, all_attns] if x is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )

class HGRNBitForCausalLM(HGRNBitPreTrainedModel):
    def __init__(self, config: HGRNBitConfig, patch_size=4, in_channels=4):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Compute the number of patches (assuming latent size 32x32)
        self.latent_size = 32  # Hardcoded for now, can be made configurable
        self.num_patches = (self.latent_size // self.patch_size) ** 2  # e.g., (32/4)^2 = 64

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))

        # DiT blocks
        self.layers = nn.ModuleList([HGRNBitBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Final projection to output noise (same shape as input latent)
        self.final_layer = nn.Sequential(
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size, self.in_channels * self.patch_size * self.patch_size, bias=False)
        )

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (batch_size, num_patches, hidden_size)
        timestep: Optional[torch.LongTensor] = None,
        label: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Add CLS token to hidden states
        batch_size = hidden_states.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_size)
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)  # (batch_size, num_patches + 1, hidden_size)

        # Prepare attention mask if provided
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=hidden_states.device)
            attention_mask = torch.cat((cls_mask, attention_mask), dim=1)

        if use_cache:
            if past_key_values is None:
                past_key_values = [layer.attn.init_state(batch_size) for layer in self.layers]
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        if self.config.use_lower_bound:
            lower_bounds = self.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            lower_bound = lower_bounds[i] if self.config.use_lower_bound else None
            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    lower_bound,
                    timestep,
                    label,
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    lower_bound=lower_bound,
                    timestep=timestep,
                    label=label,
                )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            hidden_states = hidden_states[:, 1:, :]
            noise = self.final_layer(hidden_states)
            noise = noise.view(batch_size, self.num_patches, self.in_channels, self.patch_size, self.patch_size)
            noise = noise.permute(0, 2, 1, 3, 4).contiguous() 
            noise = noise.view(batch_size,self.in_channels,self.latent_size, self.latent_size)
        next_cache = None
        if use_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(x for x in [noise, next_cache, all_hidden_states, all_attns] if x is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=noise,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )
    def __init__(self, config: HGRNBitConfig, patch_size=4, in_channels=4):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Compute the number of patches (assuming latent size 32x32)
        self.latent_size = 32  # Hardcoded for now, can be made configurable
        self.num_patches = (self.latent_size // self.patch_size) ** 2  # e.g., (32/4)^2 = 64

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))

        # DiT blocks
        self.layers = nn.ModuleList([HGRNBitBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Final projection to output noise (same shape as input latent)
        self.final_layer = nn.Sequential(
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size, self.in_channels * self.patch_size * self.patch_size, bias=False)
        )

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (batch_size, num_patches, hidden_size)
        timestep: Optional[torch.LongTensor] = None,
        label: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Add CLS token to hidden states
        batch_size = hidden_states.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_size)
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)  # (batch_size, num_patches + 1, hidden_size)

        # Prepare attention mask if provided
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=hidden_states.device)
            attention_mask = torch.cat((cls_mask, attention_mask), dim=1)

        if use_cache:
            if past_key_values is None:
                past_key_values = [layer.attn.init_state(batch_size) for layer in self.layers]
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        if self.config.use_lower_bound:
            lower_bounds = self.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            lower_bound = lower_bounds[i] if self.config.use_lower_bound else None
            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    lower_bound,
                    timestep,
                    label,
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    lower_bound=lower_bound,
                    timestep=timestep,
                    label=label,
                )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Remove CLS token and project to noise
        hidden_states = hidden_states[:, 1:, :]  # (batch_size, num_patches, hidden_size)
        noise = self.final_layer(hidden_states)  # (batch_size, num_patches, in_channels * patch_size * patch_size)
        noise = noise.view(batch_size, self.num_patches, self.in_channels, self.patch_size, self.patch_size)
        noise = noise.permute(0, 2, 1, 3, 4)  # (batch_size, in_channels, num_patches, patch_size, patch_size)
        # Reshape to (batch_size, in_channels, latent_size, latent_size)
        noise = noise.reshape(batch_size, self.in_channels, self.latent_size, self.latent_size)

        next_cache = None
        if use_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(x for x in [noise, next_cache, all_hidden_states, all_attns] if x is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=noise,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )
    def __init__(self, config: HGRNBitConfig, patch_size=4, in_channels=4):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Compute the number of patches (assuming latent size 32x32)
        self.latent_size = 32  # Hardcoded for now, can be made configurable
        self.num_patches = (self.latent_size // self.patch_size) ** 2  # e.g., (32/4)^2 = 64

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))

        # DiT blocks
        self.layers = nn.ModuleList([HGRNBitBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Final projection to output noise (same shape as input latent)
        self.final_layer = nn.Sequential(
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size, self.in_channels * self.patch_size * self.patch_size, bias=False)
        )

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (batch_size, num_patches, hidden_size)
        timestep: Optional[torch.LongTensor] = None,
        label: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Add CLS token to hidden states
        batch_size = hidden_states.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_size)
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)  # (batch_size, num_patches + 1, hidden_size)

        # Prepare attention mask if provided
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=hidden_states.device)
            attention_mask = torch.cat((cls_mask, attention_mask), dim=1)

        if use_cache:
            if past_key_values is None:
                past_key_values = [layer.attn.init_state(batch_size) for layer in self.layers]
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        if self.config.use_lower_bound:
            lower_bounds = self.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            lower_bound = lower_bounds[i] if self.config.use_lower_bound else None
            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    lower_bound,
                    timestep,
                    label,
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    lower_bound=lower_bound,
                    timestep=timestep,
                    label=label,
                )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Remove CLS token and project to noise
        hidden_states = hidden_states[:, 1:, :]  # (batch_size, num_patches, hidden_size)
        noise = self.final_layer(hidden_states)  # (batch_size, num_patches, in_channels * patch_size * patch_size)
        noise = noise.view(batch_size, self.num_patches, self.in_channels, self.patch_size, self.patch_size)
        noise = noise.permute(0, 2, 1, 3, 4)  # (batch_size, in_channels, num_patches, patch_size, patch_size)
        # Reshape to (batch_size, in_channels, latent_size, latent_size)
        noise = noise.reshape(batch_size, self.in_channels, self.latent_size, self.latent_size)

        next_cache = None
        if use_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(x for x in [noise, next_cache, all_hidden_states, all_attns] if x is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=noise,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )
    def __init__(self, config: HGRNBitConfig, patch_size=4, in_channels=4):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Compute the number of patches (assuming latent size 32x32)
        self.latent_size = 32  # Hardcoded for now, can be made configurable
        self.num_patches = (self.latent_size // self.patch_size) ** 2  # e.g., (32/4)^2 = 64

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))

        # DiT blocks
        self.layers = nn.ModuleList([HGRNBitBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Final projection to output noise (same shape as input latent)
        self.final_layer = nn.Sequential(
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size, self.in_channels * self.patch_size * self.patch_size, bias=False)
        )

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (batch_size, num_patches, hidden_size)
        timestep: Optional[torch.LongTensor] = None,
        label: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Add CLS token to hidden states
        batch_size = hidden_states.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_size)
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)  # (batch_size, num_patches + 1, hidden_size)

        # Prepare attention mask if provided
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=hidden_states.device)
            attention_mask = torch.cat((cls_mask, attention_mask), dim=1)

        if use_cache:
            if past_key_values is None:
                past_key_values = [layer.attn.init_state(batch_size) for layer in self.layers]
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        if self.config.use_lower_bound:
            lower_bounds = self.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            lower_bound = lower_bounds[i] if self.config.use_lower_bound else None
            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    lower_bound,
                    timestep,
                    label,
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    lower_bound=lower_bound,
                    timestep=timestep,
                    label=label,
                )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Remove CLS token and project to noise
        hidden_states = hidden_states[:, 1:, :]  # (batch_size, num_patches, hidden_size)
        noise = self.final_layer(hidden_states)  # (batch_size, num_patches, in_channels * patch_size * patch_size)
        noise = noise.view(batch_size, self.num_patches, self.in_channels, self.patch_size, self.patch_size)
        noise = noise.permute(0, 2, 1, 3, 4)  # (batch_size, in_channels, num_patches, patch_size, patch_size)
        # Reshape to (batch_size, in_channels, latent_size, latent_size)
        noise = noise.reshape(batch_size, self.in_channels, self.latent_size, self.latent_size)

        next_cache = None
        if use_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(x for x in [noise, next_cache, all_hidden_states, all_attns] if x is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=noise,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )
    def __init__(self, config: HGRNBitConfig, patch_size=4, in_channels=4):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Compute the number of patches (assuming latent size 32x32)
        self.latent_size = 32  # Hardcoded for now, can be made configurable
        self.num_patches = (self.latent_size // self.patch_size) ** 2  # e.g., (32/4)^2 = 64

        # Patch embedding is handled externally via patch_embedding.py
        # Expected input shape: (batch_size, num_patches + 1, hidden_size)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))

        # DiT blocks
        self.layers = nn.ModuleList([HGRNBitBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Final projection to output noise (same shape as input latent)
        self.final_layer = nn.Sequential(
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size, self.in_channels * self.patch_size * self.patch_size, bias=False)
        )

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (batch_size, num_patches, hidden_size)
        timestep: Optional[torch.LongTensor] = None,
        label: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Add CLS token to hidden states
        batch_size = hidden_states.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_size)
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)  # (batch_size, num_patches + 1, hidden_size)

        # Prepare attention mask if provided
        if attention_mask is not None:
            # Expand mask to account for CLS token
            cls_mask = torch.ones(batch_size, 1, device=hidden_states.device)
            attention_mask = torch.cat((cls_mask, attention_mask), dim=1)

        if use_cache:
            if past_key_values is None:
                past_key_values = [layer.attn.init_state(batch_size) for layer in self.layers]
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        if self.config.use_lower_bound:
            lower_bounds = self.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            lower_bound = lower_bounds[i] if self.config.use_lower_bound else None
            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    lower_bound,
                    timestep,
                    label,
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    lower_bound=lower_bound,
                    timestep=timestep,
                    label=label,
                )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Remove CLS token and project to noise
        hidden_states = hidden_states[:, 1:, :]  # (batch_size, num_patches, hidden_size)
        noise = self.final_layer(hidden_states)  # (batch_size, num_patches, in_channels * patch_size * patch_size)
        noise = noise.view(batch_size, self.num_patches, self.in_channels, self.patch_size, self.patch_size)
        noise = noise.permute(0, 2, 1, 3, 4)  # (batch_size, in_channels, num_patches, patch_size, patch_size)
        # Reshape to (batch_size, in_channels, latent_size, latent_size)
        noise = noise.view(batch_size, self.in_channels, self.latent_size, self.latent_size)

        next_cache = None
        if use_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(x for x in [noise, next_cache, all_hidden_states, all_attns] if x is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=noise,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )