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

from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.utils import RecurrentCache
from mmfreelm.modules import FusedCrossEntropyLoss, RMSNorm
from mmfreelm.modules.activations import swiglu_linear, swiglu
#from mmfreelm.ops.bitnet import BitLinear_Fuse as BitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear
from mmfreelm.models.hgrn_bit.hgrn_bit_moe import HGRNBitMoE
logger = logging.get_logger(__name__)


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


        # Modified MLP initialization
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
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            lower_bound=lower_bound
        )
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)

        return outputs


class HGRNBitPreTrainedModel(PreTrainedModel):

    config_class = HGRNBitConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['HGRNBitBlock']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d, BitLinear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class HGRNBitModel(HGRNBitPreTrainedModel):

    def __init__(self, config: HGRNBitConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))
        self.layers = nn.ModuleList([HGRNBitBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn("`HGRNBitModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

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
                    lower_bound
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    lower_bound=lower_bound
                )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
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
            attentions=all_attns
        )


class HGRNBitForCausalLM(HGRNBitPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = HGRNBitModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = BitLinear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is passed along.
        if past_key_values is not None:
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(past_key_values, input_ids.shape[1] - 1)
            input_ids, attention_mask = input_ids[:, -1:], attention_mask[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        })
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if self.config.fuse_cross_entropy:
                loss_fct = FusedCrossEntropyLoss(inplace_backward=True)
            else:
                loss_fct = nn.CrossEntropyLoss()
            # Enable model parallelism
            labels = labels.to(logits.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)), 1)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
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
        x = self.mlp(condition)
        x = self.norm(x)
        params = self.out_proj(x)
        scale, shift = params.chunk(2, dim=-1)
        return scale, shift
    

class TerneryDit(nn.Module):
    """
    Ternary Diffusion Transformer (TerneryDit) for image generation conditioned on text.
    Integrates a VAE, diffusion scheduler, patch embedding, AdaLN conditioning,
    and an HGRNBitModel transformer to produce image latents from text.
    """
    def __init__(self, 
                 vae_encoder: nn.Module,
                 vae_decoder: nn.Module,
                 noise_scheduler,
                 transformer_config,
                 text_embed_dim: int,
                 patch_size: int = 3,
                 embed_dim: int = 768,
                 in_channels: int = 4):
        super().__init__()
        # VAE encoder/decoder for latent space
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder

        # Noise scheduler for the diffusion process
        self.noise_scheduler = noise_scheduler

        # Patch embedding: conv layer to embed 3×3 patches into embed_dim tokens
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        # Compute number of patches (for 64×64 latent)
        self.num_patches_per_dim = 64 // patch_size
        num_patches = self.num_patches_per_dim ** 2
        # Positional embeddings for each patch token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # AdaLNConditioning: maps text embedding to per-layer scale/shift parameters
        num_layers = transformer_config.num_hidden_layers
        hidden_dim = transformer_config.hidden_size
        self.ada_ln = AdaLNConditioning(text_embed_dim, num_layers, hidden_dim)

        # Core transformer model (HGRNBitModel) that uses AdaLN inside its blocks
        self.transformer = HGRNBitModel(transformer_config)

        # Inverse patch embedding: ConvTranspose to reconstruct the latent
        self.depatch = nn.ConvTranspose2d(embed_dim, in_channels, 
                                          kernel_size=patch_size, stride=patch_size,
                                          output_padding=1)

    def forward(self, image: torch.Tensor, text_embedding: torch.Tensor, timesteps: torch.Tensor):
        """
        Forward pass: Encode image, add noise, apply transformer with text conditioning, and reconstruct latent.
        
        Args:
            image (torch.Tensor): [B, 3, 512, 512] input images.
            text_embedding (torch.Tensor): [B, T, D_text] precomputed text embeddings.
            timesteps (torch.Tensor): [B] diffusion timesteps for noise scheduling.
        
        Returns:
            torch.Tensor: Generated latent [B, 4, 64, 64] (to be decoded by the VAE decoder).
        """
        # Encode input image to latent space (4×64×64)
        latents = self.vae_encoder(image)  # [B, 4, 64, 64]
        # (Optionally scale latents, e.g., *0.18215, as used in some diffusion setups)
        # latents = latents * 0.18215
        
        # Add diffusion noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Patch embedding: conv + flatten to sequence of tokens
        x = self.patch_embed(noisy_latents)          # [B, embed_dim, H_p, W_p]
        B, C, H_p, W_p = x.shape
        x = x.view(B, C, H_p * W_p).transpose(1, 2)  # [B, N_tokens, embed_dim]
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Compute AdaLN scale/shift parameters from text embedding
        cond_params = self.ada_ln(text_embedding)   # returned params for each transformer layer
        
        # Core transformer forward (AdaLN conditioning is applied inside)
        out_tokens = self.transformer(x, cond_params)  # [B, N_tokens, embed_dim]
        
        # Reshape transformer output tokens back to feature map
        B, N, D = out_tokens.shape
        # Assuming square layout of tokens
        H_p = W_p = int(N ** 0.5)
        feat_map = out_tokens.transpose(1, 2).view(B, D, H_p, W_p)  # [B, embed_dim, H_p, W_p]
        
        # Reconstruct latent via inverse patch embedding
        latents_out = self.depatch(feat_map)  # [B, 4, 64, 64]
        return latents_out