# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.utils import RecurrentCache
from mmfreelm.modules import FusedCrossEntropyLoss, RMSNorm
from mmfreelm.modules.activations import swiglu
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear

logger = logging.get_logger(__name__)


class TiedLMHead(nn.Module):
    """
    Vocabulary projection whose weight is physically shared with an embedding.

    The tied path intentionally remains floating point during training. It
    avoids a second vocabulary-size shadow weight and protects output-logit
    quality; large internal HGRN and MLP matrices remain ternary-QAT layers.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    embedding.num_embeddings,
                    dtype=embedding.weight.dtype,
                    device=embedding.weight.device,
                )
            )
        else:
            self.register_parameter("bias", None)

        self.embedding = embedding

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight

    @property
    def in_features(self) -> int:
        return self.embedding.embedding_dim

    @property
    def out_features(self) -> int:
        return self.embedding.num_embeddings

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(hidden_states, self.weight, self.bias)


class HGRNBitMLP(nn.Module):
    """
    SwiGLU feed-forward network with groupwise ternary QAT projections.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "swish",
        weight_group_size: int = 128,
        weight_scale_method: str = "mean_abs",
        activation_bits: int = 8,
        activation_group_size: Optional[int] = None,
        quantization_enabled: bool = True,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive.")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        linear_kwargs = {
            "weight_group_size": weight_group_size,
            "weight_scale_method": weight_scale_method,
            "activation_bits": activation_bits,
            "activation_group_size": activation_group_size,
            "quantization_enabled": quantization_enabled,
        }

        self.gate_proj = BitLinear(
            hidden_size,
            intermediate_size * 2,
            bias=False,
            **linear_kwargs,
        )
        self.down_proj = BitLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            **linear_kwargs,
        )

    def set_quantization_enabled(self, enabled: bool) -> None:
        self.gate_proj.set_quantization_enabled(enabled)
        self.down_proj.set_quantization_enabled(enabled)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_and_value = self.gate_proj(hidden_states)
        gate, value = gate_and_value.chunk(2, dim=-1)
        return self.down_proj(swiglu(gate, value))


class HGRNBitBlock(nn.Module):
    """
    One HGRN recurrent mixer followed by either a dense SwiGLU MLP or an MoE MLP.
    """

    def __init__(
        self,
        config: HGRNBitConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe_layer = config.is_moe_layer(layer_idx)

        self.attn = HGRNBitAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_ratio=config.expand_ratio,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            conv_bias=config.conv_bias,
            share_conv_kernel=config.share_conv_kernel,
            layernorm_eps=config.rms_norm_eps,
            layer_idx=layer_idx,
            rotary_embeddings=config.rotary_embeddings,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            use_ternary_rope=config.use_ternary_rope,
            optimized_bitlinear=True,
            full_precision=False,
            weight_group_size=config.weight_quant_group_size,
            weight_scale_method=config.weight_quant_scale,
            activation_bits=config.activation_quant_bits,
            activation_group_size=config.activation_quant_group_size,
            quantization_enabled=config.quantization_warmup_steps == 0,
            decay_mode=config.decay_mode,
            decay_init=config.decay_init,
        )

        # Normalization lives inside BitLinear, which rescales its own input to
        # RMS 1. Without these, a block's output magnitude is unrelated to the
        # magnitude of the residual stream it writes into.
        self.attn_norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.mlp_norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        if self.is_moe_layer:
            from mmfreelm.models.hgrn_bit.hgrn_bit_moe import HGRNBitMoE
            self.mlp = HGRNBitMoE(config)
        else:
            self.mlp = HGRNBitMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                weight_group_size=config.weight_quant_group_size,
                weight_scale_method=config.weight_quant_scale,
                activation_bits=config.activation_quant_bits,
                activation_group_size=config.activation_quant_group_size,
                quantization_enabled=config.quantization_warmup_steps == 0,
            )

    def set_quantization_enabled(self, enabled: bool) -> None:
        self.attn.set_quantization_enabled(enabled)

        set_enabled = getattr(self.mlp, "set_quantization_enabled", None)
        if set_enabled is not None:
            set_enabled(enabled)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[RecurrentCache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[RecurrentCache],
    ]:
        del kwargs

        residual = hidden_states

        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            lower_bound=lower_bound,
        )

        residual = residual + hidden_states
        hidden_states = residual + self.mlp(self.mlp_norm(residual))

        return hidden_states, attentions, past_key_values


class HGRNBitPreTrainedModel(PreTrainedModel):
    config_class = HGRNBitConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["HGRNBitBlock"]

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ) -> None:
        """
        Initialize modules not already initialized by HGRN submodules.
        """
        if getattr(module, "_is_hf_initialized", False):
            return

        if isinstance(module, (nn.Linear, BitLinear)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

            module._is_hf_initialized = True

        elif isinstance(module, nn.Conv1d):
            # ShortConvolution subclasses nn.Conv1d. std=0.02 on a depthwise
            # k=4 kernel outputs ~0.04x its input; keep PyTorch's default init.
            module._is_hf_initialized = True

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

            module._is_hf_initialized = True

        if rescale_prenorm_residual:
            # Recurse one level: with recurse=False, `name` is always "weight"
            # and the rescale never fired. The match stays exact so it applies
            # once, at the immediate parent (HGRNBitAttention / HGRNBitMLP),
            # rather than again at every ancestor module.
            for name, parameter in module.named_parameters():
                if name in {"o_proj.weight", "down_proj.weight"}:
                    with torch.no_grad():
                        parameter.div_(
                            math.sqrt(
                                num_residuals_per_layer
                                * self.config.num_hidden_layers
                            )
                        )

    def set_quantization_enabled(self, enabled: bool) -> None:
        """
        Toggle QAT across every HGRN and MLP projection in the model.
        """
        for module in self.modules():
            if isinstance(module, HGRNBitBlock):
                module.set_quantization_enabled(enabled)


class HGRNBitModel(HGRNBitPreTrainedModel):
    """
    HGRN-Bit decoder backbone without the language-model projection.
    """

    def __init__(
        self,
        config: HGRNBitConfig,
    ) -> None:
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )

        if (
            config.use_lower_bound
            and config.decay_mode == "global_lower_bound"
        ):
            self.lower_bounds = nn.Parameter(
                torch.zeros(
                    config.num_hidden_layers,
                    int(config.hidden_size * config.expand_ratio),
                )
            )
        else:
            self.register_parameter("lower_bounds", None)

        self.layers = nn.ModuleList(
            [
                HGRNBitBlock(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings = value

    def _global_lower_bounds(self) -> Optional[torch.Tensor]:
        """
        Return original compatible lower bounds only when that decay mode is used.
        """
        if self.lower_bounds is None:
            return None

        lower_bounds = self.lower_bounds.softmax(dim=0)
        return lower_bounds.cumsum(dim=0) - lower_bounds[0]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[
            Union[RecurrentCache, Tuple[List[torch.Tensor]]]
        ] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn(
                "`HGRNBitModel` does not expose attention matrices; "
                "setting output_attentions=False.",
                stacklevel=2,
            )
            output_attentions = False

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "Specify exactly one of input_ids and inputs_embeds."
            )

        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "Specify one of input_ids or inputs_embeds."
            )

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache and not self.training)
        )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # Lift the embedding to unit RMS so the first block writes into a
        # stream of comparable magnitude instead of a 0.02-scale one.
        hidden_states = inputs_embeds * math.sqrt(self.config.hidden_size)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing; "
                "setting use_cache=False."
            )
            use_cache = False

        if use_cache:
            if past_key_values is None:
                initial_states = [
                    layer.attn.init_state(batch_size)
                    for layer in self.layers
                ]
                past_key_values = RecurrentCache.from_legacy_cache(
                    initial_states
                )
            elif not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(
                    past_key_values
                )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        lower_bounds = self._global_lower_bounds()

        total_aux_loss = hidden_states.new_zeros(())

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            lower_bound = (
                lower_bounds[layer_idx]
                if lower_bounds is not None
                else None
            )

            if self.gradient_checkpointing and self.training:
                def custom_forward(
                    hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor],
                    lower_bound: Optional[torch.Tensor],
                ) -> torch.Tensor:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        past_key_values=None,
                        use_cache=False,
                        output_attentions=False,
                        lower_bound=lower_bound,
                    )
                    return layer_outputs[0]

                hidden_states = self._gradient_checkpointing_func(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    lower_bound,
                )
                attentions = None
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
                all_attentions += (attentions,)

            aux_loss = getattr(layer.mlp, "aux_loss", None)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = (
            past_key_values.to_legacy_cache()
            if use_cache and past_key_values is not None
            else None
        )

        if not return_dict:
            outputs = (
                hidden_states,
                next_cache,
                all_hidden_states,
                all_attentions,
            )
            return outputs + (total_aux_loss,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
        output.aux_loss = total_aux_loss
        return output


class HGRNBitForCausalLM(HGRNBitPreTrainedModel, GenerationMixin):
    """
    HGRN-Bit causal language model.
    """

    _tied_weights_keys = {"lm_head.embedding.weight": "model.embeddings.weight",}
    def __init__(
        self,
        config: HGRNBitConfig,
    ) -> None:
        super().__init__(config)

        self.model = HGRNBitModel(config)
        self.vocab_size = config.vocab_size

        if config.tie_word_embeddings:
            self.lm_head = TiedLMHead(
                self.model.embeddings,
                bias=False,
            )
        else:
            self.lm_head = BitLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                weight_group_size=config.weight_quant_group_size,
                weight_scale_method=config.weight_quant_scale,
                activation_bits=config.activation_quant_bits,
                activation_group_size=config.activation_quant_group_size,
                quantization_enabled=config.quantization_warmup_steps == 0,
            )

        self.post_init()
        self.tie_weights()

    def tie_weights(
        self,
        missing_keys=None,
        recompute_mapping: bool = True,
        **kwargs,
    ) -> None:
        """
        Restore the shared input-embedding / LM-head storage.

        Transformers may pass missing_keys during from_pretrained. It is
        expected that lm_head.embedding.weight is absent because it aliases
        model.embeddings.weight and is reattached below.
        """
        del missing_keys, recompute_mapping, kwargs

        if not self.config.tie_word_embeddings:
            return

        if not isinstance(self.lm_head, TiedLMHead):
            self.lm_head = TiedLMHead(
                self.model.embeddings,
                bias=False,
            )
        else:
            self.lm_head.embedding = self.model.embeddings

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embeddings = value

        if self.config.tie_word_embeddings:
            self.tie_weights()

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if self.config.tie_word_embeddings:
            if isinstance(new_embeddings, TiedLMHead):
                self.lm_head = new_embeddings
            elif isinstance(new_embeddings, nn.Embedding):
                self.model.embeddings = new_embeddings
                self.lm_head = TiedLMHead(new_embeddings, bias=False)
            else:
                raise TypeError(
                    "A model with tie_word_embeddings=True requires "
                    "TiedLMHead or nn.Embedding as output embeddings."
                )
            self.tie_weights()
        else:
            self.lm_head = new_embeddings

    def set_decoder(self, decoder: HGRNBitModel) -> None:
        self.model = decoder

        if self.config.tie_word_embeddings:
            self.tie_weights()

    def get_decoder(self) -> HGRNBitModel:
        return self.model

    def set_quantization_enabled(self, enabled: bool) -> None:
        """
        Toggle internal projection QAT.

        The tied vocabulary head remains high precision by design. An untied
        BitLinear vocabulary head follows the same QAT setting as the backbone.
        """
        self.model.set_quantization_enabled(enabled)

        set_enabled = getattr(self.lm_head, "set_quantization_enabled", None)
        if set_enabled is not None:
            set_enabled(enabled)

    def generate(
        self,
        *args,
        **kwargs,
    ):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if "past_key_values" in str(exception):
                raise AttributeError(
                    "The selected generation strategy is not compatible with "
                    f"{self.__class__.__name__}'s recurrent cache. Use a "
                    "standard greedy, sampling, or beam generation mode."
                ) from exception
            raise

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[
            Union[RecurrentCache, Tuple[List[torch.Tensor]]]
        ] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "input_ids or inputs_embeds is required for generation."
            )

        if past_key_values is not None:
            if not isinstance(past_key_values, RecurrentCache):
                if input_ids is None:
                    raise ValueError(
                        "input_ids is required when converting a legacy cache."
                    )

                past_key_values = RecurrentCache.from_legacy_cache(
                    past_key_values,
                    input_ids.shape[1] - 1,
                )

            if input_ids is not None:
                input_ids = input_ids[:, -1:].contiguous()

            if attention_mask is not None:
                attention_mask = attention_mask[:, -1:].contiguous()

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {
                "input_ids": (
                    input_ids.contiguous()
                    if input_ids is not None
                    else None
                )
            }

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            Union[RecurrentCache, Tuple[List[torch.Tensor]]]
        ] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            hidden_states = outputs.last_hidden_state
            aux_loss = getattr(outputs, "aux_loss", None)
        else:
            hidden_states = outputs[0]
            aux_loss = outputs[-1]

        logits = self.lm_head(hidden_states)
        loss = None

        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.fuse_cross_entropy:
                loss_function = FusedCrossEntropyLoss(
                    inplace_backward=True
                )
            else:
                loss_function = nn.CrossEntropyLoss()

            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()

            loss = loss_function(
                shifted_logits.reshape(-1, self.config.vocab_size),
                shifted_labels.reshape(-1),
            )

            if aux_loss is not None:
                loss = loss + (
                    self.config.router_aux_loss_coef * aux_loss
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        causal_output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        causal_output.aux_loss = aux_loss
        return causal_output