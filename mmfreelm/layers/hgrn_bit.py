# -*- coding: utf-8 -*-
# "HGRN2: Gated Linear RNNs with State Expansion"
# https://arxiv.org/abs/2404.07904

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import Cache

from mmfreelm.models.hgrn_bit.rotary_embedding import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from mmfreelm.modules import FusedRMSNormSwishGate, ShortConvolution
from mmfreelm.modules.activations import swiglu
from mmfreelm.ops.bitnet import BitLinear as StandardBitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear
from mmfreelm.ops.hgrn.recurrent_fuse import fused_recurrent_hgrn


class HGRNBitAttention(nn.Module):
    """
    HGRN recurrent mixing block with optional causal short convolution.

    The recurrent state has fixed shape [batch, num_heads, head_dim], so unlike
    standard self-attention it does not create a KV cache that grows with the
    generated sequence length.
    """

    def __init__(
        self,
        mode: str = "fused_recurrent",
        hidden_size: int = 1024,
        num_heads: Optional[int] = None,
        expand_ratio: Union[int, float] = 1,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        share_conv_kernel: bool = True,
        layernorm_eps: float = 1e-5,
        layer_idx: Optional[int] = None,
        rotary_embeddings: bool = False,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        use_ternary_rope: bool = False,
        optimized_bitlinear: bool = True,
        full_precision: bool = False,
        weight_group_size: int = 128,
        weight_scale_method: str = "mean_abs",
        activation_bits: int = 8,
        activation_group_size: Optional[int] = None,
        quantization_enabled: bool = True,
        decay_mode: str = "independent",
        decay_init: float = 0.5,
    ) -> None:
        super().__init__()

        if mode != "fused_recurrent":
            raise ValueError(
                "Only mode='fused_recurrent' is supported; "
                f"received mode='{mode}'."
            )

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        if expand_ratio <= 0:
            raise ValueError("expand_ratio must be positive.")

        if conv_size <= 0:
            raise ValueError("conv_size must be positive.")

        if layernorm_eps <= 0:
            raise ValueError("layernorm_eps must be positive.")

        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive.")

        if decay_mode not in {"independent", "global_lower_bound"}:
            raise ValueError(
                "decay_mode must be either 'independent' or "
                "'global_lower_bound'."
            )

        if not 0.0 < decay_init < 1.0:
            raise ValueError("decay_init must be strictly between 0 and 1.")

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_ratio = expand_ratio
        self.input_dim = int(hidden_size * expand_ratio)

        if self.input_dim <= 0:
            raise ValueError("hidden_size * expand_ratio must be positive.")

        if num_heads is None:
            num_heads = max(1, self.input_dim // 64)

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if self.input_dim % num_heads != 0:
            raise ValueError(
                "input_dim must be divisible by num_heads; "
                f"got input_dim={self.input_dim}, num_heads={num_heads}."
            )

        self.num_heads = num_heads
        self.head_dim = self.input_dim // self.num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.share_conv_kernel = share_conv_kernel
        self.layer_idx = layer_idx

        self.decay_mode = decay_mode
        self.decay_init = decay_init

        if full_precision:
            linear_cls = nn.Linear
            linear_kwargs = {}
        else:
            linear_cls = (
                FusedBitLinear if optimized_bitlinear else StandardBitLinear
            )
            linear_kwargs = {
                "weight_group_size": weight_group_size,
                "weight_scale_method": weight_scale_method,
                "activation_bits": activation_bits,
                "activation_group_size": activation_group_size,
                "quantization_enabled": quantization_enabled,
            }

        self.i_proj = linear_cls(
            hidden_size,
            self.input_dim,
            bias=False,
            **linear_kwargs,
        )
        self.f_proj = linear_cls(
            hidden_size,
            self.input_dim,
            bias=False,
            **linear_kwargs,
        )
        self.g_proj = linear_cls(
            hidden_size,
            self.input_dim,
            bias=False,
            **linear_kwargs,
        )

        if self.use_short_conv:
            if self.share_conv_kernel:
                self.h_conv1d = ShortConvolution(
                    hidden_size,
                    conv_size,
                    bias=conv_bias,
                    activation="silu",
                )
            else:
                self.i_conv1d = ShortConvolution(
                    self.input_dim,
                    conv_size,
                    bias=conv_bias,
                    activation="silu",
                )
                self.f_conv1d = ShortConvolution(
                    self.input_dim,
                    conv_size,
                    bias=conv_bias,
                    activation="silu",
                )

        self.g_norm = FusedRMSNormSwishGate(
            self.input_dim,
            layernorm_eps,
        )
        self.o_proj = linear_cls(
            self.input_dim,
            hidden_size,
            bias=False,
            **linear_kwargs,
        )

        if self.decay_mode == "independent":
            initial_logit = math.log(decay_init / (1.0 - decay_init))
            self.decay_logit = nn.Parameter(
                torch.full(
                    (self.num_heads, self.head_dim),
                    initial_logit,
                )
            )
        else:
            self.register_parameter("decay_logit", None)

        self.rotary_embeddings = rotary_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_ternary_rope = use_ternary_rope

        if self.rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
                use_ternary=use_ternary_rope,
            )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if getattr(module, "_is_hf_initialized", False):
            return

        if isinstance(module, (nn.Linear, StandardBitLinear, FusedBitLinear)):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

            module._is_hf_initialized = True

    def set_quantization_enabled(self, enabled: bool) -> None:
        """
        Toggle fake quantization for all ternary projection layers.

        This supports a full-precision warm-up followed by ternary QAT without
        reconstructing the model.
        """
        for module in (self.i_proj, self.f_proj, self.g_proj, self.o_proj):
            set_enabled = getattr(module, "set_quantization_enabled", None)
            if set_enabled is not None:
                set_enabled(enabled)

    def _get_last_state(
        self,
        past_key_values: Optional[Cache],
        use_cache: bool,
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        if not use_cache or past_key_values is None:
            return None

        if self.layer_idx is None:
            raise ValueError(
                "layer_idx must be set when use_cache=True."
            )

        return past_key_values[self.layer_idx]

    def _apply_decay(
        self,
        decay: torch.Tensor,
        lower_bound: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute bounded recurrent decays.

        Independent mode gives each recurrent channel a learned baseline and
        prevents cross-layer coupling. Global-lower-bound mode preserves
        compatibility with the original model-level lower-bound mechanism.
        """
        if self.decay_mode == "independent":
            learned_lower_bound = torch.sigmoid(self.decay_logit).to(
                dtype=decay.dtype
            )
            learned_lower_bound = learned_lower_bound.unsqueeze(0).unsqueeze(2)
            return learned_lower_bound + (1.0 - learned_lower_bound) * decay

        if lower_bound is not None and self.layer_idx is not None and self.layer_idx > 0:
            if lower_bound.shape[-1] != self.input_dim:
                raise ValueError(
                    "lower_bound final dimension must match input_dim; "
                    f"got {lower_bound.shape[-1]} and {self.input_dim}."
                )

            lower_bound = rearrange(
                lower_bound,
                "(h d) -> 1 h 1 d",
                h=self.num_heads,
                d=self.head_dim,
            ).to(dtype=decay.dtype)

            return lower_bound + (1.0 - lower_bound) * decay

        return decay

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """
        Run recurrent HGRN mixing.

        `output_attentions` is accepted for Hugging Face compatibility but HGRN
        does not produce an attention matrix.
        """
        del output_attentions, kwargs

        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden]; "
                f"received shape={tuple(hidden_states.shape)}."
            )

        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}; "
                f"received {hidden_states.shape[-1]}."
            )

        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "attention_mask must have shape [batch, sequence]."
                )

            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    "attention_mask must match hidden_states batch/sequence "
                    f"dimensions; got {tuple(attention_mask.shape)} and "
                    f"{tuple(hidden_states.shape[:2])}."
                )

        mode = (
            "fused_recurrent"
            if hidden_states.shape[1] == 1
            else self.mode
        )
        last_state = self._get_last_state(past_key_values, use_cache)

        conv_state = None
        conv_state_i = None
        conv_state_f = None

        if self.use_short_conv:
            if self.share_conv_kernel:
                conv_state = last_state[0] if last_state is not None else None
                hidden_states = self.h_conv1d(
                    hidden_states,
                    cache=conv_state,
                )
                i = self.i_proj(hidden_states)
                f = self.f_proj(hidden_states)
            else:
                conv_state_i = (
                    last_state[0] if last_state is not None else None
                )
                conv_state_f = (
                    last_state[1] if last_state is not None else None
                )

                i = self.i_conv1d(
                    self.i_proj(hidden_states),
                    cache=conv_state_i,
                )
                f = self.f_conv1d(
                    self.f_proj(hidden_states),
                    cache=conv_state_f,
                )
        else:
            i = self.i_proj(hidden_states)
            f = self.f_proj(hidden_states)

        g = self.g_proj(hidden_states)
        f = torch.sigmoid(f)
        if attention_mask is not None:
            valid_tokens = attention_mask.to(dtype=i.dtype).unsqueeze(-1)
            # A padded token must neither write new content nor decay the prior state.
            i = i * valid_tokens
            g = g * valid_tokens
            f = f * valid_tokens + (1.0 - valid_tokens)

        i = rearrange(
            i,
            "b l (h d) -> b h l d",
            h=self.num_heads,
        )
        f = rearrange(
            f,
            "b l (h d) -> b h l d",
            h=self.num_heads,
        )
        g = rearrange(
            g,
            "b l (h d) -> b h l d",
            h=self.num_heads,
        )

        f = self._apply_decay(f, lower_bound)
        i = swiglu(i, 1.0 - f)

        if self.rotary_embeddings:
            sequence_length = i.shape[2]
            cos, sin = self.rotary_emb(
                i,
                seq_len=sequence_length,
            )
            i, g = apply_rotary_pos_emb(i, g, cos, sin)

        recurrent_state = last_state[-1] if last_state is not None else None

        if mode != "fused_recurrent":
            raise NotImplementedError(
                f"Not supported mode '{mode}'."
            )

        o, recurrent_state = fused_recurrent_hgrn(
            i,
            f,
            initial_state=recurrent_state,
            output_final_state=use_cache,
        )

        if past_key_values is not None and use_cache:
            if self.layer_idx is None:
                raise ValueError(
                    "layer_idx must be set when updating past_key_values."
                )

            if self.use_short_conv:
                if self.share_conv_kernel:
                    next_state = (conv_state, recurrent_state)
                else:
                    next_state = (
                        conv_state_i,
                        conv_state_f,
                        recurrent_state,
                    )
            else:
                next_state = (recurrent_state,)

            past_key_values.update(
                next_state,
                self.layer_idx,
                i.shape[2],
            )

        o = rearrange(o, "b h l d -> b l (h d)")
        g = rearrange(g, "b h l d -> b l (h d)")

        o = self.g_norm(g, o)
        o = self.o_proj(o)

        return o, None, past_key_values

    def init_state(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Allocate fixed-size inference state for one HGRN layer.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        parameter = next(self.parameters())
        state = []

        if self.use_short_conv:
            if self.share_conv_kernel:
                state.append(
                    parameter.new_zeros(
                        batch_size,
                        self.hidden_size,
                        self.conv_size,
                    )
                )
            else:
                state.extend(
                    [
                        parameter.new_zeros(
                            batch_size,
                            self.input_dim,
                            self.conv_size,
                        ),
                        parameter.new_zeros(
                            batch_size,
                            self.input_dim,
                            self.conv_size,
                        ),
                    ]
                )

        state.append(
            parameter.new_zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
            )
        )

        return tuple(state)

    def state_size(self, **kwargs) -> int:
        """
        Return per-example recurrent-state element count.

        The result is independent of generated sequence length.
        """
        del kwargs

        state_size = self.num_heads * self.head_dim

        if self.use_short_conv:
            if self.share_conv_kernel:
                state_size += self.hidden_size * self.conv_size
            else:
                state_size += 2 * self.input_dim * self.conv_size

        return state_size