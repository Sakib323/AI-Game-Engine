# -*- coding: utf-8 -*-
"""
HGRN-2 style token mixer with a matrix-valued recurrent state.

Following HGRN-2, the key is tied to the decay as k = 1 - f, so no k_proj is
needed. A q_proj IS needed -- the diagonal form had no query.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from mmfreelm.modules import FusedRMSNormSwishGate, ShortConvolution
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear
from mmfreelm.ops.gla import chunk_gla_torch

try:
    from fla.ops.gla import chunk_gla as _chunk_gla_triton
except Exception:
    _chunk_gla_triton = None


class HGRN2BitAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        expand_ratio: float = 1.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        decay_mode: str = "independent",
        decay_low_half_life: float = 8.0,
        decay_high_half_life: float = 2048.0,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        chunk_size: int = 64,
        use_triton_kernel: bool = True,
        layer_idx: Optional[int] = None,
        weight_group_size: int = 128,
        weight_scale_method: str = "mean_abs",
        activation_bits: int = 8,
        activation_group_size: Optional[int] = None,
        quantization_enabled: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.input_dim = int(hidden_size * expand_ratio)
        self.layer_idx = layer_idx
        self.chunk_size = chunk_size

        if self.input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim {self.input_dim} must divide by num_heads {num_heads}"
            )
        self.head_dim = self.input_dim // num_heads

        if decay_mode not in {"independent", "global_lower_bound"}:
            raise ValueError("decay_mode must be 'independent' or 'global_lower_bound'")
        self.decay_mode = decay_mode

        self.use_triton_kernel = use_triton_kernel and _chunk_gla_triton is not None

        linear_kwargs = dict(
            bias=False,
            weight_group_size=weight_group_size,
            weight_scale_method=weight_scale_method,
            activation_bits=activation_bits,
            activation_group_size=activation_group_size,
            quantization_enabled=quantization_enabled,
        )

        self.q_proj = BitLinear(hidden_size, self.input_dim, **linear_kwargs)
        self.i_proj = BitLinear(hidden_size, self.input_dim, **linear_kwargs)
        self.f_proj = BitLinear(hidden_size, self.input_dim, **linear_kwargs)
        self.g_proj = BitLinear(hidden_size, self.input_dim, **linear_kwargs)
        self.o_proj = BitLinear(self.input_dim, hidden_size, **linear_kwargs)

        self.f_bias = nn.Parameter(torch.zeros(self.input_dim))

        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.h_conv1d = ShortConvolution(
                hidden_size, conv_size, activation="silu", bias=conv_bias,
            )

        # This build's FusedRMSNormSwishGate is always affine and takes only
        # (hidden_size, eps); elementwise_affine is accepted for API parity
        # with the spec and has no non-default setting to forward.
        self.elementwise_affine = elementwise_affine
        self.g_norm = FusedRMSNormSwishGate(self.input_dim, eps=norm_eps)

        if decay_mode == "independent":
            self.decay_logit = nn.Parameter(
                self._log_spaced_decay_logits(
                    decay_low_half_life, decay_high_half_life,
                )
            )
        else:
            self.register_parameter("decay_logit", None)

        self.scale = self.head_dim ** -0.5

    def _log_spaced_decay_logits(self, low: float, high: float) -> torch.Tensor:
        """Per-channel decay bounds spread log-uniformly over half-lives.

        Long-range paths must EXIST at step 0. In the diagonal model decay_logit
        never moved from its init across any run -- there is no coherent
        gradient telling a model to lengthen memory it is not already using.
        """
        half_lives = torch.exp(
            torch.linspace(math.log(low), math.log(high), self.head_dim)
        )
        gates = torch.exp(math.log(0.5) / half_lives).clamp(1e-4, 1 - 1e-4)
        logits = torch.log(gates / (1.0 - gates))
        return logits.unsqueeze(0).repeat(self.num_heads, 1).contiguous()

    def set_quantization_enabled(self, enabled: bool) -> None:
        for module in (self.q_proj, self.i_proj, self.f_proj,
                       self.g_proj, self.o_proj):
            module.set_quantization_enabled(enabled)

    def _decay(self, f: torch.Tensor, lower_bound: Optional[torch.Tensor]):
        """Map sigmoid output into (lower_bound, 1). Returns LINEAR decay."""
        if self.decay_mode == "independent":
            bound = torch.sigmoid(self.decay_logit).to(f.dtype)
            bound = bound.unsqueeze(0).unsqueeze(2)
            return bound + (1.0 - bound) * f

        if lower_bound is None:
            return f
        bound = lower_bound.to(f.dtype)
        return bound + (1.0 - bound) * f

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        conv_state = None
        recurrent_state = None
        if past_key_values is not None and len(past_key_values) > 0:
            conv_state, recurrent_state = past_key_values[0], past_key_values[1]

        if self.use_short_conv:
            hidden_states = self.h_conv1d(hidden_states, conv_state)

        q = self.q_proj(hidden_states)
        v = self.i_proj(hidden_states)
        f = self.f_proj(hidden_states)
        g = self.g_proj(hidden_states)

        f = torch.sigmoid(f + self.f_bias)

        if attention_mask is not None:
            valid = attention_mask.to(dtype=v.dtype).unsqueeze(-1)
            q = q * valid
            v = v * valid
            g = g * valid
            f = f * valid + (1.0 - valid)      # pads: decay 1 -> k = 0, no write

        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)
        f = rearrange(f, "b l (h d) -> b h l d", h=self.num_heads)

        f = self._decay(f, lower_bound)
        k = 1.0 - f
        log_f = torch.log(f.float().clamp_min(1e-7))

        q = q * self.scale

        if seq_len == 1 and recurrent_state is not None:
            decay = f[:, :, 0].float().unsqueeze(-1)
            update = (
                k[:, :, 0].float().unsqueeze(-1)
                @ v[:, :, 0].float().unsqueeze(-2)
            )
            recurrent_state = decay * recurrent_state.float() + update
            o = (q[:, :, 0].float().unsqueeze(-2) @ recurrent_state).squeeze(-2)
            o = o.unsqueeze(2).to(hidden_states.dtype)
        elif self.use_triton_kernel:
            o, recurrent_state = _chunk_gla_triton(
                q, k, v, log_f,
                scale=1.0,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        else:
            pad = (-seq_len) % self.chunk_size
            if pad:
                q, k, v, log_f = (
                    torch.nn.functional.pad(t, (0, 0, 0, pad))
                    for t in (q, k, v, log_f)
                )
                log_f[:, :, seq_len:] = 0.0
            o, recurrent_state = chunk_gla_torch(
                q, k, v, log_f,
                chunk_size=self.chunk_size,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
            if pad:
                o = o[:, :, :seq_len]

        o = rearrange(o.to(hidden_states.dtype), "b h l d -> b l (h d)")

        # RMSNorm the RECURRENT OUTPUT, gate with the current token.
        # The reverse order makes history a channel-wise amplitude mask with no
        # content path -- that was the diagonal model's failure.
        o = self.g_norm(o, g)
        o = self.o_proj(o)

        new_cache = (conv_state, recurrent_state) if use_cache else None
        return o, None, new_cache

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        param = next(self.parameters())
        conv_state = param.new_zeros(
            batch_size, self.hidden_size, self.h_conv1d.kernel_size[0]
        ) if self.use_short_conv else None
        recurrent_state = param.new_zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim
        )
        return conv_state, recurrent_state

    @property
    def state_size(self) -> int:
        conv = (
            self.hidden_size * self.h_conv1d.kernel_size[0]
            if self.use_short_conv else 0
        )
        return conv + self.num_heads * self.head_dim * self.head_dim
