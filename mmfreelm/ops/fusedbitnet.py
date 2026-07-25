# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmfreelm.modules import RMSNorm
from mmfreelm.ops.bitnet import (
    DEFAULT_ACTIVATION_BITS,
    DEFAULT_WEIGHT_GROUP_SIZE,
    activation_quant,
    weight_quant,
)


def _validate_activation_bits(bits: int) -> None:
    if bits not in (4, 8, 16):
        raise ValueError(
            f"activation_bits must be one of 4, 8, or 16; received {bits}."
        )


def _validate_weight_quantization(
    group_size: int,
    scale_method: str,
) -> None:
    if group_size <= 0:
        raise ValueError(
            f"weight_group_size must be positive; received {group_size}."
        )

    if scale_method not in ("mean_abs", "rms"):
        raise ValueError(
            "weight_scale_method must be either 'mean_abs' or 'rms'; "
            f"received '{scale_method}'."
        )


class _BitLinearBase(nn.Linear):
    """
    Shared implementation for ternary training layers.

    This class retains a floating-point shadow weight for optimization. During
    the forward pass, it applies fake quantization and straight-through
    estimation (STE), so gradients update the shadow weight normally.

    This is training/evaluation code. It is not the final packed 2-bit runtime.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        weight_group_size: int = DEFAULT_WEIGHT_GROUP_SIZE,
        weight_scale_method: str = "mean_abs",
        activation_bits: int = DEFAULT_ACTIVATION_BITS,
        activation_group_size: Optional[int] = None,
        quantization_enabled: bool = True,
        norm_eps: float = 1e-8,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)

        _validate_weight_quantization(
            group_size=weight_group_size,
            scale_method=weight_scale_method,
        )
        _validate_activation_bits(activation_bits)

        if activation_group_size is not None and activation_group_size <= 0:
            raise ValueError(
                "activation_group_size must be positive or None; "
                f"received {activation_group_size}."
            )

        if norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive; received {norm_eps}.")

        self.norm = RMSNorm(in_features, eps=norm_eps)

        self.weight_group_size = weight_group_size
        self.weight_scale_method = weight_scale_method
        self.activation_bits = activation_bits
        self.activation_group_size = activation_group_size
        self.quantization_enabled = quantization_enabled

    def set_quantization_enabled(self, enabled: bool) -> None:
        """
        Enable or disable fake quantization.

        This is useful for a QAT warm-up phase. Disabling quantization does not
        remove RMSNorm and does not alter the parameter layout.
        """
        self.quantization_enabled = bool(enabled)

    def set_quantization_config(
        self,
        weight_group_size: Optional[int] = None,
        weight_scale_method: Optional[str] = None,
        activation_bits: Optional[int] = None,
        activation_group_size: Optional[int] = None,
    ) -> None:
        """
        Update quantization parameters after construction.

        This enables a training loop to switch from a full-precision warm-up to
        groupwise ternary QAT without recreating the model.
        """
        new_group_size = (
            self.weight_group_size
            if weight_group_size is None
            else weight_group_size
        )
        new_scale_method = (
            self.weight_scale_method
            if weight_scale_method is None
            else weight_scale_method
        )
        new_activation_bits = (
            self.activation_bits
            if activation_bits is None
            else activation_bits
        )

        _validate_weight_quantization(new_group_size, new_scale_method)
        _validate_activation_bits(new_activation_bits)

        if activation_group_size is not None and activation_group_size <= 0:
            raise ValueError(
                "activation_group_size must be positive or None; "
                f"received {activation_group_size}."
            )

        self.weight_group_size = new_group_size
        self.weight_scale_method = new_scale_method
        self.activation_bits = new_activation_bits
        self.activation_group_size = activation_group_size

    def quantized_weight(self) -> torch.Tensor:
        """
        Return the dequantized groupwise ternary view of the shadow weight.

        The output values are constrained to {-scale, 0, +scale} per group.
        """
        return weight_quant(
            self.weight,
            group_size=self.weight_group_size,
            scale_method=self.weight_scale_method,
        )

    def _quantize_input_ste(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_enabled:
            return x

        fake_quantized = activation_quant(
            x,
            bits=self.activation_bits,
            group_size=self.activation_group_size,
        )
        return x + (fake_quantized - x).detach()

    def _quantize_weight_ste(self) -> torch.Tensor:
        if not self.quantization_enabled:
            return self.weight

        fake_quantized = self.quantized_weight()
        return self.weight + (fake_quantized - self.weight).detach()

    def _normalized_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_x = self._normalized_input(x)
        quantized_x = self._quantize_input_ste(normalized_x)
        quantized_weight = self._quantize_weight_ste()

        return F.linear(quantized_x, quantized_weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"weight_group_size={self.weight_group_size}, "
            f"weight_scale_method='{self.weight_scale_method}', "
            f"activation_bits={self.activation_bits}, "
            f"activation_group_size={self.activation_group_size}, "
            f"quantization_enabled={self.quantization_enabled}"
        )


class BitLinear(_BitLinearBase):
    """
    RMSNorm + activation fake quantization + groupwise ternary linear layer.

    This class keeps the public name used by the prior implementation.
    """

    pass


class FusedBitLinear(_BitLinearBase):
    """
    Drop-in BitLinear implementation used by HGRN-Bit.

    The previous implementation fused normalization, activation quantization,
    and linear projection through a custom Triton autograd function. That
    function used a different global weight scale than the updated groupwise
    ternary QAT path. This implementation deliberately uses the canonical
    bitnet.py quantizer so fused and unfused modules train the same model.

    Once correctness and model quality are validated, this class is the single
    place where a future Triton W2A8 kernel should be introduced.
    """

    def _normalized_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm remains a distinct operation in the autograd graph.

        PyTorch can still fuse portions of this graph under torch.compile, and
        keeping this implementation explicit makes the QAT behavior debuggable.
        """
        return self.norm(x)


def layer_norm_linear_quant_fn(
    x: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    norm_bias: Optional[torch.Tensor],
    linear_weight: torch.Tensor,
    linear_bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
    weight_group_size: int = DEFAULT_WEIGHT_GROUP_SIZE,
    weight_scale_method: str = "mean_abs",
    activation_bits: int = DEFAULT_ACTIVATION_BITS,
    activation_group_size: Optional[int] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Compatibility implementation of the old fused function API.

    It performs the same conceptual operation with standard PyTorch autograd:
    residual add, LayerNorm/RMSNorm, activation fake quantization, groupwise
    ternary weight fake quantization, and linear projection.

    Args:
        x: Input tensor with features in its final dimension.
        norm_weight: Normalization scale, or None for unscaled normalization.
        norm_bias: LayerNorm bias. Ignored for RMSNorm.
        linear_weight: Floating-point shadow linear weight.
        linear_bias: Optional linear bias.
        residual: Optional residual added before normalization.
        eps: Normalization epsilon.
        prenorm: Return normalized residual alongside output when True.
        residual_in_fp32: Preserve residual accumulation in FP32 when True.
        is_rms_norm: Use RMSNorm instead of LayerNorm when True.
        weight_group_size: Number of weights sharing each ternary scale.
        weight_scale_method: "mean_abs" or "rms".
        activation_bits: Activation fake quantization bit width.
        activation_group_size: Optional activation group size.

    Returns:
        Linear output, or `(output, residual_out)` when `prenorm=True`.
    """
    _validate_weight_quantization(
        group_size=weight_group_size,
        scale_method=weight_scale_method,
    )
    _validate_activation_bits(activation_bits)

    if x.shape[-1] != linear_weight.shape[-1]:
        raise ValueError(
            "Input feature size must equal linear_weight.shape[-1]; "
            f"got x.shape[-1]={x.shape[-1]} and "
            f"linear_weight.shape[-1]={linear_weight.shape[-1]}."
        )

    residual_out = x

    if residual is not None:
        if residual.shape != x.shape:
            raise ValueError(
                "residual must have the same shape as x; "
                f"got residual.shape={residual.shape}, x.shape={x.shape}."
            )

        if residual_in_fp32:
            residual_out = x.float() + residual.float()
        else:
            residual_out = x + residual

    normalized_input = residual_out

    if is_rms_norm:
        variance = normalized_input.float().square().mean(
            dim=-1,
            keepdim=True,
        )
        normalized_input = normalized_input * torch.rsqrt(
            variance + eps
        ).to(normalized_input.dtype)

        if norm_weight is not None:
            normalized_input = normalized_input * norm_weight
    else:
        normalized_input = F.layer_norm(
            normalized_input,
            normalized_shape=(normalized_input.shape[-1],),
            weight=norm_weight,
            bias=norm_bias,
            eps=eps,
        )

    if residual_out.dtype != x.dtype:
        residual_out = residual_out.to(x.dtype)

    fake_quantized_input = activation_quant(
        normalized_input,
        bits=activation_bits,
        group_size=activation_group_size,
    )
    quantized_input = normalized_input + (
        fake_quantized_input - normalized_input
    ).detach()

    fake_quantized_weight = weight_quant(
        linear_weight,
        group_size=weight_group_size,
        scale_method=weight_scale_method,
    )
    quantized_weight = linear_weight + (
        fake_quantized_weight - linear_weight
    ).detach()

    output = F.linear(
        quantized_input,
        quantized_weight,
        linear_bias,
    )

    if prenorm:
        return output, residual_out

    return output


class LayerNormLinearQuantFn:
    """
    Compatibility facade for code importing the former autograd.Function.

    New model code should call `layer_norm_linear_quant_fn` directly. The class
    remains only to avoid import failures while the rest of the repository is
    migrated to groupwise ternary QAT.
    """

    @staticmethod
    def apply(
        x: torch.Tensor,
        norm_weight: Optional[torch.Tensor],
        norm_bias: Optional[torch.Tensor],
        linear_weight: torch.Tensor,
        linear_bias: Optional[torch.Tensor],
        residual: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
        weight_group_size: int = DEFAULT_WEIGHT_GROUP_SIZE,
        weight_scale_method: str = "mean_abs",
        activation_bits: int = DEFAULT_ACTIVATION_BITS,
        activation_group_size: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return layer_norm_linear_quant_fn(
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            linear_weight=linear_weight,
            linear_bias=linear_bias,
            residual=residual,
            eps=eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=is_rms_norm,
            weight_group_size=weight_group_size,
            weight_scale_method=weight_scale_method,
            activation_bits=activation_bits,
            activation_group_size=activation_group_size,
        )