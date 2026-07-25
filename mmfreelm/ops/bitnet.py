# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmfreelm.modules import RMSNorm


TERNARY_WEIGHT_BITS = 2
DEFAULT_WEIGHT_GROUP_SIZE = 128
DEFAULT_ACTIVATION_BITS = 8
QUANT_EPSILON = 1e-5


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
            f"group_size must be positive; received {group_size}."
        )

    if scale_method not in ("mean_abs", "rms"):
        raise ValueError(
            "scale_method must be either 'mean_abs' or 'rms'; "
            f"received '{scale_method}'."
        )


def _quantization_levels(bits: int) -> tuple[int, int]:
    _validate_activation_bits(bits)

    if bits == 16:
        return 0, 0

    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    return qmin, qmax


def activation_quant(
    x: torch.Tensor,
    bits: int = DEFAULT_ACTIVATION_BITS,
    group_size: Optional[int] = None,
    eps: float = QUANT_EPSILON,
) -> torch.Tensor:
    """
    Fake-quantize activations symmetrically.

    Args:
        x: Activation tensor with features in its final dimension.
        bits: Activation precision. Supported values are 4, 8, and 16.
        group_size: Optional number of final-dimension features per scale.
            None uses one scale per token/vector.
        eps: Lower bound for scales.

    Returns:
        A dequantized tensor with the same shape and dtype as x.
    """
    if bits == 16:
        return x

    if x.numel() == 0:
        return x

    qmin, qmax = _quantization_levels(bits)
    feature_size = x.shape[-1]

    if group_size is None or group_size >= feature_size:
        max_abs = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        scale = qmax / max_abs
        return (x * scale).round().clamp(qmin, qmax) / scale

    if group_size <= 0:
        raise ValueError(
            f"group_size must be positive or None; received {group_size}."
        )

    original_shape = x.shape
    flat_x = x.reshape(-1, feature_size)
    padded_feature_size = (
        (feature_size + group_size - 1) // group_size
    ) * group_size
    padding = padded_feature_size - feature_size

    if padding:
        flat_x = F.pad(flat_x, (0, padding))

    grouped_x = flat_x.reshape(-1, padded_feature_size // group_size, group_size)
    max_abs = grouped_x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    scale = qmax / max_abs
    grouped_x = (grouped_x * scale).round().clamp(qmin, qmax) / scale

    flat_x = grouped_x.reshape(-1, padded_feature_size)
    if padding:
        flat_x = flat_x[:, :feature_size]

    return flat_x.reshape(original_shape)


def _group_scale(
    grouped_weight: torch.Tensor,
    scale_method: str,
    eps: float,
) -> torch.Tensor:
    if scale_method == "mean_abs":
        return grouped_weight.abs().mean(dim=-1, keepdim=True).clamp_min(eps)

    return grouped_weight.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)


def weight_quant(
    w: torch.Tensor,
    group_size: int = DEFAULT_WEIGHT_GROUP_SIZE,
    scale_method: str = "mean_abs",
    eps: float = QUANT_EPSILON,
) -> torch.Tensor:
    """
    Groupwise ternary fake quantization for a weight tensor.

    The final dimension is split into fixed-size groups. Every group receives
    one symmetric scale and every weight is mapped to {-scale, 0, +scale}.

    This must match the future packed-inference exporter:
        00 -> 0
        01 -> +1
        10 -> -1
        11 -> reserved

    Args:
        w: Weight tensor with input features on the final dimension.
        group_size: Number of input weights sharing one ternary scale.
        scale_method: "mean_abs" or "rms".
        eps: Lower bound for scales.

    Returns:
        Dequantized ternary weights with the same shape and dtype as w.
    """
    _validate_weight_quantization(group_size, scale_method)

    if w.numel() == 0:
        return w

    input_features = w.shape[-1]
    padded_input_features = (
        (input_features + group_size - 1) // group_size
    ) * group_size
    padding = padded_input_features - input_features

    flat_w = w.reshape(-1, input_features)
    if padding:
        flat_w = F.pad(flat_w, (0, padding))

    grouped_w = flat_w.reshape(
        -1,
        padded_input_features // group_size,
        group_size,
    )

    scale = _group_scale(grouped_w, scale_method, eps)
    ternary_codes = (grouped_w / scale).round().clamp(-1, 1)
    grouped_w = ternary_codes * scale

    flat_w = grouped_w.reshape(-1, padded_input_features)
    if padding:
        flat_w = flat_w[:, :input_features]

    return flat_w.reshape_as(w)


def ternary_codes_and_scales(
    w: torch.Tensor,
    group_size: int = DEFAULT_WEIGHT_GROUP_SIZE,
    scale_method: str = "mean_abs",
    eps: float = QUANT_EPSILON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create export-ready ternary codes and scales without bit-packing.

    This is a reference utility for exporter/parity tests. The returned codes
    use int8 values {-1, 0, +1}; an exporter will later map them to 2-bit codes.
    """
    _validate_weight_quantization(group_size, scale_method)

    if w.ndim < 2:
        raise ValueError(
            "ternary_codes_and_scales expects at least a 2D weight tensor."
        )

    input_features = w.shape[-1]
    padded_input_features = (
        (input_features + group_size - 1) // group_size
    ) * group_size
    padding = padded_input_features - input_features

    flat_w = w.detach().reshape(-1, input_features)
    if padding:
        flat_w = F.pad(flat_w, (0, padding))

    grouped_w = flat_w.reshape(
        -1,
        padded_input_features // group_size,
        group_size,
    )
    scales = _group_scale(grouped_w, scale_method, eps)
    codes = (grouped_w / scales).round().clamp(-1, 1).to(torch.int8)

    if padding:
        codes = codes.reshape(-1, padded_input_features)[:, :input_features]
        codes = codes.reshape(
            *w.shape[:-1],
            input_features,
        )
    else:
        codes = codes.reshape_as(w)

    scales = scales.squeeze(-1).reshape(
        *w.shape[:-1],
        padded_input_features // group_size,
    )
    return codes, scales


class BitLinear(nn.Linear):
    """
    BitLinear training layer with RMSNorm, activation fake quantization,
    groupwise ternary weight fake quantization, and STE gradients.

    The trainable weight remains a normal floating-point shadow parameter.
    This class is for training/evaluation, not packed low-memory inference.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_group_size: int = DEFAULT_WEIGHT_GROUP_SIZE,
        weight_scale_method: str = "mean_abs",
        activation_bits: int = DEFAULT_ACTIVATION_BITS,
        activation_group_size: Optional[int] = None,
        quantization_enabled: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)

        _validate_weight_quantization(
            group_size=weight_group_size,
            scale_method=weight_scale_method,
        )
        _validate_activation_bits(activation_bits)

        self.norm = RMSNorm(in_features, eps=1e-8)
        self.weight_group_size = weight_group_size
        self.weight_scale_method = weight_scale_method
        self.activation_bits = activation_bits
        self.activation_group_size = activation_group_size
        self.quantization_enabled = quantization_enabled

    def set_quantization_enabled(self, enabled: bool) -> None:
        self.quantization_enabled = bool(enabled)

    def quantized_weight(self) -> torch.Tensor:
        return weight_quant(
            self.weight,
            group_size=self.weight_group_size,
            scale_method=self.weight_scale_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)

        if not self.quantization_enabled:
            return F.linear(x_norm, self.weight, self.bias)

        x_fake_quant = activation_quant(
            x_norm,
            bits=self.activation_bits,
            group_size=self.activation_group_size,
        )
        w_fake_quant = self.quantized_weight()

        x_quant = x_norm + (x_fake_quant - x_norm).detach()
        w_quant = self.weight + (w_fake_quant - self.weight).detach()

        return F.linear(x_quant, w_quant, self.bias)


class BitLinear_wonorm_bmm(nn.Module):
    """
    Ternary BMM layer without RMSNorm.

    The expected input layout is [..., M, K] and the weight layout is either
    [K, N] or [..., K, N]. The latter follows torch.bmm broadcasting rules
    only when the leading batch dimensions match.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_group_size: int = DEFAULT_WEIGHT_GROUP_SIZE,
        weight_scale_method: str = "mean_abs",
        activation_bits: int = DEFAULT_ACTIVATION_BITS,
        activation_group_size: Optional[int] = None,
        quantization_enabled: bool = True,
    ) -> None:
        super().__init__()

        _validate_weight_quantization(
            group_size=weight_group_size,
            scale_method=weight_scale_method,
        )
        _validate_activation_bits(activation_bits)

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.weight_group_size = weight_group_size
        self.weight_scale_method = weight_scale_method
        self.activation_bits = activation_bits
        self.activation_group_size = activation_group_size
        self.quantization_enabled = quantization_enabled

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if self.bias is not None:
            bound = 1.0 / max(1, self.in_features) ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def set_quantization_enabled(self, enabled: bool) -> None:
        self.quantization_enabled = bool(enabled)

    def quantized_weight(self) -> torch.Tensor:
        return weight_quant(
            self.weight,
            group_size=self.weight_group_size,
            scale_method=self.weight_scale_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected x.shape[-1] == {self.in_features}, "
                f"received {x.shape[-1]}."
            )

        if self.quantization_enabled:
            x_fake_quant = activation_quant(
                x,
                bits=self.activation_bits,
                group_size=self.activation_group_size,
            )
            w_fake_quant = self.quantized_weight()
            x = x + (x_fake_quant - x).detach()
            weight = self.weight + (w_fake_quant - self.weight).detach()
        else:
            weight = self.weight

        output = torch.matmul(x, weight)

        if self.bias is not None:
            output = output + self.bias

        return output