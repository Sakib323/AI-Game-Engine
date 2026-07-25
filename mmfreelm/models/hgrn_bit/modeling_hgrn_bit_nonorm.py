# -*- coding: utf-8 -*-
"""
Deprecated compatibility module.

HGRN-Bit formerly maintained a separate "nonorm" architecture. It had drifted
from the main implementation and no longer shared the groupwise ternary-QAT,
MoE, recurrent-cache, quantization-warmup, or loss behavior.

Import from `modeling_hgrn_bit` for all new code. These aliases remain so older
training scripts, checkpoint tooling, and `auto_map` entries continue to import.
"""

from __future__ import annotations

import warnings

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import (
    HGRNBitBlock,
    HGRNBitForCausalLM,
    HGRNBitMLP,
    HGRNBitModel,
    HGRNBitPreTrainedModel,
    TiedLMHead,
)

warnings.warn(
    "`modeling_hgrn_bit_nonorm` is deprecated and now aliases the canonical "
    "`modeling_hgrn_bit` implementation. Update imports when convenient.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "HGRNBitBlock",
    "HGRNBitForCausalLM",
    "HGRNBitMLP",
    "HGRNBitModel",
    "HGRNBitPreTrainedModel",
    "TiedLMHead",
]