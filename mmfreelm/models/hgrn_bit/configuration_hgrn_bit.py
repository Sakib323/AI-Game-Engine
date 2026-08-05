# -*- coding: utf-8 -*-

import math
from typing import Optional, Sequence, Union

from transformers.configuration_utils import PretrainedConfig


class HGRNBitConfig(PretrainedConfig):
    """
    Configuration for HGRN-Bit causal language models.

    Training uses BF16/FP32 shadow weights with fake quantization and STE.
    Low-bit fields describe the training quantizer and the intended deployment
    export format; they do not imply that optimizer-state memory is 2-bit.
    """

    model_type = "hgrn_bit"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        attn_mode: str = "fused_recurrent",
        token_mixer: str = "hgrn",
        gla_chunk_size: int = 64,
        num_heads: int = 1,
        expand_ratio: Union[int, float] = 1,
        use_short_conv: bool = True,
        conv_size: int = 4,
        share_conv_kernel: bool = True,
        conv_bias: bool = False,
        use_lower_bound: bool = True,
        decay_mode: str = "independent",
        decay_init: float = 0.5,
        hidden_ratio: Optional[float] = None,
        mlp_ratio: float = 8.0 / 3.0,
        intermediate_size: Optional[int] = None,
        intermediate_multiple_of: int = 256,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        fuse_cross_entropy: bool = True,
        rotary_embeddings: bool = False,
        rope_theta: float = 10000.0,
        use_ternary_rope: bool = True,
        weight_quant_bits: int = 2,
        weight_quant_group_size: int = 128,
        weight_quant_scale: str = "mean_abs",
        activation_quant_bits: int = 8,
        activation_quant_group_size: Optional[int] = None,
        quantization_warmup_steps: int = 0,
        quantize_embeddings: bool = False,
        quantize_lm_head: bool = False,
        export_weight_format: str = "ternary_2bit",
        export_scale_dtype: str = "float16",
        moe: bool = False,
        moe_layer_interval: int = 4,
        moe_layer_indices: Optional[Sequence[int]] = None,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: Optional[int] = None,
        moe_capacity_factor: float = 1.25,
        moe_shared_expert: bool = False,
        moe_shared_expert_ratio: float = 0.5,
        router_aux_loss_coef: float = 0.02,
        router_z_loss_coef: float = 1e-4,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attn_mode = attn_mode
        self.token_mixer = token_mixer
        self.gla_chunk_size = gla_chunk_size
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.share_conv_kernel = share_conv_kernel
        self.conv_bias = conv_bias

        self.use_lower_bound = use_lower_bound
        self.decay_mode = decay_mode
        self.decay_init = decay_init

        self.hidden_ratio = hidden_ratio
        self.mlp_ratio = mlp_ratio
        self.intermediate_multiple_of = intermediate_multiple_of
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else self._compute_intermediate_size(
                hidden_size=hidden_size,
                mlp_ratio=mlp_ratio,
                multiple_of=intermediate_multiple_of,
            )
        )
        self.hidden_act = hidden_act

        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy

        self.rotary_embeddings = rotary_embeddings
        self.rope_theta = rope_theta
        self.use_ternary_rope = use_ternary_rope

        self.weight_quant_bits = weight_quant_bits
        self.weight_quant_group_size = weight_quant_group_size
        self.weight_quant_scale = weight_quant_scale
        self.activation_quant_bits = activation_quant_bits
        self.activation_quant_group_size = activation_quant_group_size
        self.quantization_warmup_steps = quantization_warmup_steps
        self.quantize_embeddings = quantize_embeddings
        self.quantize_lm_head = quantize_lm_head
        self.export_weight_format = export_weight_format
        self.export_scale_dtype = export_scale_dtype

        self.moe = moe
        self.moe_layer_interval = moe_layer_interval
        self.moe_layer_indices = (
            None if moe_layer_indices is None else list(moe_layer_indices)
        )
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = (
            moe_intermediate_size
            if moe_intermediate_size is not None
            else self.intermediate_size
        )
        self.moe_capacity_factor = moe_capacity_factor
        self.moe_shared_expert = moe_shared_expert
        self.moe_shared_expert_ratio = moe_shared_expert_ratio
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

        self._validate()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @staticmethod
    def _compute_intermediate_size(
        hidden_size: int,
        mlp_ratio: float,
        multiple_of: int,
    ) -> int:
        intermediate_size = int(hidden_size * mlp_ratio)
        return max(
            multiple_of,
            math.ceil(intermediate_size / multiple_of) * multiple_of,
        )

    def _validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive.")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        input_dim = int(self.hidden_size * self.expand_ratio)
        if input_dim <= 0:
            raise ValueError("hidden_size * expand_ratio must be positive.")

        if input_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_size * expand_ratio must be divisible by num_heads; "
                f"got input_dim={input_dim}, num_heads={self.num_heads}."
            )

        if self.conv_size <= 0:
            raise ValueError("conv_size must be positive.")

        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive.")

        if self.intermediate_multiple_of <= 0:
            raise ValueError("intermediate_multiple_of must be positive.")

        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")

        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive.")

        if self.attn_mode != "fused_recurrent":
            raise ValueError(
                "Only attn_mode='fused_recurrent' is currently supported."
            )

        if self.token_mixer not in {"hgrn", "hgrn2"}:
            raise ValueError("token_mixer must be 'hgrn' or 'hgrn2'.")

        if self.gla_chunk_size <= 0:
            raise ValueError("gla_chunk_size must be positive.")

        if self.token_mixer == "hgrn2":
            input_dim = int(self.hidden_size * self.expand_ratio)
            head_dim = input_dim // self.num_heads
            if head_dim < 32:
                raise ValueError(
                    f"hgrn2 needs head_dim >= 32 for a useful matrix state; got {head_dim}."
                )

        if self.decay_mode not in {"independent", "global_lower_bound"}:
            raise ValueError(
                "decay_mode must be 'independent' or 'global_lower_bound'."
            )

        if not 0.0 < self.decay_init < 1.0:
            raise ValueError("decay_init must be strictly between 0 and 1.")

        if self.weight_quant_bits != 2:
            raise ValueError(
                "HGRN-Bit currently supports ternary deployment only: "
                "weight_quant_bits must be 2."
            )

        if self.weight_quant_group_size <= 0:
            raise ValueError("weight_quant_group_size must be positive.")

        if self.weight_quant_scale not in {"mean_abs", "rms"}:
            raise ValueError(
                "weight_quant_scale must be either 'mean_abs' or 'rms'."
            )

        if self.activation_quant_bits not in {4, 8, 16}:
            raise ValueError(
                "activation_quant_bits must be one of: 4, 8, or 16."
            )

        if (
            self.activation_quant_group_size is not None
            and self.activation_quant_group_size <= 0
        ):
            raise ValueError(
                "activation_quant_group_size must be positive or None."
            )

        if self.quantization_warmup_steps < 0:
            raise ValueError("quantization_warmup_steps must be non-negative.")

        if self.export_weight_format != "ternary_2bit":
            raise ValueError(
                "export_weight_format must be 'ternary_2bit' for this model."
            )

        if self.export_scale_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "export_scale_dtype must be float16, bfloat16, or float32."
            )

        if self.moe:
            if self.num_experts < 2:
                raise ValueError("MoE requires num_experts >= 2.")

            if not 1 <= self.num_experts_per_tok <= self.num_experts:
                raise ValueError(
                    "num_experts_per_tok must be between 1 and num_experts."
                )

            if self.moe_intermediate_size <= 0:
                raise ValueError("moe_intermediate_size must be positive.")

            if self.moe_layer_interval <= 0:
                raise ValueError("moe_layer_interval must be positive.")

            if self.moe_capacity_factor <= 0:
                raise ValueError("moe_capacity_factor must be positive.")

            if self.moe_shared_expert_ratio <= 0:
                raise ValueError("moe_shared_expert_ratio must be positive.")

            if self.moe_layer_indices is not None:
                invalid_indices = [
                    index
                    for index in self.moe_layer_indices
                    if not 0 <= index < self.num_hidden_layers
                ]
                if invalid_indices:
                    raise ValueError(
                        "moe_layer_indices contains invalid layer indices: "
                        f"{invalid_indices}."
                    )

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not self.moe:
            return False

        if not 0 <= layer_idx < self.num_hidden_layers:
            raise ValueError(
                f"layer_idx must be in [0, {self.num_hidden_layers}), "
                f"got {layer_idx}."
            )

        if self.moe_layer_indices is not None:
            return layer_idx in self.moe_layer_indices

        return (layer_idx + 1) % self.moe_layer_interval == 0