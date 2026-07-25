# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitMLP
from mmfreelm.modules import RMSNorm


class HGRNBitMoE(nn.Module):
    """
    Selective sparse MoE feed-forward layer for HGRN-Bit.

    The router remains full precision because routing errors are discrete and
    disproportionately harmful. Expert MLPs use the same groupwise ternary QAT
    projections as dense HGRNBitMLP layers.

    This module is intended to be placed only in selected decoder layers through
    HGRNBitConfig.is_moe_layer(), not in every model block.
    """

    def __init__(self, config) -> None:
        super().__init__()

        if config.num_experts < 2:
            raise ValueError("HGRNBitMoE requires num_experts >= 2.")

        if not 1 <= config.num_experts_per_tok <= config.num_experts:
            raise ValueError(
                "num_experts_per_tok must be between 1 and num_experts."
            )

        if config.moe_capacity_factor <= 0:
            raise ValueError("moe_capacity_factor must be positive.")

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.capacity_factor = config.moe_capacity_factor

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.router_z_loss_coef = config.router_z_loss_coef

        self.use_shared_expert = config.moe_shared_expert
        self.shared_expert_ratio = config.moe_shared_expert_ratio

        linear_kwargs = {
            "weight_group_size": config.weight_quant_group_size,
            "weight_scale_method": config.weight_quant_scale,
            "activation_bits": config.activation_quant_bits,
            "activation_group_size": config.activation_quant_group_size,
            "quantization_enabled": config.quantization_warmup_steps == 0,
        }

        self.experts = nn.ModuleList(
            [
                HGRNBitMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                    **linear_kwargs,
                )
                for _ in range(self.num_experts)
            ]
        )

        self.gate_norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.gate = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False,
        )

        if self.use_shared_expert:
            shared_intermediate_size = max(
                config.intermediate_multiple_of,
                int(
                    math.ceil(
                        config.moe_intermediate_size
                        * self.shared_expert_ratio
                        / config.intermediate_multiple_of
                    )
                    * config.intermediate_multiple_of
                ),
            )

            self.shared_expert = HGRNBitMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate_size,
                hidden_act=config.hidden_act,
                **linear_kwargs,
            )
        else:
            self.shared_expert = None

        self.register_buffer(
            "expert_counts",
            torch.zeros(self.num_experts, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "dropped_tokens",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "routing_steps",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )

        self.aux_loss: Optional[Tensor] = None

    def set_quantization_enabled(self, enabled: bool) -> None:
        """
        Enable or disable ternary QAT in every expert MLP.

        The router and router normalization intentionally remain full precision.
        """
        for expert in self.experts:
            expert.set_quantization_enabled(enabled)

        if self.shared_expert is not None:
            self.shared_expert.set_quantization_enabled(enabled)

    def reset_routing_stats(self) -> None:
        """
        Clear persistent monitoring statistics without changing model weights.
        """
        self.expert_counts.zero_()
        self.dropped_tokens.zero_()
        self.routing_steps.zero_()

    @property
    def expert_utilization(self) -> Tensor:
        """
        Fraction of routed token assignments received by each expert.
        """
        total_assignments = self.expert_counts.sum()

        if total_assignments.item() == 0:
            return torch.zeros_like(
                self.expert_counts,
                dtype=torch.float32,
            )

        return self.expert_counts.float() / total_assignments.float()

    def _expert_capacity(self, num_tokens: int) -> int:
        """
        Return per-expert routing capacity.

        Capacity scales by top-k so a top-2 router has room for twice as many
        total assignments as top-1 routing.
        """
        expected_assignments = (
            num_tokens * self.top_k / self.num_experts
        )
        return max(
            1,
            math.ceil(expected_assignments * self.capacity_factor),
        )

    def _router_auxiliary_loss(
        self,
        router_probs: Tensor,
        topk_indices: Tensor,
    ) -> Tensor:
        """
        Compute Switch-style load balancing plus router z-loss.

        Density uses hard assignments, while probability mass remains
        differentiable and supplies the router gradient.
        """
        num_tokens = router_probs.shape[0]

        hard_assignments = F.one_hot(
            topk_indices,
            num_classes=self.num_experts,
        ).to(dtype=router_probs.dtype)

        assignment_density = hard_assignments.sum(dim=(0, 1)) / (
            num_tokens * self.top_k
        )
        probability_density = router_probs.mean(dim=0)

        load_balance_loss = (
            self.num_experts
            * torch.sum(assignment_density * probability_density)
        )

        return load_balance_loss

    def _update_routing_stats(
        self,
        routed_counts: Tensor,
        dropped_assignments: int,
    ) -> None:
        """
        Update monitoring buffers without adding autograd work.
        """
        with torch.no_grad():
            self.expert_counts.add_(routed_counts.to(self.expert_counts.dtype))
            self.dropped_tokens.add_(int(dropped_assignments))
            self.routing_steps.add_(1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Route each token to up to `top_k` expert MLPs.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden_size].

        Returns:
            Tensor with the same shape as hidden_states.
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden]; "
                f"received {tuple(hidden_states.shape)}."
            )

        batch_size, sequence_length, hidden_size = hidden_states.shape

        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"received {hidden_size}."
            )

        num_tokens = batch_size * sequence_length
        flat_hidden_states = hidden_states.reshape(
            num_tokens,
            self.hidden_size,
        )

        router_input = self.gate_norm(flat_hidden_states)
        router_logits = self.gate(router_input)
        router_probs = torch.softmax(router_logits.float(), dim=-1).to(
            dtype=flat_hidden_states.dtype
        )

        topk_weights, topk_indices = torch.topk(
            router_probs,
            k=self.top_k,
            dim=-1,
        )

        # Top-1 is already normalized. For top-k, normalize selected expert
        # weights so each accepted token's routed mixture sums to one.
        if self.top_k > 1:
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(torch.finfo(topk_weights.dtype).eps)

        routed_outputs = torch.zeros_like(flat_hidden_states)
        routed_counts = torch.zeros(
            self.num_experts,
            dtype=torch.long,
            device=hidden_states.device,
        )

        capacity = self._expert_capacity(num_tokens)
        dropped_assignments = 0

        for expert_idx, expert in enumerate(self.experts):
            token_indices, topk_slots = torch.where(
                topk_indices == expert_idx
            )

            if token_indices.numel() == 0:
                continue

            selected_weights = topk_weights[
                token_indices,
                topk_slots,
            ]

            # Highest-weight assignments get capacity first. This is important
            # for top-k routing; taking tokens in sequence order biases routing.
            if token_indices.numel() > capacity:
                keep_order = torch.topk(
                    selected_weights,
                    k=capacity,
                    largest=True,
                    sorted=False,
                ).indices
                dropped_assignments += token_indices.numel() - capacity
                token_indices = token_indices[keep_order]
                selected_weights = selected_weights[keep_order]

            expert_output = expert(flat_hidden_states[token_indices])
            routed_outputs.index_add_(
                0,
                token_indices,
                expert_output * selected_weights.unsqueeze(-1),
            )
            routed_counts[expert_idx] = token_indices.numel()

        if self.shared_expert is not None:
            routed_outputs = routed_outputs + self.shared_expert(
                flat_hidden_states
            )

        load_balance_loss = self._router_auxiliary_loss(
            router_probs=router_probs,
            topk_indices=topk_indices,
        )
        z_loss = torch.logsumexp(
            router_logits.float(),
            dim=-1,
        ).square().mean()

        # The caller applies router_aux_loss_coef once to aux_loss. z_loss is
        # scaled here because it is a separate router-stability regularizer.
        self.aux_loss = (
            load_balance_loss
            + self.router_z_loss_coef * z_loss
        )

        self._update_routing_stats(
            routed_counts=routed_counts,
            dropped_assignments=dropped_assignments,
        )

        return routed_outputs.reshape(
            batch_size,
            sequence_length,
            self.hidden_size,
        )