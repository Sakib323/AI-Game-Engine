import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear
from mmfreelm.modules import RMSNorm

class HGRNBitMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        
        # Dynamic capacity factor (1.25x buffer is standard)
        self.capacity_factor = 1.25 

        from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitMLP

        self.experts = nn.ModuleList([
            HGRNBitMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act
            ) for _ in range(config.num_experts)
        ])
        
        # Router using Standard Linear for precision (BitLinear breaks routing convergence)
        self.gate_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        self.shared_expert = HGRNBitMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size, 
            hidden_act=config.hidden_act
        )
        # REMOVED: Shared Expert Gate (DeepSeek-V2 style: Direct addition is more stable)

        # ------------------------------------------------------------------
        # 3. Load Balancing
        # ------------------------------------------------------------------
        self.aux_loss = torch.tensor(0.0)
        self.register_buffer("expert_counts", torch.zeros(config.num_experts))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.size()
        x_flat = x.view(-1, self.hidden_size)
        
        # Independent Shared Expert (DeepSeek-V2/V3 Style: Direct Add)
        shared_output = self.shared_expert(x_flat)

        x_norm = self.gate_norm(x_flat)
        gate_logits = self.gate(x_norm)
        
        # Router Z-Loss (Stability Fix)
        # Prevent logits from exploding using LogSumExp
        z_loss = torch.logsumexp(gate_logits, dim=-1).square().mean() * 1e-4

        # Gating with top-k expert selection
        router_probs = gate_logits.softmax(dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Dynamic Expert Capacity Calculation (Token-based)
        tokens_per_expert = (batch_size * seq_len) / self.num_experts
        capacity = int(tokens_per_expert * self.capacity_factor)

        # Create a mask for mapping tokens to experts
        expert_mask = torch.zeros_like(gate_logits, dtype=top_k_weights.dtype).scatter_(
            1, top_k_indices, top_k_weights
        )
        
        # Routed output container
        routed_outputs = torch.zeros_like(x_flat)
        
        # Fix CPU Sync: Update counts using detach() to avoid blocking GPU
        with torch.no_grad():
            self.expert_counts += expert_mask.sum(dim=0).detach()
        
        for expert_idx in range(self.num_experts):
            # Get tokens assigned to this expert
            mask = expert_mask[:, expert_idx].bool()
            if mask.sum() == 0:
                continue
                
            # Apply capacity constraint
            masked_indices = torch.where(mask)[0]
            if len(masked_indices) > capacity:
                selected_indices = masked_indices[:capacity]
                # mask[masked_indices[capacity:]] = False # Optional: Update mask for loss
            else:
                selected_indices = masked_indices
                
            # Process tokens with expert
            expert_input = x_flat[selected_indices]
            expert_output = self.experts[expert_idx](expert_input)
            
            relevant_weights = torch.gather(
                top_k_weights[selected_indices], 
                1, 
                torch.where(top_k_indices[selected_indices] == expert_idx)[1].unsqueeze(1)
            ).squeeze(1)
            
            routed_outputs[selected_indices] += relevant_weights.unsqueeze(-1) * expert_output
            
        # Switch Transformer Loss (Correct Formula)
        # density = fraction of tokens in batch assigned to expert
        density = expert_mask.float().mean(dim=0)
        # probs = average probability assigned to expert
        probs = router_probs.mean(dim=0)
        # Switch Loss = dot(density, probs) * N
        switch_loss = (density * probs).sum() * self.num_experts
            
        self.aux_loss = switch_loss + z_loss

        final_output = routed_outputs + shared_output
        
        return final_output.view(batch_size, seq_len, -1)