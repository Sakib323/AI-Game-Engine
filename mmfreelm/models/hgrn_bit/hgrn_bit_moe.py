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
        self.expert_capacity = int(config.moe_intermediate_size * 1.5)  # Buffer capacity

        # Expert parameters
        from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitMLP
        self.experts = nn.ModuleList([
            HGRNBitMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act
            ) for _ in range(config.num_experts)
        ])
        
        # Router using BitLinear for ternary quantization
        self.gate_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate = BitLinear(config.hidden_size, config.num_experts, bias=False)
        
        # Load balancing loss
        self.aux_loss = torch.tensor(0.0)
        self.register_buffer("expert_counts", torch.zeros(config.num_experts))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.size()
        x_flat = x.view(-1, self.hidden_size)
        
        # Router logic with ternary quantization
        x_norm = self.gate_norm(x_flat)
        gate_logits = self.gate(x_norm)
        
        # Gating with top-k expert selection
        top_k_weights, top_k_indices = torch.topk(
            gate_logits.softmax(dim=-1), 
            self.top_k, 
            dim=-1
        )
        
        # Expert capacity calculation
        capacity = min(self.expert_capacity, batch_size * seq_len // self.num_experts)
        expert_mask = torch.zeros_like(gate_logits).scatter_(
            1, top_k_indices, top_k_weights
        )
        
        # Expert computation with load balancing
        outputs = torch.zeros_like(x_flat)
        aux_loss = 0.0
        
        for expert_idx in range(self.num_experts):
            # Get tokens assigned to this expert
            mask = expert_mask[:, expert_idx].bool()
            if mask.sum() == 0:
                continue
                
            # Apply capacity constraint
            masked_indices = torch.where(mask)[0]
            if len(masked_indices) > capacity:
                selected_indices = masked_indices[:capacity]
                mask[masked_indices[capacity:]] = False
            else:
                selected_indices = masked_indices
                
            # Process tokens with expert
            expert_input = x_flat[selected_indices]
            expert_output = self.experts[expert_idx](expert_input)
            
            # Weighted sum of expert outputs
            weights = top_k_weights[selected_indices, torch.where(top_k_indices[selected_indices] == expert_idx)[1]]
            outputs[selected_indices] += weights.unsqueeze(-1) * expert_output
            
            # Load balancing statistics
            self.expert_counts[expert_idx] += mask.sum().item()
            aux_loss += mask.float().mean() * expert_idx  # Simple load balancing
            
        # Calculate auxiliary loss
        self.aux_loss = aux_loss / self.num_experts
        
        return outputs.view(batch_size, seq_len, -1)
