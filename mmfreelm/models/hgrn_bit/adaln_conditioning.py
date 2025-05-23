from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.utils import RecurrentCache
from mmfreelm.modules import FusedCrossEntropyLoss, RMSNorm
from mmfreelm.modules.activations import swiglu_linear, swiglu
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear
from mmfreelm.models.hgrn_bit.hgrn_bit_moe import HGRNBitMoE



class AdaLNConditioning(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        # Use HGRNBitMLP (ternary MLP with BitLinear) to process the input
        self.mlp = HGRNBitMLP(
            hidden_size=input_dim,
            intermediate_size=hidden_size * 2,  # Match typical expansion
            hidden_act='swish'
        )
        # RMSNorm for normalization before output
        self.norm = RMSNorm(hidden_size * 2, eps=eps)  # Output scale and shift
        # Linear layer to produce scale and shift parameters
        self.out_proj = BitLinear(hidden_size * 2, hidden_size * 2, bias=False)

    def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process the condition through the ternary MLP
        x = self.mlp(condition)
        # Apply RMSNorm
        x = self.norm(x)
        # Produce scale and shift parameters
        params = self.out_proj(x)
        scale, shift = params.chunk(2, dim=-1)
        return scale, shift