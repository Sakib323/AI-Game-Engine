# rotary_embedding.py

import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, use_ternary=True):
        super().__init__()

        print(f"\n[RotaryEmbedding] Initialized with: "
              f"dim={dim}, max_pos={max_position_embeddings}, "
              f"base={base}, ternary={use_ternary}\n")

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.use_ternary = use_ternary
        
        # Initialize with ternary weights if specified
        if use_ternary:
            self.inv_freq = nn.Parameter(self._get_ternary_init(dim), requires_grad=False)
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype())

    def _get_ternary_init(self, dim):
        # Ternary initialization (-1, 0, 1)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        ternary_weights = torch.bernoulli(torch.abs(inv_freq)) * torch.sign(inv_freq)
        return ternary_weights

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Detach to ensure no computation graph is kept!
        self.register_buffer("cos_cached", emb.cos().to(dtype).detach(), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype).detach(), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
