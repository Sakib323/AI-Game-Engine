# -*- coding: utf-8 -*-
"""
Chunkwise-parallel gated linear attention, plain PyTorch. No Triton.

    S_t = diag(f_t) . S_{t-1} + k_t (x) v_t          S in R^{dk x dv}
    o_t = S_t^T q_t

Naively sequential in T. The chunkwise form below computes the same thing with
matmuls over chunks of C steps, reducing sequential depth from T to T/C.

With F_t the cumulative product of decays within a chunk:

    S_t = diag(F_t) S_0 + sum_{j<=t} diag(F_t / F_j) k_j v_j^T
    o_t = S_0^T (q_t . F_t) + sum_{j<=t} [(q_t . F_t) . (k_j / F_j)] v_j

so with q~_t = q_t * F_t and k~_j = k_j / F_j:

    o_t = S_0^T q~_t + sum_{j<=t} (q~_t . k~_j) v_j     <- masked matmul
    S_C = diag(F_C) [ S_0 + sum_j k~_j v_j^T ]          <- matmul

NUMERICAL NOTE: k~ divides by F_j, which grows as F decays. With chunk 64 and a
decay lower bound of 0.9, worst-case 1/F is ~800, fine in fp32. Lowering the
decay bound below ~0.85 requires a smaller chunk. The assertion guards this.
"""

import torch


def naive_gla(q, k, v, log_f):
    """Sequential reference. Testing only."""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    state = torch.zeros(B, H, Dk, Dv, device=q.device, dtype=torch.float32)
    outputs = []
    for t in range(T):
        decay = log_f[:, :, t].float().exp().unsqueeze(-1)
        update = (
            k[:, :, t].float().unsqueeze(-1) @ v[:, :, t].float().unsqueeze(-2)
        )
        state = decay * state + update
        outputs.append((q[:, :, t].float().unsqueeze(-2) @ state).squeeze(-2))
    return torch.stack(outputs, dim=2)


def chunk_gla_torch(q, k, v, log_f, chunk_size=64, initial_state=None,
                    output_final_state=False):
    """
    Args:
        q, k:   [B, H, T, Dk]
        v:      [B, H, T, Dv]
        log_f:  [B, H, T, Dk]   LOG-space decay, strictly negative
        initial_state: [B, H, Dk, Dv] or None
    Returns:
        (o, final_state_or_None) with o of shape [B, H, T, Dv]
    """
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    C = chunk_size

    assert T % C == 0, f"sequence length {T} must be divisible by chunk {C}"

    q = q.float()
    k = k.float()
    v = v.float()
    log_f = log_f.float()

    num_chunks = T // C
    q = q.reshape(B, H, num_chunks, C, Dk)
    k = k.reshape(B, H, num_chunks, C, Dk)
    v = v.reshape(B, H, num_chunks, C, Dv)
    log_f = log_f.reshape(B, H, num_chunks, C, Dk)

    cumulative = log_f.cumsum(dim=-2)

    max_inverse = (-cumulative).max().exp().item()
    assert max_inverse < 1e6, (
        f"decay too aggressive for chunk_size={C}: 1/F reaches {max_inverse:.2e}. "
        f"Lower chunk_size or raise the decay lower bound."
    )

    q_scaled = q * cumulative.exp()
    k_scaled = k * (-cumulative).exp()

    causal = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))

    state = (
        torch.zeros(B, H, Dk, Dv, device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float()
    )

    outputs = []
    for chunk in range(num_chunks):
        q_c = q_scaled[:, :, chunk]
        k_c = k_scaled[:, :, chunk]
        v_c = v[:, :, chunk]

        attention = (q_c @ k_c.transpose(-1, -2)).masked_fill(~causal, 0.0)
        output = attention @ v_c
        output = output + q_c @ state

        chunk_decay = cumulative[:, :, chunk, -1].exp()
        state = chunk_decay.unsqueeze(-1) * (
            state + k_c.transpose(-1, -2) @ v_c
        )
        outputs.append(output)

    o = torch.stack(outputs, dim=2).reshape(B, H, T, Dv)

    if output_final_state:
        return o, state
    return o, None
