import torch
from mmfreelm.ops.gla import chunk_gla_torch, naive_gla

def test_all_three_agree():
    torch.manual_seed(0)
    B, H, T, D = 4, 8, 512, 128
    device = "cuda"

    q = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)
    f = 0.9 + 0.1 * torch.rand(B, H, T, D, device=device)
    log_f, k = f.log(), 1.0 - f

    chunked, _ = chunk_gla_torch(q, k, v, log_f)

    # sequential ground truth on a shorter sequence (the loop is slow)
    reference = naive_gla(q[:, :, :128], k[:, :, :128],
                          v[:, :, :128], log_f[:, :, :128])
    error = ((chunked[:, :, :128] - reference).abs().max()
             / reference.abs().max()).item()
    assert error < 1e-4, f"chunked vs naive: {error}"

    try:
        from fla.ops.gla import chunk_gla
    except ImportError:
        return

    # fla >= 0.5 wants [B, T, H, D]. State stays [B, H, K, V].
    to_fla = lambda t: t.transpose(1, 2).contiguous()
    triton, _ = chunk_gla(
        to_fla(q).bfloat16(), to_fla(k).bfloat16(),
        to_fla(v).bfloat16(), to_fla(log_f), scale=1.0,
    )
    triton = triton.transpose(1, 2)
    error = ((chunked - triton.float()).abs().max()
             / chunked.abs().max()).item()
    # bf16 vs fp32; measured 0.0041 when correct, 1.03 when the layout is wrong
    assert error < 0.05, f"chunked vs fla: {error} -- check tensor layout"
