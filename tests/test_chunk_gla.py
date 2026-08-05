# -*- coding: utf-8 -*-
"""
Correctness of the chunkwise-parallel GLA against a sequential reference.

`chunk_gla_torch` is the oracle used to validate any future custom kernel, so
it has to agree with `naive_gla` to tight tolerance across chunk sizes. When
`fla` is installed its Triton `chunk_gla` is checked against the same
reference, since HGRN2BitAttention prefers that kernel when it is importable.
"""

import pytest
import torch

from mmfreelm.ops.gla import chunk_gla_torch, naive_gla

RELATIVE_TOLERANCE = 1e-4


def _inputs(batch=2, heads=4, seq_len=256, head_dim=64, seed=0):
    """Decays in a realistic band: lower bound 0.9, matching decay_mode init."""
    generator = torch.Generator().manual_seed(seed)

    q = torch.randn(batch, heads, seq_len, head_dim, generator=generator)
    k = torch.randn(batch, heads, seq_len, head_dim, generator=generator)
    v = torch.randn(batch, heads, seq_len, head_dim, generator=generator)

    gate = torch.rand(
        batch, heads, seq_len, head_dim, generator=generator
    )
    decay = 0.9 + 0.1 * gate
    log_f = decay.log()

    return q, k, v, log_f


def _relative_error(actual, expected):
    return (
        (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    ).item()


@pytest.mark.parametrize("chunk_size", [32, 64, 128])
def test_chunk_matches_sequential(chunk_size):
    q, k, v, log_f = _inputs()

    expected = naive_gla(q, k, v, log_f)
    actual, _ = chunk_gla_torch(q, k, v, log_f, chunk_size=chunk_size)

    error = _relative_error(actual, expected)
    assert error < RELATIVE_TOLERANCE, (
        f"chunk_size={chunk_size} relative error {error:.3e}"
    )


@pytest.mark.parametrize("chunk_size", [32, 64])
def test_final_state_matches_sequential(chunk_size):
    """The carried state must match, or multi-chunk generation drifts."""
    q, k, v, log_f = _inputs(seq_len=128)

    _, half_state = chunk_gla_torch(
        q[:, :, :64], k[:, :, :64], v[:, :, :64], log_f[:, :, :64],
        chunk_size=chunk_size, output_final_state=True,
    )
    resumed, _ = chunk_gla_torch(
        q[:, :, 64:], k[:, :, 64:], v[:, :, 64:], log_f[:, :, 64:],
        chunk_size=chunk_size, initial_state=half_state,
    )

    expected = naive_gla(q, k, v, log_f)[:, :, 64:]

    error = _relative_error(resumed, expected)
    assert error < RELATIVE_TOLERANCE, (
        f"resumed-from-state relative error {error:.3e}"
    )


def test_triton_kernel_matches_sequential():
    """Skipped unless fla and a CUDA device are both available."""
    fla_gla = pytest.importorskip("fla.ops.gla")

    if not torch.cuda.is_available():
        pytest.skip("fla's chunk_gla requires CUDA")

    q, k, v, log_f = _inputs()
    expected = naive_gla(q, k, v, log_f).cuda()

    actual, _ = fla_gla.chunk_gla(
        q.cuda(), k.cuda(), v.cuda(), log_f.cuda(),
        scale=1.0,
    )

    error = _relative_error(actual, expected)
    assert error < RELATIVE_TOLERANCE, (
        f"fla chunk_gla relative error {error:.3e}"
    )
