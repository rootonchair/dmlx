"""Fixes to mflux code that dMLX installs at load time, until they are fixed upstream.

Gated DeltaNet (the Qwen3.5 text encoder of MOSS Image): mflux inverts (I - A) inside each 64-token chunk
with repeated squaring, (I + A)(I + A^2)(I + A^4)...(I + A^32). On some prompts the powers overflow float32
in a late layer and inf - inf gives NaN. The NaN then spreads through the chunk and the next layers, so
the whole embedding, and the image, come out invalid: a black square. Measured 2026-11-07 on the q4 pack:
any prompt of about 250 tokens or more, plain prose included, went NaN in layer 25.

The reference (transformers' torch_chunk_gated_delta_rule) solves the same triangular system row by row,
which stays bounded. This replaces only that step; the rest of mflux's chunked rule is unchanged.
"""

from __future__ import annotations

import mlx.core as mx

_INSTALLED = False


def _inverse_rows(a: mx.array, chunk: int) -> mx.array:
    """(I - A)^-1 for strictly lower triangular A, row by row as in the reference."""
    rows = [mx.zeros_like(a[..., 0, :])]
    for i in range(1, chunk):
        prior = mx.stack(rows, axis=-2)  # (..., i, chunk): rows 0..i-1 of the result, minus I
        own = a[..., i, :]
        # T_i = A_i + A_i[:i] @ T[:i]  (T without its identity; the identity is added at the end)
        rows.append(own + (own[..., None, :i] @ prior)[..., 0, :])
    return mx.stack(rows, axis=-2) + mx.eye(chunk, dtype=mx.float32)


def _gated_delta_rule(query: mx.array, key: mx.array, value: mx.array, g: mx.array, beta: mx.array) -> mx.array:
    from mflux.models.common_models.qwen3_5.qwen3_5_gated_delta_net import Qwen35GatedDeltaNet

    chunk = Qwen35GatedDeltaNet.CHUNK_SIZE
    query = Qwen35GatedDeltaNet._l2_normalize(query.astype(mx.float32))
    key = Qwen35GatedDeltaNet._l2_normalize(key.astype(mx.float32))
    query, key, value = (x.astype(mx.float32).transpose(0, 2, 1, 3) for x in (query, key, value))
    g, beta = (x.astype(mx.float32).transpose(0, 2, 1) for x in (g, beta))

    batch, heads, seq_len, key_dim = key.shape
    value_dim = value.shape[-1]
    pad = (chunk - seq_len % chunk) % chunk
    query, key, value = (mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)]) for x in (query, key, value))
    g, beta = (mx.pad(x, [(0, 0), (0, 0), (0, pad)]) for x in (g, beta))
    chunks = (seq_len + pad) // chunk

    query = query * (key_dim**-0.5)
    value_beta = value * beta[..., None]
    key_beta = key * beta[..., None]
    query, key, value_beta, key_beta = (
        x.reshape(batch, heads, chunks, chunk, -1) for x in (query, key, value_beta, key_beta)
    )
    g = mx.cumsum(g.reshape(batch, heads, chunks, chunk), axis=-1)

    lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    strictly_lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_), k=-1)
    decay = mx.where(lower, mx.exp(mx.where(lower, g[..., :, None] - g[..., None, :], 0.0)), 0.0)

    a = mx.where(strictly_lower, -((key_beta @ key.swapaxes(-1, -2)) * decay), 0.0)
    inverse = _inverse_rows(a, chunk)

    value_chunks = inverse @ value_beta
    key_cumulative_decay = inverse @ (key_beta * mx.exp(g)[..., None])

    state = mx.zeros((batch, heads, key_dim, value_dim), dtype=mx.float32)
    outputs = []
    for i in range(chunks):
        q_i, k_i, g_i = query[:, :, i], key[:, :, i], g[:, :, i]
        attn = (q_i @ k_i.swapaxes(-1, -2)) * decay[:, :, i]
        v_new = value_chunks[:, :, i] - key_cumulative_decay[:, :, i] @ state
        outputs.append((q_i * mx.exp(g_i)[..., None]) @ state + attn @ v_new)
        g_last = g_i[..., -1:]
        state = state * mx.exp(g_last)[..., None] + (k_i * mx.exp(g_last - g_i)[..., None]).swapaxes(-1, -2) @ v_new
    out = mx.stack(outputs, axis=2).reshape(batch, heads, chunks * chunk, value_dim)[:, :, :seq_len]
    return out.transpose(0, 2, 1, 3)


def install_gated_delta_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from mflux.models.common_models.qwen3_5.qwen3_5_gated_delta_net import Qwen35GatedDeltaNet
    except ImportError:  # an mflux without Qwen3.5
        return
    Qwen35GatedDeltaNet._gated_delta_rule = staticmethod(_gated_delta_rule)
    _INSTALLED = True
