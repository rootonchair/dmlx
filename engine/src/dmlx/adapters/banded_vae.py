"""Exact, low-memory decode for the FLUX.2 VAE.

The stock decoder needs ~8.6 GB at 1024x1024: its 3x3 convolutions unroll their input about
nine-fold, and GroupNorm makes float32 copies of full-resolution feature maps. Tiling the
decode cuts memory but gives each tile its own GroupNorm statistics, which shifts colour
between tiles on this decoder.

This decoder keeps whole-image GroupNorm statistics (gathered band by band) and runs every
3x3 convolution in horizontal bands of rows, each reading one extra row above and below, so
only one band's temporaries exist at a time. The result matches the untiled decode up to
float rounding: MLX picks its convolution kernel by input shape, so bands accumulate in a
different order (mean difference under one 8-bit level, no regional colour shift).
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

_BAND_BYTES = 96 * 1024**2  # budget for one band's unrolled convolution input


def _rows_per_band(width: int, channels: int, height: int) -> int:
    return max(8, min(height, _BAND_BYTES // max(1, width * channels * 9 * 2)))


def _bands(height: int, rows: int):
    for r0 in range(0, height, rows):
        yield r0, min(r0 + rows, height)


def _norm_silu(x: mx.array, norm: nn.GroupNorm, precision, cast_after: bool = False, silu: bool = True):
    """Rows of silu(GroupNorm(x)) with statistics over the whole map; returns `rows(a, b)`.

    Mean and variance are gathered band by band in float32 (two passes), so neither a float32
    copy nor a normalized copy of the full map is ever held.
    """
    batch, height, width, channels = x.shape
    groups = norm.num_groups
    group_size = channels // groups
    rows = _rows_per_band(width, channels, height)
    count = height * width * group_size

    def grouped(a, b):
        return x[:, a:b].astype(mx.float32).reshape(batch, b - a, width, groups, group_size)

    total = mx.zeros((batch, 1, 1, groups, 1), dtype=mx.float32)
    for r0, r1 in _bands(height, rows):
        total = total + grouped(r0, r1).sum(axis=(1, 2, 4), keepdims=True)
        mx.eval(total)
    mean = total / count
    squares = mx.zeros_like(total)
    for r0, r1 in _bands(height, rows):
        squares = squares + ((grouped(r0, r1) - mean) ** 2).sum(axis=(1, 2, 4), keepdims=True)
        mx.eval(squares)
    scale = mx.rsqrt(squares / count + norm.eps)

    def band(a, b):
        y = ((grouped(a, b) - mean) * scale).reshape(batch, b - a, width, channels)
        y = (norm.weight * y + norm.bias).astype(precision)
        if silu:
            y = nn.silu(y)
        return y.astype(precision) if cast_after else y

    return band


def _conv_rows(source, conv: nn.Conv2d, height: int, r0: int, r1: int) -> mx.array:
    """Rows [r0, r1) of a 3x3, padding-1 convolution; `source(a, b)` returns input rows [a, b)."""
    a, b = max(r0 - 1, 0), min(r1 + 1, height)
    band = source(a, b)
    top, bottom = int(r0 == 0), int(r1 == height)
    if top or bottom:
        band = mx.pad(band, [(0, 0), (top, bottom), (0, 0), (0, 0)])
    y = mx.conv2d(band, conv.weight, stride=1, padding=(0, 1))
    if "bias" in conv:
        y = y + conv.bias
    return y


def _banded(height: int, width: int, channels: int, make_band) -> mx.array:
    out = None
    for r0, r1 in _bands(height, _rows_per_band(width, channels, height)):
        part = make_band(r0, r1)
        if out is None:
            out = mx.zeros((part.shape[0], height, *part.shape[2:]), dtype=part.dtype)
        out[:, r0:r1] = part  # written in place; avoids holding every band plus a concatenated copy
        mx.eval(out)
    return out


def _resnet(x: mx.array, block, precision) -> mx.array:
    batch, height, width, channels = x.shape
    pre1 = _norm_silu(x, block.norm1, precision)
    h = _banded(height, width, channels, lambda r0, r1: _conv_rows(pre1, block.conv1, height, r0, r1))
    del pre1
    pre2 = _norm_silu(h, block.norm2, precision)
    mid_channels = h.shape[-1]
    del h

    def out_rows(r0, r1):
        residual = x[:, r0:r1]
        if block.conv_shortcut is not None:
            residual = block.conv_shortcut(residual)
        return _conv_rows(pre2, block.conv2, height, r0, r1) + residual

    return _banded(height, width, mid_channels, out_rows)


def _upsample(x: mx.array, block) -> mx.array:
    """Nearest 2x upsample followed by a 3x3 convolution, computed band by band at the high resolution."""
    batch, height, width, channels = x.shape
    out_h = height * 2

    def source(a, b):
        lo_a, lo_b = a // 2, (b - 1) // 2 + 1
        up = mx.repeat(mx.repeat(x[:, lo_a:lo_b], 2, axis=1), 2, axis=2)
        return up[:, a - 2 * lo_a : b - 2 * lo_a]

    return _banded(out_h, width * 2, channels, lambda r0, r1: _conv_rows(source, block.conv, out_h, r0, r1))


_QUERY_CHUNK = 1024


def _attention(x: mx.array, block, precision) -> mx.array:
    """Single-head self-attention of the mid block, computed for a chunk of query positions at a time."""
    from mlx.core.fast import scaled_dot_product_attention

    batch, height, width, channels = x.shape
    tokens = height * width
    normed = _norm_silu(x, block.group_norm, precision, silu=False)(0, height).reshape(batch, tokens, channels)
    q = mx.transpose(block.to_q(normed).reshape(batch, tokens, 1, channels), (0, 2, 1, 3))
    k = mx.transpose(block.to_k(normed).reshape(batch, tokens, 1, channels), (0, 2, 1, 3))
    v = mx.transpose(block.to_v(normed).reshape(batch, tokens, 1, channels), (0, 2, 1, 3))
    mx.eval(q, k, v)
    del normed
    scale = 1 / mx.sqrt(q.shape[-1])
    flat = x.reshape(batch, tokens, channels)
    out = mx.zeros_like(flat)
    for t0 in range(0, tokens, _QUERY_CHUNK):
        t1 = min(t0 + _QUERY_CHUNK, tokens)
        attended = scaled_dot_product_attention(q[:, :, t0:t1], k, v, scale=scale)
        attended = mx.transpose(attended, (0, 2, 1, 3)).reshape(batch, t1 - t0, channels)
        out[:, t0:t1] = flat[:, t0:t1] + block.to_out(attended)
        mx.eval(out)
    return out.reshape(batch, height, width, channels)


def decode(vae, latents: mx.array) -> mx.array:
    """Drop-in for Flux2VAE.decode: same input (NCHW latents) and output (NCHW image)."""
    from mflux.models.common.config.model_config import ModelConfig

    precision = ModelConfig.precision
    if latents.ndim == 5:
        latents = latents[:, :, 0, :, :]
    latents = (latents / vae.scaling_factor) + vae.shift_factor
    latents = mx.transpose(vae.post_quant_conv(mx.transpose(latents, (0, 2, 3, 1))), (0, 3, 1, 2))
    decoder = vae.decoder
    x = mx.transpose(decoder.conv_in(latents), (0, 2, 3, 1))
    mx.eval(x)
    mid = decoder.mid_block
    x = _resnet(x, mid.resnets[0], precision)
    for attention in mid.attentions:
        mx.clear_cache()
        x = _attention(x, attention, precision)
    x = _resnet(x, mid.resnets[1], precision)
    for up_block in decoder.up_blocks:
        for resnet in up_block.resnets:
            mx.clear_cache()  # cached buffers from the previous resolution would be reused oversized
            x = _resnet(x, resnet, precision)
        for upsampler in up_block.upsamplers:
            mx.clear_cache()
            x = _upsample(x, upsampler)
    mx.clear_cache()
    batch, height, width, channels = x.shape
    pre = _norm_silu(x, decoder.conv_norm_out, precision, cast_after=True)
    del x
    out = _banded(height, width, channels, lambda r0, r1: _conv_rows(pre, decoder.conv_out, height, r0, r1))
    return mx.transpose(out, (0, 3, 1, 2))
