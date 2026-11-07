"""The Gated DeltaNet fix (MOSS Image's text encoder) against the rule's definition, token by token."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mflux.models.common_models.qwen3_5.qwen3_5_gated_delta_net")

from dmlx.adapters import mflux_fixes  # noqa: E402


def _recurrence(q, k, v, g, beta):
    """S_t = e^g_t S_{t-1} + beta_t k_t (v_t - e^g_t S_{t-1}^T k_t)^T;  o_t = S_t^T q_t / sqrt(d), in float64."""
    q, k, v, g, beta = (np.asarray(x, dtype=np.float64) for x in (q, k, v, g, beta))
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    batch, length, heads, dim = k.shape
    state = np.zeros((batch, heads, dim, v.shape[-1]))
    out = np.zeros((batch, length, heads, v.shape[-1]))
    for t in range(length):
        state = state * np.exp(g[:, t])[..., None, None]
        delta = (v[:, t] - np.einsum("bhd,bhde->bhe", k[:, t], state)) * beta[:, t][..., None]
        state = state + np.einsum("bhd,bhe->bhde", k[:, t], delta)
        out[:, t] = np.einsum("bhd,bhde->bhe", q[:, t] * dim**-0.5, state)
    return out


@pytest.mark.parametrize("length", [27, 150, 300])  # one chunk, several, and past where mflux went NaN
def test_the_chunked_rule_matches_the_recurrence(length):
    rng = np.random.default_rng(length)
    q, k, v = (rng.normal(size=(1, length, 4, 32)).astype(np.float32) for _ in range(3))
    g = (-np.abs(rng.normal(size=(1, length, 4))) * 0.05).astype(np.float32)
    beta = (1 / (1 + np.exp(-rng.normal(size=(1, length, 4))))).astype(np.float32)
    got = np.array(mflux_fixes._gated_delta_rule(*(mx.array(x) for x in (q, k, v, g, beta))))
    assert np.isfinite(got).all()
    assert np.abs(got - _recurrence(q, k, v, g, beta)).max() < 1e-3


def test_the_fix_replaces_mfluxs_rule():
    from mflux.models.common_models.qwen3_5.qwen3_5_gated_delta_net import Qwen35GatedDeltaNet

    mflux_fixes.install_gated_delta_fix()
    assert Qwen35GatedDeltaNet._gated_delta_rule is mflux_fixes._gated_delta_rule
