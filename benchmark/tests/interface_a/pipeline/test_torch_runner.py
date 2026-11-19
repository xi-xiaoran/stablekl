from fractions import Fraction as F
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from klbench.fp import BINARY16, round_fraction
from klbench.torchrun.runner import _bits_to_tensor, _round_output, _call_batch

def test_bit_roundtrip_preserves_signed_zero_and_subnormals():
    raw = np.asarray([[0, 32768, 1, 31743]], dtype=np.uint16)
    result = _bits_to_tensor(raw, 'fp16', torch)
    assert np.array_equal(result.numpy().view(np.uint16), raw)

def test_b64_uses_single_rounding_on_midpoint():
    exact = F(1) + F(1, 2048) + F(1, 2 ** 40)
    t = torch.tensor([float(exact)], dtype=torch.float64)
    rounded, high, dtype = _round_output(t, 'fp16', 'B64')
    assert rounded == [15361]
    assert high == [format(int(t.numpy().view(np.uint64)[0]), '016x')]
    assert dtype == 'float64'

def test_official_kl_registry_and_grad_are_reached():
    data = {'sample_id': np.asarray(['x']), '_parameter_shapes': {'probs': {'shape': [1], 'differentiable': True}}, 'p__probs': np.asarray([[14336]], dtype=np.uint16), 'q__probs': np.asarray([[13312]], dtype=np.uint16)}
    results = _call_batch('bernoulli_probs', 'fp16', 'B64', data, [0], 'cpu')
    assert results[0]['status'] == 'OK', results
    assert results[0]['grad']['P__probs']['status'] == 'OK'
    assert results[0]['grad']['Q__probs']['status'] == 'OK'

def test_f6_backward_unsupported_cpu_preserves_forward(monkeypatch):
    data = {'sample_id': np.asarray(['x']), '_parameter_shapes': {'probs': {'shape': [1], 'differentiable': True}}, 'p__probs': np.asarray([[14336]], dtype=np.uint16), 'q__probs': np.asarray([[13312]], dtype=np.uint16)}

    def unsupported_backward(self, *args, **kwargs):
        raise RuntimeError('"some_kernel" not implemented for \'Half\'')
    monkeypatch.setattr(torch.Tensor, 'backward', unsupported_backward)
    result = _call_batch('bernoulli_probs', 'fp16', 'B64', data, [0], 'cpu')[0]
    assert result['status'] == 'BACKWARD_UNSUPPORTED_CPU'
    assert result['kl_bits'] is not None
    assert result['grad']['P__probs']['status'] == 'BACKWARD_UNSUPPORTED_CPU'
