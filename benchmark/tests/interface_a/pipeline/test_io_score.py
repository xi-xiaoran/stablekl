from __future__ import annotations
from fractions import Fraction as F
import numpy as np
import pytest
from klbench.fp import BINARY16, round_fraction
from klbench.pipeline_io import load_unit, save_unit, verify
from klbench.score.runner import _cos_l2, _d, _eq, _wilson, score_one, _gradient_coverage, _cert_failure_count

def _fixed():
    t = round_fraction(F(1, 16), BINARY16)
    return {'sample_id': np.asarray(['id0', 'id1', 'id2']), 'kl_status': np.asarray(['CERTIFIED'] * 3), 'kl_truth_bits': np.asarray([t, t, 0], dtype=np.uint16), 'kl_lower': np.asarray([str(F(1, 16)), str(F(1, 16)), '0']), 'kl_upper': np.asarray([str(F(1, 16)), str(F(1, 16)), '0']), 'kl_overflow': np.asarray([False] * 3), 'group': np.asarray(['B_1', 'B_1', 'identity']), 'grad__P__probs__bits': np.asarray([[0], [0], [0]], dtype=np.uint16), 'grad__P__probs__status': np.asarray([['CERTIFIED']] * 3), 'grad__P__probs__exact_zero': np.asarray([[True]] * 3)}

def _observed():
    return {'sample_id': np.asarray(['id0', 'id1', 'id2']), 'status': np.asarray(['OK', 'CALL_ERROR', 'OK']), 'kl_bits': np.asarray([11264, 0, 0], dtype=np.uint16), 'grad__P__probs__bits': np.asarray([[0], [0], [15360]], dtype=np.uint16), 'grad__P__probs__status': np.asarray(['OK', 'GRAD_MISSING', 'OK'])}

def test_metric_error_keeps_call_errors_in_denominator():
    fixed = _fixed()
    observed = _observed()
    first, _, _ = score_one(fixed, observed, 0, 'fp16')
    assert first['KL_CR'] is False
    failed, _, _ = score_one(fixed, observed, 1, 'fp16')
    assert failed['KL_CR'] is True
    assert failed['G_CR:P.probs[0]'] is True
    anomalous, _, _ = score_one(fixed, observed, 2, 'fp16')
    assert anomalous['G_SPURIOUS:P.probs[0]'] is True
    assert anomalous['KL_NONZERO_AT_IDENTITY'] is False

def test_signed_zero_equality_and_normal_range():
    assert _eq(_d(32768, BINARY16), _d(0, BINARY16))
    assert not _eq(_d(32256, BINARY16), _d(32256, BINARY16))
    assert not _eq(_d(31744, BINARY16), _d(64512, BINARY16))

def test_cosine_and_l2_applicability():
    zero = _d(0, BINARY16)
    one = _d(15360, BINARY16)
    assert _cos_l2([zero], [zero]) == (None, None)
    relative, cos = _cos_l2([zero, one], [one, zero])
    assert relative > 0 and cos == 0

def test_wilson_bounds_and_na():
    assert _wilson(0, 0) == (None, None)
    lo, hi = _wilson(7, 10)
    assert 0 < lo < 0.7 < hi < 1

def test_atomic_npz_manifest_and_resume(tmp_path):
    root = tmp_path
    (root / 'src' / 'klbench').mkdir(parents=True)
    p = root / 'example.npz'
    data = {'numbers': np.asarray([1, 2, 3], dtype=np.uint16), 'rational': np.asarray(['3/7'])}
    assert save_unit(p, data, {'stage': 'test'}, project_root=root)
    loaded, meta = load_unit(p)
    assert loaded['rational'].tolist() == ['3/7']
    assert meta['stage'] == 'test'
    assert not save_unit(p, data, {'stage': 'different'}, project_root=root)
    p.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='SHA256'):
        verify(p)

def test_f6_backward_unsupported_excludes_only_gradients():
    frozen = _fixed()
    raw = _observed()
    raw['status'] = raw['status'].astype('U32')
    raw['grad__P__probs__status'] = raw['grad__P__probs__status'].astype('U32')
    raw['status'][0] = 'BACKWARD_UNSUPPORTED_CPU'
    raw['grad__P__probs__status'][0] = 'BACKWARD_UNSUPPORTED_CPU'
    metrics, _, _ = score_one(frozen, raw, 0, 'fp16')
    assert metrics['KL_CR'] is False
    assert 'G_CR:P.probs[0]' not in metrics
    coverage = _gradient_coverage(frozen, raw)
    assert coverage['eligible_components'] == 3
    assert coverage['unsupported_components'] == 1
    assert coverage['covered_components'] == 2
    assert coverage['coverage'] == pytest.approx(2 / 3)

def test_summary_cert_failure_counter_is_not_all_rejections():
    info = {'rejections': {'UNCERT_BIN': 2, 'UNCERT_GRAD': 3, 'XCHECK_MISMATCH': 1, 'BIN_FULL': 17, 'DOMAIN': 5, 'ROUNDED_TO_IDENTICAL': 6}}
    assert _cert_failure_count(info) == 6
