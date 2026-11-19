from __future__ import annotations
import ast
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp
import pytest
from flint import arb
from klbench.fp import BINARY16, BINARY32, round_fraction
from klbench.ref.arbutil import arb_bounds, arb_precision
from klbench.ref.certify import CertStatus, certify_sample
from klbench.ref.configs import get_config
from klbench.ref.configs.common import rational_power_equal
from klbench.ref.configs.special import trigamma_arb
from klbench.ref.configs.geometric import DefinitionSumUnavailable
IDS = ('binomial_probs', 'binomial_logits', 'geometric_probs', 'geometric_logits', 'poisson', 'exponential', 'gamma', 'beta', 'uniform', 'pareto', 'cauchy', 'gumbel', 'laplace')

def _case(config_id: str, i: int):
    if i >= 40:
        return _stress_case(config_id, i)
    a = F(8 + i % 9, 8)
    b = F(9 + 3 * i % 11, 8)
    if config_id == 'binomial_probs':
        n = (1, 10, 100, 1000)[i % 4]
        return ({'total_count': [F(n)], 'probs': [F(3 + i % 9, 16)]}, {'total_count': [F(n)], 'probs': [F(4 + 3 * i % 9, 16)]})
    if config_id == 'binomial_logits':
        n = (1, 10, 100, 1000)[i % 4]
        return ({'total_count': [F(n)], 'logits': [F(i % 9 - 4, 8)]}, {'total_count': [F(n)], 'logits': [F(i % 9 - 4, 8) + F(1 + i % 5, 64)]})
    if config_id == 'geometric_probs':
        return ({'probs': [F(2 + i % 9, 16)]}, {'probs': [F(3 + 3 * i % 9, 16)]})
    if config_id == 'geometric_logits':
        return ({'logits': [F(i % 9 - 4, 8)]}, {'logits': [F(i % 9 - 4, 8) + F(1 + i % 5, 64)]})
    if config_id in ('poisson', 'exponential'):
        return ({'rate': [a]}, {'rate': [b]})
    if config_id == 'gamma':
        return ({'concentration': [a], 'rate': [b]}, {'concentration': [F(9 + 5 * i % 13, 8)], 'rate': [F(9 + 7 * i % 9, 8)]})
    if config_id == 'beta':
        return ({'concentration1': [a], 'concentration0': [b]}, {'concentration1': [F(9 + 5 * i % 13, 8)], 'concentration0': [F(9 + 7 * i % 9, 8)]})
    if config_id == 'uniform':
        low = F(i % 5 - 2, 4)
        high = low + F(1 + i % 4, 2)
        return ({'low': [low], 'high': [high]}, {'low': [low - F(1 + i % 5, 16)], 'high': [high + F(1 + i * 3 % 5, 16)]})
    if config_id == 'pareto':
        return ({'scale': [F(4 + i % 5, 2)], 'alpha': [a]}, {'scale': [F(1 + i % 3, 2)], 'alpha': [b]})
    if config_id in ('cauchy', 'gumbel', 'laplace'):
        m = F(i % 7 - 3, 4)
        return ({'loc': [m], 'scale': [a]}, {'loc': [m + F(0 if i % 11 == 0 else 1 + i % 5, 8)], 'scale': [b]})
    raise AssertionError(config_id)

def _stress_case(config_id: str, i: int):
    p, q = _case(config_id, i - 40)
    if i < 44:
        key = 'probs' if 'probs' in q else 'logits' if 'logits' in q else 'concentration' if 'concentration' in q else 'concentration1' if 'concentration1' in q else 'rate' if 'rate' in q else 'alpha' if 'alpha' in q else 'high' if 'high' in q else 'loc'
        q = {k: list(v) for k, v in p.items()}
        q[key][0] += F(1, 2 ** 20)
        return (p, q)
    r2 = i >= 47
    tiny = F(1, 65536 if r2 else 256)
    if config_id in ('binomial_probs', 'geometric_probs'):
        p['probs'] = [tiny]
        q['probs'] = [tiny * 2]
    elif config_id in ('binomial_logits', 'geometric_logits'):
        p['logits'] = [F(-12 if r2 else -6)]
        q['logits'] = [p['logits'][0] + F(1, 8)]
    elif config_id in ('poisson', 'exponential'):
        p['rate'] = [tiny]
        q['rate'] = [tiny * 2]
    elif config_id == 'gamma':
        p = {'concentration': [F(1, 16 if r2 else 8)], 'rate': [tiny]}
        q = {'concentration': [F(1, 8 if r2 else 4)], 'rate': [tiny * 2]}
    elif config_id == 'beta':
        p = {'concentration1': [F(1, 16 if r2 else 8)], 'concentration0': [F(3, 2)]}
        q = {'concentration1': [F(1, 8 if r2 else 4)], 'concentration0': [F(2)]}
    elif config_id == 'uniform':
        p = {'low': [F(1)], 'high': [F(1) + tiny]}
        q = {'low': [F(1) - tiny], 'high': [F(1) + 2 * tiny]}
    elif config_id == 'pareto':
        p = {'scale': [tiny * 2], 'alpha': [F(1024 if r2 else 128)]}
        q = {'scale': [tiny], 'alpha': [F(2048 if r2 else 256)]}
    else:
        p = {'loc': [F(10)], 'scale': [tiny]}
        q = {'loc': [F(10) + tiny / 4], 'scale': [tiny * 2]}
    return (p, q)

def _mpf(x: F):
    return mp.mpf(x.numerator) / mp.mpf(x.denominator)

def _pair_to_bits(p, q, fmt):
    return ({k: [round_fraction(v, fmt) for v in values] for k, values in p.items()}, {k: [round_fraction(v, fmt) for v in values] for k, values in q.items()})

def _small_case(config_id: str):
    if config_id in ('binomial_probs', 'binomial_logits'):
        key = 'probs' if config_id.endswith('probs') else 'logits'
        p = {'total_count': [F(10)], key: [F(1, 2)]}
    elif config_id.startswith('geometric_'):
        p = {'probs' if config_id.endswith('probs') else 'logits': [F(1, 2)]}
    elif config_id in ('poisson', 'exponential'):
        p = {'rate': [F(1)]}
    elif config_id == 'gamma':
        p = {'concentration': [F(2)], 'rate': [F(1)]}
    elif config_id == 'beta':
        p = {'concentration1': [F(2)], 'concentration0': [F(2)]}
    elif config_id == 'uniform':
        p = {'low': [F(0)], 'high': [F(1)]}
    elif config_id == 'pareto':
        p = {'scale': [F(1)], 'alpha': [F(2)]}
    else:
        p = {'loc': [F(0)], 'scale': [F(1)]}
    key = 'probs' if 'probs' in p else 'logits' if 'logits' in p else 'concentration' if 'concentration' in p else 'concentration1' if 'concentration1' in p else 'alpha' if 'alpha' in p else 'rate' if 'rate' in p else 'high' if 'high' in p else 'loc'
    if config_id in ('cauchy', 'laplace', 'gumbel'):
        key = 'loc'
    return (p, key)

@pytest.mark.parametrize('config_id', IDS)
def test_interfaces_and_arb_identity(config_id):
    config = get_config(config_id)
    p, _ = _small_case(config_id)
    assert config.domain_ok(p, p) is True
    assert config.semantic_equal(p, p)
    assert callable(config.kl_definition_check)
    assert callable(config.kl_mpmath_alt)
    with arb_precision(256):
        lo, hi = arb_bounds(config.kl_arb(p, p, 256))
    assert lo <= 0 <= hi
    g = config.grad_arb(p, p, 256)
    assert set(g) == {'P', 'Q'}
    for side in ('P', 'Q'):
        assert set(g[side]) == set(config.exact_zero_mask(p, p)[side])
        for name, values in g[side].items():
            assert len(values) == 1
            if config.exact_zero_mask(p, p)[side][name][0]:
                l, u = arb_bounds(values[0])
                assert l <= 0 <= u

@pytest.mark.parametrize('config_id', IDS)
@pytest.mark.parametrize('fmt', [BINARY16, BINARY32], ids=['fp16', 'fp32'])
def test_tiny_ulp_and_identity_certificates(config_id, fmt):
    p, key = _small_case(config_id)
    pbits, qbits = _pair_to_bits(p, p, fmt)
    identity = certify_sample(config_id, fmt, pbits, pbits, sample_type='identity')
    assert identity.kl.status == CertStatus.CERTIFIED
    assert identity.kl.truth_bits == 0
    assert identity.kl.group == 'identity'
    nondiff_expected = config_id in ('uniform', 'pareto')
    statuses = [g.status for side in identity.gradients.values() for x in side.values() for g in x]
    assert all((s in (CertStatus.NONDIFF, CertStatus.CERTIFIED) for s in statuses))
    assert (CertStatus.NONDIFF in statuses) == nondiff_expected
    assert all((g.truth_bits == 0 for side in identity.gradients.values() for x in side.values() for g in x if g.status == CertStatus.CERTIFIED))
    qbits = {k: list(v) for k, v in pbits.items()}
    qbits[key][0] += 1
    one_ulp = certify_sample(config_id, fmt, pbits, qbits, sample_type='ulp')
    assert not one_ulp.semantic_identity
    assert one_ulp.kl.status == CertStatus.CERTIFIED
    assert all((g.status in (CertStatus.CERTIFIED, CertStatus.NONDIFF) for side in one_ulp.gradients.values() for x in side.values() for g in x))

def test_support_nondifferentiability_and_smooth_laplace_cusp():
    uniform = get_config('uniform')
    p = {'low': [F(0)], 'high': [F(1)]}
    q = {'low': [F(0)], 'high': [F(2)]}
    mask = uniform.nondiff_mask(p, q)
    assert mask['P']['low'] == [True] and mask['Q']['low'] == [True]
    assert mask['P']['high'] == [False] and mask['Q']['high'] == [False]
    assert uniform.domain_ok(p, {'low': [F(1)], 'high': [F(2)]}) is not True
    pareto = get_config('pareto')
    p = {'scale': [F(2)], 'alpha': [F(1)]}
    q = {'scale': [F(2)], 'alpha': [F(2)]}
    assert pareto.nondiff_mask(p, q)['P']['scale'] == [True]
    assert pareto.nondiff_mask(p, q)['Q']['scale'] == [True]
    laplace = get_config('laplace')
    p = {'loc': [F(1)], 'scale': [F(2)]}
    q = {'loc': [F(1)], 'scale': [F(1)]}
    assert laplace.exact_zero_mask(p, q)['P']['loc'] == [True]
    assert laplace.exact_grad(p, q)['P']['loc'] == [F(0)]

def test_rational_nonidentity_kl_and_large_exponent_logic():
    pareto = get_config('pareto')
    p = {'scale': [F(2)], 'alpha': [F(1, 2)]}
    q = {'scale': [F(1)], 'alpha': [F(1)]}
    assert pareto.exact_kl(p, q) == F(1)
    with arb_precision(256):
        lo, hi = arb_bounds(pareto.kl_arb(p, q, 256))
    assert lo <= F(1) <= hi
    beta = get_config('beta')
    p = {'concentration1': [F(2)], 'concentration0': [F(3)]}
    q = {'concentration1': [F(3)], 'concentration0': [F(2)]}
    assert beta.exact_kl(p, q) == F(1, 2)
    with arb_precision(256):
        lo, hi = arb_bounds(beta.kl_arb(p, q, 256))
    assert lo <= F(1, 2) <= hi
    assert rational_power_equal(F(1, 2), F(2 ** 120), F(1, 2)) is False
    assert rational_power_equal(F(1, 16), F(1, 2), F(1, 4))

@pytest.mark.parametrize('config_id', IDS)
def test_exact_rationals_within_arb_intervals(config_id):
    config = get_config(config_id)
    p, q = _case(config_id, 5)
    with arb_precision(256):
        grads = config.grad_arb(p, q, 256)
        exact = config.exact_grad(p, q) if hasattr(config, 'exact_grad') else {}
        for side in exact:
            for name, values in exact[side].items():
                for j, number in enumerate(values):
                    if number is not None:
                        lo, hi = arb_bounds(grads[side][name][j])
                        assert lo <= number <= hi, (config_id, side, name, number, lo, hi)
        if hasattr(config, 'exact_kl'):
            number = config.exact_kl(p, q)
            if number is not None:
                lo, hi = arb_bounds(config.kl_arb(p, q, 256))
                assert lo <= number <= hi

def test_normal_exact_grad_extension_matches_arb():
    cfg = get_config('normal')
    p = {'loc': [F(3, 2)], 'scale': [F(7, 4)]}
    q = {'loc': [F(1, 4)], 'scale': [F(2)]}
    exact = cfg.exact_grad(p, q)
    with arb_precision(256):
        balls = cfg.grad_arb(p, q, 256)
        for side in ('P', 'Q'):
            for name in ('loc', 'scale'):
                lo, hi = arb_bounds(balls[side][name][0])
                assert lo <= exact[side][name][0] <= hi

@pytest.mark.parametrize('x', [F(1, 4), F(1), F(7, 4), F(5), F(100)])
def test_hurwitz_zeta_trigamma_matches_mpmath(x):
    with arb_precision(256), mp.workdps(95):
        lo, hi = arb_bounds(trigamma_arb(arb(x.numerator) / arb(x.denominator)))
        expected = mp.polygamma(1, _mpf(x))
        assert _mpf(lo) - mp.mpf('1e-65') <= expected <= _mpf(hi) + mp.mpf('1e-65')

@pytest.mark.parametrize('config_id', IDS)
@pytest.mark.slow
def test_50_independent_definition_sums_or_integrals(config_id):
    cfg = get_config(config_id)
    skipped_unverifiable = 0
    actually_verified = 0
    with mp.workdps(90):
        for i in range(50):
            p, q = _case(config_id, i)
            assert cfg.domain_ok(p, q) is True
            try:
                direct = cfg.kl_definition_check(p, q, 90)
            except DefinitionSumUnavailable:
                skipped_unverifiable += 1
                continue
            actually_verified += 1
            closed = cfg.kl_mpmath_alt(p, q, 90)
            tail = cfg.definition_tail_bound(p, q, 90) if hasattr(cfg, 'definition_tail_bound') else mp.mpf(0)
            if config_id.startswith('geometric_'):
                assert tail is not None and tail <= mp.mpf('1e-30')
            assert abs(direct - closed) <= max(tail * mp.mpf('1.000001'), mp.mpf('1e-32')), (config_id, i, mp.nstr(direct, 20), mp.nstr(closed, 20), mp.nstr(tail, 20))
    assert actually_verified > 0, f'{config_id}: no meaningful definition-level verification'
    print(f'{config_id} definition checks: verified={actually_verified}, unverifiable_skipped={skipped_unverifiable}')

@pytest.mark.parametrize('config_id', IDS)
@pytest.mark.slow
def test_50_richardson_constructor_gradients(config_id):
    cfg = get_config(config_id)
    with mp.workdps(115):
        for i in range(50):
            p, q = _case(config_id, i)
            masks = cfg.nondiff_mask(p, q) if hasattr(cfg, 'nondiff_mask') else {}
            with arb_precision(320):
                grads = cfg.grad_arb(p, q, 320)
            exact = cfg.exact_grad(p, q) if hasattr(cfg, 'exact_grad') else {}
            for side, params in (('P', p), ('Q', q)):
                for name, values in grads[side].items():
                    if masks and masks[side][name][0]:
                        continue
                    if exact and exact[side][name][0] is not None:
                        lo, hi = arb_bounds(values[0])
                        assert lo <= exact[side][name][0] <= hi
                    x = params[name][0]
                    h = (abs(x) if x else F(1)) / 2 ** 32

                    def f(step):
                        pp = {k: list(v) for k, v in p.items()}
                        qq = {k: list(v) for k, v in q.items()}
                        (pp if side == 'P' else qq)[name][0] = x + step
                        return cfg.kl_mpmath_alt(pp, qq, 115)

                    def center(step):
                        return (f(step) - f(-step)) / (2 * _mpf(step))

                    def extrap(step):
                        return (4 * center(step / 2) - center(step)) / 3
                    v1 = extrap(h * 2)
                    v2 = extrap(h)
                    lo, hi = arb_bounds(values[0])
                    mid = (_mpf(lo) + _mpf(hi)) / 2
                    estimate = 16 * abs(v2 - v1) + mp.power(2, -100) * max(1, abs(v2))
                    assert abs(mid - v2) <= estimate + (_mpf(hi) - _mpf(lo)) / 2, (config_id, i, side, name, mp.nstr(v2, 18), mp.nstr(mid, 18))

def test_ref_tree_does_not_import_torch():
    ref_dir = Path(__file__).parents[3] / 'interface_a' / 'klbench' / 'ref'
    for path in ref_dir.rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all((n.name.split('.')[0] != 'torch' for n in node.names)), path
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split('.')[0] != 'torch', path
