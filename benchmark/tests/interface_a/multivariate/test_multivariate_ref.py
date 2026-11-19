from __future__ import annotations
import ast
import pathlib
import random
import subprocess
import sys
from fractions import Fraction as F
import mpmath as mp
import numpy as np
import pytest
from klbench.fp import BINARY16, BINARY32
from klbench.fp.bits import exact_value
from klbench.fp.rounding import round_fraction
from klbench.mv_dims import MV_SUBCONFIGS, split_quota
from klbench.ref.arbutil import arb_bounds
from klbench.ref.certify import CertStatus, certify_sample
from klbench.ref.configs import CONFIGS, MV_CONFIGS
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parents[2] / 'interface_a' / 'klbench'
rng = random.Random(7)

def q32(x):
    return F(float(np.float32(x)))
TOL = F(1, 10 ** 13)

def mid(ball):
    lo, hi = arb_bounds(ball)
    return (lo + hi) / 2

def rel_close(a, b, tol):
    a = mp.mpf(a.numerator) / a.denominator if isinstance(a, F) else a
    b = mp.mpf(b.numerator) / b.denominator if isinstance(b, F) else b
    return abs(a - b) <= tol * max(abs(a), abs(b), mp.mpf(10) ** (-300))

def test_new_truth_modules_never_import_torch():
    for name in ('contbern', 'categorical', 'dirichlet', 'mvn'):
        tree = ast.parse((SRC / 'ref' / 'configs' / f'{name}.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
                assert not any((m.split('.')[0] == 'torch' for m in mods)), name

def test_quota_split_sums_to_spec_quota():
    for base, subs in MV_SUBCONFIGS.items():
        for quota in (10000, 1000):
            assert sum((split_quota(s, quota) for s in subs)) == quota, base

def make_pair(config_id):
    c = CONFIGS[config_id]
    shapes = {n: int(np.prod(s['shape'])) for n, s in c.PARAM_SPEC.items()}
    if config_id.startswith('contbern_probs'):
        a = q32(rng.uniform(0.05, 0.95))
        return ({'probs': [a]}, {'probs': [q32(float(a) * 1.01)]})
    if config_id.startswith('contbern_logits'):
        a = q32(rng.uniform(-6, 6))
        return ({'logits': [a]}, {'logits': [q32(float(a) + 0.03)]})
    if config_id.startswith('categorical_probs'):
        x = [q32(rng.uniform(0.1, 3)) for _ in range(shapes['probs'])]
        return ({'probs': x}, {'probs': [q32(float(v) * np.exp(rng.uniform(-0.05, 0.05))) for v in x]})
    if config_id.startswith('categorical_logits'):
        z = [q32(rng.uniform(-5, 5)) for _ in range(shapes['logits'])]
        return ({'logits': z}, {'logits': [q32(float(v) + rng.uniform(-0.05, 0.05)) for v in z]})
    if config_id.startswith('dirichlet'):
        a = [q32(np.exp(rng.uniform(-3, 3))) for _ in range(shapes['concentration'])]
        return ({'concentration': a}, {'concentration': [q32(float(v) * 1.02) for v in a]})
    if config_id.startswith('mvn'):
        d = shapes['loc']
        low = [[q32(np.exp(rng.uniform(-1, 1))) if i == j else q32(rng.uniform(-0.4, 0.4)) if j < i else F(0) for j in range(d)] for i in range(d)]
        lq = [[q32(float(v) * 1.01) if j <= i else F(0) for j, v in enumerate(r)] for i, r in enumerate(low)]
        loc = [q32(rng.uniform(-5, 5)) for _ in range(d)]
        return ({'loc': loc, 'scale_tril': [v for r in low for v in r]}, {'loc': [q32(float(v) + 0.02) for v in loc], 'scale_tril': [v for r in lq for v in r]})
    d, n = (shapes['loc'], shapes['cov_factor'])
    w = [q32(rng.uniform(-1, 1)) for _ in range(n)]
    dg = [q32(np.exp(rng.uniform(-1, 1))) for _ in range(d)]
    loc = [q32(rng.uniform(-5, 5)) for _ in range(d)]
    return ({'loc': loc, 'cov_factor': w, 'cov_diag': dg}, {'loc': [q32(float(v) + 0.02) for v in loc], 'cov_factor': [q32(float(v) + 0.01) for v in w], 'cov_diag': [q32(float(v) * 1.01) for v in dg]})

@pytest.mark.parametrize('config_id', MV_CONFIGS)
def test_arb_matches_mpmath_alternative_form(config_id):
    c = CONFIGS[config_id]
    for _ in range(5):
        p, q = make_pair(config_id)
        assert c.domain_ok(p, q) is True
        ball = c.kl_arb(p, q, 256)
        lo, hi = arb_bounds(ball)
        assert lo > 0
        with mp.workdps(90):
            alt = c.kl_mpmath_alt(p, q, 90)
            centre = (lo + hi) / 2
            m = mp.mpf(centre.numerator) / centre.denominator
            assert abs(alt - m) <= mp.mpf(10) ** (-13) * abs(m), (config_id, alt, m)

@pytest.mark.parametrize('config_id', ['contbern_probs', 'contbern_logits'])
def test_contbern_definition_integral(config_id):
    c = CONFIGS[config_id]
    for _ in range(6):
        p, q = make_pair(config_id)
        with mp.workdps(40):
            integ = c.kl_definition_check(p, q, 40)
        assert rel_close(mid(c.kl_arb(p, q, 256)), integ, mp.mpf(10) ** (-13))

def test_contbern_near_half_definition_integral():
    c = CONFIGS['contbern_probs']
    for off in (F(0), F(1, 2 ** 24), F(-3, 2 ** 20), F(5, 2 ** 12)):
        p, q = ({'probs': [F(1, 2) + off]}, {'probs': [F(1, 2) + off + F(1, 2 ** 9)]})
        with mp.workdps(40):
            integ = c.kl_definition_check(p, q, 40)
        assert rel_close(mid(c.kl_arb(p, q, 256)), integ, mp.mpf(10) ** (-13))

def test_dirichlet_k2_equals_beta():
    beta, dk2 = (CONFIGS['beta'], CONFIGS['dirichlet_K2'])
    for _ in range(10):
        p, q = make_pair('dirichlet_K2')
        pb = {'concentration1': [p['concentration'][0]], 'concentration0': [p['concentration'][1]]}
        qb = {'concentration1': [q['concentration'][0]], 'concentration0': [q['concentration'][1]]}
        a, b = (mid(dk2.kl_arb(p, q, 256)), mid(beta.kl_arb(pb, qb, 256)))
        assert abs(a - b) <= TOL * abs(b)

def test_mvn_diagonal_equals_sum_of_normals():
    normal = CONFIGS['normal']
    for d in (2, 4):
        c = CONFIGS[f'mvn_d{d}']
        sp = [q32(np.exp(rng.uniform(-1, 1))) for _ in range(d)]
        sq = [q32(float(v) * 1.03) for v in sp]
        mu_p = [q32(rng.uniform(-3, 3)) for _ in range(d)]
        mu_q = [q32(float(v) + 0.1) for v in mu_p]
        diag = lambda s: [s[i] if i == j else F(0) for i in range(d) for j in range(d)]
        kl = mid(c.kl_arb({'loc': mu_p, 'scale_tril': diag(sp)}, {'loc': mu_q, 'scale_tril': diag(sq)}, 256))
        total = sum((mid(normal.kl_arb({'loc': [a], 'scale': [s]}, {'loc': [b], 'scale': [t]}, 256)) for a, b, s, t in zip(mu_p, mu_q, sp, sq)))
        assert abs(kl - total) <= TOL * abs(total)

def test_lowrank_zero_factor_is_diagonal_mvn():
    for d in (2, 4):
        for r in (1, 2):
            c = CONFIGS[f'lowrank_mvn_d{d}_r{r}']
            dg_p = [F(4) ** rng.randrange(-2, 3) for _ in range(d)]
            dg_q = [v * F(9, 4) for v in dg_p]
            mu_p = [q32(rng.uniform(-3, 3)) for _ in range(d)]
            mu_q = [q32(float(v) - 0.2) for v in mu_p]
            zero = [F(0)] * (d * r)
            kl = mid(c.kl_arb({'loc': mu_p, 'cov_factor': zero, 'cov_diag': dg_p}, {'loc': mu_q, 'cov_factor': zero, 'cov_diag': dg_q}, 256))
            m = CONFIGS[f'mvn_d{d}']
            sqrt2 = lambda v: F(int(round(float(v) ** 0.5 * 2 ** 20)), 2 ** 20)
            diag = lambda s: [sqrt2(s[i]) if i == j else F(0) for i in range(d) for j in range(d)]
            ref = mid(m.kl_arb({'loc': mu_p, 'scale_tril': diag(dg_p)}, {'loc': mu_q, 'scale_tril': diag(dg_q)}, 256))
            assert abs(kl - ref) <= TOL * abs(ref)

@pytest.mark.parametrize('config_id', [c for c in MV_CONFIGS if c.startswith(('mvn', 'lowrank', 'categorical_probs'))])
def test_exact_gradients_inside_arb_balls(config_id):
    c = CONFIGS[config_id]
    for _ in range(4):
        p, q = make_pair(config_id)
        ex, g = (c.exact_grad(p, q), c.grad_arb(p, q, 256))
        for side in ('P', 'Q'):
            for name, vals in ex[side].items():
                for i, v in enumerate(vals):
                    if v is not None:
                        lo, hi = arb_bounds(g[side][name][i])
                        assert lo <= v <= hi, (config_id, side, name, i)

def _bits(values, fmt):
    return [round_fraction(F(v), fmt) for v in values]

@pytest.mark.parametrize('fmt', [BINARY16, BINARY32])
def test_contbern_half_and_neighbours_certify(fmt):
    half = round_fraction(F(1, 2), fmt)
    for pb, qb in ((half, half + 1), (half - 1, half), (half, half - 1), (half + 1, half + 3)):
        cert = certify_sample('contbern_probs', fmt, {'probs': [pb]}, {'probs': [qb]}, sample_type='ulp')
        assert cert.kl.status == CertStatus.CERTIFIED
        assert all((v.status == CertStatus.CERTIFIED for s in cert.gradients.values() for vs in s.values() for v in vs))
    for zb, wb in ((0, 1), (1, 0), (32769 if fmt is BINARY16 else 2147483649, 1)):
        cert = certify_sample('contbern_logits', fmt, {'logits': [zb]}, {'logits': [wb]}, sample_type='ulp')
        assert cert.kl.status == CertStatus.CERTIFIED

@pytest.mark.parametrize('config_id', MV_CONFIGS)
def test_identity_is_exact_zero(config_id):
    p, _ = make_pair(config_id)
    bits = {n: _bits(v, BINARY32) for n, v in p.items()}
    cert = certify_sample(config_id, 'fp32', bits, {k: list(v) for k, v in bits.items()}, sample_type='identity')
    assert cert.kl.exact_zero and cert.kl.truth_bits == 0
    for side in cert.gradients.values():
        for vals in side.values():
            assert all((v.exact_zero and v.truth_bits == 0 or v.status == CertStatus.NONDIFF for v in vals))

def test_zero_kl_nonidentical_tags():
    x = [F(1, 4), F(3, 8), F(5, 16), F(1, 16)]
    bits = {'probs': _bits(x, BINARY32)}
    cert = certify_sample('categorical_probs_K4', 'fp32', bits, {'probs': _bits([2 * v for v in x], BINARY32)})
    assert cert.sample_tag == 'ZERO_KL_NONIDENTICAL'
    z = {'logits': _bits(x, BINARY32)}
    cert = certify_sample('categorical_logits_K4', 'fp32', z, {'logits': _bits([v + 1 for v in x], BINARY32)})
    assert cert.sample_tag == 'ZERO_KL_NONIDENTICAL'
    w = [F(1, 2), F(-1, 4)]
    base = {'loc': _bits([F(1), F(2)], BINARY32), 'cov_diag': _bits([F(1), F(2)], BINARY32)}
    cert = certify_sample('lowrank_mvn_d2_r1', 'fp32', {**base, 'cov_factor': _bits(w, BINARY32)}, {**base, 'cov_factor': _bits([-v for v in w], BINARY32)})
    assert cert.sample_tag == 'ZERO_KL_NONIDENTICAL'

def test_mvn_upper_triangle_is_not_applicable():
    p, q = make_pair('mvn_d4')
    bits_p = {n: _bits(v, BINARY32) for n, v in p.items()}
    bits_q = {n: _bits(v, BINARY32) for n, v in q.items()}
    cert = certify_sample('mvn_d4', 'fp32', bits_p, bits_q, sample_type='ulp')
    for side in ('P', 'Q'):
        for idx, v in enumerate(cert.gradients[side]['scale_tril']):
            if idx % 4 > idx // 4:
                assert v.status == CertStatus.NONDIFF
            else:
                assert v.status == CertStatus.CERTIFIED

def test_formula_check_script_passes():
    res = subprocess.run([sys.executable, str(HERE / 'multivariate_formula_check.py')], capture_output=True, text=True, timeout=3000)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
