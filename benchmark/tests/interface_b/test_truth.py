import sys
from fractions import Fraction as Fr
import numpy as np, mpmath as mp
from kldivbench import truth as T, gen as G
from klbench.fp import get_format, round_fraction, exact_value
assert 'torch' not in sys.modules, 'the truth modules must not import torch'
mp.mp.dps = 60
NINF = float('-inf')
checks = 0

def b(v, dt='fp32'):
    return int(G.to_bits(v, dt))

def val(bits, dt='fp32'):
    return float(G.from_bits(np.array(bits, dtype=G.NP[dt][1]), dt))

def ok(cond, msg):
    global checks
    checks += 1
    assert cond, msg

def elem(x, t, lt, dt='fp32'):
    f = get_format(dt)
    L, gx, gt = T.element(b(x, dt), b(t, dt), f, lt)
    return [(v.status, None if not v.scored else v.rounded(f)[0]) for v in (L, gx, gt)]

def is_bits(bits, x, dt='fp32'):
    return bits == b(x, dt) or (val(bits, dt) == 0 and x == 0)
(Ls, Lb), (xs, xb), (ts, tb) = elem(-1.0, 0.0, False)
ok(Ls == T.EXZ and is_bits(Lb, 0.0) and (xs == T.EXGZ) and is_bits(xb, 0.0) and (ts == T.ONESIDED) and (val(tb) == NINF), 't=0, x finite')
(Ls, Lb), (xs, xb), (ts, tb) = elem(NINF, 0.5, False)
ok(Ls == T.MINF and val(Lb) == float('inf') and (xs == T.LIMIT) and (val(xb) == -0.5) and (ts == T.LIMIT) and (val(tb) == float('inf')), 't>0, x=-inf')
(Ls, Lb), (xs, xb), (ts, tb) = elem(NINF, 0.0, False)
ok(Ls == T.CONVZ and is_bits(Lb, 0.0) and (xs == T.LIMIT) and is_bits(xb, 0.0) and (ts == T.UNDEF) and (tb is None), 't=0, x=-inf')
ok(all((s == T.DOMAIN for s, _ in elem(-1.0, -0.5, False))), 't<0')
(Ls, Lb), (xs, xb), (ts, tb) = elem(0.0, 1.0, False)
ok(Ls == T.EXZ and val(xb) == -1.0 and (ts == T.CERT) and (val(tb) == 1.0), 't=1, x=0')
(Ls, Lb), (xs, xb), (ts, tb) = elem(1.0, 1.0, False)
ok(Ls == T.CERT and val(Lb) == -1.0 and (val(xb) == -1.0) and (ts == T.EXGZ) and is_bits(tb, 0.0), 't=1, x=1: exact zero gradient, L = -1')
(Ls, Lb), (xs, xb), (ts, tb) = elem(-1.5, -1.5, True)
ok(Ls == T.EXZ and xs == T.CERT and (val(xb) == float(np.float32(-mp.e ** mp.mpf(-1.5)))) and (val(tb) == float(np.float32(mp.e ** mp.mpf(-1.5)))), 't=x')
(Ls, Lb), (xs, xb), (ts, tb) = elem(1.0, 0.0, True)
ok(val(Lb) == -1.0 and val(xb) == -1.0 and (ts == T.EXGZ) and is_bits(tb, 0.0), 'x=t+1: L=-1, dL/dt exactly 0')
(Ls, Lb), (xs, xb), (ts, tb) = elem(-0.75, -1.75, True)
ok(ts == T.EXGZ and Ls == T.CERT, 'x=t+1 away from 0')
(Ls, Lb), (xs, xb), (ts, tb) = elem(-1.0, NINF, True)
ok(Ls == T.CONVZ and xs == T.LIMIT and is_bits(xb, 0.0) and (ts == T.LIMIT) and is_bits(tb, 0.0), 't=-inf, x finite')
(Ls, Lb), (xs, xb), (ts, tb) = elem(NINF, -1.0, True)
ok(Ls == T.MINF and xs == T.LIMIT and (val(xb) == float(np.float32(-mp.e ** (-1)))) and (val(tb) == float('inf')), 't finite, x=-inf')
(Ls, Lb), (xs, xb), (ts, tb) = elem(NINF, NINF, True)
ok(Ls == T.CONVZ and xs == T.LIMIT and (ts == T.UNDEF), 't=-inf, x=-inf')
for bad in ((float('inf'), 0.5), (float('nan'), 0.5), (-1.0, float('nan'))):
    try:
        elem(*bad, False)
        ok(False, 'unsupported input accepted')
    except T.Unsupported:
        ok(True, '')
for c in (mp.mpf('0.5'), mp.mpf(3)):
    t = mp.mpf('1e-9')
    ok(abs(t * (mp.log(t) - (mp.log(t) - c / t)) - c) < mp.mpf('1e-40'), 'path limit')
rng = np.random.default_rng(1)
for dt in ('fp16', 'fp32'):
    f = get_format(dt)
    fl = G.NP[dt][0]
    for lt in (False, True):
        for _ in range(300):
            x = fl(rng.uniform(-8, 2))
            t = fl(rng.uniform(-8, 2)) if lt else fl(np.exp(rng.uniform(-8, 3)))
            X = mp.mpf(float(x))
            Tt = mp.mpf(float(t))
            fun = (lambda tt, xx: mp.e ** tt * (tt - xx)) if lt else lambda tt, xx: tt * mp.log(tt * mp.e ** (-xx))
            (Ls, Lb), (xs, xb), (ts, tb) = elem(float(x), float(t), lt, dt)
            h = mp.mpf('1e-25')
            ref = [fun(Tt, X), (fun(Tt, X + h) - fun(Tt, X - h)) / (2 * h), (fun(Tt + h, X) - fun(Tt - h, X)) / (2 * h)]
            for got, r, nm in zip((Lb, xb, tb), ref, ('L', 'gx', 'gt')):
                want = round_fraction(Fr(mp.nstr(r, 45)), f)
                ok(got == want or (val(got, dt) == 0 and val(want, dt) == 0), f'{dt} lt={lt} {nm}: {got} vs {want} at x={x}, t={t}')
xb_ = [[b(-1.0), b(-2.0), b(-0.5)]]
tb_ = [[b(0.25), b(0.5), b(0.25)]]
o = T.certify_batch(xb_, tb_, 'fp32', False)
S = sum((mp.mpf(t) * (mp.log(t) - x) for x, t in ((-1, 0.25), (-2, 0.5), (-0.5, 0.25))))
ok(o['S_bits']['sum'] == round_fraction(Fr(mp.nstr(S, 50)), 'fp32') and o['S_bits']['mean'] == round_fraction(Fr(mp.nstr(S / 3, 50)), 'fp32') and (o['S_bits']['batchmean'] == o['S_bits']['sum']), 'reductions')
o = T.certify_batch([[b(NINF), b(-1.0)]], [[b(0.5), b(0.5)]], 'fp32', False)
ok(o['S_st'] == T.MINF and val(o['S_bits']['sum']) == float('inf'), 'row with +inf')
o = T.certify_batch([[b(-1.0), b(-1.0)]], [[b(-0.5), b(0.5)]], 'fp32', False)
ok(o['S_st'] == T.DOMAIN and o['L_st'][1] == T.CERT, 'row with domain error: other element still certified')
o = T.certify_batch([[b(-1.0), b(0.0)]], [[b(0.0), b(1.0)]], 'fp32', False)
ok(o['S_st'] == T.EXZ and o['gt_st'] == [T.ONESIDED, T.CERT], 'all-exact-zero row; per-component statuses')
for _ in range(50):
    zp = [Fr(float(np.float32(v))) for v in rng.uniform(-5, 5, 4)]
    zq = [Fr(float(np.float32(v))) for v in rng.uniform(-5, 5, 4)]
    a = T.interval(T.e2e_row(zp, zq))
    c = T.interval(T.e2e_row_alt(zp, zq))
    ok(a[0] <= c[1] and c[0] <= a[1], 'e2e forms overlap')
unit = G.Unit('S1', 'B_9', 'R0', True)
seen = []
for e in (-15, -14):
    zp = G.to_bits([0.0, 0.0], 'fp16')
    zq = G.to_bits([2.0 ** e, 0.0], 'fp16')
    x, t = G.make_xt(zp, zq, 'fp16', True)
    ok([format(v, '04X') for v in x] == ['B98C', 'B98C'] and [format(v, '04X') for v in t] == ['B98C', 'B98C'], 'counterexample bit patterns')
    lo, hi = T.interval(T.e2e_row(G.exact(zp, 'fp16'), G.exact(zq, 'fp16')))
    ref = mp.log(mp.cosh(mp.mpf(2) ** e / 2))
    ok(abs(mp.mpf(lo.numerator) / lo.denominator - ref) < ref * mp.mpf('1e-8') and G.classify(lo, hi) == 'B_9', 'T_e2e = log cosh(u/2), group [1e-10, 1e-9)')
    ok(unit.add(x, t, zp, zq, e2e_lo=str(lo), e2e_hi=str(hi)), 'sample must be kept')
    seen.append(lo)
    o = T.certify_batch([x], [t], 'fp16', True)
    ok(o['S_st'] == T.EXZ, 'T_op exactly 0')
ok(len(unit.rows) == 2 and seen[0] != seen[1] and (unit.rows[0]['xt_hash'] == unit.rows[1]['xt_hash']), 'both kept, different T_e2e, same (x, t) hash')
ok(not unit.add(unit.rows[0]['x'], unit.rows[0]['t'], unit.rows[0]['zp'], unit.rows[0]['zq']), 'an identical source is a duplicate')
u2 = G.Unit('S4', '', '', False)
ok(u2.add([1, 2], [3, 4]) and (not u2.add([1, 2], [3, 4])), 'layers without logits de-duplicate on (x, t)')
print('all', checks, 'checks passed')
