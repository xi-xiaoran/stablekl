import sys, json, tempfile, struct, hashlib, math
from fractions import Fraction as Fr
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from flint import arb
from klbench.fp import get_format, round_fraction, bits_to_fraction
from kldivbench import truth as T, score as S, torchrun as R, pipeline as P, aggregate as A, store, gen as G
N = [0]

def ok(c, msg):
    N[0] += 1
    if not c:
        raise AssertionError(msg)
NPF = {'fp16': (np.float16, np.uint16), 'fp32': (np.float32, np.uint32)}

def bits(v, dt='fp32'):
    f, u = NPF[dt]
    return np.asarray(v, dtype=f).view(u).tolist()

def val(b, dt='fp32'):
    f, u = NPF[dt]
    return float(np.asarray(b, dtype=u).view(f))

def run(x, t, dt, lt, kl=F.kl_div, mutate=None, zp=None, zq=None, method='A'):
    fmt = get_format(dt)
    xb = [bits(r, dt) for r in x]
    tb = [bits(r, dt) for r in t]
    tr = T.certify_batch(xb, tb, fmt, lt, zp=zp, zq=zq)
    tc = S.TruthCtx(xb, tb, fmt, lt, zp, zq)
    outs = R.run_batch(xb, tb, dt, lt, kl=kl)
    if mutate:
        mutate(outs[method], tr)
    acc = S.Acc()
    flags, info = S.score_method(acc, tc, tr, outs[method], fmt)
    return (acc, flags, tr, outs, tc)

def expected(B, K):
    return dict(batches=1, rows=B * K // K, elements=B * K)
rng = np.random.default_rng(7)

def rand_batch(B, K, lt):
    zq = rng.uniform(-3, 3, (B, K))
    zp = rng.uniform(-3, 3, (B, K))
    x = torch.log_softmax(torch.tensor(zq), -1).numpy().astype(np.float32)
    p = torch.log_softmax(torch.tensor(zp), -1).numpy()
    return (x, (p if lt else np.exp(p)).astype(np.float32))

def raising(only=None):

    def kl(x, t, reduction, log_target):
        if only is None or reduction in only:
            raise RuntimeError('injected')
        return F.kl_div(x, t, reduction=reduction, log_target=log_target)
    return kl
for B, K in ((1, 4), (8, 4)):
    x, t = rand_batch(B, K, False)
    n = B * K
    good, *_ = run(x, t, 'fp32', False)
    c0 = good.cnt
    acc, flags, *_ = run(x, t, 'fp32', False, kl=raising())
    c = acc.cnt
    ok(c['E_CR|n'] == n == c['E_CR|bad'], 'all-call failure: every element is in the denominator and judged wrong')
    ok(all((c[f'S_CR:{r}|n'] == 1 == c[f'S_CR:{r}|bad'] for r in S.REDS)), 'reductions judged wrong')
    ok(all((c[f'G_CR:{s}:{r}|n'] == n == c[f'G_CR:{s}:{r}|bad'] for s in ('gx', 'gt') for r in S.REDS)), 'gradients judged wrong')
    ok(all((c[f'{k}|n'] == c0[f'{k}|n'] for k in ('E_CR', 'S_CR:sum', 'S_CR:mean', 'G_CR:gx:sum', 'G_CR:gt:batchmean'))), 'denominators identical with and without the exception')
    ok(c['CALL_ERROR:sum'] == 1 and c['CALL_ERROR:none'] == 1 and (flags['LAYER_A'] is True), 'exception counted, sample fails level A')
    ok(A.reconcile('u', 'k', c, dict(batches=1, rows=B, elements=n), None) == [], 'buckets reconcile under exceptions')
    acc, flags, *_ = run(x, t, 'fp32', False, kl=raising(('mean',)))
    c = acc.cnt
    ok(c['S_CR:mean|bad'] == 1 and c['S_CR:sum|bad'] == c0['S_CR:sum|bad'] and (c['G_CR:gx:mean|bad'] == n) and (c['G_CR:gx:sum|bad'] == c0['G_CR:gx:sum|bad']), 'a failure of one reduction only affects that reduction')
    ok(A.reconcile('u', 'k', c, dict(batches=1, rows=B, elements=n), None) == [], 'buckets reconcile under a partial exception')

def nograd(x, t, reduction, log_target):
    return F.kl_div(x.detach(), t.detach(), reduction=reduction, log_target=log_target)
x, t = rand_batch(1, 4, False)
acc, flags, *_ = run(x, t, 'fp32', False, kl=nograd)
c = acc.cnt
ok(c['BACKWARD_ERROR:sum'] + c['GRAD_MISSING:sum'] == 1 and c['G_CR:gx:sum|bad'] == 4 and (c['S_CR:sum|n'] == 1), 'missing gradient: forward still scored, gradient components judged wrong')

def nankl(x, t, reduction, log_target):
    return F.kl_div(x, t, reduction=reduction, log_target=log_target) * float('nan')
acc, flags, *_ = run(x, t, 'fp32', False, kl=nankl)
c = acc.cnt
ok(c['E_NAN|bad'] == 4 == c['E_NAN|n'] and c['E_CR|bad'] == 4 and (c['S_NAN:sum|bad'] == 1) and flags['LAYER_C'], 'NaN outputs are in the denominator and judged wrong')
ok(c['E_ULP8|n'] == 0 and c['E_ULP8|na'] == 4, 'ULP metrics need a finite output')
x, t = rand_batch(8, 4, False)

def flip_row3(o, tr):
    for red in S.REDS:
        g = o[red + '_gx']
        for i in range(12, 16):
            g[i] ^= 2147483648
acc, flags, tr, outs, tc = run(x, t, 'fp32', False, mutate=flip_row3)
c = acc.cnt
ok(c['GV_n:gx:sum'] == 8 and c['GV_REVERSED:gx:sum|n'] == 8 and (c['GV_REVERSED:gx:sum|bad'] == 1), 'one reversed row out of 8 is one bad vector out of 8')
ok(c['GV_REL_L2_1:gx:sum|bad'] == 1 and c['GV_COS_9:gx:sum|bad'] == 1 and (c['GV_REVERSED:gt:sum|bad'] == 0), 'only that row, only that side')
gy = [bits_to_fraction(b, 'fp32').value for b in outs['A']['sum_gx']]
gT = [bits_to_fraction(b, 'fp32').value for b in tr['gx_bits']['sum']]
ok(sum((a * b for a, b in zip(gy, gT))) > 0, 'the whole B x K vector is NOT reversed: a batch-level vector would hide the row')
ok(c['G_SIGN:gx:sum|bad'] == 4 and c['G_SIGN:gx:sum|rbad'] == 1 and (c['G_SIGN:gx:sum|rn'] == 8) and (c['G_SIGN:gx:sum|sbad'] == 1), 'component / row / sample counting')

def rot(a):

    def f(o, tr):
        for red in S.REDS:
            o[red + '_gx'] = bits([-0.5 - a, -0.5 + a]) if red != 'mean' else bits([(-0.5 - a) / 2, (-0.5 + a) / 2])
    return f
for a_, want in ((0.125, (1, 1, 0)), (0.03125, (1, 0, 0)), (0.5, (1, 1, 1)), (0.0, (0, 0, 0))):
    acc, *_ = run([[-1.0, -2.0]], [[0.5, 0.5]], 'fp32', False, mutate=rot(a_))
    c = acc.cnt
    got = tuple((c[f'GV_{l}:gx:sum|bad'] for _, l in S.COSS))
    ok(got == want and c['GV_COS_999:gx:sum|n'] == 1, f'cosine flags for a = {a_}: {got}')
    ok(c['GV_REL_L2_1:gx:sum|bad'] == (a_ > 0) and c['GV_REVERSED:gx:sum|bad'] == 0, 'relative L2 = 2a')

def zero_g(o, tr):
    for red in S.REDS:
        o[red + '_gx'] = [0, 0]
acc, *_ = run([[-1.0, -2.0]], [[0.5, 0.5]], 'fp32', False, mutate=zero_g)
c = acc.cnt
ok(c['GV_COS_999:gx:sum|na'] == 1 and c['GV_REL_L2_1:gx:sum|bad'] == 1 and (c['GV_VANISH_VEC:gx:sum|bad'] == 1) and (c['G_VANISH:gx:sum|bad'] == 2), 'zero output vector: cosine not applicable, relative error 100%')
acc, flags, tr, *_ = run([[-1.0, -2.0]], [[0.0, 1.0]], 'fp32', False)
c = acc.cnt
ok(c['GV_NOT_APPLICABLE:gt:sum'] == 1 and c['GV_n:gt:sum'] == 0 and (c['GV_n:gx:sum'] == 1) and (c['G_CR:gt:sum|n'] == 1), 'ONE_SIDED_INF component excludes the t-side vector only')
ok(c['BG:gt:sum:ONE_SIDED_INF:-inf|n'] == 1 and c['BG:gt:sum:ONE_SIDED_INF:-inf|BG_NAN'] == 1, 'boundary table: official NaN at t = 0')
ok(A.reconcile('u', 'k', c, dict(batches=1, rows=1, elements=2), None) == [], 'buckets reconcile with boundary components')
f16 = get_format('fp16')
found = None
for n_ in range(1, 3000):
    if n_ % 3 == 0 or (n_ > 2048 and n_ % 2):
        continue
    Tex = Fr(n_, 3)
    tbit = round_fraction(Tex, f16)
    Trn = bits_to_fraction(tbit, f16).value
    y0 = round_fraction(Tex * Fr(101, 100), f16)
    for yb in range(y0 - 2, y0 + 3):
        yv = bits_to_fraction(yb, f16).value
        if (abs(yv - Tex) >= Tex / 100) != (abs(yv - Trn) >= Trn / 100):
            found = (n_, yb, abs(yv - Tex) >= Tex / 100)
            break
    if found:
        break
ok(found is not None, 'a case where rounded truth and exact truth disagree on the 1% threshold exists')
n_, yb, exact_flag = found

def put_mean(o, tr):
    o['mean'] = yb
acc, flags, tr, *_ = run([[-float(n_), 0.0, 0.0]], [[0.0, 0.0, 0.0]], 'fp16', True, mutate=put_mean)
ok(tr['S_bits']['mean'] == round_fraction(Fr(n_, 3), f16), 'truth of the mean is RN(n / 3)')
ok(flags['S_REL1:mean'] == exact_flag, f'REL1 follows the exact truth, not the rounded one (n = {n_}, y = {yb:#06x}, exact says {exact_flag})')
x, t = ([[-1.25, -0.75]], [[0.3, 0.7]])

def scale_sum(f_):

    def g(o, tr):
        o['sum'] = bits(val(tr['S_bits']['sum']) * f_)
    return g
acc, flags, *_ = run(x, t, 'fp32', False, mutate=scale_sum(1.02))
ok(flags['S_REL1:sum'] is True and flags['S_REL10:sum'] is False and (acc.cnt['S_REL1_UNDECIDED:sum'] == 0), '2% error')
acc, flags, *_ = run(x, t, 'fp32', False, mutate=scale_sum(1.0))
ok(flags['S_REL1:sum'] is False and flags['S_CR:sum'] is False, 'correctly rounded output')
acc, flags, *_ = run(x, t, 'fp32', False, mutate=scale_sum(-3.0))
ok(flags['S_REL100:sum'] is True and flags['S_SIGN_MISMATCH:sum'] is True, 'sign flipped, 400% error')

def zero_sum(o, tr):
    o['sum'] = 0
acc, flags, *_ = run(x, t, 'fp32', False, mutate=zero_sum)
ok(flags['S_REL100:sum'] is True and flags['S_FALSE_ZERO:sum'] is True and (acc.cnt['S_REL100_UNDECIDED:sum'] == 0), 'output 0 for a non-zero truth: exactly 100%, decided')
f16_ = get_format('fp16')
zq_ = [[Fr(1, 1 << 15), Fr(0)]]
zp_ = [[Fr(0), Fr(0)]]
xb_ = [T.rn_log_softmax(zq_[0], 'fp16')]
tb_ = [T.rn_log_softmax(zp_[0], 'fp16')]
tr_ = T.certify_batch(xb_, tb_, f16_, True, zp=zp_, zq=zq_)
tc_ = S.TruthCtx(xb_, tb_, f16_, True, zp_, zq_)
aT = S.Acc()
fl_ = S.score_truth(aT, tc_, tr_, f16_)
ok(tr_['S_st'] == T.EXZ and fl_['R_REL100'] is True and (fl_['R_NEG'] is False) and (fl_['R_NONZERO'] is False) and (aT.cnt['R_REL100_UNDECIDED'] == 0), 'T_op = 0 exactly against T_e2e > 0')
f32 = get_format('fp32')
for u in (0.5, 2.0 ** (-8), 3.0):
    zp = [[Fr(0), Fr(0)]]
    zq = [[Fr(u), Fr(0)]]
    xb = [T.rn_log_softmax(zq[0], 'fp32')]
    tb = [T.rn_softmax(zp[0], 'fp32')]
    tr = T.certify_batch(xb, tb, f32, False, zp=zp, zq=zq)
    import math
    ok(all((tr['E2E_q'][r] & T.Q_CERT for r in S.REDS)), 'T_e2e certified at all three scales')
    E = tr['E2E_val']
    ok(E.lo is not None and round_fraction(E.lo, f32) == round_fraction(E.hi, f32) == tr['E2E_bits']['sum'], 'both endpoints of the enclosure round to the stored bits')
    ok(abs(val(tr['E2E_bits']['sum']) - math.log(math.cosh(u / 2))) <= 2e-07 * math.log(math.cosh(u / 2)), 'closed form log cosh(u / 2)')
    ok(tr['E2E_bits']['mean'] == round_fraction(Fr(1, 2) * E.lo, f32) == round_fraction(Fr(1, 2) * E.hi, f32), 'mean scale certified on its own')
    tc = S.TruthCtx(xb, tb, f32, False, zp, zq)
    accT = S.Acc()
    fl = S.score_truth(accT, tc, tr, f32)
    ok('R_REL1' in fl and 'R_NEG' in fl and (accT.cnt['R_REL1_UNDECIDED'] == 0) and all((len(accT.stats[f'R_ABS:{r}']) == 1 for r in S.REDS)), 'representation metrics decided with certified balls')
    ok(abs(accT.stats['R_ABS:mean'][0] * 2 - accT.stats['R_ABS:sum'][0]) <= 1e-12 * accT.stats['R_ABS:sum'][0], "R_ABS is reported at every reduction's own scale")
    am = S.Acc()
    o_ = R.run_batch(xb, tb, 'fp32', False)['A']
    S.score_method(am, tc, tr, o_, f32)
    ok(all((len(am.stats[f'TOT_ABS:{r}']) == 1 for r in S.REDS)), 'TOT_ABS recorded for sum, mean and batchmean separately')
    e_ = float(tr['E2E_val'].lo)
    ok(abs(am.stats['TOT_ABS:mean'][0] - abs(val(o_['mean']) - e_ / 2)) <= 1e-09 * e_, 'TOT_ABS:mean = |mean output - T_e2e / (B K)|')
zp = [[Fr(1, 3).limit_denominator(1 << 20) * 0 + Fr(3, 8), Fr(-5, 4), Fr(1, 2)]]
xb = [T.rn_log_softmax(zp[0], 'fp32')]
tb = [T.rn_softmax(zp[0], 'fp32')]
tr = T.certify_batch(xb, tb, f32, False, zp=zp, zq=zp)
tc = S.TruthCtx(xb, tb, f32, False, zp, zp)
accT = S.Acc()
fl = S.score_truth(accT, tc, tr, f32)
ok(tr['E2E_st'] == T.EXZ and 'R_REL1' not in fl and (accT.cnt['R_REL1|na'] == 1) and ('R_NONZERO' in fl), 'T_e2e = 0: only R_NEG / R_NONZERO / R_ABS')
v = T.Val(T.CERT, Fr(1, 1 << 30))
(b1, q1), (b2, q2) = v.certify(f16, (Fr(1), Fr(1 << 20)))
ok(b1 == 0 and q1 == T.Q_CERT | T.Q_RZ and (q2 == T.Q_CERT) and (b2 != 0), 'ROUNDS_TO_ZERO is a flag of one output at one scale')
v = T.Val(T.CERT, Fr(10 ** 6))
(b1, q1), (b2, q2) = v.certify(f16, (Fr(1), Fr(1, 1000)))
ok(b1 == f16.infinity_bits and q1 == T.Q_CERT | T.Q_OVF and (q2 == T.Q_CERT), 'ROUNDING_OVERFLOW is a flag of one output at one scale')
ok(T.Val(T.MINF, T.PINF).certify(f16)[0] == (f16.infinity_bits, T.Q_CERT), 'a mathematical +inf is not a rounding overflow')
ok(T.Val(T.EXZ, Fr(0)).certify(f16)[0] == (0, T.Q_CERT), 'an exact zero is not ROUNDS_TO_ZERO')
hard = T.Val(T.CERT, fn=lambda: arb(1) + arb(2) ** (-24) + (arb(2).sqrt() ** 2 - 2))
r = hard.certify(f32, (Fr(1), Fr(1, 3)))
ok(r[0] == (None, 0) and r[1][1] & T.Q_CERT and (r[1][0] == round_fraction((1 + Fr(1, 1 << 24)) / 3, f32)), 'a hard case at one scale does not block the other scale')
tiny = 2.0 ** (-24)
acc, flags, tr, outs, tc = run([[-1.0, -2.0, -3.0, -4.0]], [[tiny, 0.5, 0.25, 0.125]], 'fp16', False)
c = acc.cnt
ok(tr['gx_q']['sum'][0] == T.Q_CERT and tr['gx_q']['mean'][0] == T.Q_CERT | T.Q_RZ and (tr['gx_st'][0] == T.CERT), "x-gradient -t: exact at 'sum', ROUNDS_TO_ZERO at 'mean'")
ok(outs['A']['mean_gx'][0] & 32767 == 0 and c['G_VANISH:gx:mean|na'] == 1 and (c['G_VANISH:gx:mean|bad'] == 0), 'official 0 at a ROUNDS_TO_ZERO component is not G_VANISH')
acc, flags, tr, outs, tc = run([[10.0, -3.0, -3.0, -3.0]], [[11.5, -20.0, -20.0, -20.0]], 'fp16', True)
c = acc.cnt
ok(tr['gx_q']['sum'][0] & T.Q_OVF and (not tr['gx_q']['mean'][0] & T.Q_OVF) and tr['S_q']['sum'] & T.Q_OVF and (not tr['S_q']['mean'] & T.Q_OVF), 'overflow flags per scale')
ok(c['S_CR:sum|bad'] == 0 and c['S_FALSE_INF:sum|na'] == 1, "inf at 'sum' is the correctly rounded result")
ok(c['S_FALSE_INF:mean|bad'] == 1 and c['S_FALSE_INF_INTERMEDIATE_OVERFLOW:mean'] == 1, "inf at 'mean' is FALSE_INF, tagged INTERMEDIATE_OVERFLOW")
ok(c['G_FALSE_INF:gx:mean|bad'] == 1 and c['G_FALSE_INF_INTERMEDIATE_OVERFLOW:gx:mean'] == 1 and (c['GV_NOT_APPLICABLE:gx:sum'] == 1) and (c['GV_n:gx:mean'] == 1), 'gradient: same; vector excluded only where truth overflows')
okB = S.Acc()
S.score_method(okB, tc, tr, outs['B64'], f16)
ok(okB.cnt['S_FALSE_INF:mean|bad'] == 0 and okB.cnt['S_CR:mean|bad'] == 0, 'B64 gives the finite, correctly rounded mean')
pc = __import__('collections').Counter()
S.pair(pc, {'m': True, 'k': False, 'z': True}, {'m': False, 'k': False})
ok(pc == {'m|fixed': 1, 'k|both_good': 1}, 'pairing')
lo, hi = A.wilson(0, 10)
ok(lo == 0 and abs(hi - 0.27753) < 0.0001, 'Wilson interval')
lo, hi = A.wilson(5, 10)
ok(abs(lo - 0.2366) < 0.001 and abs(hi - 0.7634) < 0.001, 'Wilson interval')
ok(R.run_batch.__defaults__[0] is torch.nn.functional.kl_div, 'the adapter calls torch.nn.functional.kl_div')
x, t = rand_batch(1, 16, False)
xt, tt = (torch.tensor(x), torch.tensor(t))
ok(torch.equal(F.kl_div(xt, tt, reduction='none'), torch.xlogy(tt, tt) - tt * xt), 'element-wise output is bitwise xlogy(t, t) - t * x')
with tempfile.TemporaryDirectory() as td:
    q = dict(S1={'R0': 8, 'R1': 8, 'R2': 0}, S1t={'R0': 8, 'R1': 0, 'R2': 0}, S2=8, S2b=8, S3=8, S4=8, S5=8, S6=8, S7=8)
    for lt, dt, K in ((False, 'fp16', 4), (True, 'fp32', 2)):
        root = Path(td) / store.cfg_name(lt, dt, K)
        kw = dict(root=root, lt=lt, dt=dt, K=K, q=q)
        P.stage_gen(**kw)
        pend = sorted((root / 'pending_s1t').glob('*.npz'))
        ok(len(pend) == 45 and (not list((root / 'inputs').glob('S1t*'))), 'S1t waits for torch')
        P.stage_fill(**kw)
        nrow = 0
        for p in sorted((root / 'inputs').glob('*.npz')):
            a, meta = store.load(p)
            for i in range(meta['n_rows']):
                m = json.loads(str(a['meta'][i]))
                nrow += 1
                ok(m['xt_hash'] == G.xt_hash(a['x'][i].tolist(), a['t'][i].tolist()), f'xt_hash matches the stored (x, t): {p.name} row {i}')
            if meta['layer'] == 'S1t' and meta['n_rows']:
                ok(a['x'].any() and 'filled_by' in meta, 'S1t inputs are the real torch outputs')
                x0, t0 = R.torch_inputs(a['zp'][0].tolist(), a['zq'][0].tolist(), dt, lt)
                ok(x0 == a['x'][0].tolist() and t0 == a['t'][0].tolist(), 'S1t inputs reproduce')
        P.stage_certify(**kw)
        P.stage_run(**kw)
        P.stage_score(**kw)
        ok(P.stage_run(**kw)['batches'] == 0, 'completed units are not recomputed')
    rec = A.main(['--root', td])
    ok(rec == [], f'denominators reconcile with the inputs: {rec[:3]}')
    import csv
    rows = list(csv.DictReader(open(Path(td) / 'report' / 'metrics.csv', encoding='utf-8')))
    s1 = {(r['group'], r['region']) for r in rows if r['layer'] == 'S1' and r['log_target'] == 'False'}
    ok(len(s1) > 10 and all((g.startswith('B_') and rg in ('R0', 'R1', 'R2') for g, rg in s1)), 'S1 metrics are reported per KL group and region')
    for lt, dt, K in ((False, 'fp16', 4), (True, 'fp32', 2)):
        root = Path(td) / store.cfg_name(lt, dt, K)
        for B in (1, 8):
            want = sum((store.verify(p)['n_rows'] // B * B * K for p in (root / 'inputs').glob('S1__*.npz')))
            got = sum((int(r['n']) for r in rows if r['layer'] == 'S1' and r['log_target'] == str(lt) and (r['dtype'] == dt) and (int(r['B']) == B) and (r['method'] == 'A') and (r['metric'] == 'E_CR') and (r['counting_unit'] == 'component')))
            ok(got == want and want > 0, f'S1 element denominators merge back to the input count ({got} vs {want})')
    cnts = list(csv.DictReader(open(Path(td) / 'report' / 'counts.csv', encoding='utf-8')))
    ok(not any((r['counter'].startswith('CALL_ERROR') for r in cnts)), 'no call errors in the real run')
    prs = list(csv.DictReader(open(Path(td) / 'report' / 'pairs.csv', encoding='utf-8')))
    ok(any((r['metric'] == 'S_CR:sum' and int(r['fixed']) > 0 for r in prs)) and {r['method'] for r in prs} == {'B32', 'B64'}, 'A / B pairing table')
    bg = list(csv.DictReader(open(Path(td) / 'report' / 'boundary.csv', encoding='utf-8')))
    ok(any((r['status'] == 'LIMIT_GRAD' for r in bg)) and any((r['status'] == 'ONE_SIDED_INF' for r in bg)), 'boundary table')
    p = sorted((Path(td) / 'ltF_fp16_K4' / 'out').glob('*.npz'))[0]
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 1
    p.write_bytes(bytes(raw))
    try:
        store.load(p)
        ok(False, 'tampered file accepted')
    except ValueError:
        ok(True, '')
import subprocess, os
env = dict(os.environ)
r = subprocess.run([sys.executable, '-c', "import kldivbench.truth, kldivbench.gen, kldivbench.score, kldivbench.xcheck, kldivbench.aggregate, kldivbench.pipeline, sys; assert 'torch' not in sys.modules"], env=env, capture_output=True)
ok(r.returncode == 0, 'truth / gen / score / xcheck / aggregate / pipeline do not import torch: ' + r.stderr.decode()[-300:])
r = subprocess.run([sys.executable, '-c', "import kldivbench.torchrun, kldivbench.pipeline, kldivbench.store, sys; assert 'flint' not in sys.modules and 'mpmath' not in sys.modules"], env=env, capture_output=True)
ok(r.returncode == 0, 'torchrun / pipeline / store do not import flint or mpmath: ' + r.stderr.decode()[-300:])
from kldivbench import xcheck as X
import collections
xb = [bits([-1.25, -0.75, -2.0])]
tb = [bits([0.3, 0.5, 0.2])]
vv = [T.row_vals(xb[0], tb[0], f32, False)]
tr = T.certify_batch(xb, tb, f32, False, vals=vv)
c = collections.Counter()
X.check_batch(c, xb, tb, f32, False, vv, tr['S_val'])
ok(c['XCHECK_elem_L_n'] == 3 and c['XCHECK_S_n'] == 1 and (not any((k.endswith('_bad') and v for k, v in c.items() if k != 'examples'))), 'mpmath second form agrees')
ks = X.pick(list(range(100, 356)), [5] * 255 + [1], 'seed', 8)
ok({0, 255} <= ks and 9 <= len(ks) <= 12 and (ks == X.pick(list(range(100, 356)), [5] * 255 + [1], 'seed', 8)) and (ks != X.pick(list(range(100, 356)), [5] * 255 + [1], 'other', 8)), 'cross-check sample: extremes + seeded random')
Sv = tr['S_val']
Sv.lo += Fr(1, 10 ** 20)
Sv.hi += Fr(1, 10 ** 20)
c = collections.Counter()
X.check_batch(c, xb, tb, f32, False, vv, Sv)
ok(c['XCHECK_S_bad'] == 1 and c['examples'], 'a truth shifted by 1e-20 is caught by the cross-check')
print(f'all {N[0]} checks passed')
from kldivbench import gensupp as GS

def cell(zp, zq, dt):
    f, u = NPF[dt]
    zpb = np.asarray(zp, dtype=f).view(u)
    zqb = np.asarray(zq, dtype=f).view(u)
    lo, hi = T.interval(T.e2e_row(G.exact(zpb, dt), G.exact(zqb, dt)))
    return (G.classify(lo, hi), G.worst(G.region_of(zpb, dt), G.region_of(zqb, dt)), zpb, zqb, float(lo))
for dt in ('fp16', 'fp32'):
    g, reg, zpb, zqb, lo = cell([0.0] * 1024, [5.0] + [0.0] * 1023, dt)
    ok(g == 'B_c' and reg == 'R0' and (abs(lo - (math.log((1023 + math.exp(5)) / 1024) - 5 / 1024)) < 1e-09), f'K = 1024 counterexample is in (B_c, R0): {g} {reg} {lo}')
for K_, want in ((2, 7.11e-15), (4, 5.33e-15)):
    g, reg, zpb, zqb, lo = cell([0.0] * K_, [2.0 ** (-22)] + [0.0] * (K_ - 1), 'fp16')
    ok(g == 'B_14' and reg == 'R0' and (abs(lo - want) < 0.01 * want), f'FP16 K = {K_} counterexample is in (B_14, R0): {g} {lo}')
    for lt_ in (False, True):
        x_, t_ = G.make_xt(zpb, zqb, 'fp16', lt_)
        u_ = G.Unit('S1s', 'B_14', 'R0', True)
        ok(u_.add(x_, t_, zpb, zqb) and (not u_.add(x_, t_, zpb, zqb)), 'accepted once; same provenance is a duplicate')
        if lt_:
            ok(list(x_) == list(t_), 'log_target=True: x and t are bitwise identical although the logits differ')
u_ = GS.gen_cell(False, 'fp16', 2, 'S1', 'B_14', 'R0', 20, set())
ok(len(u_.rows) == 20 and 'STATUS' not in u_.counters, f'supplement fills (FP16, K = 2, B_14, R0): {u_.counters}')
for r_ in u_.rows:
    g, reg, *_ = cell(np.asarray(r_['zp'], dtype=np.uint16).view(np.float16), np.asarray(r_['zq'], dtype=np.uint16).view(np.float16), 'fp16')
    ok((g, reg) == ('B_14', 'R0') and r_['direction'] in GS.FAMILIES, 'every supplementary row is certified into its own cell')
u_ = GS.gen_cell(True, 'fp32', 256, 'S1', 'B_c', 'R0', 6, set())
ok(len(u_.rows) == 6 and u_.counters.get('ACCEPTED_supp_interp', 0) >= 1, f'supplement fills (FP32, K = 256, B_c, R0): {u_.counters}')
u2 = GS.gen_cell(True, 'fp32', 256, 'S1', 'B_c', 'R0', 6, set())
ok([r_['zq'] for r_ in u2.rows] == [r_['zq'] for r_ in u_.rows], 'supplement is deterministic')
ex = {(tuple(r_['zp']), tuple(r_['zq'])) for r_ in u_.rows[:3]}
u3 = GS.gen_cell(True, 'fp32', 256, 'S1', 'B_c', 'R0', 6, ex)
ok(u3.counters.get('DUPLICATE_OF_FROZEN') == 3 and (not {(tuple(r_['zp']), tuple(r_['zq'])) for r_ in u3.rows} & ex), 'rows already in the frozen unit are not repeated')
u_ = GS.gen_cell(False, 'fp16', 2, 'S1t', 'B_14', 'R0', 4, set())
ok(u_.layer == 'S1ts' and all((r_['needs_torch_inputs'] and r_['xt_hash'] is None for r_ in u_.rows)), 'S1t supplement waits for torch inputs')
print(f'all {N[0]} checks passed (including supplement tests)')
