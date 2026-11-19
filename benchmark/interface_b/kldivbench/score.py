from __future__ import annotations
from collections import Counter, defaultdict
from fractions import Fraction as Fr
from flint import arb
from klbench.fp import get_format, bits_to_fraction, ulp_distance
from klbench.ref.arbutil import fraction_to_arb, arb_precision
from . import truth as T
REGULAR = (T.CERT, T.EXGZ)
EXACT_ZERO = (T.EXZ, T.CONVZ, T.EXGZ)
RELS = ((Fr(1, 100), 'REL1'), (Fr(1, 10), 'REL10'), (Fr(1), 'REL100'))
COSS = ((Fr(999, 1000), 'COS_999'), (Fr(99, 100), 'COS_99'), (Fr(9, 10), 'COS_9'))
REDS = ('sum', 'mean', 'batchmean')
SCORE_LADDER = (256, 1024, 4096)
LEVEL_A = ('CR',)
LEVEL_B = ('ULP8', 'REL1', 'REL10', 'REL100', 'REL_L2_1', 'COS_999', 'COS_99', 'COS_9')
LEVEL_C = ('NAN', 'FALSE_INF', 'MISSED_INF', 'SIGN_MISMATCH', 'FALSE_ZERO', 'NONZERO_AT_EXACT_ZERO', 'SIGN', 'VANISH', 'SPURIOUS', 'REVERSED', 'VANISH_VEC')
PAIR_CATS = ('fixed', 'new_bad', 'persistent', 'both_good')

def _kind(bits, fmt):
    d = bits_to_fraction(int(bits), fmt)
    return ('nan', None) if d.kind == 'nan' else ('inf', d.sign) if d.kind == 'infinity' else ('fin', d.value)

def decide(fn):
    r = None
    for p in SCORE_LADDER:
        with arb_precision(p):
            r = fn()
        if isinstance(r, list):
            if all((v is not None for v in r)):
                return r
        elif r is not None:
            return r
    return r

def _cmp(a, b):
    if a > b or (a.is_exact() and b.is_exact() and (a == b)):
        return True
    if a < b:
        return False
    return None

def _sign(ball):
    return 1 if ball > 0 else -1 if ball < 0 else 0 if ball.is_exact() and ball == 0 else None

class Acc:

    def __init__(self):
        self.cnt = Counter()
        self.stats = defaultdict(list)
        self.top = []
        self.ccases = []

class Emit:

    def __init__(self, acc):
        self.acc = acc
        self.flags = {}
        self.rows = {}

    def __call__(self, name, bad, row=None):
        c = self.acc.cnt
        if bad is None:
            c[name + '|na'] += 1
            return
        bad = bool(bad)
        c[name + '|n'] += 1
        c[name + '|bad'] += bad
        self.flags[name] = self.flags.get(name, False) or bad
        if row is not None:
            self.rows[name, row] = self.rows.get((name, row), False) or bad

    def close(self):
        c = self.acc.cnt
        for (n, _), b in self.rows.items():
            c[n + '|rn'] += 1
            c[n + '|rbad'] += b
        lay = {'LAYER_A': None, 'LAYER_B': None, 'LAYER_C': None}
        for n, b in self.flags.items():
            if n[:2] not in ('E_', 'S_', 'G_', 'GV'):
                continue
            m = n.split(':')[0].split('_', 1)[1]
            k = 'LAYER_A' if m in LEVEL_A else 'LAYER_B' if m in LEVEL_B else 'LAYER_C' if m in LEVEL_C else None
            if k:
                lay[k] = bool(lay[k]) or b
        for k, v in lay.items():
            if v is not None:
                self.flags[k] = v
        for n, b in self.flags.items():
            c[n + '|sn'] += 1
            c[n + '|sbad'] += b
        return self.flags

class TruthCtx:

    def __init__(self, xbits, tbits, fmt, log_target, zp=None, zq=None):
        self.fmt = fmt
        self.B = len(xbits)
        self.K = len(xbits[0])
        self.lt = log_target
        self.x = xbits
        self.t = tbits
        vals = [T.row_vals(xr, tr, fmt, log_target) for xr, tr in zip(xbits, tbits)]
        self.L = [v[0] for r in vals for v in r]
        self.gx = [v[1] for r in vals for v in r]
        self.gt = [v[2] for r in vals for v in r]
        self.S = T.reduce_sum(self.L)
        self.E = T.e2e_batch(zp, zq) if zp is not None else None
        self.scales = T.batch_scales(self.B, self.K)
        self._b = {}

    def ball(self, val):
        from flint import ctx
        k = (id(val), ctx.prec)
        if k not in self._b:
            self._b[k] = val.ball()
        return self._b[k]

    def interm_overflow(self, i):
        try:
            x = T.decode(self.x[i // self.K][i % self.K], self.fmt)
            t = T.decode(self.t[i // self.K][i % self.K], self.fmt)
        except T.Unsupported:
            return False
        m = self.fmt.sign_mask - 1
        inf = self.fmt.infinity_bits

        def ovf(v):
            b = v.certify(self.fmt)[0][0]
            return b is not None and b & m == inf
        if t in (T.NINF,) or x == T.NINF:
            return False
        if self.lt:
            return ovf(T.Val(T.CERT, fn=lambda: fraction_to_arb(t).exp()))
        if t <= 0:
            return False
        return ovf(T.Val(T.CERT, t * x)) or (t != 1 and ovf(T.Val(T.CERT, fn=lambda: fraction_to_arb(t) * fraction_to_arb(t).log())))

def rel_flags(tc, y, val, s):
    if val.exact is not None:
        tv = val.exact * s
        r = abs(y - tv)
        return [r >= th * abs(tv) for th, _ in RELS]
    if y == 0:
        return [True] * len(RELS)

    def f():
        tb = tc.ball(val) * fraction_to_arb(s)
        a = abs(fraction_to_arb(y) - tb)
        b = abs(tb)
        return [_cmp(a, b * fraction_to_arb(th)) for th, _ in RELS]
    return decide(f)

def cmp_scalar(E, tc, name, suffix, y, tb, q, st, val, s, fmt, row=None, grad=False, tag=None):
    c = E.acc.cnt
    n = lambda m: f'{name}_{m}{suffix}'
    if st in (T.UNDEF, T.DOMAIN):
        c[n('NOT_SCORED')] += 1
        return None
    if not q & T.Q_CERT:
        c[n('UNCERT')] += 1
        return None
    if y is None:
        E(n('CR'), True, row)
        c[n('NO_OUTPUT')] += 1
        return None
    m = fmt.sign_mask - 1
    ky, vy = _kind(y, fmt)
    kt, vt = _kind(tb, fmt)
    same = ky != 'nan' and (y == tb or (y & m == 0 and tb & m == 0))
    E(n('CR'), not same, row)
    nan = ky == 'nan'
    minf = st == T.MINF
    ovf = bool(q & T.Q_OVF)
    rz = bool(q & T.Q_RZ)
    ez = st in EXACT_ZERO
    E(n('NAN'), nan, row)
    if not grad:
        E(n('MISSED_INF'), not (ky == 'inf' and vy > 0) if minf else None, row)
    finf = ky == 'inf' if not (minf or ovf) else None
    E(n('FALSE_INF'), finf, row)
    if finf and tag is not None:
        c[n('FALSE_INF_' + tag())] += 1
    if ovf and (not same):
        c[n('OVERFLOW_MISMATCH')] += 1
    ysgn = None if nan else (-1 if y & fmt.sign_mask else 1) if y & m else 0
    tsgn = 0 if ez else -1 if tb & fmt.sign_mask else 1
    E(n('SIGN' if grad else 'SIGN_MISMATCH'), None if nan or ez else ysgn != 0 and ysgn != tsgn, row)
    E(n('VANISH' if grad else 'FALSE_ZERO'), None if ez or rz or nan else ysgn == 0 and tb & m != 0, row)
    E(n('SPURIOUS' if grad else 'NONZERO_AT_EXACT_ZERO'), nan or ysgn != 0 if ez else None, row)
    if ky == 'fin' and kt == 'fin':
        d = ulp_distance(int(y), int(tb), fmt)
        E(n('ULP8'), d >= 8, row)
        if d > c[n('maxULP')]:
            c[n('maxULP')] = d
        if not grad:
            if vt != 0 and abs(vt) >= fmt.minimum_normal and (not ez):
                fl = [False] * 3 if d == 0 else rel_flags(tc, vy, val, s)
                for (_, lab), b in zip(RELS, fl):
                    if b is None:
                        c[n(lab + '_UNDECIDED')] += 1
                    E(n(lab), b, row)
            else:
                for _, lab in RELS:
                    E(n(lab), None, row)
            c[n('out_negative')] += vy < 0
            c[n('truth_negative')] += tsgn < 0
            c[n('neg_n')] += 1
        return d
    E(n('ULP8'), None, row)
    if not grad:
        for _, lab in RELS:
            E(n(lab), None, row)
    return None

def boundary(c, key, y, tb, q, fmt):
    if not q & T.Q_CERT:
        c[key + '|UNCERT'] += 1
        return
    m = fmt.sign_mask - 1
    kt, vt = _kind(tb, fmt)
    cls = ('+inf' if vt > 0 else '-inf') if kt == 'inf' and (not q & T.Q_OVF) else 'zero' if tb & m == 0 and (not q & T.Q_RZ) else 'finite'
    k = f'{key}:{cls}'
    c[k + '|n'] += 1
    if y is None:
        c[k + '|BG_NO_OUTPUT'] += 1
        return
    ky, vy = _kind(y, fmt)
    if ky == 'nan':
        c[k + '|BG_NAN'] += 1
    elif y == tb or (y & m == 0 and tb & m == 0):
        c[k + '|BG_MATCH'] += 1
    elif cls in ('+inf', '-inf'):
        c[k + ('|BG_MISSED_POS_INF' if cls == '+inf' else '|BG_MISSED_NEG_INF')] += 1
    else:
        c[k + '|BG_OTHER'] += 1

def vector_metrics(E, tc, side, red, r, ys, tbs, qs, sts, vals, s, fmt):
    c = E.acc.cnt
    sfx = f':{side}:{red}'
    names = ['REL_L2_1'] + [l for _, l in COSS] + ['REVERSED', 'VANISH_VEC']

    def na(reason):
        c[f'GV_{reason}{sfx}'] += 1
        for nm in names:
            E(f'GV_{nm}{sfx}', None)
    if not all((st in REGULAR for st in sts)) or not all((q & T.Q_CERT for q in qs)) or any((q & T.Q_OVF for q in qs)):
        return na('NOT_APPLICABLE')
    if ys is None:
        return na('NO_OUTPUT')
    c[f'GV_n{sfx}'] += 1
    m = fmt.sign_mask - 1
    kinds = [_kind(y, fmt) for y in ys]
    tnz = any((st != T.EXGZ for st in sts))
    if not all((k == 'fin' for k, _ in kinds)):
        E(f'GV_REL_L2_1{sfx}', True if tnz else None)
        for nm in names[1:]:
            E(f'GV_{nm}{sfx}', None)
        return
    yv = [v for _, v in kinds]
    ynz = any((v != 0 for v in yv))
    E(f'GV_VANISH_VEC{sfx}', not ynz and any((tb & m != 0 for tb in tbs)))
    if not tnz:
        for nm in names[:-1]:
            E(f'GV_{nm}{sfx}', None)
        return
    if all((y == tb or (y | tb) & m == 0 for y, tb in zip(ys, tbs))) and all((abs(_kind(tb, fmt)[1]) >= fmt.minimum_normal or st == T.EXGZ for tb, st in zip(tbs, sts))):
        res = [False] * 5
    else:

        def f():
            sa = fraction_to_arb(s)
            tb_ = [tc.ball(v) * sa for v in vals]
            ya = [fraction_to_arb(v) for v in yv]
            err2 = sum(((a - b) * (a - b) for a, b in zip(ya, tb_)), arb(0))
            tt = sum((b * b for b in tb_), arb(0))
            out = [_cmp(err2, tt * fraction_to_arb(Fr(1, 10000)))]
            if not ynz:
                return out + [0, 0, 0, 0]
            yy = sum((a * a for a in ya), arb(0))
            cos = sum((a * b for a, b in zip(ya, tb_)), arb(0)) / (yy * tt).sqrt()
            for th, _ in COSS:
                g = _cmp(cos, fraction_to_arb(th))
                out.append(None if g is None else not g)
            g = _cmp(cos, arb(0))
            out.append(None if g is None else not g)
            return out
        res = decide(f)
    for nm, b in zip(names[:-1], res):
        if nm != 'REL_L2_1' and (not ynz):
            b = None
        elif b is None:
            c[f'GV_{nm}_UNDECIDED{sfx}'] += 1
        E(f'GV_{nm}{sfx}', b)

def score_method(acc, tc, truth, out, fmt):
    E = Emit(acc)
    c = acc.cnt
    B, K = (tc.B, tc.K)
    n = B * K
    m = fmt.sign_mask - 1
    inf = fmt.infinity_bits
    c['batches'] += 1
    for red in ('none',) + REDS:
        if out['call_status'][red] != 'OK':
            c[f'CALL_ERROR:{red}'] += 1
        if out['grad_status'][red] != 'OK':
            c[f"{out['grad_status'][red]}:{red}"] += 1
    yn = out['none']
    maxd = 0
    for i in range(n):
        d = cmp_scalar(E, tc, 'E', '', None if yn is None else yn[i], truth['L_bits'][i], truth['L_q'][i], truth['L_st'][i], tc.L[i], Fr(1), fmt, row=i // K, tag=lambda i=i: 'INTERMEDIATE_OVERFLOW' if tc.interm_overflow(i) else 'OTHER')
        if d and d > maxd:
            maxd = d

    def s_tag(red):

        def f():
            if truth['S_q']['sum'] & T.Q_OVF or any((q & T.Q_OVF for q in truth['L_q'])) or any((tc.interm_overflow(i) for i in range(n))):
                return 'INTERMEDIATE_OVERFLOW'

            def g():
                tot = arb(0)
                for v in tc.L:
                    b = tc.ball(v)
                    if b is not None:
                        tot += abs(b)
                return bool(tot > fraction_to_arb(bits_to_fraction(inf - 1, fmt).value))
            with arb_precision(256):
                big = g()
            return 'PARTIAL_SUM_OVERFLOW_POSSIBLE' if big else 'OTHER'
        return f
    sd = {}
    for red in REDS:
        sd[red] = cmp_scalar(E, tc, 'S', ':' + red, out[red], truth['S_bits'][red], truth['S_q'][red], truth['S_st'], tc.S, tc.scales[red], fmt, tag=s_tag(red))
    for red in REDS:
        s = tc.scales[red]
        for side, vals in (('gx', tc.gx), ('gt', tc.gt)):
            g = out[f'{red}_{side}']
            sts = truth[side + '_st']
            tbs = truth[side + '_bits'][red]
            qs = truth[side + '_q'][red]
            q1 = truth[side + '_q']['sum']
            for i in range(n):
                y = None if g is None else g[i]
                if sts[i] in (T.ONESIDED, T.LIMIT):
                    boundary(c, f'BG:{side}:{red}:{sts[i]}', y, tbs[i], qs[i], fmt)
                else:
                    cmp_scalar(E, tc, 'G', f':{side}:{red}', y, tbs[i], qs[i], sts[i], vals[i], s, fmt, row=i // K, grad=True, tag=lambda i=i, q1=q1: 'INTERMEDIATE_OVERFLOW' if q1[i] & T.Q_OVF or tc.interm_overflow(i) else 'OTHER')
            for r in range(B):
                sl = slice(r * K, (r + 1) * K)
                vector_metrics(E, tc, side, red, r, None if g is None else g[sl], tbs[sl], qs[sl], sts[sl], vals[sl], s, fmt)
        gx, gt = (out[f'{red}_gx'], out[f'{red}_gt'])
        for r in range(B):
            sl = slice(r * K, (r + 1) * K)
            vector_metrics(E, tc, 'both', red, r, None if gx is None or gt is None else list(gx[sl]) + list(gt[sl]), list(truth['gx_bits'][red][sl]) + list(truth['gt_bits'][red][sl]), list(truth['gx_q'][red][sl]) + list(truth['gt_q'][red][sl]), list(truth['gx_st'][sl]) + list(truth['gt_st'][sl]), tc.gx[sl] + tc.gt[sl], s, fmt)
    if out['none_gx'] is not None and out['sum_gx'] is not None:
        c['GNONE_n'] += 1
        c['GNONE_ne_SUM'] += list(out['none_gx']) != list(out['sum_gx']) or list(out['none_gt']) != list(out['sum_gt'])
    if out.get('Y_seq') is not None and out['sum'] is not None and truth['S_q']['sum'] & T.Q_CERT and (truth['S_st'] != T.DOMAIN):
        ts = truth['S_bits']['sum']
        ys = out['sum']
        c['ATTR_n'] += 1
        c['ATTR_sum_ne_seq'] += ys != out['Y_seq']
        fin = lambda b: b & m < inf
        if fin(ys) and fin(out['Y_seq']):
            acc.stats['ATTR_ULP_sum_vs_seq'].append(ulp_distance(int(ys), int(out['Y_seq']), fmt))
        ye = out.get('Y_exact')
        if ye is not None:
            c['ATTR_exact_n'] += 1
            c['ATTR_sum_ne_exactsum'] += ys != ye
            c['ATTR_exactsum_ne_truth'] += ye != ts
            if fin(ys) and fin(ye) and fin(ts):
                a = ulp_distance(int(ye), int(ts), fmt)
                b = ulp_distance(int(ys), int(ye), fmt)
                c['ATTR_fin_n'] += 1
                c['ATTR_exactsum_ULP8'] += a >= 8
                c['ATTR_summation_ULP8'] += b >= 8
                acc.stats['ATTR_ULP_elementwise_part'].append(a)
                acc.stats['ATTR_ULP_summation_part'].append(b)
        if out.get('sum_noncontig') is not None:
            c['REPRO_noncontig_n'] += 1
            c['REPRO_noncontig_ne'] += out['sum_noncontig'] != ys
            if fin(ys) and fin(out['sum_noncontig']):
                acc.stats['REPRO_ULP_noncontig'].append(ulp_distance(int(ys), int(out['sum_noncontig']), fmt))
    if tc.E is not None:
        for red in REDS:
            y = out[red]
            eb = truth['E2E_bits'][red]
            eq = truth['E2E_q'][red]
            s = tc.scales[red]
            sfx = ':' + red
            if not eq & T.Q_CERT:
                c['TOT_E2E_UNCERT' + sfx] += 1
                continue
            if y is None:
                c['TOT_NO_OUTPUT' + sfx] += 1
                for nm in ('ULP8', 'REL1', 'REL10', 'REL100', 'NEG', 'NONZERO'):
                    E(f'TOT_{nm}{sfx}', None)
                continue
            ky, vy = _kind(y, fmt)
            if ky == 'fin':
                d = ulp_distance(int(y), int(eb), fmt)
                E('TOT_ULP8' + sfx, d >= 8)
                acc.stats['TOT_ULP' + sfx].append(d)
            else:
                E('TOT_ULP8' + sfx, None)
                c['TOT_nonfinite' + sfx] += 1
            pos = tc.E.status != T.EXZ
            if not pos:
                fl = [None] * 3
            elif ky != 'fin':
                fl = [True] * 3
            else:
                fl = rel_flags(tc, vy, tc.E, s)
            for (_, lab), b in zip(RELS, fl):
                if pos and b is None:
                    c[f'TOT_{lab}_UNDECIDED{sfx}'] += 1
                E(f'TOT_{lab}{sfx}', b)
            E('TOT_NEG' + sfx, None if ky == 'nan' else vy < 0 if ky == 'fin' else vy < 0)
            E('TOT_NONZERO' + sfx, None if ky == 'nan' else ky == 'inf' or vy != 0)
            if ky == 'fin':
                with arb_precision(256):
                    acc.stats['TOT_ABS' + sfx].append(float(abs(fraction_to_arb(vy) - tc.ball(tc.E) * fraction_to_arb(s)).mid()))
    flags = E.close()
    return (flags, {'E_maxULP': maxd, 'S_ULP': sd})

def score_truth(acc, tc, truth, fmt):
    E = Emit(acc)
    c = acc.cnt
    c['batches'] += 1
    S, Ev = (tc.S, tc.E)
    if S.status == T.DOMAIN or S.infinite:
        c['R_NOT_SCORED'] += 1
        return E.close()
    for red in REDS:
        sb, sq, eb, eq = (truth['S_bits'][red], truth['S_q'][red], truth['E2E_bits'][red], truth['E2E_q'][red])
        if sq & T.Q_CERT and eq & T.Q_CERT and (not sq & T.Q_OVF):
            d = ulp_distance(int(sb), int(eb), fmt)
            E(f'R_ULP8:{red}', d >= 8)
            acc.stats[f'R_ULP:{red}'].append(d)
        else:
            c[f'R_ULP_UNCERT:{red}'] += 1

    def sgn():
        return _sign(tc.ball(S))
    sg = 0 if S.status == T.EXZ else decide(sgn) if S.exact is None else (S.exact > 0) - (S.exact < 0)
    if sg is None:
        c['R_SIGN_UNDECIDED'] += 1
    E('R_NEG', None if sg is None else sg < 0)
    E('R_NONZERO', None if sg is None else sg != 0)
    if Ev.status == T.EXZ:
        fl = [None] * 3
    elif sg == 0:
        fl = [True] * 3
    else:

        def f():
            sb_, eb_ = (tc.ball(S), tc.ball(Ev))
            a = abs(sb_ - eb_)
            return [_cmp(a, eb_ * fraction_to_arb(th)) for th, _ in RELS]
        fl = decide(f)
    for (_, lab), b in zip(RELS, fl):
        if Ev.status != T.EXZ and b is None:
            c[f'R_{lab}_UNDECIDED'] += 1
        E('R_' + lab, b)
    with arb_precision(256):
        d = abs(tc.ball(S) - tc.ball(Ev))
        for red in REDS:
            acc.stats[f'R_ABS:{red}'].append(float((d * fraction_to_arb(tc.scales[red])).mid()))
    return E.close()

def pair(acc_pair, fa, fb):
    for k, x in fa.items():
        y = fb.get(k)
        if y is None:
            continue
        acc_pair[k + '|' + ('persistent' if x and y else 'fixed' if x else 'new_bad' if y else 'both_good')] += 1
