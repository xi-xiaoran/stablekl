from __future__ import annotations
from collections import Counter
from fractions import Fraction as Fr
from flint import arb
from klbench.fp import bits_to_fraction, ulp_distance
from kldivbench import score as S, truth as T
from . import bounds as BD
from .bounds import A
REDS = ('sum', 'mean', 'batchmean')
BC = set(S.LEVEL_B) | set(S.LEVEL_C)
TH2 = Fr(1, 10000)

def _same(y, tb, fmt):
    if y is None:
        return False
    m = fmt.sign_mask - 1
    d = bits_to_fraction(int(y), fmt)
    return d.kind != 'nan' and (y == tb or (y & m == 0 and tb & m == 0))

def _flags(tc, name, sfx, y, tb, q, st, val, s, fmt, grad):
    E = S.Emit(S.Acc())
    S.cmp_scalar(E, tc, name, sfx, y, tb, q, st, val, s, fmt, grad=grad)
    return {k: v for k, v in E.flags.items() if k.split(':')[0].split('_', 1)[1] in BC}

def components(tc, truth):
    n = tc.B * tc.K
    for i in range(n):
        yield ('E', ('none', 'out', i), truth['L_bits'][i], truth['L_q'][i], truth['L_st'][i], tc.L[i], Fr(1), False, False)
    for red in REDS:
        yield (f'S:{red}', (red, 'out', 0), truth['S_bits'][red], truth['S_q'][red], truth['S_st'], tc.S, tc.scales[red], False, False)
    for red in REDS:
        for side, vals in (('gx', tc.gx), ('gt', tc.gt)):
            sts = truth[side + '_st']
            tbs = truth[side + '_bits'][red]
            qs = truth[side + '_q'][red]
            for i in range(n):
                bg = sts[i] in (T.ONESIDED, T.LIMIT)
                yield (('BG:' if bg else 'G:') + f'{side}:{red}', (red, side, i), tbs[i], qs[i], sts[i], vals[i], tc.scales[red], True, bg)

def _get(out, key):
    red, part, i = key
    v = out[red] if part == 'out' else out[f'{red}_{part}']
    if v is None:
        return None
    return v if part == 'out' and red != 'none' else v[i]

def h1_batch(cnt, reg, ex, tc, truth, oA, oM, fmt, where):
    for cls, key, tb, q, st, val, s, grad, bg in components(tc, truth):
        if st in (T.UNDEF, T.DOMAIN) or not q & T.Q_CERT:
            cnt[cls + '|unscored'] += 1
            continue
        yA, yM = (_get(oA, key), _get(oM, key))
        a, m = (_same(yA, tb, fmt), _same(yM, tb, fmt))
        cnt[cls + '|n'] += 1
        cnt[cls + '|crA'] += a
        cnt[cls + '|crM'] += m
        cnt[cls + ('|both_cr' if a and m else '|fixed' if m else '|regressed' if a else '|both_not_cr')] += 1
        cnt[cls + '|differ'] += yM != yA
        if m:
            continue
        if not a and yM != yA and (yM is not None) and (yA is not None):
            fa, fm_, ft = (bits_to_fraction(int(v), fmt) for v in (yA, yM, tb))
            if fa.kind == fm_.kind == ft.kind == 'finite':
                da, dm = (ulp_distance(int(yA), int(tb), fmt), ulp_distance(int(yM), int(tb), fmt))
                cnt[cls + ('|bnc_closer' if dm < da else '|bnc_farther' if dm > da else '|bnc_same_distance')] += 1
            else:
                cnt[cls + ('|bnc_A_nonfinite' if fa.kind != 'finite' and fm_.kind == 'finite' else '|bnc_other_nonfinite')] += 1
        if a:
            d = None
            if yM is not None and bits_to_fraction(int(yM), fmt).kind == 'finite' and (bits_to_fraction(int(tb), fmt).kind == 'finite'):
                d = ulp_distance(int(yM), int(tb), fmt)
            reg.append(dict(where, cls=cls, key=list(key), y=yM, truth=int(tb), ulp=d, truth_finite=bits_to_fraction(int(tb), fmt).kind == 'finite'))
        if yM == yA or bg:
            continue
        name, sfx = ('E', '') if cls == 'E' else (cls.split(':')[0], ':' + cls.split(':', 1)[1])
        fA = _flags(tc, name, sfx, yA, tb, q, st, val, s, fmt, grad)
        fM = _flags(tc, name, sfx, yM, tb, q, st, val, s, fmt, grad)
        for k, v in fM.items():
            if v and (not fA.get(k, False)):
                cnt[f'H1a_new_bad|{k}'] += 1
                if len(ex) < 50:
                    ex.append(dict(where, metric=k, key=list(key), yA=yA, yM=yM, truth=int(tb)))

def eimpl_batch(cnt, tc, truth, outs, fmt):
    B, K = (tc.B, tc.K)
    m_ = fmt.sign_mask - 1
    val = lambda b: bits_to_fraction(int(b), fmt)
    for red in REDS:
        s = tc.scales[red]
        for side, vals in (('gx', tc.gx), ('gt', tc.gt)):
            sts = truth[side + '_st']
            tbs = truth[side + '_bits'][red]
            qs = truth[side + '_q'][red]
            sfx = f':{side}:{red}'
            for r in range(B):
                sl = range(r * K, (r + 1) * K)
                if not all((sts[i] in S.REGULAR for i in sl)) or not all((qs[i] & T.Q_CERT for i in sl)) or any((qs[i] & T.Q_OVF for i in sl)):
                    cnt['EV_NOT_APPLICABLE' + sfx] += 1
                    continue
                if all((sts[i] == T.EXGZ for i in sl)):
                    cnt['EV_ZERO_TRUTH' + sfx] += 1
                    continue
                g = [val(tbs[i]).value for i in sl]
                tt = {}

                def norm2():
                    from flint import ctx
                    if ctx.prec not in tt:
                        tt[ctx.prec] = sum(((tc.ball(vals[i]) * A(s)) ** 2 for i in sl), arb(0))
                    return tt[ctx.prec]
                small = any((tbs[i] & m_ == 0 and sts[i] != T.EXGZ or 0 < abs(g[k]) < fmt.minimum_normal for k, i in enumerate(sl)))
                if small:

                    def ff():
                        e2 = sum(((A(g[k]) - tc.ball(vals[i]) * A(s)) ** 2 for k, i in enumerate(sl)), arb(0))
                        return S._cmp(e2, norm2() * A(TH2))
                    b = S.decide(ff)
                else:
                    b = False
                cnt['EFORMAT_REL1' + sfx + ('|und' if b is None else '|n')] += 1
                cnt['EFORMAT_REL1' + sfx + '|bad'] += bool(b)
                for meth, o in outs.items():
                    y = o[f'{red}_{side}']
                    pre = f'{meth}|'
                    if y is None:
                        cnt[pre + 'EIMPL_NO_OUTPUT' + sfx] += 1
                        continue
                    ys = [val(y[i]) for i in sl]
                    cnt[pre + 'EIMPL_REL1' + sfx + '|n'] += 1
                    for k, i in enumerate(sl):
                        if g[k] != 0:
                            cnt[pre + 'EIMPL_SIGN' + sfx + '|n'] += 1
                            cnt[pre + 'EIMPL_ZERO' + sfx + '|n'] += 1
                            if ys[k].kind == 'nan':
                                cnt[pre + 'EIMPL_SIGN' + sfx + '|bad'] += 1
                                continue
                            v = ys[k].value if ys[k].kind == 'finite' else ys[k].sign
                            cnt[pre + 'EIMPL_SIGN' + sfx + '|bad'] += (v > 0) != (g[k] > 0) and v != 0
                            cnt[pre + 'EIMPL_ZERO' + sfx + '|bad'] += v == 0
                    if not all((v.kind == 'finite' for v in ys)):
                        cnt[pre + 'EIMPL_REL1' + sfx + '|bad'] += 1
                        continue
                    e2 = sum(((v.value - g[k]) ** 2 for k, v in enumerate(ys)), Fr(0))
                    if e2 == 0:
                        continue
                    bi = S.decide(lambda: S._cmp(A(e2), norm2() * A(TH2)))
                    if bi is None:
                        cnt[pre + 'EIMPL_REL1' + sfx + '|und'] += 1
                    cnt[pre + 'EIMPL_REL1' + sfx + '|bad'] += bool(bi)

def check_regression(tc, truth, r, method, dt, lt, value, fmt):
    if value is None:
        return 'violation'
    cls = r['cls']
    red, part, i = r['key']
    K = tc.K
    x = lambda j: T.decode(tc.x[j // K][j % K], fmt)
    t = lambda j: T.decode(tc.t[j // K][j % K], fmt)

    def f():
        if cls == 'E':
            eb = tc.ball(tc.L[i])
            b = BD.elem_bound(method, dt, lt, x(i), t(i), abs(eb))
            return False if b is None else BD.within(value, eb, b)
        if cls.startswith('S:'):
            s = tc.scales[red]
            n = tc.B * K
            sb = arb(0)
            sa = arb(0)
            for j in range(n):
                lb = tc.ball(tc.L[j])
                if lb is None:
                    return False
                if tc.L[j].status == T.CERT and tc.L[j].exact is None:
                    bj = BD.elem_bound(method, dt, lt, x(j), t(j), abs(lb))
                    if bj is None:
                        return False
                    sb += bj
                sa += abs(lb)
            eb = tc.ball(tc.S) * A(s)
            return BD.within(value, eb, BD.sum_bound(method, dt, n, sb, sa, abs(eb), s))
        side = part
        s = tc.scales[red]
        v = (tc.gx if side == 'gx' else tc.gt)[i]
        eb = tc.ball(v)
        if eb is None:
            return False
        eb = eb * A(s)
        return BD.within(value, eb, BD.grad_bound(method, dt, lt, side, x(i), t(i), abs(eb), s))
    if cls.startswith('BG:') and r.get('truth_finite') is not True:
        return 'violation'
    res = S.decide(f)
    return 'undecided' if res is None else 'exception' if res else 'violation'
