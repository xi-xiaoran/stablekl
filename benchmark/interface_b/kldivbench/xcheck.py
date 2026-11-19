from __future__ import annotations
import random
from fractions import Fraction as Fr
import mpmath as mp
from . import truth as T
DPS = 60

def _m(v):
    return mp.mpf(v.numerator) / mp.mpf(v.denominator)

def _inside(v, val, scale_abs, c, key, info):
    c[key + '_n'] += 1
    tol = mp.mpf(10) ** (-45) * (scale_abs + mp.mpf(10) ** (-300))
    if not _m(val.lo) - tol <= v <= _m(val.hi) + tol:
        c[key + '_bad'] += 1
        if len(c.setdefault('examples', [])) < 20:
            c['examples'].append(dict(key=key, mp=mp.nstr(v, 30), lo=float(val.lo), hi=float(val.hi), **info))

def element_alt(x, t, log_target):
    if not log_target:
        lt = mp.log(t)
        return (t * mp.log(t * mp.exp(-x)), -t, mp.log(t * mp.exp(1 - x)), abs(t) * (abs(x) + abs(lt) + 1) + abs(x) + abs(lt) + 1)
    e = mp.exp(t)
    return (e * t - e * x, -1 / mp.exp(-t), e * t - e * x + e, e * (abs(t) + abs(x) + 1))

def pick(xr, tr, seed, picks, sign_mask=1 << 31):
    K = len(xr)
    if K <= picks:
        return set(range(K))
    key = lambda b: -(b & sign_mask - 1) if b & sign_mask else b
    idx = {max(range(K), key=lambda i: key(xr[i])), min(range(K), key=lambda i: key(xr[i])), max(range(K), key=lambda i: key(tr[i])), min(range(K), key=lambda i: key(tr[i]))}
    rnd = random.Random(seed)
    idx.update(rnd.sample(range(K), picks))
    return idx

def check_batch(c, xbits, tbits, fmt, log_target, vals, S, picks=8, seeds=None):
    with mp.workdps(DPS):
        tot = mp.mpf(0)
        mag = mp.mpf(0)
        ok_sum = S.exact is None and S.lo is not None
        for r, (xr, tr) in enumerate(zip(xbits, tbits)):
            K = len(xr)
            chosen = pick(xr, tr, seeds[r], picks, fmt.sign_mask) if seeds is not None else set(range(0, K, max(1, K // picks)))
            for i in range(K):
                L, gx, gt = vals[r][i]
                if L.status != T.CERT:
                    continue
                x = _m(T.decode(xr[i], fmt))
                t = _m(T.decode(tr[i], fmt))
                aL, agx, agt, m = element_alt(x, t, log_target)
                tot += aL
                mag += m
                if i not in chosen:
                    continue
                info = dict(row=r, i=i, x=int(xr[i]), t=int(tr[i]))
                for nm, v, a in (('L', L, aL), ('gx', gx, agx), ('gt', gt, agt)):
                    if v.status == T.CERT and v.exact is None and (v.lo is not None):
                        _inside(a, v, m, c, 'XCHECK_elem_' + nm, info)
                    elif v.status == T.CERT and v.exact is not None and (v.exact not in (T.PINF, T.NINF)):
                        c['XCHECK_exact_' + nm + '_n'] += 1
                        c['XCHECK_exact_' + nm + '_bad'] += abs(a - _m(v.exact)) > mp.mpf(10) ** (-45) * (m + 1)
        if ok_sum:
            _inside(tot, S, mag, c, 'XCHECK_S', dict(rows=len(xbits)))

def check_e2e(c, zps, zqs, E):
    if E.exact is not None or E.lo is None:
        return
    with mp.workdps(DPS):
        tot = mp.mpf(0)
        mag = mp.mpf(0)
        for zp, zq in zip(zps, zqs):
            a = [_m(v) for v in zp]
            b = [_m(v) for v in zq]
            ma, mb = (max(a), max(b))
            sa = mp.fsum((mp.exp(v - ma) for v in a))
            sb = mp.fsum((mp.exp(v - mb) for v in b))
            tot += mp.fsum((mp.exp(u - ma) / sa * (u - v) for u, v in zip(a, b))) - (ma + mp.log(sa)) + (mb + mp.log(sb))
            mag += max((abs(v) for v in a + b)) + 1
        _inside(tot, E, mag, c, 'XCHECK_E2E', dict(rows=len(zps)))
