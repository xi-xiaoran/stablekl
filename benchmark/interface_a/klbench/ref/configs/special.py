from __future__ import annotations
from fractions import Fraction
from math import factorial
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import arb_precision
from .common import unpack, arbs, mps, masks, spec, rational_grads, positive_domain
Params = dict[str, list[Fraction]]
_GAMMA = ('concentration', 'rate')
_BETA = ('concentration1', 'concentration0')

def trigamma_arb(x: arb) -> arb:
    return arb(2).zeta(x)

def _log_beta(a: arb, b: arb) -> arb:
    return a.lgamma() + b.lgamma() - (a + b).lgamma()

class Gamma:
    PARAM_SPEC = spec(_GAMMA, _GAMMA)

    @staticmethod
    def domain_ok(p: Params, q: Params) -> bool | str:
        return positive_domain(p, q, _GAMMA, _GAMMA)

    @staticmethod
    def semantic_equal(p: Params, q: Params) -> bool:
        a, b, c, d = unpack(p, q, _GAMMA)
        return a == c and b == d

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            a, b, c, d = unpack(p, q, _GAMMA)
            x, y, z, w = arbs((a, b, c, d))
            return (x - z) * x.digamma() - x.lgamma() + z.lgamma() + z * ((y - w) / w).log1p() + x * (w - y) / y

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int):
        with arb_precision(prec):
            a, b, c, d = arbs(unpack(p, q, _GAMMA))
            return {'P': {'concentration': [(a - c) * trigamma_arb(a) + d / b - 1], 'rate': [c / b - a * d / (b * b)]}, 'Q': {'concentration': [-a.digamma() + c.digamma() + (b / d).log()], 'rate': [a / b - c / d]}}

    @staticmethod
    def exact_grad(p: Params, q: Params):
        a, b, c, d = unpack(p, q, _GAMMA)
        return rational_grads(_GAMMA, (d / b - 1 if a == c else None, c / b - a * d / b ** 2), (Fraction(0) if a == c and b == d else None, a / b - c / d))

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        a, b, c, d = unpack(p, q, _GAMMA)
        return masks(_GAMMA, (a == c and b == d, c * b == a * d), (a == c and b == d, a * d == c * b))

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            a, b, c, d = mps(unpack(p, q, _GAMMA))
            return (a - c) * mp.digamma(a) - mp.loggamma(a) + mp.loggamma(c) + c * (mp.log(b) - mp.log(d)) + a * (d / b - 1)

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            a, b, c, d = mps(unpack(p, q, _GAMMA))
            logp0 = a * mp.log(b) - mp.loggamma(a)
            logq0 = c * mp.log(d) - mp.loggamma(c)

            def f(t):
                logx = mp.log(t / b)
                logratio = logp0 - logq0 + (a - c - 0) * logx + (d - b) * t / b
                density = mp.exp((a - 1) * mp.log(t) - t - mp.loggamma(a))
                return density * logratio
            first = mp.quad(lambda u: f(u ** 64) * 64 * u ** 63, [0, mp.mpf(1) / 2, 1])
            return first + mp.quad(f, [1, 16, mp.inf])

class Beta:
    PARAM_SPEC = spec(_BETA, _BETA)

    @staticmethod
    def domain_ok(p: Params, q: Params) -> bool | str:
        return positive_domain(p, q, _BETA, _BETA)

    @staticmethod
    def semantic_equal(p: Params, q: Params) -> bool:
        a, b, c, d = unpack(p, q, _BETA)
        return a == c and b == d

    @staticmethod
    def exact_kl(p: Params, q: Params) -> Fraction | None:
        a, b, c, d = unpack(p, q, _BETA)
        if all((x.denominator == 1 for x in (a, b, c, d))):
            ia, ib, ic, id_ = map(int, (a, b, c, d))
            swapped = a == d and b == c
            if max(ia, ib, ic, id_) > 4096 and (not swapped):
                return None
            equal_normalizers = swapped
            if not swapped:
                bp = Fraction(factorial(ia - 1) * factorial(ib - 1), factorial(ia + ib - 1))
                bq = Fraction(factorial(ic - 1) * factorial(id_ - 1), factorial(ic + id_ - 1))
                equal_normalizers = bp == bq
            if equal_normalizers and max(ia, ib) <= 4096:
                hp = _harmonic(ia - 1)
                hq = _harmonic(ib - 1)
                hs = _harmonic(ia + ib - 1)
                return (a - c) * (hp - hs) + (b - d) * (hq - hs)
        return None

    @staticmethod
    def exact_grad(p: Params, q: Params):
        a, b, c, d = unpack(p, q, _BETA)
        if not all((x.denominator == 1 for x in (a, b, c, d))):
            return rational_grads(_BETA, (None, None), (None, None))
        ia, ib, ic, id_ = map(int, (a, b, c, d))
        if max(ia, ib, ic, id_) > 4096:
            return rational_grads(_BETA, (None, None), (None, None))
        h2a = _harmonic(ia - 1, 2)
        h2b = _harmonic(ib - 1, 2)
        h2s = _harmonic(ia + ib - 1, 2)
        ha = _harmonic(ia - 1)
        hb = _harmonic(ib - 1)
        hs = _harmonic(ia + ib - 1)
        hc = _harmonic(ic - 1)
        hd = _harmonic(id_ - 1)
        ht = _harmonic(ic + id_ - 1)
        gp_a = -(a - c) * h2a + (a + b - c - d) * h2s if b == d else None
        gp_b = -(b - d) * h2b + (a + b - c - d) * h2s if a == c else None
        return rational_grads(_BETA, (gp_a, gp_b), (hc - ht - ha + hs, hd - ht - hb + hs))

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            a, b, c, d = arbs(unpack(p, q, _BETA))
            return _log_beta(c, d) - _log_beta(a, b) + (a - c) * (a.digamma() - (a + b).digamma()) + (b - d) * (b.digamma() - (a + b).digamma())

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int):
        with arb_precision(prec):
            a, b, c, d = arbs(unpack(p, q, _BETA))
            s = a + b
            t = c + d
            return {'P': {'concentration1': [(a - c) * trigamma_arb(a) - (s - t) * trigamma_arb(s)], 'concentration0': [(b - d) * trigamma_arb(b) - (s - t) * trigamma_arb(s)]}, 'Q': {'concentration1': [c.digamma() - t.digamma() - a.digamma() + s.digamma()], 'concentration0': [d.digamma() - t.digamma() - b.digamma() + s.digamma()]}}

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        a, b, c, d = unpack(p, q, _BETA)
        same = a == c and b == d
        return masks(_BETA, (same, same), (same, same))

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            a, b, c, d = mps(unpack(p, q, _BETA))
            logb = lambda x, y: mp.loggamma(x) + mp.loggamma(y) - mp.loggamma(x + y)
            return logb(c, d) - logb(a, b) + (a - c) * mp.digamma(a) + (b - d) * mp.digamma(b) - (a + b - c - d) * mp.digamma(a + b)

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            a, b, c, d = mps(unpack(p, q, _BETA))
            logbp = mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)
            logbq = mp.loggamma(c) + mp.loggamma(d) - mp.loggamma(c + d)

            def f_logs(lx, ly):
                pdf = mp.exp((a - 1) * lx + (b - 1) * ly - logbp)
                return pdf * ((a - c) * lx + (b - d) * ly + logbq - logbp)

            def left(u):
                y = u ** 64 / 2
                return f_logs(mp.log(y), mp.log1p(-y)) * 32 * u ** 63

            def right(u):
                y = u ** 64 / 2
                return f_logs(mp.log1p(-y), mp.log(y)) * 32 * u ** 63
            return mp.quad(left, [0, mp.mpf(1) / 2, 1]) + mp.quad(right, [0, mp.mpf(1) / 2, 1])

def _harmonic(n: int, power: int=1) -> Fraction:
    return sum((Fraction(1, k ** power) for k in range(1, n + 1)), Fraction(0))
