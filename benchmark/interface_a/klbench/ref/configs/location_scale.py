from __future__ import annotations
from fractions import Fraction
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import arb_precision
from .common import unpack, arbs, mps, masks, spec, rational_grads, positive_domain
Params = dict[str, list[Fraction]]
_NAMES = ('loc', 'scale')

def _domain(p: Params, q: Params) -> bool | str:
    return positive_domain(p, q, _NAMES, ('scale',))

def _equal(p: Params, q: Params) -> bool:
    a, b, c, d = unpack(p, q, _NAMES)
    return a == c and b == d

class Cauchy:
    PARAM_SPEC = spec(_NAMES, ('scale',))
    domain_ok = staticmethod(_domain)
    semantic_equal = staticmethod(_equal)

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            mp_, bp, mq, bq = unpack(p, q, _NAMES)
            delta = mp_ - mq
            diff = bp - bq
            x, y = arbs((bp, bq))
            return (arbs((diff * diff + delta * delta,))[0] / (4 * x * y)).log1p()

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int):
        with arb_precision(prec):
            m, b, n, c = arbs(unpack(p, q, _NAMES))
            delta = m - n
            s = b + c
            denominator = s * s + delta * delta
            return {'P': {'loc': [2 * delta / denominator], 'scale': [2 * s / denominator - 1 / b]}, 'Q': {'loc': [-2 * delta / denominator], 'scale': [2 * s / denominator - 1 / c]}}

    @staticmethod
    def exact_grad(p: Params, q: Params):
        m, b, n, c = unpack(p, q, _NAMES)
        delta = m - n
        s = b + c
        den = s * s + delta * delta
        return rational_grads(_NAMES, (2 * delta / den, 2 * s / den - 1 / b), (-2 * delta / den, 2 * s / den - 1 / c))

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        m, b, n, c = unpack(p, q, _NAMES)
        delta = m - n
        den = (b + c) ** 2 + delta ** 2
        return masks(_NAMES, (delta == 0, 2 * b * (b + c) == den), (delta == 0, 2 * c * (b + c) == den))

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            m, b, n, c = mps(unpack(p, q, _NAMES))
            return mp.log(((b + c) ** 2 + (m - n) ** 2) / (4 * b * c))

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            m, b, n, c = mps(unpack(p, q, _NAMES))

            def f(theta):
                x = m + b * mp.tan(theta)
                return mp.log(c / b) + mp.log(1 + ((x - n) / c) ** 2) - mp.log(1 + ((x - m) / b) ** 2)
            return mp.quad(f, [-mp.pi / 2, -mp.pi / 4, 0, mp.pi / 4, mp.pi / 2]) / mp.pi

class Laplace:
    PARAM_SPEC = spec(_NAMES, ('scale',))
    domain_ok = staticmethod(_domain)
    semantic_equal = staticmethod(_equal)

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            m, b, n, c = unpack(p, q, _NAMES)
            a = abs(m - n)
            x, y = arbs((b, c))
            amplitude = arbs((a,))[0]
            return ((y - x) / x).log1p() + amplitude / y + x / y * (-amplitude / x).exp() - 1

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int):
        with arb_precision(prec):
            m, b, n, c = unpack(p, q, _NAMES)
            a = abs(m - n)
            x, y, z = arbs((b, c, a))
            e = (-z / x).exp()
            sign = 1 if m > n else -1 if m < n else 0
            dloc = sign * -(-z / x).expm1() / y
            return {'P': {'loc': [dloc], 'scale': [-1 / x + e * (1 + z / x) / y]}, 'Q': {'loc': [-dloc], 'scale': [1 / y - (z + x * e) / (y * y)]}}

    @staticmethod
    def exact_grad(p: Params, q: Params):
        m, b, n, c = unpack(p, q, _NAMES)
        if m == n:
            return rational_grads(_NAMES, (Fraction(0), -1 / b + 1 / c), (Fraction(0), 1 / c - b / c ** 2))
        return rational_grads(_NAMES, (None, None), (None, None))

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        m, b, n, c = unpack(p, q, _NAMES)
        d = m == n
        return masks(_NAMES, (d, d and b == c), (d, d and b == c))

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            m, b, n, c = mps(unpack(p, q, _NAMES))
            a = abs(m - n)
            return mp.log(c / b) + a / c + b / c * mp.exp(-a / b) - 1

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            m, b, n, c = mps(unpack(p, q, _NAMES))

            def integrand(t):
                x = m + b * t
                lp = -mp.log(2 * b) - abs(t)
                lq = -mp.log(2 * c) - abs(x - n) / c
                return mp.exp(-abs(t)) / 2 * (lp - lq)
            knot = (n - m) / b
            points = sorted({-mp.inf, mp.mpf(0), knot, mp.inf})
            return mp.quad(integrand, points)

class Gumbel:
    PARAM_SPEC = spec(_NAMES, ('scale',))
    domain_ok = staticmethod(_domain)
    semantic_equal = staticmethod(_equal)

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            m, b, n, c = arbs(unpack(p, q, _NAMES))
            delta = m - n
            r = b / c
            t = delta / c
            a = (-t).exp() * (1 + r).gamma()
            return ((c - b) / b).log1p() + t + arb.const_euler() * r + a - 1 - arb.const_euler()

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int):
        with arb_precision(prec):
            m, b, n, c = arbs(unpack(p, q, _NAMES))
            t = (m - n) / c
            r = b / c
            e = (-t).exp() * (1 + r).gamma()
            psi = (1 + r).digamma()
            gamma = arb.const_euler()
            dloc = (1 - e) / c
            return {'P': {'loc': [dloc], 'scale': [-1 / b + gamma / c + e * psi / c]}, 'Q': {'loc': [-dloc], 'scale': [(1 + (e - 1) * t - gamma * r - e * r * psi) / c]}}

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        same = _equal(p, q)
        return masks(_NAMES, (same, same), (same, same))

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            m, b, n, c = mps(unpack(p, q, _NAMES))
            r = b / c
            t = (m - n) / c
            return mp.log(c / b) + t + mp.euler * r + mp.exp(-t) * mp.gamma(1 + r) - 1 - mp.euler

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            m, b, n, c = mps(unpack(p, q, _NAMES))

            def f(u):
                x = m - b * mp.log(u)
                tp = (x - m) / b
                tq = (x - n) / c
                lp = -mp.log(b) - tp - mp.exp(-tp)
                lq = -mp.log(c) - tq - mp.exp(-tq)
                return mp.exp(-u) * (lp - lq)
            return mp.quad(f, [0, mp.mpf(1) / 16, 1, 16, mp.inf])
