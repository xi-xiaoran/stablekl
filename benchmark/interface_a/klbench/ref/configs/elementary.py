from __future__ import annotations
from fractions import Fraction
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import arb_precision, fraction_to_arb
from .common import unpack, arbs, mps, spec, masks, rational_grads, positive_domain
Params = dict[str, list[Fraction]]
_RATE = ('rate',)

class Poisson:
    PARAM_SPEC = spec(_RATE, _RATE)

    @staticmethod
    def domain_ok(p: Params, q: Params) -> bool | str:
        return positive_domain(p, q, _RATE, _RATE)

    @staticmethod
    def semantic_equal(p: Params, q: Params) -> bool:
        return p['rate'][0] == q['rate'][0]

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            a, b = unpack(p, q, _RATE)
            x, y = arbs((a, b))
            delta = x - y
            return x * (delta / y).log1p() - delta

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int):
        with arb_precision(prec):
            a, b = unpack(p, q, _RATE)
            x, y = arbs((a, b))
            return {'P': {'rate': [((x - y) / y).log1p()]}, 'Q': {'rate': [1 - x / y]}}

    @staticmethod
    def exact_grad(p: Params, q: Params):
        a, b = unpack(p, q, _RATE)
        return rational_grads(_RATE, (None,), (1 - a / b,))

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        same = Poisson.semantic_equal(p, q)
        return masks(_RATE, (same,), (same,))

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            a, b = mps(unpack(p, q, _RATE))
            return a * (mp.log(a) - mp.log(b)) + b - a

    @staticmethod
    def definition_tail_bound(p: Params, q: Params, dps: int=100, count: int | None=None) -> Any:
        with mp.workdps(dps):
            a, b = mps(unpack(p, q, _RATE))
            n = count or int(max(96, a + 30 * mp.sqrt(a) + 100))
            if n <= a:
                raise ValueError('Poisson tail requires N > mean')
            f_n = mp.exp(-a + n * mp.log(a) - mp.loggamma(n + 1))
            f_prev = mp.exp(-a + (n - 1) * mp.log(a) - mp.loggamma(n))
            return abs(mp.log(a / b)) * a * f_prev / (1 - a / n) + abs(b - a) * f_n / (1 - a / (n + 1))

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            a, b = mps(unpack(p, q, _RATE))
            n = int(max(96, a + 30 * mp.sqrt(a) + 100))
            log_ratio = mp.log(a / b)
            weight = mp.exp(-a)
            values = []
            for k in range(n):
                values.append(weight * (k * log_ratio + b - a))
                weight *= a / (k + 1)
            return mp.fsum(values)

class Exponential:
    PARAM_SPEC = spec(_RATE, _RATE)

    @staticmethod
    def domain_ok(p: Params, q: Params) -> bool | str:
        return positive_domain(p, q, _RATE, _RATE)

    @staticmethod
    def semantic_equal(p: Params, q: Params) -> bool:
        return p['rate'][0] == q['rate'][0]

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            a, b = unpack(p, q, _RATE)
            x, y = arbs((a, b))
            t = (x - y) / y
            return t.log1p() - t / (1 + t)

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int):
        with arb_precision(prec):
            a, b = arbs(unpack(p, q, _RATE))
            return {'P': {'rate': [(a - b) / (a * a)]}, 'Q': {'rate': [1 / a - 1 / b]}}

    @staticmethod
    def exact_grad(p: Params, q: Params):
        a, b = unpack(p, q, _RATE)
        return rational_grads(_RATE, ((a - b) / a ** 2,), (1 / a - 1 / b,))

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        same = Exponential.semantic_equal(p, q)
        return masks(_RATE, (same,), (same,))

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            a, b = mps(unpack(p, q, _RATE))
            return mp.log(a / b) + b / a - 1

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            a, b = mps(unpack(p, q, _RATE))
            return mp.quad(lambda u: mp.exp(-u) * (mp.log(a / b) + (b - a) * u / a), [0, 1, 16, 100, mp.inf])
