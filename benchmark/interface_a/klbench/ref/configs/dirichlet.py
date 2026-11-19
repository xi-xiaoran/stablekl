from __future__ import annotations
from fractions import Fraction
from math import factorial
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import arb_precision, fraction_to_arb
from .special import trigamma_arb
Params = dict[str, list[Fraction]]
_EXACT_INT_LIMIT = 4096
_HARMONIC_CACHE: dict[int, list[Fraction]] = {1: [Fraction(0)], 2: [Fraction(0)]}

def _harmonic(n: int, power: int=1) -> Fraction:
    table = _HARMONIC_CACHE[power]
    while len(table) <= n:
        j = len(table)
        table.append(table[-1] + Fraction(1, j ** power))
    return table[max(n, 0)]

def _vec(params: Params, k: int) -> list[Fraction]:
    if set(params) != {'concentration'} or len(params['concentration']) != k:
        raise ValueError(f"Expected 'concentration' of length {k}")
    return list(params['concentration'])

def _mp(x: Fraction) -> Any:
    return mp.mpf(x.numerator) / mp.mpf(x.denominator)

def _all_int(values) -> bool:
    return all((v.denominator == 1 for v in values))

def make(k: int):

    class Dirichlet:
        K = k
        PARAM_SPEC = {'concentration': {'shape': (k,), 'differentiable': True, 'domain': 'positive'}}

        @staticmethod
        def domain_ok(p: Params, q: Params) -> bool | str:
            try:
                a, b = (_vec(p, k), _vec(q, k))
            except (KeyError, ValueError) as exc:
                return str(exc)
            return True if all((v > 0 for v in a + b)) else 'concentration must be positive'

        @staticmethod
        def semantic_equal(p: Params, q: Params) -> bool:
            return _vec(p, k) == _vec(q, k)

        @staticmethod
        def kl_arb(p: Params, q: Params, prec: int) -> arb:
            with arb_precision(prec):
                a, b = (_vec(p, k), _vec(q, k))
                a0, b0 = (sum(a), sum(b))
                xa = [fraction_to_arb(v) for v in a]
                xb = [fraction_to_arb(v) for v in b]
                s0 = fraction_to_arb(a0)
                psi0 = s0.digamma()
                out = s0.lgamma() - fraction_to_arb(b0).lgamma()
                for ai, bi, xi, yi in zip(a, b, xa, xb):
                    out += yi.lgamma() - xi.lgamma()
                    if ai != bi:
                        out += fraction_to_arb(ai - bi) * (xi.digamma() - psi0)
                return out

        @staticmethod
        def grad_arb(p: Params, q: Params, prec: int):
            with arb_precision(prec):
                a, b = (_vec(p, k), _vec(q, k))
                a0, b0 = (sum(a), sum(b))
                s0, t0 = (fraction_to_arb(a0), fraction_to_arb(b0))
                tri0 = trigamma_arb(s0)
                gp, gq = ([], [])
                for ai, bi in zip(a, b):
                    x, y = (fraction_to_arb(ai), fraction_to_arb(bi))
                    gp.append(fraction_to_arb(ai - bi) * trigamma_arb(x) - fraction_to_arb(a0 - b0) * tri0)
                    gq.append(y.digamma() - t0.digamma() - x.digamma() + s0.digamma())
                return {'P': {'concentration': gp}, 'Q': {'concentration': gq}}

        @staticmethod
        def exact_zero_mask(p: Params, q: Params):
            a, b = (_vec(p, k), _vec(q, k))
            same_sum = sum(a) == sum(b)
            mask = [same_sum and ai == bi for ai, bi in zip(a, b)]
            return {'P': {'concentration': list(mask)}, 'Q': {'concentration': list(mask)}}

        @staticmethod
        def exact_grad(p: Params, q: Params):
            a, b = (_vec(p, k), _vec(q, k))
            none = {'P': {'concentration': [None] * k}, 'Q': {'concentration': [None] * k}}
            if not _all_int(a + b):
                return none
            a0, b0 = (sum(a), sum(b))
            if max(a0, b0) > _EXACT_INT_LIMIT:
                return none
            ia0, ib0 = (int(a0), int(b0))
            gp: list[Fraction | None] = []
            gq: list[Fraction | None] = []
            for ai, bi in zip(a, b):
                ia, ib = (int(ai), int(bi))
                gq.append(_harmonic(ib - 1) - _harmonic(ib0 - 1) - _harmonic(ia - 1) + _harmonic(ia0 - 1))
                if ai - bi == a0 - b0:
                    gp.append(-(ai - bi) * _harmonic(ia - 1, 2) + (a0 - b0) * _harmonic(ia0 - 1, 2))
                else:
                    gp.append(None)
            return {'P': {'concentration': gp}, 'Q': {'concentration': gq}}

        @staticmethod
        def exact_kl(p: Params, q: Params) -> Fraction | None:
            a, b = (_vec(p, k), _vec(q, k))
            if a == b or not _all_int(a + b):
                return None
            a0, b0 = (sum(a), sum(b))
            if max(a0, b0) > _EXACT_INT_LIMIT:
                return None
            if sorted(a) != sorted(b):
                if max(a0, b0) > 512:
                    return None
                num_a = 1
                for v in a:
                    num_a *= factorial(int(v) - 1)
                num_b = 1
                for v in b:
                    num_b *= factorial(int(v) - 1)
                if Fraction(num_a, factorial(int(a0) - 1)) != Fraction(num_b, factorial(int(b0) - 1)):
                    return None
            h0 = _harmonic(int(a0) - 1)
            return sum(((ai - bi) * (_harmonic(int(ai) - 1) - h0) for ai, bi in zip(a, b)), Fraction(0))

        @staticmethod
        def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
            with mp.workdps(dps):
                a = [_mp(v) for v in _vec(p, k)]
                b = [_mp(v) for v in _vec(q, k)]
                a0, b0 = (mp.fsum(a), mp.fsum(b))
                log_b = lambda v, s: mp.fsum((mp.loggamma(x) for x in v)) - mp.loggamma(s)
                return log_b(b, b0) - log_b(a, a0) + mp.fsum(((x - y) * mp.digamma(x) for x, y in zip(a, b))) - (a0 - b0) * mp.digamma(a0)

        @staticmethod
        def kl_definition_check(p: Params, q: Params, dps: int=60) -> Any:
            return Dirichlet.kl_mpmath_alt(p, q, dps)
    Dirichlet.__name__ = Dirichlet.__qualname__ = f'DirichletK{k}'
    return Dirichlet
