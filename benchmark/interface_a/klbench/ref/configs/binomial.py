from __future__ import annotations
from fractions import Fraction
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import arb_precision
from . import bernoulli
from .common import mpf
Params = dict[str, list[Fraction]]
_COUNTS = (1, 10, 100, 1000)

def _unpack(p: Params, q: Params, name: str) -> tuple[int, Fraction, Fraction]:
    if set(p) != {'total_count', name} or set(q) != {'total_count', name} or any((len(v[k]) != 1 for v in (p, q) for k in ('total_count', name))):
        raise ValueError('Binomial expects scalar total_count and probability parameter')
    n = p['total_count'][0]
    if n != q['total_count'][0] or n.denominator != 1 or int(n) not in _COUNTS:
        raise ValueError('P and Q total_count must be identical and in {1,10,100,1000}')
    return (int(n), p[name][0], q[name][0])

def _one(p: Params, name: str) -> Params:
    return {name: p[name]}

class _Binomial:
    NAME: str
    BASE: Any

    @classmethod
    def domain_ok(cls, p: Params, q: Params) -> bool | str:
        try:
            _unpack(p, q, cls.NAME)
        except ValueError as exc:
            return str(exc)
        return cls.BASE.domain_ok(_one(p, cls.NAME), _one(q, cls.NAME))

    @classmethod
    def semantic_equal(cls, p: Params, q: Params) -> bool:
        _unpack(p, q, cls.NAME)
        return cls.BASE.semantic_equal(_one(p, cls.NAME), _one(q, cls.NAME))

    @classmethod
    def kl_arb(cls, p: Params, q: Params, prec: int) -> arb:
        n, _, _ = _unpack(p, q, cls.NAME)
        with arb_precision(prec):
            return n * cls.BASE.kl_arb(_one(p, cls.NAME), _one(q, cls.NAME), prec)

    @classmethod
    def grad_arb(cls, p: Params, q: Params, prec: int) -> dict[str, dict[str, list[arb]]]:
        n, _, _ = _unpack(p, q, cls.NAME)
        with arb_precision(prec):
            grads = cls.BASE.grad_arb(_one(p, cls.NAME), _one(q, cls.NAME), prec)
            return {side: {cls.NAME: [n * grads[side][cls.NAME][0]]} for side in ('P', 'Q')}

    @classmethod
    def exact_zero_mask(cls, p: Params, q: Params) -> dict[str, dict[str, list[bool]]]:
        return cls.BASE.exact_zero_mask(_one(p, cls.NAME), _one(q, cls.NAME))

    @classmethod
    def kl_mpmath_alt(cls, p: Params, q: Params, dps: int) -> Any:
        n, _, _ = _unpack(p, q, cls.NAME)
        return n * cls.BASE.kl_mpmath_alt(_one(p, cls.NAME), _one(q, cls.NAME), dps)

    @classmethod
    def kl_definition_check(cls, p: Params, q: Params, dps: int=100) -> Any:
        n, _, _ = _unpack(p, q, cls.NAME)
        with mp.workdps(dps):
            if cls.NAME == 'probs':
                a, b = (mpf(p[cls.NAME][0]), mpf(q[cls.NAME][0]))
            else:
                z, w = (mpf(p[cls.NAME][0]), mpf(q[cls.NAME][0]))
                a, b = (1 / (1 + mp.exp(-z)), 1 / (1 + mp.exp(-w)))
            a0, a1 = (mp.log(a / b), mp.log((1 - a) / (1 - b)))
            f = (1 - a) ** n
            terms = []
            for k in range(n + 1):
                terms.append(f * (k * a0 + (n - k) * a1))
                if k < n:
                    f *= mp.mpf(n - k) / (k + 1) * a / (1 - a)
            return mp.fsum(terms)

class Probs(_Binomial):
    NAME = 'probs'
    BASE = bernoulli.Probs
    PARAM_SPEC = {'total_count': {'shape': (1,), 'differentiable': False, 'domain': 'exact integer in {1,10,100,1000}'}, 'probs': {'shape': (1,), 'differentiable': True, 'domain': '0 < p < 1'}}

    @classmethod
    def exact_grad(cls, p: Params, q: Params):
        n, a, b = _unpack(p, q, 'probs')
        return {'P': {'probs': [None]}, 'Q': {'probs': [n * (b - a) / (b * (1 - b))]}}

class Logits(_Binomial):
    NAME = 'logits'
    BASE = bernoulli.Logits
    PARAM_SPEC = {'total_count': {'shape': (1,), 'differentiable': False, 'domain': 'exact integer in {1,10,100,1000}'}, 'logits': {'shape': (1,), 'differentiable': True, 'domain': 'finite real'}}
