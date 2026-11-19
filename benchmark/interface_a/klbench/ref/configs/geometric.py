from __future__ import annotations
from fractions import Fraction
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import arb_precision, fraction_to_arb
from .common import mpf, rational_grads
from . import bernoulli
Params = dict[str, list[Fraction]]

def _one(p: Params, name: str) -> Fraction:
    if set(p) != {name} or len(p[name]) != 1:
        raise ValueError(f'Geometric expects scalar {name}')
    return p[name][0]

def _softplus(z: arb, exact: Fraction) -> arb:
    return z + (-z).exp().log1p() if exact >= 0 else z.exp().log1p()
DEFINITION_TAIL_TARGET = '1e-30'
DEFINITION_MAX_TERMS = 1000000

def _definition(prob_p: Any, prob_q: Any, count: int | None=None):
    p, q = (prob_p, prob_q)
    a, b = (1 - p, 1 - q)
    a0, a1 = (mp.log(p / q), mp.log(a / b))
    if count is None:

        def bound(n):
            return a ** n * (abs(a0) + abs(a1) * (n + a / p))
        high = 1
        while high <= DEFINITION_MAX_TERMS and bound(high) > mp.mpf(DEFINITION_TAIL_TARGET):
            high *= 2
        if high > DEFINITION_MAX_TERMS:
            if bound(DEFINITION_MAX_TERMS) > mp.mpf(DEFINITION_TAIL_TARGET):
                return (None, None, 'SUM_UNVERIFIABLE')
            high = DEFINITION_MAX_TERMS
        low = max(0, high // 2)
        while low + 1 < high:
            mid = (low + high) // 2
            if bound(mid) <= mp.mpf(DEFINITION_TAIL_TARGET):
                high = mid
            else:
                low = mid
        count = high
    weight = p
    terms = []
    for k in range(count):
        terms.append(weight * (a0 + k * a1))
        weight *= a
    tail_bound = a ** count * (abs(a0) + abs(a1) * (count + a / p))
    return (mp.fsum(terms), tail_bound, 'VERIFIED')

class Probs:
    PARAM_SPEC = {'probs': {'shape': (1,), 'differentiable': True, 'domain': '0 < p < 1'}}

    @staticmethod
    def domain_ok(p: Params, q: Params) -> bool | str:
        try:
            a, b = (_one(p, 'probs'), _one(q, 'probs'))
        except ValueError as exc:
            return str(exc)
        return True if 0 < a < 1 and 0 < b < 1 else '0 < p,q < 1 required'

    @staticmethod
    def semantic_equal(p: Params, q: Params) -> bool:
        return _one(p, 'probs') == _one(q, 'probs')

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            a, b = (_one(p, 'probs'), _one(q, 'probs'))
            x, y = (fraction_to_arb(a), fraction_to_arb(b))
            delta = x - y
            return (delta / y).log1p() + (1 - x) / x * (-delta / (1 - y)).log1p()

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int) -> dict[str, dict[str, list[arb]]]:
        with arb_precision(prec):
            a, b = (_one(p, 'probs'), _one(q, 'probs'))
            x, y = (fraction_to_arb(a), fraction_to_arb(b))
            logtail = ((y - x) / (1 - y)).log1p()
            return {'P': {'probs': [-logtail / (x * x)]}, 'Q': {'probs': [-1 / y + (1 - x) / (x * (1 - y))]}}

    @staticmethod
    def exact_grad(p: Params, q: Params):
        a, b = (_one(p, 'probs'), _one(q, 'probs'))
        return rational_grads(('probs',), (None,), (-1 / b + (1 - a) / (a * (1 - b)),))

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        same = Probs.semantic_equal(p, q)
        return {'P': {'probs': [same]}, 'Q': {'probs': [same]}}

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            a, b = (mpf(_one(p, 'probs')), mpf(_one(q, 'probs')))
            return mp.log(a / b) + (1 - a) / a * mp.log((1 - a) / (1 - b))

    @staticmethod
    def definition_tail_bound(p: Params, q: Params, dps: int=100, count: int | None=None):
        with mp.workdps(dps):
            a, b = (mpf(_one(p, 'probs')), mpf(_one(q, 'probs')))
            return _definition(a, b, count)[1]

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            a, b = (mpf(_one(p, 'probs')), mpf(_one(q, 'probs')))
            s, bound, status = _definition(a, b)
            if status == 'SUM_UNVERIFIABLE':
                raise DefinitionSumUnavailable('Geometric sum requires >1,000,000 terms')
            return s

class Logits:
    PARAM_SPEC = {'logits': {'shape': (1,), 'differentiable': True, 'domain': 'finite real'}}

    @staticmethod
    def domain_ok(p: Params, q: Params) -> bool | str:
        try:
            _one(p, 'logits')
            _one(q, 'logits')
        except ValueError as exc:
            return str(exc)
        return True

    @staticmethod
    def semantic_equal(p: Params, q: Params) -> bool:
        return _one(p, 'logits') == _one(q, 'logits')

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            z, w = (_one(p, 'logits'), _one(q, 'logits'))
            x, y = (fraction_to_arb(z), fraction_to_arb(w))
            return x - y + (1 + (-x).exp()) * (_softplus(y, w) - _softplus(x, z))

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int) -> dict[str, dict[str, list[arb]]]:
        with arb_precision(prec):
            z, w = (_one(p, 'logits'), _one(q, 'logits'))
            x, y = (fraction_to_arb(z), fraction_to_arb(w))
            p1 = 1 / (1 + (-x).exp())
            q1 = 1 / (1 + (-y).exp())
            return {'P': {'logits': [-(-x).exp() * (_softplus(y, w) - _softplus(x, z))]}, 'Q': {'logits': [q1 / p1 - 1]}}

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        same = Logits.semantic_equal(p, q)
        return {'P': {'logits': [same]}, 'Q': {'logits': [same]}}

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            z, w = (mpf(_one(p, 'logits')), mpf(_one(q, 'logits')))
            a, b = (1 / (1 + mp.exp(-z)), 1 / (1 + mp.exp(-w)))
            return mp.log(a / b) + (1 - a) / a * mp.log((1 - a) / (1 - b))

    @staticmethod
    def definition_tail_bound(p: Params, q: Params, dps: int=100, count: int | None=None):
        with mp.workdps(dps):
            z, w = (mpf(_one(p, 'logits')), mpf(_one(q, 'logits')))
            return _definition(1 / (1 + mp.exp(-z)), 1 / (1 + mp.exp(-w)), count)[1]

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            z, w = (mpf(_one(p, 'logits')), mpf(_one(q, 'logits')))
            value, _, status = _definition(1 / (1 + mp.exp(-z)), 1 / (1 + mp.exp(-w)))
            if status == 'SUM_UNVERIFIABLE':
                raise DefinitionSumUnavailable('Geometric sum requires >1,000,000 terms')
            return value

class DefinitionSumUnavailable(ValueError):
    pass
