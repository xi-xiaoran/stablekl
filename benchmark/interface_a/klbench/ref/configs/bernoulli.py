from __future__ import annotations
from fractions import Fraction
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import arb_precision, fraction_to_arb
Params = dict[str, list[Fraction]]

def _get_one(params: Params, name: str) -> Fraction:
    if set(params) != {name} or len(params[name]) != 1:
        raise ValueError(f'Expected one scalar {name!r}')
    return params[name][0]

def _mp_frac(x: Fraction) -> Any:
    return mp.mpf(x.numerator) / mp.mpf(x.denominator)

def _softplus_arb(z: arb, sign: int) -> arb:
    return z + (-z).exp().log1p() if sign >= 0 else z.exp().log1p()

class Probs:
    PARAM_SPEC = {'probs': {'shape': (1,), 'differentiable': True, 'domain': '0 < probs < 1'}}

    @staticmethod
    def domain_ok(p: Params, q: Params) -> bool | str:
        try:
            a, b = (_get_one(p, 'probs'), _get_one(q, 'probs'))
        except (KeyError, ValueError) as exc:
            return str(exc)
        return True if 0 < a < 1 and 0 < b < 1 else 'Probability must lie strictly within (0,1)'

    @staticmethod
    def semantic_equal(p: Params, q: Params) -> bool:
        return _get_one(p, 'probs') == _get_one(q, 'probs')

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            x, y = (fraction_to_arb(_get_one(p, 'probs')), fraction_to_arb(_get_one(q, 'probs')))
            delta = x - y
            return x * (delta / y).log1p() + (1 - x) * (-delta / (1 - y)).log1p()

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int) -> dict[str, dict[str, list[arb]]]:
        with arb_precision(prec):
            x, y = (fraction_to_arb(_get_one(p, 'probs')), fraction_to_arb(_get_one(q, 'probs')))
            gap = x - y
            gp = (gap / y).log1p() - (-gap / (1 - y)).log1p()
            gq = -gap / (y * (1 - y))
            return {'P': {'probs': [gp]}, 'Q': {'probs': [gq]}}

    @staticmethod
    def exact_zero_mask(p: Params, q: Params) -> dict[str, dict[str, list[bool]]]:
        same = _get_one(p, 'probs') == _get_one(q, 'probs')
        return {'P': {'probs': [same]}, 'Q': {'probs': [same]}}

    @staticmethod
    def exact_grad(p: Params, q: Params):
        a, b = (_get_one(p, 'probs'), _get_one(q, 'probs'))
        return {'P': {'probs': [None]}, 'Q': {'probs': [(b - a) / (b * (1 - b))]}}

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            x, y = (_mp_frac(_get_one(p, 'probs')), _mp_frac(_get_one(q, 'probs')))
            return x * mp.log(x / y) + (1 - x) * mp.log((1 - x) / (1 - y))

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            x, y = (_mp_frac(_get_one(p, 'probs')), _mp_frac(_get_one(q, 'probs')))
            return mp.fsum([x * (mp.log(x) - mp.log(y)), (1 - x) * (mp.log(1 - x) - mp.log(1 - y))])

class Logits:
    PARAM_SPEC = {'logits': {'shape': (1,), 'differentiable': True, 'domain': 'finite real logits'}}

    @staticmethod
    def domain_ok(p: Params, q: Params) -> bool | str:
        try:
            _get_one(p, 'logits')
            _get_one(q, 'logits')
        except (KeyError, ValueError) as exc:
            return str(exc)
        return True

    @staticmethod
    def semantic_equal(p: Params, q: Params) -> bool:
        return _get_one(p, 'logits') == _get_one(q, 'logits')

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            u, v = (_get_one(p, 'logits'), _get_one(q, 'logits'))
            z, w = (fraction_to_arb(u), fraction_to_arb(v))
            prob = 1 / (1 + (-z).exp())
            return _softplus_arb(w, 1 if v >= 0 else -1) - _softplus_arb(z, 1 if u >= 0 else -1) + prob * (z - w)

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int) -> dict[str, dict[str, list[arb]]]:
        with arb_precision(prec):
            z, w = (fraction_to_arb(_get_one(p, 'logits')), fraction_to_arb(_get_one(q, 'logits')))
            prob, other = (1 / (1 + (-z).exp()), 1 / (1 + (-w).exp()))
            return {'P': {'logits': [prob * (1 - prob) * (z - w)]}, 'Q': {'logits': [other - prob]}}

    @staticmethod
    def exact_zero_mask(p: Params, q: Params) -> dict[str, dict[str, list[bool]]]:
        same = _get_one(p, 'logits') == _get_one(q, 'logits')
        return {'P': {'logits': [same]}, 'Q': {'logits': [same]}}

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            z, w = (_mp_frac(_get_one(p, 'logits')), _mp_frac(_get_one(q, 'logits')))
            log_p, log_q = (-mp.log(1 + mp.exp(-z)), -mp.log(1 + mp.exp(-w)))
            log_1mp, log_1mq = (-mp.log(1 + mp.exp(z)), -mp.log(1 + mp.exp(w)))
            return mp.exp(log_p) * (log_p - log_q) + mp.exp(log_1mp) * (log_1mp - log_1mq)

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            z, w = (_mp_frac(_get_one(p, 'logits')), _mp_frac(_get_one(q, 'logits')))
            lp, lq = (-mp.log(1 + mp.exp(-z)), -mp.log(1 + mp.exp(-w)))
            lm_p, lm_q = (-mp.log(1 + mp.exp(z)), -mp.log(1 + mp.exp(w)))
            return mp.fsum([mp.exp(lp) * (lp - lq), mp.exp(lm_p) * (lm_p - lm_q)])
