from __future__ import annotations
from fractions import Fraction
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import arb_precision, fraction_to_arb
Params = dict[str, list[Fraction]]
PARAMS = ('loc', 'scale')

def _unpack(p: Params, q: Params):
    for v in (p, q):
        if set(v) != set(PARAMS) or any((len(v[k]) != 1 for k in PARAMS)):
            raise ValueError('Normal expects scalar loc and scale')
    return (p['loc'][0], p['scale'][0], q['loc'][0], q['scale'][0])

def _mp_frac(x: Fraction) -> Any:
    return mp.mpf(x.numerator) / mp.mpf(x.denominator)

class Normal:
    PARAM_SPEC = {'loc': {'shape': (1,), 'differentiable': True, 'domain': 'finite real'}, 'scale': {'shape': (1,), 'differentiable': True, 'domain': 'scale > 0'}}

    @staticmethod
    def domain_ok(p: Params, q: Params) -> bool | str:
        try:
            _, sp, _, sq = _unpack(p, q)
        except ValueError as exc:
            return str(exc)
        return True if sp > 0 and sq > 0 else 'Both Normal scales must be positive'

    @staticmethod
    def semantic_equal(p: Params, q: Params) -> bool:
        mp_, sp, mq, sq = _unpack(p, q)
        return mp_ == mq and sp == sq

    @staticmethod
    def exact_kl(p: Params, q: Params) -> Fraction | None:
        mp_, sp, mq, sq = _unpack(p, q)
        if sp == sq:
            return (mp_ - mq) ** 2 / (2 * sq ** 2)
        return None

    @staticmethod
    def exact_grad(p: Params, q: Params) -> dict[str, dict[str, list[Fraction]]]:
        m1, s1, m2, s2 = _unpack(p, q)
        d = m1 - m2
        return {'P': {'loc': [d / s2 ** 2], 'scale': [s1 / s2 ** 2 - 1 / s1]}, 'Q': {'loc': [-d / s2 ** 2], 'scale': [1 / s2 - (s1 ** 2 + d ** 2) / s2 ** 3]}}

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            mp_, sp, mq, sq = (fraction_to_arb(v) for v in _unpack(p, q))
            ratio_minus_one = (sp - sq) / sq
            shifted_loc = (mp_ - mq) / sq
            return shifted_loc * shifted_loc / 2 + ratio_minus_one + ratio_minus_one * ratio_minus_one / 2 - ratio_minus_one.log1p()

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int) -> dict[str, dict[str, list[arb]]]:
        with arb_precision(prec):
            mp_, sp, mq, sq = (fraction_to_arb(v) for v in _unpack(p, q))
            delta = mp_ - mq
            return {'P': {'loc': [delta / (sq * sq)], 'scale': [sp / (sq * sq) - 1 / sp]}, 'Q': {'loc': [-delta / (sq * sq)], 'scale': [1 / sq - (sp * sp + delta * delta) / (sq * sq * sq)]}}

    @staticmethod
    def exact_zero_mask(p: Params, q: Params) -> dict[str, dict[str, list[bool]]]:
        mp_, sp, mq, sq = _unpack(p, q)
        zero_loc = mp_ == mq
        return {'P': {'loc': [zero_loc], 'scale': [sp * sp == sq * sq]}, 'Q': {'loc': [zero_loc], 'scale': [sq * sq == sp * sp + (mp_ - mq) * (mp_ - mq)]}}

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            m1, s1, m2, s2 = (_mp_frac(v) for v in _unpack(p, q))
            return mp.log(s2 / s1) + (s1 * s1 + (m1 - m2) ** 2) / (2 * s2 * s2) - mp.mpf(1) / 2

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            m1, s1, m2, s2 = (_mp_frac(v) for v in _unpack(p, q))
            norm = mp.sqrt(2 * mp.pi)

            def pdf(x):
                return mp.exp(-((x - m1) / s1) ** 2 / 2) / (s1 * norm)

            def log_pdf_ratio(x):
                return mp.log(s2 / s1) - (x - m1) ** 2 / (2 * s1 * s1) + (x - m2) ** 2 / (2 * s2 * s2)
            return mp.quad(lambda x: pdf(x) * log_pdf_ratio(x), [-mp.inf, m1 - 12 * s1, m1, m1 + 12 * s1, mp.inf])
