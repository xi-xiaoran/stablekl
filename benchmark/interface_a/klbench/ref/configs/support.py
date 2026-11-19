from __future__ import annotations
from fractions import Fraction
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import arb_precision, fraction_to_arb
from .common import unpack, arbs, mps, masks, spec, rational_grads, rational_power_equal
Params = dict[str, list[Fraction]]
_LIMITS = ('low', 'high')
_PARETO = ('scale', 'alpha')

class Uniform:
    PARAM_SPEC = spec(_LIMITS)

    @staticmethod
    def domain_ok(p: Params, q: Params) -> bool | str:
        try:
            lp, hp, lq, hq = unpack(p, q, _LIMITS)
        except ValueError as exc:
            return str(exc)
        if not (lp < hp and lq < hq):
            return 'Each Uniform interval must have strictly positive width'
        return True if lq <= lp and hp <= hq else 'Uniform Q support must contain P support'

    @staticmethod
    def semantic_equal(p: Params, q: Params) -> bool:
        lp, hp, lq, hq = unpack(p, q, _LIMITS)
        return lp == lq and hp == hq

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            lp, hp, lq, hq = unpack(p, q, _LIMITS)
            wp, wq = (hp - lp, hq - lq)
            if wp == wq:
                return arb(0)
            return (fraction_to_arb(wq - wp) / fraction_to_arb(wp)).log1p()

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int):
        with arb_precision(prec):
            lp, hp, lq, hq = unpack(p, q, _LIMITS)
            wp, wq = (fraction_to_arb(hp - lp), fraction_to_arb(hq - lq))
            return {'P': {'low': [1 / wp], 'high': [-1 / wp]}, 'Q': {'low': [-1 / wq], 'high': [1 / wq]}}

    @staticmethod
    def exact_grad(p: Params, q: Params):
        lp, hp, lq, hq = unpack(p, q, _LIMITS)
        wp, wq = (hp - lp, hq - lq)
        return rational_grads(_LIMITS, (1 / wp, -1 / wp), (-1 / wq, 1 / wq))

    @staticmethod
    def nondiff_mask(p: Params, q: Params):
        lp, hp, lq, hq = unpack(p, q, _LIMITS)
        return masks(_LIMITS, (lp == lq, hp == hq), (lp == lq, hp == hq))

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        return masks(_LIMITS, (False, False), (False, False))

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            lp, hp, lq, hq = mps(unpack(p, q, _LIMITS))
            return mp.log((hq - lq) / (hp - lp))

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            lp, hp, lq, hq = mps(unpack(p, q, _LIMITS))
            return mp.quad(lambda t: mp.log((hq - lq) / (hp - lp)), [0, mp.mpf(1) / 2, 1])

class Pareto:
    PARAM_SPEC = spec(_PARETO, _PARETO)

    @staticmethod
    def domain_ok(p: Params, q: Params) -> bool | str:
        try:
            sp, ap, sq, aq = unpack(p, q, _PARETO)
        except ValueError as exc:
            return str(exc)
        if min(sp, ap, sq, aq) <= 0:
            return 'Pareto scale and alpha must be strictly positive'
        return True if sp >= sq else 'Pareto P scale must be >= Q scale'

    @staticmethod
    def semantic_equal(p: Params, q: Params) -> bool:
        sp, ap, sq, aq = unpack(p, q, _PARETO)
        return sp == sq and ap == aq

    @staticmethod
    def exact_kl(p: Params, q: Params) -> Fraction | None:
        sp, ap, sq, aq = unpack(p, q, _PARETO)
        if rational_power_equal(sq / sp, aq, ap / aq):
            return aq / ap - 1
        return None

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            sp, ap, sq, aq = unpack(p, q, _PARETO)
            xp, yp, xq, yq = arbs((sp, ap, sq, aq))
            return ((yp - yq) / yq).log1p() + yq * ((xp - xq) / xq).log1p() + (yq - yp) / yp

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int):
        with arb_precision(prec):
            sp, ap, sq, aq = unpack(p, q, _PARETO)
            xp, yp, xq, yq = arbs((sp, ap, sq, aq))
            return {'P': {'scale': [yq / xp], 'alpha': [1 / yp - yq / (yp * yp)]}, 'Q': {'scale': [-yq / xq], 'alpha': [-1 / yq + ((xp - xq) / xq).log1p() + 1 / yp]}}

    @staticmethod
    def exact_grad(p: Params, q: Params):
        sp, ap, sq, aq = unpack(p, q, _PARETO)
        return rational_grads(_PARETO, (aq / sp, 1 / ap - aq / ap ** 2), (-aq / sq, 1 / ap - 1 / aq if sp == sq else None))

    @staticmethod
    def nondiff_mask(p: Params, q: Params):
        sp, ap, sq, aq = unpack(p, q, _PARETO)
        return masks(_PARETO, (sp == sq, False), (sp == sq, False))

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        sp, ap, sq, aq = unpack(p, q, _PARETO)
        return masks(_PARETO, (False, ap == aq), (False, sp == sq and ap == aq))

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            sp, ap, sq, aq = mps(unpack(p, q, _PARETO))
            return mp.log(ap / aq) + aq * mp.log(sp / sq) + aq / ap - 1

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=100) -> Any:
        with mp.workdps(dps):
            sp, ap, sq, aq = mps(unpack(p, q, _PARETO))
            const = mp.log(ap / aq) + aq * mp.log(sp / sq)
            return mp.quad(lambda t: ap * mp.exp(-ap * t) * (const + (aq - ap) * t), [0, 1 / ap, 16 / ap, mp.inf])
