from __future__ import annotations
from fractions import Fraction
from functools import lru_cache
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import arb_precision, fraction_to_arb
Params = dict[str, list[Fraction]]
SERIES_ETA = Fraction(1, 16)
SERIES_PROB_HALFWIDTH = Fraction(1, 128)

@lru_cache(maxsize=None)
def bernoulli_numbers(n_max: int) -> tuple[Fraction, ...]:
    out: list[Fraction] = []
    a = [Fraction(0)] * (n_max + 1)
    for m in range(n_max + 1):
        a[m] = Fraction(1, m + 1)
        for j in range(m, 0, -1):
            a[j - 1] = j * (a[j - 1] - a[j])
        out.append(a[0])
    return tuple(out)

@lru_cache(maxsize=None)
def _series_coefficients(n_terms: int) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    b = bernoulli_numbers(2 * n_terms)
    out = []
    fact = 1
    for k in range(1, n_terms + 1):
        fact *= (2 * k - 1) * (2 * k)
        c1 = b[2 * k] / fact
        out.append((c1 / (2 * k), c1, (2 * k - 1) * c1))
    return tuple(out)

def _n_terms(prec: int) -> int:
    return max(4, (prec + 80) // 13 + 1)

def _series(eta: arb, prec: int) -> tuple[arb, arb, arb]:
    n = _n_terms(prec)
    coeffs = _series_coefficients(n)
    e2 = eta * eta
    a0 = eta / 2
    a1 = arb(1) / 2
    a2 = arb(0)
    power = arb(1)
    for k, (ca, ca1, ca2) in enumerate(coeffs, start=1):
        a2 += fraction_to_arb_q(ca2) * power
        a1 += fraction_to_arb_q(ca1) * power * eta
        a0 += fraction_to_arb_q(ca) * power * e2
        power *= e2
    eta_up = abs(eta).upper()
    two_pi_low = (2 * arb.pi()).lower()
    y = (eta_up / two_pi_low).upper()
    if not y < arb(1) / 16:
        raise ValueError('ContinuousBernoulli series branch outside its radius')
    y2 = (y * y).upper()
    one = arb(1)
    t_a = (4 * y2 ** (n + 1) / ((2 * n + 2) * (one - y2).lower())).upper()
    t_a1 = (4 * y ** (2 * n + 1) / (two_pi_low * (one - y2).lower())).upper()
    t_a2 = (4 * (2 * n + 1) * y2 ** n / (two_pi_low * two_pi_low * (one - 3 * y2).lower())).upper()
    return (a0 + arb(0, t_a), a1 + arb(0, t_a1), a2 + arb(0, t_a2))

def fraction_to_arb_q(value: Fraction) -> arb:
    return fraction_to_arb(value)

def _direct(eta: arb) -> tuple[arb, arb, arb]:
    a0 = (eta.expm1() / eta).log()
    a1 = 1 / (1 - (-eta).exp()) - 1 / eta
    s = (eta / 2).sinh()
    a2 = 1 / (eta * eta) - 1 / (4 * s * s)
    return (a0, a1, a2)

def _a_funcs(eta: arb, use_series: bool, prec: int) -> tuple[arb, arb, arb]:
    return _series(eta, prec) if use_series else _direct(eta)

def _get_one(params: Params, name: str) -> Fraction:
    if set(params) != {name} or len(params[name]) != 1:
        raise ValueError(f'Expected one scalar {name!r}')
    return params[name][0]

def _mp(x: Fraction) -> Any:
    return mp.mpf(x.numerator) / mp.mpf(x.denominator)

def _kl_from_a(ap, aq, eta_gap_qp: arb) -> arb:
    return aq[0] - ap[0] - eta_gap_qp * ap[1]

class Probs:
    PARAM_SPEC = {'probs': {'shape': (1,), 'differentiable': True, 'domain': '0 < probs < 1'}}

    @staticmethod
    def domain_ok(p: Params, q: Params) -> bool | str:
        try:
            a, b = (_get_one(p, 'probs'), _get_one(q, 'probs'))
        except (KeyError, ValueError) as exc:
            return str(exc)
        return True if 0 < a < 1 and 0 < b < 1 else 'probs must lie strictly within (0,1)'

    @staticmethod
    def semantic_equal(p: Params, q: Params) -> bool:
        return _get_one(p, 'probs') == _get_one(q, 'probs')

    @staticmethod
    def _parts(p: Params, q: Params, prec: int):
        lp, lq = (_get_one(p, 'probs'), _get_one(q, 'probs'))
        half = Fraction(1, 2)
        eta_p = fraction_to_arb((2 * lp - 1) / (1 - lp)).log1p()
        eta_q = fraction_to_arb((2 * lq - 1) / (1 - lq)).log1p()
        gap_qp = fraction_to_arb((lq - lp) / (lp * (1 - lq))).log1p()
        ap = _a_funcs(eta_p, abs(lp - half) < SERIES_PROB_HALFWIDTH, prec)
        aq = _a_funcs(eta_q, abs(lq - half) < SERIES_PROB_HALFWIDTH, prec)
        return (lp, lq, gap_qp, ap, aq)

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            _, _, gap, ap, aq = Probs._parts(p, q, prec)
            return _kl_from_a(ap, aq, gap)

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int):
        with arb_precision(prec):
            lp, lq, gap, ap, aq = Probs._parts(p, q, prec)
            dp = fraction_to_arb(1 / (lp * (1 - lp)))
            dq = fraction_to_arb(1 / (lq * (1 - lq)))
            return {'P': {'probs': [-gap * ap[2] * dp]}, 'Q': {'probs': [(aq[1] - ap[1]) * dq]}}

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        same = _get_one(p, 'probs') == _get_one(q, 'probs')
        return {'P': {'probs': [same]}, 'Q': {'probs': [same]}}

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            lp, lq = (_get_one(p, 'probs'), _get_one(q, 'probs'))

            def log_c(lam: Fraction):
                if lam == Fraction(1, 2):
                    return mp.log(2)
                t = _mp(1 - 2 * lam)
                return mp.log(2 * mp.atanh(t) / t)

            def mean(lam: Fraction):
                if lam == Fraction(1, 2):
                    return mp.mpf(1) / 2
                t = _mp(1 - 2 * lam)
                return _mp(lam) / _mp(2 * lam - 1) + 1 / (2 * mp.atanh(t))
            x, y = (_mp(lp), _mp(lq))
            logit_gap = mp.log(x) - mp.log1p(-x) - mp.log(y) + mp.log1p(-y)
            return log_c(lp) - log_c(lq) + mp.log1p(-x) - mp.log1p(-y) + mean(lp) * logit_gap

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=60) -> Any:
        with mp.workdps(dps):
            lp, lq = (_get_one(p, 'probs'), _get_one(q, 'probs'))

            def logc(lam: Fraction):
                if lam == Fraction(1, 2):
                    return mp.log(2)
                t = _mp(1 - 2 * lam)
                return mp.log(2 * mp.atanh(t) / t)
            x, y = (_mp(lp), _mp(lq))
            cp, cq = (logc(lp), logc(lq))

            def logf(c, lam, t):
                return c + t * mp.log(lam) + (1 - t) * mp.log1p(-lam)

            def integrand(t):
                a = logf(cp, x, t)
                return mp.exp(a) * (a - logf(cq, y, t))
            return mp.quad(integrand, [0, mp.mpf(1) / 2, 1])

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
    def _parts(p: Params, q: Params, prec: int):
        zp, zq = (_get_one(p, 'logits'), _get_one(q, 'logits'))
        ap = _a_funcs(fraction_to_arb(zp), abs(zp) < SERIES_ETA, prec)
        aq = _a_funcs(fraction_to_arb(zq), abs(zq) < SERIES_ETA, prec)
        return (zp, zq, ap, aq)

    @staticmethod
    def kl_arb(p: Params, q: Params, prec: int) -> arb:
        with arb_precision(prec):
            zp, zq, ap, aq = Logits._parts(p, q, prec)
            return _kl_from_a(ap, aq, fraction_to_arb(zq - zp))

    @staticmethod
    def grad_arb(p: Params, q: Params, prec: int):
        with arb_precision(prec):
            zp, zq, ap, aq = Logits._parts(p, q, prec)
            return {'P': {'logits': [fraction_to_arb(zp - zq) * ap[2]]}, 'Q': {'logits': [aq[1] - ap[1]]}}

    @staticmethod
    def exact_zero_mask(p: Params, q: Params):
        same = _get_one(p, 'logits') == _get_one(q, 'logits')
        return {'P': {'logits': [same]}, 'Q': {'logits': [same]}}

    @staticmethod
    def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
        with mp.workdps(dps):
            zp, zq = (_get_one(p, 'logits'), _get_one(q, 'logits'))

            def log_c(z: Fraction):
                if z == 0:
                    return mp.log(2)
                x = _mp(z)
                return mp.log(x / mp.tanh(x / 2))

            def mean(z: Fraction):
                if z == 0:
                    return mp.mpf(1) / 2
                x = _mp(z)
                th = mp.tanh(x / 2)
                return (1 + th) / (2 * th) - 1 / x

            def log1m_sigmoid(z: Fraction):
                x = _mp(z)
                return -mp.log1p(mp.exp(x)) if x < 0 else -x - mp.log1p(mp.exp(-x))
            return log_c(zp) - log_c(zq) + log1m_sigmoid(zp) - log1m_sigmoid(zq) + mean(zp) * _mp(zp - zq)

    @staticmethod
    def kl_definition_check(p: Params, q: Params, dps: int=60) -> Any:
        with mp.workdps(dps):
            zp, zq = (_mp(_get_one(p, 'logits')), _mp(_get_one(q, 'logits')))

            def a(z):
                return mp.mpf(0) if z == 0 else mp.log(mp.expm1(z) / z)
            ap, aq = (a(zp), a(zq))

            def integrand(t):
                lp = zp * t - ap
                return mp.exp(lp) * (lp - (zq * t - aq))
            return mp.quad(integrand, [0, mp.mpf(1) / 2, 1])
