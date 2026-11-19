from __future__ import annotations
from fractions import Fraction
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import arb_precision, fraction_to_arb
Params = dict[str, list[Fraction]]

def _vec(params: Params, name: str, k: int) -> list[Fraction]:
    if set(params) != {name} or len(params[name]) != k:
        raise ValueError(f'Expected {name!r} of length {k}')
    return list(params[name])

def _mp(x: Fraction) -> Any:
    return mp.mpf(x.numerator) / mp.mpf(x.denominator)

def make_probs(k: int):

    class Probs:
        K = k
        PARAM_SPEC = {'probs': {'shape': (k,), 'differentiable': True, 'domain': 'positive raw (unnormalized) vector'}}

        @staticmethod
        def domain_ok(p: Params, q: Params) -> bool | str:
            try:
                x, y = (_vec(p, 'probs', k), _vec(q, 'probs', k))
            except (KeyError, ValueError) as exc:
                return str(exc)
            if any((v <= 0 for v in x + y)):
                return 'raw probs must be strictly positive'
            return True

        @staticmethod
        def semantic_equal(p: Params, q: Params) -> bool:
            x, y = (_vec(p, 'probs', k), _vec(q, 'probs', k))
            sx, sy = (sum(x), sum(y))
            return all((a * sy == b * sx for a, b in zip(x, y)))

        @staticmethod
        def _rational(p: Params, q: Params):
            x, y = (_vec(p, 'probs', k), _vec(q, 'probs', k))
            sx, sy = (sum(x), sum(y))
            pr = [a / sx for a in x]
            r = [(a * sy - b * sx) / (b * sx) for a, b in zip(x, y)]
            return (sx, sy, pr, r)

        @staticmethod
        def kl_arb(p: Params, q: Params, prec: int) -> arb:
            with arb_precision(prec):
                _, _, pr, r = Probs._rational(p, q)
                return sum((fraction_to_arb(pi) * fraction_to_arb(ri).log1p() for pi, ri in zip(pr, r)), arb(0))

        @staticmethod
        def grad_arb(p: Params, q: Params, prec: int):
            with arb_precision(prec):
                sx, sy, pr, r = Probs._rational(p, q)
                logs = [fraction_to_arb(ri).log1p() for ri in r]
                kl = sum((fraction_to_arb(pi) * li for pi, li in zip(pr, logs)), arb(0))
                inv_sx = fraction_to_arb(1 / sx)
                return {'P': {'probs': [(li - kl) * inv_sx for li in logs]}, 'Q': {'probs': [fraction_to_arb(-ri / sy) for ri in r]}}

        @staticmethod
        def exact_grad(p: Params, q: Params):
            _, sy, _, r = Probs._rational(p, q)
            return {'P': {'probs': [None] * k}, 'Q': {'probs': [-ri / sy for ri in r]}}

        @staticmethod
        def exact_zero_mask(p: Params, q: Params):
            same = Probs.semantic_equal(p, q)
            _, _, _, r = Probs._rational(p, q)
            return {'P': {'probs': [same] * k}, 'Q': {'probs': [same or ri == 0 for ri in r]}}

        @staticmethod
        def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
            with mp.workdps(dps):
                x = [_mp(v) for v in _vec(p, 'probs', k)]
                y = [_mp(v) for v in _vec(q, 'probs', k)]
                sx, sy = (mp.fsum(x), mp.fsum(y))
                return mp.fsum((a / sx * (mp.log(a / sx) - mp.log(b / sy)) for a, b in zip(x, y)))

        @staticmethod
        def kl_definition_check(p: Params, q: Params, dps: int=60) -> Any:
            with mp.workdps(dps):
                x = _vec(p, 'probs', k)
                y = _vec(q, 'probs', k)
                sx, sy = (sum(x), sum(y))
                return mp.fsum((_mp(a / sx) * mp.log(_mp(a / sx / (b / sy))) for a, b in zip(x, y)))
    Probs.__name__ = Probs.__qualname__ = f'CategoricalProbsK{k}'
    return Probs

def make_logits(k: int):

    class Logits:
        K = k
        PARAM_SPEC = {'logits': {'shape': (k,), 'differentiable': True, 'domain': 'finite real logits'}}

        @staticmethod
        def domain_ok(p: Params, q: Params) -> bool | str:
            try:
                _vec(p, 'logits', k)
                _vec(q, 'logits', k)
            except (KeyError, ValueError) as exc:
                return str(exc)
            return True

        @staticmethod
        def semantic_equal(p: Params, q: Params) -> bool:
            z, w = (_vec(p, 'logits', k), _vec(q, 'logits', k))
            c = [a - b for a, b in zip(z, w)]
            return all((v == c[0] for v in c))

        @staticmethod
        def _parts(p: Params, q: Params):
            z, w = (_vec(p, 'logits', k), _vec(q, 'logits', k))
            zmax, wmax = (max(z), max(w))
            ez = [fraction_to_arb(a - zmax).exp() for a in z]
            ew = [fraction_to_arb(b - wmax).exp() for b in w]
            sz, sw = (sum(ez, arb(0)), sum(ew, arb(0)))
            pr = [e / sz for e in ez]
            qr = [e / sw for e in ew]
            kref = z.index(zmax)
            cref = z[kref] - w[kref]
            d = [a - b - cref for a, b in zip(z, w)]
            big_l = sum((qi * fraction_to_arb(di).expm1() for qi, di in zip(qr, d)), arb(0)).log1p()
            kl = sum((pi * fraction_to_arb(di) for pi, di in zip(pr, d)), arb(0)) - big_l
            return (pr, qr, d, big_l, kl)

        @staticmethod
        def kl_arb(p: Params, q: Params, prec: int) -> arb:
            with arb_precision(prec):
                return Logits._parts(p, q)[4]

        @staticmethod
        def grad_arb(p: Params, q: Params, prec: int):
            with arb_precision(prec):
                pr, qr, d, big_l, kl = Logits._parts(p, q)
                return {'P': {'logits': [pi * (fraction_to_arb(di) - big_l - kl) for pi, di in zip(pr, d)]}, 'Q': {'logits': [qi - pi for pi, qi in zip(pr, qr)]}}

        @staticmethod
        def exact_zero_mask(p: Params, q: Params):
            same = Logits.semantic_equal(p, q)
            return {'P': {'logits': [same] * k}, 'Q': {'logits': [same] * k}}

        @staticmethod
        def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
            with mp.workdps(dps):
                z = [_mp(v) for v in _vec(p, 'logits', k)]
                w = [_mp(v) for v in _vec(q, 'logits', k)]

                def lse(v):
                    m = max(v)
                    return m + mp.log(mp.fsum((mp.exp(a - m) for a in v)))
                lz, lw = (lse(z), lse(w))
                return mp.fsum((mp.exp(a - lz) * (a - lz - (b - lw)) for a, b in zip(z, w)))

        @staticmethod
        def kl_definition_check(p: Params, q: Params, dps: int=60) -> Any:
            return Logits.kl_mpmath_alt(p, q, dps)
    Logits.__name__ = Logits.__qualname__ = f'CategoricalLogitsK{k}'
    return Logits
