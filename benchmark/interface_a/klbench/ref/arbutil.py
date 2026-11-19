from __future__ import annotations
from contextlib import contextmanager
from fractions import Fraction
from typing import Iterator
from flint import arb, ctx

@contextmanager
def arb_precision(bits: int) -> Iterator[None]:
    previous = ctx.prec
    try:
        ctx.prec = bits
        yield
    finally:
        ctx.prec = previous

def fraction_to_arb(value: Fraction) -> arb:
    value = Fraction(value)
    denom = value.denominator
    if not denom & denom - 1:
        result = arb((value.numerator, -(denom.bit_length() - 1)))
        if not result.is_exact():
            raise ValueError('The dyadic input exceeds the active Arb precision')
        return result
    return arb(value.numerator) / arb(denom)

def exact_arb_to_fraction(number: arb) -> Fraction:
    if not number.is_finite() or not number.is_exact():
        raise ValueError('Arb endpoint must be finite and exact')
    mantissa, exponent = number.man_exp()
    m, e = (int(mantissa), int(exponent))
    return Fraction(m << e) if e >= 0 else Fraction(m, 1 << -e)

def arb_bounds(number: arb) -> tuple[Fraction, Fraction]:
    if not number.is_finite():
        raise ValueError('Nonfinite ball: handle MATH_INF before endpoint extraction')
    lower = exact_arb_to_fraction(number.lower())
    upper = exact_arb_to_fraction(number.upper())
    if lower > upper:
        raise AssertionError('Invalid Arb enclosure')
    return (lower, upper)
