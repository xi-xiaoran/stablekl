from __future__ import annotations
from fractions import Fraction
from .formats import FloatFormat, get_format

def _nearest_even(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError('Expected positive ratio')
    integer, remainder = divmod(numerator, denominator)
    double_remainder = remainder * 2
    return integer + int(double_remainder > denominator or (double_remainder == denominator and bool(integer & 1)))

def _scale_round(number: Fraction, exponent: int) -> int:
    if exponent >= 0:
        return _nearest_even(number.numerator << exponent, number.denominator)
    return _nearest_even(number.numerator, number.denominator << -exponent)

def _floor_log2(number: Fraction) -> int:
    k = number.numerator.bit_length() - number.denominator.bit_length()
    if k >= 0 and number.numerator < number.denominator << k:
        k -= 1
    elif k < 0 and number.numerator << -k < number.denominator:
        k -= 1
    return k

def round_fraction(x: Fraction | int, fmt: str | FloatFormat, sign_hint: int | None=None) -> int:
    spec = get_format(fmt)
    x = Fraction(x)
    if sign_hint not in (None, -1, 1):
        raise ValueError('sign_hint must be None, -1, or +1')
    sign = spec.sign_mask if x < 0 or (x == 0 and sign_hint == -1) else 0
    if x == 0:
        return sign
    absolute = abs(x)
    if absolute >= spec.overflow_threshold:
        return sign | spec.infinity_bits
    exponent = _floor_log2(absolute)
    if exponent < spec.emin:
        significand = _scale_round(absolute, spec.fraction_bits - spec.emin)
        if significand == 0:
            return sign
        if significand >= 1 << spec.fraction_bits:
            return sign | 1 << spec.fraction_bits
        return sign | significand
    significand = _scale_round(absolute, spec.fraction_bits - exponent)
    if significand == 1 << spec.precision:
        exponent += 1
        significand >>= 1
    if exponent > spec.emax:
        return sign | spec.infinity_bits
    return sign | exponent + spec.bias << spec.fraction_bits | significand - (1 << spec.fraction_bits)
