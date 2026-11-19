from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from .formats import FloatFormat, get_format

@dataclass(frozen=True)
class Decoded:
    kind: str
    sign: int
    value: Fraction | None = None
    payload: int | None = None

    @property
    def is_zero(self) -> bool:
        return self.kind == 'finite' and self.value == 0

def bits_to_fraction(bits: int, fmt: str | FloatFormat) -> Decoded:
    spec = get_format(fmt)
    if not isinstance(bits, int) or not 0 <= bits < 1 << spec.bits:
        raise ValueError(f'{bits!r} is not a {spec.bits}-bit unsigned integer')
    sign = -1 if bits & spec.sign_mask else 1
    fractional = bits & spec.fraction_mask
    exp_field = bits >> spec.fraction_bits & (1 << spec.exponent_bits) - 1
    if exp_field == (1 << spec.exponent_bits) - 1:
        if fractional:
            return Decoded('nan', sign, payload=fractional)
        return Decoded('infinity', sign)
    if exp_field == 0:
        mantissa = fractional
        exponent = spec.emin - spec.fraction_bits
    else:
        mantissa = 1 << spec.fraction_bits | fractional
        exponent = exp_field - spec.bias - spec.fraction_bits
    return Decoded('finite', sign, Fraction(sign * mantissa) * spec.power_of_two(exponent))

def exact_value(bits: int, fmt: str | FloatFormat) -> Fraction:
    decoded = bits_to_fraction(bits, fmt)
    if decoded.kind != 'finite':
        raise ValueError('Mathematical input must be finite')
    assert decoded.value is not None
    return decoded.value

def fraction_to_bits(decoded: Decoded, fmt: str | FloatFormat) -> int:
    from .rounding import round_fraction
    spec = get_format(fmt)
    sign = spec.sign_mask if decoded.sign == -1 else 0
    if decoded.kind == 'infinity':
        return sign | spec.infinity_bits
    if decoded.kind == 'nan':
        if decoded.payload is None or not 0 < decoded.payload <= spec.fraction_mask:
            raise ValueError('Missing/invalid NaN payload')
        return sign | spec.infinity_bits | decoded.payload
    if decoded.kind != 'finite' or decoded.value is None:
        raise ValueError('Invalid tagged float value')
    return round_fraction(decoded.value, spec, sign_hint=decoded.sign)
