from __future__ import annotations
from .formats import FloatFormat, get_format
from .bits import bits_to_fraction

def ord(bits: int, fmt: str | FloatFormat) -> int:
    spec = get_format(fmt)
    decoded = bits_to_fraction(bits, spec)
    if decoded.kind == 'nan':
        raise ValueError('NaN cannot be ordinally ordered')
    magnitude = bits & spec.sign_mask - 1
    return -magnitude if decoded.sign < 0 else magnitude

def ulp_distance(left: int, right: int, fmt: str | FloatFormat) -> int:
    return abs(ord(left, fmt) - ord(right, fmt))
