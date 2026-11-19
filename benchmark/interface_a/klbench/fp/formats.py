from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

@dataclass(frozen=True)
class FloatFormat:
    name: str
    exponent_bits: int
    fraction_bits: int
    bias: int

    @property
    def bits(self) -> int:
        return self.exponent_bits + self.fraction_bits + 1

    @property
    def precision(self) -> int:
        return self.fraction_bits + 1

    @property
    def emin(self) -> int:
        return 1 - self.bias

    @property
    def emax(self) -> int:
        return (1 << self.exponent_bits) - 2 - self.bias

    @property
    def sign_mask(self) -> int:
        return 1 << self.bits - 1

    @property
    def fraction_mask(self) -> int:
        return (1 << self.fraction_bits) - 1

    @property
    def infinity_bits(self) -> int:
        return (1 << self.exponent_bits) - 1 << self.fraction_bits

    @property
    def maximum_finite_bits(self) -> int:
        return self.infinity_bits - 1

    @property
    def minimum_normal(self) -> Fraction:
        return self.power_of_two(self.emin)

    @property
    def minimum_subnormal(self) -> Fraction:
        return self.power_of_two(self.emin - self.fraction_bits)

    @property
    def eps(self) -> Fraction:
        return self.power_of_two(-self.fraction_bits)

    @property
    def maximum(self) -> Fraction:
        return Fraction((1 << self.precision) - 1) * self.power_of_two(self.emax - self.fraction_bits)

    @property
    def overflow_threshold(self) -> Fraction:
        return self.maximum + self.power_of_two(self.emax - self.fraction_bits - 1)

    @staticmethod
    def power_of_two(k: int) -> Fraction:
        return Fraction(1 << k) if k >= 0 else Fraction(1, 1 << -k)
BINARY16 = FloatFormat('fp16', 5, 10, 15)
BINARY32 = FloatFormat('fp32', 8, 23, 127)
FORMATS = MappingProxyType({'fp16': BINARY16, 'binary16': BINARY16, 'fp32': BINARY32, 'binary32': BINARY32})

def get_format(fmt: str | FloatFormat) -> FloatFormat:
    if isinstance(fmt, FloatFormat):
        if fmt not in (BINARY16, BINARY32):
            raise ValueError('Only IEEE binary16/binary32 are supported')
        return fmt
    try:
        return FORMATS[fmt.lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f'Unknown IEEE format {fmt!r}') from exc
