from __future__ import annotations
from fractions import Fraction
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import fraction_to_arb
Params = dict[str, list[Fraction]]

def unpack(p: Params, q: Params, names: tuple[str, ...]) -> tuple[Fraction, ...]:
    for obj in (p, q):
        if set(obj) != set(names) or any((len(obj[name]) != 1 for name in names)):
            raise ValueError(f'Expected scalar constructor arguments {names}')
    return tuple((p[name][0] for name in names)) + tuple((q[name][0] for name in names))

def positive_domain(p: Params, q: Params, names: tuple[str, ...], positive: tuple[str, ...]) -> bool | str:
    try:
        unpack(p, q, names)
    except ValueError as exc:
        return str(exc)
    for side in (p, q):
        if any((side[name][0] <= 0 for name in positive)):
            return f'{positive} must be positive'
    return True

def arbs(values: tuple[Fraction, ...]) -> tuple[arb, ...]:
    return tuple((fraction_to_arb(value) for value in values))

def mpf(value: Fraction) -> Any:
    return mp.mpf(value.numerator) / mp.mpf(value.denominator)

def mps(values: tuple[Fraction, ...]) -> tuple[Any, ...]:
    return tuple((mpf(value) for value in values))

def masks(names: tuple[str, ...], p_values: tuple[bool, ...], q_values: tuple[bool, ...]) -> dict[str, dict[str, list[bool]]]:
    return {'P': {n: [v] for n, v in zip(names, p_values)}, 'Q': {n: [v] for n, v in zip(names, q_values)}}

def rational_grads(names: tuple[str, ...], p_values: tuple[Fraction | None, ...], q_values: tuple[Fraction | None, ...]) -> dict[str, dict[str, list[Fraction | None]]]:
    return {'P': {n: [v] for n, v in zip(names, p_values)}, 'Q': {n: [v] for n, v in zip(names, q_values)}}

def spec(names: tuple[str, ...], positive: tuple[str, ...]=(), *, domain_override: dict[str, str] | None=None) -> dict[str, dict[str, Any]]:
    domain_override = domain_override or {}
    return {n: {'shape': (1,), 'differentiable': True, 'domain': domain_override.get(n, 'positive' if n in positive else 'finite real')} for n in names}

def log_ratio_arb(x: arb, y: arb, difference: Fraction) -> arb:
    return (fraction_to_arb(difference) / y).log1p()

def log_ratio_mp(x: Any, y: Any) -> Any:
    return mp.log(x / y)

def finite_diff_precision(x: Any, dps: int) -> Any:
    return max(abs(x), mp.mpf(1) / 10000) * mp.power(2, -max(32, dps * 3 // 4))

def integer_nth_root_exact(number: int, exponent: int) -> int | None:
    if number < 1 or exponent < 1:
        raise ValueError('Expect positive integer and positive root index')
    if number == 1 or exponent == 1:
        return number
    if exponent >= number.bit_length():
        return None
    low, high = (1, 1 << (number.bit_length() + exponent - 1) // exponent)
    while low <= high:
        mid = (low + high) // 2
        value = mid ** exponent
        if value == number:
            return mid
        if value < number:
            low = mid + 1
        else:
            high = mid - 1
    return None

def rational_power_equal(base: Fraction, exponent: Fraction, target: Fraction) -> bool:
    if min(base, exponent, target) <= 0:
        raise ValueError('Positive rational arguments required')
    u, v = (exponent.numerator, exponent.denominator)
    bnum = integer_nth_root_exact(base.numerator, v)
    bden = integer_nth_root_exact(base.denominator, v)
    if bnum is None or bden is None:
        return False
    tnum = integer_nth_root_exact(target.numerator, u)
    tden = integer_nth_root_exact(target.denominator, u)
    return tnum == bnum and tden == bden
