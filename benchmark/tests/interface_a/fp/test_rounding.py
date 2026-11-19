from __future__ import annotations
from fractions import Fraction
import random
import struct
import pytest
from klbench.fp import BINARY16, BINARY32, bits_to_fraction, fraction_to_bits, exact_value, ord, round_fraction, ulp_distance

def test_format_constants() -> None:
    assert BINARY16.minimum_subnormal == Fraction(1, 1 << 24)
    assert BINARY32.minimum_subnormal == Fraction(1, 1 << 149)
    assert BINARY16.maximum == 65504
    assert BINARY16.overflow_threshold == 65520
    assert BINARY32.eps == Fraction(1, 1 << 23)

def test_fp16_all_65536_patterns_round_trip() -> None:
    for raw in range(1 << 16):
        decoded = bits_to_fraction(raw, BINARY16)
        assert fraction_to_bits(decoded, BINARY16) == raw

def test_fp16_all_adjacent_finite_midpoints_and_offsets() -> None:
    finite = [i for i in range(1 << 16) if bits_to_fraction(i, BINARY16).kind == 'finite']
    finite.sort(key=lambda raw: ord(raw, BINARY16))
    for raw_left, raw_right in zip(finite, finite[1:]):
        left, right = (exact_value(raw_left, BINARY16), exact_value(raw_right, BINARY16))
        if left == right:
            continue
        midpoint = (left + right) / 2
        step = (right - left) / (1 << 60)
        expected_tie = raw_left if not raw_left & 1 else raw_right
        assert ord(round_fraction(midpoint, BINARY16), BINARY16) == ord(expected_tie, BINARY16)
        assert ord(round_fraction(midpoint - step, BINARY16), BINARY16) == ord(raw_left, BINARY16)
        assert ord(round_fraction(midpoint + step, BINARY16), BINARY16) == ord(raw_right, BINARY16)

def test_special_values_and_ulp_order() -> None:
    for fmt in (BINARY16, BINARY32):
        pos_zero, neg_zero = (0, fmt.sign_mask)
        assert bits_to_fraction(neg_zero, fmt).sign == -1
        assert bits_to_fraction(pos_zero, fmt).sign == 1
        assert ord(pos_zero, fmt) == ord(neg_zero, fmt) == 0
        assert ulp_distance(pos_zero, neg_zero, fmt) == 0
        assert ulp_distance(fmt.sign_mask | 1, 1, fmt) == 2
        assert round_fraction(-fmt.minimum_subnormal / 2, fmt) == neg_zero
        assert round_fraction(fmt.minimum_subnormal / 2, fmt) == pos_zero
        assert round_fraction(fmt.minimum_subnormal / 2 + fmt.minimum_subnormal / (1 << 65), fmt) == 1
        assert round_fraction(fmt.maximum, fmt) == fmt.maximum_finite_bits
        assert round_fraction(fmt.overflow_threshold, fmt) == fmt.infinity_bits
        assert round_fraction(fmt.overflow_threshold - Fraction(1, 1 << 300), fmt) == fmt.maximum_finite_bits
        assert bits_to_fraction(fmt.infinity_bits, fmt).kind == 'infinity'
        assert bits_to_fraction(fmt.infinity_bits | 1, fmt).kind == 'nan'
        with pytest.raises(ValueError):
            ord(fmt.infinity_bits | 1, fmt)

def test_fp32_exponent_boundaries_and_subnormal_boundaries() -> None:
    fmt = BINARY32
    for exponent_field in range(1, (1 << fmt.exponent_bits) - 1):
        boundary = exponent_field << fmt.fraction_bits
        assert round_fraction(exact_value(boundary, fmt), fmt) == boundary
        previous = boundary - 1
        assert round_fraction(exact_value(previous, fmt), fmt) == previous
        midpoint = (exact_value(previous, fmt) + exact_value(boundary, fmt)) / 2
        assert round_fraction(midpoint, fmt) == boundary
    assert exact_value(fmt.fraction_mask, fmt) + fmt.minimum_subnormal == fmt.minimum_normal
    for b in (0, 1, 2, 3, 4194303, 4194304, 8388605, 8388606, 8388607):
        assert round_fraction(exact_value(b, fmt), fmt) == b
    for power in range(23):
        crossing = 1 << power
        assert round_fraction(exact_value(crossing, fmt), fmt) == crossing
        if crossing > 1:
            previous = crossing - 1
            midpoint = (exact_value(previous, fmt) + exact_value(crossing, fmt)) / 2
            expected = previous if not previous & 1 else crossing
            assert round_fraction(midpoint, fmt) == expected
    for b in (1, 8388606, 8388607):
        assert round_fraction(-exact_value(b, fmt), fmt) == fmt.sign_mask | b

def test_direct_versus_double_rounding() -> None:
    x = Fraction(1) + Fraction(1, 1 << 11) + Fraction(1, 1 << 25)
    assert round_fraction(x, BINARY16) == 15361
    via32 = round_fraction(x, BINARY32)
    assert round_fraction(exact_value(via32, BINARY32), BINARY16) == 15360

@pytest.mark.slow
def test_fp32_million_random_roundtrip_and_midpoint() -> None:
    rng = random.Random(20261105)
    fmt = BINARY32
    for _ in range(1000000):
        bits = rng.getrandbits(32)
        decoded = bits_to_fraction(bits, fmt)
        assert fraction_to_bits(decoded, fmt) == bits
        if decoded.kind != 'finite':
            continue
        positive = bits & fmt.sign_mask - 1
        if positive >= fmt.maximum_finite_bits:
            continue
        a, b = (exact_value(positive, fmt), exact_value(positive + 1, fmt))
        middle = (a + b) / 2
        assert round_fraction(middle, fmt) == (positive if not positive & 1 else positive + 1)

@pytest.mark.slow
def test_million_float64_against_numpy_each_format() -> None:
    import numpy as np
    rng = random.Random(101)
    for fmt, dtype in ((BINARY16, np.float16), (BINARY32, np.float32)):
        mask = (1 << fmt.bits) - 1
        for _ in range(1000000):
            bits64 = rng.getrandbits(64)
            value = struct.unpack('<d', struct.pack('<Q', bits64))[0]
            if not value == value or value in (float('inf'), -float('inf')):
                continue
            predicted = round_fraction(Fraction.from_float(value), fmt, sign_hint=-1 if value == 0 and bits64 >> 63 else 1)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                observed = np.array([value], dtype=np.float64).astype(dtype)
            raw = int(observed.view(np.uint16 if fmt is BINARY16 else np.uint32)[0]) & mask
            assert predicted == raw, (fmt.name, hex(bits64), hex(predicted), hex(raw))
