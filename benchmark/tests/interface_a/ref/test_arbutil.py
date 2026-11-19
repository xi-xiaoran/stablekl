from fractions import Fraction
from flint import arb
from klbench.ref.arbutil import arb_precision, fraction_to_arb, arb_bounds

def test_dyadics_exact_and_endpoints() -> None:
    with arb_precision(256):
        for value in (Fraction(0), Fraction(1, 1 << 149), Fraction(-3, 1 << 50), Fraction(65504), Fraction(2 ** 120 + 1, 1 << 160)):
            ball = fraction_to_arb(value)
            assert ball.is_exact()
            assert ball.rad() == 0
            assert arb_bounds(ball) == (value, value)

def test_one_third_enclosed_without_decimal_rounding() -> None:
    with arb_precision(128):
        approximate = fraction_to_arb(Fraction(1, 3))
        low, high = arb_bounds(approximate)
        assert low <= Fraction(1, 3) <= high
        assert low < high

def test_log_interval_contains_high_precision_mpmath_value() -> None:
    import mpmath as mp
    with arb_precision(128):
        bounds = arb_bounds(fraction_to_arb(Fraction(3, 2)).log())
    with mp.workdps(100):
        actual = mp.log(mp.mpf(3) / 2)
        low = mp.mpf(bounds[0].numerator) / bounds[0].denominator
        high = mp.mpf(bounds[1].numerator) / bounds[1].denominator
        assert low <= actual <= high
