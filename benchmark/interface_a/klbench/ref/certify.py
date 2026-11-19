from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, cast
import mpmath as mp
from ..fp import FloatFormat, get_format, round_fraction
from ..fp.bits import exact_value
from .arbutil import arb_bounds
from .constants import ARB_PRECISIONS, KL_BINS, XCHECK_EXTRA_BITS
from .configs import get_config
Params = dict[str, list[Fraction]]
BitParams = dict[str, list[int]]

class CertStatus(str, Enum):
    CERTIFIED = 'CERTIFIED'
    UNCERT_BIN = 'UNCERT_BIN'
    UNCERT_VALUE = 'UNCERT_VALUE'
    UNCERT_GRAD = 'UNCERT_GRAD'
    XCHECK_MISMATCH = 'XCHECK_MISMATCH'
    MATH_INF = 'MATH_INF'
    NONDIFF = 'NONDIFF'
    INFEASIBLE = 'INFEASIBLE'
    INSUFFICIENT_SAMPLES = 'INSUFFICIENT_SAMPLES'

@dataclass(frozen=True)
class ValueCertificate:
    lower: Fraction | None
    upper: Fraction | None
    truth_bits: int | None
    precision_bits: int | None
    status: CertStatus
    group: str | None = None
    exact_zero: bool = False
    rounding_overflow: bool = False
    error_type: str | None = None
    error_message: str | None = None

@dataclass(frozen=True)
class CertificationIssue:
    stage: str
    precision_bits: int
    error_type: str
    message: str
    component: str | None = None

@dataclass(frozen=True)
class XCheck:
    status: str
    low_precision_bits: int | None
    high_precision_bits: int | None
    low_value: str | None = None
    high_value: str | None = None
    note: str | None = None

@dataclass(frozen=True)
class SampleCertificate:
    config_id: str
    dtype: str
    sample_type: str
    kl: ValueCertificate
    gradients: dict[str, dict[str, list[ValueCertificate]]]
    xcheck: XCheck
    semantic_identity: bool
    exact_bitwise_identity: bool
    sample_tag: str | None = None
    rejection_reason: str | None = None
    issues: tuple[CertificationIssue, ...] = ()

def classify_bin(lower: Fraction, upper: Fraction) -> str | None:
    if not 0 <= lower <= upper:
        return None
    for name, minimum, maximum in KL_BINS:
        if minimum <= lower and upper < maximum:
            return name
    return None

def _proved_out_of_range(lower: Fraction, upper: Fraction) -> bool:
    return upper < KL_BINS[-1][1] or lower >= KL_BINS[0][2]

def _certify_rational(value: Fraction, fmt: FloatFormat, group: str | None=None, exact_zero: bool=False) -> ValueCertificate:
    bits = round_fraction(value, fmt)
    return ValueCertificate(value, value, bits, None, CertStatus.CERTIFIED, group, exact_zero, bits & fmt.sign_mask - 1 == fmt.infinity_bits)

def _assert_exact_in_arb(ball: Any, exact: Fraction, label: str) -> None:
    lo, hi = arb_bounds(ball)
    if not lo <= exact <= hi:
        raise AssertionError(f'{label}: exact rational {exact} outside 128-bit Arb [{lo}, {hi}]')

def _certify_ball(ball: Any, fmt: FloatFormat, precision: int, *, group: str | None=None, is_gradient: bool=False, exact_zero: bool=False) -> ValueCertificate:
    low, high = arb_bounds(ball)
    left, right = (round_fraction(low, fmt), round_fraction(high, fmt))
    same = left == right or (left & fmt.sign_mask - 1 == 0 and right & fmt.sign_mask - 1 == 0)
    status = CertStatus.CERTIFIED if same else CertStatus.UNCERT_GRAD if is_gradient else CertStatus.UNCERT_VALUE
    bits = left if same else None
    return ValueCertificate(low, high, bits, precision, status, group, exact_zero, bits is not None and bits & fmt.sign_mask - 1 == fmt.infinity_bits)

def _xcheck(config: Any, p: Params, q: Params, certificate: ValueCertificate) -> XCheck:
    if certificate.lower is None or certificate.upper is None:
        return XCheck('NOT_RUN', None, None, note='No finite Arb interval')
    last_bits = certificate.precision_bits or ARB_PRECISIONS[0]
    bits1 = last_bits + XCHECK_EXTRA_BITS
    bits2 = bits1 + XCHECK_EXTRA_BITS
    dps1 = bits1 * 302 // 1000 + 5
    dps2 = bits2 * 302 // 1000 + 5
    first = config.kl_mpmath_alt(p, q, dps1)
    second = config.kl_mpmath_alt(p, q, dps2)
    with mp.workdps(dps2 + 10):
        lo = mp.mpf(certificate.lower.numerator) / mp.mpf(certificate.lower.denominator)
        hi = mp.mpf(certificate.upper.numerator) / mp.mpf(certificate.upper.denominator)
        midpoint, radius = ((lo + hi) / 2, (hi - lo) / 2)
        discrepancy = abs(second - midpoint)
        change = abs(second - first)
        numerical_floor = mp.power(2, -bits2) * max(mp.mpf(1), abs(second))
        mismatch = discrepancy > max(2 * (radius + change), numerical_floor)
    return XCheck(CertStatus.XCHECK_MISMATCH.value if mismatch else 'PASS', bits1, bits2, mp.nstr(first, 22), mp.nstr(second, 22), 'Diagnostic only: never overrides rigorous Arb certification')

def _to_exact_parameters(bits: BitParams, fmt: FloatFormat) -> Params:
    return {name: [exact_value(bit, fmt) for bit in values] for name, values in bits.items()}

def certify_sample(config_id: str, fmt: str | FloatFormat, p_bits: BitParams, q_bits: BitParams, *, sample_type: str='nonzero') -> SampleCertificate:
    if sample_type not in ('nonzero', 'identity', 'ulp'):
        raise ValueError('Invalid sample_type')
    spec = get_format(fmt)
    config = get_config(config_id)
    p, q = (_to_exact_parameters(p_bits, spec), _to_exact_parameters(q_bits, spec))
    valid = config.domain_ok(p, q)
    if valid is not True:
        raise ValueError(f'Invalid mathematical distribution inputs: {valid}')
    raw_equal = p_bits == q_bits
    semantic_equal = bool(config.semantic_equal(p, q))
    if sample_type == 'identity' and (not raw_equal):
        raise ValueError('Identity-group samples require bitwise identical P and Q')
    if sample_type == 'nonzero' and semantic_equal:
        tag = 'ZERO_KL_NONIDENTICAL' if not raw_equal else 'IDENTICAL_OUTSIDE_IDENTITY_GROUP'
    else:
        tag = None
    gradient_zero = config.exact_zero_mask(p, q)
    nondiff = config.nondiff_mask(p, q) if hasattr(config, 'nondiff_mask') else {side: {name: [False] * len(mask) for name, mask in coords.items()} for side, coords in gradient_zero.items()}
    exact_grads = config.exact_grad(p, q) if hasattr(config, 'exact_grad') else {side: {name: [None] * len(mask) for name, mask in coords.items()} for side, coords in gradient_zero.items()}
    for side, coords in gradient_zero.items():
        assert set(nondiff[side]) == set(coords) == set(exact_grads[side])
        for name, mask in coords.items():
            assert len(nondiff[side][name]) == len(mask) == len(exact_grads[side][name])
            assert not any((a and b for a, b in zip(mask, nondiff[side][name]))), f'Conflicting exact-zero and nondifferentiable marks: {side}.{name}'
    issues: list[CertificationIssue] = []

    def reject_out_of_range(value: ValueCertificate) -> SampleCertificate:
        return SampleCertificate(config_id, spec.name, sample_type, value, {}, XCheck('NOT_RUN', None, None, note='Candidate rejected: certified outside the KL bins'), semantic_equal, raw_equal, 'OUT_OF_RANGE', 'OUT_OF_RANGE', tuple(issues))
    exact_kl: Fraction | None = Fraction(0) if semantic_equal else None
    if exact_kl is None and hasattr(config, 'exact_kl'):
        exact_kl = config.exact_kl(p, q)
    kl: ValueCertificate | None = None
    if exact_kl is not None:
        _assert_exact_in_arb(config.kl_arb(p, q, ARB_PRECISIONS[0]), exact_kl, f'{config_id}.exact_kl')
        bin_name = classify_bin(exact_kl, exact_kl) if exact_kl > 0 else 'identity'
        kl = _certify_rational(exact_kl, spec, bin_name, exact_zero=exact_kl == 0)
        if sample_type == 'nonzero' and exact_kl > 0 and _proved_out_of_range(exact_kl, exact_kl):
            return reject_out_of_range(kl)
    grads: dict[str, dict[str, list[ValueCertificate | None]]] = {side: {name: [None] * len(mask) for name, mask in coords.items()} for side, coords in gradient_zero.items()}
    if any((value is not None for sides in exact_grads.values() for values in sides.values() for value in values)):
        first_arb_grads = config.grad_arb(p, q, ARB_PRECISIONS[0])
        for side, coords in exact_grads.items():
            for name, values in coords.items():
                for index, value in enumerate(values):
                    if value is not None and (not nondiff[side][name][index]):
                        _assert_exact_in_arb(first_arb_grads[side][name][index], value, f'{config_id}.{side}.{name}[{index}]')
    for side, coords in grads.items():
        for name, values in coords.items():
            for index in range(len(values)):
                if nondiff[side][name][index]:
                    values[index] = ValueCertificate(None, None, None, None, CertStatus.NONDIFF)
                elif exact_grads[side][name][index] is not None and (not gradient_zero[side][name][index]):
                    value = exact_grads[side][name][index]
                    assert isinstance(value, Fraction), 'exact_grad must return Fraction or None'
                    values[index] = _certify_rational(value, spec, exact_zero=value == 0)
    for precision in ARB_PRECISIONS:
        if kl is None or kl.status in (CertStatus.UNCERT_BIN, CertStatus.UNCERT_VALUE):
            try:
                ball = config.kl_arb(p, q, precision)
                new_kl = _certify_ball(ball, spec, precision)
            except ValueError as exc:
                issues.append(CertificationIssue('KL', precision, type(exc).__name__, str(exc)))
                new_kl = ValueCertificate(None, None, None, precision, CertStatus.UNCERT_VALUE, error_type=type(exc).__name__, error_message=str(exc))
            if new_kl.lower is not None and new_kl.upper is not None:
                group = classify_bin(new_kl.lower, new_kl.upper)
                if sample_type == 'nonzero' and group is None:
                    if _proved_out_of_range(new_kl.lower, new_kl.upper):
                        return reject_out_of_range(new_kl)
                    new_kl = ValueCertificate(new_kl.lower, new_kl.upper, new_kl.truth_bits, precision, CertStatus.UNCERT_BIN, None, new_kl.exact_zero, new_kl.rounding_overflow)
                else:
                    new_kl = ValueCertificate(new_kl.lower, new_kl.upper, new_kl.truth_bits, precision, new_kl.status, group, new_kl.exact_zero, new_kl.rounding_overflow)
            kl = new_kl
        pending = any((v is None or v.status not in (CertStatus.CERTIFIED, CertStatus.NONDIFF) for side in grads.values() for coordinates in side.values() for v in coordinates))
        if pending:
            try:
                grad_balls = config.grad_arb(p, q, precision)
            except ValueError as exc:
                issues.append(CertificationIssue('GRAD', precision, type(exc).__name__, str(exc), 'all'))
                for side, coordinates in grads.items():
                    for name, values in coordinates.items():
                        for index, previous in enumerate(values):
                            if previous is None or previous.status not in (CertStatus.CERTIFIED, CertStatus.NONDIFF):
                                values[index] = ValueCertificate(None, None, None, precision, CertStatus.UNCERT_GRAD, error_type=type(exc).__name__, error_message=str(exc))
                continue
            for side, coordinates in grads.items():
                for name, values in coordinates.items():
                    for index, previous in enumerate(values):
                        if previous is not None and previous.status in (CertStatus.CERTIFIED, CertStatus.NONDIFF):
                            continue
                        component = f'{side}.{name}[{index}]'
                        try:
                            grad_ball = grad_balls[side][name][index]
                            if gradient_zero[side][name][index]:
                                lower, upper = arb_bounds(grad_ball)
                                if not lower <= 0 <= upper:
                                    raise AssertionError(f'Incorrect exact_zero_mask for {config_id} {component}: Arb interval [{lower}, {upper}] excludes 0')
                                exact = exact_grads[side][name][index]
                                assert exact is None or exact == 0, f'Inconsistent exact_grad and exact_zero_mask: {component}'
                                values[index] = _certify_rational(Fraction(0), spec, exact_zero=True)
                            else:
                                values[index] = _certify_ball(grad_ball, spec, precision, is_gradient=True)
                        except ValueError as exc:
                            issues.append(CertificationIssue('GRAD', precision, type(exc).__name__, str(exc), component))
                            values[index] = ValueCertificate(None, None, None, precision, CertStatus.UNCERT_GRAD, exact_zero=gradient_zero[side][name][index], error_type=type(exc).__name__, error_message=str(exc))
        kl_done = kl is not None and kl.status == CertStatus.CERTIFIED
        grad_done = all((v is not None and v.status in (CertStatus.CERTIFIED, CertStatus.NONDIFF) for side in grads.values() for coords in side.values() for v in coords))
        if kl_done and grad_done:
            break
    assert kl is not None
    assert all((v is not None for side in grads.values() for coords in side.values() for v in coords)), 'Incomplete gradient certificate'
    final_grad: dict[str, dict[str, list[ValueCertificate]]] = {side: {name: [cast(ValueCertificate, v) for v in vals] for name, vals in coords.items()} for side, coords in grads.items()}
    try:
        xcheck = _xcheck(config, p, q, kl)
    except (ValueError, OverflowError) as exc:
        issues.append(CertificationIssue('XCHECK', kl.precision_bits or ARB_PRECISIONS[0], type(exc).__name__, str(exc)))
        xcheck = XCheck('NOT_RUN', None, None, note=f'{type(exc).__name__}: {exc}')
    return SampleCertificate(config_id, spec.name, sample_type, kl, final_grad, xcheck, semantic_equal, raw_equal, tag, None, tuple(issues))
