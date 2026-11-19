from __future__ import annotations
from fractions import Fraction as Fr
from flint import arb

def A(v):
    v = Fr(v)
    return arb(v.numerator) / arb(v.denominator)
U = {'fp16': Fr(1, 2 ** 11), 'fp32': Fr(1, 2 ** 24), 'fp64': Fr(1, 2 ** 53)}
ETA = {'fp16': Fr(1, 2 ** 25), 'fp32': Fr(1, 2 ** 150), 'fp64': Fr(1, 2 ** 1075)}
MINN = {'fp16': Fr(1, 2 ** 14), 'fp32': Fr(1, 2 ** 126), 'fp64': Fr(1, 2 ** 1022)}
WIDE = {('C', 'fp16'): 'fp16', ('C', 'fp32'): 'fp32', ('D32', 'fp16'): 'fp32', ('D32', 'fp32'): 'fp32', ('D64', 'fp16'): 'fp64', ('D64', 'fp32'): 'fp64'}

def _exp_term(dt, t, cof):
    e = A(t).exp()
    eta = A(ETA[dt])
    if e >= A(MINN[dt]):
        return (2.0, arb(0))
    return (12.0, 2 * eta * cof if cof < 1.5 else arb(0))

def _lost(method, dt, t):
    w = WIDE[method, dt]
    return 4 * A(ETA[w]) if A(t).exp() < A(MINN[w]) else arb(0)

def elem_bound(method, dt, lt, x, t, yabs):
    u = A(U[WIDE[method, dt]])
    if method == 'C':
        if not lt:
            return None
        k, ab = _exp_term(dt, t, abs(A(t - x)))
        return (k + 2.1) * u * yabs + ab + A(ETA[dt])
    if lt:
        return 4.1 * u * yabs + _lost(method, dt, t) * abs(A(t - x))
    return A(t) * (0.8 * u + 1.1 * u * u * (abs(A(t).log()) + abs(A(x)) + 2)) + 2.1 * u * yabs

def grad_bound(method, dt, lt, side, x, t, gabs, s):
    w = WIDE[method, dt]
    u = A(U[w])
    sa = A(s)
    if method == 'C':
        eta = A(ETA[w])
        if not lt:
            return 3.1 * u * gabs + eta if side == 'gx' else sa * 1.6 * u + 4.2 * u * gabs + eta
        if side == 'gx':
            k, ab = _exp_term(dt, t, sa)
            return (k + 3.1) * u * gabs + ab + eta
        k, ab = _exp_term(dt, t, sa * abs(A(t - x + 1)))
        return sa * A(t).exp() * 1.1 * u * u * (abs(A(t)) + abs(A(x)) + 1) + (k + 4.2) * u * gabs + ab + eta
    if not lt:
        if side == 'gx':
            return 2.1 * u * gabs
        l = abs(A(t).log())
        return sa * (0.8 * u + 1.1 * u * u * (l + abs(A(x)) + 2)) + 3.2 * u * gabs
    lost = _lost(method, dt, t)
    if side == 'gx':
        return 4.1 * u * gabs + lost * sa
    return sa * A(t).exp() * 1.1 * u * u * (abs(A(t)) + abs(A(x)) + 1) + 5.2 * u * gabs + lost * sa * abs(A(t - x + 1))

def sum_bound(method, dt, n, sum_elem_bounds, sum_abs, sabs, s):
    u = A(U[WIDE[method, dt]])
    lg = max(1, (2 * n - 1).bit_length())
    if method == 'C':
        return A(s) * (sum_elem_bounds + n * u * sum_abs) + 2.2 * u * sabs + A(ETA[dt])
    return A(s) * (sum_elem_bounds + 1.01 * n * lg * u * u * sum_abs) + 2.2 * u * sabs

def within(value, exact_ball, bound):
    d = abs(A(value) - exact_ball)
    if d <= bound:
        return True
    if d > bound:
        return False
    return None
