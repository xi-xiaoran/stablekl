"""Repaired KL divergence between two ContinuousBernoulli (probs and logits) distributions: formula, analytic gradients, and the
entry point run_rewrite_torch used by the benchmark (dtype in, compute dtype, outputs per component).
The derivation is in the paper's appendix on interface A."""
from __future__ import annotations
import math
from decimal import Decimal
from fractions import Fraction as Fr
import numpy as np
FMT = {'fp16': dict(np=np.float16, u=2.0 ** (-11), eta=2.0 ** (-24), minnorm=2.0 ** (-14), split=65.0), 'fp32': dict(np=np.float32, u=2.0 ** (-24), eta=2.0 ** (-149), minnorm=2.0 ** (-126), split=4097.0), 'fp64': dict(np=np.float64, u=2.0 ** (-53), eta=2.0 ** (-1074), minnorm=2.0 ** (-1022), split=134217729.0)}
SAFETY = 1.02
KTERMS = {'fp16': 9, 'fp32': 17, 'fp64': 36}
L1N = {'fp16': 4, 'fp32': 6, 'fp64': 10}
SEG_LO, SEG_HI = (-0.75, 3.0)
OFFICIAL_FN = '_kl_continuous_bernoulli_continuous_bernoulli'
GRAD_TRUTH = 'frozen_interval'
LN2_FR = Fr(Decimal('0.693147180559945309417232121458176568075500134'))

def q_tail_bound(z, K):
    return z ** K / ((2 * K + 3) * (1 - z))

def l1_tail(r, n):
    return abs(r) ** n / ((n + 1) * (1 - abs(r)))

class _ROps:
    trace = True

    def __init__(self, dt, lam):
        f = FMT[dt]
        self.dt = dt
        self.T = f['np']
        self.u = f['u']
        self.h = f['eta'] / 2 or 5e-324
        self.lam = lam
        self.mn = f['minnorm']
        self.OT = float(np.finfo(self.T).max) * (1 + self.u)
        self.C = f['split']
        self.exact = 0
        self.tags = []

    def A(self, x):
        return min(abs(float(x)), self.OT)

    def sb(self, x):
        return 1.0 if abs(float(x)) < self.mn else 0.0

    def R(self, z):
        return 0.0 if self.exact else self.u * self.A(z) + self.h * self.sb(z)

    def inp(self, x):
        return (self.T(x), 0.0)

    def const(self, c):
        v = self.T(float(c))
        return (v, float(abs(Fr(float(v)) - Fr(c))))

    @staticmethod
    def ovf(z, *ins):
        return not math.isfinite(float(z)) and all((math.isfinite(float(x[0])) for x in ins))

    def add(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] + b[0])
        if self.ovf(z, a, b):
            return (z, math.inf)
        return (z, a[1] + b[1] + (self.u * self.A(z) if abs(float(z)) >= self.mn and (not self.exact) else 0.0))

    def sub(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] - b[0])
        if self.ovf(z, a, b):
            return (z, math.inf)
        return (z, a[1] + b[1] + (self.u * self.A(z) if abs(float(z)) >= self.mn and (not self.exact) else 0.0))

    def mul(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] * b[0])
        if self.ovf(z, a, b):
            return (z, math.inf)
        return (z, self.A(b[0]) * a[1] + self.A(a[0]) * b[1] + self.R(z))

    def div(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] / b[0])
        bb = abs(float(b[0]))
        if bb == math.inf:
            return (z, a[1] / bb if a[1] < math.inf and b[1] < math.inf else math.inf)
        if self.ovf(z, a, b) and bb > 0:
            return (z, math.inf)
        return (z, (a[1] + self.A(z) * b[1]) / bb + self.R(z) if bb > 0 else math.inf)

    def fn(self, name, a):
        with np.errstate(all='ignore'):
            z = self.T(getattr(np, name)(np.array([a[0]], dtype=self.T))[0])
        xv = float(a[0])
        e = float(a[1])
        if not math.isfinite(e) or (name in ('exp', 'expm1') and self.ovf(z, a)):
            return (z, math.inf)
        if name in ('exp', 'expm1'):
            lx = min(xv, math.log(self.OT))
            if e == 0:
                prop = 0.0
            elif e < 1:
                prop = math.exp(lx) * math.expm1(e)
            else:
                prop = math.exp(lx + e) if lx + e < 709.7 else math.inf
        else:
            y = abs(xv if name == 'log' else 1.0 + xv)
            prop = 0.0 if e == 0 else -math.log1p(-e / y) if e < y else math.inf
        return (z, prop + self.lam * self.R(z))

    def where(self, c, fa, fb):
        return fa() if c else fb()

    def wheret(self, c, fa, fb):
        return fa() if c else fb()

    def iwhere(self, c, a, b):
        return a if c else b

    def lt(self, a, b):
        return float(a[0]) < (float(b[0]) if isinstance(b, tuple) else b)

    def le(self, a, b):
        return float(a[0]) <= (float(b[0]) if isinstance(b, tuple) else b)

    def eq(self, a, b):
        return float(a[0]) == (float(b[0]) if isinstance(b, tuple) else b)

    def land(self, a, b):
        return a and b

    def lor(self, a, b):
        return a or b

    def lnot(self, a):
        return not a

    def isfinite(self, a):
        return math.isfinite(float(a[0]))

    def abs(self, a):
        return (self.T(abs(a[0])), a[1])

    def neg(self, a):
        return (self.T(-a[0]), a[1])

    def frexp(self, a):
        m, e = math.frexp(float(a[0]))
        return ((self.T(m), float(np.ldexp(a[1], -e)) if math.isfinite(a[1]) else math.inf), e)

    def ld2(self, x, k):
        k = int(k)
        with np.errstate(all='ignore'):
            z = self.T(float(np.ldexp(np.float64(x[0]), k)))
            ek = float(np.ldexp(np.float64(x[1]), k))
        if self.ovf(z, x):
            return (z, math.inf)
        exact = float(z) == float(np.ldexp(np.float64(x[0]), k))
        if float(x[0]) != 0 and k < 0 and (float(z) == 0 or (self.dt == 'fp64' and abs(float(z)) < self.mn)):
            return (z, ek + self.h)
        return (z, ek + (0.0 if exact else self.R(z)))

    def iclamp(self, e, lo, hi):
        return max(lo, min(hi, e))

    def imax(self, a, b):
        return max(a, b)

    def iabs(self, e):
        return abs(e)

    def two_prod(self, x, y):
        T = self.T
        C = T(self.C)

        def split(v):
            c = T(v * C)
            hi = T(c - T(c - v))
            return (hi, T(v - hi))
        with np.errstate(all='ignore'):
            p = T(x[0] * y[0])
            hx, lx = split(x[0])
            hy, ly = split(y[0])
            e = T(T(T(T(hx * hy) - p) + T(hx * ly)) + T(lx * hy))
            e = T(e + T(lx * ly))
        return ((p, 0.0), (e, 0.0))

    def two_sum(self, a, b):
        T = self.T
        if float(a[0]) == 0:
            return ((b[0], a[1] + b[1]), (T(0), 0.0))
        if float(b[0]) == 0:
            return ((a[0], a[1] + b[1]), (T(0), 0.0))
        with np.errstate(all='ignore'):
            s = T(a[0] + b[0])
            bb = T(s - a[0])
            e = T(T(a[0] - T(s - bb)) + T(b[0] - bb))
        return ((s, a[1] + b[1] + (0.0 if self.exact else abs(float(e)))), (e, 0.0))

    def extra(self, a, ef):
        return (a[0], a[1] + ef())

    def seterr(self, a, ef):
        return (a[0], ef())

    def clamp0(self, a):
        return (max(a[0], self.T(0)), a[1] if math.isfinite(float(a[0])) else math.inf)

    def pz(self, a):
        with np.errstate(all='ignore'):
            return (self.T(a[0] + self.T(0)), a[1])

    def tag(self, t):
        self.tags.append(t)

    def fval(self, a):
        return float(a[0])

    def rint(self, a):
        with np.errstate(all='ignore'):
            return (self.T(np.rint(a[0])), 0.0)

    def toint(self, a):
        return int(float(a[0]))

    def max2(self, a, b):
        return a if float(a[0]) >= float(b[0]) else b

class _TOps:
    trace = False

    def __init__(self, torch, ref, dt):
        self.t = torch
        self.ref = ref
        self.dt = dt
        self.C = FMT[dt]['split']
        self.iz = torch.zeros_like(ref, dtype=torch.int32)

    def inp(self, x):
        return x

    def const(self, c):
        return self.t.full_like(self.ref, float(c))

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        return a / b

    def fn(self, name, a):
        return getattr(self.t, name)(a)

    def where(self, c, fa, fb):
        return self.t.where(c, fa(), fb())

    def wheret(self, c, fa, fb):
        return tuple((self.t.where(c, x, y) for x, y in zip(fa(), fb())))

    def _i(self, a):
        return a if hasattr(a, 'dtype') else self.iz + int(a)

    def iwhere(self, c, a, b):
        return self.t.where(c, self._i(a), self._i(b))

    def lt(self, a, b):
        return a < b

    def le(self, a, b):
        return a <= b

    def eq(self, a, b):
        return a == b

    def land(self, a, b):
        return a & b

    def lor(self, a, b):
        return a | b

    def lnot(self, a):
        return ~a

    def isfinite(self, a):
        return self.t.isfinite(a)

    def abs(self, a):
        return a.abs()

    def neg(self, a):
        return -a

    def frexp(self, a):
        m, e = self.t.frexp(a.double())
        return (m.to(a.dtype), e)

    def ld2(self, x, k):
        return (x.double() * self.t.exp2(self._i(k).double())).to(x.dtype)

    def iclamp(self, e, lo, hi):
        return self.t.clamp(self._i(e), lo, hi)

    def imax(self, a, b):
        return self.t.maximum(self._i(a), self._i(b))

    def iabs(self, e):
        return self._i(e).abs()

    def two_prod(self, x, y):
        C = self.const(self.C)

        def split(v):
            c = v * C
            hi = c - (c - v)
            return (hi, v - hi)
        p = x * y
        hx, lx = split(x)
        hy, ly = split(y)
        e = hx * hy - p + hx * ly + lx * hy
        e = e + lx * ly
        return (p, e)

    def two_sum(self, a, b):
        s = a + b
        bb = s - a
        e = a - (s - bb) + (b - bb)
        return (s, e)

    def extra(self, a, ef):
        return a

    def seterr(self, a, ef):
        return a

    def clamp0(self, a):
        return self.t.clamp_min(a, 0.0)

    def pz(self, a):
        return a + 0.0

    def tag(self, t):
        pass

    def rint(self, a):
        return self.t.round(a.double()).to(a.dtype)

    def toint(self, a):
        return a.to(self.t.int32)

    def max2(self, a, b):
        return self.t.where(a >= b, a, b)

def _horner(o, x, coefs):
    p = o.const(coefs[-1])
    for c in coefs[-2::-1]:
        p = o.add(o.mul(p, x), o.const(c))
    return p

def mprod(o, facs, divs=()):
    acc = None
    ex = 0
    for f in facs:
        m, e = f if isinstance(f, list) else o.frexp(f)
        acc = m if acc is None else o.mul(acc, m)
        ex = ex + e
    for f in divs:
        m, e = f if isinstance(f, list) else o.frexp(f)
        acc = o.div(acc if acc is not None else o.const(1), m)
        ex = ex - e
    return o.ld2(acc, ex)

def _phi(o, coef, x, dt, lx=None):
    K = KTERMS[dt]
    seg = o.land(o.le(o.const(SEG_LO), x), o.le(x, o.const(SEG_HI)))

    def inside():
        s = o.div(x, o.add(o.const(2), x))
        z = o.mul(s, s)
        q = _horner(o, z, [Fr(1, 2 * k + 1) for k in range(1, K + 1)])
        if o.trace:
            q = o.extra(q, lambda: q_tail_bound(float(z[0]), K))
        F = o.sub(x, o.mul(o.mul(o.const(2), z), q))
        return o.mul(o.mul(coef, s), F)
    return o.where(seg, inside, lambda: o.mul(coef, o.sub(x, lx if lx is not None else o.fn('log1p', x))))

def _ie(o, e):
    if o.trace:
        return o.const(int(e))
    return e.to(o.ref.dtype)

def _phh(o, x, dt):
    K = KTERMS[dt]
    two = o.const(2)
    tz = o.add(two, x)
    s = o.div(x, tz)
    z = o.mul(s, s)
    q = _horner(o, z, [Fr(1, 2 * k + 1) for k in range(1, K + 1)])
    if o.trace:
        q = o.extra(q, lambda: q_tail_bound(float(z[0]), K))
    return o.div(o.sub(o.const(1), o.div(o.mul(o.mul(two, x), q), o.mul(tz, tz))), tz)
E1N = {'fp16': 4, 'fp32': 5, 'fp64': 9}

def _E1(o, x, dt):
    n = E1N[dt]

    def ser():
        p = _horner(o, x, [Fr(1, math.factorial(k + 1)) for k in range(n)])
        if o.trace:
            p = o.extra(p, lambda: abs(float(x[0])) ** n / math.factorial(n + 1) * 1.1)
        return p
    return o.where(o.lt(o.abs(x), 2.0 ** (-6)), ser, lambda: o.div(o.fn('expm1', x), x))
TH = 2.0 ** (-6)
PSN = {'fp16': 4, 'fp32': 5, 'fp64': 9}

def _psi2(o, t, dt):
    n = PSN[dt]
    p = _horner(o, t, [Fr(1, math.factorial(k + 2)) for k in range(n)])
    if o.trace:
        p = o.extra(p, lambda: abs(float(t[0])) ** n / math.factorial(n + 2) * 1.1)
    return p
TA = 4.0
TLO = 1.5
KS = {'fp16': 17, 'fp32': 27, 'fp64': 50}
NB = 130

def _bern(n):
    a = [Fr(0)] * (n + 1)
    out = []
    for m in range(n + 1):
        a[m] = Fr(1, m + 1)
        for j in range(m, 0, -1):
            a[j - 1] = j * (a[j - 1] - a[j])
        out.append(a[0])
    return out

def _tables():
    B = _bern(2 * NB)
    b = {}
    f = 1
    for k in range(1, NB + 1):
        f *= (2 * k - 1) * (2 * k)
        b[k] = B[2 * k] / f * 16 ** (k - 1)
    coef, tail = ({}, {})
    w = 16 / (4 * math.pi ** 2)
    rem = 0.25 * w ** (NB + 1) / (1 - w) * (2 * NB + 3) * (NB + 2) * 4
    floor = {'g0': 0.0372, 'g1': 0.0672, 'g2': 0.0431, 'gc': 0.0299, 'S': 0.0216, 'Q': 0.0431}
    for dt, K in KS.items():
        coef[dt] = {'g0': [b[k] / (2 * k) for k in range(1, K + 1)], 'g1': [b[k] for k in range(1, K + 1)], 'g2': [(2 * k - 1) * b[k] for k in range(1, K + 1)], 'gc': [b[k] * (1 - Fr(1, 2 * k)) for k in range(1, K + 1)]}
        rest = range(K + 1, NB + 1)
        tail[dt] = {'g0': float(sum((abs(b[k]) / (2 * k) for k in rest))) + rem, 'g1': float(sum((abs(b[k]) for k in rest))) + rem, 'g2': float(sum(((2 * k - 1) * abs(b[k]) for k in rest))) + rem, 'gc': float(sum((abs(b[k]) for k in rest))) + rem, 'S': float(sum((abs(b[k]) / (2 * k) * (2 * k - 1) * k for k in rest))) + rem, 'Q': float(sum((abs(b[k]) * (2 * k - 1) for k in rest))) + rem}
        for n_, t_ in tail[dt].items():
            assert t_ / floor[n_] <= FMT[dt]['u'] / 16, (dt, n_, t_)
    return (coef, tail)
COEF, TAIL = _tables()

def _G(o, x, dt, name):
    y = o.ld2(x, -2)
    z = o.mul(y, y)
    p = _horner(o, z, COEF[dt][name])
    if o.trace:
        p = o.extra(p, lambda: TAIL[dt][name] * min(1.0, float(z[0]) + z[1]) ** KS[dt])
    return p

def _exn(o, x):
    E = o.fn('exp', o.neg(x))
    if o.trace and float(E[0]) == 0.0 and math.isfinite(x[1]):
        tb = max(math.exp(-(float(x[0]) - x[1])), 5e-324) * 1.000001
        if tb < E[1]:
            E = o.seterr(E, lambda: tb)
    return E

def _gval(o, x, E, dt):

    def small():
        return o.mul(o.mul(x, x), _G(o, x, dt, 'g0'))

    def mid():
        xh = o.ld2(x, -1)
        eh = o.fn('exp', xh)
        return o.fn('log', o.div(o.ld2(o.sub(eh, o.div(o.const(1), eh)), -1), xh))

    def big():
        return o.add(o.sub(o.ld2(x, -1), o.fn('log', x)), o.fn('log1p', o.neg(E)))
    return o.where(o.le(x, TA), small, lambda: o.where(o.le(x, 16.0), mid, big))

def _g1val(o, x, E, dt):
    one = o.const(1)

    def small():
        return o.mul(x, _G(o, x, dt, 'g1'))

    def mid():
        xh = o.ld2(x, -1)
        e = o.fn('exp', o.neg(xh))
        return o.add(o.ld2(o.mul(xh, _G(o, xh, dt, 'g1')), -1), o.ld2(o.div(o.sub(one, e), o.add(one, e)), -2))

    def big():
        return o.add(o.sub(o.const(Fr(1, 2)), o.div(one, x)), o.div(E, o.sub(one, E)))
    return o.where(o.le(x, TA), small, lambda: o.where(o.le(x, 8.0), mid, big))

def _cval(o, x, E, dt):
    one = o.const(1)

    def small():
        return o.mul(o.mul(x, x), _G(o, x, dt, 'gc'))

    def mid():
        xh = o.ld2(x, -1)
        e = o.fn('exp', o.neg(xh))
        b = o.sub(o.sub(o.const(LN2_FR), o.div(o.mul(xh, e), o.add(one, e))), o.fn('log1p', e))
        return o.add(o.mul(o.mul(xh, xh), _G(o, xh, dt, 'gc')), b)

    def big():
        eh = o.fn('exp', o.neg(o.ld2(x, -1)))
        return o.sub(o.add(o.sub(o.fn('log', x), one), o.div(o.mul(o.mul(x, eh), eh), o.sub(one, E))), o.fn('log1p', o.neg(E)))
    return o.where(o.le(x, TA), small, lambda: o.where(o.le(x, 8.0), mid, big))

def _g2(o, x, E, dt):
    one = o.const(1)
    sm = o.le(x, TA)

    def big():
        t = o.div(o.mul(x, o.fn('exp', o.neg(o.ld2(x, -1)))), o.sub(one, E))
        return o.sub(one, o.mul(t, t))
    return (o.where(sm, lambda: _G(o, x, dt, 'g2'), big), o.where(sm, lambda: one, lambda: x))

def _logq(o, a, b, dt):
    ma, ea = o.frexp(a)
    mb, eb = o.frexp(b)
    m2, e2 = o.frexp(o.div(ma, mb))
    e = ea - eb + e2
    pos = e >= 1
    mm = o.where(pos, lambda: o.add(m2, m2), lambda: m2)
    return o.add(o.fn('log', mm), o.mul(_ie(o, o.iwhere(pos, e - 1, e)), o.const(LN2_FR)))

def _eta(o, lam, mh, dt):
    one = o.const(1)

    def mid():
        o.exact = 1 if o.trace else 0
        t = o.sub(o.add(lam, lam), one)
        o.exact = 0
        return o.fn('log1p', o.div(t, mh))

    def lo():
        return o.sub(o.fn('log', lam), o.fn('log1p', o.neg(lam)))

    def hi():
        return o.sub(o.fn('log1p', o.neg(mh)), o.fn('log', mh))
    return o.where(o.lt(lam, 0.25), lo, lambda: o.where(o.le(lam, 0.75), mid, hi))
TIE_TAU = 1.0 / 16
TIE_FMT = {'fp16': (11, -14), 'fp32': (24, -126)}

def _tie_round(o, A, C, tgt):
    prec, emin = TIE_FMT[tgt]
    if not o.trace:
        t = o.t
        s = A + C
        e = A - s + C
        _, ex = t.frexp(s.double())
        ulp = t.ldexp(t.ones_like(s).double(), t.clamp_min(ex - 1, emin) - (prec - 1)).to(s.dtype)
        r = s.to({'fp16': t.float16, 'fp32': t.float32}[tgt]).to(s.dtype)
        mid = ((s - r).abs() * 2 == ulp) & (e != 0)
        return t.where(mid, t.nextafter(s, s + t.sign(e) * s.abs()), s)
    T = o.T
    sv = o.add(A, C)
    with np.errstate(all='ignore'):
        e = T(T(A[0] - sv[0]) + C[0])
        _, ex = np.frexp(float(sv[0]))
        ulp = 2.0 ** (max(int(ex) - 1, emin) - (prec - 1))
        r = float(T(np.array(sv[0]).astype(np.float16 if tgt == 'fp16' else np.float32)))
        if abs(float(sv[0]) - r) * 2 == ulp and float(e) != 0:
            nv = np.nextafter(sv[0], T(sv[0] + np.sign(e) * abs(sv[0])))
            o.tag('Tn')
            return (nv, sv[1] + abs(float(nv) - float(sv[0])))
    return sv

def _tie_grads(o, p, q, dh, dl, dt, tgt):
    K = KS[dt]
    C1, C2 = (COEF[dt]['g1'], COEF[dt]['g2'])
    one = o.const(1)
    c12 = o.const(12)
    yp = o.ld2(p, -2)
    yq = o.ld2(q, -2)
    z = o.mul(yp, yp)
    rP = o.mul(o.mul(c12, z), _horner(o, z, C2[1:]))
    pm = one
    V = one
    SQ = None
    for m in range(1, 2 * K - 1):
        pm = o.mul(pm, yp)
        V = o.add(o.mul(yq, V), pm)
        if m % 2 == 0:
            t_ = o.mul(o.const(C1[m // 2]), V)
            SQ = t_ if SQ is None else o.add(SQ, t_)
    rQ = o.mul(c12, SQ)
    if o.trace:
        zm = min(1.0, max(abs(float(yp[0])) + yp[1], abs(float(yq[0])) + yq[1])) ** (2 * K)
        rP = o.extra(rP, lambda: 12 * TAIL[dt]['g2'] * zm)
        rQ = o.extra(rQ, lambda: 12 * TAIL[dt]['Q'] * zm)
    hi = o.div(dh, c12)
    o.exact = 1 if o.trace else 0
    th, tl = o.two_sum(o.ld2(hi, 3), o.ld2(hi, 2))
    r = o.sub(o.sub(dh, th), tl)
    o.exact = 0
    if dl is not None:
        r = o.add(r, dl)
    Cq = o.div(o.add(r, o.mul(dh, rQ)), c12)
    Cp = o.div(o.add(r, o.mul(dh, rP)), c12)
    return (_tie_round(o, o.neg(hi), o.neg(Cp), tgt), _tie_round(o, hi, Cq, tgt))

def _make_algo(kind):

    def algo(o, a, c, dt, tgt=None):
        one = o.const(1)
        if kind == 'logits':
            p, q = (a, c)
            o.exact = 1 if o.trace else 0
            dh, dl = o.two_sum(q, o.neg(p))
            o.exact = 0
            dv = o.seterr(dh, lambda: abs(float(dl[0]))) if o.trace else dh
            cp, cq = ([], [])
        else:
            mph, mpl = o.two_sum(one, o.neg(a))
            mqh, mql = o.two_sum(one, o.neg(c))
            p = _eta(o, a, mph, dt)
            q = _eta(o, c, mqh, dt)
            o.exact = 1 if o.trace else 0
            th, tl = o.two_sum(c, o.neg(a))
            o.exact = 0
            r1 = o.add(o.div(th, a), o.div(tl, a))
            L1 = o.where(o.land(o.le(o.const(-0.5), r1), o.le(r1, 1.0)), lambda: o.fn('log1p', r1), lambda: _logq(o, c, a, dt))
            r2a = o.div(th, mqh)
            r2 = o.add(r2a, o.div(o.sub(tl, o.mul(r2a, mql)), mqh))
            L2 = o.where(o.land(o.le(o.const(-0.5), r2), o.le(r2, 1.0)), lambda: o.fn('log1p', r2), lambda: o.add(_logq(o, mph, mqh, dt), o.sub(o.div(mpl, mph), o.div(mql, mqh))))
            dh = o.add(L1, L2)
            dl = None
            dv = dh
            cp, cq = ([a, mph], [c, mqh])
        xp = o.abs(p)
        xq = o.abs(q)
        Ep = _exn(o, xp)
        Eq = _exn(o, xq)
        pneg = o.lt(p, 0.0)
        qneg = o.lt(q, 0.0)
        opp = o.lor(o.land(pneg, o.lt(o.neg(q), 0.0)), o.land(qneg, o.lt(o.neg(p), 0.0)))
        mx = o.max2(xp, xq)
        mn = o.where(o.le(xp, xq), lambda: xp, lambda: xq)
        inA = o.le(mx, TA)
        inB = o.land(o.lnot(inA), o.land(o.lnot(opp), o.le(o.const(TLO), mn)))
        if o.trace:
            o.tag('A' if inA else 'B' if inB else 'C')

        def gmul(f, divs):
            r = mprod(o, [dh] + f, divs)
            return r if dl is None else o.add(r, mprod(o, [dl] + f, divs))

        def gradp():
            v, dx = _g2(o, xp, Ep, dt)
            return o.neg(gmul([v], [dx, dx] + cp))

        def fA():
            K = KS[dt]
            C0, C1 = (COEF[dt]['g0'], COEF[dt]['g1'])
            yp = o.ld2(p, -2)
            yq = o.ld2(q, -2)
            pm = one
            H = one
            V = one
            S0 = o.const(C0[0])
            SQ = o.const(C1[0])
            for m in range(1, 2 * K - 1):
                pm = o.mul(pm, yp)
                H = o.add(o.mul(yq, H), o.mul(o.const(m + 1), pm))
                V = o.add(o.mul(yq, V), pm)
                if m % 2 == 0:
                    S0 = o.add(S0, o.mul(o.const(C0[m // 2]), H))
                    SQ = o.add(SQ, o.mul(o.const(C1[m // 2]), V))
            if o.trace:
                zm = min(1.0, max(abs(float(yp[0])) + yp[1], abs(float(yq[0])) + yq[1])) ** (2 * K)
                S0 = o.extra(S0, lambda: TAIL[dt]['S'] * zm)
                SQ = o.extra(SQ, lambda: TAIL[dt]['Q'] * zm)
            d2 = dh if dl is None else o.add(dh, o.ld2(dl, 1))
            return (o.clamp0(mprod(o, [dh, d2, S0])), gradp(), gmul([SQ], cq))

        def fB():
            D = o.where(pneg, lambda: o.neg(dv), lambda: dv)
            Dh = o.where(pneg, lambda: o.neg(dh), lambda: dh)
            D2 = Dh if dl is None else o.add(Dh, o.ld2(o.where(pneg, lambda: o.neg(dl), lambda: dl), 1))
            om = o.sub(one, Ep)
            omq = o.sub(one, Eq)
            ehp = o.fn('exp', o.neg(o.ld2(xp, -1)))
            ehq = o.fn('exp', o.neg(o.ld2(xq, -1)))
            rho = o.div(D, xp)
            lx = o.fn('log', o.div(xq, xp))

            def P(z, lz=None):
                return o.where(o.le(o.abs(z), TH), lambda: _phh(o, z, dt), lambda: o.div(o.div(_phi(o, one, z, dt, lx=lz), z), z))
            Pr = P(rho, lx)
            dsm = o.le(o.abs(D), 1.0)
            Ed = o.sub(Ep, Eq)
            if o.trace:
                o.tag('Bs' if dsm else 'Bl')
            tau = o.where(dsm, lambda: o.div(o.mul(Ep, o.neg(o.fn('expm1', o.neg(D)))), om), lambda: o.div(Ed, om))
            tauh = o.where(dsm, lambda: o.div(o.mul(o.mul(o.mul(xp, ehp), ehp), _E1(o, o.neg(D), dt)), om), lambda: mprod(o, [xp, Ed], divs=[D, om]))
            r2 = o.div(o.mul(o.mul(tauh, tauh), P(tau)), Pr)

            def r3a():
                nD = o.neg(D)
                t = o.mul(xp, ehp)
                ps = o.where(o.le(o.abs(D), TH), lambda: _psi2(o, nD, dt), lambda: o.div(o.div(_phi(o, one, o.fn('expm1', nD), dt, lx=nD), D), D))
                return o.div(o.mul(o.mul(t, t), ps), om)

            def r3b():
                return mprod(o, [xp, xp, o.sub(Eq, o.mul(Ep, o.sub(one, D)))], divs=[D, D, om])
            r3 = o.div(o.where(o.le(o.const(-2), D), r3a, r3b), Pr)
            kl = o.clamp0(mprod(o, [Dh, D2, Pr, o.sub(o.sub(one, r2), r3)], divs=[xp, xp]))
            T = o.where(dsm, lambda: o.div(o.mul(o.mul(o.mul(xq, ehq), o.mul(xp, ehq)), _E1(o, D, dt)), o.mul(om, omq)), lambda: mprod(o, [xp, xq, Ed], divs=[D, om, omq]))
            return (kl, gradp(), gmul([o.sub(one, T)], [xp, xq] + cq))

        def fC():
            g1p = _g1val(o, xp, Ep, dt)
            g1q = _g1val(o, xq, Eq, dt)
            base = o.add(_gval(o, xq, Eq, dt), _cval(o, xp, Ep, dt))
            t = o.mul(xq, g1p)
            kl = o.clamp0(o.where(opp, lambda: o.add(base, t), lambda: o.sub(base, t)))
            v = o.sub(o.where(qneg, lambda: o.neg(g1q), lambda: g1q), o.where(pneg, lambda: o.neg(g1p), lambda: g1p))
            return (kl, gradp(), mprod(o, [v], divs=cq) if cq else v)
        kl, gp, gq = o.wheret(inA, fA, lambda: o.wheret(inB, fB, fC))
        if tgt is not None and kind == 'logits':
            tiny = o.land(o.le(xp, TIE_TAU), o.le(xq, TIE_TAU))
            if not o.trace:
                tp, tq = _tie_grads(o, p, q, dh, dl, dt, tgt)
                gp = o.t.where(tiny, tp, gp)
                gq = o.t.where(tiny, tq, gq)
            elif tiny:
                o.tag('T')
                gp, gq = _tie_grads(o, p, q, dh, dl, dt, tgt)
        return [kl, o.pz(gp), o.pz(gq)]
    return algo
_ALGO = {'probs': _make_algo('probs'), 'logits': _make_algo('logits')}
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    _DT = {torch.float16: 'fp16', torch.float32: 'fp32', torch.float64: 'fp64'}

    def _make(kind):

        class _F(torch.autograd.Function):

            @staticmethod
            def forward(ctx, a, c, tgt=None):
                dt = _DT[a.dtype]
                o = _TOps(torch, a, dt)
                outs = _ALGO[kind](o, a, c, dt, tgt)
                same = a == c
                zero = torch.zeros_like(a)
                outs = [torch.where(same, zero, x) for x in outs]
                ctx.save_for_backward(outs[1], outs[2])
                return outs[0]

            @staticmethod
            def backward(ctx, go):
                ga, gc = ctx.saved_tensors
                return (go * ga, go * gc, None)
        _F.__name__ = f'ContinuousBernoulli{kind.capitalize()}KLC'
        return _F
    KLC = {'probs': _make('probs'), 'logits': _make('logits')}
SUBCONFIGS = {'contbern_probs': ('probs', {'fp16': 124070, 'fp32': 218000}), 'contbern_logits': ('logits', {'fp16': 158691, 'fp32': 218000})}
TRANSCENDENTAL_FUNCS = ('log1p', 'log', 'expm1', 'exp')
NAME = None
KIND = None
PARAMS = None
COMPS = None
EXPECTED_ROWS = None
TRANSCENDENTAL_COMPS = None

def _comps(kind):
    return ['KL', f'P.{kind}', f'Q.{kind}']

def configure(sub):
    global NAME, KIND, PARAMS, COMPS, EXPECTED_ROWS, TRANSCENDENTAL_COMPS
    KIND, EXPECTED_ROWS = SUBCONFIGS[sub]
    NAME = sub
    PARAMS = (KIND,)
    COMPS = _comps(KIND)
    TRANSCENDENTAL_COMPS = tuple(COMPS)

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    a = to_t(P[KIND], dt_in).reshape(-1).to(TD[cdt]).requires_grad_(True)
    c = to_t(Q[KIND], dt_in).reshape(-1).to(TD[cdt]).requires_grad_(True)
    kl = KLC[KIND].apply(a, c, dt_in if cdt != dt_in else None)
    kl.sum().backward()
    return {'KL': kl.detach(), f'P.{KIND}': a.grad.detach(), f'Q.{KIND}': c.grad.detach()}

def exact_grads(*ex):
    return None

def identity(*ex):
    return ex[0] == ex[1]

def calibrate_exp_pos(seed=11, dts=('fp16', 'fp32', 'fp64')):
    from decimal import localcontext
    TS = {'fp16': 11.089866488461016, 'fp32': 88.72283905206835, 'fp64': 709.782712893384}
    res = {}
    with localcontext() as ctx:
        ctx.prec = 40
        for k_, dt in enumerate(('fp16', 'fp32', 'fp64')):
            if dt not in dts:
                continue
            rng = np.random.default_rng(seed + k_)
            T = FMT[dt]['np']
            T1 = TS[dt]
            UI = {'fp16': np.uint16, 'fp32': np.uint32, 'fp64': np.uint64}[dt]
            if dt == 'fp16':
                xs = np.arange(1, 31744, dtype=np.uint16).view(np.float16)
                xs = xs[xs.astype(float) <= T1]
            else:
                lo = -30 if dt == 'fp32' else -60
                xs = np.unique(np.concatenate([np.exp2(rng.uniform(lo, math.log2(T1), 20000)), rng.uniform(0, T1, 10000)]).astype(T))
                xs = xs[(xs > 0) & (xs.astype(float) <= T1)]
            res[dt] = 0
            for fn in ('exp', 'expm1'):
                if torch is not None:
                    yv = np.asarray(getattr(torch, fn)(torch.from_numpy(xs)).numpy()).astype(T)
                    src = 'torch'
                else:
                    with np.errstate(all='ignore'):
                        yv = getattr(np, fn)(xs).astype(T)
                    src = 'numpy'
                dmax, wx = (0, None)
                for xv, yy in zip(xs.tolist(), yv):
                    ex = Decimal(xv).exp() - (1 if fn == 'expm1' else 0)
                    cc = T(float(ex))
                    cands = [np.nextafter(cc, T(-np.inf)), cc, np.nextafter(cc, T(np.inf))]
                    cr = min(cands, key=lambda q_: (abs(Decimal(float(q_)) - ex), int(np.array(q_).view(UI)) & 1))
                    d = abs(int(np.array(yy).view(UI)) - int(np.array(cr).view(UI)))
                    if d > dmax:
                        dmax, wx = (d, xv)
                res[dt] = max(res[dt], dmax)
                print(f'calibrate_exp_pos {dt} {fn} ({src}) n={len(xs)} max ULP vs correctly rounded d={dmax} (lambda would be 2d+1={2 * dmax + 1}) at x={wx!r}', flush=True)
    return res
_LAM_POS = {}

def lam_pos(dt):
    if dt not in _LAM_POS:
        _LAM_POS[dt] = 2 * calibrate_exp_pos(dts=(dt,))[dt] + 1
        print(f'fix_contbern.running_bounds: {dt} lambda_pos = {_LAM_POS[dt]}; the bound uses max(lambda from p2_accept, lambda_pos)', flush=True)
    return _LAM_POS[dt]
REPLAY_TARGET = True

def _trace(kind, vals, dt, lam, tgt=None):
    o = _ROps(dt, lam)
    a, c = (o.inp(x) for x in vals)
    names = _comps(kind)
    if float(a[0]) == float(c[0]):
        z = (o.T(0), 0.0)
        return (dict(zip(names, [z] * 3)), {'_branch': 'id'})
    res = _ALGO[kind](o, a, c, dt, tgt)
    return (dict(zip(names, res)), {'_branch': '/'.join(sorted(set(o.tags)))})
_CACHE = {}

def replay(*args, kind=None, tgt=None):
    *vals, dt = args
    lamc = max(3, _LAM_POS.get(dt, 3))
    kind = kind or KIND
    out, flags = _trace(kind, vals, dt, lamc, tgt)
    res = {k: float(v[0]) for k, v in out.items()}
    res.update(flags)
    res['_in'] = tuple((float(x) for x in vals))
    res['_kind'] = kind
    _CACHE.clear()
    _CACHE[res['_in'], dt, lamc, tgt] = out
    return res

def running_bounds(v, dt, lam, tgt=None):
    L = max(lam, lam_pos(dt))
    out = _CACHE.get((v['_in'], dt, L, tgt))
    if out is None:
        out, _ = _trace(v['_kind'], list(v['_in']), dt, L, tgt)
    return {k: math.inf if math.isnan(float(e)) else SAFETY * float(e) for k, (_, e) in out.items()}
