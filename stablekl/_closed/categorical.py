"""Repaired KL divergence between two Categorical (probs and logits) distributions: formula, analytic gradients, and the
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
OFFICIAL_FN = '_kl_categorical_categorical'
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
            ax, ay = (abs(float(x[0])), abs(float(y[0])))
            ep = ax * y[1] + ay * x[1] + x[1] * y[1]
            if 0 < ax * ay < 2 * self.h / (self.u * self.u):
                ep += 8 * self.h
        return ((p, ep), (e, 0.0))

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
SENT = -30000

def _grow(o, E, b):
    Q = b
    out = []
    for e in E:
        Q, h = o.two_sum(Q, e)
        out.append(h)
    out.append(Q)
    return out

def _expansion(o, xs):
    E = [xs[0]]
    for x in xs[1:]:
        E = _grow(o, E, x)
    return E

def _esum(o, E):
    acc = E[0]
    for e in E[1:]:
        acc = o.add(acc, e)
    return acc

def _hilo(o, E):
    hi = _esum(o, E)
    o.exact = 1 if o.trace else 0
    R = _grow(o, E, o.neg(o.seterr(hi, lambda: 0.0)))
    o.exact = 0
    return (hi, _esum(o, R))
TH = 2.0 ** (-6)
PSN = {'fp16': 4, 'fp32': 5, 'fp64': 9}
EXMAX = {'fp16': 16, 'fp32': 128, 'fp64': 1024}
PRE = {'fp16': (6, 7), 'fp32': (0, 0), 'fp64': (0, 0)}
LN2H = {dt: Fr(float(FMT[dt]['np'](math.log(2.0)))) for dt in FMT}
LN2L = {dt: LN2_FR - LN2H[dt] for dt in FMT}

def _npair(o, m, e):
    m2, e2 = o.frexp(m)
    if o.trace and float(m2[0]) == 0 and (m2[1] > 0):
        return [m2, e]
    return [m2, o.iwhere(o.eq(m2, 0), SENT, e + e2)]

def _pv(o, F):
    return o.ld2(F[0], F[1])

def _pneg(o, F):
    return [o.neg(F[0]), F[1]]

def _pmul(o, facs, divs=()):
    acc = None
    ex = 0
    for f in facs:
        m, e = f if isinstance(f, list) else o.frexp(f)
        acc = m if acc is None else o.mul(acc, m)
        ex = ex + e
    for f in divs:
        m, e = f if isinstance(f, list) else o.frexp(f)
        acc = o.div(acc, m)
        ex = ex - e
    return [acc, ex]

def _psum(o, Fs):
    if o.trace:
        return _psum_trace(o, Fs)
    Emax = Fs[0][1]
    for F in Fs[1:]:
        Emax = o.imax(Emax, F[1])
    acc = None
    for F in Fs:
        v = o.ld2(F[0], F[1] - Emax)
        acc = v if acc is None else o.add(acc, v)
    return _npair(o, acc, Emax)

def _ldx(x, k):
    try:
        return math.ldexp(x, int(k))
    except OverflowError:
        return math.inf

def _psum_trace(o, Fs):
    nz = [F for F in Fs if float(F[0][0]) != 0]
    zs = [F for F in Fs if float(F[0][0]) == 0 and F[0][1] > 0]
    if not nz:
        if not zs:
            return [(o.T(0), 0.0), SENT]
        es = max((F[1] for F in zs))
        return [(o.T(0), sum((_ldx(F[0][1], F[1] - es) for F in zs))), es]
    Emax = max((F[1] for F in nz))
    acc = None
    for F in Fs:
        if float(F[0][0]) == 0:
            v = (o.T(0), _ldx(F[0][1], F[1] - Emax) if F[0][1] > 0 else 0.0)
        else:
            v = o.ld2(F[0], F[1] - Emax)
        acc = v if acc is None else o.add(acc, v)
    return _npair(o, acc, Emax)

def _pwhere(o, c, fa, fb):
    if o.trace:
        return fa() if c else fb()
    A, B = (fa(), fb())
    return [o.t.where(c, A[0], B[0]), o.iwhere(c, A[1], B[1])]

def _l1(o, x, dt):
    n = L1N[dt]

    def ser():
        p = _horner(o, o.neg(x), [Fr(1, k + 1) for k in range(n)])
        if o.trace:
            p = o.extra(p, lambda: l1_tail(float(x[0]), n))
        return p
    return o.where(o.le(o.abs(x), TH), ser, lambda: o.div(o.fn('log1p', x), x))

def _psi2(o, t, dt):
    n = PSN[dt]
    p = _horner(o, t, [Fr(1, math.factorial(k + 2)) for k in range(n)])
    if o.trace:
        p = o.extra(p, lambda: abs(float(t[0])) ** n / math.factorial(n + 2) * 1.1)
    return p

def _algo_probs(o, X, Y, dt):
    K_ = len(X)
    one = o.const(1)
    PM, PS = PRE[dt]
    FX = [o.frexp(x) for x in X]
    FY = [o.frexp(y) for y in Y]
    emx = FX[0][1]
    for f in FX[1:] + FY:
        emx = o.imax(emx, f[1])
    sh = o.imax(emx - (EXMAX[dt] - 5), 0)
    Xs = [o.ld2(x, 0 - sh) for x in X]
    Ys = [o.ld2(y, 0 - sh) for y in Y]
    o.exact = 1 if o.trace else 0
    Es = _expansion(o, Xs)
    Eq = _expansion(o, Ys)
    o.exact = 0

    def total(E):
        hi, lo = _hilo(o, E)
        S = o.seterr(hi, lambda: abs(float(lo[0])) + lo[1])
        return o.frexp(S)
    mSx, eSx = total(Es)
    mSy, eSy = total(Eq)
    cSx = [o.ld2(c, PS - eSx) for c in Es]
    cSy = [o.ld2(c, PS - eSy) for c in Eq]
    pF = [[o.div(FX[k][0], mSx), FX[k][1] - eSx - sh] for k in range(K_)]
    terms = []
    lrFs = []
    NFs = []
    for k in range(K_):
        mx, ex = FX[k]
        my, ey = FY[k]
        eA = ey + eSx
        eB = ex + eSy
        E = o.imax(eA, eB)
        my2 = o.ld2(my, PM + (eA - E))
        mx2 = o.ld2(mx, PM + (eB - E))
        o.exact = 1 if o.trace else 0
        tl = []
        for c in cSx:
            p_, e_ = o.two_prod(my2, c)
            tl += [p_, e_]
        for c in cSy:
            p_, e_ = o.two_prod(mx2, c)
            tl += [o.neg(p_), o.neg(e_)]
        EN = _expansion(o, tl)
        o.exact = 0
        Nv = _esum(o, EN)
        eN = E - PS - PM
        NFs.append([Nv, eN])
        rhoF = _npair(o, o.div(Nv, o.mul(mx, mSy)), eN - ex - eSy)
        rv = _pv(o, rhoF)
        Rm2, Re2 = o.frexp(o.div(o.mul(my, mSx), o.mul(mx, mSy)))
        logR = o.add(o.fn('log', Rm2), o.mul(_ie(o, ey - ex + eSx - eSy + Re2), o.const(LN2_FR)))
        far = o.lor(o.lt(rv, -0.5), o.lnot(o.isfinite(rv)))
        lrv = o.where(far, lambda: logR, lambda: o.fn('log1p', rv))
        small = o.le(o.abs(rv), TH)
        fin = o.isfinite(rv)
        pk = pF[k]
        if o.trace:
            o.tag('ps' if small else 'pm' if fin else 'pinf')
            o.tag('pfar') if far else None
        lrFs.append(_pwhere(o, small, lambda: [o.mul(rhoF[0], _l1(o, rv, dt)), rhoF[1]], lambda: [lrv, 0]))

        def t_small():
            return _pmul(o, [pk, rhoF, rhoF, [_phh(o, rv, dt), 0]])

        def t_mid():
            return [o.mul(pk[0], _phi(o, one, rv, dt, lx=lrv)), pk[1]]

        def t_inf():
            return _psum(o, [_pmul(o, [pk, rhoF]), [o.neg(o.mul(pk[0], lrv)), pk[1]]])
        terms.append(_pwhere(o, small, t_small, lambda: _pwhere(o, fin, t_mid, t_inf)))
    KLF = _psum(o, terms)
    kl = o.clamp0(_pv(o, KLF))
    LF = {}
    for k in range(K_):
        for j in range(k + 1, K_):
            eA = FY[k][1] + FX[j][1]
            eB = FX[k][1] + FY[j][1]
            E = o.imax(eA, eB)
            o.exact = 1 if o.trace else 0
            p1, e1 = o.two_prod(o.ld2(FY[k][0], PM + (eA - E)), o.ld2(FX[j][0], PM))
            p2, e2 = o.two_prod(o.ld2(FX[k][0], PM + (eB - E)), o.ld2(FY[j][0], PM))
            ED = _expansion(o, [p1, e1, o.neg(p2), o.neg(e2)])
            o.exact = 0
            den = o.mul(FX[k][0], FY[j][0])
            dF = _npair(o, o.div(_esum(o, ED), den), E - 2 * PM - FX[k][1] - FY[j][1])
            dv = _pv(o, dF)
            Rm2, Re2 = o.frexp(o.div(o.mul(FY[k][0], FX[j][0]), den))
            logR = o.add(o.fn('log', Rm2), o.mul(_ie(o, eA - eB + Re2), o.const(LN2_FR)))
            far = o.lor(o.lt(dv, -0.5), o.lnot(o.isfinite(dv)))
            lv = o.where(far, lambda: logR, lambda: o.fn('log1p', dv))
            f = _pwhere(o, o.le(o.abs(dv), TH), lambda: [o.mul(dF[0], _l1(o, dv, dt)), dF[1]], lambda: [lv, 0])
            LF[k, j] = f
            LF[j, k] = _pneg(o, f)
    GP, GQ = ([], [])
    for k in range(K_):
        tb = [_pmul(o, [pF[j], LF[j, k]]) for j in range(K_) if j != k]
        bF = _psum(o, tb)
        mb = _psum(o, [[o.abs(f[0]), f[1]] for f in tb])
        aF = _pneg(o, _psum(o, [lrFs[k], KLF]))
        ma = _psum(o, [[o.abs(lrFs[k][0]), lrFs[k][1]], KLF])
        useb = o.lt(_psum(o, [mb, _pneg(o, ma)])[0], 0.0)
        if o.trace:
            o.tag('Pb' if useb else 'Pa')
        sF = _pwhere(o, useb, lambda: bF, lambda: aF)
        GP.append(o.ld2(o.div(sF[0], mSx), sF[1] - eSx - sh))
        Nv, eN = NFs[k]
        GQ.append(o.ld2(o.div(Nv, o.mul(o.mul(FY[k][0], mSx), mSy)), eN - FY[k][1] - eSx - eSy - sh))
    return [kl] + [o.pz(g) for g in GP] + [o.pz(g) for g in GQ]

def _expF(o, hi, lo, dt):
    kf = o.rint(o.mul(hi, o.const(1 / LN2_FR)))
    p_, e_ = o.two_prod(o.ld2(kf, -4), o.const(16 * LN2H[dt]))
    r = o.sub(o.sub(o.sub(hi, p_), e_), o.mul(kf, o.const(LN2L[dt])))
    return [o.fn('exp', o.add(r, lo)), o.toint(kf)]

def _algo_logits(o, Z, W, dt):
    K_ = len(Z)
    one = o.const(1)
    zmax = Z[0]
    for z in Z[1:]:
        zmax = o.max2(zmax, z)
    o.exact = 1 if o.trace else 0
    AH = [o.two_sum(z, o.neg(zmax)) for z in Z]
    o.exact = 0
    eF = [_expF(o, ah, al, dt) for ah, al in AH]
    Zs = None
    for f in eF:
        v = _pv(o, f)
        Zs = v if Zs is None else o.add(Zs, v)
    pF = [[o.div(f[0], Zs), f[1]] for f in eF]

    def expan(xs):
        o.exact = 1 if o.trace else 0
        E = _expansion(o, xs)
        o.exact = 0
        return _hilo(o, E)
    Wr = W[K_ - 1]
    for k in range(K_ - 2, -1, -1):
        Wr = o.where(o.eq(Z[k], zmax), lambda k=k: W[k], lambda Wr=Wr: Wr)
    base = lambda k: [W[k], o.neg(Z[k]), o.neg(Wr), zmax]
    cl = []
    for k in range(K_):
        s1h, s1l = expan(base(k))
        cl += [_pmul(o, [pF[k], _npair(o, s1h, 0)]), _pmul(o, [pF[k], _npair(o, s1l, 0)])]
    c = _pv(o, _psum(o, cl))
    c = o.seterr(c, lambda: 0.0)
    TH_, TL_ = ([], [])
    for k in range(K_):
        th, tl = expan(base(k) + [o.neg(c)])
        TH_.append(th)
        TL_.append(tl)
    thF = [_npair(o, th, 0) for th in TH_]
    tlF = [_npair(o, tl, 0) for tl in TL_]
    tv = [o.seterr(th, lambda tl=tl: abs(float(tl[0])) + tl[1]) for th, tl in zip(TH_, TL_)]
    WF = []
    for k in range(K_):
        o.exact = 1 if o.trace else 0
        s_, e2 = o.two_sum(AH[k][0], o.seterr(TH_[k], lambda: 0.0))
        o.exact = 0
        f = _expF(o, s_, o.add(o.add(e2, AH[k][1]), TL_[k]), dt)
        WF.append([o.div(f[0], Zs), f[1]])
    SWF = _psum(o, WF)
    MF = _psum(o, [_pmul(o, [pF[k], thF[k]]) for k in range(K_)] + [_pmul(o, [pF[k], tlF[k]]) for k in range(K_)])
    st = []
    for k in range(K_):
        pk = pF[k]
        t = tv[k]
        small = o.le(o.abs(t), TH)
        mid = o.le(t, 1.0)
        if o.trace:
            o.tag('ls' if small else 'lm' if mid else 'lb')

        def t_small(k=k, pk=pk, t=t):
            return _pmul(o, [pk, thF[k], thF[k], [_psi2(o, t, dt), 0]])

        def t_mid(pk=pk, t=t):
            return [o.mul(pk[0], _phi(o, one, o.fn('expm1', t), dt, lx=t)), pk[1]]

        def t_big(k=k, pk=pk):
            return _psum(o, [WF[k], _pneg(o, pk), _pneg(o, _pmul(o, [pk, thF[k]]))])
        st.append(_pwhere(o, small, t_small, lambda: _pwhere(o, mid, t_mid, t_big)))
    SF = _psum(o, st)
    EF = _psum(o, [MF, SF])
    Ev = _pv(o, EF)
    near = o.le(Ev, 1.0)
    esm = o.le(o.abs(Ev), TH)
    if o.trace:
        o.tag('Ln' if near else 'Lf')

    def L_far():
        return [o.add(o.fn('log', SWF[0]), o.mul(_ie(o, SWF[1]), o.const(LN2_FR))), 0]
    LF = _pwhere(o, esm, lambda: [o.mul(EF[0], _l1(o, Ev, dt)), EF[1]], lambda: _pwhere(o, near, lambda: [o.fn('log1p', Ev), 0], L_far))

    def kl_near():
        phE = _pwhere(o, esm, lambda: _pmul(o, [EF, EF, [_phh(o, Ev, dt), 0]]), lambda: [_phi(o, one, Ev, dt), 0])
        return _psum(o, [SF, _pneg(o, phE)])
    KLF = _pwhere(o, near, kl_near, lambda: _psum(o, [LF, _pneg(o, MF)]))
    kl = o.clamp0(_pv(o, KLF))
    GZ, GW = ([], [])
    for k in range(K_):
        pk = pF[k]
        dF = _psum(o, [thF[k], tlF[k], _pneg(o, MF)])
        GZ.append(o.neg(_pv(o, _pmul(o, [pk, dF]))))
        sF = _psum(o, [thF[k], tlF[k], _pneg(o, LF)])
        sv = _pv(o, sF)
        small = o.le(o.abs(sv), TH)
        mid = o.le(sv, 1.0)
        if o.trace:
            o.tag('gs' if small else 'gm' if mid else 'gb')

        def g_small(pk=pk, sF=sF, sv=sv):
            return _pmul(o, [pk, sF, [_E1(o, sv, dt), 0]])

        def g_mid(pk=pk, sv=sv):
            return [o.mul(pk[0], o.fn('expm1', sv)), pk[1]]

        def g_big(k=k, pk=pk):
            return _psum(o, [[o.div(WF[k][0], SWF[0]), WF[k][1] - SWF[1]], _pneg(o, pk)])
        GW.append(_pv(o, _pwhere(o, small, g_small, lambda: _pwhere(o, mid, g_mid, g_big))))
    return [kl] + [o.pz(g) for g in GZ] + [o.pz(g) for g in GW]
_ALGO = {'probs': _algo_probs, 'logits': _algo_logits}
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    _DT = {torch.float16: 'fp16', torch.float32: 'fp32', torch.float64: 'fp64'}

    def _make(kind):

        class _F(torch.autograd.Function):

            @staticmethod
            def forward(ctx, a, c):
                dt = _DT[a.dtype]
                K_ = a.shape[1]
                A = [a[:, j] for j in range(K_)]
                C = [c[:, j] for j in range(K_)]
                o = _TOps(torch, A[0], dt)
                outs = _ALGO[kind](o, A, C, dt)
                same = (a == c).all(dim=1)
                zero = torch.zeros_like(A[0])
                outs = [torch.where(same, zero, x) for x in outs]
                ga = torch.stack(outs[1:K_ + 1], dim=1)
                gc = torch.stack(outs[K_ + 1:], dim=1)
                ctx.save_for_backward(ga, gc)
                return outs[0]

            @staticmethod
            def backward(ctx, go):
                ga, gc = ctx.saved_tensors
                return (go[:, None] * ga, go[:, None] * gc)
        _F.__name__ = f'Categorical{kind.capitalize()}KLC'
        return _F
    KLC = {'probs': _make('probs'), 'logits': _make('logits')}
SUBCONFIGS = {'categorical_probs_K2': ('probs', 2, {'fp16': 54478, 'fp32': 72710}), 'categorical_probs_K4': ('probs', 4, {'fp16': 49302, 'fp32': 72645}), 'categorical_probs_K16': ('probs', 16, {'fp16': 50649, 'fp32': 72645}), 'categorical_logits_K2': ('logits', 2, {'fp16': 54132, 'fp32': 67366}), 'categorical_logits_K4': ('logits', 4, {'fp16': 55690, 'fp32': 67317}), 'categorical_logits_K16': ('logits', 16, {'fp16': 55924, 'fp32': 67317})}
TRANSCENDENTAL_FUNCS = ('log1p', 'log', 'expm1', 'exp')
NAME = None
KIND = None
KDIM = None
PARAMS = None
COMPS = None
EXPECTED_ROWS = None
TRANSCENDENTAL_COMPS = None

def _comps(kind, k):
    return ['KL'] + [f'P.{kind}[{j}]' for j in range(k)] + [f'Q.{kind}[{j}]' for j in range(k)]

def configure(sub):
    global NAME, KIND, KDIM, PARAMS, COMPS, EXPECTED_ROWS, TRANSCENDENTAL_COMPS
    KIND, KDIM, EXPECTED_ROWS = SUBCONFIGS[sub]
    NAME = sub
    PARAMS = (KIND,)
    COMPS = _comps(KIND, KDIM)
    TRANSCENDENTAL_COMPS = tuple(COMPS)

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    a = to_t(P[KIND], dt_in).to(TD[cdt]).requires_grad_(True)
    c = to_t(Q[KIND], dt_in).to(TD[cdt]).requires_grad_(True)
    kl = KLC[KIND].apply(a, c)
    kl.sum().backward()
    vals = {'KL': kl.detach()}
    for j in range(KDIM):
        vals[f'P.{KIND}[{j}]'] = a.grad[:, j].detach()
        vals[f'Q.{KIND}[{j}]'] = c.grad[:, j].detach()
    return vals

def exact_grads(*ex):
    return None

def identity(*ex):
    h = len(ex) // 2
    a, c = (ex[:h], ex[h:])
    if KIND == 'probs':
        sa, sc = (sum(a), sum(c))
        return all((x * sc == y * sa for x, y in zip(a, c)))
    d = [x - y for x, y in zip(a, c)]
    return all((v == d[0] for v in d))

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
        print(f'fix_categorical.running_bounds: {dt} lambda_pos = {_LAM_POS[dt]}; the bound uses max(lambda from p2_accept, lambda_pos)', flush=True)
    return _LAM_POS[dt]

def _trace(kind, vals, dt, lam):
    o = _ROps(dt, lam)
    k = len(vals) // 2
    X = [o.inp(x) for x in vals]
    A, C = (X[:k], X[k:])
    names = _comps(kind, k)
    if all((float(a[0]) == float(c[0]) for a, c in zip(A, C))):
        z = (o.T(0), 0.0)
        return (dict(zip(names, [z] * len(names))), {'_branch': 'id'})
    res = _ALGO[kind](o, A, C, dt)
    return (dict(zip(names, res)), {'_branch': '/'.join(sorted(set(o.tags)))})
_CACHE = {}

def replay(*args, kind=None):
    *vals, dt = args
    lamc = max(3, _LAM_POS.get(dt, 3))
    kind = kind or KIND
    out, flags = _trace(kind, vals, dt, lamc)
    res = {k: float(v[0]) for k, v in out.items()}
    res.update(flags)
    res['_in'] = tuple((float(x) for x in vals))
    res['_kind'] = kind
    _CACHE.clear()
    _CACHE[res['_in'], dt, lamc] = out
    return res

def running_bounds(v, dt, lam):
    L = max(lam, lam_pos(dt))
    out = _CACHE.get((v['_in'], dt, L))
    if out is None:
        out, _ = _trace(v['_kind'], list(v['_in']), dt, L)
    return {k: math.inf if math.isnan(float(e)) else SAFETY * float(e) for k, (_, e) in out.items()}
