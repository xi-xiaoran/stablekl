"""Repaired KL divergence between two MultivariateNormal and LowRankMultivariateNormal distributions: formula, analytic gradients, and the
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
OFFICIAL_FN = '_kl_multivariatenormal_multivariatenormal'
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

class _DD:

    def __init__(self, o):
        self.o = o

    def _ex(self, on):
        self.o.exact = (1 if self.o.trace else 0) if on else 0

    def lift(self, x):
        return None if x is None else (x, None)

    def _norm(self, s, e):
        self._ex(1)
        h, l = self.o.two_sum(s, e)
        self._ex(0)
        return (h, l)

    def neg(self, a):
        return None if a is None else (self.o.neg(a[0]), None if a[1] is None else self.o.neg(a[1]))

    def add(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        o = self.o
        self._ex(1)
        s, e = o.two_sum(a[0], b[0])
        self._ex(0)
        for l in (a[1], b[1]):
            if l is not None:
                e = o.add(e, l)
        return self._norm(s, e)

    def sub(self, a, b):
        return self.add(a, self.neg(b))

    def mul(self, a, b):
        if a is None or b is None:
            return None
        o = self.o
        self._ex(1)
        p, e = o.two_prod(a[0], b[0])
        self._ex(0)
        if b[1] is not None:
            e = o.add(e, o.mul(a[0], b[1]))
        if a[1] is not None:
            e = o.add(e, o.mul(a[1], b[0]))
        return self._norm(p, e)

    def val(self, a):
        if a is None:
            return self.o.const(0)
        return a[0] if a[1] is None else self.o.add(a[0], a[1])

    def _approx(self, b):
        o = self.o
        if not o.trace or b[1] is None:
            return b[0]
        return o.seterr(b[0], lambda: abs(float(b[1][0])) + b[0][1] + b[1][1])

    def div(self, a, b):
        if a is None:
            return None
        o = self.o
        q1 = o.div(a[0], b[0])
        if o.trace:
            q1 = o.seterr(q1, lambda: 0.0)
        r = self.sub(a, self.mul((q1, None), b))
        q2 = o.div(self.val(r), self._approx(b))
        return self._norm(q1, q2)

    def sqrt(self, a):
        o = self.o
        s1 = _sqrt(o, a[0])
        if o.trace:
            s1 = o.seterr(s1, lambda: 0.0)
        r = self.val(self.sub(a, self.mul((s1, None), (s1, None))))
        q2 = o.div(r, o.ld2(s1, 1))
        if o.trace:
            s1v = abs(float(s1[0]))
            q2 = o.extra(q2, lambda: 1.01 * (abs(float(r[0])) + r[1]) ** 2 / (8 * s1v ** 3) if s1v > 0 else math.inf)
        return self._norm(s1, q2)

    def half(self, a):
        return None if a is None else (self.o.ld2(a[0], -1), None if a[1] is None else self.o.ld2(a[1], -1))

    def twice(self, a):
        return None if a is None else (self.o.ld2(a[0], 1), None if a[1] is None else self.o.ld2(a[1], 1))

def _cr_sqrt(o, x, s):
    t = o.t
    f = FMT[o.dt]
    inf = t.full_like(s, math.inf)
    ok = t.isfinite(x) & (x >= f['minnorm'] / f['u'] ** 2) & (x <= f['u'] ** 2 / f['minnorm'])
    for _ in range(2):
        p, e = o.two_prod(s, s)
        rh, rl = o.two_sum(x - p, -e)
        up = t.nextafter(s, inf)
        dn = t.nextafter(s, -inf)
        tu = s * (up - s)
        tl = -(s * (s - dn))
        go_up = (rh > tu) | (rh == tu) & (rl > 0)
        go_dn = (rh < tl) | (rh == tl) & (rl <= 0)
        s = t.where(ok & go_up, up, t.where(ok & go_dn, dn, s))
    return s

def check_sqrt(n=1000000, seed=3):
    left = 0
    g = torch.Generator().manual_seed(seed)
    for dt, td in (('fp32', torch.float32), ('fp64', torch.float64)):
        x = (torch.rand(n, generator=g, dtype=torch.float64) * 4 + 0.01).to(td) * (2.0 ** torch.randint(-20, 21, (n,), generator=g)).to(td)
        ref = np.sqrt(x.numpy())
        raw = x.sqrt()
        o = _TOps(torch, x, dt)
        fix = _cr_sqrt(o, x, raw)
        iv = np.int32 if dt == 'fp32' else np.int64
        du = np.abs(raw.numpy().view(iv).astype(np.int64) - ref.view(iv).astype(np.int64))
        nfix = int((fix.numpy() != ref).sum())
        left += nfix
        print(f'sqrt check {dt}: n={n} platform != correctly rounded: {int((du > 0).sum())} (max {int(du.max())} ulp); after _cr_sqrt: {nfix}')
    return left

def _sqrt(o, x):
    if not o.trace:
        return _cr_sqrt(o, x, x.sqrt())
    with np.errstate(all='ignore'):
        z = o.T(np.sqrt(x[0]))
    lo = float(x[0]) - x[1]
    if not lo > 0 or not math.isfinite(x[1]):
        return (z, math.inf)
    return (z, x[1] / (2 * math.sqrt(lo)) + o.R(z))

def _fs(f, L, B):
    d = len(L)
    m = len(B[0])
    X = [[None] * m for _ in range(d)]
    for j in range(m):
        for i in range(d):
            acc = B[i][j]
            for t in range(i):
                acc = f.sub(acc, f.mul(L[i][t], X[t][j]))
            X[i][j] = f.div(acc, L[i][i])
    return X

def _bs(f, L, B, lower=False):
    d = len(L)
    m = len(B[0])
    X = [[None] * m for _ in range(d)]
    for j in range(m):
        for i in range(d - 1, (j if lower else 0) - 1, -1):
            acc = B[i][j]
            for t in range(i + 1, d):
                acc = f.sub(acc, f.mul(L[t][i], X[t][j]))
            X[i][j] = f.div(acc, L[i][i])
    return X

def _tr(A):
    return [list(r) for r in zip(*A)]

def _kl(o, f, N, v, dt):
    one = o.const(1)
    d = len(N)
    acc = None
    sq = None
    for i in range(d):
        n = N[i][i]
        if n is not None:
            if n[1] is None:
                ph = _phi(o, one, n[0], dt)
            else:
                o.exact = 1 if o.trace else 0
                nv, res = o.two_sum(n[0], n[1])
                o.exact = 0
                ph = o.add(_phi(o, one, nv, dt), o.div(o.mul(res, nv), o.add(one, nv)))
                if o.trace:
                    ph = o.extra(ph, lambda: (abs(float(res[0])) + nv[1]) ** 2)
            acc = f.add(acc, (ph, None))
        for j in range(i + 1):
            sq = f.add(sq, f.mul(N[i][j], N[i][j]))
        sq = f.add(sq, f.mul(v[i], v[i]))
    return o.clamp0(f.val(f.add(acc, f.half(sq))))

def _algo_mvn(o, P, Q, dt, d, r=None):
    f = _DD(o)
    mp_, lp = (P[:d], P[d:])
    mq, lq = (Q[:d], Q[d:])
    Lp = [[f.lift(lp[i * d + j]) if j <= i else None for j in range(d)] for i in range(d)]
    Lq = [[f.lift(lq[i * d + j]) if j <= i else None for j in range(d)] for i in range(d)]
    dl = [[f.sub(f.lift(m1), f.lift(m2))] for m1, m2 in zip(mp_, mq)]
    D = [[f.sub(Lp[i][j], Lq[i][j]) if j <= i else None for j in range(d)] for i in range(d)]
    N = _fs(f, Lq, D)
    v = [row[0] for row in _fs(f, Lq, dl)]
    kl = _kl(o, f, N, v, dt)
    gmu = [row[0] for row in _bs(f, Lq, [[x] for x in v])]
    Y = _bs(f, Lq, N, lower=True)
    W = [[None] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            acc = f.twice(N[i][i]) if i == j else N[i][j]
            for t in range(j + 1):
                acc = f.add(acc, f.mul(N[i][t], N[j][t]))
            W[i][j] = f.add(acc, f.mul(v[i], v[j]))
    Z = _bs(f, Lq, W, lower=True)
    gLp = []
    gLq = []
    for i in range(d):
        for j in range(i + 1):
            y = Y[i][j]
            if i == j:
                y = f.add(y, f.div(N[i][i], Lp[i][i]))
            gLp.append(f.val(y))
            gLq.append(o.neg(f.val(Z[i][j])))
    gm = [f.val(g) for g in gmu]
    return [kl] + gm + gLp + [o.neg(g) for g in gm] + gLq

def _algo_lowrank(o, P, Q, dt, d, r):
    f = _DD(o)
    one = f.lift(o.const(1))
    mp_, wp, dp = (P[:d], P[d:d + d * r], P[d + d * r:])
    mq, wq, dq = (Q[:d], Q[d:d + d * r], Q[d + d * r:])
    Wp = [[f.lift(wp[i * r + t]) for t in range(r)] for i in range(d)]
    Wq = [[f.lift(wq[i * r + t]) for t in range(r)] for i in range(d)]
    S = [[None] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            acc = None
            for t in range(r):
                acc = f.add(acc, f.mul(Wq[i][t], Wq[j][t]))
            S[i][j] = f.add(acc, f.lift(dq[i])) if i == j else acc
    Lq = [[None] * d for _ in range(d)]
    for j in range(d):
        s = S[j][j]
        for t in range(j):
            s = f.sub(s, f.mul(Lq[j][t], Lq[j][t]))
        Lq[j][j] = f.sqrt(s)
        for i in range(j + 1, d):
            acc = S[i][j]
            for t in range(j):
                acc = f.sub(acc, f.mul(Lq[i][t], Lq[j][t]))
            Lq[i][j] = f.div(acc, Lq[j][j])
    dW = [[f.sub(Wp[i][t], Wq[i][t]) for t in range(r)] for i in range(d)]
    E = [[None] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            acc = None
            for t in range(r):
                acc = f.add(f.add(acc, f.mul(dW[i][t], Wp[j][t])), f.mul(Wq[i][t], dW[j][t]))
            if i == j:
                acc = f.add(acc, f.sub(f.lift(dp[i]), f.lift(dq[i])))
            E[i][j] = acc
            E[j][i] = acc
    Xf = _tr(_fs(f, Lq, _tr(_fs(f, Lq, E))))
    X = [[Xf[max(i, j)][min(i, j)] for j in range(d)] for i in range(d)]
    N = [[None] * d for _ in range(d)]
    G = [[None] * d for _ in range(d)]
    for j in range(d):
        s = X[j][j]
        for t in range(j):
            s = f.sub(s, f.mul(G[j][t], G[j][t]))
        N[j][j] = None if s is None else f.div(s, f.add(one, f.sqrt(f.add(one, s))))
        G[j][j] = f.add(one, N[j][j])
        for i in range(j + 1, d):
            acc = X[i][j]
            for t in range(j):
                acc = f.sub(acc, f.mul(G[i][t], G[j][t]))
            G[i][j] = f.div(acc, G[j][j])
            N[i][j] = G[i][j]
    v = [row[0] for row in _fs(f, Lq, [[f.sub(f.lift(m1), f.lift(m2))] for m1, m2 in zip(mp_, mq)])]
    kl = _kl(o, f, N, v, dt)
    gmu = [f.val(row[0]) for row in _bs(f, Lq, [[x] for x in v])]
    dmu = [f.sub(f.lift(m1), f.lift(m2)) for m1, m2 in zip(mp_, mq)]
    EM = [[None] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            EM[i][j] = f.add(E[i][j], f.mul(dmu[i], dmu[j]))
            EM[j][i] = EM[i][j]
    Kf = _tr(_fs(f, Lq, _tr(_fs(f, Lq, EM))))
    Kq = [[Kf[max(i, j)][min(i, j)] for j in range(d)] for i in range(d)]
    Hq = _bs(f, Lq, _tr(_bs(f, Lq, Kq)))
    R = _bs(f, G, _fs(f, G, X))
    Hp = _bs(f, Lq, _tr(_bs(f, Lq, R)))

    def gw(H, W_, sign):
        out = []
        for i in range(d):
            for t in range(r):
                acc = None
                for j in range(d):
                    acc = f.add(acc, f.mul(H[i][j], W_[j][t]))
                out.append(f.val(acc) if sign > 0 else o.neg(f.val(acc)))
        return out
    gdp = [f.val(f.half(Hp[i][i])) for i in range(d)]
    gdq = [o.neg(f.val(f.half(Hq[i][i]))) for i in range(d)]
    return [kl] + gmu + gw(Hp, Wp, 1) + gdp + [o.neg(g) for g in gmu] + gw(Hq, Wq, -1) + gdq

def _sem_guard(f):
    return (f['minnorm'] / f['u'] ** 2, 1.0 / math.sqrt(f['minnorm']))

def _sem_same_torch(o, P, Q, d, r):
    t = o.t
    f = FMT[o.dt]
    lo, big = _sem_guard(f)
    mp_, wp, dp = (P[:d], P[d:d + d * r], P[d + d * r:])
    mq, wq, dq = (Q[:d], Q[d:d + d * r], Q[d + d * r:])
    same = t.ones_like(P[0], dtype=t.bool)
    for a, b in zip(mp_, mq):
        same = same & (a == b)

    def grow(exp, x):
        out = []
        for c in exp:
            x, h = o.two_sum(x, c)
            out.append(h)
        return out + [x]
    for i in range(d):
        for j in range(i + 1):
            exp = []
            for w, sgn in ((wp, 1), (wq, -1)):
                for k in range(r):
                    a, b = (w[i * r + k], w[j * r + k])
                    pr, er = o.two_prod(a, b)
                    zero = (a == 0) | (b == 0)
                    same = same & (zero | (pr.abs() >= lo) & (a.abs() <= big) & (b.abs() <= big))
                    pr = t.where(zero, t.zeros_like(pr), pr)
                    er = t.where(zero, t.zeros_like(er), er)
                    exp = grow(grow(exp, er if sgn > 0 else -er), pr if sgn > 0 else -pr)
            if i == j:
                exp = grow(grow(exp, dp[i]), -dq[i])
            for c in exp:
                same = same & (c == 0)
    return same

def _sem_same_exact(vals, d, r, dt):
    f = FMT[dt]
    T = f['np']
    lo, big = _sem_guard(f)
    h = len(vals) // 2
    P = [T(x) for x in vals[:h]]
    Q = [T(x) for x in vals[h:]]
    if any((float(a) != float(b) for a, b in zip(P[:d], Q[:d]))):
        return False
    wp, dp = (P[d:d + d * r], P[d + d * r:])
    wq, dq = (Q[d:d + d * r], Q[d + d * r:])
    for i in range(d):
        for j in range(i + 1):
            acc = Fr(0)
            for w, sgn in ((wp, 1), (wq, -1)):
                for k in range(r):
                    a, b = (w[i * r + k], w[j * r + k])
                    if float(a) == 0 or float(b) == 0:
                        continue
                    with np.errstate(all='ignore'):
                        pr = abs(float(T(a * b)))
                    if not (pr >= lo and abs(float(a)) <= big and (abs(float(b)) <= big)):
                        return False
                    acc += sgn * Fr(float(a)) * Fr(float(b))
            if i == j:
                acc += Fr(float(dp[i])) - Fr(float(dq[i]))
            if acc != 0:
                return False
    return True
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    _DT = {torch.float16: 'fp16', torch.float32: 'fp32', torch.float64: 'fp64'}

    def _make(algo, d, r):

        class _F(torch.autograd.Function):

            @staticmethod
            def forward(ctx, A, C):
                dt = _DT[A.dtype]
                o = _TOps(torch, A[:, 0], dt)
                m = A.shape[1]
                outs = algo(o, [A[:, i] for i in range(m)], [C[:, i] for i in range(m)], dt, d, r)
                same = (A == C).all(dim=1)
                zero = torch.zeros_like(A[:, 0])
                if algo is _algo_lowrank:
                    same = same | _sem_same_torch(o, [A[:, i] for i in range(m)], [C[:, i] for i in range(m)], d, r)
                outs = [torch.where(same, zero, o.pz(x)) for x in outs]
                ctx.save_for_backward(torch.stack(outs[1:], dim=1))
                return outs[0]

            @staticmethod
            def backward(ctx, go):
                g, = ctx.saved_tensors
                h = g.shape[1] // 2
                idx = ctx_idx
                gA = torch.zeros(g.shape[0], msize, dtype=g.dtype, device=g.device)
                gC = torch.zeros_like(gA)
                gA[:, idx] = g[:, :h]
                gC[:, idx] = g[:, h:]
                return (go[:, None] * gA, go[:, None] * gC)
        ctx_idx, msize = _layout(algo is _algo_mvn, d, r)
        return _F
SUBCONFIGS = {'mvn_d2': ('mvn', 2, None, 101000), 'mvn_d4': ('mvn', 4, None, 101000), 'lowrank_mvn_d2_r1': ('lowrank', 2, 1, 50500), 'lowrank_mvn_d2_r2': ('lowrank', 2, 2, 50500), 'lowrank_mvn_d4_r1': ('lowrank', 4, 1, 50500), 'lowrank_mvn_d4_r2': ('lowrank', 4, 2, 50500)}
SKIP_DTYPES = {'fp16': 'torch has no fp16 (Half) linear-algebra kernels: the official KL does not run in fp16, so there is nothing to fix or to compare with'}
TRANSCENDENTAL_FUNCS = ('log1p', 'log')
NAME = None
KIND = None
DIM = None
RANK = None
PARAMS = None
COMPS = None
EXPECTED_ROWS = None
TRANSCENDENTAL_COMPS = None
SIZES = None

def _layout(is_mvn, d, r):
    if is_mvn:
        return (list(range(d)) + [d + i * d + j for i in range(d) for j in range(i + 1)], d + d * d)
    return (list(range(d + d * r + d)), d + d * r + d)

def _comps(kind, d, r):
    out = ['KL']
    for side in ('P', 'Q'):
        out += [f'{side}.loc[{i}]' for i in range(d)]
        if kind == 'mvn':
            out += [f'{side}.scale_tril[{i * d + j}]' for i in range(d) for j in range(i + 1)]
        else:
            out += [f'{side}.cov_factor[{i}]' for i in range(d * r)] + [f'{side}.cov_diag[{i}]' for i in range(d)]
    return out
_ALGOS = {'mvn': _algo_mvn, 'lowrank': _algo_lowrank}

def configure(sub):
    global NAME, KIND, DIM, RANK, PARAMS, COMPS, EXPECTED_ROWS, TRANSCENDENTAL_COMPS, SIZES, OFFICIAL_FN
    KIND, DIM, RANK, n32 = SUBCONFIGS[sub]
    NAME = sub
    PARAMS = ('loc', 'scale_tril') if KIND == 'mvn' else ('loc', 'cov_factor', 'cov_diag')
    SIZES = (DIM, DIM * DIM) if KIND == 'mvn' else (DIM, DIM * RANK, DIM)
    COMPS = _comps(KIND, DIM, RANK)
    TRANSCENDENTAL_COMPS = ('KL',)
    EXPECTED_ROWS = {'fp16': 0, 'fp32': n32}
    OFFICIAL_FN = '_kl_multivariatenormal_multivariatenormal' if KIND == 'mvn' else '_kl_lowrankmultivariatenormal_lowrankmultivariatenormal'
_FCACHE = {}

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    key = (KIND, DIM, RANK)
    if key not in _FCACHE:
        _FCACHE[key] = _make(_ALGOS[KIND], DIM, RANK)
    A = torch.cat([to_t(P[k], dt_in).reshape(-1, s) for k, s in zip(PARAMS, SIZES)], dim=1).to(TD[cdt]).requires_grad_(True)
    C = torch.cat([to_t(Q[k], dt_in).reshape(-1, s) for k, s in zip(PARAMS, SIZES)], dim=1).to(TD[cdt]).requires_grad_(True)
    kl = _FCACHE[key].apply(A, C)
    kl.sum().backward()
    out = {'KL': kl.detach()}
    off = dict(zip(PARAMS, np.cumsum((0,) + SIZES[:-1])))
    for comp in COMPS[1:]:
        side, rest = comp.split('.')
        name, i = rest[:-1].split('[')
        out[comp] = (A if side == 'P' else C).grad[:, int(off[name]) + int(i)].detach()
    return out

def exact_grads(*ex):
    return None

def identity(*ex):
    h = len(ex) // 2
    return list(ex[:h]) == list(ex[h:])

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
        print(f'fix_mvn.running_bounds: {dt} lambda_pos = {_LAM_POS[dt]}; the bound uses max(lambda from p2_accept, lambda_pos)', flush=True)
    return _LAM_POS[dt]

def _trace(sub, vals, dt, lam):
    kind, d, r, _ = SUBCONFIGS[sub]
    o = _ROps(dt, lam)
    xs = [o.inp(x) for x in vals]
    h = len(xs) // 2
    names = _comps(kind, d, r)
    if [float(x[0]) for x in xs[:h]] == [float(x[0]) for x in xs[h:]]:
        z = (o.T(0), 0.0)
        return (dict(zip(names, [z] * len(names))), {'_branch': 'id'})
    if kind == 'lowrank' and _sem_same_exact([x[0] for x in xs], d, r, dt):
        z = (o.T(0), 0.0)
        return (dict(zip(names, [z] * len(names))), {'_branch': 'id_sem'})
    res = _ALGOS[kind](o, xs[:h], xs[h:], dt, d, r)
    return (dict(zip(names, [o.pz(x) for x in res])), {'_branch': kind})
_CACHE = {}

def replay(*args, sub=None, tgt=None):
    *vals, dt = args
    lamc = max(3, _LAM_POS.get(dt, 3))
    sub = sub or NAME
    out, flags = _trace(sub, vals, dt, lamc)
    res = {k: float(v[0]) for k, v in out.items()}
    res.update(flags)
    res['_in'] = tuple((float(x) for x in vals))
    res['_sub'] = sub
    _CACHE.clear()
    _CACHE[res['_in'], dt, lamc] = out
    return res

def running_bounds(v, dt, lam, tgt=None):
    L = max(lam, lam_pos(dt))
    out = _CACHE.get((v['_in'], dt, L))
    if out is None:
        out, _ = _trace(v['_sub'], list(v['_in']), dt, L)
    return {k: math.inf if math.isnan(float(e)) else SAFETY * float(e) for k, (_, e) in out.items()}
