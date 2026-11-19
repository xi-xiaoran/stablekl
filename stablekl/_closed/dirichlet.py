"""Repaired KL divergence between two Dirichlet distributions: formula, analytic gradients, and the
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
NQ3 = {'fp16': 5, 'fp32': 7, 'fp64': 13}
NQK = {'fp16': 3, 'fp32': 6, 'fp64': 10}
QTAIL = {'fp16': 1e-05, 'fp32': 1e-09, 'fp64': 1e-17}
ABG = {'fp16': 3, 'fp32': 5, 'fp64': 10}
KAS = {'fp16': 7, 'fp32': 9, 'fp64': 12}
MH = {'fp16': 9, 'fp32': 16, 'fp64': 29}
T_NEAR = 1.0
SZ = 12
NSMALL = 2.0 ** (-8)
OFFICIAL_FN = '_kl_dirichlet_dirichlet'
GRAD_TRUTH = 'frozen_interval'
_BERN = (Fr(1, 6), Fr(-1, 30), Fr(1, 42), Fr(-1, 30), Fr(5, 66), Fr(-691, 2730), Fr(7, 6), Fr(-3617, 510), Fr(43867, 798), Fr(-174611, 330), Fr(854513, 138), Fr(-236364091, 2730), Fr(8553103, 6), Fr(-23749461029, 870))
G_AS = _BERN
H3_AS = tuple((-(2 * k + 3) * _BERN[k] for k in range(14)))
W_AS = tuple((_BERN[k - 1] / (2 * k * (2 * k - 1)) for k in range(1, 15)))
WP_AS = tuple((_BERN[k - 1] / (2 * k) for k in range(1, 15)))
H_CO = tuple((Fr(1, 2 * m + 1) for m in range(1, 40)))
HP_CO = tuple((Fr(m, 2 * m + 1) for m in range(1, 40)))
LN2_FR = Fr(Decimal('0.693147180559945309417232121458176568075500134'))
GLN = {7: (('0.949107912342758486268223805382149294018745422363', '0.741531185599394460083999547350686043500900268555', '0.405845151377397184155881859624059870839118957520', '0.000000000000000000000000000000000000000000000000', '-0.405845151377397184155881859624059870839118957520', '-0.741531185599394460083999547350686043500900268555', '-0.949107912342758486268223805382149294018745422363'), ('0.129484966168869702896060402963485103100538253784', '0.279705391489276644634287549706641584634780883789', '0.381830050505118923087621851664152927696704864502', '0.417959183673469403252909160073613747954368591309', '0.381830050505118923087621851664152927696704864502', '0.279705391489276644634287549706641584634780883789', '0.129484966168869702896060402963485103100538253784')), 13: (('0.984183054718588135045820308732800185680389404297', '0.917598399222977922917721116391476243734359741211', '0.801578090733309878146428673062473535537719726562', '0.642349339440340227902481728960992768406867980957', '0.448492751036446868351248440376366488635540008545', '0.230458315955134801500392427442420739680528640747', '0.000000000000000000000000000000000000000000000000', '-0.230458315955134801500392427442420739680528640747', '-0.448492751036446868351248440376366488635540008545', '-0.642349339440340227902481728960992768406867980957', '-0.801578090733309878146428673062473535537719726562', '-0.917598399222977922917721116391476243734359741211', '-0.984183054718588135045820308732800185680389404297'), ('0.040484004765315877161224733526978525333106517792', '0.092121499837728451631768677998479688540101051331', '0.138873510219787249519995953050965908914804458618', '0.178145980761945738057860921799147035926580429077', '0.207816047536888509617014619834662880748510360718', '0.226283180262897232193353147522429935634136199951', '0.232551553230873897515351700349128805100917816162', '0.226283180262897232193353147522429935634136199951', '0.207816047536888509617014619834662880748510360718', '0.178145980761945738057860921799147035926580429077', '0.138873510219787249519995953050965908914804458618', '0.092121499837728451631768677998479688540101051331', '0.040484004765315877161224733526978525333106517792')), 3: (('0.774596669241483377035853079956479922166584341059', '0.000000000000000000000000000000000000000000000000', '-0.774596669241483377035853079956479922166584341059'), ('0.555555555555555555555555555555555555555555555556', '0.888888888888888888888888888888888888888888888889', '0.555555555555555555555555555555555555555555555556')), 4: (('0.861136311594052575223946488892809505095725379629', '0.339981043584856264802665759103244687200575869770', '-0.339981043584856264802665759103244687200575869770', '-0.861136311594052575223946488892809505095725379629'), ('0.347854845137453857373063949221999407410889836016', '0.652145154862546142626936050778000592764910163984', '0.652145154862546142626936050778000592764910163984', '0.347854845137453857373063949221999407410889836016')), 5: (('0.906179845938663992797626878299392965125651910762', '0.538469310105683091036314420700208804967286606905', '0.000000000000000000000000000000000000000000000000', '-0.538469310105683091036314420700208804967286606905', '-0.906179845938663992797626878299392965125651910762'), ('0.236926885056189087514264040719917362643260002300', '0.478628670499366468041291514835638192912295553254', '0.568888888888888888888888888888888888888888888889', '0.478628670499366468041291514835638192912295553254', '0.236926885056189087514264040719917362643260002300')), 6: (('0.932469514203152027812301554493994609134765737713', '0.661209386466264513661399595019905347006448564396', '0.238619186083196908630501721680711935418610630141', '-0.238619186083196908630501721680711935418610630141', '-0.661209386466264513661399595019905347006448564396', '-0.932469514203152027812301554493994609134765737713'), ('0.171324492379170345040296142172732893526822501484', '0.360761573048138607569833513837716111661521892747', '0.467913934572691047389870343989550994811655605769', '0.467913934572691047389870343989550994811655605769', '0.360761573048138607569833513837716111661521892747', '0.171324492379170345040296142172732893526822501484')), 8: (('0.960289856497536231683560868569472990428235234302', '0.796666477413626739591553936475830436837171731615', '0.525532409916328985817739049189246349041964243949', '0.183434642495649804939476142360183980666757812912', '-0.183434642495649804939476142360183980666757812912', '-0.525532409916328985817739049189246349041964243949', '-0.796666477413626739591553936475830436837171731615', '-0.960289856497536231683560868569472990428235234302'), ('0.101228536290376259152531354309962190115394091052', '0.222381034453374470544355994426240884430130870051', '0.313706645877887287337962201986601313260328999002', '0.362683783378361982965150449277195612194146039895', '0.362683783378361982965150449277195612194146039895', '0.313706645877887287337962201986601313260328999002', '0.222381034453374470544355994426240884430130870051', '0.101228536290376259152531354309962190115394091052')), 10: (('0.973906528517171720077964012084452053428269946693', '0.865063366688984510732096688423493048527543014965', '0.679409568299024406234327365114873575769294711835', '0.433395394129247190799265943165784162200071837656', '0.148874338981631210884826001129719984617564859420', '-0.148874338981631210884826001129719984617564859420', '-0.433395394129247190799265943165784162200071837656', '-0.679409568299024406234327365114873575769294711835', '-0.865063366688984510732096688423493048527543014965', '-0.973906528517171720077964012084452053428269946693'), ('0.066671344308688137593568809893331792857864834320', '0.149451349150580593145776339657697332402556639669', '0.219086362515982043995534934228163192458771870523', '0.269266719309996355091226921569469352859759938461', '0.295524224714752870173892994651338329421046717026', '0.295524224714752870173892994651338329421046717026', '0.269266719309996355091226921569469352859759938461', '0.219086362515982043995534934228163192458771870523', '0.149451349150580593145776339657697332402556639669', '0.066671344308688137593568809893331792857864834320'))}
GLU = {N: tuple((((1 + Fr(Decimal(x))) / 2, Fr(Decimal(w)) / 2, (1 - Fr(Decimal(x))) / 2) for x, w in zip(*GLN[N]))) for N in GLN}

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
        self.h = f['eta'] / 2
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

def _log1p_mf(o, rf, lr):
    r = o.ld2(rf[0], rf[1])

    def big():
        mm, ee = o.frexp(rf[0])
        return o.add(o.fn('log', mm), o.mul(_ie(o, rf[1] + ee), o.const(LN2_FR)))
    return o.where(o.lt(r, -0.5), lambda: lr, lambda: o.where(o.isfinite(r), lambda: o.fn('log1p', r), big))

def _ie(o, e):
    if o.trace:
        return o.const(int(e))
    return e.to(o.ref.dtype)

def _cphi_mf(o, coef, rf, lr, dt):
    r = o.ld2(rf[0], rf[1])
    return o.where(o.isfinite(r), lambda: _phi(o, coef, r, dt, lx=lr), lambda: o.sub(mprod(o, [coef, rf]), o.mul(coef, _log1p_mf(o, rf, lr))))

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

def _shift(o, s, AB):
    n = o.const(0)
    masks = []
    for j in range(AB):
        m = o.lt(s, AB - j)
        masks.append(m)
        n = o.where(m, lambda: o.add(n, o.const(1)), lambda: n)
    return (n, masks)

def _g(o, s, dt):
    AB = ABG[dt]
    K = KAS[dt]
    one = o.const(1)
    half = o.const(Fr(1, 2))
    n, masks = _shift(o, s, AB)
    acc = o.const(0)
    for j in range(AB):

        def term(j=j):
            t = o.add(s, o.const(j))
            q = o.div(s, t)
            t1 = o.add(t, one)
            return o.add(acc, o.div(o.mul(half, o.mul(q, q)), o.mul(t1, t1)))
        acc = o.where(masks[j], term, lambda: acc)
    y = o.add(s, n)
    iy = o.div(one, y)
    iy2 = o.mul(iy, iy)
    ga = o.mul(iy, _horner(o, iy2, G_AS[:K]))
    if o.trace:
        ga = o.extra(ga, lambda: abs(float(G_AS[K])) * float(y[0]) ** (-(2 * K + 1)))
    r = o.div(s, y)
    return o.add(acc, o.mul(o.mul(r, r), ga))

def _h3(o, s, dt):
    AB = ABG[dt]
    K = KAS[dt]
    one = o.const(1)
    n, masks = _shift(o, s, AB)
    acc = o.const(0)
    for j in range(AB):

        def term(j=j):
            A = o.add(s, o.const(j))
            B = o.add(A, one)
            qa = o.div(s, A)
            qb = o.div(s, B)
            return o.sub(acc, o.div(o.mul(o.mul(qa, qa), o.add(qa, qb)), o.mul(B, B)))
        acc = o.where(masks[j], term, lambda: acc)
    y = o.add(s, n)
    iy = o.div(one, y)
    iy2 = o.mul(iy, iy)
    ha = o.mul(iy, _horner(o, iy2, H3_AS[:K]))
    if o.trace:
        ha = o.extra(ha, lambda: abs(float(H3_AS[K])) * float(y[0]) ** (-(2 * K + 1)))
    r = o.div(s, y)
    return o.add(acc, o.mul(o.mul(o.mul(r, r), r), ha))

def _wpair(o, a, ia, dt):
    AB = ABG[dt]
    K = KAS[dt]
    M = MH[dt]
    one = o.const(1)
    half = o.const(Fr(1, 2))
    zero = o.const(0)
    n, masks = _shift(o, a, AB)
    W = zero
    Wp = zero
    V = zero
    Vp = zero
    for j in range(AB):

        def hterms(j=j):
            t = o.add(a, o.const(j))

            def ser():
                u = o.div(one, o.add(o.add(t, t), one))
                u2 = o.mul(u, u)
                hs = o.mul(u2, _horner(o, u2, H_CO[:M]))
                hps = o.mul(o.mul(o.const(-4), o.mul(u, u2)), _horner(o, u2, HP_CO[:M]))
                if o.trace:
                    hs = o.extra(hs, lambda: float(u2[0]) ** (M + 1) / ((2 * M + 3) * (1 - float(u2[0]))))
                    hps = o.extra(hps, lambda: 2 * float(u[0]) ** (2 * M + 3) / (1 - float(u2[0])))
                return (hs, hps)

            def dire():
                lg = o.sub(o.fn('log1p', t), o.fn('log', t))
                return (o.sub(o.mul(o.add(t, half), lg), one), o.sub(o.sub(lg, o.div(half, t)), o.div(half, o.add(t, one))))
            if o.trace:
                return dire() if float(t[0]) < 0.5 else ser()
            m = t < 0.5
            d1, d2 = dire()
            s1, s2 = ser()
            return (o.t.where(m, d1, s1), o.t.where(m, d2, s2))

        def addw(j=j):
            hj, hpj = hterms()
            return (o.add(W, hj), o.add(Wp, hpj))
        if o.trace:
            if masks[j]:
                W, Wp = addw()
        else:
            hj, hpj = hterms()
            W = o.t.where(masks[j], W + hj, W)
            Wp = o.t.where(masks[j], Wp + hpj, Wp)
        if j == 0:
            sm = o.lt(a, 0.5)
            V = o.where(sm, lambda: o.sub(o.sub(o.mul(o.add(a, half), o.fn('log1p', a)), o.mul(a, o.fn('log', a))), one), lambda: zero)
            Vp = o.where(sm, lambda: o.sub(o.sub(o.fn('log1p', a), o.fn('log', a)), o.div(half, o.add(a, one))), lambda: zero)
        elif o.trace:
            if masks[j]:
                hj, hpj = hterms()
                V = o.add(V, hj)
                Vp = o.add(Vp, hpj)
        else:
            V = o.t.where(masks[j], V + hj, V)
            Vp = o.t.where(masks[j], Vp + hpj, Vp)
    y = o.add(a, n)
    iy = o.where(o.eq(n, 0), lambda: ia, lambda: o.div(one, y))
    iy2 = o.mul(iy, iy)
    wa = o.mul(iy, _horner(o, iy2, W_AS[:K]))
    wpa = o.mul(o.neg(iy2), _horner(o, iy2, WP_AS[:K]))
    if o.trace:
        yv = 1 / float(iy[0]) if float(iy[0]) != 0 else math.inf
        if yv == 0:
            yv = 1e-300
        wa = o.extra(wa, lambda: abs(float(W_AS[K])) * yv ** (-(2 * K + 1)))
        wpa = o.extra(wpa, lambda: abs(float(WP_AS[K])) * yv ** (-(2 * K + 2)))
    return (o.add(W, wa), o.add(Wp, wpa), o.add(V, wa), o.add(Vp, wpa))

def _xdK(o, x, y, hx, hs, xs, dt):

    def direct():
        return o.sub(hx, o.mul(xs, hs))

    def quad():
        Tk = o.fn('log1p', o.div(y, x))
        N = NQK[dt]
        J = o.const(0)
        for ui, vi, _ in GLU[N]:
            tau = o.mul(Tk, o.const(ui))
            t = o.mul(x, o.fn('exp', tau))
            f = o.sub(o.sub(o.const(Fr(1, 2)), _g(o, t, dt)), _h3(o, t, dt))
            J = o.add(J, o.mul(o.const(vi), o.mul(f, o.fn('exp', o.neg(tau)))))
        if o.trace:
            J = o.extra(J, lambda: QTAIL[dt] * abs(float(J[0])) * 1.5)
        mT, eT = o.frexp(Tk)
        return (o.mul(mT, J), eT)
    if o.trace:
        if float(x[0]) <= float(y[0]):
            o.tag('Kd')
            return [direct(), 0]
        o.tag('Kq')
        m, e = quad()
        return [m, e]
    near = x <= y
    m, e = quad()
    d = direct()
    return [o.t.where(near, d, m), o.t.where(near, o.iz, e)]
E1N = {'fp16': 4, 'fp32': 5, 'fp64': 9}

def _E1(o, x, dt):
    n = E1N[dt]

    def ser():
        p = _horner(o, x, [Fr(1, math.factorial(k + 1)) for k in range(n)])
        if o.trace:
            p = o.extra(p, lambda: abs(float(x[0])) ** n / math.factorial(n + 1) * 1.1)
        return p
    return o.where(o.lt(o.abs(x), 2.0 ** (-6)), ser, lambda: o.div(o.fn('expm1', x), x))
T_PANEL = {2: 2.0, 4: 4.0}

def _Rpanels(o, y, T, P, dt):
    one = o.const(1)
    IR = o.const(0)
    N = NQ3[dt]
    for k in range(P):
        for ui, vi, wi in GLU[N]:
            uu = (k + ui) / P
            om = (P - k - 1 + wi) / P
            tau = o.mul(T, o.const(uu))
            arg = o.mul(T, o.const(om))
            E1 = _E1(o, arg, dt)
            hm = o.sub(_h3(o, o.mul(y, o.fn('exp', tau)), dt), one)
            IR = o.add(IR, o.mul(o.mul(o.const(vi / P * om), o.fn('exp', o.neg(tau))), o.mul(E1, hm)))
    if o.trace:
        IR = o.extra(IR, lambda: QTAIL[dt] * abs(float(IR[0])) * 1.5)
    mT, eT = o.frexp(T)
    return mprod(o, [[o.mul(mT, mT), 2 * eT], IR], [y])

def _dpsi1(o, y, dlt, dt):
    one = o.const(1)
    acc = o.const(0)
    N = NQ3[dt]
    for ui, vi, _ in GLU[N]:
        z = o.add(one, o.add(y, o.mul(dlt, o.const(ui))))
        ps1 = o.add(o.div(o.add(o.const(Fr(1, 2)), _g(o, z, dt)), o.mul(z, z)), o.div(one, z))
        acc = o.add(acc, o.mul(o.const(vi), ps1))
    if o.trace:
        acc = o.extra(acc, lambda: QTAIL[dt] * abs(float(acc[0])) * 1.5)
    return o.mul(dlt, acc)

def _pair(o, y, dltf, xin, hy, dt):
    one = o.const(1)
    half = o.const(Fr(1, 2))
    zero = o.const(0)
    dm, de = dltf
    sc = de != 0 if o.trace else o._i(de) != 0
    r = o.where(sc, lambda: o.ld2(o.div(dm, y), de), lambda: o.div(dm, y))
    dWp = lambda Wp_: o.where(sc, lambda: o.ld2(o.mul(dm, Wp_), de), lambda: o.mul(dm, Wp_))
    x = xin
    T = o.where(o.lor(o.lt(r, -0.5), o.lnot(o.isfinite(r))), lambda: o.sub(o.fn('log', x), o.fn('log', y)), lambda: o.fn('log1p', r))
    near = o.le(o.abs(T), o.const(T_NEAR))

    def nearf():
        N = NQ3[dt]
        I3 = zero
        IR = zero
        for ui, vi, wi in GLU[N]:
            tau = o.mul(T, o.const(ui))
            arg = o.mul(T, o.const(wi))
            E1 = _E1(o, arg, dt)
            hm = o.sub(_h3(o, o.mul(y, o.fn('exp', tau)), dt), one)
            cw = o.mul(o.const(vi), o.const(wi))
            I3 = o.add(I3, o.mul(o.mul(cw, o.const(wi)), o.mul(o.mul(E1, E1), hm)))
            IR = o.add(IR, o.mul(o.mul(cw, o.fn('exp', o.neg(tau))), o.mul(E1, hm)))
        if o.trace:
            I3 = o.extra(I3, lambda: QTAIL[dt] * abs(float(I3[0])) * 1.5)
            IR = o.extra(IR, lambda: QTAIL[dt] * abs(float(IR[0])) * 1.5)
        mT, eT = o.frexp(T)
        T3 = mprod(o, [[o.mul(o.mul(o.mul(mT, mT), mT), o.mul(half, I3)), 3 * eT]])
        R = mprod(o, [[o.mul(mT, mT), 2 * eT], IR], [y])
        Bv = o.add(T3, o.mul(o.mul(o.mul(half, r), r), hy))
        dL = o.add(R, o.div(o.mul(r, hy), y))
        return (T3, R, Bv, dL)

    def farf():
        ix = o.where(o.isfinite(x), lambda: o.div(one, x), lambda: o.div(o.div(one, y), o.add(one, r)))
        Wx, Wpx, Ux, Upx = _wpair(o, x, ix, dt)
        Wy, Wpy, Uy, Upy = _wpair(o, y, o.div(one, y), dt)
        uvar = o.land(o.lt(x, 0.5), o.lt(y, 0.5))

        def wv():
            Bv = o.add(_phi(o, half, r, dt, lx=T), o.sub(o.sub(Wx, Wy), dWp(Wpy)))
            dL = o.add(o.sub(Wpx, Wpy), o.sub(o.div(half, y), o.mul(half, ix)))
            return (Bv, dL)

        def uv():
            Bv = o.add(_phi(o, one, r, dt, lx=T), o.sub(o.sub(Ux, Uy), dWp(Upy)))
            dL = o.add(o.sub(Upx, Upy), o.sub(o.div(one, y), ix))
            return (Bv, dL)
        Bv, dL = o.wheret(uvar, uv, wv)
        T3 = o.sub(Bv, o.mul(o.mul(o.mul(half, r), r), hy))
        R = o.sub(dL, o.div(o.mul(r, hy), y))
        return (T3, R, Bv, dL)
    aT = o.abs(T)
    if o.trace:
        if float(T[0]) == 0:
            o.tag('0')
            return (zero, zero, zero, zero, True, T, True)
        if near:
            o.tag('N')
            return nearf() + (True, T, True)
        T3f, Rf, Bvf, dLf = farf()
        ok4 = False
        for P in (2, 4):
            if float(aT[0]) <= T_PANEL[P]:
                o.tag('P%d' % P)
                Rf = _Rpanels(o, y, T, P, dt)
                ok4 = True
                break
        else:
            o.tag('F')
        return (T3f, Rf, Bvf, dLf, False, T, ok4)
    nr = nearf()
    fr = list(farf())
    R2 = _Rpanels(o, y, T, 2, dt)
    R4 = _Rpanels(o, y, T, 4, dt)
    fr[1] = o.t.where(aT <= T_PANEL[2], R2, o.t.where(aT <= T_PANEL[4], R4, fr[1]))
    z0 = T == 0
    res = tuple((o.t.where(z0, zero, o.t.where(near, x1, x2)) for x1, x2 in zip(nr, fr)))
    return res + (near | z0, T, (aT <= T_PANEL[4]) | z0)

def _pair_bigs(o, sg, sq16, dltf, hy, dt):
    one = o.const(1)
    half = o.const(Fr(1, 2))
    zero = o.const(0)
    K = KAS[dt]
    dm, de = dltf
    r = o.ld2(o.div(dm, sg), de - 4)
    T = o.where(o.lt(r, -0.5), lambda: o.sub(o.fn('log', sq16), o.fn('log', sg)), lambda: o.fn('log1p', r))
    near = o.le(o.abs(T), o.const(T_NEAR))
    ms, es = o.frexp(sg)
    yF = [ms, es + 4]

    def h3big(t16):
        it = o.ld2(o.div(one, t16), -4)
        ha = o.mul(it, _horner(o, o.mul(it, it), H3_AS[:K]))
        if o.trace:
            ha = o.extra(ha, lambda: abs(float(H3_AS[K])) * (16 * float(t16[0])) ** (-(2 * K + 1)))
        return ha
    N = NQ3[dt]
    I3 = zero
    IR = zero
    for ui, vi, wi in GLU[N]:
        tau = o.mul(T, o.const(ui))
        arg = o.mul(T, o.const(wi))
        E1 = _E1(o, arg, dt)
        hm = o.sub(h3big(o.mul(sg, o.fn('exp', tau))), one)
        cw = o.mul(o.const(vi), o.const(wi))
        I3 = o.add(I3, o.mul(o.mul(cw, o.const(wi)), o.mul(o.mul(E1, E1), hm)))
        IR = o.add(IR, o.mul(o.mul(cw, o.fn('exp', o.neg(tau))), o.mul(E1, hm)))
    if o.trace:
        I3 = o.extra(I3, lambda: QTAIL[dt] * abs(float(I3[0])) * 1.5)
        IR = o.extra(IR, lambda: QTAIL[dt] * abs(float(IR[0])) * 1.5)
    mT, eT = o.frexp(T)
    T3 = mprod(o, [[o.mul(o.mul(o.mul(mT, mT), mT), o.mul(half, I3)), 3 * eT]])
    R = mprod(o, [[o.mul(mT, mT), 2 * eT], IR], [yF])
    Bv = o.add(T3, o.mul(o.mul(o.mul(half, r), r), hy))
    dL = o.add(R, mprod(o, [r, hy], [yF]))
    if o.trace:
        if float(T[0]) == 0:
            o.tag('0')
            return (zero, zero, zero, zero, True, T, True)
        o.tag('Nb' if near else 'Fb')
        return (T3, R, Bv, dL, near, T, near)
    z0 = T == 0
    res = tuple((o.t.where(z0, zero, x) for x in (T3, R, Bv, dL)))
    return res + (near | z0, T, near | z0)
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

def _dd_ratio(o, nh, nl, dh_, dl_, shift=0):
    mn, en = o.frexp(nh)
    md, ed = o.frexp(dh_)
    rho = o.seterr(o.div(mn, md), lambda: 0.0)
    o.exact = 1 if o.trace else 0
    p, e = o.two_prod(rho, md)
    rem = o.sub(mn, p)
    o.exact = 0
    rem = o.sub(rem, e)
    lo = o.div(o.sub(o.add(rem, o.ld2(nl, 0 - en)), o.mul(rho, o.ld2(dl_, 0 - ed))), md)
    return (rho, lo, o.iwhere(o.eq(mn, 0), SENT, en - ed + shift))

def _ratio_f(o, num, den, shift=0):
    mn, en = o.frexp(num)
    md, ed = o.frexp(den)
    return [o.div(mn, md), o.iwhere(o.eq(mn, 0), SENT, en - ed + shift)]

def _sub_f(o, xf, yf):
    E = o.imax(xf[1], yf[1])
    return [o.sub(o.ld2(xf[0], xf[1] - E), o.ld2(yf[0], yf[1] - E)), E]

def _allpairs_lt8(o, xs):
    c = None
    eight = o.const(8)
    for i, xi in enumerate(xs):
        for j, xj in enumerate(xs):
            if i == j:
                continue
            t = o.lt(xi, o.mul(eight, xj))
            c = t if c is None else o.land(c, t)
    return c

def _algo(o, A, C, dt):
    K_ = len(A)
    one = o.const(1)
    zero = o.const(0)
    half = o.const(Fr(1, 2))
    o.exact = 1 if o.trace else 0
    Es = _expansion(o, A)
    Eq = _expansion(o, C)
    Eoth = [_grow(o, Es, o.neg(a)) for a in A]
    HL = [o.two_sum(c, o.neg(a)) for a, c in zip(A, C)]
    HLv = [x for hl in HL for x in hl]
    ED = _expansion(o, HLv)
    o.exact = 0
    sqE = _esum(o, Eq)

    def sq_naive():
        acc = C[0]
        for c in C[1:]:
            acc = o.add(acc, c)
        return acc
    sq = o.where(o.isfinite(sqE), lambda: sqE, sq_naive)
    oth = [_esum(o, E) for E in Eoth]
    s_hi, s_lo = _hilo(o, Es)
    D_hi, D_lo = _hilo(o, ED)
    s = o.seterr(s_hi, lambda: abs(float(s_lo[0])) + s_lo[1])
    D = o.seterr(D_hi, lambda: abs(float(D_lo[0])) + D_lo[1])

    def Dhalf():
        hv = [o.ld2(x, -1) for x in HLv]
        o.exact = 1 if o.trace else 0
        E2 = _expansion(o, hv)
        o.exact = 0
        hi, lo = _hilo(o, E2)
        return (o.seterr(hi, lambda: abs(float(lo[0])) + lo[1]), lo)
    if o.trace:
        if math.isfinite(float(D[0])):
            Df = [D, 0]
            Dl = D_lo
        else:
            o.tag('Dh')
            dh_, Dl = Dhalf()
            Df = [dh_, 1]
    else:
        fD = o.t.isfinite(D)
        dh_, dl_ = Dhalf()
        Df = [o.t.where(fD, D, dh_), o.t.where(fD, o.iz, o.iz + 1)]
        Dl = o.t.where(fD, D_lo, dl_)
    H = [o.seterr(hl[0], lambda hl=hl: abs(float(hl[1][0]))) for hl in HL]
    big = not math.isfinite(float(s_hi[0])) if o.trace else ~o.t.isfinite(s_hi)

    def pick(fb, fn):
        if o.trace:
            return fb() if big else fn()
        b_, n_ = (fb(), fn())
        if isinstance(b_, (tuple, list)):
            return type(b_)((o.t.where(big, x, y) for x, y in zip(b_, n_)))
        return o.t.where(big, b_, n_)
    if not o.trace or big:
        o.tag('sbig') if o.trace else None
        a16 = [o.ld2(a, -4) for a in A]
        c16 = [o.ld2(c, -4) for c in C]
        o.exact = 1 if o.trace else 0
        E16 = _expansion(o, a16)
        Eq16 = _expansion(o, c16)
        Eo16 = [_grow(o, E16, o.neg(x)) for x in a16]
        o.exact = 0
        sg_hi, sg_lo = _hilo(o, E16)
        sq16 = _esum(o, Eq16)
        sg = o.seterr(sg_hi, lambda: abs(float(sg_lo[0])) + sg_lo[1])
        oth_b = [o.ld2(_esum(o, E), 4) for E in Eo16]
        isg = o.ld2(o.div(one, sg), -4)
    oth = [pick(lambda j=j: oth_b[j], lambda j=j: oth[j]) for j in range(K_)]

    def sF_normal():
        m_, e_ = o.frexp(s)
        return (m_, e_)

    def sF_big():
        m_, e_ = o.frexp(sg)
        return (m_, e_ + 4)
    sF = list(pick(sF_big, sF_normal))
    ga_r, ga_l, ga_E = pick(lambda: _dd_ratio(o, o.seterr(Df[0], lambda: 0.0), Dl, o.seterr(sg_hi, lambda: 0.0), sg_lo, Df[1] - 4), lambda: _dd_ratio(o, o.seterr(Df[0], lambda: 0.0), Dl, o.seterr(s_hi, lambda: 0.0), s_lo, Df[1]))
    af, dfl = ([], [])
    for hl, a in zip(HL, A):
        ar, al_, aE = _dd_ratio(o, hl[0], hl[1], a, zero)
        af.append([o.add(ar, al_), aE])
        E_ = o.imax(aE, ga_E)
        x1 = o.ld2(ar, aE - E_)
        y1 = o.ld2(ga_r, ga_E - E_)
        o.exact = 1 if o.trace else 0
        dh, de = o.two_sum(x1, o.neg(y1))
        o.exact = 0
        dlo = o.sub(o.ld2(al_, aE - E_), o.ld2(ga_l, ga_E - E_))
        dfl.append([o.add(dh, o.add(de, dlo)), E_])
    MA = [o.frexp(a) for a in A]
    MC = [o.frexp(c) for c in C]
    rf = [[o.neg(o.mul(o.div(ma[0], mc[0]), d_[0])), ma[1] - mc[1] + d_[1]] for ma, mc, d_ in zip(MA, MC, dfl)]
    rv = [o.ld2(x[0], x[1]) for x in rf]

    def g_big():
        ga = o.mul(isg, _horner(o, o.mul(isg, isg), G_AS[:KAS[dt]]))
        if o.trace:
            ga = o.extra(ga, lambda: abs(float(G_AS[KAS[dt]])) * (16 * float(sg[0])) ** (-(2 * KAS[dt] + 1)))
        return ga
    hs = pick(lambda: o.add(half, g_big()), lambda: o.add(half, _g(o, s, dt)))
    KS = pick(lambda: o.ld2(o.div(hs, sg), -4), lambda: o.div(hs, s))
    K1 = o.add(one, KS)
    HG = [o.add(half, _g(o, a, dt)) for a in A]
    XS = [pick(lambda a=a: o.ld2(o.div(a, sg), -4), lambda a=a: o.div(a, s)) for a in A]
    XD = [_xdK(o, a, y, hg, hs, xs, dt) for a, y, hg, xs in zip(A, oth, HG, XS)]
    PR = [_pair(o, a, [h, 0], c, hg, dt) for a, h, c, hg in zip(A, H, C, HG)]
    T3s, Rs, Bvs, dLs, ns, Ts, ks = pick(lambda: _pair_bigs(o, sg, sq16, Df, hs, dt), lambda: _pair(o, s, Df, sq, hs, dt))
    alln = ns
    ok4 = ks
    for p_ in PR:
        alln = o.land(alln, p_[4])
        ok4 = o.land(ok4, p_[6])
    LR = [o.sub(Ts, p_[5]) for p_ in PR]
    if o.trace:
        o.tag('S2' if alln else 'S1')

    def p1term(c, r_f, r_v, lr):
        seg = o.land(o.le(o.const(SEG_LO), r_v), o.le(r_v, o.const(SEG_HI)))
        return o.where(seg, lambda: mprod(o, [c, r_f, r_f, _phh(o, r_v, dt)]), lambda: _cphi_mf(o, c, r_f, lr, dt))
    P1 = None
    for c, r_f, r_v, lr in zip(C, rf, rv, LR):
        t = p1term(c, r_f, r_v, lr)
        P1 = t if P1 is None else o.add(P1, t)

    def second():
        Q = None
        for xd, a_f in zip(XD, af):
            t = mprod(o, [xd, a_f, a_f])
            Q = t if Q is None else o.add(Q, t)
        for a, d_ in zip(A, dfl):
            Q = o.add(Q, mprod(o, [KS, a, d_, d_]))
        T3 = None
        for p_ in PR:
            T3 = p_[0] if T3 is None else o.add(T3, p_[0])
        return o.add(o.add(o.clamp0(P1), o.mul(half, o.clamp0(Q))), o.sub(T3, T3s))

    def first():
        Bv = None
        for p_ in PR:
            Bv = p_[2] if Bv is None else o.add(Bv, p_[2])
        return o.add(o.clamp0(P1), o.sub(Bv, Bvs))
    if o.trace:
        kl = second() if alln else first()
    else:
        kl = o.t.where(alln, second(), first())

    def l1r(r, lr):
        return o.where(o.lt(o.abs(r), NSMALL * 4), lambda: _horner(o, o.neg(r), [Fr(1, k + 1) for k in range(L1N[dt])]), lambda: o.where(o.isfinite(r), lambda: o.div(o.where(o.lt(r, -0.5), lambda: lr, lambda: o.fn('log1p', r)), r), lambda: o.const(0)))

    def rterm(r_f, r_v, lr, mc, ma):
        L = l1r(r_v, lr)
        if o.trace and abs(float(r_v[0])) < NSMALL * 4:
            L = o.extra(L, lambda: l1_tail(float(r_v[0]), L1N[dt]))
        inner = o.add(mprod(o, [[mc[0], mc[1]], KS], [[ma[0], ma[1]]]), L)
        return mprod(o, [r_f, inner])
    A_ = o.abs
    GA, GC, MGA, MGC = ([], [], [], [])
    for j in range(K_):
        ta1 = mprod(o, [XD[j], H[j]], [A[j], A[j]])
        ta2 = mprod(o, [dfl[j], K1])
        rc = rterm(rf[j], rv[j], LR[j], MC[j], MA[j])
        GA.append(o.sub(o.neg(ta1), ta2))
        MGA.append(o.add(A_(ta1), A_(ta2)))
        GC.append(o.add(o.sub(ta1, rc), o.sub(PR[j][1], Rs)))
        MGC.append(o.add(o.add(A_(ta1), A_(rc)), o.add(A_(PR[j][1]), A_(Rs))))

    def oldc(j):
        l1 = _log1p_mf(o, rf[j], LR[j])
        dL = PR[j][3]
        return (o.sub(o.sub(dL, dLs), l1), o.add(o.add(A_(dL), A_(dLs)), A_(l1)))
    u3 = mprod(o, [Df, KS], [sF])

    def olda(j):
        u1 = mprod(o, [C[j], rf[j]], [A[j]])
        u2 = mprod(o, [af[j], HG[j]], [A[j]])
        return (o.add(o.sub(u1, u2), u3), o.add(o.add(A_(u1), A_(u2)), A_(u3)))
    allsmall = None
    for x in list(A) + list(C):
        t = o.lt(x, 0.5)
        allsmall = t if allsmall is None else o.land(allsmall, t)
    allsmall = o.land(o.land(allsmall, _allpairs_lt8(o, A)), _allpairs_lt8(o, C))
    allsmall = o.land(allsmall, o.lnot(alln))

    def smallq():
        ps = _dpsi1(o, s, o.ld2(Df[0], Df[1]), dt)
        out = []
        for j in range(K_):
            pj = _dpsi1(o, A[j], H[j], dt)

            def comb(sh, j=j, pj=pj):
                if sh and o.trace:
                    o.tag('sk')
                sc_ = [o.const(1), -sh]
                r1_ = mprod(o, [H[j], sc_], [A[j], C[j]])
                rs = mprod(o, [Df, sc_], [s, sq])
                pc = o.sub(pj, ps)
                if sh:
                    pc = o.ld2(pc, -sh)
                g = o.add(o.sub(r1_, rs), pc)
                return o.ld2(g, sh) if sh else g
            g0 = comb(0)
            out.append(o.where(o.isfinite(g0), lambda g0=g0: g0, lambda comb=comb: comb(8)))
        return out
    if o.trace:
        sm = smallq() if allsmall else None
    else:
        sm = smallq()
    for j in range(K_):
        gn, mg = (GA[j], MGA[j])
        go, mo = olda(j)
        if o.trace:
            if not math.isfinite(float(gn[0])):
                o.tag('fa')
                gn = go
        else:
            gn = o.t.where(o.t.isfinite(gn), gn, go)
        GA[j] = gn
        gn, mg = (GC[j], MGC[j])
        go, mo = oldc(j)
        if o.trace:
            if not math.isfinite(float(gn[0])) or not ok4 or float(mo[0]) < float(mg[0]):
                o.tag('fc')
                gn = go
            if sm is not None:
                o.tag('sc')
                gn = sm[j]
        else:
            keep = o.t.isfinite(gn) & ok4 & ~(mo < mg)
            gn = o.t.where(keep, gn, go)
            gn = o.t.where(allsmall, sm[j], gn)
        GC[j] = gn
    kl = o.clamp0(kl)
    outs = [kl] + [o.pz(g) for g in GA] + [o.pz(g) for g in GC]
    if o.trace:
        if big and (not ns):
            o.tag('sfar')
            nan = (o.T(np.nan), math.inf)
            return [nan] * len(outs)
        return outs
    sf = ~big | ns
    nanv = o.const(float('nan'))
    return [o.t.where(sf, x, nanv) for x in outs]
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    _DT = {torch.float16: 'fp16', torch.float32: 'fp32', torch.float64: 'fp64'}

    class DirichletKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, a, c):
            dt = _DT[a.dtype]
            K_ = a.shape[1]
            A = [a[:, j] for j in range(K_)]
            C = [c[:, j] for j in range(K_)]
            o = _TOps(torch, A[0], dt)
            outs = _algo(o, A, C, dt)
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
SUBCONFIGS = {'dirichlet_K2': (2, {'fp16': 42446, 'fp32': 67348}), 'dirichlet_K4': (4, {'fp16': 36678, 'fp32': 66669}), 'dirichlet_K16': (16, {'fp16': 34659, 'fp32': 66651})}
PARAMS = ('concentration',)
TRANSCENDENTAL_FUNCS = ('log1p', 'log', 'expm1', 'exp')
NAME = None
KDIM = None
COMPS = None
EXPECTED_ROWS = None
TRANSCENDENTAL_COMPS = None

def configure(sub):
    global NAME, KDIM, COMPS, EXPECTED_ROWS, TRANSCENDENTAL_COMPS
    KDIM, EXPECTED_ROWS = SUBCONFIGS[sub]
    NAME = sub
    COMPS = ['KL'] + [f'P.concentration[{j}]' for j in range(KDIM)] + [f'Q.concentration[{j}]' for j in range(KDIM)]
    TRANSCENDENTAL_COMPS = tuple(COMPS)

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    a = to_t(P['concentration'], dt_in).to(TD[cdt]).requires_grad_(True)
    c = to_t(Q['concentration'], dt_in).to(TD[cdt]).requires_grad_(True)
    kl = DirichletKLC.apply(a, c)
    kl.sum().backward()
    vals = {'KL': kl.detach()}
    for j in range(KDIM):
        vals[f'P.concentration[{j}]'] = a.grad[:, j].detach()
        vals[f'Q.concentration[{j}]'] = c.grad[:, j].detach()
    return vals

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
        print(f'fix_dirichlet.running_bounds: {dt} lambda_pos = {_LAM_POS[dt]}; the bound uses max(lambda from p2_accept, lambda_pos)', flush=True)
    return _LAM_POS[dt]

def _comps(k):
    return ['KL'] + [f'P.concentration[{j}]' for j in range(k)] + [f'Q.concentration[{j}]' for j in range(k)]

def _trace(vals, dt, lam):
    o = _ROps(dt, lam)
    k = len(vals) // 2
    X = [o.inp(x) for x in vals]
    A, C = (X[:k], X[k:])
    names = _comps(k)
    if all((float(a[0]) == float(c[0]) for a, c in zip(A, C))):
        z = (o.T(0), 0.0)
        return (dict(zip(names, [z] * len(names))), {'_branch': 'id'})
    res = _algo(o, A, C, dt)
    return (dict(zip(names, res)), {'_branch': '/'.join(sorted(set(o.tags)))})
_CACHE = {}

def replay(*args):
    *vals, dt = args
    lamc = max(3, _LAM_POS.get(dt, 3))
    out, flags = _trace(vals, dt, lamc)
    res = {k: float(v[0]) for k, v in out.items()}
    res.update(flags)
    res['_in'] = tuple((float(x) for x in vals))
    _CACHE.clear()
    _CACHE[res['_in'], dt, lamc] = out
    return res

def running_bounds(v, dt, lam):
    L = max(lam, lam_pos(dt))
    out = _CACHE.get((v['_in'], dt, L))
    if out is None:
        out, _ = _trace(list(v['_in']), dt, L)
    return {k: math.inf if math.isnan(float(e)) else SAFETY * float(e) for k, (_, e) in out.items()}
