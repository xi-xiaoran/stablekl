"""Repaired KL divergence between two Beta distributions: formula, analytic gradients, and the
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
OFFICIAL_FN = '_kl_beta_beta'
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

def _algo(o, a, b, c, d, dt):
    one = o.const(1)
    zero = o.const(0)
    half = o.const(Fr(1, 2))
    ma, ea = o.frexp(a)
    mb, eb = o.frexp(b)
    mc, ec = o.frexp(c)
    md, ed = o.frexp(d)
    E = ea + ed - ec - eb
    o.exact = 1 if o.trace else 0
    p1, e1 = o.two_prod(ma, md)
    p2, e2 = o.two_prod(mc, mb)
    Ec = o.iclamp(E, -2, 2)
    p1s = o.ld2(p1, Ec)
    e1s = o.ld2(e1, Ec)
    s1 = o.sub(p1s, p2)
    t1, t2 = o.two_sum(e1s, o.neg(e2))
    sg, tau1 = o.two_sum(s1, t1)
    Nn = o.add(sg, o.add(tau1, t2))
    o.exact = 0
    u = FMT[dt]['u']
    hh = FMT[dt]['eta'] / 2
    mn = FMT[dt]['minnorm']
    nearE = abs(E) <= 2 if o.trace else E.abs() <= 2
    ex0 = o.land(nearE, o.land(o.eq(p1s, p2), o.eq(e1s, e2)))
    small = o.land(nearE, o.land(o.lt(o.abs(Nn), NSMALL), o.lnot(ex0)))
    R_ = o.div(p1, p2)
    zx = o.seterr(o.div(Nn, p2), lambda: 0.0)
    if o.trace:
        zx = o.seterr(zx, lambda: 5 * u * abs(float(zx[0])) + hh * (1.0 if abs(float(zx[0])) < mn else 0.0))
    z = o.where(nearE, lambda: o.where(ex0, lambda: zero, lambda: zx), lambda: o.sub(o.ld2(R_, E), one))
    zs = o.div(o.mul(Nn, o.const(2 ** SZ)), p2)
    zz = o.where(small, lambda: zs, lambda: z)
    zsh = o.iwhere(small, SZ, 0)
    qs = o.div(o.mul(Nn, o.const(2 ** SZ)), p1s)
    qx = o.div(Nn, p1s)
    qq = o.where(small, lambda: qs, lambda: qx)
    if o.trace:
        o.tag('Ez' if ex0 else 'Es' if small else 'Ex' if nearE else 'Ef')
    s = o.add(a, b)
    bs = o.div(b, s)
    as_ = o.div(a, s)
    caM = o.div(mc, ma)
    caE = ec - ea

    def nearfacs():
        r1f = [o.mul(bs, zz), o.iwhere(small, -SZ, 0) if not o.trace else -zsh]
        r0f = [o.neg(o.mul(as_, qq)), o.iwhere(small, -SZ, 0) if not o.trace else -zsh]
        baf = [o.mul(caM, zz), caE - zsh]
        return (r1f, r0f, baf)
    if o.trace:
        if nearE:
            r1f, r0f, baf = nearfacs()
        elif E > 0:
            r1f = [o.sub(o.mul(bs, R_), o.ld2(bs, 0 - E)), E]
            r0f = [o.sub(o.ld2(o.div(as_, R_), 0 - E), as_), 0]
            baf = [o.sub(o.mul(caM, R_), o.ld2(caM, 0 - E)), caE + E]
        else:
            r1f = [o.sub(o.ld2(o.mul(bs, R_), E), bs), 0]
            r0f = [o.sub(o.div(as_, R_), o.ld2(as_, E)), 0 - E]
            baf = [o.sub(o.ld2(o.mul(caM, R_), E), caM), caE]
    else:
        n1, n0, nb = nearfacs()
        pos = E > 0
        r1F = o.t.where(pos, o.sub(o.mul(bs, R_), o.ld2(bs, -E)), o.sub(o.ld2(o.mul(bs, R_), E), bs))
        r0F = o.t.where(pos, o.sub(o.ld2(o.div(as_, R_), -E), as_), o.sub(o.div(as_, R_), o.ld2(as_, E)))
        baF = o.t.where(pos, o.sub(o.mul(caM, R_), o.ld2(caM, -E)), o.sub(o.ld2(o.mul(caM, R_), E), caM))
        r1f = [o.t.where(nearE, n1[0], r1F), o.t.where(nearE, n1[1], o.t.where(pos, E, o.iz))]
        r0f = [o.t.where(nearE, n0[0], r0F), o.t.where(nearE, n0[1], o.t.where(pos, o.iz, -E))]
        baf = [o.t.where(nearE, nb[0], baF), o.t.where(nearE, nb[1], o.t.where(pos, caE + E, caE))]
    r1 = o.ld2(r1f[0], r1f[1])
    r0 = o.ld2(r0f[0], r0f[1])
    ba = o.ld2(baf[0], baf[1])
    h1, l1 = o.two_sum(c, o.neg(a))
    h2, l2 = o.two_sum(d, o.neg(b))
    sh, el = o.two_sum(h1, h2)
    D = o.add(sh, o.add(el, o.add(l1, l2)))

    def Dhalf():
        hh, eh = o.two_sum(o.ld2(h1, -1), o.ld2(h2, -1))
        return o.add(hh, o.add(eh, o.ld2(o.add(l1, l2), -1)))
    if o.trace:
        Df = [D, 0] if math.isfinite(float(D[0])) else [Dhalf(), 1]
    else:
        fD = o.t.isfinite(D)
        Df = [o.t.where(fD, D, Dhalf()), o.t.where(fD, o.iz, o.iz + 1)]
    al = o.div(h1, a)
    be = o.div(h2, b)
    hs = o.add(half, _g(o, s, dt))
    hga = o.add(half, _g(o, a, dt))
    hgb = o.add(half, _g(o, b, dt))
    KS = o.div(hs, s)
    xa = _xdK(o, a, b, hga, hs, as_, dt)
    xb = _xdK(o, b, a, hgb, hs, bs, dt)
    T3a, Ra, Bva, dLa, na, Ta, ka = _pair(o, a, [h1, 0], c, hga, dt)
    T3b, Rb, Bvb, dLb, nb_, Tb, kb = _pair(o, b, [h2, 0], d, hgb, dt)
    T3s, Rs, Bvs, dLs, ns, Ts, ks = _pair(o, s, Df, o.add(c, d), hs, dt)
    ok4 = o.land(o.land(ka, kb), ks)
    lr1 = o.sub(Ts, Ta)
    lr0 = o.sub(Ts, Tb)
    alln = o.land(o.land(na, nb_), ns)
    if o.trace:
        o.tag('S2' if alln else 'S1')

    def phi_small(coef_m, coef_e, rf):
        return mprod(o, [[o.mul(o.mul(coef_m, rf[0]), rf[0]), coef_e + 2 * rf[1]], _phh(o, o.ld2(rf[0], rf[1]), dt)])
    if o.trace:
        if small:
            P1 = o.add(phi_small(mc, ec, r1f), phi_small(md, ed, r0f))
        else:
            P1 = o.add(_cphi_mf(o, c, r1f, lr1, dt), _cphi_mf(o, d, r0f, lr0, dt))
    else:
        P1s = o.add(phi_small(mc, ec, r1f), phi_small(md, ed, r0f))
        P1n = o.add(_cphi_mf(o, c, r1f, lr1, dt), _cphi_mf(o, d, r0f, lr0, dt))
        P1 = o.t.where(small, P1s, P1n)

    def second():
        Qa = mprod(o, [xa, al, al])
        Qb = mprod(o, [xb, be, be])
        Qc = mprod(o, [KS, bs, a, baf, baf])
        Q = o.add(o.add(Qa, Qb), Qc)
        kl = o.add(o.add(o.clamp0(P1), o.mul(half, o.clamp0(Q))), o.sub(o.add(T3a, T3b), T3s))
        K1 = o.add(one, KS)
        ga_ = o.add(o.neg(mprod(o, [xa, h1], [a, a])), mprod(o, [bs, baf, K1]))
        gb_ = o.sub(o.neg(mprod(o, [xb, h2], [b, b])), mprod(o, [as_, baf, K1]))
        ta1 = mprod(o, [xa, h1], [a, a])
        tb1 = mprod(o, [xb, h2], [b, b])
        ta2 = mprod(o, [bs, baf, K1])
        tb2 = mprod(o, [as_, baf, K1])
        rc_ = rterm(r1f, r1, lr1, mc, ec, ma, ea)
        rd_ = rterm(r0f, r0, lr0, md, ed, mb, eb)
        ga_ = o.add(o.neg(ta1), ta2)
        gb_ = o.sub(o.neg(tb1), tb2)
        gc_ = o.add(o.sub(ta1, rc_), o.sub(Ra, Rs))
        gd_ = o.add(o.sub(tb1, rd_), o.sub(Rb, Rs))
        A_ = o.abs
        mags = (o.add(A_(ta1), A_(ta2)), o.add(A_(tb1), A_(tb2)), o.add(o.add(A_(ta1), A_(rc_)), o.add(A_(Ra), A_(Rs))), o.add(o.add(A_(tb1), A_(rd_)), o.add(A_(Rb), A_(Rs))))
        return (kl, ga_, gb_, gc_, gd_, mags)

    def first():
        kl = o.add(o.clamp0(P1), o.sub(o.add(Bva, Bvb), Bvs))
        return kl

    def l1r(r, lr):
        return o.where(o.lt(o.abs(r), NSMALL * 4), lambda: _horner(o, o.neg(r), [Fr(1, k + 1) for k in range(L1N[dt])]), lambda: o.where(o.isfinite(r), lambda: o.div(o.where(o.lt(r, -0.5), lambda: lr, lambda: o.fn('log1p', r)), r), lambda: o.const(0)))

    def rterm(rf, rr, lr, cm, ce, dm, de):
        L = l1r(rr, lr)
        if o.trace and abs(float(rr[0])) < NSMALL * 4:
            L = o.extra(L, lambda: l1_tail(float(rr[0]), L1N[dt]))
        inner = o.add(mprod(o, [[cm, ce], KS], [[dm, de]]), L)
        return mprod(o, [rf, inner])
    kl2, ga_, gb_, gc_, gd_, mags = second()
    if o.trace:
        kl = kl2 if alln else first()
    else:
        kl = o.t.where(alln, kl2, first())

    def oldc():
        l1 = _log1p_mf(o, r1f, lr1)
        return (o.sub(o.sub(dLa, dLs), l1), o.add(o.add(o.abs(dLa), o.abs(dLs)), o.abs(l1)))

    def oldd():
        l0 = _log1p_mf(o, r0f, lr0)
        return (o.sub(o.sub(dLb, dLs), l0), o.add(o.add(o.abs(dLb), o.abs(dLs)), o.abs(l0)))

    def gsK():
        return mprod(o, [Df, KS], [s])

    def olda():
        u1 = mprod(o, [c, r1f], [a])
        u2 = mprod(o, [al, hga], [a])
        u3 = gsK()
        return (o.add(o.sub(u1, u2), u3), o.add(o.add(o.abs(u1), o.abs(u2)), o.abs(u3)))

    def oldb():
        u1 = mprod(o, [d, r0f], [b])
        u2 = mprod(o, [be, hgb], [b])
        u3 = gsK()
        return (o.add(o.sub(u1, u2), u3), o.add(o.add(o.abs(u1), o.abs(u2)), o.abs(u3)))
    sq_ = o.add(c, d)
    allsmall = o.land(o.land(o.lt(a, 0.5), o.lt(b, 0.5)), o.land(o.lt(c, 0.5), o.lt(d, 0.5)))
    allsmall = o.land(allsmall, o.land(o.land(o.lt(a, o.mul(o.const(8), b)), o.lt(b, o.mul(o.const(8), a))), o.land(o.lt(c, o.mul(o.const(8), d)), o.lt(d, o.mul(o.const(8), c)))))
    allsmall = o.land(allsmall, o.lnot(alln))

    def smallq():
        ps = _dpsi1(o, s, o.ld2(Df[0], Df[1]), dt)
        p1_ = _dpsi1(o, a, h1, dt)
        p0_ = _dpsi1(o, b, h2, dt)

        def rat(sh):
            sc_ = [o.const(1), -sh]
            return (mprod(o, [h1, sc_], [a, c]), mprod(o, [h2, sc_], [b, d]), mprod(o, [Df, sc_], [s, sq_]))

        def comb(sh):
            if sh and o.trace:
                o.tag('sk')
            r1_, r0_, rs = rat(sh)
            pc = o.sub(p1_, ps)
            pd = o.sub(p0_, ps)
            if sh:
                pc = o.ld2(pc, -sh)
                pd = o.ld2(pd, -sh)
            gc = o.add(o.sub(r1_, rs), pc)
            gd = o.add(o.sub(r0_, rs), pd)
            if sh:
                gc = o.ld2(gc, sh)
                gd = o.ld2(gd, sh)
            return (gc, gd)
        gcs, gds = comb(0)
        gcs = o.where(o.isfinite(gcs), lambda: gcs, lambda: comb(8)[0])
        gds = o.where(o.isfinite(gds), lambda: gds, lambda: comb(8)[1])
        return (gcs, gds)
    new = [ga_, gb_, gc_, gd_]
    olds = (olda, oldb, oldc, oldd)
    if o.trace:
        sm = smallq() if allsmall else None
    else:
        sm = smallq()
    for k in range(4):
        gn, mg = (new[k], mags[k])
        go, mo = olds[k]()
        if o.trace:
            use_old = not math.isfinite(float(gn[0])) or (k >= 2 and (not ok4 or float(mo[0]) < float(mg[0])))
            if use_old:
                o.tag('f' + 'abcd'[k])
                gn, mg = (go, mo)
            if k >= 2 and sm is not None:
                o.tag('s' + 'abcd'[k])
                gn = sm[k - 2]
            new[k] = gn
        else:
            keep = o.t.isfinite(gn)
            if k >= 2:
                keep = keep & ok4 & ~(mo < mg)
            g1 = o.t.where(keep, gn, go)
            m1 = o.t.where(keep, mg, mo)
            if k >= 2:
                g1 = o.t.where(allsmall, sm[k - 2], g1)
            new[k] = g1
    ga_, gb_, gc_, gd_ = new
    kl = o.clamp0(kl)
    if o.trace:
        if not math.isfinite(float(s[0])):
            o.tag('sovf')
            nan = (o.T(np.nan), math.inf)
            return (nan, nan, nan, nan, nan)
    else:
        sf = o.t.isfinite(s)
        nanv = o.const(float('nan'))
        kl, ga_, gb_, gc_, gd_ = (o.t.where(sf, x, nanv) for x in (kl, ga_, gb_, gc_, gd_))
    return (kl, ga_, gb_, gc_, gd_)
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    _DT = {torch.float16: 'fp16', torch.float32: 'fp32', torch.float64: 'fp64'}

    class BetaKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, a, b, c, d):
            dt = _DT[a.dtype]
            o = _TOps(torch, a, dt)
            kl, g1, g2, g3, g4 = _algo(o, a, b, c, d, dt)
            same = (a == c) & (b == d)
            zero = torch.zeros_like(a)
            kl = torch.where(same, zero, kl)
            outs = [torch.where(same, zero, x) for x in (g1, g2, g3, g4)]
            ctx.save_for_backward(*outs)
            return kl

        @staticmethod
        def backward(ctx, go):
            return tuple((go * x for x in ctx.saved_tensors))
NAME = 'beta'
PARAMS = ('concentration1', 'concentration0')
COMPS = ['KL', 'P.concentration1', 'P.concentration0', 'Q.concentration1', 'Q.concentration0']
EXPECTED_ROWS = {'fp16': 127313, 'fp32': 201843}
TRANSCENDENTAL_COMPS = tuple(COMPS)
TRANSCENDENTAL_FUNCS = ('log1p', 'log', 'expm1', 'exp')

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    leaves = [to_t(x, dt_in).to(TD[cdt]).requires_grad_(True) for x in (P['concentration1'], P['concentration0'], Q['concentration1'], Q['concentration0'])]
    kl = BetaKLC.apply(*leaves)
    kl.sum().backward()
    vals = {'KL': kl.detach()}
    for name, leaf in zip(COMPS[1:], leaves):
        vals[name] = leaf.grad.detach()
    return vals

def exact_grads(a, b, c, d):
    return None

def identity(a, b, c, d):
    return a == c and b == d

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
        print(f'fix_beta.running_bounds: {dt} lambda_pos = {_LAM_POS[dt]}; the bound uses max(lambda from p2_accept, lambda_pos)', flush=True)
    return _LAM_POS[dt]

def _trace(a, b, c, d, dt, lam):
    o = _ROps(dt, lam)
    A, B_, C, D_ = (o.inp(x) for x in (a, b, c, d))
    if float(A[0]) == float(C[0]) and float(B_[0]) == float(D_[0]):
        z = (o.T(0), 0.0)
        out = dict(zip(COMPS, [z] * 5))
        return (out, {'_branch': 'id'})
    res = _algo(o, A, B_, C, D_, dt)
    return (dict(zip(COMPS, res)), {'_branch': '/'.join(o.tags)})
_CACHE = {}

def replay(a, b, c, d, dt):
    lamc = max(3, _LAM_POS.get(dt, 3))
    out, flags = _trace(a, b, c, d, dt, lamc)
    res = {k: float(v[0]) for k, v in out.items()}
    res.update(flags)
    res['_in'] = (float(a), float(b), float(c), float(d))
    _CACHE.clear()
    _CACHE[res['_in'], dt, lamc] = out
    return res

def running_bounds(v, dt, lam):
    L = max(lam, lam_pos(dt))
    out = _CACHE.get((v['_in'], dt, L))
    if out is None:
        out, _ = _trace(*v['_in'], dt, L)
    return {k: math.inf if math.isnan(float(e)) else SAFETY * float(e) for k, (_, e) in out.items()}
