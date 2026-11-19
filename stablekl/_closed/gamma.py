"""Repaired KL divergence between two Gamma distributions: formula, analytic gradients, and the
entry point run_rewrite_torch used by the benchmark (dtype in, compute dtype, outputs per component).
The derivation is in the paper's appendix on interface A."""
from __future__ import annotations
import math
from decimal import Decimal
from fractions import Fraction as Fr
import numpy as np
FMT = {'fp16': dict(np=np.float16, u=2.0 ** (-11), eta=2.0 ** (-24), minnorm=2.0 ** (-14), emax=15, split=65.0), 'fp32': dict(np=np.float32, u=2.0 ** (-24), eta=2.0 ** (-149), minnorm=2.0 ** (-126), emax=127, split=4097.0), 'fp64': dict(np=np.float64, u=2.0 ** (-53), eta=2.0 ** (-1074), minnorm=2.0 ** (-1022), emax=1023, split=134217729.0)}
SAFETY = 1.02
KTERMS = {'fp16': 9, 'fp32': 17, 'fp64': 36}
SEG_LO, SEG_HI = (-0.75, 3.0)
NQ = {'fp16': 5, 'fp32': 7, 'fp64': 13}
QTAIL = {'fp16': 1e-05, 'fp32': 1e-09, 'fp64': 1e-17}
ABG = {'fp16': 3, 'fp32': 5, 'fp64': 10}
KAS = {'fp16': 7, 'fp32': 9, 'fp64': 12}
MH = {'fp16': 9, 'fp32': 16, 'fp64': 29}
T_NEAR = 1.0
SZ = 12
NSMALL = 2.0 ** (-8)
SBQ = {'fp16': 12, 'fp32': 0, 'fp64': 0}
OFFICIAL_FN = '_kl_gamma_gamma'
GRAD_TRUTH = 'frozen_interval'
_BERN = (Fr(1, 6), Fr(-1, 30), Fr(1, 42), Fr(-1, 30), Fr(5, 66), Fr(-691, 2730), Fr(7, 6), Fr(-3617, 510), Fr(43867, 798), Fr(-174611, 330), Fr(854513, 138), Fr(-236364091, 2730), Fr(8553103, 6), Fr(-23749461029, 870))
G_AS = _BERN
W_AS = tuple((_BERN[k - 1] / (2 * k * (2 * k - 1)) for k in range(1, 15)))
WP_AS = tuple((_BERN[k - 1] / (2 * k) for k in range(1, 15)))
H_CO = tuple((Fr(1, 2 * m + 1) for m in range(1, 40)))
HP_CO = tuple((Fr(m, 2 * m + 1) for m in range(1, 40)))
LN2_FR = Fr(Decimal('0.693147180559945309417232121458176568075500134'))
_GL = {5: (('0.906179845938663992797626878299392965125651911', '0.538469310105683091036314420700208804967286607', '0', '-0.538469310105683091036314420700208804967286607', '-0.906179845938663992797626878299392965125651911'), ('0.236926885056189087514264040719917362643260002', '0.478628670499366468041291514835638192912295553', '0.568888888888888888888888888888888888888888889', '0.478628670499366468041291514835638192912295553', '0.236926885056189087514264040719917362643260002')), 7: (('0.949107912342758524526189684047851262400770938', '0.741531185599394439863864773280788407074147647', '0.405845151377397166906606412076961463347382014', '0', '-0.405845151377397166906606412076961463347382014', '-0.741531185599394439863864773280788407074147647', '-0.949107912342758524526189684047851262400770938'), ('0.129484966168869693270611432679082018328587402', '0.279705391489276667901467771423779582486925065', '0.381830050505118944950369775488975133878365084', '0.417959183673469387755102040816326530612244898', '0.381830050505118944950369775488975133878365084', '0.279705391489276667901467771423779582486925065', '0.129484966168869693270611432679082018328587402')), 13: (('0.984183054718588149472829448807109611064990562', '0.917598399222977965206547836500719512390474790', '0.801578090733309912794206489582859890305615725', '0.642349339440340220643984606995515650071697398', '0.448492751036446852877912852127639867801921667', '0.230458315955134794065528121097988835211542376', '0', '-0.230458315955134794065528121097988835211542376', '-0.448492751036446852877912852127639867801921667', '-0.642349339440340220643984606995515650071697398', '-0.801578090733309912794206489582859890305615725', '-0.917598399222977965206547836500719512390474790', '-0.984183054718588149472829448807109611064990562'), ('0.040484004765315879520021592200986060041986546', '0.092121499837728447914421775953797120923684000', '0.138873510219787238463601776868871467621862718', '0.178145980761945738280046691996097995512812651', '0.207816047536888502312523219306052763386582609', '0.226283180262897238412090186039776618434757738', '0.232551553230873910194589515268835948156627477', '0.226283180262897238412090186039776618434757738', '0.207816047536888502312523219306052763386582609', '0.178145980761945738280046691996097995512812651', '0.138873510219787238463601776868871467621862718', '0.092121499837728447914421775953797120923684000', '0.040484004765315879520021592200986060041986546'))}
GLCV = {N: tuple((((1 + Fr(Decimal(x))) / 2, Fr(Decimal(w)) / 2) for x, w in zip(*_GL[N]))) for N in _GL}

def g_tail(y, K):
    return abs(float(G_AS[K])) * y ** (-(2 * K + 1))

def w_tail(y, K):
    return abs(float(W_AS[K])) * y ** (-(2 * K + 1))

def wp_tail(y, K):
    return abs(float(WP_AS[K])) * y ** (-(2 * K + 2))

def h_tail(u, M):
    return u ** (2 * M + 2) / ((2 * M + 3) * (1 - u * u))

def hp_tail(u, M):
    return 2 * u ** (2 * M + 3) / (1 - u * u)

def q_tail_bound(z, K):
    return z ** K / ((2 * K + 3) * (1 - z))
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    _DT = {torch.float16: 'fp16', torch.float32: 'fp32', torch.float64: 'fp64'}

    def _phiF(xs, s, K):
        z = s * s
        q = torch.full_like(s, 1.0 / (2 * K + 1))
        for k in range(K - 1, 0, -1):
            q = q * z + torch.full_like(s, 1.0 / (2 * k + 1))
        return xs - torch.full_like(s, 2.0) * z * q

    def _horner(x, coefs):
        p = torch.full_like(x, float(coefs[-1]))
        for c in coefs[-2::-1]:
            p = p * x + float(c)
        return p

    def _xfc(x, f, c, mn):
        q = f / c
        xf = x * f
        ok1 = torch.isfinite(q) & (q.abs() >= mn)
        ok2 = torch.isfinite(xf) & ((xf.abs() >= mn) | (x == 0))
        return torch.where(ok1, x * q, torch.where(ok2, xf / c, x / c * f))

    def _two_prod(x, y, C):

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

    def _two_sum(a, b):
        s = a + b
        bb = s - a
        e = a - (s - bb) + (b - bb)
        return (s, e)

    def _g(s, dt):
        AB = ABG[dt]
        K = KAS[dt]
        one = torch.ones_like(s)
        zero = torch.zeros_like(s)
        half = torch.full_like(s, 0.5)
        n = torch.clamp(torch.ceil(torch.full_like(s, float(AB)) - s), 0.0, float(AB))
        acc = zero
        for j in range(AB):
            t = s + float(j)
            q = s / t
            term = half * (q * q) / ((t + one) * (t + one))
            acc = acc + torch.where(n > float(j), term, zero)
        y = s + n
        iy = one / y
        iy2 = iy * iy
        ga = iy * _horner(iy2, G_AS[:K])
        r = s / y
        return acc + r * r * ga

    def _wpair(a, dt):
        AB = ABG[dt]
        K = KAS[dt]
        M = MH[dt]
        one = torch.ones_like(a)
        zero = torch.zeros_like(a)
        half = torch.full_like(a, 0.5)
        n = torch.clamp(torch.ceil(torch.full_like(a, float(AB)) - a), 0.0, float(AB))
        W = zero
        Wp = zero
        V = zero
        Vp = zero
        for j in range(AB):
            t = a + float(j)
            u = one / (t + t + one)
            u2 = u * u
            hs = u2 * _horner(u2, H_CO[:M])
            hps = torch.full_like(a, -4.0) * (u * u2) * _horner(u2, HP_CO[:M])
            lg = torch.log1p(t) - torch.log(t)
            hd = (t + half) * lg - one
            hpd = lg - half / t - half / (t + one)
            small = t < 0.5
            hj = torch.where(small, hd, hs)
            hpj = torch.where(small, hpd, hps)
            m = n > float(j)
            W = W + torch.where(m, hj, zero)
            Wp = Wp + torch.where(m, hpj, zero)
            if j == 0:
                V = (a + half) * torch.log1p(a) - a * torch.log(a) - one
                Vp = torch.log1p(a) - torch.log(a) - half / (a + one)
            else:
                V = V + torch.where(m, hj, zero)
                Vp = Vp + torch.where(m, hpj, zero)
        y = a + n
        iy = one / y
        iy2 = iy * iy
        wa = iy * _horner(iy2, W_AS[:K])
        wpa = (zero - iy2) * _horner(iy2, WP_AS[:K])
        return (W + wa, Wp + wpa, V + wa, Vp + wpa)

    class GammaKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, ap, bp, aq, bq):
            dt = _DT[ap.dtype]
            K = KTERMS[dt]
            N = NQ[dt]
            one = torch.ones_like(ap)
            zero = torch.zeros_like(ap)
            half = torch.full_like(ap, 0.5)
            d = aq - ap
            al = d / ap
            rq = aq / ap
            mnT = FMT[dt]['minnorm']
            Tlog = torch.where((rq >= mnT) & torch.isfinite(rq), torch.log(rq), torch.log(aq) - torch.log(ap))
            T = torch.where((al < -0.5) | torch.isinf(al), Tlog, torch.log1p(al))
            mA, eA = torch.frexp(ap.double())
            mB, eB = torch.frexp(bq.double())
            mC, eC = torch.frexp(aq.double())
            mD, eD = torch.frexp(bp.double())
            E = eA + eB - eC - eD
            Cs = FMT[dt]['split']
            p1, e1 = _two_prod(mA.to(ap.dtype), mB.to(ap.dtype), Cs)
            p2, e2 = _two_prod(mC.to(ap.dtype), mD.to(ap.dtype), Cs)
            sc = torch.where(E == -2, torch.full_like(ap, 0.25), torch.where(E == -1, torch.full_like(ap, 0.5), torch.where(E == 0, one, torch.where(E == 1, torch.full_like(ap, 2.0), torch.full_like(ap, 4.0)))))
            p1s = p1 * sc
            e1s = e1 * sc
            s1 = p1s - p2
            t1, t2 = _two_sum(e1s, zero - e2)
            sg, tau1 = _two_sum(s1, t1)
            Nn = sg + (tau1 + t2)
            zx = Nn / p2
            nearE = E.abs() <= 2
            ex0 = nearE & (p1s == p2) & (e1s == e2)
            zx = torch.where(ex0, zero, zx)
            R = p1 / p2
            ld2 = lambda x_, k_: (x_.double() * torch.exp2(k_.double())).to(ap.dtype)
            Pf = ld2(R, E)
            z = torch.where(nearE, zx, Pf - one)
            bigE = E >= 3
            small = nearE & (Nn.abs() < NSMALL) & ~ex0
            zs = Nn * float(2 ** SZ) / p2
            mCd = mC.to(ap.dtype)

            def X(m_, e_):
                xb = ld2(mCd * R / m_ - ld2(mCd / m_, zero.to(E.dtype) - E), E + eC - e_)
                xs = ld2(mCd * zs / m_, eC - e_ - SZ)
                xn = ld2(mCd * z / m_, eC - e_)
                return torch.where(bigE, xb, torch.where(small, xs, xn))
            LZ = torch.where(nearE, torch.log1p(zx), torch.log(R) + E.to(ap.dtype) * float(LN2_FR))
            segz = (z >= SEG_LO) & (z <= SEG_HI)
            sz = z / (torch.full_like(z, 2.0) + z)
            two = torch.full_like(z, 2.0)
            tz = two + z
            qz = torch.full_like(z, 1.0 / (2 * K + 1))
            for k in range(K - 1, 0, -1):
                qz = qz * (sz * sz) + torch.full_like(z, 1.0 / (2 * k + 1))
            phh = (one - two * z * qz / (tz * tz)) / tz
            kls = ld2(mCd * zs * zs * phh, eC - 2 * SZ)
            klb = ld2(mCd * R - ld2(mCd * (one + LZ), zero.to(E.dtype) - E), E + eC)
            kl1 = torch.where(small, kls, torch.where(segz, aq * sz * _phiF(z, sz, K), torch.where(bigE, klb, aq * (z - LZ))))
            sega = (al >= SEG_LO) & (al <= SEG_HI)
            sa = al / (torch.full_like(al, 2.0) + al)
            phX = torch.where(torch.isinf(al), al, al - T)
            pha = torch.where(sega, sa * _phiF(al, sa, K), phX)
            mAd = mA.to(ap.dtype)
            md, ed = torch.frexp(d.double())
            mdd = md.to(ap.dtype)
            kl2I = ld2(mdd / mAd, ed - eA - 1) - half * T
            kl2 = torch.where(sega, half * sa * _phiF(al, sa, K), torch.where(torch.isinf(al), kl2I, half * phX))
            Bq = zero
            Lq = zero
            for c_i, v_i in GLCV[N]:
                tau = T * float(c_i)
                Wt = T * float(v_i)
                s = ap * torch.exp(tau)
                gg = _g(s, dt)
                Bq = Bq + Wt * float(2 ** SBQ[dt]) * torch.expm1(T - tau) * gg
                Lq = Lq + Wt * torch.exp(zero - tau) * gg
            Bq = ld2(Bq, torch.full_like(E, -SBQ[dt]))
            dLq = Lq / ap + half * al / aq
            Wp_, Wpp, Vp_, Vpp = _wpair(ap, dt)
            Wq_, Wpq, Vq_, Vpq = _wpair(aq, dt)
            peel = torch.maximum(ap, aq) < 0.5
            hpha = torch.where(torch.isinf(al), kl2I, half * pha)
            alq = torch.where(torch.isinf(al), ld2(mdd / mAd / mCd, ed - eA - eC), al / aq)
            Bf = torch.where(peel, hpha + (Vq_ - Vp_ - d * Vpp), Wq_ - Wp_ - d * Wpp)
            dLf = torch.where(peel, alq + (Vpq - Vpp), Wpq - half / aq - (Wpp - half / ap))
            near = T.abs() <= T_NEAR
            Bw = torch.where(near, Bq, Bf)
            dL = torch.where(near, dLq, dLf)
            kl = torch.clamp_min(kl1, 0.0) + torch.clamp_min(kl2, 0.0) + torch.clamp_min(Bw, 0.0)
            hg = half + _g(ap, dt)
            hm = mdd / mAd * hg / mAd
            t1 = X(mAd, eA)
            t2 = ld2(hm, ed - 2 * eA)
            e1 = torch.clamp(E, min=0) + eC - eA
            S = torch.where(d == 0, e1, torch.maximum(e1, ed - 2 * eA))
            gS = ld2(X(mAd, eA + S) - ld2(hm, ed - 2 * eA - S), S)
            g_ap = torch.where(torch.isfinite(t1) & torch.isfinite(t2), t1 - t2, gS)
            g_bp = zero - X(mD.to(ap.dtype), eD)
            g_aq = dL - LZ
            g_bq = X(mB.to(ap.dtype), eB)
            same = (ap == aq) & (bp == bq)
            kl = torch.where(same, zero, kl)
            outs = [torch.where(same, zero, x) for x in (g_ap, g_bp, g_aq, g_bq)]
            ctx.save_for_backward(*outs)
            return kl

        @staticmethod
        def backward(ctx, go):
            return tuple((go * x for x in ctx.saved_tensors))
NAME = 'gamma'
PARAMS = ('concentration', 'rate')
COMPS = ['KL', 'P.concentration', 'P.rate', 'Q.concentration', 'Q.rate']
EXPECTED_ROWS = {'fp16': 129580, 'fp32': 201487}
TRANSCENDENTAL_COMPS = ('KL', 'P.concentration', 'P.rate', 'Q.concentration', 'Q.rate')
TRANSCENDENTAL_FUNCS = ('log1p', 'log', 'expm1', 'exp')

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    leaves = [to_t(x, dt_in).to(TD[cdt]).requires_grad_(True) for x in (P['concentration'], P['rate'], Q['concentration'], Q['rate'])]
    kl = GammaKLC.apply(*leaves)
    kl.sum().backward()
    vals = {'KL': kl.detach()}
    for name, leaf in zip(COMPS[1:], leaves):
        vals[name] = leaf.grad.detach()
    return {k: v.cpu().numpy().astype(NPF[cdt]) for k, v in vals.items()}

def exact_grads(ap, bp, aq, bq):
    return None

def identity(ap, bp, aq, bq):
    return ap == aq and bp == bq

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
        print(f'fix_gamma.running_bounds: {dt} lambda_pos = {_LAM_POS[dt]}; the bound uses max(lambda from p2_accept, lambda_pos)', flush=True)
    return _LAM_POS[dt]

class _Tr:

    def __init__(self, dt, lam):
        f = FMT[dt]
        self.dt = dt
        self.T = f['np']
        self.u = f['u']
        self.h = f['eta'] / 2
        self.lam = lam
        self.mn = f['minnorm']
        self.OT = float(np.finfo(self.T).max) * (1 + self.u)

    def A(self, x):
        return min(abs(float(x)), self.OT)

    def sb(self, x):
        return 1.0 if abs(float(x)) < self.mn else 0.0

    def R(self, z):
        return self.u * self.A(z) + self.h * self.sb(z)

    def inp(self, x):
        return (self.T(x), 0.0)

    def const(self, c):
        v = self.T(float(c))
        return (v, float(abs(Fr(float(v)) - c)))

    def add(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] + b[0])
        return (z, a[1] + b[1] + (self.u * self.A(z) if abs(float(z)) >= self.mn else 0.0))

    def sub(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] - b[0])
        return (z, a[1] + b[1] + (self.u * self.A(z) if abs(float(z)) >= self.mn else 0.0))

    def mul(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] * b[0])
        return (z, self.A(b[0]) * a[1] + self.A(a[0]) * b[1] + self.R(z))

    def div(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] / b[0])
        bb = abs(float(b[0]))
        if bb == math.inf:
            return (z, a[1] / bb if a[1] < math.inf else math.inf)
        return (z, (a[1] + self.A(z) * b[1]) / bb + self.R(z) if bb > 0 else float('inf'))

    def fn(self, name, a):
        with np.errstate(all='ignore'):
            z = self.T(getattr(np, name)(np.array([a[0]], dtype=self.T))[0])
        xv = float(a[0])
        e = float(a[1])
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

    def extra(self, a, e):
        return (a[0], a[1] + e)

    def val(self, v, e=0.0):
        return (self.T(v), e)

    def clamp0(self, a):
        return (max(a[0], self.T(0)), a[1] if math.isfinite(float(a[0])) else math.inf)

def _tr_phiF(tr, x, s, dt):
    K = KTERMS[dt]
    z = tr.mul(s, s)
    q = tr.const(Fr(1, 2 * K + 1))
    for k in range(K - 1, 0, -1):
        q = tr.add(tr.mul(q, z), tr.const(Fr(1, 2 * k + 1)))
    q = tr.extra(q, q_tail_bound(float(z[0]), K))
    return tr.sub(x, tr.mul(tr.mul(tr.inp(2.0), z), q))

def _tr_horner(tr, x, coefs):
    p = tr.const(coefs[-1])
    for c in coefs[-2::-1]:
        p = tr.add(tr.mul(p, x), tr.const(c))
    return p

def _tr_aphi(tr, a, x, lg, dt):
    xv = float(x[0])
    if SEG_LO <= xv <= SEG_HI:
        s = tr.div(x, tr.add(tr.inp(2.0), x))
        F_ = _tr_phiF(tr, x, s, dt)
        return (tr.mul(tr.mul(a, s), F_), tr.mul(s, F_), 'S')
    ph = tr.sub(x, lg)
    return (tr.mul(a, ph), ph, 'X')

def _tr_g(tr, s, dt):
    AB = ABG[dt]
    K = KAS[dt]
    T = tr.T
    one = tr.inp(1.0)
    half = tr.inp(0.5)
    nv = float(np.clip(np.ceil(T(T(AB) - s[0])), 0.0, float(AB)))
    acc = tr.inp(0.0)
    for j in range(AB):
        if not nv > j:
            break
        t = tr.add(s, tr.inp(float(j)))
        q = tr.div(s, t) if j else (T(1.0), 0.0)
        t1 = tr.add(t, one)
        acc = tr.add(acc, tr.div(tr.mul(half, tr.mul(q, q)), tr.mul(t1, t1)))
    y = tr.add(s, tr.inp(nv))
    iy = tr.div(one, y)
    iy2 = tr.mul(iy, iy)
    ga = tr.mul(iy, _tr_horner(tr, iy2, G_AS[:K]))
    ga = tr.extra(ga, g_tail(float(y[0]), K) * (1 + 4 * tr.u))
    r = tr.div(s, y) if nv > 0 else (T(1.0), 0.0)
    return tr.add(acc, tr.mul(tr.mul(r, r), ga))

def _tr_wpair(tr, a, dt, want_v):
    AB = ABG[dt]
    K = KAS[dt]
    M = MH[dt]
    T = tr.T
    one = tr.inp(1.0)
    half = tr.inp(0.5)
    nv = float(np.clip(np.ceil(T(T(AB) - a[0])), 0.0, float(AB)))
    W = tr.inp(0.0)
    Wp = tr.inp(0.0)
    V = Vp = None
    for j in range(AB):
        if not nv > j and (not (want_v and j == 0)):
            break
        t = tr.add(a, tr.inp(float(j)))
        if nv > j:
            if float(t[0]) < 0.5:
                lg = tr.sub(tr.fn('log1p', t), tr.fn('log', t))
                hj = tr.sub(tr.mul(tr.add(t, half), lg), one)
                hpj = tr.sub(tr.sub(lg, tr.div(half, t)), tr.div(half, tr.add(t, one)))
            else:
                u = tr.div(one, tr.add(tr.add(t, t), one))
                u2 = tr.mul(u, u)
                uv = float(u[0])
                hj = tr.extra(tr.mul(u2, _tr_horner(tr, u2, H_CO[:M])), h_tail(uv, M))
                hpj = tr.mul(tr.mul(tr.inp(-4.0), tr.mul(u, u2)), _tr_horner(tr, u2, HP_CO[:M]))
                hpj = tr.extra(hpj, hp_tail(uv, M))
            W = tr.add(W, hj)
            Wp = tr.add(Wp, hpj)
        if want_v:
            if j == 0:
                l1 = tr.fn('log1p', a)
                la = tr.fn('log', a)
                V = tr.sub(tr.sub(tr.mul(tr.add(a, half), l1), tr.mul(a, la)), one)
                Vp = tr.sub(tr.sub(l1, la), tr.div(half, tr.add(a, one)))
            elif nv > j:
                V = tr.add(V, hj)
                Vp = tr.add(Vp, hpj)
    y = tr.add(a, tr.inp(nv))
    iy = tr.div(one, y)
    iy2 = tr.mul(iy, iy)
    yv = float(y[0])
    wa = tr.extra(tr.mul(iy, _tr_horner(tr, iy2, W_AS[:K])), w_tail(yv, K) * (1 + 4 * tr.u))
    wpa = tr.extra(tr.mul(tr.sub(tr.inp(0.0), iy2), _tr_horner(tr, iy2, WP_AS[:K])), wp_tail(yv, K) * (1 + 4 * tr.u))
    out = (tr.add(W, wa), tr.add(Wp, wpa))
    if want_v:
        out += (tr.add(V, wa), tr.add(Vp, wpa))
    return out

def _two_prod_np(x, y, T, C):
    with np.errstate(all='ignore'):

        def split(v):
            c = T(v * T(C))
            hi = T(c - T(c - v))
            return (hi, T(v - hi))
        p = T(x * y)
        hx, lx = split(x)
        hy, ly = split(y)
        e = T(T(T(T(hx * hy) - p) + T(hx * ly)) + T(lx * hy))
        e = T(e + T(lx * ly))
    return (p, e)

def _two_sum_np(a, b, T):
    with np.errstate(all='ignore'):
        s = T(a + b)
        bb = T(s - a)
        e = T(T(a - T(s - bb)) + T(b - bb))
    return (s, e)

def _trace(ap, bp, aq, bq, dt, lam):
    tr = _Tr(dt, lam)
    T = tr.T
    N = NQ[dt]
    u = tr.u
    apv, bpv, aqv, bqv = (float(x) for x in (ap, bp, aq, bq))
    ap, bp, aq, bq = (tr.inp(v) for v in (apv, bpv, aqv, bqv))
    one = tr.inp(1.0)
    zero = tr.inp(0.0)
    half = tr.inp(0.5)
    if apv == aqv and bpv == bqv:
        return ({k: zero for k in COMPS}, {'_branch': 'id'})
    d = tr.sub(aq, ap)
    al = tr.div(d, ap)
    alv = float(al[0])
    if alv < -0.5 or math.isinf(alv):
        rq = tr.div(aq, ap)
        Tl = tr.fn('log', rq) if FMT[dt]['minnorm'] <= float(rq[0]) < math.inf else tr.sub(tr.fn('log', aq), tr.fn('log', ap))
    else:
        Tl = tr.fn('log1p', al)
    (mA, eA), (mB, eB), (mC, eC), (mD, eD) = (math.frexp(v) for v in (apv, bqv, aqv, bpv))
    E = eA + eB - eC - eD
    C = FMT[dt]['split']
    p1, e1 = _two_prod_np(T(mA), T(mB), T, C)
    p2, e2 = _two_prod_np(T(mC), T(mD), T, C)
    with np.errstate(all='ignore'):
        sc = T({-2: 0.25, -1: 0.5, 0: 1.0, 1: 2.0}.get(E, 4.0))
        p1s = T(p1 * sc)
        e1s = T(e1 * sc)
        s1 = T(p1s - p2)
        t1, t2 = _two_sum_np(e1s, T(T(0) - e2), T)
        sg, tau1 = _two_sum_np(s1, t1, T)
        Nn = T(sg + T(tau1 + t2))
        zxv = T(Nn / p2)
        R = T(p1 / p2)
    if abs(E) <= 2:
        if p1s == p2 and e1s == e2:
            z = (T(0), 0.0)
            zb = 'E0'
        else:
            z = (zxv, 5 * u * abs(float(zxv)) + tr.h * tr.sb(zxv))
            zb = 'Ex'
        LZ = tr.fn('log1p', z)
    Rt = (R, 3 * u * abs(float(R)))
    if abs(E) > 2:
        with np.errstate(all='ignore'):
            Pv = T(float(np.ldexp(np.float64(R), E)))
        P = (Pv, float(np.ldexp(Rt[1], E)) + tr.R(Pv))
        z = tr.sub(P, one)
        zb = 'Ef'
        LZ = tr.add(tr.fn('log', Rt), tr.mul(tr.inp(float(T(E))), tr.const(LN2_FR)))
    bigE = E >= 3
    small = abs(E) <= 2 and abs(float(Nn)) < NSMALL and (zb != 'E0')

    def ld2(x_, k):
        with np.errstate(all='ignore'):
            z_ = T(float(np.ldexp(np.float64(x_[0]), int(k))))
            ek = float(np.ldexp(np.float64(x_[1]), int(k)))
        return (z_, ek + tr.R(z_))
    mCt = (T(mC), 0.0)
    if small:
        zs = (T(T(Nn * T(2.0 ** SZ)) / p2), 4 * u * abs(float(T(T(Nn * T(2.0 ** SZ)) / p2))))
    K_ = KTERMS[dt]

    def X(m_, e_):
        if bigE:
            return ld2(tr.sub(tr.div(tr.mul(mCt, Rt), m_), ld2(tr.div(mCt, m_), -E)), E + eC - e_)
        if small:
            return ld2(tr.div(tr.mul(mCt, zs), m_), eC - e_ - SZ)
        return ld2(tr.div(tr.mul(mCt, z), m_), eC - e_)
    if small:
        two = tr.inp(2.0)
        tz = tr.add(two, z)
        s_ = tr.div(z, tz)
        ss = tr.mul(s_, s_)
        qz = tr.const(Fr(1, 2 * K_ + 1))
        for k in range(K_ - 1, 0, -1):
            qz = tr.add(tr.mul(qz, ss), tr.const(Fr(1, 2 * k + 1)))
        qz = tr.extra(qz, q_tail_bound(float(ss[0]), K_))
        phh = tr.div(tr.sub(one, tr.div(tr.mul(tr.mul(two, z), qz), tr.mul(tz, tz))), tz)
        kl1 = ld2(tr.mul(tr.mul(tr.mul(mCt, zs), zs), phh), eC - 2 * SZ)
        sz = 's'
    elif bigE and (not SEG_LO <= float(z[0]) <= SEG_HI):
        kl1 = ld2(tr.sub(tr.mul(mCt, Rt), ld2(tr.mul(mCt, tr.add(one, LZ)), -E)), E + eC)
        sz = 'X'
    else:
        kl1, _, sz = _tr_aphi(tr, aq, z, LZ, dt)
    md, ed = math.frexp(float(d[0]))
    mAt = (T(mA), 0.0)
    mdt = ld2(d, -ed)
    alinf = math.isinf(float(al[0]))
    if alinf:
        pha = al
        kl2 = tr.sub(ld2(tr.div(mdt, mAt), ed - eA - 1), tr.mul(half, Tl))
        sa = 'I'
    else:
        kl2, pha, sa = _tr_aphi(tr, half, al, Tl, dt)
    near = abs(float(Tl[0])) <= T_NEAR
    if near:
        Bq = zero
        Lq = zero
        for c_i, v_i in GLCV[N]:
            tau = tr.mul(Tl, tr.const(c_i))
            Wt = tr.mul(Tl, tr.const(v_i))
            s = tr.mul(ap, tr.fn('exp', tau))
            gg = _tr_g(tr, s, dt)
            Bq = tr.add(Bq, tr.mul(tr.mul(tr.mul(Wt, tr.inp(float(2 ** SBQ[dt]))), tr.fn('expm1', tr.sub(Tl, tau))), gg))
            Lq = tr.add(Lq, tr.mul(tr.mul(Wt, tr.fn('exp', tr.sub(zero, tau))), gg))
        Bq = ld2(Bq, -SBQ[dt])
        Bq = tr.extra(Bq, QTAIL[dt] * abs(float(Bq[0])) * 1.5)
        Lq = tr.extra(Lq, QTAIL[dt] * abs(float(Lq[0])) * 1.5)
        Bw = Bq
        dL = tr.add(tr.div(Lq, ap), tr.div(tr.mul(half, al), aq))
        fb = 'N'
    else:
        peel = max(apv, aqv) < 0.5
        Wp_, Wpp, *VP = _tr_wpair(tr, ap, dt, peel)
        Wq_, Wpq, *VQ = _tr_wpair(tr, aq, dt, peel)
        if peel:
            Vp_, Vpp = VP
            Vq_, Vpq = VQ
            Bw = tr.add(kl2 if alinf else tr.mul(half, pha), tr.sub(tr.sub(Vq_, Vp_), tr.mul(d, Vpp)))
            alq = ld2(tr.div(tr.div(mdt, mAt), mCt), ed - eA - eC) if alinf else tr.div(al, aq)
            dL = tr.add(alq, tr.sub(Vpq, Vpp))
            fb = 'Fv'
        else:
            Bw = tr.sub(tr.sub(Wq_, Wp_), tr.mul(d, Wpp))
            dL = tr.sub(tr.sub(Wpq, tr.div(half, aq)), tr.sub(Wpp, tr.div(half, ap)))
            fb = 'Fw'
    kl = tr.add(tr.add(tr.clamp0(kl1), tr.clamp0(kl2)), tr.clamp0(Bw))
    hg = tr.add(half, _tr_g(tr, ap, dt))
    sa_extra = ''
    hm = tr.div(tr.mul(tr.div(mdt, mAt), hg), mAt)
    t1 = X(mAt, eA)
    t2 = ld2(hm, ed - 2 * eA)
    if math.isfinite(float(t1[0])) and math.isfinite(float(t2[0])):
        g_ap = tr.sub(t1, t2)
    else:
        e1 = max(E, 0) + eC - eA
        S = e1 if float(d[0]) == 0 else max(e1, ed - 2 * eA)
        g_ap = ld2(tr.sub(X(mAt, eA + S), ld2(hm, ed - 2 * eA - S)), S)
        sa_extra = '+S'
    g_bp = tr.sub(zero, X((T(mD), 0.0), eD))
    g_aq = tr.sub(dL, LZ)
    g_bq = X((T(mB), 0.0), eB)
    return ({'KL': kl, 'P.concentration': g_ap, 'P.rate': g_bp, 'Q.concentration': g_aq, 'Q.rate': g_bq}, {'_branch': f'{fb}/{zb}/{sz}{sa}{sa_extra}'})

def replay(ap, bp, aq, bq, dt):
    out, flags = _trace(ap, bp, aq, bq, dt, 1)
    res = {k: float(v[0]) for k, v in out.items()}
    res.update(flags)
    res['_in'] = (float(ap), float(bp), float(aq), float(bq))
    return res

def running_bounds(v, dt, lam):
    out, _ = _trace(*v['_in'], dt, max(lam, lam_pos(dt)))
    return {k: math.inf if math.isnan(float(e)) else SAFETY * float(e) for k, (_, e) in out.items()}
