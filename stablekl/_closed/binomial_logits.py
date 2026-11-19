"""Repaired KL divergence between two Binomial (logits) distributions: formula, analytic gradients, and the
entry point run_rewrite_torch used by the benchmark (dtype in, compute dtype, outputs per component).
The derivation is in the paper's appendix on interface A."""
from __future__ import annotations
import math
from fractions import Fraction as Fr
import numpy as np
FMT = {'fp16': dict(np=np.float16, u=2.0 ** (-11), eta=2.0 ** (-24), minnorm=2.0 ** (-14), emax=15), 'fp32': dict(np=np.float32, u=2.0 ** (-24), eta=2.0 ** (-149), minnorm=2.0 ** (-126), emax=127), 'fp64': dict(np=np.float64, u=2.0 ** (-53), eta=2.0 ** (-1074), minnorm=2.0 ** (-1022), emax=1023)}
SAFETY = 1.02
KTERMS = {'fp16': 9, 'fp32': 17, 'fp64': 36}
TSUB = {'fp16': 9.70406052783923, 'fp32': 87.3365447505531, 'fp64': 708.3964185322641}
SEG_LO, SEG_HI = (-0.75, 3.0)
OFFICIAL_FN = '_kl_binomial_binomial'
GRAD_TRUTH = 'frozen_interval'
TRANSCENDENTAL_FUNCS = ('log1p', 'log', 'expm1', 'exp')

def q_tail_bound(z, K):
    return z ** K / ((2 * K + 3) * (1 - z))
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    _DT = {torch.float16: 'fp16', torch.float32: 'fp32', torch.float64: 'fp64'}

    def _phiS(xs, s, K):
        z = s * s
        q = torch.full_like(s, 1.0 / (2 * K + 1))
        for k in range(K - 1, 0, -1):
            q = q * z + torch.full_like(s, 1.0 / (2 * k + 1))
        return xs * s - torch.full_like(s, 2.0) * (s * z) * q

    def _phiSn(xs, s, K, n):
        z = s * s
        q = torch.full_like(s, 1.0 / (2 * K + 1))
        for k in range(K - 1, 0, -1):
            q = q * z + torch.full_like(s, 1.0 / (2 * k + 1))
        n2 = n * 2.0
        return n * xs * s - n2 * s * s * s * q

    def _inseg(x):
        return torch.isfinite(x) & (x >= SEG_LO) & (x <= SEG_HI)

    def _sig(x):
        e = torch.exp(torch.zeros_like(x) - x.abs())
        one = torch.ones_like(x)
        return torch.where(x >= 0, one / (one + e), e / (one + e))

    def _sp(x):
        return torch.clamp_min(x, 0) + torch.log1p(torch.exp(torch.zeros_like(x) - x.abs()))

    def _LD(a, b):
        bn = (a < 0) & (b < 0)
        return torch.where(bn, a - b + (_sp(b) - _sp(a)), _sp(torch.zeros_like(b) - b) - _sp(torch.zeros_like(a) - a))

    def _mulsig(F, x, T1):
        one = torch.ones_like(x)
        e = torch.exp(torch.clamp_max(x, 0))
        h2 = torch.exp(torch.clamp_max(x, 0) * 0.5)
        h4 = torch.exp(torch.clamp_max(x, 0) * 0.25)
        s2 = F * h2 * h2 / (one + e)
        s4 = F * h4 * h4 * h4 * h4 / (one + e)
        return torch.where(x >= -T1, _sig(x) * F, torch.where(x >= -2 * T1, s2, s4))

    def _mulsig_n(F, x, n, T1):
        one = torch.ones_like(x)
        e = torch.exp(torch.clamp_max(x, 0))
        h2 = torch.exp(torch.clamp_max(x, 0) * 0.5)
        h4 = torch.exp(torch.clamp_max(x, 0) * 0.25)
        s2 = F * h2 * (n * h2) / (one + e)
        s4 = F * h4 * (n * h4) * h4 * h4 / (one + e)
        return torch.where(x >= -T1, _sig(x) * n * F, torch.where(x >= -2 * T1, s2, s4))

    def _nsp(n, x, T1):
        one = torch.ones_like(x)
        zero = torch.zeros_like(x)
        t = torch.exp(torch.clamp_max(x, 0))
        F = torch.where(t > 0, torch.log1p(t) * (one + t) / torch.where(t > 0, t, one), one)
        return torch.where(x < 0, _mulsig_n(F, x, n, T1), n * _sp(torch.clamp_min(x, 0)))

    def _tie_round(A, C, tgt):
        prec, emin = TIE_FMT[tgt]
        s = A + C
        e = A - s + C
        one = torch.ones_like(s)
        _, ex = torch.frexp(s)
        ulp = torch.ldexp(one, torch.clamp_min(ex - 1, emin) - (prec - 1))
        r = s.to(_TD[tgt]).to(s.dtype)
        r = torch.where(torch.isinf(r) & torch.isfinite(s), torch.sign(s) * TIE_MAX[tgt], r)
        mid = ((s - r).abs() * 2 == ulp) & (e != 0)
        return torch.where(mid, torch.nextafter(s, s + torch.sign(e) * s.abs()), s)

    def _tie_aware(z, w, g_z, g_w, tgt, n):
        zero = torch.zeros_like(z)
        tz_ = (z.abs() <= TIE_TAU) & (z != w) & torch.isfinite(w)
        tw_ = tz_ & (w.abs() <= TIE_TAU)
        zz = torch.where(tz_, z, zero)
        wP = torch.where(tz_, w, zero)
        ww = torch.where(tw_, w, zero)
        zq = torch.where(tw_, z, zero)
        z2 = zz * zz
        bp = torch.full_like(z, TIE_BP[4])
        for c in TIE_BP[3::-1]:
            bp = bp * z2 + c
        bp = bp * z2
        hs = [torch.ones_like(z)]
        zk = torch.ones_like(z)
        for k in range(1, 11):
            zk = zk * zq
            hs.append(zk + ww * hs[k - 1])
        bq = torch.full_like(z, TIE_BQ[4]) * hs[10]
        for k in (3, 2, 1, 0):
            bq = bq + TIE_BQ[k] * hs[2 * k + 2]
        nz = n * zz
        nw = n * wP
        sP = nz - nw
        bb = sP - nz
        eP = nz - (sP - bb) + (zero - nw - bb)
        AP = sP * 0.25
        APl = eP * 0.25
        AQ = zero - AP
        AQl = zero - APl
        tz = _tie_round(AP, APl + AP * bp, tgt)
        tw = _tie_round(AQ, AQl + AQ * bq, tgt)
        return (torch.where(tz_, tz, g_z), torch.where(tw_, tw, g_w))
    _TD = {'fp16': torch.float16, 'fp32': torch.float32}

    class BinomialLogitsKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, n, z, w, tgt=None):
            dt = _DT[z.dtype]
            K = KTERMS[dt]
            one = torch.ones_like(z)
            zero = torch.zeros_like(z)
            half = torch.full_like(z, 0.5)
            two = torch.full_like(z, 2.0)
            z_in, w_in = (z, w)
            zi, wi = (torch.isinf(z), torch.isinf(w))
            z = torch.where(zi | wi, zero, z)
            w = torch.where(zi | wi, zero, w)
            p, omp = (_sig(z), _sig(zero - z))
            q, omq = (_sig(w), _sig(zero - w))
            dl = z - w
            fdl = torch.isfinite(dl)
            dlh = z * half - w * half
            small = fdl & (dl.abs() <= 2)
            dls = torch.where(small, dl, zero)
            T1 = TSUB[dt]
            Em = zero - torch.expm1(zero - dls)
            Ds = torch.where(w > T1, _mulsig(p * Em, zero - w, T1), torch.where(z < -T1, _mulsig(omq * Em, z, T1), p * omq * Em))
            bp = (z >= 0) & (w >= 0)
            bn = (z < 0) & (w < 0)
            zgw = z >= w
            ez = torch.exp(zero - z.abs())
            ew = torch.exp(zero - w.abs())
            ea = torch.exp(torch.clamp_max(w - z, 0))
            eb = torch.exp(torch.clamp_max(z - w, 0))
            Dc = torch.where(zgw, _mulsig(one - ea * (one + ew) / (one + ez), zero - w, T1), zero - _mulsig(one - eb * (one + ez) / (one + ew), zero - z, T1))
            Dd = torch.where(zgw, _mulsig(one - ea * (one + ez) / (one + ew), z, T1), zero - _mulsig(one - eb * (one + ew) / (one + ez), w, T1))
            useC = bp & (torch.maximum(z, w) > T1)
            useD = bn & (torch.minimum(z, w) < -T1)
            D = torch.where(small, Ds, torch.where(bp, torch.where(useC, Dc, omq - omp), torch.where(useD, Dd, p - q)))
            nEm = n * Em
            Dsn = torch.where(w > T1, _mulsig(p * nEm, zero - w, T1), torch.where(z < -T1, _mulsig(omq * nEm, z, T1), p * nEm * omq))
            Dcn = torch.where(zgw, _mulsig(n * (one - ea * (one + ew) / (one + ez)), zero - w, T1), zero - _mulsig(n * (one - eb * (one + ez) / (one + ew)), zero - z, T1))
            Ddn = torch.where(zgw, _mulsig(n * (one - ea * (one + ez) / (one + ew)), z, T1), zero - _mulsig(n * (one - eb * (one + ew) / (one + ez)), w, T1))
            Dn = torch.where(small, Dsn, torch.where(bp, torch.where(useC, Dcn, n * (omq - omp)), torch.where(useD, Ddn, n * (p - q))))
            L1 = _LD(w, z)
            L2 = _LD(zero - w, zero - z)
            psub = z < -T1
            osub = z > T1
            x1 = torch.where(psub, torch.expm1(L1), (zero - Dn) / (n * p))
            x2 = torch.where(osub, torch.expm1(L2), Dn / (n * omp))
            in1, in2 = (_inseg(x1), _inseg(x2))
            s1 = torch.where(in1, torch.where(psub, x1 / (two + x1), (zero - Dn) / (n * (p + q))), zero)
            s2 = torch.where(in2, torch.where(osub, x2 / (two + x2), Dn / (n * (omp + omq))), zero)
            phi1 = _phiSn(torch.where(in1, x1, zero), s1, K, n)
            phi2 = _phiSn(torch.where(in2, x2, zero), s2, K, n)
            sc1 = psub & (in1 | (x1 < SEG_LO))
            sc2 = osub & (in2 | (x2 < SEG_LO))
            t1x = zero - Dn - n * p * L1
            t1x = torch.where(torch.isfinite(t1x), t1x, n * (zero - D - p * L1))
            t2x = Dn - n * omp * L2
            t2x = torch.where(torch.isfinite(t2x), t2x, n * (D - omp * L2))
            t1o = _mulsig(n, w, T1) - _mulsig_n(one + L1, z, n, T1)
            t2o = _mulsig(n, zero - w, T1) - _mulsig_n(one + L2, zero - z, n, T1)
            t1 = torch.where(sc1, torch.where(in1, _mulsig(phi1, z, T1), _mulsig_n(x1 - L1, z, n, T1)), torch.where(in1, p * phi1, torch.where(psub, t1o, t1x)))
            t2 = torch.where(sc2, torch.where(in2, _mulsig(phi2, zero - z, T1), _mulsig_n(x2 - L2, zero - z, n, T1)), torch.where(in2, omp * phi2, torch.where(osub, t2o, t2x)))
            kl = torch.clamp_min(t1, 0) + torch.clamp_min(t2, 0)
            T1 = TSUB[dt]
            az = z.abs()
            e = torch.exp(zero - az)
            ope2 = (one + e) * (one + e)
            dd = torch.where(fdl, dl, dlh)
            h2 = torch.exp(zero - az * half)
            h4 = torch.exp(zero - az * torch.full_like(z, 0.25))
            pp = p * omp
            g1 = torch.where(dd.abs() >= 1, pp * dd * n, pp * (n * dd))
            g2 = dd * h2 * (n * h2) / ope2
            g4 = dd * h4 * (n * h4) * h4 * h4 / ope2
            gz = torch.where(az <= T1, g1, torch.where(az <= 2 * T1, g2, g4))
            g_z = torch.where(fdl, gz, gz * two)
            g_w = zero - Dn
            if tgt is not None:
                g_z, g_w = _tie_aware(z, w, g_z, g_w, tgt, n)
            same = z_in == w_in
            kinf = wi & ~same
            wf = torch.where(wi, zero, w_in)
            zneg = z_in < 0
            klZ = torch.where(zneg, _nsp(n, wf, T1), _nsp(n, zero - wf, T1))
            gwZ = torch.where(zneg, _mulsig(n, wf, T1), zero - _mulsig(n, zero - wf, T1))
            kl = torch.where(same, zero, torch.where(kinf, torch.full_like(z, math.inf), torch.where(zi, klZ, kl)))
            g_z = torch.where(same | zi | wi, zero, g_z)
            g_w = torch.where(same | kinf, zero, torch.where(zi, gwZ, g_w))
            ctx.save_for_backward(g_z, g_w)
            return kl

        @staticmethod
        def backward(ctx, g, *rest):
            a, b = ctx.saved_tensors
            return (None, g * a, g * b, None)
TIE_TAU = 1.0 / 16
REPLAY_TARGET = True
TIE_BP_FR = (Fr(-1, 4), Fr(1, 24), Fr(-17, 2880), Fr(31, 40320), Fr(-691, 7257600))
TIE_BQ_FR = (Fr(-1, 12), Fr(1, 120), Fr(-17, 20160), Fr(31, 362880), Fr(-691, 79833600))
TIE_BP = tuple((float(c) for c in TIE_BP_FR))
TIE_BQ = tuple((float(c) for c in TIE_BQ_FR))
TIE_TRUNC = 1e-19
TIE_FMT = {'fp16': (11, -14), 'fp32': (24, -126)}
TIE_MAX = {'fp16': 65504.0, 'fp32': 3.4028234663852886e+38}
NAME = 'binomial_logits'
PARAMS = ('total_count', 'logits')
COMPS = ['KL', 'P.logits', 'Q.logits']
EXPECTED_ROWS = {'fp16': 143715, 'fp32': 202000}
TRANSCENDENTAL_COMPS = ('KL', 'P.logits', 'Q.logits')

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    n = to_t(P['total_count'], dt_in).to(TD[cdt])
    leaves = [to_t(x, dt_in).to(TD[cdt]).requires_grad_(True) for x in (P['logits'], Q['logits'])]
    kl = BinomialLogitsKLC.apply(n, *leaves, dt_in if cdt != dt_in else None)
    kl.sum().backward()
    vals = {'KL': kl.detach(), 'P.logits': leaves[0].grad.detach(), 'Q.logits': leaves[1].grad.detach()}
    return {k: v.cpu().numpy().astype(NPF[cdt]) for k, v in vals.items()}

def exact_grads(n_p, z, n_q, w):
    return None

def identity(n_p, z, n_q, w):
    return z == w and n_p == n_q

class _Tr:

    def __init__(self, dt, lam):
        f = FMT[dt]
        self.dt = dt
        self.T = f['np']
        self.u = f['u']
        self.mn = f['minnorm']
        self.h = f['eta'] / 2 if f['eta'] / 2 > 0 else f['eta']
        self.lam = lam
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

    def neg(self, a):
        return (self.T(self.T(0) - a[0]), a[1])

    def mul(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] * b[0])
        return (z, self.A(b[0]) * a[1] + self.A(a[0]) * b[1] + self.R(z))

    def scale2(self, a, c):
        with np.errstate(all='ignore'):
            z = self.T(a[0] * self.T(c))
        return (z, abs(c) * a[1] + self.h * self.sb(z) * (c < 1))

    def div(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] / b[0])
        bb = abs(float(b[0]))
        return (z, (a[1] + self.A(z) * b[1]) / bb + self.R(z) if bb > 0 else float('inf'))

    def fn(self, name, a):
        with np.errstate(all='ignore'):
            z = self.T(getattr(np, name)(a[0]))
        xv = float(a[0])
        der = {'expm1': lambda: self.A(z) + 1.0, 'exp': lambda: self.A(z), 'log1p': lambda: 1.0 / (1.0 + xv), 'log': lambda: 1.0 / xv}[name]()
        return (z, der * a[1] + self.lam * self.R(z))

    def absv(self, a):
        return (self.T(abs(a[0])), a[1])

    def clamp0(self, a):
        return (max(a[0], self.T(0)), a[1])

    def clamp_max0(self, a):
        return (min(a[0], self.T(0)), a[1])

    def extra(self, a, e):
        return (a[0], a[1] + e)

def _tr_phiS(tr, x, s, dt):
    K = KTERMS[dt]
    z = tr.mul(s, s)
    q = tr.const(Fr(1, 2 * K + 1))
    for k in range(K - 1, 0, -1):
        q = tr.add(tr.mul(q, z), tr.const(Fr(1, 2 * k + 1)))
    q = tr.extra(q, q_tail_bound(float(z[0]), K))
    return tr.sub(tr.mul(x, s), tr.mul(tr.scale2(tr.mul(s, z), 2.0), q))

def _seg(a):
    v = float(a[0])
    return math.isfinite(v) and SEG_LO <= v <= SEG_HI

def _tr_phiSn(tr, x, s, dt, nn):
    K = KTERMS[dt]
    z = tr.mul(s, s)
    q = tr.const(Fr(1, 2 * K + 1))
    for k in range(K - 1, 0, -1):
        q = tr.add(tr.mul(q, z), tr.const(Fr(1, 2 * k + 1)))
    q = tr.extra(q, q_tail_bound(float(z[0]), K))
    n2 = tr.scale2(nn, 2.0)
    return tr.sub(tr.mul(tr.mul(nn, x), s), tr.mul(tr.mul(tr.mul(tr.mul(n2, s), s), s), q))

def _tr_tie(tr, z, w, tgt, nn, both):
    zz = z
    z2 = tr.mul(zz, zz)
    bp = tr.const(TIE_BP_FR[4])
    for c in TIE_BP_FR[3::-1]:
        bp = tr.add(tr.mul(bp, z2), tr.const(c))
    bp = tr.mul(bp, z2)
    T = tr.T
    nz, nw = (tr.mul(nn, zz), tr.mul(nn, w))
    sP = T(nz[0] - nw[0])
    bb = T(sP - nz[0])
    eP = T(T(nz[0] - T(sP - bb)) + T(T(T(0) - nw[0]) - bb))
    AP = (T(sP * T(0.25)), 0.0)
    APl = T(eP * T(0.25))
    AQ = (T(T(0) - AP[0]), 0.0)
    AQl = T(T(0) - APl)
    prec, emin = TIE_FMT[tgt]
    tags = ''

    def rnd(A, C):
        nonlocal tags
        sv = tr.add(A, C)
        e = T(T(A[0] - sv[0]) + C[0])
        m, ex = np.frexp(float(sv[0]))
        ulp = 2.0 ** (max(ex - 1, emin) - (prec - 1))
        with np.errstate(all='ignore'):
            r = float(T(np.array(sv[0]).astype(np.float16 if tgt == 'fp16' else np.float32)))
        if math.isinf(r) and math.isfinite(float(sv[0])):
            r = math.copysign(TIE_MAX[tgt], float(sv[0]))
        if abs(float(sv[0]) - r) * 2 == ulp and e != 0:
            nv = np.nextafter(sv[0], T(sv[0] + np.sign(e) * abs(sv[0])))
            tags += 'n'
            return (nv, sv[1] + abs(float(nv) - float(sv[0])) + TIE_TRUNC * abs(float(A[0])))
        tags += '.'
        return (sv[0], sv[1] + TIE_TRUNC * abs(float(A[0])))
    cz = tr.mul(AP, bp)
    cz = (T(APl + cz[0]), cz[1] + tr.u * abs(float(APl + cz[0])))
    gz = rnd(AP, cz)
    gw = None
    if both:
        ww = w
        hs = [tr.inp(1.0)]
        zk = tr.inp(1.0)
        for k in range(1, 11):
            zk = tr.mul(zk, zz)
            hs.append(tr.add(zk, tr.mul(ww, hs[k - 1])))
        bq = tr.mul(tr.const(TIE_BQ_FR[4]), hs[10])
        for k in (3, 2, 1, 0):
            bq = tr.add(bq, tr.mul(tr.const(TIE_BQ_FR[k]), hs[2 * k + 2]))
        cw = tr.mul(AQ, bq)
        cw = (T(AQl + cw[0]), cw[1] + tr.u * abs(float(AQl + cw[0])))
        gw = rnd(AQ, cw)
    return (gz, gw, 'T' + tags)

def _trace(n, z, w, dt, lam, tgt=None):
    tr = _Tr(dt, lam)
    T = tr.T
    nn = tr.inp(n)
    z, w = (tr.inp(z), tr.inp(w))
    one = tr.inp(1.0)

    def sig(x):
        e = tr.fn('exp', tr.neg(tr.absv(x)))
        return tr.div(one, tr.add(one, e)) if float(x[0]) >= 0 else tr.div(e, tr.add(one, e))

    def sp(x):
        return tr.add(tr.clamp0(x), tr.fn('log1p', tr.fn('exp', tr.neg(tr.absv(x)))))

    def LD(a, b):
        if float(a[0]) < 0 and float(b[0]) < 0:
            return tr.add(tr.sub(a, b), tr.sub(sp(b), sp(a)))
        return tr.sub(sp(tr.neg(b)), sp(tr.neg(a)))
    T1 = TSUB[dt]
    two = tr.inp(2.0)

    def mulsig(F, x):
        xv = float(x[0])
        if xv >= -T1:
            return tr.mul(sig(x), F)
        e = tr.fn('exp', x)
        ope = tr.add(one, e)
        if xv >= -2 * T1:
            h = tr.fn('exp', tr.scale2(x, 0.5))
            return tr.div(tr.mul(tr.mul(F, h), h), ope)
        h = tr.fn('exp', tr.scale2(x, 0.25))
        return tr.div(tr.mul(tr.mul(tr.mul(tr.mul(F, h), h), h), h), ope)

    def mulsig_n(F, x):
        xv = float(x[0])
        if xv >= -T1:
            return tr.mul(tr.mul(sig(x), nn), F)
        e = tr.fn('exp', x)
        ope = tr.add(one, e)
        if xv >= -2 * T1:
            h = tr.fn('exp', tr.scale2(x, 0.5))
            return tr.div(tr.mul(tr.mul(F, h), tr.mul(nn, h)), ope)
        h = tr.fn('exp', tr.scale2(x, 0.25))
        return tr.div(tr.mul(tr.mul(tr.mul(tr.mul(F, h), tr.mul(nn, h)), h), h), ope)
    zv0, wv0 = (float(z[0]), float(w[0]))
    zero0 = tr.inp(0.0)
    if zv0 == wv0:
        return ({'KL': zero0, 'P.logits': zero0, 'Q.logits': zero0}, {'_branch': 'id'})
    if math.isinf(wv0):
        return ({'KL': tr.inp(math.inf), 'P.logits': zero0, 'Q.logits': zero0}, {'_branch': 'E/inf'})
    if math.isinf(zv0):

        def nsp(x):
            if float(x[0]) >= 0:
                return tr.mul(nn, sp(tr.clamp0(x)))
            t = tr.fn('exp', tr.clamp_max0(x))
            F = tr.div(tr.mul(tr.fn('log1p', t), tr.add(one, t)), t) if float(t[0]) > 0 else (one[0], 0.0)
            return mulsig_n(F, x)
        if zv0 < 0:
            return ({'KL': nsp(w), 'P.logits': zero0, 'Q.logits': mulsig(nn, w)}, {'_branch': 'E/p0'})
        return ({'KL': nsp(tr.neg(w)), 'P.logits': zero0, 'Q.logits': tr.neg(mulsig(nn, tr.neg(w)))}, {'_branch': 'E/p1'})
    p, omp = (sig(z), sig(tr.neg(z)))
    q, omq = (sig(w), sig(tr.neg(w)))
    dl = tr.sub(z, w)
    fdl = bool(np.isfinite(dl[0]))
    zv, wv = (float(z[0]), float(w[0]))
    if fdl and abs(float(dl[0])) <= 2:
        Em = tr.neg(tr.fn('expm1', tr.neg(dl)))
        nEm = tr.mul(nn, Em)
        if wv > T1:
            D = mulsig(tr.mul(p, Em), tr.neg(w))
            Dn = mulsig(tr.mul(p, nEm), tr.neg(w))
            bd = 'sw'
        elif zv < -T1:
            D = mulsig(tr.mul(omq, Em), z)
            Dn = mulsig(tr.mul(omq, nEm), z)
            bd = 'sz'
        else:
            D = tr.mul(tr.mul(p, omq), Em)
            Dn = tr.mul(tr.mul(p, nEm), omq)
            bd = 's'
    elif zv >= 0 and wv >= 0:
        if max(zv, wv) > T1:
            ez, ew = (tr.fn('exp', tr.neg(tr.absv(z))), tr.fn('exp', tr.neg(tr.absv(w))))
            if zv >= wv:
                ea = tr.fn('exp', tr.clamp_max0(tr.sub(w, z)))
                om = tr.sub(one, tr.div(tr.mul(ea, tr.add(one, ew)), tr.add(one, ez)))
                D = mulsig(om, tr.neg(w))
                Dn = mulsig(tr.mul(nn, om), tr.neg(w))
            else:
                eb = tr.fn('exp', tr.clamp_max0(tr.sub(z, w)))
                om = tr.sub(one, tr.div(tr.mul(eb, tr.add(one, ez)), tr.add(one, ew)))
                D = tr.neg(mulsig(om, tr.neg(z)))
                Dn = tr.neg(mulsig(tr.mul(nn, om), tr.neg(z)))
            bd = 'C'
        else:
            D = tr.sub(omq, omp)
            Dn = tr.mul(nn, D)
            bd = 'c'
    elif zv < 0 and wv < 0 and (min(zv, wv) < -T1):
        ez, ew = (tr.fn('exp', tr.neg(tr.absv(z))), tr.fn('exp', tr.neg(tr.absv(w))))
        if zv >= wv:
            ea = tr.fn('exp', tr.clamp_max0(tr.sub(w, z)))
            om = tr.sub(one, tr.div(tr.mul(ea, tr.add(one, ez)), tr.add(one, ew)))
            D = mulsig(om, z)
            Dn = mulsig(tr.mul(nn, om), z)
        else:
            eb = tr.fn('exp', tr.clamp_max0(tr.sub(z, w)))
            om = tr.sub(one, tr.div(tr.mul(eb, tr.add(one, ew)), tr.add(one, ez)))
            D = tr.neg(mulsig(om, w))
            Dn = tr.neg(mulsig(tr.mul(nn, om), w))
        bd = 'Dd'
    else:
        D = tr.sub(p, q)
        Dn = tr.mul(nn, D)
        bd = 'd'

    def term(sub, L, x, c, cx, cq, Dlin, Dlin_n, snum, sden):
        if _seg(x):
            s = tr.div(x, tr.add(two, x)) if sub else tr.div(snum, sden)
            ph = _tr_phiSn(tr, x, s, dt, nn)
            return (mulsig(ph, cx), 's') if sub else (tr.mul(c, ph), 'S')
        if sub and float(x[0]) < SEG_LO:
            return (mulsig_n(tr.sub(x, L), cx), 'x')
        if sub:
            return (tr.sub(mulsig(nn, cq), mulsig_n(tr.add(one, L), cx)), 'o')
        t = tr.sub(Dlin_n, tr.mul(tr.mul(nn, c), L))
        if not math.isfinite(float(t[0])):
            return (tr.mul(nn, tr.sub(Dlin, tr.mul(c, L))), 'Xf')
        return (t, 'X')
    L1 = LD(w, z)
    L2 = LD(tr.neg(w), tr.neg(z))
    psub = float(z[0]) < -T1
    osub = float(z[0]) > T1
    x1 = tr.fn('expm1', L1) if psub else tr.div(tr.neg(Dn), tr.mul(nn, p))
    x2 = tr.fn('expm1', L2) if osub else tr.div(Dn, tr.mul(nn, omp))
    t1, c1 = term(psub, L1, x1, p, z, w, tr.neg(D), tr.neg(Dn), tr.neg(Dn), tr.mul(nn, tr.add(p, q)))
    t2, c2 = term(osub, L2, x2, omp, tr.neg(z), tr.neg(w), D, Dn, Dn, tr.mul(nn, tr.add(omp, omq)))
    KL = tr.add(tr.clamp0(t1), tr.clamp0(t2))
    dd = dl if fdl else tr.sub(tr.scale2(z, 0.5), tr.scale2(w, 0.5))
    az = tr.absv(z)
    T1 = TSUB[dt]
    if float(az[0]) <= T1:
        pp = tr.mul(p, omp)
        gz = tr.mul(tr.mul(pp, dd), nn) if abs(float(dd[0])) >= 1 else tr.mul(pp, tr.mul(nn, dd))
        bz = '1'
    else:
        e = tr.fn('exp', tr.neg(az))
        ope = tr.add(one, e)
        ope2 = tr.mul(ope, ope)
        if float(az[0]) <= 2 * T1:
            h = tr.fn('exp', tr.neg(tr.scale2(az, 0.5)))
            gz = tr.div(tr.mul(tr.mul(dd, h), tr.mul(nn, h)), ope2)
            bz = '2'
        else:
            h = tr.fn('exp', tr.neg(tr.scale2(az, 0.25)))
            gz = tr.div(tr.mul(tr.mul(tr.mul(tr.mul(dd, h), tr.mul(nn, h)), h), h), ope2)
            bz = '4'
    if not fdl:
        gz = tr.scale2(gz, 2.0)
    gw = tr.neg(Dn)
    br = f"{c1}{c2}/{bd}/g{bz}{('' if fdl else '/h')}"
    if tgt is not None and abs(zv) <= TIE_TAU and (zv != wv) and math.isfinite(wv):
        both = abs(wv) <= TIE_TAU
        gz, gw2, tt = _tr_tie(tr, z, w, tgt, nn, both)
        br += '/' + tt
        if both:
            gw = gw2
    return ({'KL': KL, 'P.logits': gz, 'Q.logits': gw}, {'_branch': br})

def replay(n, z, n_q, w, dt, tgt=None):
    out, flags = _trace(n, z, w, dt, 1, tgt)
    res = {k: float(v[0]) for k, v in out.items()}
    res.update(flags)
    res['_in'] = (float(n), float(z), float(w))
    return res

def running_bounds(v, dt, lam, tgt=None):
    out, _ = _trace(*v['_in'], dt, lam, tgt)
    return {k: SAFETY * float(e) for k, (_, e) in out.items()}
