"""Repaired KL divergence between two Laplace distributions: formula, analytic gradients, and the
entry point run_rewrite_torch used by the benchmark (dtype in, compute dtype, outputs per component).
The derivation is in the paper's appendix on interface A."""
from __future__ import annotations
import math
from fractions import Fraction as Fr
import numpy as np
NAME = 'laplace'
PARAMS = ('loc', 'scale')
COMPS = ['KL', 'P.loc', 'P.scale', 'Q.loc', 'Q.scale']
EXPECTED_ROWS = {'fp16': 137692, 'fp32': 202000}
TRANSCENDENTAL_COMPS = ('KL', 'P.loc', 'P.scale', 'Q.loc', 'Q.scale')
TRANSCENDENTAL_FUNCS = ('log1p', 'log', 'expm1', 'exp')
GRAD_TRUTH = 'frozen_interval'
FMT = {'fp16': dict(np=np.float16, u=2.0 ** (-11), eta=2.0 ** (-24), minnorm=2.0 ** (-14), emax=15), 'fp32': dict(np=np.float32, u=2.0 ** (-24), eta=2.0 ** (-149), minnorm=2.0 ** (-126), emax=127), 'fp64': dict(np=np.float64, u=2.0 ** (-53), eta=2.0 ** (-1074), minnorm=2.0 ** (-1022), emax=1023)}
SAFETY = 1.02
KTERMS = {'fp16': 9, 'fp32': 17, 'fp64': 36}
KPSI = {'fp16': 7, 'fp32': 11, 'fp64': 18}
KCHI = {'fp16': 8, 'fp32': 12, 'fp64': 19}
MSC = {'fp16': 12, 'fp32': 25, 'fp64': 54}
SEG_LO, SEG_HI = (-0.75, 3.0)
PSI_C = [Fr(1, math.factorial(k + 2)) for k in range(40)]
CHI_C = [Fr(j + 1, math.factorial(j + 2)) for j in range(40)]

def q_tail_bound(z, K):
    return z ** K / ((2 * K + 3) * (1 - z))
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    _DT = {torch.float16: 'fp16', torch.float32: 'fp32', torch.float64: 'fp64'}

    class LaplaceKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, pl, ps, ql, qs):
            dt = _DT[ps.dtype]
            MINN = FMT[dt]['minnorm']
            one = torch.ones_like(ps)
            zero = torch.zeros_like(ps)
            half = torch.full_like(ps, 0.5)
            two = torch.full_like(ps, 2)
            inf = torch.full_like(ps, float('inf'))
            fin = lambda a, b: torch.where(torch.isnan(a), b, a)
            dm = pl - ql
            fdm = torch.isfinite(dm)
            dmh = pl * half - ql * half
            neg = torch.where(fdm, dm < 0, dmh < 0)
            dd = torch.where(fdm, dm, dmh).abs()
            t = torch.where(fdm, dd / ps, two * (dd / ps))
            w = torch.where(fdm, dd / qs, two * (dd / qs))
            ds = ps - qs
            x = ds / qs
            fx = torch.isfinite(x)
            xf = torch.where(fx, x, zero)
            inS = fx & (x >= SEG_LO) & (x <= SEG_HI)
            inA = fx & (x > SEG_HI)
            dss = torch.where(inS, ds, zero)
            xs = torch.where(inS, x, zero)
            ss = ps + qs
            fss = torch.isfinite(ss)
            s = torch.where(fss, dss / torch.where(fss, ss, one), dss * half / (ps * half + qs * half))
            z = s * s
            K = KTERMS[dt]
            q = torch.full_like(ps, 1.0 / (2 * K + 1))
            for k in range(K - 1, 0, -1):
                q = q * z + torch.full_like(ps, 1.0 / (2 * k + 1))
            phiS = xs * s - two * (s * z) * q
            LA = torch.log1p(torch.where(inA, x, zero))
            r = ps / qs
            rn = r >= MINN
            Lab = torch.log(ps) - torch.log(qs)
            LB = torch.where(rn, torch.log(torch.where(rn, r, one)), Lab)
            L = torch.where(inA, LA, LB)
            phi = torch.clamp_min(torch.where(inS, phiS, xf - L), 0)
            tsml = t <= 1
            ts = torch.where(tsml, t, zero)
            KP, KX = (KPSI[dt], KCHI[dt])
            P = torch.full_like(ps, float(PSI_C[KP - 1]))
            for k in range(KP - 2, -1, -1):
                P = torch.full_like(ps, float(PSI_C[k])) - ts * P
            X = torch.full_like(ps, float(CHI_C[KX - 1]))
            for j in range(KX - 2, -1, -1):
                X = torch.full_like(ps, float(CHI_C[j])) - ts * X
            m1 = zero - torch.expm1(zero - t)
            Et = torch.exp(zero - t)
            tf = torch.isfinite(t)
            tsub = t < MINN
            m = MSC[dt]
            up = torch.full_like(ps, 2.0 ** m)
            dn = torch.full_like(ps, 2.0 ** (-m))
            dl = torch.where(fdm, dd * up, dd * up * two)
            tb3 = dl / ps / qs * dn
            tb = torch.where(tsub, torch.where((w >= MINN) & torch.isfinite(w), w / ps, torch.where(torch.isfinite(tb3), tb3, t / qs)), t / qs)
            T2s = w * ts * P
            T2l = w - r * m1
            T2 = torch.where(tsml, T2s, T2l)
            kl = torch.where(fx, torch.clamp_min(phi + T2, 0), inf)
            g_loc = torch.where(tsub, tb, m1 / qs)
            g_pl = torch.where(neg, zero - g_loc, g_loc)
            g_ql = zero - g_pl
            chib = ts * X * tb
            y = ds / ps
            ab1 = torch.where(qs > ps, x / ps, y / qs)
            gS = fin(ab1 - chib, (y - ts * ts * X) / qs)
            Etp = torch.where(tf, Et * (one + t), zero)
            gL1 = Etp / qs - one / ps
            gL = torch.where(torch.isfinite(gL1), gL1, (Etp * r - one) / ps)
            g_ps = torch.where(tsml, gS, gL)
            T2b = torch.where(tsml, w * P * tb, T2l / qs)
            g_qs = fin(zero - (x / qs + T2b), zero - (x + T2) / qs)
            g_qs = torch.where(fx, g_qs, zero - inf)
            ctx.save_for_backward(g_pl, g_ps, g_ql, g_qs)
            return kl

        @staticmethod
        def backward(ctx, g):
            a, b, c, d = ctx.saved_tensors
            return (g * a, g * b, g * c, g * d)

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    leaves = [to_t(x, dt_in).to(TD[cdt]).requires_grad_(True) for x in (P['loc'], P['scale'], Q['loc'], Q['scale'])]
    kl = LaplaceKLC.apply(*leaves)
    kl.sum().backward()
    vals = {'KL': kl.detach()}
    for name, leaf in zip(('P.loc', 'P.scale', 'Q.loc', 'Q.scale'), leaves):
        vals[name] = leaf.grad.detach()
    return {k: v.cpu().numpy().astype(NPF[cdt]) for k, v in vals.items()}

def exact_grads(pl, ps, ql, qs):
    return None

def identity(pl, ps, ql, qs):
    return pl == ql and ps == qs

class _Tr:

    def __init__(self, dt, lam):
        f = FMT[dt]
        self.dt = dt
        self.T = f['np']
        self.u = f['u']
        self.h = f['eta'] / 2
        self.mn = f['minnorm']
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

    def extra(self, a, e):
        return (a[0], a[1] + e)

def _trace(pl, ps, ql, qs, dt, lam):
    tr = _Tr(dt, lam)
    T = tr.T
    MINN = FMT[dt]['minnorm']
    v = {}
    pl, ps, ql, qs = (tr.inp(pl), tr.inp(ps), tr.inp(ql), tr.inp(qs))
    fin = lambda a: bool(np.isfinite(a[0]))
    nan = lambda a: bool(np.isnan(a[0]))
    dm = tr.sub(pl, ql)
    fdm = fin(dm)
    if fdm:
        neg = bool(dm[0] < 0)
        dd = tr.absv(dm)
        t = tr.div(dd, ps)
        w = tr.div(dd, qs)
    else:
        dmh = tr.sub(tr.scale2(pl, 0.5), tr.scale2(ql, 0.5))
        neg = bool(dmh[0] < 0)
        dd = tr.absv(dmh)
        t = tr.scale2(tr.div(dd, ps), 2.0)
        w = tr.scale2(tr.div(dd, qs), 2.0)
    ds = tr.sub(ps, qs)
    x = tr.div(ds, qs)
    fx = fin(x)
    xv = float(x[0])
    inS = fx and SEG_LO <= xv <= SEG_HI
    inA = fx and xv > SEG_HI
    r = tr.div(ps, qs)
    br = 'S' if inS else 'A' if inA else 'B' if fx else 'INF'
    if inS:
        K = KTERMS[dt]
        ss = tr.add(ps, qs)
        if fin(ss):
            s = tr.div(ds, ss)
        else:
            s = tr.div(tr.scale2(ds, 0.5), tr.add(tr.scale2(ps, 0.5), tr.scale2(qs, 0.5)))
        z = tr.mul(s, s)
        q = tr.const(Fr(1, 2 * K + 1))
        for k in range(K - 1, 0, -1):
            q = tr.add(tr.mul(q, z), (T(1.0 / (2 * k + 1)), float(abs(Fr(float(T(1.0 / (2 * k + 1)))) - Fr(1, 2 * k + 1)))))
        q = tr.extra(q, q_tail_bound(float(z[0]), K))
        phit = tr.sub(tr.mul(x, s), tr.mul(tr.scale2(tr.mul(s, z), 2.0), q))
    elif fx:
        if inA:
            L = tr.fn('log1p', x)
        elif float(r[0]) >= MINN:
            L = tr.fn('log', r)
        else:
            L = tr.sub(tr.fn('log', ps), tr.fn('log', qs))
        phit = tr.sub(x, L)
    if fx:
        phi = tr.clamp0(phit)
    tv = float(t[0])
    tsml = tv <= 1
    tsub = tv < MINN
    tf = fin(t)
    if tsml:
        KP, KX = (KPSI[dt], KCHI[dt])
        P = tr.const(PSI_C[KP - 1])
        for k in range(KP - 2, -1, -1):
            P = tr.sub(tr.const(PSI_C[k]), tr.mul(t, P))
        P = tr.extra(P, abs(tv) ** KP * float(PSI_C[KP]))
        X = tr.const(CHI_C[KX - 1])
        for j in range(KX - 2, -1, -1):
            X = tr.sub(tr.const(CHI_C[j]), tr.mul(t, X))
        X = tr.extra(X, abs(tv) ** KX * float(CHI_C[KX]))
    m1 = tr.neg(tr.fn('expm1', tr.neg(t)))
    sb = ''
    if not tsub:
        tb = tr.div(t, qs)
    elif float(w[0]) >= MINN and fin(w):
        tb = tr.div(w, ps)
        sb = 'w'
    else:
        m = MSC[dt]
        dl = tr.scale2(dd, 2.0 ** m)
        if not fdm:
            dl = tr.scale2(dl, 2.0)
        tb = tr.scale2(tr.div(tr.div(dl, ps), qs), 2.0 ** (-m))
        sb = 'm'
        if not fin(tb):
            tb = tr.div(t, qs)
            sb = 't'
    if tsub:
        tb = tr.extra(tb, tr.A(tb[0]) * tv)
    T2 = tr.mul(tr.mul(w, t), P) if tsml else tr.sub(w, tr.mul(r, m1))
    if fx:
        KL = tr.clamp0(tr.add(phi, T2))
    else:
        KL = (T(np.inf), 0.0)
    g_loc = tb if tsub else tr.div(m1, qs)
    gpl = tr.neg(g_loc) if neg else g_loc
    gql = tr.neg(gpl)
    one = tr.inp(1.0)
    if tsml:
        chib = tr.mul(tr.mul(t, X), tb)
        y = tr.div(ds, ps)
        ab1 = tr.div(x, ps) if float(qs[0]) > float(ps[0]) else tr.div(y, qs)
        gps = tr.sub(ab1, chib)
        rA = 'S'
        if nan(gps):
            gps = tr.div(tr.sub(y, tr.mul(tr.mul(t, t), X)), qs)
            rA = 'Sf'
    else:
        Et = tr.fn('exp', tr.neg(t))
        Etp = tr.mul(Et, tr.add(one, t)) if tf else (T(0), 0.0)
        gps = tr.sub(tr.div(Etp, qs), tr.div(one, ps))
        rA = 'L'
        if not fin(gps):
            gps = tr.div(tr.sub(tr.mul(Etp, r), one), ps)
            rA = 'Lf'
    T2b = tr.mul(tr.mul(w, P), tb) if tsml else tr.div(T2, qs)
    gqs = tr.neg(tr.add(tr.div(x, qs), T2b))
    rQ = 'N'
    if nan(gqs):
        gqs = tr.neg(tr.div(tr.add(x, T2), qs))
        rQ = 'Nf'
    if not fx:
        gqs = (T(-np.inf), 0.0)
        rQ = 'Ninf'
    out = {'KL': KL, 'P.loc': gpl, 'P.scale': gps, 'Q.loc': gql, 'Q.scale': gqs}
    return (out, {'_branch': f"{br}/{('s' if tsml else 'l')}{('u' + sb if tsub else '')}/{rA}/{rQ}"})

def replay(pl, ps, ql, qs, dt):
    out, flags = _trace(pl, ps, ql, qs, dt, 1)
    res = {k: float(v[0]) for k, v in out.items()}
    res.update(flags)
    res['_in'] = (float(pl), float(ps), float(ql), float(qs))
    return res

def running_bounds(v, dt, lam):
    out, _ = _trace(*v['_in'], dt, lam)
    return {k: SAFETY * float(e) for k, (_, e) in out.items()}
