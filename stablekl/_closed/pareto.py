"""Repaired KL divergence between two Pareto distributions: formula, analytic gradients, and the
entry point run_rewrite_torch used by the benchmark (dtype in, compute dtype, outputs per component).
The derivation is in the paper's appendix on interface A."""
from __future__ import annotations
import math
from fractions import Fraction as Fr
import numpy as np
NAME = 'pareto'
PARAMS = ('scale', 'alpha')
COMPS = ['KL', 'P.scale', 'P.alpha', 'Q.scale', 'Q.alpha']
EXPECTED_ROWS = {'fp16': 80000, 'fp32': 126950}
TRANSCENDENTAL_COMPS = ('KL', 'Q.alpha')
TRANSCENDENTAL_FUNCS = ('log1p', 'log')
GRAD_TRUTH = 'frozen_interval'
FMT = {'fp16': dict(np=np.float16, u=2.0 ** (-11), eta=2.0 ** (-24), minnorm=2.0 ** (-14), emax=15), 'fp32': dict(np=np.float32, u=2.0 ** (-24), eta=2.0 ** (-149), minnorm=2.0 ** (-126), emax=127), 'fp64': dict(np=np.float64, u=2.0 ** (-53), eta=2.0 ** (-1074), minnorm=2.0 ** (-1022), emax=1023)}
SAFETY = 1.02
KTERMS = {'fp16': 9, 'fp32': 17, 'fp64': 36}
SEG_LO, SEG_HI = (-0.75, 3.0)

def q_tail_bound(z, K):
    return z ** K / ((2 * K + 3) * (1 - z))
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    _DT = {torch.float16: 'fp16', torch.float32: 'fp32', torch.float64: 'fp64'}

    class ParetoKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, sp, ap, sq, aq):
            dt = _DT[sp.dtype]
            MINN = FMT[dt]['minnorm']
            one = torch.ones_like(sp)
            zero = torch.zeros_like(sp)
            half = torch.full_like(sp, 0.5)
            two = torch.full_like(sp, 2)
            inf = torch.full_like(sp, float('inf'))
            x = (sp - sq) / sq
            fx = torch.isfinite(x)
            L = torch.where(fx, torch.log1p(torch.where(fx, x, zero)), torch.log(sp) - torch.log(sq))
            da = aq - ap
            y = da / ap
            fy = torch.isfinite(y)
            yf = torch.where(fy, y, zero)
            inS = fy & (y >= SEG_LO) & (y <= SEG_HI)
            inA = fy & (y > SEG_HI)
            das = torch.where(inS, da, zero)
            ys = torch.where(inS, y, zero)
            ss = aq + ap
            fss = torch.isfinite(ss)
            s = torch.where(fss, das / torch.where(fss, ss, one), das * half / (aq * half + ap * half))
            z = s * s
            K = KTERMS[dt]
            q = torch.full_like(sp, 1.0 / (2 * K + 1))
            for k in range(K - 1, 0, -1):
                q = q * z + torch.full_like(sp, 1.0 / (2 * k + 1))
            phiS = ys * s - two * (s * z) * q
            LA = torch.log1p(torch.where(inA, y, zero))
            ra = aq / ap
            rn = ra >= MINN
            Lab = torch.log(aq) - torch.log(ap)
            LB = torch.where(rn, torch.log(torch.where(rn, ra, one)), Lab)
            phi = torch.clamp_min(torch.where(inS, phiS, yf - torch.where(inA, LA, LB)), 0)
            kl = torch.where(fy & (sp >= sq), torch.clamp_min(aq * L + phi, 0), inf)
            g_sp = aq / sp
            g_sq = zero - aq / sq
            g_ap = (ap - aq) / ap / ap
            term = torch.where(ap > aq, da / ap / aq, da / aq / ap)
            g_aq = torch.where(torch.isfinite(term), L + term, (aq * L + da / ap) / aq)
            ok = sp >= sq
            g_sp, g_ap = (torch.where(ok, g_sp, zero), torch.where(ok, g_ap, zero))
            g_sq, g_aq = (torch.where(ok, g_sq, zero), torch.where(ok, g_aq, zero))
            ctx.save_for_backward(g_sp, g_ap, g_sq, g_aq)
            return kl

        @staticmethod
        def backward(ctx, g):
            a, b, c, d = ctx.saved_tensors
            return (g * a, g * b, g * c, g * d)

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    leaves = [to_t(x, dt_in).to(TD[cdt]).requires_grad_(True) for x in (P['scale'], P['alpha'], Q['scale'], Q['alpha'])]
    kl = ParetoKLC.apply(*leaves)
    kl.sum().backward()
    vals = {'KL': kl.detach()}
    for name, leaf in zip(('P.scale', 'P.alpha', 'Q.scale', 'Q.alpha'), leaves):
        vals[name] = leaf.grad.detach()
    return {k: v.cpu().numpy().astype(NPF[cdt]) for k, v in vals.items()}

def exact_grads(sp, ap, sq, aq):
    return None

def identity(sp, ap, sq, aq):
    return sp == sq and ap == aq

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

def _trace(sp, ap, sq, aq, dt, lam):
    tr = _Tr(dt, lam)
    T = tr.T
    MINN = FMT[dt]['minnorm']
    sp, ap, sq, aq = (tr.inp(sp), tr.inp(ap), tr.inp(sq), tr.inp(aq))
    fin = lambda a: bool(np.isfinite(a[0]))
    x = tr.div(tr.sub(sp, sq), sq)
    if fin(x):
        L = tr.fn('log1p', x)
        bl = '1p'
    else:
        L = tr.sub(tr.fn('log', sp), tr.fn('log', sq))
        bl = 'lg'
    da = tr.sub(aq, ap)
    y = tr.div(da, ap)
    fy = fin(y)
    yv = float(y[0])
    inS = fy and SEG_LO <= yv <= SEG_HI
    inA = fy and yv > SEG_HI
    br = 'S' if inS else 'A' if inA else 'B' if fy else 'INF'
    if inS:
        K = KTERMS[dt]
        ss = tr.add(aq, ap)
        if fin(ss):
            s = tr.div(da, ss)
        else:
            s = tr.div(tr.scale2(da, 0.5), tr.add(tr.scale2(aq, 0.5), tr.scale2(ap, 0.5)))
        z = tr.mul(s, s)
        q = tr.const(Fr(1, 2 * K + 1))
        for k in range(K - 1, 0, -1):
            q = tr.add(tr.mul(q, z), tr.const(Fr(1, 2 * k + 1)))
        q = tr.extra(q, q_tail_bound(float(z[0]), K))
        phit = tr.sub(tr.mul(y, s), tr.mul(tr.scale2(tr.mul(s, z), 2.0), q))
    elif fy:
        ra = tr.div(aq, ap)
        if inA:
            LL = tr.fn('log1p', y)
        elif float(ra[0]) >= MINN:
            LL = tr.fn('log', ra)
        else:
            LL = tr.sub(tr.fn('log', aq), tr.fn('log', ap))
        phit = tr.sub(y, LL)
    if fy and float(sp[0]) >= float(sq[0]):
        KL = tr.clamp0(tr.add(tr.mul(aq, L), tr.clamp0(phit)))
    else:
        KL = (T(np.inf), 0.0)
    gsp = tr.div(aq, sp)
    gsq = tr.neg(tr.div(aq, sq))
    gap = tr.div(tr.div(tr.sub(ap, aq), ap), ap)
    term = tr.div(tr.div(da, ap), aq) if float(ap[0]) > float(aq[0]) else tr.div(tr.div(da, aq), ap)
    if fin(term):
        gaq = tr.add(L, term)
        bt = ''
    else:
        gaq = tr.div(tr.add(tr.mul(aq, L), tr.div(da, ap)), aq)
        bt = '/merged'
    out = {'KL': KL, 'P.scale': gsp, 'P.alpha': gap, 'Q.scale': gsq, 'Q.alpha': gaq}
    if float(sp[0]) < float(sq[0]):
        for c in ('P.scale', 'P.alpha', 'Q.scale', 'Q.alpha'):
            out[c] = (T(0), 0.0)
    return (out, {'_branch': f"{br}/{bl}/{('ap>aq' if float(ap[0]) > float(aq[0]) else 'ap<=aq')}{bt}"})

def replay(sp, ap, sq, aq, dt):
    out, flags = _trace(sp, ap, sq, aq, dt, 1)
    res = {k: float(v[0]) for k, v in out.items()}
    res.update(flags)
    res['_in'] = (float(sp), float(ap), float(sq), float(aq))
    return res

def running_bounds(v, dt, lam):
    out, _ = _trace(*v['_in'], dt, lam)
    return {k: SAFETY * float(e) for k, (_, e) in out.items()}
