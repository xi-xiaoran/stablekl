"""Repaired KL divergence between two Bernoulli (probs) distributions: formula, analytic gradients, and the
entry point run_rewrite_torch used by the benchmark (dtype in, compute dtype, outputs per component).
The derivation is in the paper's appendix on interface A."""
from __future__ import annotations
import math
from fractions import Fraction as Fr
import numpy as np
FMT = {'fp16': dict(np=np.float16, u=2.0 ** (-11), eta=2.0 ** (-24), minnorm=2.0 ** (-14), emax=15), 'fp32': dict(np=np.float32, u=2.0 ** (-24), eta=2.0 ** (-149), minnorm=2.0 ** (-126), emax=127), 'fp64': dict(np=np.float64, u=2.0 ** (-53), eta=2.0 ** (-1074), minnorm=2.0 ** (-1022), emax=1023)}
SAFETY = 1.02
KTERMS = {'fp16': 9, 'fp32': 17, 'fp64': 36}
SEG_LO, SEG_HI = (-0.75, 3.0)
OFFICIAL_FN = '_kl_bernoulli_bernoulli'
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

    def _inseg(x):
        return torch.isfinite(x) & (x >= SEG_LO) & (x <= SEG_HI)

    class BernoulliProbsKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, p, q):
            dt = _DT[p.dtype]
            MINN = FMT[dt]['minnorm']
            K = KTERMS[dt]
            one = torch.ones_like(p)
            zero = torch.zeros_like(p)
            pe0, pe1, qe0, qe1 = (p == 0, p == 1, q == 0, q == 1)
            edge = pe0 | pe1 | qe0 | qe1
            p_in, q_in = (p, q)
            half = torch.full_like(p, 0.5)
            p = torch.where(edge, half, p)
            q = torch.where(edge, half, q)
            D = p - q
            omp = one - p
            omq = one - q
            x1 = (zero - D) / p
            x2 = D / omp
            in1, in2 = (_inseg(x1), _inseg(x2))
            s1 = torch.where(in1, (zero - D) / (p + q), zero)
            s2 = torch.where(in2, D / (omp + omq), zero)
            phi1 = _phiS(torch.where(in1, x1, zero), s1, K)
            phi2 = _phiS(torch.where(in2, x2, zero), s2, K)

            def Lf(x, a, b):
                ok = torch.isfinite(x) & (x >= SEG_LO)
                r = a / b
                rok = torch.isfinite(r) & (r >= MINN)
                return torch.where(ok, torch.log1p(torch.where(ok, x, zero)), torch.where(rok, torch.log(torch.where(rok, r, one)), torch.log(a) - torch.log(b)))
            L1 = Lf(x1, q, p)
            L2 = Lf(x2, omq, omp)
            t1 = torch.where(in1, p * phi1, zero - D - p * L1)
            t2 = torch.where(in2, omp * phi2, D - omp * L2)
            kl = torch.clamp_min(t1, 0) + torch.clamp_min(t2, 0)
            g_p = L2 - L1
            g_q = (zero - D) / q / omq
            inf = torch.full_like(p, math.inf)
            kinf = qe0 & ~pe0 | qe1 & ~pe1
            qf = torch.where(kinf, half, q_in)
            klE = torch.where(pe0, zero - torch.log1p(zero - qf), zero - torch.log(qf))
            gpE = torch.where(pe0, torch.where(qe0, inf, zero - inf), torch.where(qe1, zero - inf, inf))
            gqE = torch.where(pe0, one / (one - qf), (zero - one) / qf)
            kl = torch.where(edge, torch.where(kinf, inf, klE), kl)
            g_p = torch.where(edge, torch.where(kinf, zero, gpE), g_p)
            g_q = torch.where(edge, torch.where(kinf, zero, gqE), g_q)
            ctx.save_for_backward(g_p, g_q)
            return kl

        @staticmethod
        def backward(ctx, g):
            a, b = ctx.saved_tensors
            return (g * a, g * b)
NAME = 'bernoulli_probs'
PARAMS = ('probs',)
COMPS = ['KL', 'P.probs', 'Q.probs']
EXPECTED_ROWS = {'fp16': 125099, 'fp32': 202000}
TRANSCENDENTAL_COMPS = ('KL', 'P.probs')
TRANSCENDENTAL_FUNCS = ('log1p', 'log')

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    leaves = [to_t(x, dt_in).to(TD[cdt]).requires_grad_(True) for x in (P['probs'], Q['probs'])]
    kl = BernoulliProbsKLC.apply(*leaves)
    kl.sum().backward()
    vals = {'KL': kl.detach(), 'P.probs': leaves[0].grad.detach(), 'Q.probs': leaves[1].grad.detach()}
    return {k: v.cpu().numpy().astype(NPF[cdt]) for k, v in vals.items()}

def exact_grads(p, q):
    return None

def identity(p, q):
    return p == q

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

def _trace(p, q, dt, lam):
    tr = _Tr(dt, lam)
    T = tr.T
    MINN = FMT[dt]['minnorm']
    pv, qv = (float(p), float(q))
    p, q = (tr.inp(p), tr.inp(q))
    one = tr.inp(1.0)
    if pv in (0.0, 1.0) or qv in (0.0, 1.0):
        zero, inf = (tr.inp(0.0), tr.inp(math.inf))
        if qv == 0 and pv != 0 or (qv == 1 and pv != 1):
            return ({'KL': inf, 'P.probs': zero, 'Q.probs': zero}, {'_branch': 'E/inf'})
        if pv == 0:
            return ({'KL': tr.neg(tr.fn('log1p', tr.neg(q))), 'P.probs': tr.inp(math.inf if qv == 0 else -math.inf), 'Q.probs': tr.div(one, tr.sub(one, q))}, {'_branch': 'E/p0'})
        return ({'KL': tr.neg(tr.fn('log', q)), 'P.probs': tr.inp(-math.inf if qv == 1 else math.inf), 'Q.probs': tr.div(tr.neg(one), q)}, {'_branch': 'E/p1'})
    D = tr.sub(p, q)
    omp = tr.sub(one, p)
    omq = tr.sub(one, q)
    x1 = tr.div(tr.neg(D), p)
    x2 = tr.div(D, omp)

    def Lf(x, a, b):
        v = float(x[0])
        if math.isfinite(v) and v >= SEG_LO:
            return (tr.fn('log1p', x), '1p')
        r = tr.div(a, b)
        if math.isfinite(float(r[0])) and float(r[0]) >= MINN:
            return (tr.fn('log', r), 'lr')
        return (tr.sub(tr.fn('log', a), tr.fn('log', b)), 'ld')
    L1, b1 = Lf(x1, q, p)
    L2, b2 = Lf(x2, omq, omp)
    if _seg(x1):
        t1 = tr.mul(p, _tr_phiS(tr, x1, tr.div(tr.neg(D), tr.add(p, q)), dt))
        c1 = 'S'
    else:
        t1 = tr.sub(tr.neg(D), tr.mul(p, L1))
        c1 = 'X'
    if _seg(x2):
        t2 = tr.mul(omp, _tr_phiS(tr, x2, tr.div(D, tr.add(omp, omq)), dt))
        c2 = 'S'
    else:
        t2 = tr.sub(D, tr.mul(omp, L2))
        c2 = 'X'
    KL = tr.add(tr.clamp0(t1), tr.clamp0(t2))
    gp = tr.sub(L2, L1)
    gq = tr.div(tr.div(tr.neg(D), q), omq)
    return ({'KL': KL, 'P.probs': gp, 'Q.probs': gq}, {'_branch': f'{c1}{c2}/{b1}/{b2}'})

def replay(p, q, dt):
    out, flags = _trace(p, q, dt, 1)
    res = {k: float(v[0]) for k, v in out.items()}
    res.update(flags)
    res['_in'] = (float(p), float(q))
    return res

def running_bounds(v, dt, lam):
    out, _ = _trace(*v['_in'], dt, lam)
    return {k: SAFETY * float(e) for k, (_, e) in out.items()}
