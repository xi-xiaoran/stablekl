"""Repaired KL divergence between two Geometric (logits) distributions: formula, analytic gradients, and the
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
OFFICIAL_FN = '_kl_geometric_geometric'
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

    def _psig(s, K):
        z = s * s
        q = torch.full_like(s, 1.0 / (2 * K + 1))
        for k in range(K - 1, 0, -1):
            q = q * z + torch.full_like(s, 1.0 / (2 * k + 1))
        a = z * q
        oms = torch.ones_like(s) - s
        return (s - a * oms, oms * (torch.ones_like(s) + a))

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

    def _mulexp(F, x, T1):
        ax = x.abs()
        m1 = F * torch.exp(x)
        h2 = torch.exp(x * 0.5)
        m2 = F * h2 * h2
        h4 = torch.exp(x * 0.25)
        m4 = F * h4 * h4 * h4 * h4
        return torch.where(ax <= T1, m1, torch.where(ax <= 2 * T1, m2, m4))

    def _tie_round(A, C, tgt):
        prec, emin = TIE_FMT[tgt]
        s = A + C
        e = A - s + C
        one = torch.ones_like(s)
        _, ex = torch.frexp(s)
        ulp = torch.ldexp(one, torch.clamp_min(ex - 1, emin) - (prec - 1))
        r = s.to(_TD[tgt]).to(s.dtype)
        mid = ((s - r).abs() * 2 == ulp) & (e != 0)
        return torch.where(mid, torch.nextafter(s, s + torch.sign(e) * s.abs()), s)

    def _tie_aware(z, w, g_z, g_w, tgt):
        zero = torch.zeros_like(z)
        one = torch.ones_like(z)
        tiny = (z.abs() <= TIE_TAU) & (w.abs() <= TIE_TAU) & (z != w)
        zz = torch.where(tiny, z, zero)
        ww = torch.where(tiny, w, zero)
        d = ww - zz
        pe = torch.full_like(z, TIE_PE[-1])
        for c in TIE_PE[-2::-1]:
            pe = pe * d + c
        E2 = d * d * pe
        E1 = d * 0.5 + E2
        y = ww * 0.5
        y2 = y * y
        pt = torch.full_like(z, TIE_PT[-1])
        for c in TIE_PT[-2::-1]:
            pt = pt * y2 + c
        T3 = y2 * y * pt
        Tt = y + T3
        Bw = zero - zz * 0.5 + E2 - T3 - Tt * E1
        u = zz * 0.5
        H = [one]
        uk = one
        for m in range(1, 12):
            uk = uk * u
            H.append(uk + y * H[m - 1])
        R = H[11] * TIE_C[5]
        for k in (4, 3, 2, 1, 0):
            R = R + H[2 * k + 1] * TIE_C[k]
        mz = zero - zz
        pm = torch.full_like(z, TIE_EM[-1])
        for c in TIE_EM[-2::-1]:
            pm = pm * mz + c
        em = mz * pm
        Bz = em + R + em * R
        AQ = (ww - zz) * 0.5
        AP = (zz - ww) * 0.5
        tz = _tie_round(AP, AP * Bz, tgt)
        tw = _tie_round(AQ, AQ * Bw, tgt)
        return (torch.where(tiny, tz, g_z), torch.where(tiny, tw, g_w))
    _TD = {'fp16': torch.float16, 'fp32': torch.float32}

    class GeometricLogitsKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, z, w, tgt=None):
            dt = _DT[z.dtype]
            K = KTERMS[dt]
            T1 = TSUB[dt]
            one = torch.ones_like(z)
            zero = torch.zeros_like(z)
            half = torch.full_like(z, 0.5)
            two = torch.full_like(z, 2.0)
            z_in, w_in = (z, w)
            zi, wi = (torch.isinf(z), torch.isinf(w))
            z = torch.where(zi | wi, zero, z)
            w = torch.where(zi | wi, zero, w)
            dl = z - w
            mdl = zero - dl
            nb = mdl <= 2
            b2 = ~nb & (w >= 0)
            mw = zero - w
            bb = dl - z
            dlo = z - (dl - bb) + (mw - bb)
            dlo = torch.where(dl.abs() <= 2 * T1, dlo, zero)
            F1b = (one - torch.exp(dl)) / (one + torch.exp(mw))
            F1c = (one - torch.exp(dl)) * _sig(mw)
            F1c = F1c - F1c * dlo
            F1 = torch.where(nb, torch.expm1(mdl), torch.where(b2, F1b, F1c))

            def mX1(F):
                return torch.where(nb, _mulsig(F, mw, T1), torch.where(b2, _mulexp(F, zero - z, T1), _mulexp(F, mdl, T1)))
            x1 = mX1(F1)
            nc = dl <= 2
            c2 = ~nc & (w <= 0)
            F2b = (one - torch.exp(mdl)) / (one + torch.exp(w))
            F2c = (one - torch.exp(mdl)) * _sig(w)
            F2c = F2c + F2c * dlo
            x2 = torch.where(nc, _mulsig(torch.expm1(dl), w, T1), torch.where(c2, _mulexp(F2b, z, T1), _mulexp(F2c, dl, T1)))
            in1, in2 = (_inseg(x1), _inseg(x2))
            s1 = torch.where(in1, x1 / (two + x1), zero)
            s2 = torch.where(in2, x2 / (two + x2), zero)
            ps1, _ = _psig(s1, K)
            ps2, gs2 = _psig(s2, K)
            ok1 = torch.isfinite(x1) & (x1 >= SEG_LO)
            ok2 = torch.isfinite(x2) & (x2 >= SEG_LO)
            L1 = torch.where(ok1, torch.log1p(torch.where(ok1, x1, zero)), _LD(w, z))
            L2 = torch.where(ok2, torch.log1p(torch.where(ok2, x2, zero)), _LD(zero - w, zero - z))
            psi1 = torch.where(in1, ps1, one - L1 / x1)
            psi2 = torch.where(in2, ps2, one - L2 / x2)
            kl = mX1(F1 * (psi1 - psi2))
            g_w = x1
            g_z = torch.where(in2, mX1((zero - F1) * gs2), _mulexp(L2, zero - z, T1))
            if tgt is not None:
                g_z, g_w = _tie_aware(z, w, g_z, g_w, tgt)
            same = z_in == w_in
            kinf = wi & ~same
            wf = torch.where(wi, zero, w_in)
            zpos = zi & (z_in > 0)
            kl = torch.where(same, zero, torch.where(kinf, torch.full_like(z, math.inf), torch.where(zpos, _sp(zero - wf), kl)))
            g_z = torch.where(same | zi | wi, zero, g_z)
            g_w = torch.where(same | kinf, zero, torch.where(zpos, zero - _sig(zero - wf), g_w))
            pz = zi & (z_in < 0)
            nan = torch.full_like(z, math.nan)
            kl = torch.where(pz, nan, kl)
            g_z = torch.where(pz, nan, g_z)
            g_w = torch.where(pz, nan, g_w)
            ctx.save_for_backward(g_z, g_w)
            return kl

        @staticmethod
        def backward(ctx, g, *rest):
            a, b = ctx.saved_tensors
            return (g * a, g * b, None)
TIE_TAU = 1.0 / 16
REPLAY_TARGET = True
_FACT = [math.factorial(k) for k in range(15)]
TIE_PE_FR = tuple((Fr(1, _FACT[k + 1]) for k in range(2, 13)))
TIE_PT_FR = (Fr(-1, 3), Fr(2, 15), Fr(-17, 315), Fr(62, 2835), Fr(-1382, 155925))
TIE_C_FR = (Fr(1, 2), Fr(-1, 12), Fr(1, 45), Fr(-17, 2520), Fr(31, 14175), Fr(-691, 935550))
TIE_EM_FR = tuple((Fr(1, _FACT[k]) for k in range(1, 13)))
TIE_PE, TIE_PT, TIE_C, TIE_EM = (tuple((float(c) for c in t)) for t in (TIE_PE_FR, TIE_PT_FR, TIE_C_FR, TIE_EM_FR))
TIE_TRUNC = 1e-19
TIE_FMT = {'fp16': (11, -14), 'fp32': (24, -126)}
NAME = 'geometric_logits'
PARAMS = ('logits',)
COMPS = ['KL', 'P.logits', 'Q.logits']
EXPECTED_ROWS = {'fp16': 150178, 'fp32': 202000}
TRANSCENDENTAL_COMPS = ('KL', 'P.logits', 'Q.logits')

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    leaves = [to_t(x, dt_in).to(TD[cdt]).requires_grad_(True) for x in (P['logits'], Q['logits'])]
    kl = GeometricLogitsKLC.apply(*leaves, dt_in if cdt != dt_in else None)
    kl.sum().backward()
    vals = {'KL': kl.detach(), 'P.logits': leaves[0].grad.detach(), 'Q.logits': leaves[1].grad.detach()}
    return {k: v.cpu().numpy().astype(NPF[cdt]) for k, v in vals.items()}

def exact_grads(z, w):
    return None

def identity(z, w):
    return z == w

def calibrate_exp_pos(seed=11):
    from decimal import Decimal, getcontext
    getcontext().prec = 40
    rng = np.random.default_rng(seed)
    P = {'fp16': 11, 'fp32': 24, 'fp64': 53}
    TT = {'fp16': torch.float16, 'fp32': torch.float32, 'fp64': torch.float64}
    for dt in ('fp16', 'fp32', 'fp64'):
        T = FMT[dt]['np']
        T1 = TSUB[dt]
        if dt == 'fp16':
            xs = np.arange(1, 31744, dtype=np.uint16).view(np.float16)
            xs = xs[xs.astype(float) <= T1]
        else:
            lo = -30 if dt == 'fp32' else -60
            xs = np.unique(np.concatenate([np.exp2(rng.uniform(lo, math.log2(T1), 20000)), rng.uniform(0, T1, 10000)]).astype(T))
            xs = xs[(xs > 0) & (xs.astype(float) <= T1)]
        UI = {'fp16': np.uint16, 'fp32': np.uint32, 'fp64': np.uint64}[dt]
        for fn, top in (('exp', T1), ('expm1', 2.0)):
            xa = xs[xs.astype(float) <= top]
            y = getattr(torch, fn)(torch.from_numpy(xa).to(TT[dt])).numpy().astype(T)
            dmax, wx = (0, None)
            for x, yy in zip(xa.tolist(), y):
                ex = Decimal(x).exp() - (1 if fn == 'expm1' else 0)
                c = T(float(ex))
                cands = [np.nextafter(c, T(-np.inf)), c, np.nextafter(c, T(np.inf))]
                cr = min(cands, key=lambda t: (abs(Decimal(float(t)) - ex), int(np.array(t).view(UI)) & 1))
                d = abs(int(np.array(yy).view(UI)) - int(np.array(cr).view(UI)))
                if d > dmax:
                    dmax, wx = (d, x)
            print(f'calibrate_exp_pos {dt} {fn} n={len(xa)} max ULP vs correctly rounded d={dmax} (lambda would be 2d+1={2 * dmax + 1}) at x={wx!r}', flush=True)

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

    def subx(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] - b[0])
        return (z, a[1] + b[1])

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
        if bb == math.inf:
            return (z, a[1] / bb if a[1] < math.inf else math.inf)
        return (z, (a[1] + self.A(z) * b[1]) / bb + self.R(z) if bb > 0 else float('inf'))

    def fn(self, name, a):
        with np.errstate(all='ignore'):
            z = self.T(getattr(np, name)(np.array([a[0]], dtype=self.T))[0])
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

def _tr_psig(tr, s, dt):
    K = KTERMS[dt]
    z = tr.mul(s, s)
    q = tr.const(Fr(1, 2 * K + 1))
    for k in range(K - 1, 0, -1):
        q = tr.add(tr.mul(q, z), tr.const(Fr(1, 2 * k + 1)))
    q = tr.extra(q, q_tail_bound(float(z[0]), K))
    a = tr.mul(z, q)
    one = tr.inp(1.0)
    oms = tr.sub(one, s)
    return (tr.sub(s, tr.mul(a, oms)), tr.mul(oms, tr.add(one, a)))

def _seg(a):
    v = float(a[0])
    return math.isfinite(v) and SEG_LO <= v <= SEG_HI

class _Ops:

    def __init__(self, tr, dt):
        self.tr = tr
        self.T1 = TSUB[dt]
        self.one = tr.inp(1.0)

    def sig(self, x):
        tr, one = (self.tr, self.one)
        e = tr.fn('exp', tr.neg(tr.absv(x)))
        return tr.div(one, tr.add(one, e)) if float(x[0]) >= 0 else tr.div(e, tr.add(one, e))

    def sp(self, x):
        tr = self.tr
        return tr.add(tr.clamp0(x), tr.fn('log1p', tr.fn('exp', tr.neg(tr.absv(x)))))

    def LD(self, a, b):
        tr = self.tr
        if float(a[0]) < 0 and float(b[0]) < 0:
            return tr.add(tr.sub(a, b), tr.sub(self.sp(b), self.sp(a)))
        return tr.sub(self.sp(tr.neg(b)), self.sp(tr.neg(a)))

    def mulsig(self, F, x):
        tr, one, T1 = (self.tr, self.one, self.T1)
        xv = float(x[0])
        if xv >= -T1:
            return tr.mul(self.sig(x), F)
        e = tr.fn('exp', x)
        ope = tr.add(one, e)
        if xv >= -2 * T1:
            h = tr.fn('exp', tr.scale2(x, 0.5))
            return tr.div(tr.mul(tr.mul(F, h), h), ope)
        h = tr.fn('exp', tr.scale2(x, 0.25))
        return tr.div(tr.mul(tr.mul(tr.mul(tr.mul(F, h), h), h), h), ope)

    def mulexp(self, F, x):
        tr, T1 = (self.tr, self.T1)
        ax = abs(float(x[0]))
        if ax <= T1:
            return tr.mul(F, tr.fn('exp', x))
        if ax <= 2 * T1:
            h = tr.fn('exp', tr.scale2(x, 0.5))
            return tr.mul(tr.mul(F, h), h)
        h = tr.fn('exp', tr.scale2(x, 0.25))
        return tr.mul(tr.mul(tr.mul(tr.mul(F, h), h), h), h)

def _tr_tie(tr, z, w, tgt):
    zz, ww = (z, w)
    one = tr.inp(1.0)
    zero = tr.inp(0.0)
    d = tr.sub(ww, zz)
    pe = tr.const(TIE_PE_FR[-1])
    for c in TIE_PE_FR[-2::-1]:
        pe = tr.add(tr.mul(pe, d), tr.const(c))
    E2 = tr.mul(tr.mul(d, d), pe)
    E1 = tr.add(tr.scale2(d, 0.5), E2)
    y = tr.scale2(ww, 0.5)
    y2 = tr.mul(y, y)
    pt = tr.const(TIE_PT_FR[-1])
    for c in TIE_PT_FR[-2::-1]:
        pt = tr.add(tr.mul(pt, y2), tr.const(c))
    T3 = tr.mul(tr.mul(y2, y), pt)
    Tt = tr.add(y, T3)
    Bw = tr.sub(tr.sub(tr.add(tr.sub(zero, tr.scale2(zz, 0.5)), E2), T3), tr.mul(Tt, E1))
    u = tr.scale2(zz, 0.5)
    H = [one]
    uk = one
    for m in range(1, 12):
        uk = tr.mul(uk, u)
        H.append(tr.add(uk, tr.mul(y, H[m - 1])))
    R = tr.mul(H[11], tr.const(TIE_C_FR[5]))
    for k in (4, 3, 2, 1, 0):
        R = tr.add(R, tr.mul(H[2 * k + 1], tr.const(TIE_C_FR[k])))
    mz = tr.sub(zero, zz)
    pm = tr.const(TIE_EM_FR[-1])
    for c in TIE_EM_FR[-2::-1]:
        pm = tr.add(tr.mul(pm, mz), tr.const(c))
    em = tr.mul(mz, pm)
    Bz = tr.add(tr.add(em, R), tr.mul(em, R))
    AQ = tr.scale2(tr.sub(ww, zz), 0.5)
    AP = tr.scale2(tr.sub(zz, ww), 0.5)
    prec, emin = TIE_FMT[tgt]
    T = tr.T
    tags = ''

    def rnd(A, C):
        nonlocal tags
        sv = tr.add(A, C)
        e = T(T(A[0] - sv[0]) + C[0])
        m, ex = np.frexp(float(sv[0]))
        ulp = 2.0 ** (max(ex - 1, emin) - (prec - 1))
        r = float(T(np.array(sv[0]).astype(np.float16 if tgt == 'fp16' else np.float32)))
        if abs(float(sv[0]) - r) * 2 == ulp and e != 0:
            nv = np.nextafter(sv[0], T(sv[0] + np.sign(e) * abs(sv[0])))
            tags += 'n'
            return (nv, sv[1] + abs(float(nv) - float(sv[0])) + TIE_TRUNC * abs(float(A[0])))
        tags += '.'
        return (sv[0], sv[1] + TIE_TRUNC * abs(float(A[0])))
    gz = rnd(AP, tr.mul(AP, Bz))
    gw = rnd(AQ, tr.mul(AQ, Bw))
    return (gz, gw, 'T' + tags)

def _trace(z, w, dt, lam, tgt=None):
    tr = _Tr(dt, lam)
    O = _Ops(tr, dt)
    z, w = (tr.inp(z), tr.inp(w))
    one = tr.inp(1.0)
    zero0 = tr.inp(0.0)
    two = tr.inp(2.0)
    zv, wv = (float(z[0]), float(w[0]))
    if math.isinf(zv) and zv < 0:
        nan = tr.inp(math.nan)
        return ({'KL': nan, 'P.logits': nan, 'Q.logits': nan}, {'_branch': 'E/p0'})
    if zv == wv:
        return ({'KL': zero0, 'P.logits': zero0, 'Q.logits': zero0}, {'_branch': 'id'})
    if math.isinf(wv):
        return ({'KL': tr.inp(math.inf), 'P.logits': zero0, 'Q.logits': zero0}, {'_branch': 'E/inf'})
    if math.isinf(zv):
        return ({'KL': O.sp(tr.neg(w)), 'P.logits': zero0, 'Q.logits': tr.neg(O.sig(tr.neg(w)))}, {'_branch': 'E/p1'})
    dl = tr.sub(z, w)
    dlv = float(dl[0])
    T = tr.T
    with np.errstate(all='ignore'):
        mwv = T(T(0) - w[0])
        bbv = T(dl[0] - z[0])
        dlov = T(T(z[0] - T(dl[0] - bbv)) + T(mwv - bbv))
    if not abs(float(dl[0])) <= 2 * O.T1:
        dlov = T(0)

    def corrected(Fc, sgn):
        d = tr.inp(dlov)
        prod = tr.mul(Fc, d)
        r = tr.add(Fc, prod) if sgn > 0 else tr.sub(Fc, prod)
        return tr.extra(r, abs(float(Fc[0])) * float(dlov) ** 2)
    if -dlv <= 2 or (math.isinf(dlv) and dlv > 0):
        mdl = tr.neg(dl)
        F1 = tr.fn('expm1', mdl)
        mX1 = lambda F: O.mulsig(F, tr.neg(w))
        b1 = '1'
    elif wv >= 0:
        F1 = tr.div(tr.sub(one, tr.fn('exp', dl)), tr.add(one, tr.fn('exp', tr.neg(w))))
        mX1 = lambda F: O.mulexp(F, tr.neg(z))
        b1 = '2'
    else:
        dlx = (dl[0], 0.0)
        mdl = tr.neg(dlx)
        E = tr.extra(tr.fn('exp', dlx), abs(float(np.exp(float(dlx[0])))) * abs(float(dlov)))
        F1 = corrected(tr.mul(tr.sub(one, E), O.sig(tr.neg(w))), -1)
        mX1 = lambda F: O.mulexp(F, mdl)
        b1 = '3'
    x1 = mX1(F1)
    dl2 = tr.sub(z, w)
    if dlv <= 2 or (math.isinf(dlv) and dlv < 0):
        x2 = O.mulsig(tr.fn('expm1', dl2), w)
        b2 = '1'
    elif wv <= 0:
        F2 = tr.div(tr.sub(one, tr.fn('exp', tr.neg(dl2))), tr.add(one, tr.fn('exp', w)))
        x2 = O.mulexp(F2, z)
        b2 = '2'
    else:
        dlx = (dl[0], 0.0)
        E = tr.extra(tr.fn('exp', tr.neg(dlx)), abs(float(np.exp(-float(dlx[0])))) * abs(float(dlov)))
        F2 = corrected(tr.mul(tr.sub(one, E), O.sig(w)), +1)
        x2 = O.mulexp(F2, dlx)
        b2 = '3'

    def L(x, a, b):
        v = float(x[0])
        if math.isfinite(v) and v >= SEG_LO:
            return tr.fn('log1p', x)
        return O.LD(a, b)
    if _seg(x1):
        psi1, _ = _tr_psig(tr, tr.div(x1, tr.add(two, x1)), dt)
        c1 = 'S'
    else:
        psi1 = tr.sub(one, tr.div(L(x1, w, z), x1))
        c1 = 'X'
    nw, nz = (tr.neg(w), tr.neg(z))
    if _seg(x2):
        psi2, g2 = _tr_psig(tr, tr.div(x2, tr.add(two, x2)), dt)
        c2 = 'S'
    else:
        L2 = L(x2, nw, nz)
        psi2 = tr.sub(one, tr.div(L2, x2))
        c2 = 'X'
    KL = mX1(tr.mul(F1, tr.sub(psi1, psi2)))
    gw = x1
    gz = mX1(tr.mul(tr.neg(F1), g2)) if c2 == 'S' else O.mulexp(L2, nz)
    br = f'x{b1}{b2}/{c1}{c2}'
    if tgt is not None and abs(zv) <= TIE_TAU and (abs(wv) <= TIE_TAU):
        gz, gw, tt = _tr_tie(tr, z, w, tgt)
        br += '/' + tt
    return ({'KL': KL, 'P.logits': gz, 'Q.logits': gw}, {'_branch': br})

def replay(z, w, dt, tgt=None):
    out, flags = _trace(z, w, dt, 1, tgt)
    res = {k: float(v[0]) for k, v in out.items()}
    res.update(flags)
    res['_in'] = (float(z), float(w))
    return res

def running_bounds(v, dt, lam, tgt=None):
    out, _ = _trace(*v['_in'], dt, lam, tgt)
    return {k: SAFETY * float(e) for k, (_, e) in out.items()}
