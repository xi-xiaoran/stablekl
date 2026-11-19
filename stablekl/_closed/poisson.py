"""Repaired KL divergence between two Poisson distributions: formula, analytic gradients, and the
entry point run_rewrite_torch used by the benchmark (dtype in, compute dtype, outputs per component).
The derivation is in the paper's appendix on interface A."""
from __future__ import annotations
import math
from decimal import Decimal, localcontext
from fractions import Fraction
import numpy as np
NAME = 'poisson'
PARAMS = ('rate',)
COMPS = ['KL', 'P.rate', 'Q.rate']
EXPECTED_ROWS = {'fp16': 123498, 'fp32': 200000}
TRANSCENDENTAL_COMPS = ('KL', 'P.rate')
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
    _MINN = {torch.float16: 2.0 ** (-14), torch.float32: 2.0 ** (-126), torch.float64: 2.0 ** (-1022)}
    _K = {torch.float16: KTERMS['fp16'], torch.float32: KTERMS['fp32'], torch.float64: KTERMS['fp64']}

    class PoissonKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, a, b):
            one = torch.ones_like(a)
            zero = torch.zeros_like(a)
            d = b - a
            x = d / a
            fin = torch.isfinite(x)
            inS = fin & (x >= SEG_LO) & (x <= SEG_HI)
            inL1 = fin & (x >= SEG_LO)
            LA = torch.log1p(torch.where(inL1, x, zero))
            r = b / a
            rn = r >= _MINN[a.dtype]
            Lab = torch.log(b) - torch.log(a)
            LB = torch.where(rn, torch.log(torch.where(rn, r, one)), Lab)
            L = torch.where(inL1, LA, torch.where(fin, LB, Lab))
            ds = torch.where(inS, d, zero)
            half = torch.full_like(a, 0.5)
            Sab = a + b
            fS = torch.isfinite(Sab)
            s = torch.where(fS, ds / torch.where(fS, Sab, one), ds * half / (a * half + b * half))
            z = s * s
            q = torch.full_like(a, 1.0 / (2 * _K[a.dtype] + 1))
            for k in range(_K[a.dtype] - 1, 0, -1):
                q = q * z + torch.full_like(a, 1.0 / (2 * k + 1))
            t1 = ds * s
            t2 = 2 * (a * s * z) * q
            KLs = torch.clamp_min(t1 - t2, 0)
            inB = fin & (x < SEG_LO)
            xf = torch.where(fin, x, zero)
            KLB = a * torch.clamp_min(xf - L, 0)
            KLAC = torch.clamp_min(d - a * L, 0)
            kl = torch.where(inS, KLs, torch.where(inB, KLB, KLAC))
            ga = zero - L
            gb = d / b
            ctx.save_for_backward(ga, gb)
            return kl

        @staticmethod
        def backward(ctx, g):
            ga, gb = ctx.saved_tensors
            return (g * ga, g * gb)

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    a = to_t(P['rate'], dt_in).to(TD[cdt]).requires_grad_(True)
    b = to_t(Q['rate'], dt_in).to(TD[cdt]).requires_grad_(True)
    kl = PoissonKLC.apply(a, b)
    kl.sum().backward()
    vals = {'KL': kl.detach(), 'P.rate': a.grad.detach(), 'Q.rate': b.grad.detach()}
    return {k: v.cpu().numpy().astype(NPF[cdt]) for k, v in vals.items()}

def _ln_ratio(a: Fraction, b: Fraction) -> Fraction:
    if a == b:
        return Fraction(0)
    X = a / b
    with localcontext() as ctx:
        ctx.prec = 100 + len(str(X.numerator)) + len(str(X.denominator))
        return Fraction(Decimal(X.numerator).ln() - Decimal(X.denominator).ln())

def exact_grads(a, b):
    return {'P.rate': _ln_ratio(a, b), 'Q.rate': 1 - a / b}

def identity(a, b):
    return a == b

def replay(a, b, dt):
    f = FMT[dt]
    T = f['np']
    a, b = (T(a), T(b))
    v = {}
    with np.errstate(all='ignore'):
        v['d'] = T(b - a)
        v['x'] = T(v['d'] / a)
        fin = bool(np.isfinite(v['x']))
        xv = float(v['x'])
        inS = fin and SEG_LO <= xv <= SEG_HI
        inL1 = fin and xv >= SEG_LO
        v['r'] = T(b / a)
        rn = float(v['r']) >= f['minnorm']
        v['la'], v['lb'] = (T(np.log(a)), T(np.log(b)))
        v['Lab'] = T(v['lb'] - v['la'])
        if inL1:
            lsrc = 'log1p'
            v['L'] = T(np.log1p(v['x']))
        elif fin:
            lsrc = 'logr' if rn else 'lab'
            v['L'] = T(np.log(v['r'])) if rn else v['Lab']
        else:
            lsrc = 'lab'
            v['L'] = v['Lab']
        if inS:
            br = 'S'
            K = KTERMS[dt]
            x, d = (v['x'], v['d'])
            v['Sab'] = T(a + b)
            v['_fS'] = bool(np.isfinite(v['Sab']))
            if v['_fS']:
                v['s'] = T(d / v['Sab'])
            else:
                v['Sab'] = T(T(a * T(0.5)) + T(b * T(0.5)))
                v['s'] = T(T(d * T(0.5)) / v['Sab'])
            v['z'] = T(v['s'] * v['s'])
            qs = [T(1.0 / (2 * K + 1))]
            for k in range(K - 1, 0, -1):
                qs.append(T(T(qs[-1] * v['z']) + T(1.0 / (2 * k + 1))))
            v['q'] = qs[-1]
            v['t1'] = T(d * v['s'])
            v['as'] = T(a * v['s'])
            v['asz'] = T(v['as'] * v['z'])
            v['t2'] = T(T(T(2) * v['asz']) * v['q'])
            v['t'] = T(v['t1'] - v['t2'])
        elif fin and xv < SEG_LO:
            br = 'B'
            qs = None
            v['xL'] = T(v['x'] - v['L'])
            v['t'] = T(a * max(v['xL'], T(0)))
        else:
            br = 'A' if fin else 'C'
            qs = None
            v['aL'] = T(a * v['L'])
            v['t'] = T(v['d'] - v['aL'])
        v['KL'] = max(v['t'], T(0))
        clamped = float(v['t']) < 0 or (br == 'B' and float(v['xL']) < 0)
        v['ga'] = T(T(0) - v['L'])
        v['gb'] = T(v['d'] / b)
    fS = v.pop('_fS', None)
    out = {k: float(x) for k, x in v.items()}
    out.update({'_fS': fS, '_branch': br, '_lsrc': lsrc, '_rn': rn, '_clamped': clamped, '_in': (float(a), float(b)), '_qs': [float(t) for t in qs] if qs is not None else None, 'P.rate': out['ga'], 'Q.rate': out['gb']})
    return out

def running_bounds(v, dt, lam):
    f = FMT[dt]
    u, h = (f['u'], f['eta'] / 2)
    mn = f['minnorm']
    sub = lambda x: 1.0 if abs(x) < mn else 0.0
    OTf = float(np.finfo(f['np']).max) * (1 + u)
    A = lambda x: min(abs(x), OTf)
    a, b = v['_in']
    e_d = u * A(v['d'])
    e_x = e_d / a + u * A(v['x']) + h * sub(v['x'])
    e_Lab = lam * u * abs(math.log(a)) + lam * u * abs(math.log(b)) + u * A(v['Lab'])
    if v['_lsrc'] == 'log1p':
        e_L = e_x / (1 + v['x']) + lam * u * A(v['L']) + h * sub(v['L'])
    elif v['_lsrc'] == 'logr':
        e_L = u * A(v['r']) / v['r'] + lam * u * A(v['L'])
    else:
        e_L = e_Lab
    if v['_branch'] == 'S':
        K = KTERMS[dt]
        x, d = (v['x'], v['d'])
        Sab, sv, z = (v['Sab'], v['s'], v['z'])
        e_S = u * A(Sab)
        dd = d if v['_fS'] else d / 2
        e_dd = e_d if v['_fS'] else e_d / 2
        e_s = e_dd / Sab + abs(sv) * e_S / Sab + u * A(sv) + h * sub(sv)
        e_z = 2 * abs(sv) * e_s + u * A(z) + h * sub(z)
        qs = v['_qs']
        e_q = u * qs[0]
        for i, k in enumerate(range(K - 1, 0, -1)):
            qprev = qs[i]
            prod = qprev * z
            e_q = e_q * z + qprev * e_z + u * A(prod) + h * sub(prod) + u / (2 * k + 1) + u * A(qs[i + 1])
        e_q += q_tail_bound(z, K)
        e_t1 = e_d * abs(sv) + abs(d) * e_s + u * A(v['t1']) + h * sub(v['t1'])
        e_as = a * e_s + u * A(v['as']) + h * sub(v['as'])
        e_asz = e_as * z + abs(v['as']) * e_z + u * A(v['asz']) + h * sub(v['asz'])
        e_t2 = 2 * e_asz * v['q'] + 2 * abs(v['asz']) * e_q + u * A(v['t2']) + h * sub(v['t2'])
        e_KL = e_t1 + e_t2 + u * A(v['t']) + h * sub(v['t'])
    elif v['_branch'] == 'B':
        e_xL = e_x + e_L + u * A(v['xL']) + h * sub(v['xL'])
        e_KL = a * e_xL + u * A(v['t']) + h * sub(v['t'])
    else:
        e_aL = a * e_L + u * A(v['aL']) + h * sub(v['aL'])
        e_KL = e_d + e_aL + u * A(v['t']) + h * sub(v['t'])
    e_ga = e_L
    e_gb = e_d / b + u * A(v['gb']) + h * sub(v['gb'])
    return {k: SAFETY * x for k, x in {'KL': e_KL, 'P.rate': e_ga, 'Q.rate': e_gb}.items()}
