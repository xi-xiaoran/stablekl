"""Repaired KL divergence between two Exponential distributions: formula, analytic gradients, and the
entry point run_rewrite_torch used by the benchmark (dtype in, compute dtype, outputs per component).
The derivation is in the paper's appendix on interface A."""
from __future__ import annotations
import math
import numpy as np
NAME = 'exponential'
PARAMS = ('rate',)
COMPS = ['KL', 'P.rate', 'Q.rate']
EXPECTED_ROWS = {'fp16': 106000, 'fp32': 202000}
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

    class ExponentialKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, a, b):
            one = torch.ones_like(a)
            zero = torch.zeros_like(a)
            half = torch.full_like(a, 0.5)
            d = b - a
            x = d / a
            fin = torch.isfinite(x)
            inS = fin & (x >= SEG_LO) & (x <= SEG_HI)
            inA = fin & (x > SEG_HI)
            xf = torch.where(fin, x, zero)
            xs = torch.where(inS, x, zero)
            tx = torch.full_like(a, 2) + xs
            s = xs / tx
            z = s * s
            q = torch.full_like(a, 1.0 / (2 * _K[a.dtype] + 1))
            for k in range(_K[a.dtype] - 1, 0, -1):
                q = q * z + torch.full_like(a, 1.0 / (2 * k + 1))
            t1 = xs * xs / tx
            t2 = 2 * (s * z) * q
            KLs = torch.clamp_min(t1 - t2, 0)
            LA = torch.log1p(torch.where(inA, x, zero))
            r = b / a
            rn = r >= _MINN[a.dtype]
            Lab = torch.log(b) - torch.log(a)
            LB = torch.where(rn, torch.log(torch.where(rn, r, one)), Lab)
            L = torch.where(inA, LA, LB)
            KLab = torch.where(inS, KLs, torch.clamp_min(xf - L, 0))
            xh = d * half / a
            KLc = 2 * (xh - Lab * half)
            kl = torch.where(fin, KLab, KLc)
            gp = (zero - x) / a
            gq = torch.where(fin, xf / b, one / a - one / b)
            ctx.save_for_backward(gp, gq)
            return kl

        @staticmethod
        def backward(ctx, g):
            gp, gq = ctx.saved_tensors
            return (g * gp, g * gq)

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    a = to_t(P['rate'], dt_in).to(TD[cdt]).requires_grad_(True)
    b = to_t(Q['rate'], dt_in).to(TD[cdt]).requires_grad_(True)
    kl = ExponentialKLC.apply(a, b)
    kl.sum().backward()
    vals = {'KL': kl.detach(), 'P.rate': a.grad.detach(), 'Q.rate': b.grad.detach()}
    return {k: v.cpu().numpy().astype(NPF[cdt]) for k, v in vals.items()}

def exact_grads(a, b):
    return {'P.rate': (a - b) / (a * a), 'Q.rate': 1 / a - 1 / b}

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
        inS = fin and SEG_LO <= float(v['x']) <= SEG_HI
        inA = fin and float(v['x']) > SEG_HI
        v['r'] = T(b / a)
        rn = float(v['r']) >= f['minnorm']
        v['la'], v['lb'] = (T(np.log(a)), T(np.log(b)))
        v['Lab'] = T(v['lb'] - v['la'])
        if inS:
            br = 'S'
            K = KTERMS[dt]
            x = v['x']
            v['tx'] = T(T(2) + x)
            v['s'] = T(x / v['tx'])
            v['z'] = T(v['s'] * v['s'])
            qs = [T(1.0 / (2 * K + 1))]
            for k in range(K - 1, 0, -1):
                qs.append(T(T(qs[-1] * v['z']) + T(1.0 / (2 * k + 1))))
            v['q'] = qs[-1]
            v['_qs'] = qs
            v['xx'] = T(x * x)
            v['t1'] = T(v['xx'] / v['tx'])
            v['sz'] = T(v['s'] * v['z'])
            v['t2'] = T(T(T(2) * v['sz']) * v['q'])
            v['t'] = T(v['t1'] - v['t2'])
        elif inA:
            br = 'A'
            v['L'] = T(np.log1p(v['x']))
        elif fin:
            br = 'B'
            v['L'] = T(np.log(v['r'])) if rn else v['Lab']
        else:
            br = 'C'
        if fin:
            if not inS:
                v['t'] = T(v['x'] - v['L'])
            v['KL'] = max(v['t'], T(0))
            clamped = float(v['t']) < 0
        else:
            v['xh'] = T(T(v['d'] * T(0.5)) / a)
            v['Lh'] = T(v['Lab'] * T(0.5))
            v['KLh'] = T(v['xh'] - v['Lh'])
            v['KL'] = T(T(2) * v['KLh'])
            clamped = False
        v['gp'] = T(T(T(0) - v['x']) / a)
        if fin:
            v['gq'] = T(v['x'] / b)
        else:
            v['ia'], v['ib'] = (T(T(1) / a), T(T(1) / b))
            v['gq'] = T(v['ia'] - v['ib'])
    qs = v.pop('_qs', None)
    out = {k: float(x) for k, x in v.items()}
    out['_qs'] = [float(t) for t in qs] if qs is not None else None
    out.update({'_branch': br, '_rn': rn, '_clamped': clamped, '_in': (float(a), float(b)), 'P.rate': out['gp'], 'Q.rate': out['gq']})
    return out

def running_bounds(v, dt, lam):
    f = FMT[dt]
    u, h = (f['u'], f['eta'] / 2)
    mn = f['minnorm']
    sub = lambda x: 1.0 if 0 < abs(x) < mn else 0.0
    OTf = float(np.finfo(f['np']).max) * (1 + u)
    A = lambda x: min(abs(x), OTf)
    a, b = v['_in']
    e_d = u * A(v['d'])
    e_x = e_d / a + u * A(v['x']) + h * sub(v['x'])
    e_la = lam * u * abs(math.log(a))
    e_lb = lam * u * abs(math.log(b))
    e_Lab = e_la + e_lb + u * A(v['Lab'])
    br = v['_branch']
    if br == 'S':
        K = KTERMS[dt]
        x = v['x']
        tx, sv, z = (v['tx'], v['s'], v['z'])
        e_tx = e_x + u * A(tx)
        e_s = e_x / abs(tx) + abs(sv) * e_tx / abs(tx) + u * A(sv) + h * sub(sv)
        e_z = 2 * abs(sv) * e_s + u * A(z) + h * sub(z)
        qs = v['_qs']
        e_q = u * qs[0]
        for i, k in enumerate(range(K - 1, 0, -1)):
            qprev = qs[i]
            prod = qprev * z
            e_q = e_q * z + qprev * e_z + u * A(prod) + h * sub(prod) + u / (2 * k + 1) + u * A(qs[i + 1])
        e_q += q_tail_bound(z, K)
        e_xx = 2 * abs(x) * e_x + u * A(v['xx']) + h * sub(v['xx'])
        e_t1 = e_xx / tx + v['t1'] * e_tx / tx + u * A(v['t1']) + h * sub(v['t1'])
        e_sz = e_s * z + abs(sv) * e_z + u * A(v['sz']) + h * sub(v['sz'])
        e_t2 = 2 * e_sz * v['q'] + 2 * abs(v['sz']) * e_q + u * A(v['t2']) + h * sub(v['t2'])
        e_KL = e_t1 + e_t2 + u * A(v['t']) + h * sub(v['t'])
    elif br == 'A':
        x = v['x']
        e_L = e_x / (1 + x) + lam * u * A(v['L']) + h * sub(v['L'])
    elif br == 'B':
        if v['_rn']:
            e_r = u * A(v['r'])
            e_L = e_r / v['r'] + lam * u * A(v['L'])
        else:
            e_L = e_Lab
    if br in ('A', 'B'):
        e_KL = e_x + e_L + u * A(v['t']) + h * sub(v['t'])
    elif br == 'C':
        e_xh = 0.5 * e_d / a + u * A(v['xh']) + h * sub(v['xh'])
        e_KL = 2 * (e_xh + 0.5 * e_Lab + u * A(v['KLh']) + h * sub(v['KLh']))
    if br in ('S', 'A', 'B'):
        e_gp = e_x / a + u * A(v['gp']) + h * sub(v['gp'])
        e_gq = e_x / b + u * A(v['gq']) + h * sub(v['gq'])
    else:
        e_gp = u * A(v['gp'])
        e_ia = u * A(v['ia']) + h * sub(v['ia'])
        e_ib = u * A(v['ib']) + h * sub(v['ib'])
        e_gq = e_ia + e_ib + u * A(v['gq']) + h * sub(v['gq'])
    return {k: SAFETY * x for k, x in {'KL': e_KL, 'P.rate': e_gp, 'Q.rate': e_gq}.items()}
