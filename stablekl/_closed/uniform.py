"""Repaired KL divergence between two Uniform distributions: formula, analytic gradients, and the
entry point run_rewrite_torch used by the benchmark (dtype in, compute dtype, outputs per component).
The derivation is in the paper's appendix on interface A."""
from __future__ import annotations
import math
import numpy as np
NAME = 'uniform'
PARAMS = ('low', 'high')
COMPS = ['KL', 'P.low', 'P.high', 'Q.low', 'Q.high']
EXPECTED_ROWS = {'fp16': 67292, 'fp32': 114506}
FMT = {'fp16': dict(np=np.float16, u=2.0 ** (-11), eta=2.0 ** (-24), minnorm=2.0 ** (-14), emax=15), 'fp32': dict(np=np.float32, u=2.0 ** (-24), eta=2.0 ** (-149), minnorm=2.0 ** (-126), emax=127), 'fp64': dict(np=np.float64, u=2.0 ** (-53), eta=2.0 ** (-1074), minnorm=2.0 ** (-1022), emax=1023)}
SAFETY = 1.02
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    _EMAX = {torch.float16: 15, torch.float32: 127, torch.float64: 1023}

    class UniformKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, pl, ph, ql, qh):
            dt = pl.dtype
            half = torch.full_like(pl, 0.5)
            mx = torch.maximum(torch.maximum(pl.abs(), ph.abs()), torch.maximum(ql.abs(), qh.abs()))
            H = torch.where(mx >= 2.0 ** (_EMAX[dt] - 1), half, torch.ones_like(pl))
            pl_, ph_, ql_, qh_ = (pl * H, ph * H, ql * H, qh * H)
            d1 = qh_ - ph_
            d2 = pl_ - ql_
            wps = ph_ - pl_
            s = d1 + d2
            r = s / wps
            ok = torch.isfinite(r)
            kl1 = torch.log1p(torch.where(ok, r, torch.zeros_like(r)))
            wp = ph - pl
            wq = qh - ql
            fp, fq = (torch.isfinite(wp), torch.isfinite(wq))
            wph = ph * half - pl * half
            wqh = qh * half - ql * half
            l2 = torch.log(torch.full_like(pl, 2))
            Lp = torch.where(fp, torch.log(torch.where(fp, wp, torch.ones_like(wp))), torch.log(wph) + l2)
            Lq = torch.where(fq, torch.log(torch.where(fq, wq, torch.ones_like(wq))), torch.log(wqh) + l2)
            kl2 = Lq - Lp
            out = (ql > pl) | (qh < ph)
            kl = torch.where(ok, kl1, kl2)
            kl = torch.where(out, torch.full_like(kl, float('inf')), kl)
            gp = torch.where(fp, 1 / torch.where(fp, wp, torch.ones_like(wp)), half / wph)
            gq = torch.where(fq, 1 / torch.where(fq, wq, torch.ones_like(wq)), half / wqh)
            zero = torch.zeros_like(gp)
            gp = torch.where(out, zero, gp)
            gq = torch.where(out, zero, gq)
            ctx.save_for_backward(gp, gq)
            return kl

        @staticmethod
        def backward(ctx, g):
            gp, gq = ctx.saved_tensors
            return (g * gp, g * -gp, g * -gq, g * gq)

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    leaves = [to_t(x, dt_in).to(TD[cdt]).requires_grad_(True) for x in (P['low'], P['high'], Q['low'], Q['high'])]
    kl = UniformKLC.apply(*leaves)
    kl.sum().backward()
    vals = {'KL': kl.detach()}
    for name, leaf in zip(('P.low', 'P.high', 'Q.low', 'Q.high'), leaves):
        vals[name] = leaf.grad.detach()
    return {k: v.cpu().numpy().astype(NPF[cdt]) for k, v in vals.items()}

def exact_grads(pl, ph, ql, qh):
    wp, wq = (ph - pl, qh - ql)
    return {'P.low': 1 / wp, 'P.high': -1 / wp, 'Q.low': -1 / wq, 'Q.high': 1 / wq}

def identity(pl, ph, ql, qh):
    return pl == ql and ph == qh

def replay(pl, ph, ql, qh, dt):
    f = FMT[dt]
    T = f['np']
    pl, ph, ql, qh = (T(pl), T(ph), T(ql), T(qh))
    v = {}
    with np.errstate(all='ignore'):
        hflag = 1 if max((abs(float(x)) for x in (pl, ph, ql, qh))) >= 2.0 ** (f['emax'] - 1) else 0
        H = T(0.5) if hflag else T(1)
        v['pl_'], v['ph_'], v['ql_'], v['qh_'] = (T(pl * H), T(ph * H), T(ql * H), T(qh * H))
        v['d1'] = T(v['qh_'] - v['ph_'])
        v['d2'] = T(v['pl_'] - v['ql_'])
        v['wps'] = T(v['ph_'] - v['pl_'])
        v['s'] = T(v['d1'] + v['d2'])
        v['r'] = T(v['s'] / v['wps'])
        v['wp'] = T(ph - pl)
        v['wq'] = T(qh - ql)
        v['fp'] = bool(np.isfinite(v['wp']))
        v['fq'] = bool(np.isfinite(v['wq']))
        half = T(0.5)
        v['wph'] = T(T(ph * half) - T(pl * half))
        v['wqh'] = T(T(qh * half) - T(ql * half))
        outside = bool(ql > pl or qh < ph)
        if outside:
            v['branch'] = 0
            v['KL'] = T(np.inf)
        elif np.isfinite(v['r']):
            v['branch'] = 1
            v['KL'] = T(np.log1p(v['r']))
        else:
            v['branch'] = 2
            v['l2'] = T(np.log(T(2)))
            v['Lp'] = T(np.log(v['wp'])) if v['fp'] else T(T(np.log(v['wph'])) + v['l2'])
            v['Lq'] = T(np.log(v['wq'])) if v['fq'] else T(T(np.log(v['wqh'])) + v['l2'])
            v['KL'] = T(v['Lq'] - v['Lp'])
        if outside:
            v['gp'] = T(0)
            v['gq'] = T(0)
        else:
            v['gp'] = T(T(1) / v['wp']) if v['fp'] else T(half / v['wph'])
            v['gq'] = T(T(1) / v['wq']) if v['fq'] else T(half / v['wqh'])
    out = {k: float(x) for k, x in v.items() if k not in ('branch', 'fp', 'fq')}
    out['_branch'], out['_h'], out['_fp'], out['_fq'] = (v['branch'], hflag, v['fp'], v['fq'])
    out['P.low'], out['P.high'], out['Q.low'], out['Q.high'] = (out['gp'], -out['gp'], -out['gq'], out['gq'])
    out['_in'] = (float(pl), float(ph), float(ql), float(qh))
    return out

def running_bounds(v, dt, lam):
    f = FMT[dt]
    u, h = (f['u'], f['eta'] / 2)
    mn = f['minnorm']
    sub = lambda x: 1.0 if abs(x) < mn else 0.0
    OTf = float(np.finfo(f['np']).max) * (1 + u)
    A = lambda x: min(abs(x), OTf)
    pl, ph, ql, qh = v['_in']
    if v['_branch'] == 0:
        return {k: 0.0 for k in COMPS}
    e_in = {k: h * sub(v[k]) if v['_h'] else 0.0 for k in ('pl_', 'ph_', 'ql_', 'qh_')}
    e_d1 = e_in['qh_'] + e_in['ph_'] + u * A(v['d1'])
    e_d2 = e_in['pl_'] + e_in['ql_'] + u * A(v['d2'])
    e_wps = e_in['ph_'] + e_in['pl_'] + u * A(v['wps'])
    e_s = e_d1 + e_d2 + u * A(v['s'])
    e_wp = u * A(v['wp'])
    e_wq = u * A(v['wq'])
    e_wph = u * A(v['wph'])
    e_wqh = u * A(v['wqh'])
    if v['_branch'] == 1:
        r, wps = (v['r'], v['wps'])
        e_r = e_s / wps + A(r) * e_wps / wps + u * A(r) + h * sub(r)
        e_KL = e_r / (1 + r) + (0.0 if sub(r) and r != 0 else lam * u * A(v['KL']) + h * sub(v['KL']))
    else:
        L2 = math.log(2)
        e_l2 = lam * u * L2
        if v['_fp']:
            e_Lp = e_wp / v['wp'] + lam * u * A(v['Lp'])
        else:
            lwh = math.log(v['wph'])
            e_Lp = e_wph / v['wph'] + lam * u * abs(lwh) + e_l2 + u * A(v['Lp'])
        if v['_fq']:
            e_Lq = e_wq / v['wq'] + lam * u * A(v['Lq'])
        else:
            lwh = math.log(v['wqh'])
            e_Lq = e_wqh / v['wqh'] + lam * u * abs(lwh) + e_l2 + u * A(v['Lq'])
        e_KL = e_Lp + e_Lq + u * A(v['KL'])
    if v['_fp']:
        e_gp = e_wp / v['wp'] ** 2 + u * A(v['gp']) + h * sub(v['gp'])
    else:
        e_gp = 0.5 * e_wph / v['wph'] ** 2 + u * A(v['gp']) + h * sub(v['gp'])
    if v['_fq']:
        e_gq = e_wq / v['wq'] ** 2 + u * A(v['gq']) + h * sub(v['gq'])
    else:
        e_gq = 0.5 * e_wqh / v['wqh'] ** 2 + u * A(v['gq']) + h * sub(v['gq'])
    return {k: SAFETY * x for k, x in {'KL': e_KL, 'P.low': e_gp, 'P.high': e_gp, 'Q.low': e_gq, 'Q.high': e_gq}.items()}

def closed_form_normal(dt, lam, kl):
    u = FMT[dt]['u']
    return {'KL': SAFETY * (4 + lam) * u * abs(kl), 'grad_rel': SAFETY * 2 * u}
