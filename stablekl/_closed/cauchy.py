"""Repaired KL divergence between two Cauchy distributions: formula, analytic gradients, and the
entry point run_rewrite_torch used by the benchmark (dtype in, compute dtype, outputs per component).
The derivation is in the paper's appendix on interface A."""
from __future__ import annotations
import math
import numpy as np
import torch
NAME = "cauchy"
PARAMS = ("loc", "scale")
TD = {'fp16': torch.float16, 'fp32': torch.float32, 'fp64': torch.float64}
UI = {'fp16': np.uint16, 'fp32': np.uint32, 'fp64': np.uint64}
NPF = {'fp16': np.float16, 'fp32': np.float32, 'fp64': np.float64}
LIM = {torch.float16: 14, torch.float32: 60, torch.float64: 500}
EMAX = {torch.float16: 15, torch.float32: 127, torch.float64: 1023}
COMPS = ['KL', 'P.loc', 'P.scale', 'Q.loc', 'Q.scale']

def pow2(k, like):
    return torch.exp2(k.to(torch.float64)).to(like.dtype)

def _exp(x):
    _, e = torch.frexp(x.to(torch.float32) if x.dtype == torch.float16 else x)
    return e.to(torch.int64)

def _split(k, like):
    L = LIM[like.dtype]
    k1 = k.clamp(-L, L)
    k2 = k - k1
    out = (pow2(k1, like), pow2(k2, like))
    for c in out:
        mant, _ = torch.frexp(c.to(torch.float32) if c.dtype == torch.float16 else c)
        assert bool(torch.all(mant == 0.5)) and bool(torch.all(torch.isfinite(c))), 'pow2 not exact'
    return out

class CauchyKLC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pl, ps, ql, qs):
        dt = ps.dtype
        hflag = (torch.maximum(pl.abs(), ql.abs()) >= 2.0 ** (EMAX[dt] - 1)).to(torch.int64)
        H = torch.where(hflag == 1, torch.full_like(pl, 0.5), torch.ones_like(pl))
        d2 = pl * H - ql * H
        e1 = _exp(torch.maximum(ps, qs))
        e2 = _exp(d2.abs()) + hflag
        e = torch.where(d2 == 0, e1, torch.maximum(e1, e2))
        k = 4 - e
        c1, c2 = _split(k, ps)
        i1, i2 = _split(-k, ps)
        cb1, cb2 = _split(k + hflag, ps)
        a = (ps - qs) * c1 * c2
        b = d2 * cb1 * cb2
        p_ = ps * c1 * c2
        q_ = qs * c1 * c2
        B2 = b * b
        num = a * a + B2
        den = 4 * (p_ * q_)
        r = num / den
        S = p_ + q_
        Np = S * S + B2
        ok = torch.isfinite(r)
        kl1 = torch.log1p(torch.where(ok, r, torch.zeros_like(r)))
        two = torch.full_like(ps, 2)
        tN = torch.log(Np) + (-2 * k).to(dt) * torch.log(two)
        kl2 = tN - (torch.log(ps) + torch.log(qs) + torch.log(torch.full_like(ps, 4)))
        ctx.save_for_backward(a, b, S, B2, Np, ps, qs, i1, i2, k)
        return torch.where(ok, kl1, kl2)

    @staticmethod
    def backward(ctx, g):
        a, b, S, B2, Np, ps, qs, i1, i2, k = ctx.saved_tensors
        aS = a * S
        b2x = 2 * b
        Npc = Np * i1 * i2
        fin = torch.isfinite(Npc)
        ck = pow2(torch.where(fin, torch.zeros_like(k), k), b)
        gPl = torch.where(fin, b2x / torch.where(fin, Npc, torch.ones_like(Npc)), b2x / Np * ck)
        gPs = (aS - B2) / Np / ps
        gQs = (-aS - B2) / Np / qs
        return (g * gPl, g * gPs, g * -gPl, g * gQs)

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    leaves = [to_t(x, dt_in).to(TD[cdt]).requires_grad_(True) for x in (P["loc"], P["scale"], Q["loc"], Q["scale"])]
    kl = CauchyKLC.apply(*leaves)
    kl.sum().backward()
    vals = {"KL": kl.detach()}
    for name, leaf in zip(("P.loc", "P.scale", "Q.loc", "Q.scale"), leaves):
        vals[name] = leaf.grad.detach()
    return {k: v.cpu().numpy().astype(NPF[cdt]) for k, v in vals.items()}
