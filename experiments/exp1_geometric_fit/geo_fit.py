import sys, math, warnings, json
import torch, mpmath as mp
from torch.distributions import Geometric, kl_divergence
import os
from stablekl._closed import geometric_probs as FP, geometric_logits as FL
warnings.simplefilter('ignore')
torch.set_num_threads(2)
mp.mp.dps = 60

def true_kl(p, q):
    if p == q:
        return 0.0
    if p == 0:
        return float('nan')
    if p == 1:
        return math.inf if q == 0 else float(-mp.log(mp.mpf(q)))
    if q == 0 or q == 1:
        return math.inf
    p, q = (mp.mpf(p), mp.mpf(q))
    return float(mp.log(p / q) + (1 - p) / p * mp.log((1 - p) / (1 - q)))

def sig(th):
    th = mp.mpf(th)
    return 1 / (1 + mp.exp(-th))

def loss(method, param, direction, th, star, dt):
    if param == 'probs':
        a = torch.sigmoid(th.to(dt))
        b = torch.full_like(a, star)
    else:
        a = th.to(dt)
        b = torch.full_like(a, star)
    P, Q = (a, b) if direction == 'rev' else (b, a)
    if method == 'official':
        mk = (lambda v: Geometric(probs=v)) if param == 'probs' else lambda v: Geometric(logits=v)
        return kl_divergence(mk(P), mk(Q)).sum()
    if param == 'probs':
        if method == 'C':
            return FP.GeometricProbsKLC.apply(P, Q).sum()
        return FP.GeometricProbsKLC.apply(P.double(), Q.double()).to(dt).sum()
    if method == 'C':
        return FL.GeometricLogitsKLC.apply(P, Q, None).sum()
    return FL.GeometricLogitsKLC.apply(P.double(), Q.double(), {torch.float16: 'fp16', torch.float32: 'fp32'}[dt]).to(dt).sum()

def run(method, param, direction, pstar, dt, steps=3000, lr=0.01, p0=None):
    p0 = min(0.5, 3 * pstar) if p0 is None else p0
    th_star = math.log(pstar) - math.log1p(-pstar)
    star = float(torch.tensor(pstar if param == 'probs' else th_star, dtype=dt))
    p_opt = star if param == 'probs' else float(sig(star))
    th = torch.tensor([math.log(p0) - math.log1p(-p0)], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([th], lr=lr)
    neg = zero = nonfin = 0
    tail = []
    for s in range(steps):
        L = loss(method, param, direction, th, star, dt)
        v = L.item()
        neg += v < 0
        zero += v == 0
        nonfin += not math.isfinite(v)
        opt.zero_grad()
        L.backward()
        if not torch.isfinite(th.grad).all():
            nonfin += 1
            break
        opt.step()
        if s >= steps - 500:
            pt = float(torch.sigmoid(th.detach().to(dt)).double()) if param == 'probs' else float(sig(float(th.detach().to(dt))))
            tail.append(abs(pt - p_opt) / p_opt)
    pt = float(torch.sigmoid(th.detach().to(dt)).double()) if param == 'probs' else float(sig(float(th.detach().to(dt))))
    kl = true_kl(pt, p_opt) if direction == 'rev' else true_kl(p_opt, pt)
    L0, g0 = opt_check(method, param, direction, star, dt)
    return dict(final_rel_err=abs(pt - p_opt) / p_opt, tail_mean_rel_err=sum(tail) / max(len(tail), 1), true_kl=kl, neg=neg, zero=zero, nonfinite=nonfin, loss_at_opt=L0, grad_at_opt=g0)

def opt_check(method, param, direction, star, dt):
    a = torch.tensor([star], dtype=dt, requires_grad=True)
    b = torch.tensor([star], dtype=dt)
    P, Q = (a, b) if direction == 'rev' else (b, a)
    if method == 'official':
        mk = (lambda v: Geometric(probs=v)) if param == 'probs' else lambda v: Geometric(logits=v)
        L = kl_divergence(mk(P), mk(Q)).sum()
    elif param == 'probs':
        L = (FP.GeometricProbsKLC.apply(P, Q) if method == 'C' else FP.GeometricProbsKLC.apply(P.double(), Q.double()).to(dt)).sum()
    else:
        tg = {torch.float16: 'fp16', torch.float32: 'fp32'}[dt]
        L = (FL.GeometricLogitsKLC.apply(P, Q, None) if method == 'C' else FL.GeometricLogitsKLC.apply(P.double(), Q.double(), tg).to(dt)).sum()
    L.backward()
    return (L.item(), a.grad.item())
if __name__ == '__main__':
    res = []
    for dt in (torch.float16, torch.float32):
        for param in ('probs', 'logits'):
            for direction in ('rev', 'fwd'):
                for pstar in (0.9, 0.3, 0.01, 0.001):
                    for method in ('official', 'C', 'D'):
                        r = run(method, param, direction, pstar, dt)
                        r.update(dt=str(dt)[-7:], param=param, dir=direction, pstar=pstar, method=method)
                        res.append(r)
                        print(f"{r['dt']} {param:6s} {direction} p*={pstar:<6g} {method:8s} | rel err final {r['final_rel_err']:.2e} tail {r['tail_mean_rel_err']:.2e} | true KL {r['true_kl']:.2e} | loss<0 {r['neg']:4d} =0 {r['zero']:4d} nonfin {r['nonfinite']} | at optimum: loss {r['loss_at_opt']:.2e} grad {r['grad_at_opt']:.2e}", flush=True)
    json.dump(res, open('geo_fit.json', 'w'), indent=1)
