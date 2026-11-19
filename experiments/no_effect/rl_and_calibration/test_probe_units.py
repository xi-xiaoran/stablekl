import os, sys, math, torch, mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe as P
from softmax_fix import fixed_log_softmax
fails = 0

def check(ok, msg):
    global fails
    fails += not ok
    print(('ok  ' if ok else 'FAIL ') + msg)
mp.mp.prec = 200
z = torch.tensor([[9.0, -9.0, -9.0]], requires_grad=True)
loss = -fixed_log_softmax(z)[0, 0]
loss.backward()
check(abs(loss.item() - 3.045997e-08) < 1e-13 and abs(z.grad[0, 0].item() + 3.045996e-08) < 1e-13, f"fixed log-softmax reproduces the paper's example: {loss.item():.6e}, {z.grad.tolist()}")
for x in (1e-09, -3e-06, 0.001, -0.2, 0.49, 0.51, 2.0, -5.0):
    t = torch.tensor([x], dtype=torch.float64)
    ex = mp.expm1(mp.mpf(x)) - mp.mpf(x)
    check(abs(float((mp.mpf(P.psi(t).item()) - ex) / ex)) < 1e-14, f'psi({x}) relative error {float((mp.mpf(P.psi(t).item()) - ex) / ex):.1e}')
torch.manual_seed(0)
V = 40
for scale in (0.0001, 0.01, 0.5):
    zr = torch.randn(3, V, dtype=torch.float64) * 2
    zp = zr + scale * torch.randn(3, V, dtype=torch.float64)
    a = torch.tensor([0, 7, 39])
    ex = P.exact(zp, zr, a)
    for i in range(3):
        Z, R = ([mp.mpf(v) for v in zp[i].tolist()], [mp.mpf(v) for v in zr[i].tolist()])
        lz, lr = (mp.log(mp.fsum((mp.exp(v) for v in Z))), mp.log(mp.fsum((mp.exp(v) for v in R))))
        q = [mp.exp(v - lz) for v in Z]
        p = [mp.exp(v - lr) for v in R]
        r = Z[a[i]] - lz - (R[a[i]] - lr)
        k3 = mp.exp(-r) + r - 1
        kf = mp.fsum((pj * (mp.log(pj) - mp.log(qj)) for pj, qj in zip(p, q)))
        kr = mp.fsum((qj * (mp.log(qj) - mp.log(pj)) for pj, qj in zip(p, q)))
        gf = [qj - pj for pj, qj in zip(p, q)]
        gr = [qj * (mp.log(qj) - mp.log(pj) - kr) for pj, qj in zip(p, q)]
        e = lambda v, w: float(abs(mp.mpf(v) - w) / abs(w)) if w != 0 else float(abs(mp.mpf(v)))
        errs = [e(ex['k1'][0][i].item(), r), e(ex['k3'][0][i].item(), k3), e(ex['fwd'][0][i].item(), kf), e(ex['rev'][0][i].item(), kr), max((float(abs(mp.mpf(ex['fwd'][1][i, j].item()) - gf[j])) for j in range(V))) / float(max((abs(x) for x in gf))), max((float(abs(mp.mpf(ex['rev'][1][i, j].item()) - gr[j])) for j in range(V))) / float(max((abs(x) for x in gr)))]
        check(max(errs) < 1e-11, f'scale {scale:g} row {i}: FP64 references vs mpmath, max rel {max(errs):.1e}')
acc = P.Acc()
acc.add(torch.zeros(2), torch.tensor([[0.0, 1e-07], [0.0, 0.0]]), torch.zeros(2), torch.zeros(2, 2), op=torch.zeros(2, dtype=torch.float64))
r = acc.result()
check(abs(r['grad_max_abs_where_exact_zero'] - 1e-07) < 1e-12 and r['grad_cosine'] is None and (r['zero_ref_tokens'] == 2), f"zero exact gradient: reports max |g| = {r['grad_max_abs_where_exact_zero']}, cosine {r['grad_cosine']}")
rv = torch.tensor([1e-06, 2e-06, 3e-06], dtype=torch.float64)
rg = torch.tensor([[0.001, -0.001]] * 3, dtype=torch.float64)
g = rg.float()
acc = P.Acc()
acc.add(torch.full((3,), float('nan')), g, rv, rg, op=rv)
r = acc.result()
check(r['mean_rel'] is None and r['value_nonfinite'] == 3 and (r['value_ge_10pct'] == 1.0) and (r['op_ge_1pct'] == 1.0) and (abs(r['ref_mean'] - 2e-06) < 1e-18), f"all-NaN values: mean_rel {r['mean_rel']}, non-finite {r['value_nonfinite']}, >=10% share {r['value_ge_10pct']}, ref_mean {r['ref_mean']}")
acc = P.Acc()
acc.add(torch.tensor([float('inf')] * 3), g, rv, rg)
r = acc.result()
check(r['mean_rel'] is None and r['value_nonfinite'] == 3 and (r['value_rel_median'] == float('inf')), f"all-inf values: mean_rel {r['mean_rel']}, median {r['value_rel_median']}")
acc = P.Acc()
acc.add(torch.tensor([1e-06, float('inf'), 3e-06]), g, rv, rg)
r = acc.result()
check(r['mean_rel'] is None and r['value_nonfinite'] == 1 and (abs(r['value_ge_10pct'] - 1 / 3) < 1e-12) and (r['grad_nonfinite'] == 0) and (r['grad_cosine'] is not None), f"mixed inf value: mean_rel {r['mean_rel']}, >=10% share {r['value_ge_10pct']:.3f}, grad cosine {r['grad_cosine']}")
gb = g.clone()
gb[1, 0] = float('nan')
acc = P.Acc()
acc.add(rv.float(), gb, rv, rg)
r = acc.result()
check(r['mean_rel'] is not None and r['mean_rel'] < 1e-06 and (r['value_nonfinite'] == 0) and (r['grad_nonfinite'] == 1) and (r['grad_cosine'] is None) and (r['grad_rel_p95'] == float('inf')), f"NaN gradient only: value mean_rel {r['mean_rel']:.1e} kept, grad non-finite {r['grad_nonfinite']}, cosine {r['grad_cosine']}, grad p95 {r['grad_rel_p95']}")
acc = P.Acc()
acc.add(rv.float(), g, rv, rg)
r = acc.result()
check(r['mean_rel'] < 1e-06 and r['value_nonfinite'] == 0 and (r['grad_nonfinite'] == 0) and (abs(r['grad_cosine'] - 1) < 1e-06), f"all finite: mean_rel {r['mean_rel']:.1e}, cosine {r['grad_cosine']:.7f}")
print('probe unit tests:', 'PASS' if fails == 0 else f'{fails} FAILED')
sys.exit(1 if fails else 0)
