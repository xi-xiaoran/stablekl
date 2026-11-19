import os, sys
import numpy as np, torch, mpmath as mp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import run_eval as R
dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
torch.manual_seed(0)
T, V = (48, 32000)
fails = 0
base = torch.randn(T, V, dtype=torch.float64) * 2.5
for scale, label in ((0.003, 'W8-like'), (0.03, 'W4-like')):
    b = base.half().to(dev)
    q = (base + scale * torch.randn(T, V, dtype=torch.float64)).half().to(dev)
    q[:4] = b[:4]
    out = R.methods(b, q, None)
    assert len({v[0].numel() for v in out.values()}) == 1, 'token counts differ'
    for dn in ('fp16', 'bf16', 'fp32'):
        per, cv, given, comp = out[f'logitsA:{dn}']
        per, given = (per.cpu().numpy(), given.cpu().numpy())
        cr = R.cr_frac(per, given, dn)
        zero_ok = bool((per[:4] == 0).all())
        mean_rel = abs(cv['batchmean'] - per.mean()) / abs(per.mean())
        tol = {'fp16': 0.002, 'bf16': 0.016, 'fp32': 1e-06}[dn]
        crv = R._round_once(torch.from_numpy(given), R.DT[dn]).double().numpy()
        bad = np.flatnonzero(per != crv)
        fi = torch.finfo(R.DT[dn])
        one_unit = len(bad) == 1 and abs(per[bad[0]] - crv[bad[0]]) <= fi.eps * abs(crv[bad[0]]) + fi.tiny * fi.eps
        ok = (cr >= 0.999 or one_unit) and zero_ok and (mean_rel < tol)
        fails += not ok
        off = out[f'kl_div:{dn}'][0].cpu().numpy()
        print(f"{label} {dn}: logitsA CR {cr:.4f}, identical rows -> 0: {zero_ok}, chunk vs mean of rows {mean_rel:.1e}  | official kl_div CR {R.cr_frac(off, given if dn == 'fp32' else out[f'kl_div:{dn}'][2].cpu().numpy(), dn):.3f}, negative tokens {int((off < 0).sum())}  {('ok' if ok else 'FAIL')}")
mp.mp.prec = 150
b = base.half()
q = (base + 0.003 * torch.randn(T, V, dtype=torch.float64)).half()
ref, naive = (R.exact_kl(b.float(), q.float()), R.naive_kl(b.float(), q.float()))
for i in (5, 6):
    bv, qv = ([mp.mpf(x) for x in b[i].double().tolist()], [mp.mpf(x) for x in q[i].double().tolist()])
    lsb, lsq = (mp.log(mp.fsum((mp.exp(x) for x in bv))), mp.log(mp.fsum((mp.exp(x) for x in qv))))
    ex = mp.fsum((mp.exp(x - lsb) * (x - lsb - (y - lsq)) for x, y in zip(bv, qv)))
    er, en = (float(abs(mp.mpf(ref[i].item()) - ex) / ex), float(abs(mp.mpf(naive[i].item()) - ex) / ex))
    ok = er <= 1e-13
    fails += not ok
    print(f"reference vs 150-bit mpmath, row {i} (KL {float(ex):.2e}): cancellation-free {er:.1e}, naive {en:.1e}  {('ok' if ok else 'FAIL')}")
print('logits-path (exp1) tests:', 'PASS' if fails == 0 else f'{fails} FAILED')
sys.exit(1 if fails else 0)
