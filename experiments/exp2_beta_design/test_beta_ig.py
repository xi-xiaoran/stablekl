import os, sys, time, numpy as np, torch, mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beta_ig as B
mp.mp.dps = 60
fails = 0

def check(ok, msg):
    global fails
    fails += not ok
    print(('ok  ' if ok else 'FAIL ') + msg, flush=True)

def kl_mp(a1, b1, a2, b2):
    a1, b1, a2, b2 = map(mp.mpf, (a1, b1, a2, b2))
    lB = lambda a, b: mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)
    return lB(a2, b2) - lB(a1, b1) + (a1 - a2) * mp.digamma(a1) + (b1 - b2) * mp.digamma(b1) + (a2 - a1 + b2 - b1) * mp.digamma(a1 + b1)

def eig_mp(a, b):
    return float((mp.mpf(a) * kl_mp(a + 1, b, a, b) + mp.mpf(b) * kl_mp(a, b + 1, a, b)) / (a + b))
rng = np.random.default_rng(0)
grid = [(1, 1), (2, 1), (1, 7), (50, 50), (3, 997)] + [tuple((int(x) for x in np.maximum(1, np.round(10 ** rng.uniform(0, 5.3, 2))))) for _ in range(60)] + [(150000, 150000), (1, 250000), (2000000, 3000000)]
tr = np.array([eig_mp(a, b) for a, b in grid])
ex = np.array([float(B.exact_eig(a, b)) for a, b in grid])
check(float(np.max(np.abs(ex / tr - 1))) < 1e-07, f'exact_eig (FP64 identity) vs 60-digit arithmetic on {len(grid)} posteriors up to 5e6 pulls: max rel err {np.max(np.abs(ex / tr - 1)):.1e}')
small = [g for g in grid if sum(g) <= 400000]
trs = np.array([eig_mp(a, b) for a, b in small])
rw = np.array([B.eig_rewrite(float(a), float(b)) for a, b in small])
of = np.array([B.eig_official(float(a), float(b)) for a, b in small])
print(f'   Rewrite FP32 vs truth on {len(small)} posteriors (a + b <= 4e5): max rel err {np.max(np.abs(rw / trs - 1)):.1e}; official FP32: max rel err {np.max(np.abs(of / trs - 1)):.1e}, non-positive {int((of <= 0).sum())}')
st_ = np.array([B.eig_single('strict', float(a), float(b)) for a, b in small])
iw = int(np.argmax(np.abs(rw / trs - 1)))
print(f'   worst Rewrite case: posterior Beta{small[iw]}, truth {trs[iw]:.4e}, Rewrite {rw[iw]:.4e}; strict mode max rel err {np.max(np.abs(st_ / trs - 1)):.1e}')
check(float(np.max(np.abs(rw / trs - 1))) < 0.0001 and float(np.max(np.abs(st_ / trs - 1))) < 1e-06, 'Rewrite FP32 EIG within 1e-4 and strict mode within 1e-6 of the truth in the range of the experiment')
aa = np.array([float(a) for a, b in small])
bb = np.array([float(b) for a, b in small])
check(all((np.array_equal(B._tables(p, aa, bb), np.array([B.eig_single(p, x, y) for x, y in small])) for p in ('official', 'rewrite', 'strict'))), 'batched tables are bitwise identical to single evaluations (official, rewrite, strict)')
r = np.array([0.3, 0.6, 0.8])
U = np.random.default_rng(1).random((3, 500), dtype=np.float32)
st = np.zeros(3, dtype=int)
a = B.run('exact', r, U, st, 300)
b = B.run('exact', r, U, st, 300)
check(a['pulls'] == b['pulls'] and sum(a['pulls']) == 300 and (a['wrong'] == 0) and (max(a['pulls']) - min(a['pulls']) <= 30), f"exact policy: deterministic, never wrong by its own measure, pulls {a['pulls']}")
c = B.run('rewrite', r, U, st, 300)
check(sum(c['pulls']) == 300 and c['wrong'] == 0, f"rewrite policy on a short run: pulls {c['pulls']}, wrong {c['wrong']}")
w = B.run('exact', r, U, np.array([100, 0, 20]), 200)
check(w['pulls'][0] >= 100 and sum(w['pulls']) == 320, f"warm start counts the prior pulls: {w['pulls']}")
rr = np.array([0.2, 0.5, 0.7, 0.9])
UU = np.random.default_rng(5).random((4, 3000), dtype=np.float32)
s0 = np.zeros(4, dtype=int)
for pol in ('exact', 'official', 'rewrite'):
    x1, y1 = (B.run(pol, rr, UU, s0, 2000), B.run_batch(pol, rr, UU, s0, 2000, 1))
    check(x1['pulls'] == y1['pulls'] and x1['wrong'] == y1['wrong'] and (abs(x1['post_var_sum'] - y1['post_var_sum']) == 0), f"batch size 1 reproduces the sequential run exactly ({pol}: pulls {y1['pulls']})")
yb = B.run_batch('exact', rr, UU, np.array([50, 0, 0, 10]), 2000, 100)
check(sum(yb['pulls']) == 2060 and all(((p - s) % 100 == 0 for p, s in zip(yb['pulls'], [50, 0, 0, 10]))) and (yb['wrong'] == 0), f"batches of 100: pulls {yb['pulls']}")
x = np.round(10 ** np.random.default_rng(3).uniform(0, 5, 100000))
t0 = time.time()
B._tables('official', x, x[::-1].copy())
to_ = time.time() - t0
t0 = time.time()
B._tables('rewrite', x, x[::-1].copy())
tr_ = time.time() - t0
print(f'   100,000 posteriors in a batch: official {to_:.2f} s, rewrite {tr_:.2f} s')
print('beta_ig tests:', 'PASS' if fails == 0 else f'{fails} FAILED')
sys.exit(1 if fails else 0)
