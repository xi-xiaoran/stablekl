from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from fingerprint import code_md5
import argparse, hashlib, importlib.util, json, os, sys, time
import numpy as np, torch
from scipy.special import digamma
from torch.distributions import Beta, kl_divergence
HERE = os.path.dirname(os.path.abspath(__file__))

def _load(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m
from stablekl._closed import beta as FIX
F32 = torch.float32

def exact_eig(a, b):
    a, b = (np.float64(a), np.float64(b))
    s = a + b
    ps = digamma(s + 1)
    k1 = np.log(s / a) + digamma(a + 1) - ps
    k0 = np.log(s / b) + digamma(b + 1) - ps
    return (a * k1 + b * k0) / s

def _tables(policy, a, b, chunk=200000):
    if policy == 'exact':
        return exact_eig(a, b)
    out = np.empty(len(a))
    for r in range(0, len(a), chunk):
        A, Bq = (torch.tensor(a[r:r + chunk], dtype=F32), torch.tensor(b[r:r + chunk], dtype=F32))
        one = torch.ones_like(A)
        if policy == 'official':
            k1 = kl_divergence(Beta(A + one, Bq), Beta(A, Bq))
            k0 = kl_divergence(Beta(A, Bq + one), Beta(A, Bq))
        elif policy == 'rewrite':
            with torch.no_grad():
                k1 = FIX.BetaKLC.apply(A + one, Bq, A, Bq)
                k0 = FIX.BetaKLC.apply(A, Bq + one, A, Bq)
        elif policy == 'strict':
            with torch.no_grad():
                D = lambda x: x.double()
                k1 = FIX.BetaKLC.apply(D(A + one), D(Bq), D(A), D(Bq)).to(F32)
                k0 = FIX.BetaKLC.apply(D(A), D(Bq + one), D(A), D(Bq)).to(F32)
        S = A + Bq
        out[r:r + chunk] = (A / S * k1 + Bq / S * k0).double().numpy()
    return out

def eig_single(policy, a, b):
    return float(_tables(policy, np.array([float(a)]), np.array([float(b)]))[0])
eig_official = lambda a, b: eig_single('official', a, b)
eig_rewrite = lambda a, b: eig_single('rewrite', a, b)
POLICIES = ('exact', 'official', 'rewrite', 'strict')

def run(policy, rates, U, start, T, tol=0.001):
    K = len(rates)
    succ = np.concatenate([np.zeros((K, 1)), np.cumsum(U[:, :int(start.max()) + T] < rates[:, None], axis=1)], axis=1)
    J = np.arange(T + 1)[None, :] + start[:, None]
    A = 1.0 + np.take_along_axis(succ, J, axis=1)
    Bq = 1.0 + J - (A - 1.0)
    EX = exact_eig(A.ravel(), Bq.ravel()).reshape(K, T + 1)
    ES = EX if policy == 'exact' else _tables(policy, A.ravel(), Bq.ravel()).reshape(K, T + 1)
    c = np.zeros(K, dtype=np.int64)
    est = ES[:, 0].copy()
    ex = EX[:, 0].copy()
    wrong = 0
    first = None
    info = 0.0
    bad = int(((est <= 0) | ~np.isfinite(est)).sum())
    curve = []
    relmax = float(np.max(np.abs(est / ex - 1)))
    marks = set((int(x) for x in np.unique(np.round(np.logspace(1, np.log10(T), 60))))) | {T}
    for t in range(1, T + 1):
        e = np.where(np.isfinite(est), est, -np.inf)
        i = int(np.argmax(e))
        w = ex[i] < (1 - tol) * ex.max()
        wrong += w
        first = t if w and first is None else first
        info += ex[i]
        c[i] += 1
        est[i] = ES[i, c[i]]
        ex[i] = EX[i, c[i]]
        bad += not np.isfinite(est[i]) or est[i] <= 0
        relmax = max(relmax, abs(est[i] / ex[i] - 1))
        if t in marks:
            a, b = (A[np.arange(K), c], Bq[np.arange(K), c])
            curve.append((t, int(wrong), float((a * b / ((a + b) ** 2 * (a + b + 1))).sum()), int((start + c).min()), int((start + c).max())))
    a, b = (A[np.arange(K), c], Bq[np.arange(K), c])
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return dict(policy=policy, wrong=int(wrong), wrong_frac=wrong / T, first_wrong=first, info=info, post_var_sum=float(var.sum()), post_var_max=float(var.max()), pulls=(start + c).tolist(), new_pulls=c.tolist(), nonpositive_or_nonfinite=int(bad), max_rel_err_eig=relmax, sq_err=float(((a / (a + b) - rates) ** 2).sum()), curve=curve)
SETTINGS = {'A': dict(K=10, T=100000, warm=False), 'B': dict(K=10, T=20000, warm=True)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', required=True)
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--out', required=True)
    ap.add_argument('--T', type=int, default=0)
    a = ap.parse_args()
    S = dict(SETTINGS[a.setting])
    S['T'] = a.T or S['T']
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    meta = dict(setting=a.setting, **S, torch=torch.__version__, code={f: code_md5(f, HERE) for f in ('beta_ig.py', 'fix_beta.py')})
    for seed in (int(x) for x in a.seeds.split(',')):
        path = os.path.join(a.out, f'{a.setting}_seed{seed}.json')
        if os.path.exists(path):
            print(path, 'exists, skipped')
            continue
        rng = np.random.default_rng(1000 + seed)
        rates = rng.uniform(0.05, 0.95, S['K'])
        start = np.round(10 ** rng.uniform(2, 5, S['K'])).astype(int) if S['warm'] else np.zeros(S['K'], dtype=int)
        U = rng.random((S['K'], int(start.max()) + S['T'] + 1), dtype=np.float32)
        res = {}
        for pol in POLICIES:
            t0 = time.time()
            res[pol] = run(pol, rates, U, start, S['T'])
            res[pol]['seconds'] = time.time() - t0
            r = res[pol]
            print(f"{a.setting} seed {seed} {pol:8s}: wrong {r['wrong_frac']:.4f} (first {r['first_wrong']}), var sum {r['post_var_sum']:.3e}, pulls min/max {min(r['pulls'])}/{max(r['pulls'])}, {r['seconds']:.0f} s", flush=True)
        json.dump(dict(meta=meta, seed=seed, rates=rates.tolist(), start=start.tolist(), results=res), open(path + '.tmp', 'w'))
        os.replace(path + '.tmp', path)

def run_batch(policy, rates, U, start, T, m, tol=0.001):
    K = len(rates)
    R = T // m
    succ = np.concatenate([np.zeros((K, 1)), np.cumsum(U[:, :int(start.max()) + T] < rates[:, None], axis=1)], axis=1)
    J = np.arange(R + 1)[None, :] * m + start[:, None]
    A = 1.0 + np.take_along_axis(succ, J, axis=1)
    Bq = 1.0 + J - (A - 1.0)
    EX = exact_eig(A.ravel(), Bq.ravel()).reshape(K, R + 1)
    ES = EX if policy == 'exact' else _tables(policy, A.ravel(), Bq.ravel()).reshape(K, R + 1)
    c = np.zeros(K, dtype=np.int64)
    est = ES[:, 0].copy()
    ex = EX[:, 0].copy()
    wrong = 0
    first = None
    lost = 0.0
    bad = int(((est <= 0) | ~np.isfinite(est)).sum())
    relmax = float(np.max(np.abs(est / ex - 1)))
    for r in range(1, R + 1):
        e = np.where(np.isfinite(est), est, -np.inf)
        i = int(np.argmax(e))
        w = ex[i] < (1 - tol) * ex.max()
        wrong += w
        first = r if w and first is None else first
        lost += 1 - ex[i] / ex.max()
        c[i] += 1
        est[i] = ES[i, c[i]]
        ex[i] = EX[i, c[i]]
        bad += not np.isfinite(est[i]) or est[i] <= 0
        relmax = max(relmax, abs(est[i] / ex[i] - 1))
    a, b = (A[np.arange(K), c], Bq[np.arange(K), c])
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return dict(policy=policy, m=m, rounds=R, wrong=int(wrong), wrong_frac=wrong / R, first_wrong=first, mean_step_regret=lost / R, post_var_sum=float(var.sum()), post_var_max=float(var.max()), pulls=(start + c * m).tolist(), nonpositive_or_nonfinite=int(bad), n_values=int(K + R), max_rel_err_eig=relmax, sq_err=float(((a / (a + b) - rates) ** 2).sum()))

def main_batch(out, seeds=(0, 1, 2, 3, 4), T=100000, K=10, ms=(1, 100, 1000, 10000)):
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(2)
    meta = dict(T=T, K=K, ms=list(ms), torch=torch.__version__, code={f: code_md5(f, HERE) for f in ('beta_ig.py', 'fix_beta.py')})
    for warm in (False, True):
        for seed in seeds:
            path = os.path.join(out, f"batch_{('warm' if warm else 'cold')}_seed{seed}.json")
            if os.path.exists(path):
                print(path, 'exists, skipped')
                continue
            rng = np.random.default_rng(1000 + seed)
            rates = rng.uniform(0.05, 0.95, K)
            start = np.round(10 ** rng.uniform(2, 5, K)).astype(int) if warm else np.zeros(K, dtype=int)
            U = rng.random((K, int(start.max()) + T + 1), dtype=np.float32)
            res = {}
            for m in ms:
                for pol in ('exact', 'official', 'rewrite'):
                    t0 = time.time()
                    r = run_batch(pol, rates, U, start, T, m)
                    r['seconds'] = time.time() - t0
                    res[f'{m}|{pol}'] = r
                e, o, w = (res[f'{m}|exact'], res[f'{m}|official'], res[f'{m}|rewrite'])
                print(f"{('warm' if warm else 'cold')} seed {seed} m {m:5d}: official wrong {o['wrong_frac']:.3f}, var x{o['post_var_sum'] / e['post_var_sum']:.3f}, pulls {min(o['pulls'])}/{max(o['pulls'])} | rewrite wrong {w['wrong_frac']:.4f}, var x{w['post_var_sum'] / e['post_var_sum']:.4f} | exact pulls {min(e['pulls'])}/{max(e['pulls'])}", flush=True)
            json.dump(dict(meta=meta, warm=warm, seed=seed, rates=rates.tolist(), start=start.tolist(), results=res), open(path + '.tmp', 'w'))
            os.replace(path + '.tmp', path)
if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'batch':
        main_batch(sys.argv[2])
    else:
        main()
