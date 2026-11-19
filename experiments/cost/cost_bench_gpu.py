import argparse, sys, os, time, json, importlib.util, platform, statistics, numpy as np, torch, torch.nn.functional as F
from torch.distributions import Normal, Exponential, Gamma, Bernoulli, Laplace, Poisson, Beta, kl_divergence
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stablekl import interface_b as K
from stablekl.logits import logits_kl

def run(dev, out, quick):
    cuda = dev.type == 'cuda'
    sync = (lambda: torch.cuda.synchronize()) if cuda else lambda: None
    R = dict(env=dict(torch=torch.__version__, device=torch.cuda.get_device_name(0) if cuda else platform.processor() or 'cpu', threads=torch.get_num_threads(), python=platform.python_version()), closed=[], kldiv=[], logits=[], failed=[])

    def bench(fn, n):
        fn()
        sync()
        res = []
        for _ in range(3 if quick else 7):
            t = time.perf_counter()
            for _ in range(n):
                fn()
            sync()
            res.append((time.perf_counter() - t) / n * 1000.0)
        return statistics.median(res)

    def load(name):
        return importlib.import_module('stablekl._closed.' + name)
    TD = {'fp32': torch.float32, 'fp64': torch.float64}
    NPF = {'fp32': np.float32, 'fp64': np.float64}
    to_t = lambda x, dt: torch.tensor(np.asarray(x), dtype=TD[dt], device=dev)
    for N in (4096,) if quick else (65536, 1048576):
        rng = np.random.default_rng(0)
        cases = {'Exponential': ('exponential', dict(rate=rng.uniform(0.5, 2, N)), lambda P: Exponential(P['rate'])), 'Laplace': ('laplace', dict(loc=rng.normal(size=N), scale=rng.uniform(0.5, 2, N)), lambda P: Laplace(P['loc'], P['scale'])), 'Poisson': ('poisson', dict(rate=rng.uniform(0.5, 20, N)), lambda P: Poisson(P['rate'])), 'Normal': ('normal', dict(loc=rng.normal(size=N), scale=rng.uniform(0.5, 2, N)), lambda P: Normal(P['loc'], P['scale'])), 'Bernoulli (logits)': ('bernoulli_logits', dict(logits=rng.normal(size=N) * 2), lambda P: Bernoulli(logits=P['logits'])), 'Gamma': ('gamma', dict(concentration=rng.uniform(0.5, 5, N), rate=rng.uniform(0.5, 2, N)), lambda P: Gamma(P['concentration'], P['rate'])), 'Beta': ('beta', dict(concentration1=rng.uniform(0.5, 5, N), concentration0=rng.uniform(0.5, 5, N)), lambda P: Beta(P['concentration1'], P['concentration0']))}
        for name, (mod, Pn, mk) in cases.items():
            try:
                M = load(mod)
                Qn = {k: v * (1 + 0.001 * rng.normal(size=N)) for k, v in Pn.items()}

                def off(dt='fp32'):
                    P = {k: to_t(v, 'fp32').to(TD[dt]).requires_grad_(True) for k, v in Pn.items()}
                    Q = {k: to_t(v, 'fp32').to(TD[dt]).requires_grad_(True) for k, v in Qn.items()}
                    kl_divergence(mk(P), mk(Q)).sum().backward()
                n = 10 if N <= 65536 else 3
                a = bench(off, n)
                a64 = bench(lambda: off('fp64'), n)
                c = bench(lambda: M.run_rewrite_torch(TD, to_t, 'fp32', 'fp32', Pn, Qn, NPF), max(1, n // 3))
                d = bench(lambda: M.run_rewrite_torch(TD, to_t, 'fp32', 'fp64', Pn, Qn, NPF), max(1, n // 3))
                R['closed'].append(dict(name=name, n=N, official_ms=a, official_fp64_ms=a64, rewrite_ms=c, ours_ms=d))
                print(f'[{dev.type}] {name:20s} N={N:8d} official {a:8.3f} ms | FP64 {a64 / a:5.1f}x | Rewrite {c / a:6.1f}x | Ours {d / a:6.1f}x', flush=True)
            except Exception as e:
                R['failed'].append(dict(case=f'closed:{name}:N={N}', error=f'{type(e).__name__}: {str(e)[:300]}'))
                print(f'[{dev.type}] {name} N={N}: FAILED {type(e).__name__}: {str(e)[:160]}', flush=True)
            if cuda:
                torch.cuda.empty_cache()
    torch.manual_seed(0)
    nb = K._near_boundary
    for B, V in ((16, 256),) if quick else ((64, 1024), (64, 32768)):
        for dname, dt in (('fp32', torch.float32), ('fp16', torch.float16)):
            for lt in (False, True):
                try:
                    x0 = torch.log_softmax(torch.randn(B, V) * 3, -1).to(dev, dt)
                    t0 = (torch.log_softmax(torch.randn(B, V) * 3, -1) if lt else torch.softmax(torch.randn(B, V) * 3, -1)).to(dev, dt)

                    def off(d=dt):
                        x = x0.to(d).clone().requires_grad_(True)
                        F.kl_div(x, t0.to(d), reduction='batchmean', log_target=lt).backward()

                    def fix():
                        x = x0.clone().requires_grad_(True)
                        K.kl_div(x, t0, reduction='batchmean', log_target=lt).backward()
                    n = 20 if V == 1024 else 3
                    a = bench(off, n)
                    a64 = bench(lambda: off(torch.float64), n)
                    c = bench(fix, max(1, n // 4))
                    K._near_boundary = lambda v, bound, d: torch.zeros(v.shape, dtype=torch.bool, device=v.device)
                    try:
                        u = bench(fix, max(1, n // 4))
                    finally:
                        K._near_boundary = nb
                    R['kldiv'].append(dict(dtype=dname, log_target=lt, B=B, V=V, official_ms=a, official_fp64_ms=a64, repaired_ms=c, repaired_nocheck_ms=u))
                    print(f'[{dev.type}] kl_div {dname} log_target={lt!s:5s} {B}x{V}: official {a:8.3f} ms | FP64 {a64 / a:5.1f}x | repaired {c / a:6.1f}x | check skipped {u / a:6.1f}x', flush=True)
                except Exception as e:
                    K._near_boundary = nb
                    R['failed'].append(dict(case=f'kldiv:{dname}:lt={lt}:{B}x{V}', error=f'{type(e).__name__}: {str(e)[:300]}'))
                    print(f'[{dev.type}] kl_div {dname} lt={lt} {B}x{V}: FAILED {type(e).__name__}: {str(e)[:160]}', flush=True)
    for dname, dt in (('fp32', torch.float32), ('bf16', torch.bfloat16), ('fp16', torch.float16)):
        if dname == 'fp16' and (not cuda):
            continue
        for B, V in ((8, 512),) if quick else ((64, 1024), (64, 32768), (32, 151936), (512, 151936)):
            if not cuda and B * V > 32 * 151936:
                continue
            try:
                t = (torch.randn(B, V) * 3).to(dev, dt)
                s0 = (t.float() + 0.01 * torch.randn(B, V, device=dev)).to(dt)

                def off(d=None):
                    s = s0.clone().requires_grad_(True)
                    ss, tt = (s, t) if d is None else (s.to(d), t.to(d))
                    F.kl_div(F.log_softmax(ss, -1), F.log_softmax(tt, -1), reduction='batchmean', log_target=True).backward()

                def ours():
                    s = s0.clone().requires_grad_(True)
                    logits_kl(t, s).backward()
                n = 20 if V == 1024 else 4
                a = bench(off, n)
                a32 = bench(lambda: off(torch.float32), n)
                a64 = bench(lambda: off(torch.float64), n)
                b = bench(ours, n)
                R['logits'].append(dict(dtype=dname, B=B, V=V, official_ms=a, official_fp32_ms=a32, official_fp64_ms=a64, logits_path_ms=b))
                print(f'[{dev.type}] logits {dname} {B}x{V}: official {a:8.3f} ms | FP32 {a32 / a:5.1f}x | FP64 {a64 / a:5.1f}x | logits path {b / a:6.1f}x', flush=True)
            except Exception as e:
                R['failed'].append(dict(case=f'logits:{dname}:{B}x{V}', error=f'{type(e).__name__}: {str(e)[:300]}'))
                print(f'[{dev.type}] logits {dname} {B}x{V}: FAILED {type(e).__name__}: {str(e)[:160]}', flush=True)
            if cuda:
                torch.cuda.empty_cache()
    os.makedirs(out, exist_ok=True)
    json.dump(R, open(os.path.join(out, f'cost_{dev.type}.json'), 'w'), indent=1)
    print(f"saved cost_{dev.type}.json ({len(R['failed'])} failed cases)", flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--devices', default='cuda,cpu')
    ap.add_argument('--quick', type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    for d in a.devices.split(','):
        run(torch.device(d), a.out, bool(a.quick))
    print('done')
