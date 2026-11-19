import sys, time, json, importlib.util, platform, statistics, numpy as np, torch, torch.nn.functional as F
from torch.distributions import Normal, Exponential, Gamma, Bernoulli, Laplace, Poisson, Beta, Cauchy, kl_divergence
torch.set_num_threads(2)
R = dict(env=dict(torch=torch.__version__, threads=2, machine=platform.machine(), python=platform.python_version()), closed=[], kldiv=[], logits=[])

def bench(fn, n):
    fn()
    out = []
    for _ in range(7):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        out.append((time.perf_counter() - t) / n * 1000.0)
    return statistics.median(out)

def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m
N = 65536
rng = np.random.default_rng(0)
TD = {'fp32': torch.float32, 'fp64': torch.float64}
NPF = {'fp32': np.float32, 'fp64': np.float64}
to_t = lambda x, dt: torch.tensor(np.asarray(x), dtype=TD[dt])
cases = {'Exponential': ('exponential', dict(rate=rng.uniform(0.5, 2, N)), lambda P: Exponential(P['rate'])), 'Laplace': ('laplace', dict(loc=rng.normal(size=N), scale=rng.uniform(0.5, 2, N)), lambda P: Laplace(P['loc'], P['scale'])), 'Poisson': ('poisson', dict(rate=rng.uniform(0.5, 20, N)), lambda P: Poisson(P['rate'])), 'Cauchy': ('cauchy', dict(loc=rng.normal(size=N), scale=rng.uniform(0.5, 2, N)), lambda P: Cauchy(P['loc'], P['scale'])), 'Normal': ('normal', dict(loc=rng.normal(size=N), scale=rng.uniform(0.5, 2, N)), lambda P: Normal(P['loc'], P['scale'])), 'Bernoulli (logits)': ('bernoulli_logits', dict(logits=rng.normal(size=N) * 2), lambda P: Bernoulli(logits=P['logits'])), 'Gamma': ('gamma', dict(concentration=rng.uniform(0.5, 5, N), rate=rng.uniform(0.5, 2, N)), lambda P: Gamma(P['concentration'], P['rate'])), 'Beta': ('beta', dict(concentration1=rng.uniform(0.5, 5, N), concentration0=rng.uniform(0.5, 5, N)), lambda P: Beta(P['concentration1'], P['concentration0']))}
for name, (path, Pn, mk) in cases.items():
    try:
        M = importlib.import_module('stablekl._closed.' + path)
        Qn = {k: v * (1 + 0.001 * rng.normal(size=N)) for k, v in Pn.items()}

        def off(dt='fp32'):
            P = {k: to_t(v, 'fp32').to(TD[dt]).requires_grad_(True) for k, v in Pn.items()}
            Q = {k: to_t(v, 'fp32').to(TD[dt]).requires_grad_(True) for k, v in Qn.items()}
            kl_divergence(mk(P), mk(Q)).sum().backward()
        a = bench(off, 10)
        a64 = bench(lambda: off('fp64'), 10)
        c = bench(lambda: M.run_rewrite_torch(TD, to_t, 'fp32', 'fp32', Pn, Qn, NPF), 3)
        d = bench(lambda: M.run_rewrite_torch(TD, to_t, 'fp32', 'fp64', Pn, Qn, NPF), 3)
        R['closed'].append(dict(name=name, n=N, official_ms=a, official_fp64_ms=a64, rewrite_ms=c, ours_ms=d))
        print(f'{name:20s} official {a:.2f} ms | FP64 {a64 / a:.1f}x | Rewrite {c / a:.1f}x | Ours {d / a:.1f}x', flush=True)
    except Exception as e:
        print(f'{name}: skipped ({type(e).__name__}: {str(e)[:140]})', flush=True)
from stablekl import interface_b as K
torch.manual_seed(0)
B, V = (64, 1024)
nb = K._near_boundary
for dname, dt in (('fp32', torch.float32), ('fp16', torch.float16)):
    for lt in (False, True):
        x0 = torch.log_softmax(torch.randn(B, V) * 3, -1).to(dt)
        t0 = (torch.log_softmax(torch.randn(B, V) * 3, -1) if lt else torch.softmax(torch.randn(B, V) * 3, -1)).to(dt)

        def off(d=dt):
            x = x0.to(d).clone().requires_grad_(True)
            F.kl_div(x, t0.to(d), reduction='batchmean', log_target=lt).backward()

        def fix():
            x = x0.clone().requires_grad_(True)
            K.kl_div(x, t0, reduction='batchmean', log_target=lt).backward()
        a = bench(off, 30)
        a64 = bench(lambda: off(torch.float64), 30)
        c = bench(fix, 5)
        K._near_boundary = lambda v, bound, d: torch.zeros(v.shape, dtype=torch.bool)
        u = bench(fix, 5)
        K._near_boundary = nb
        R['kldiv'].append(dict(dtype=dname, log_target=lt, B=B, V=V, official_ms=a, official_fp64_ms=a64, repaired_ms=c, repaired_nocheck_ms=u))
        print(f'kl_div {dname} log_target={lt}: official {a:.3f} ms | FP64 {a64 / a:.1f}x | repaired {c / a:.0f}x | check skipped {u / a:.0f}x', flush=True)
from stablekl.logits import logits_kl
for dname, dt in (('fp32', torch.float32), ('bf16', torch.bfloat16)):
    for B, V in ((64, 1024), (64, 32768), (32, 151936)):
        t = (torch.randn(B, V) * 3).to(dt)
        s0 = (t.float() + 0.01 * torch.randn(B, V)).to(dt)

        def off(d=None):
            s = s0.clone().requires_grad_(True)
            ss, tt = (s, t) if d is None else (s.to(d), t.to(d))
            F.kl_div(F.log_softmax(ss, -1), F.log_softmax(tt, -1), reduction='batchmean', log_target=True).backward()

        def ours():
            s = s0.clone().requires_grad_(True)
            logits_kl(t, s).backward()
        n = 20 if V == 1024 else 4
        a = bench(off, n)
        a64 = bench(lambda: off(torch.float64), n)
        b = bench(ours, n)
        R['logits'].append(dict(dtype=dname, B=B, V=V, official_ms=a, official_fp64_ms=a64, logits_path_ms=b))
        print(f'logits {dname} {B}x{V}: official {a:.2f} ms | FP64 {a64 / a:.1f}x | logits path {b / a:.1f}x', flush=True)
json.dump(R, open('cost_bench.json', 'w'), indent=1)
print('saved cost_bench.json')
