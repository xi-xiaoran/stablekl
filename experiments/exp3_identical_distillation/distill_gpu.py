import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from fingerprint import code_md5
import argparse, copy, hashlib, json, math, os, sys, time, warnings
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stablekl.logits import logits_kl
warnings.simplefilter('ignore')
try:
    from stablekl import interface_b as KF
except Exception as e:
    KF = None
    KF_ERR = f'{type(e).__name__}: {e}'
DT = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}

def mlp(V, seed):
    g = torch.Generator().manual_seed(seed)
    m = nn.Sequential(nn.Linear(32, 256), nn.GELU(), nn.Linear(256, V))
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (1 / math.sqrt(p.shape[-1]) if p.dim() > 1 else 0.1))
        m[2].weight.mul_(3.0)
    return m

class OursGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, t):
        ctx.save_for_backward(x, t)
        return F.kl_div(x, t, reduction='batchmean')

    @staticmethod
    def backward(ctx, G):
        x, t = ctx.saved_tensors
        gx, _ = KF._KLGrad.apply(x, t, G.reshape(()), (False, 'D64', 'sum', x.numel(), x.shape[0]))
        return (gx, None)

def loss_fn(method, s, t, dt):
    sd, td = (s.to(dt), t.to(dt))
    if method == 'kl_div':
        return F.kl_div(sd.log_softmax(-1), td.softmax(-1), reduction='batchmean')
    if method == 'kl_div_lt':
        return F.kl_div(sd.log_softmax(-1), td.log_softmax(-1), reduction='batchmean', log_target=True)
    if method == 'categorical':
        return kl_divergence(Categorical(logits=td), Categorical(logits=sd)).mean()
    if method == 'upcast64':
        s64, t64 = (sd.double(), td.double())
        return F.kl_div(s64.log_softmax(-1), t64.softmax(-1), reduction='batchmean').to(dt)
    if method == 'ours_kldiv_grad':
        return OursGrad.apply(sd.log_softmax(-1), td.softmax(-1))
    if method == 'ours_kldiv':
        return KF.kl_div(sd.log_softmax(-1), td.softmax(-1), reduction='batchmean', method='D64')
    if method == 'logitsA':
        return logits_kl(td, sd)
    raise ValueError(method)

def true_kl(t, s):
    lt, ls = (t.double().log_softmax(-1), s.double().log_softmax(-1))
    return (lt.exp() * (lt - ls)).sum(-1).mean().item()

def run(dev, V, dtn, method, opt_name, seed, steps=300, lr=None, kick=0.0, log_at=(1, 10, 30, 50, 100, 300)):
    dt = DT[dtn]
    teacher = mlp(V, seed)
    student = copy.deepcopy(teacher)
    if kick:
        g = torch.Generator().manual_seed(500 + seed)
        with torch.no_grad():
            for p in student.parameters():
                p.mul_(1 + kick * torch.randn(p.shape, generator=g))
    teacher, student = (teacher.to(dev), student.to(dev))
    teacher.requires_grad_(False)
    W0 = torch.cat([p.detach().flatten() for p in teacher.parameters()])
    nW0 = W0.norm().item()
    lr = lr if lr is not None else 0.001 if opt_name == 'adam' else 0.1
    opt = torch.optim.Adam(student.parameters(), lr=lr, eps=1e-08) if opt_name == 'adam' else torch.optim.SGD(student.parameters(), lr=lr)
    gx = torch.Generator().manual_seed(10000 + seed)
    xe = torch.randn(512, 32, generator=torch.Generator().manual_seed(99)).to(dev)
    with torch.no_grad():
        te = teacher(xe)
    out = dict(V=V, dt=dtn, method=method, opt=opt_name, seed=seed, lr=lr, kick=kick, steps=steps, log={})
    t0 = time.time()
    for s in range(1, steps + 1):
        x = torch.randn(64, 32, generator=gx).to(dev)
        with torch.no_grad():
            t = teacher(x)
        L = loss_fn(method, student(x), t, dt)
        opt.zero_grad()
        L.backward()
        if s == 1:
            out['loss0'] = L.item()
            out['gnorm0'] = math.sqrt(sum((float(p.grad.double().pow(2).sum()) for p in student.parameters())))
        if not all((torch.isfinite(p.grad).all() for p in student.parameters())):
            out['nonfinite_step'] = s
            break
        opt.step()
        out['steps_done'] = s
        if s in log_at:
            with torch.no_grad():
                W = torch.cat([p.detach().flatten() for p in student.parameters()])
                se = student(xe)
                out['log'][s] = dict(rel_dW=(W - W0).norm().item() / nW0, max_dW=(W - W0).abs().max().item(), true_kl=true_kl(te, se), top1_disagree=(se.argmax(-1) != te.argmax(-1)).float().mean().item())
    out.setdefault('steps_done', 0)
    out['status'] = 'complete' if out['steps_done'] == steps else 'nonfinite'
    h = hashlib.sha256()
    for p in student.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    out['W_final_sha256'] = h.hexdigest()
    out['seconds'] = round(time.time() - t0, 2)
    return out

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    raw = open(path).read()
    lines = raw.split('\n')
    rows = []
    if lines and lines[-1] == '':
        lines = lines[:-1]
    for i, l in enumerate(lines):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            print(f'{path}: dropped a truncated last line', flush=True)
            with open(path, 'w') as f:
                f.write(''.join((json.dumps(r) + '\n' for r in rows)))
            return rows
    if raw and (not raw.endswith('\n')):
        print(f'{path}: added a missing final newline', flush=True)
        with open(path, 'a') as f:
            f.write('\n')
    return rows

def fingerprint(a, dev):
    h = {}
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ('distill_gpu.py', 'logits_kl.py'):
        h[fn] = code_md5(fn, here)
    if KF is not None:
        h['kldivfix.py'] = hashlib.md5(open(KF.__file__, 'rb').read()).hexdigest()
    return dict(code=h, torch=torch.__version__, device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', smoke=a.smoke, no_fix=a.no_fix)

def grad0(dev, V, dtn, seed):
    dt = DT[dtn]
    m = mlp(V, seed).to(dev)
    gx = torch.Generator().manual_seed(10000 + seed)
    x = torch.randn(64, 32, generator=gx).to(dev)
    with torch.no_grad():
        z = m(x)
    res = dict(V=V, dt=dtn, seed=seed)
    G = {}
    for meth in ('kl_div', 'upcast64', 'logitsA') + (('ours_kldiv_grad',) if KF is not None and dtn != 'bf16' else ()):
        s = z.clone().to(dt).requires_grad_(True)
        t = z.to(dt)
        if meth == 'kl_div':
            L = F.kl_div(s.log_softmax(-1), t.softmax(-1), reduction='batchmean')
        elif meth == 'upcast64':
            L = F.kl_div(s.double().log_softmax(-1), t.double().softmax(-1), reduction='batchmean')
        elif meth == 'ours_kldiv_grad':
            L = OursGrad.apply(s.log_softmax(-1), t.softmax(-1))
        else:
            L = logits_kl(t, s)
        L.backward()
        g = s.grad.double()
        G[meth] = s.grad.clone()
        res[meth] = dict(loss=L.item(), nonzero_frac=float((g != 0).float().mean()), max_abs=float(g.abs().max()), norm=float(g.norm()))
    if 'ours_kldiv_grad' in G:
        res['ours_grad_equals_official'] = bool(torch.equal(G['ours_kldiv_grad'], G['kl_div']))
        res['ours_grad_equal_frac'] = float((G['ours_kldiv_grad'] == G['kl_div']).float().mean())
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--smoke', action='store_true', help='tiny grid for a local check')
    ap.add_argument('--no-fix', action='store_true', help='skip the interface-B fix (set by the job if its unit tests fail in this environment)')
    a = ap.parse_args()
    global KF, KF_ERR
    if a.no_fix:
        KF = None
        KF_ERR = 'disabled: kldivfix unit tests failed in this environment (see the job log)'
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', kldivfix=None if KF is None else getattr(KF, '__version__', '?'), kldivfix_error=None if KF is not None else KF_ERR)
    print(env, flush=True)
    fp = fingerprint(a, dev)
    fpath = os.path.join(a.out, 'fingerprint.json')
    if os.path.exists(fpath):
        old = json.load(open(fpath))
        if old != fp:
            sys.exit(f'RESUME REFUSED: {a.out} was written by a different code / environment / grid\n old {old}\n now {fp}')
        print('resume: fingerprint matches', flush=True)
    else:
        json.dump(fp, open(fpath, 'w'), indent=1)
    json.dump(env, open(os.path.join(a.out, 'env.json'), 'w'), indent=1)
    Vs = (16, 1024) if a.smoke else (1024, 32768, 151936)
    seeds = range(2) if a.smoke else range(10)
    steps = 20 if a.smoke else 300
    logs = (1, 10, 20) if a.smoke else (1, 10, 30, 50, 100, 300)
    have_fix = KF is not None
    gpath = os.path.join(a.out, 'grad0.jsonl')
    gdone = set()
    gdone = {(r['V'], r['dt'], r['seed']) for r in read_jsonl(gpath)}
    with open(gpath, 'a') as f:
        for V in Vs:
            for dtn in DT:
                for seed in seeds:
                    if (V, dtn, seed) in gdone:
                        continue
                    f.write(json.dumps(grad0(dev, V, dtn, seed)) + '\n')
                    f.flush()
    print('step-0 gradients done', flush=True)
    plan = []
    for V in Vs:
        for dtn in DT:
            meths = ['kl_div', 'kl_div_lt', 'categorical', 'upcast64', 'logitsA'] + (['ours_kldiv_grad'] if have_fix and dtn != 'bf16' else [])
            for meth in meths:
                for opt in ('adam', 'sgd'):
                    for seed in seeds:
                        plan.append(dict(V=V, dtn=dtn, method=meth, opt_name=opt, seed=seed, steps=steps, log_at=logs))
    big = Vs[-1]
    if not a.smoke:
        for lr in (0.0001, 0.01):
            for seed in seeds:
                plan.append(dict(V=big, dtn='fp32', method='kl_div', opt_name='adam', seed=seed, steps=steps, lr=lr, log_at=logs))
        for meth in ('kl_div', 'upcast64', 'logitsA'):
            for seed in seeds:
                plan.append(dict(V=big, dtn='fp32', method=meth, opt_name='adam', seed=seed, steps=steps, kick=1e-06, log_at=logs))
        for seed in seeds:
            plan.append(dict(V=big, dtn='fp32', method='upcast64', opt_name='adam', seed=seed, steps=1000, log_at=(1, 10, 30, 50, 100, 300, 500, 700, 1000)))
    if have_fix:
        for dtn in ('fp32', 'fp16'):
            for opt, ns in (('adam', 10), ('sgd', 3)):
                for seed in list(seeds)[:2 if a.smoke else ns]:
                    plan.append(dict(V=Vs[0], dtn=dtn, method='ours_kldiv', opt_name=opt, seed=seed, steps=steps, log_at=logs))
    rpath = os.path.join(a.out, 'runs.jsonl')
    key = lambda d: (d['V'], d.get('dtn', d.get('dt')), d['method'], d.get('opt_name', d.get('opt')), d['seed'], d.get('lr') or 0, d.get('kick', 0.0), d['steps'])
    done = set()
    if True:
        for r in read_jsonl(rpath):
            lr_default = 0.001 if r['opt'] == 'adam' else 0.1
            done.add((r['V'], r['dt'], r['method'], r['opt'], r['seed'], 0 if r['lr'] == lr_default else r['lr'], r['kick'], r['steps']))
    t0 = time.time()
    n = 0
    with open(rpath, 'a') as f:
        for p in plan:
            if key(p) in done:
                continue
            r = run(dev, **{k: v for k, v in p.items()})
            f.write(json.dumps(r) + '\n')
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f'{n} runs, {time.time() - t0:.0f} s', flush=True)
    rows = read_jsonl(rpath)
    st = {}
    for r in rows:
        st[r['status']] = st.get(r['status'], 0) + 1
    ok = len(rows) == len(plan)
    print(f'{len(rows)} / {len(plan)} planned runs recorded ({st}); {n} run in this job, {time.time() - t0:.0f} s', flush=True)
    json.dump(dict(planned=len(plan), recorded=len(rows), by_status=st, kldivfix=have_fix), open(os.path.join(a.out, 'status.json'), 'w'))
    if not ok:
        sys.exit(f'recorded runs ({len(rows)}) != planned ({len(plan)})')
if __name__ == '__main__':
    main()
