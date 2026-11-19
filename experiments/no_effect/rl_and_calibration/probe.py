from __future__ import annotations
import argparse, glob, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from softmax_fix import fixed_log_softmax
from stablekl.logits import logits_kl
DT = {'fp32': torch.float32, 'bf16': torch.bfloat16}
_PSI = [1.0 / math.factorial(k) for k in range(2, 24)]
_PHI = [(k + 1) / math.factorial(k + 2) for k in range(18)]

def psi(x):
    small = x.abs() < 0.5
    xs = torch.where(small, x, torch.zeros_like(x))
    acc = torch.full_like(x, _PSI[-1])
    for c in reversed(_PSI[:-1]):
        acc = acc * xs + c
    return torch.where(small, xs * xs * acc, torch.expm1(x) - x)

def phi64(w):
    small = w.abs() < 0.1
    ws = torch.where(small, w, torch.zeros_like(w))
    acc = torch.full_like(w, _PHI[-1])
    for c in reversed(_PHI[:-1]):
        acc = acc * ws + c
    big = torch.where(torch.isneginf(w), torch.ones_like(w), w * torch.exp(w) - torch.expm1(w))
    return torch.where(small, ws * ws * acc, big)

def exact(zp, zr, a):
    zp, zr = (zp.double(), zr.double())
    u = zp - zr
    p = torch.softmax(zr, -1)
    Delta = torch.log1p((p * torch.expm1(u)).sum(-1, keepdim=True))
    s = u - Delta
    q = p * torch.exp(s)
    r = s.gather(-1, a[:, None]).squeeze(-1)
    oh = torch.zeros_like(q).scatter(-1, a[:, None], 1.0)
    dr = oh - q
    k3 = psi(-r)
    g_k1 = dr
    g_k3 = (-torch.expm1(-r))[:, None] * dr
    kl_f = (q * phi64(-s)).sum(-1)
    g_f = p * torch.expm1(s)
    kl_r = (p * phi64(s)).sum(-1)
    g_r = q * (s - kl_r[:, None])
    return dict(k1=(r, g_k1), k3=(k3, g_k3), fwd=(kl_f, g_f), rev=(kl_r, g_r))

def lsm(z, which):
    return fixed_log_softmax(z) if which == 'fixed' else torch.log_softmax(z, -1)

def estimators(zp, zr, a, P, which):
    out = {}
    for name in ('k1', 'k3_official', 'k3_fixed'):
        x = zp.to(DT[P]).clone().requires_grad_(True)
        lp = lsm(x, which).gather(-1, a[:, None]).squeeze(-1)
        lr = lsm(zr.to(DT[P]), which).gather(-1, a[:, None]).squeeze(-1)
        d = lr - lp
        v = lp - lr if name == 'k1' else torch.exp(d) - d - 1 if name == 'k3_official' else psi(d)
        g, = torch.autograd.grad(v.sum(), x)
        dd = lr.detach().double() - lp.detach().double()
        op = -dd if name == 'k1' else psi(dd)
        out[name] = (v.detach(), g, op)
    return out

def full_kl(zp, zr, P, which, kldivfix):
    out = {}
    for name in ('fwd_lt_official', 'fwd_lt_fix', 'fwd_prob_official', 'fwd_prob_fix', 'rev_lt_official', 'rev_lt_fix', 'fwd_logits'):
        fix = name.endswith('_fix')
        if fix and (kldivfix is None or P == 'bf16'):
            continue
        vals, grads, ops = ([], [], [])
        for i in range(zp.shape[0]):
            x = zp[i:i + 1].to(DT[P]).clone().requires_grad_(True)
            zri = zr[i:i + 1].to(DT[P])
            kl = kldivfix.kl_div if fix else F.kl_div
            kw = dict(method='D64') if fix else {}
            if name.startswith('fwd_lt'):
                v = kl(lsm(x, which), lsm(zri, which), reduction='sum', log_target=True, **kw)
            elif name.startswith('fwd_prob'):
                v = kl(lsm(x, which), lsm(zri, which).exp(), reduction='sum', **kw)
            elif name.startswith('rev_lt'):
                v = kl(lsm(zri, which), lsm(x, which), reduction='sum', log_target=True, **kw)
            else:
                v = logits_kl(zri, x)
            g, = torch.autograd.grad(v, x)
            vals.append(v.detach().reshape(()))
            grads.append(g)
            with torch.no_grad():
                xs, zs_ = (lsm(x.detach(), which).double(), lsm(zri, which))
                if name.startswith('fwd_lt'):
                    t = zs_.double()
                    ops.append((t.exp() * (t - xs)).sum())
                elif name.startswith('fwd_prob'):
                    t = zs_.exp().double()
                    ops.append(torch.xlogy(t, t).sum() - (t * xs).sum())
                elif name.startswith('rev_lt'):
                    xr, t = (zs_.double(), xs)
                    ops.append((t.exp() * (t - xr)).sum())
                else:
                    ops.append(torch.tensor(float('nan'), dtype=torch.float64, device=x.device))
        out[name] = (torch.stack(vals), torch.cat(grads), torch.stack(ops))
    return out

def rel(v, ref):
    v, ref = (v.double(), ref.double())
    d = (v - ref).abs()
    m = ref.abs()
    return torch.where(m > 0, d / m, torch.where(d == 0, torch.zeros_like(d), torch.full_like(d, float('inf'))))

class Acc:

    def __init__(self):
        self.vr, self.vo, self.gr = ([], [], [])
        self.vsum = self.rsum = self.dot = self.n2 = self.r2 = 0.0
        self.n = self.v_nonfinite = self.g_nonfinite = 0
        self.gmax_at_zero = 0.0
        self.zero_ref_tokens = 0

    def add(self, v, g, rv, rg, op=None):
        vfin = torch.isfinite(v)
        gfin = torch.isfinite(g).all(-1)
        self.v_nonfinite += int((~vfin).sum())
        self.g_nonfinite += int((~gfin).sum())
        self.n += v.numel()
        e = rel(v, rv)
        e = torch.where(vfin, e, torch.full_like(e, float('inf')))
        self.vr.append(e.cpu())
        if op is not None and (not torch.isnan(op).all()):
            eo = rel(v, op)
            self.vo.append(torch.where(vfin, eo, torch.full_like(eo, float('inf'))).cpu())
        self.rsum += float(rv.double().sum())
        self.vsum += float(v.double().sum())
        gd, rgd = (g.double(), rg.double())
        rn = rgd.norm(dim=-1)
        zero = rn == 0
        self.zero_ref_tokens += int(zero.sum())
        if (zero & gfin).any():
            self.gmax_at_zero = max(self.gmax_at_zero, float(gd[zero & gfin].abs().max()))
        if (zero & ~gfin).any():
            self.gmax_at_zero = float('inf')
        nz = ~zero
        if nz.any():
            ge = (gd[nz] - rgd[nz]).norm(dim=-1) / rn[nz]
            self.gr.append(torch.where(gfin[nz], ge, torch.full_like(ge, float('inf'))).cpu())
        self.dot += float((gd * rgd).sum())
        self.n2 += float((gd ** 2).sum())
        self.r2 += float((rgd ** 2).sum())

    def result(self):
        cat = lambda L: torch.cat(L).numpy() if L else np.array([])
        vr, vo, gr = (cat(self.vr), cat(self.vo), cat(self.gr))
        q = lambda x, p: float(np.percentile(x, p, method='higher')) if x.size else None
        share = lambda x, t: float(np.mean(x >= t)) if x.size else None
        if self.v_nonfinite or not math.isfinite(self.vsum):
            mean_rel = None
        elif self.rsum:
            mean_rel = abs(self.vsum / self.rsum - 1)
        else:
            mean_rel = 0.0 if self.vsum == 0 else float('inf')
        cos_ok = self.g_nonfinite == 0 and math.isfinite(self.dot) and (self.n2 > 0) and (self.r2 > 0)
        return dict(tokens=self.n, value_nonfinite=self.v_nonfinite, grad_nonfinite=self.g_nonfinite, ref_mean=self.rsum / max(self.n, 1), mean_rel=mean_rel, value_rel_median=q(vr, 50), value_rel_p95=q(vr, 95), value_ge_1pct=share(vr, 0.01), value_ge_10pct=share(vr, 0.1), op_rel_median=q(vo, 50), op_rel_p95=q(vo, 95), op_ge_1pct=share(vo, 0.01), grad_rel_median=q(gr, 50), grad_rel_p95=q(gr, 95), grad_cosine=self.dot / math.sqrt(self.n2 * self.r2) if cos_ok else None, zero_ref_tokens=self.zero_ref_tokens, grad_max_abs_where_exact_zero=self.gmax_at_zero if self.zero_ref_tokens else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='local model directory (HF snapshot)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--text-file', default=None)
    ap.add_argument('--windows', type=int, default=8)
    ap.add_argument('--window-len', type=int, default=512)
    ap.add_argument('--full-windows', type=int, default=2, help='windows used for the (per-row, slower) full-distribution KL')
    ap.add_argument('--sigmas', default='0,1e-3,1e-2,3e-2')
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--no-fix', action='store_true')
    ap.add_argument('--dtype', default='bfloat16')
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kldivfix = None
    if not a.no_fix:
        from stablekl import interface_b as kldivfix
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.text_file:
        text = open(a.text_file, encoding='utf-8').read()
    else:
        from datasets import load_dataset
        ds = None
        for name in ('Salesforce/wikitext', 'wikitext'):
            try:
                ds = load_dataset(name, 'wikitext-2-raw-v1', split='test')
                break
            except Exception as e:
                last = e
        if ds is None:
            raise SystemExit(f'WikiText-2 not available offline: {last}')
        text = '\n\n'.join(ds['text'])
    ids = tok(text, return_tensors=None)['input_ids']
    W, L = (a.windows, a.window_len)
    if len(ids) < W * L:
        raise SystemExit('text too short')
    windows = torch.tensor([ids[i * L:(i + 1) * L] for i in range(W)])
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype), local_files_only=True).to(dev).eval()

    @torch.no_grad()
    def logits():
        return torch.stack([model(windows[w:w + 1].to(dev)).logits[0].cpu() for w in range(W)])
    Zr = logits()
    V = Zr.shape[-1]

    def sample(Z, seed):
        g = torch.Generator(device='cpu').manual_seed(seed)
        return torch.stack([torch.multinomial(torch.softmax(Z[w].float(), -1), 1, generator=g).squeeze(-1) for w in range(W)])
    A_ref = sample(Zr, 0)
    lin = [m for n, m in model.named_modules() if n.startswith('model.layers.') and isinstance(m, torch.nn.Linear)]
    orig = [m.weight.data.clone() for m in lin]
    meta = dict(model=a.model, dtype=a.dtype, torch=torch.__version__, device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', tokens_sha256=hashlib.sha256(windows.numpy().tobytes()).hexdigest(), vocab=V, windows=W, window_len=L, full_windows=a.full_windows, kldivfix=None if kldivfix is None else kldivfix.__version__, pairs={})
    res = {}
    for sig in [float(x) for x in a.sigmas.split(',')]:
        gn = torch.Generator(device='cpu').manual_seed(123)
        with torch.no_grad():
            for m, w0 in zip(lin, orig):
                m.weight.data = w0 if sig == 0 else (w0.float() + sig * w0.float().abs() * torch.randn(w0.shape, generator=gn).to(w0.device)).to(w0.dtype)
        Zp = logits()
        for m, w0 in zip(lin, orig):
            m.weight.data = w0
        A = sample(Zp, 1)
        info = dict(identical_to_reference=bool(torch.equal(Zp, Zr)))
        accs = {}
        for w in range(W):
            for s0 in range(0, L, a.chunk):
                zp, zr, act = (Zp[w, s0:s0 + a.chunk].to(dev), Zr[w, s0:s0 + a.chunk].to(dev), A[w, s0:s0 + a.chunk].to(dev))
                ex = exact(zp, zr, act)
                act_r = A_ref[w, s0:s0 + a.chunk].to(dev)
                exr = exact(zp, zr, act_r)
                for name, (v, gr, op) in estimators(zp, zr, act_r, 'fp32', 'fixed').items():
                    accs.setdefault(('fixed', 'fp32', name + '@ref_actions'), Acc()).add(v, gr, *exr['k1' if name == 'k1' else 'k3'], op=op)
                for which in ('fixed', 'torch'):
                    for P in ('fp32', 'bf16'):
                        for name, (v, gr, op) in estimators(zp, zr, act, P, which).items():
                            key = 'k1' if name == 'k1' else 'k3'
                            accs.setdefault((which, P, name), Acc()).add(v, gr, *ex[key], op=op)
                        if w < a.full_windows and which == 'fixed':
                            for name, (v, gr, op) in full_kl(zp, zr, P, which, kldivfix).items():
                                key = 'fwd' if name.startswith('fwd') else 'rev'
                                accs.setdefault((which, P, name), Acc()).add(v, gr, *ex[key], op=op)
        res[str(sig)] = {f'{which}|{P}|{name}': acc.result() for (which, P, name), acc in accs.items()}
        meta['pairs'][str(sig)] = info
        info.update(mean_k3_exact=res[str(sig)]['fixed|fp32|k3_official']['ref_mean'], mean_forward_kl_exact=res[str(sig)].get('fixed|fp32|fwd_lt_official', {}).get('ref_mean'))
        print(f"[sigma {sig:g}] identical logits: {info['identical_to_reference']}; mean k3 (exact) {res[str(sig)]['fixed|fp32|k3_official']['ref_mean']:.3e}; mean forward KL (exact) {res[str(sig)].get('fixed|fp32|fwd_lt_official', {}).get('ref_mean', float('nan')):.3e}; {time.time() - t0:.0f} s", flush=True)
    meta['seconds'] = round(time.time() - t0, 1)
    json.dump(dict(meta=meta, results=res), open(out / 'probe_results.json', 'w'), indent=1)
    write_md(out, meta, res)
    print('done:', out / 'PROBE_SUMMARY.md', flush=True)

def write_md(out, meta, res):
    f = lambda x: '—' if x is None else f'{x:.1e}' if isinstance(x, float) else str(x)
    fm = lambda x: 'invalid (non-finite value)' if x is None else f'{x:.1e}'
    pc = lambda x: '—' if x is None else f'{100 * x:.1f}%'
    L = ['# Numerical probe on real LLM logits of the KL used in RL / distillation (not an RLHF training reproduction)', '', "Main rows: actions sampled from each policy; rows marked '@ref_actions': actions sampled from the reference (ablation).  A non-finite method value counts as an error >= 100% and makes the batch-mean error 'invalid'.", '', f"Model {meta['model']} ({meta['dtype']}), vocab {meta['vocab']}, {meta['windows']} x {meta['window_len']} tokens (full-distribution KL on {meta['full_windows']} windows), device {meta['device']}, kldivfix {meta['kldivfix']}.  Reference: FP64 from the same logits, cancellation-free.  'fixed' = Barron et al. fixed log-softmax; 'torch' = torch.log_softmax.", '']
    for sig, r in res.items():
        L += [f"## sigma = {sig} (logits identical to the reference: {meta['pairs'][sig]['identical_to_reference']})", '', '| softmax | logprob dtype | method | exact mean | batch-mean rel. err | value err (total) median / p95 | total >= 1% / >= 10% | operator err median / p95, >= 1% | grad rel. err median / p95 | grad cosine | max abs grad where exact grad = 0 | non-finite value / grad |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for k in sorted(r):
            x = r[k]
            which, P, name = k.split('|')
            cos = '—' if x['grad_cosine'] is None else f"{x['grad_cosine']:.6f}"
            L.append(f"| {which} | {P} | {name} | {f(x['ref_mean'])} | {fm(x['mean_rel'])} | {f(x['value_rel_median'])} / {f(x['value_rel_p95'])} | {pc(x['value_ge_1pct'])} / {pc(x['value_ge_10pct'])} | {f(x['op_rel_median'])} / {f(x['op_rel_p95'])}, {pc(x['op_ge_1pct'])} | {f(x['grad_rel_median'])} / {f(x['grad_rel_p95'])} | {cos} | {f(x['grad_max_abs_where_exact_zero'])} | {x['value_nonfinite']} / {x['grad_nonfinite']} |")
        L.append('')
    (out / 'PROBE_SUMMARY.md').write_text('\n'.join(L), encoding='utf-8')
if __name__ == '__main__':
    main()
