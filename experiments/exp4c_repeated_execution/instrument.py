from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'no_effect' / 'llm_distillation'))
from fingerprint import code_md5
import argparse, hashlib, json, math, os, time
import torch, torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
import gkd as P
from stablekl.logits import _phi, _round_once
from rtn import quantise_decoder_linears
METHODS = ('kl_div_prob', 'kl_div_logt', 'categorical')
REDS = ('batchmean', 'sum', 'mean', 'none', 'none_float')
CAT_REDS = ('batchmean', 'none_float')
DT = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32, 'fp64': torch.float64}
_PSI = [1.0 / math.factorial(k + 2) for k in range(18)]
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = ('instrument.py', 'gkd.py', 'logits_kl.py', 'rtn.py')

def official(method, p, q, red):
    n, V = q.shape
    if method == 'categorical':
        try:
            k = kl_divergence(Categorical(logits=p), Categorical(logits=q))
        except ValueError:
            return float('nan')
        return float(k.mean()) if red == 'batchmean' else float(k.float().mean())
    lq = F.log_softmax(q, -1)
    lt = method == 'kl_div_logt'
    tgt = F.log_softmax(p, -1) if lt else F.softmax(p, -1)
    if red == 'batchmean':
        return float(F.kl_div(lq, tgt, reduction='batchmean', log_target=lt))
    if red == 'sum':
        return float(F.kl_div(lq, tgt, reduction='sum', log_target=lt)) / n
    if red == 'mean':
        return float(F.kl_div(lq, tgt, reduction='mean', log_target=lt)) * V
    e = F.kl_div(lq, tgt, reduction='none', log_target=lt)
    return float(e.sum(-1).mean()) if red == 'none' else float(e.float().sum(-1).mean())

def logits_path(p, q, sub=128):
    rows = []
    for r in range(0, p.shape[0], sub):
        A, C = (p[r:r + sub].double(), q[r:r + sub].double())
        u = A - C
        qq = torch.softmax(C, -1)
        w = u - torch.log1p((qq * torch.expm1(u)).sum(-1, keepdim=True))
        rows.append((qq * _phi(w)).sum(-1))
    return float(_round_once(torch.cat(rows).mean(), p.dtype))

def finite_rows(tensors, sub=512):
    n = tensors[0].shape[0]
    good = torch.ones(n, dtype=torch.bool, device=tensors[0].device)
    bad = []
    for t in tensors:
        ok = torch.cat([torch.isfinite(t[r:r + sub]).all(-1) for r in range(0, n, sub)])
        bad.append(int((~ok).sum()))
        good &= ok
    return (good, bad)

def exact_mean(Pm, Qm, dev, sub=128):
    tot = 0.0
    for r in range(0, Pm.shape[0], sub):
        tot += float(P.exact_kl64(Pm[r:r + sub].to(dev), Qm[r:r + sub].to(dev)).sum())
    return tot / Pm.shape[0]

def run_setting(fn, Pm, Qm, chunk, idx, cdt, dev):
    n = Pm.shape[0]
    tot = 0.0
    for r in range(0, n, chunk):
        ii = slice(r, min(r + chunk, n)) if idx is None else idx[r:r + chunk]
        p = Pm[ii].to(dev, cdt)
        q = Qm[ii].to(dev, cdt)
        v = fn(p, q)
        if not math.isfinite(v):
            return None
        tot += v * p.shape[0]
    return tot / n

def psi(x):
    small = x.abs() < 0.5
    xs = torch.where(small, x, torch.zeros_like(x))
    acc = torch.full_like(x, _PSI[-1])
    for c in reversed(_PSI[:-1]):
        acc = acc * xs + c
    return torch.where(small, xs * xs * acc, torch.expm1(x) - x)

def estimators(Pm, Qm, dev, seed, sub=128):
    g = torch.Generator(device=dev).manual_seed(seed)
    lp32, lq32, lp64, lq64 = ([], [], [], [])
    for r in range(0, Pm.shape[0], sub):
        p, q = (Pm[r:r + sub].to(dev), Qm[r:r + sub].to(dev))
        a = torch.multinomial(torch.softmax(p.double(), -1), 1, generator=g)
        lp32.append(F.log_softmax(p.float(), -1).gather(-1, a))
        lq32.append(F.log_softmax(q.float(), -1).gather(-1, a))
        lp64.append(F.log_softmax(p.double(), -1).gather(-1, a))
        lq64.append(F.log_softmax(q.double(), -1).gather(-1, a))
    lp32, lq32, lp64, lq64 = (torch.cat(v).flatten() for v in (lp32, lq32, lp64, lq64))
    x32, x64 = (lq32 - lp32, lq64 - lp64)
    return dict(k1_fp32=float((lp32 - lq32).mean()), k1_fp64=float((lp64 - lq64).mean()), k3_official_fp32=float((torch.exp(x32) - x32 - 1).mean()), k3_fixed_fp32=float(psi(x32).mean()), k3_fp64=float(psi(x64).mean()), abs_log_ratio_median=float(x64.abs().median()), abs_log_ratio_max=float(x64.abs().max()), seq_log_ratio_note='x = lp_train - lp_rollout per sampled token')

@torch.no_grad()
def forward_full(model, T, bs, dev):
    return torch.cat([model(input_ids=T[b:b + bs].to(dev)).logits[:, :-1].reshape(-1, model.config.vocab_size) for b in range(0, len(T), bs)])

@torch.no_grad()
def forward_cached(model, T, bs, dev):
    out = []
    for b in range(0, len(T), bs):
        ids = T[b:b + bs].to(dev)
        pkv = None
        steps = []
        for t in range(ids.shape[1] - 1):
            o = model(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            steps.append(o.logits[:, 0])
        out.append(torch.stack(steps, 1).reshape(-1, model.config.vocab_size))
    return torch.cat(out)

def load(path, dt, dev, attn='sdpa'):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt, attn_implementation=attn, local_files_only=True).to(dev).eval()

def part_R(a, T, dev, out):
    fdt = DT[a.fwd]
    names = a.variants.split(',')
    cc = lambda: torch.cuda.empty_cache() if dev.type == 'cuda' else None
    base = load(a.model, fdt, dev)
    L = [forward_full(base, T, a.bs, dev)]
    del base
    cc()
    for name in names:
        bits, grp = (int(x) for x in name[1:].split('g')) if 'g' in name else (int(name[1:]), None)
        m = load(a.model, fdt, dev)
        quantise_decoder_linears(m, bits, grp)
        L.append(forward_full(m, T, a.bs, dev))
        del m
        cc()
    good, bad = finite_rows(L)
    rows_all = L[0].shape[0]
    L = [t[good] for t in L]
    cc()
    N = L[0].shape[0]
    print(f'R {a.fwd}: {rows_all} positions, non-finite per tensor (base, {a.variants}): {bad}, kept {N}', flush=True)
    if N < a.min_rows:
        raise SystemExit(f'fewer than {a.min_rows} finite positions')
    Pm = L[0]
    gen = torch.Generator().manual_seed(0)
    perms = [torch.randperm(N, generator=gen).to(dev) for _ in range(2)]
    chunks = sorted({c for c in (1, 16, 128, 512, N) if c <= N})
    big = [c for c in chunks if c >= 128]
    for name, Qm in zip(names, L[1:]):
        ref = exact_mean(Pm, Qm, dev)
        t0 = time.time()
        rec = dict(part='R', fwd=a.fwd, variant=name, rows=N, rows_before_filter=rows_all, nonfinite_rows=dict(zip(['base'] + names, bad)), ref=ref, top1_disagree=float((Pm.argmax(-1) != Qm.argmax(-1)).float().mean()), reported={})
        for cname in (a.fwd, 'fp32'):
            cdt = DT[cname]
            for meth in METHODS + ('logits_path',):
                vals = {}
                reds = ('-',) if meth == 'logits_path' else CAT_REDS if meth == 'categorical' else REDS
                fn_of = lambda red: logits_path if meth == 'logits_path' else lambda p, q: official(meth, p, q, red)
                for c in chunks:
                    for red in reds:
                        vals[f'gpu|chunk{c}|{red}|order0'] = run_setting(fn_of(red), Pm, Qm, c, None, cdt, dev)
                for c in big:
                    for k, pm in enumerate(perms):
                        vals[f'gpu|chunk{c}|{reds[0]}|order{k + 1}'] = run_setting(fn_of(reds[0]), Pm, Qm, c, pm, cdt, dev)
                if a.cpu and 128 in chunks:
                    Pc, Qc = (Pm.cpu(), Qm.cpu())
                    vals[f'cpu|chunk128|{reds[0]}|order0'] = run_setting(fn_of(reds[0]), Pc, Qc, 128, None, cdt, torch.device('cpu'))
                    del Pc, Qc
                rec['reported'][f'{cname}|{meth}'] = vals
        rec['seconds'] = time.time() - t0
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f"R {a.fwd} {name}: ref {ref:.4e}, top-1 {rec['top1_disagree']:.4f}, {rec['seconds']:.0f} s", flush=True)

def part_M(a, T, dev, out):
    L = {}
    for d in a.dtypes.split(','):
        m = load(a.model, DT[d], dev)
        L[f'{d}|full_b{a.bs}'] = forward_full(m, T, a.bs, dev)
        L[f'{d}|full_b1'] = forward_full(m, T, 1, dev)
        L[f'{d}|cached_b{a.bs}'] = forward_cached(m, T, a.bs, dev)
        del m
        m = load(a.model, DT[d], dev, 'eager')
        L[f'{d}|eager_b{a.bs}'] = forward_full(m, T, a.bs, dev)
        del m
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'M forward {d}: done', flush=True)
    keys = list(L)
    nonfinite = dict(zip(keys, finite_rows([L[k] for k in keys])[1]))
    rows_all = L[keys[0]].shape[0]
    print(f'M: {rows_all} positions, non-finite per configuration: {nonfinite}', flush=True)
    ds = a.dtypes.split(',')
    A = lambda d: f'{d}|full_b{a.bs}'
    pairs = [(f'{d}|cached_b{a.bs}', A(d)) for d in ds] + [(f'{d}|full_b1', A(d)) for d in ds] + [(f'{d}|eager_b{a.bs}', A(d)) for d in ds]
    pairs += [(A(x), A(y)) for x, y in (('fp32', 'fp16'), ('fp32', 'bf16'), ('fp16', 'bf16')) if x in ds and y in ds]
    for pn, qn in pairs:
        good, _ = finite_rows([L[pn], L[qn]])
        kept = int(good.sum())
        if kept < a.min_rows:
            rec = dict(part='M', P=pn, Q=qn, skipped=f'only {kept} of {rows_all} positions are finite in both', nonfinite_rows=nonfinite)
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"M {pn} vs {qn}: SKIPPED ({rec['skipped']})", flush=True)
            continue
        Pm, Qm = (L[pn][good], L[qn][good])
        same = Pm.dtype == Qm.dtype
        ref = exact_mean(Pm, Qm, dev)
        rec = dict(part='M', P=pn, Q=qn, rows=kept, rows_before_filter=rows_all, nonfinite_rows=nonfinite, ref=ref, max_abs_logit_diff=float((Pm.double() - Qm.double()).abs().max()), rows_identical=float((Pm == Qm).all(-1).float().mean()) if same else None, top1_disagree=float((Pm.argmax(-1) != Qm.argmax(-1)).float().mean()), reported={}, estimators=estimators(Pm, Qm, dev, 0))
        cds = ([next((k for k, v in DT.items() if v == Pm.dtype))] if same and Pm.dtype != torch.float32 else []) + ['fp32', 'fp64']
        for cname in cds:
            for meth in METHODS:
                rec['reported'][f'{cname}|{meth}'] = run_setting(lambda p, q: official(meth, p, q, 'batchmean'), Pm, Qm, 128, None, DT[cname], dev)
            if cname != 'fp64':
                rec['reported'][f'{cname}|logits_path'] = run_setting(logits_path, Pm, Qm, 128, None, DT[cname], dev)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f"M {pn} vs {qn}: {kept} rows, ref {ref:.4e}, max |dz| {rec['max_abs_logit_diff']:.3e}, " + ', '.join((f'{k} {v:.3e}' if v is not None else f'{k} nonfinite' for k, v in rec['reported'].items() if k.startswith('fp32'))), flush=True)
        del Pm, Qm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=('R', 'M'), required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fwd', default='fp16')
    ap.add_argument('--variants', default='w8,w7,w6,w5,w4g128')
    ap.add_argument('--dtypes', default='fp32,fp16,bf16')
    ap.add_argument('--n-seq', type=int, default=8)
    ap.add_argument('--seq', type=int, default=513)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--text-file', default=None)
    ap.add_argument('--cpu', type=int, default=1)
    ap.add_argument('--min-rows', type=int, default=256)
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} exists: refusing to append to an earlier run')
    from transformers import AutoTokenizer
    T, _ = P.load_tokens(AutoTokenizer.from_pretrained(a.model), a.text_file, a.n_seq, 1, a.seq)
    meta = dict(code={f: code_md5(f, HERE) for f in CODE}, args=vars(a), torch=torch.__version__, transformers=__import__('transformers').__version__, device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', tokens_sha256=hashlib.sha256(T.numpy().tobytes()).hexdigest())
    json.dump(meta, open(a.out + '.meta.json', 'w'), indent=1)
    print(meta, flush=True)
    (part_R if a.part == 'R' else part_M)(a, T, dev, a.out)
    print('done', flush=True)
if __name__ == '__main__':
    main()
