from __future__ import annotations
import argparse, gc, glob, hashlib, json, math, os, sys, time, warnings
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dequant import dequant_awq, dequant_gptq, rtn
from stablekl.logits import logits_kl, _round_once
DT = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

def snapshot(repo, hub):
    root = os.path.join(hub, 'models--' + repo.replace('/', '--'))
    ref = os.path.join(root, 'refs', 'main')
    if os.path.isfile(ref):
        d = os.path.join(root, 'snapshots', open(ref).read().strip())
        if os.path.isdir(d):
            return d
    d = sorted(glob.glob(os.path.join(root, 'snapshots', '*')))
    if len(d) != 1:
        raise SystemExit(f'cannot choose a snapshot of {repo} in {hub}: {d}')
    return d[0]

def load_sd(path):
    from safetensors.torch import load_file
    sd = {}
    for f in sorted(glob.glob(os.path.join(path, '*.safetensors'))):
        sd.update(load_file(f))
    return sd

def build(variant, base_path, hub, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16).to(device).eval()
    kind = variant['kind']
    info = dict(kind=kind)
    if kind == 'base':
        return (model, info)
    lin = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n != 'lm_head'}
    if kind == 'rtn':
        for n, m in lin.items():
            m.weight.data = rtn(m.weight.data, variant['bits'], variant.get('group')).to(device)
        info.update(bits=variant['bits'], group=variant.get('group'), layers=len(lin))
        return (model, info)
    qpath = snapshot(variant['repo'], hub)
    cfg = json.load(open(os.path.join(qpath, 'config.json')))['quantization_config']
    sd = load_sd(qpath)
    bits, group = (cfg['bits'], cfg['group_size'])
    done = 0
    copied = 0
    info.update(repo=variant['repo'], snapshot=qpath, quantization_config=cfg)
    if kind == 'gptq' and cfg.get('desc_act'):
        raise SystemExit('desc_act=True is not handled')
    for n, m in lin.items():
        if n + '.qweight' not in sd:
            raise SystemExit(f"{variant['repo']}: no packed weight for {n}")
        if kind == 'awq':
            W = dequant_awq(sd[n + '.qweight'], sd[n + '.qzeros'], sd[n + '.scales'], bits, group)
        else:
            W = dequant_gptq(sd[n + '.qweight'], sd[n + '.qzeros'], sd[n + '.scales'], sd.get(n + '.g_idx'), bits, group)
        if tuple(W.shape) != tuple(m.weight.shape):
            raise SystemExit(f'{n}: shape {tuple(W.shape)} vs {tuple(m.weight.shape)}')
        m.weight.data = W.to(device)
        done += 1
    own = model.state_dict()
    for k, v in sd.items():
        if k.endswith(('.qweight', '.qzeros', '.scales', '.g_idx')):
            continue
        if k in own:
            if own[k].shape != v.shape:
                raise SystemExit(f'{k}: shape mismatch')
            own[k].copy_(v.to(own[k].dtype))
            copied += 1
    info.update(dequantised_layers=done, copied_tensors=copied)
    return (model, info)

@torch.no_grad()
def forward_logits(model, windows, device):
    out, nll, n = ([], 0.0, 0)
    for w in windows:
        ids = torch.tensor(w, device=device).unsqueeze(0)
        lg = model(ids).logits[0]
        lp = lg.double().log_softmax(-1)
        nll -= lp[:-1].gather(1, ids[0, 1:, None]).sum().item()
        n += ids.shape[1] - 1
        out.append(lg.cpu())
    L = torch.stack(out)
    return (L, dict(ppl=math.exp(nll / n), nonfinite_logits=int((~torch.isfinite(L)).sum()), logits_dtype=str(L.dtype)))
_PHI_C = [(k + 1) / math.factorial(k + 2) for k in range(16)]

def _phi_ref(w):
    small = w.abs() < 0.1
    ws = torch.where(small, w, torch.zeros_like(w))
    acc = torch.full_like(w, _PHI_C[-1])
    for c in reversed(_PHI_C[:-1]):
        acc = acc * ws + c
    big = torch.where(torch.isneginf(w), torch.ones_like(w), w * torch.exp(w) - torch.expm1(w))
    return torch.where(small, ws * ws * acc, big)

def exact_kl(b, q):
    bd, qd = (b.double(), q.double())
    u = bd - qd
    sq = torch.softmax(qd, -1)
    w = u - torch.log1p((sq * torch.expm1(u)).sum(-1, keepdim=True))
    return (sq * _phi_ref(w)).sum(-1)

def naive_kl(b, q):
    lb, lq = (b.double().log_softmax(-1), q.double().log_softmax(-1))
    return (lb.exp() * (lb - lq)).sum(-1)
FIX_STATE = {'error': None, 'chunks_ok': 0, 'chunks_failed': 0}

def methods(b, q, kldivfix):
    r = {}
    T = b.shape[0]
    nanT = torch.full((T,), float('nan'), dtype=torch.float64, device=b.device)
    placeholder = set()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for dn, dt in DT.items():
            bb, qq = (b.to(dt), q.to(dt))
            lq, lb, pb = (qq.log_softmax(-1), bb.log_softmax(-1), bb.softmax(-1))
            g_kl = torch.xlogy(pb.double(), pb.double()).sum(-1) - (pb.double() * lq.double()).sum(-1)
            g_lt = (lb.double().exp() * (lb.double() - lq.double())).sum(-1)
            g_cat = exact_kl(bb, qq)
            cv = lambda f, **k: {red: f(reduction=red, **k).double().item() for red in ('batchmean', 'mean')}
            r[f'kl_div:{dn}'] = (F.kl_div(lq, pb, reduction='none').sum(-1).double(), cv(lambda **k: F.kl_div(lq, pb, **k)), g_kl)
            r[f'kl_div_lt:{dn}'] = (F.kl_div(lq, lb, reduction='none', log_target=True).sum(-1).double(), cv(lambda **k: F.kl_div(lq, lb, **k), log_target=True), g_lt)
            c = kl_divergence(Categorical(logits=bb), Categorical(logits=qq))
            r[f'categorical:{dn}'] = (c.double(), {'batchmean': c.mean().double().item()}, g_cat)
            per_l = torch.stack([logits_kl(bb[i:i + 1], qq[i:i + 1]) for i in range(T)]).double()
            r[f'logitsA:{dn}'] = (per_l, {'batchmean': logits_kl(bb, qq).double().item()}, g_cat)
            if kldivfix is not None and dn in ('fp16', 'fp32'):
                key = f'ours_kl_div:{dn}'
                if FIX_STATE['error'] is None:
                    try:
                        per = torch.stack([kldivfix.kl_div(lq[i], pb[i], reduction='sum', method='D64') for i in range(T)]).double()
                        r[key] = (per, cv(lambda **k: kldivfix.kl_div(lq, pb, method='D64', **k)), g_kl)
                        FIX_STATE['chunks_ok'] += 1
                    except Exception as e:
                        FIX_STATE['error'] = f'{type(e).__name__}: {e}'[:300]
                        print('kldivfix failed:', FIX_STATE['error'], flush=True)
                if FIX_STATE['error'] is not None and key not in r:
                    r[key] = (nanT, {'batchmean': float('nan'), 'mean': float('nan')}, g_kl)
                    placeholder.add(key)
                    FIX_STATE['chunks_failed'] += 1
            del bb, qq, lq, lb, pb
    return {k: v + (k not in placeholder,) for k, v in r.items()}

def expected_methods(kldivfix):
    keys = {f'{m}:{dn}' for m in ('kl_div', 'kl_div_lt', 'categorical', 'logitsA') for dn in DT}
    if kldivfix is not None:
        keys |= {f'ours_kl_div:{dn}' for dn in ('fp16', 'fp32')}
    return keys

def rel_or_inf(x, ref):
    if not np.isfinite(x):
        return None
    if ref == 0:
        return 0.0 if x == 0 else float('inf')
    return abs(x - ref) / abs(ref)

def cr_frac(v, ref, dn):
    r = _round_once(torch.from_numpy(np.ascontiguousarray(ref, dtype=np.float64)), DT[dn]).double().numpy()
    return float(np.mean((v == r) & np.isfinite(v))) if v.size else None

def err_stats(v, ref, prefix):
    with np.errstate(all='ignore'):
        rel = np.where(ref != 0, np.abs(v - ref) / np.abs(ref), np.where(v == ref, 0.0, np.inf))
    rel = np.where(np.isfinite(v), rel, np.inf)
    fr = rel[np.isfinite(rel)]
    return {f'{prefix}_err_ge_1pct': float(np.mean(rel >= 0.01)), f'{prefix}_err_ge_10pct': float(np.mean(rel >= 0.1)), f'{prefix}_err_ge_100pct': float(np.mean(rel >= 1.0)), f'{prefix}_err_median_finite': float(np.median(fr)) if fr.size else None}

def ref_check(b, q, n_mp=4, n_rand=2, n_given=2):
    import mpmath as mp
    k1 = exact_kl(b, q)
    k2 = naive_kl(b, q)
    d12 = (k1 - k2).abs()
    nz = k1 != 0
    c = dict(two_fp64_max_rel=(d12[nz] / k1[nz].abs()).max().item() if nz.any() else 0.0, two_fp64_max_abs_at_zero=d12[~nz].max().item() if (~nz).any() else 0.0, mpmath_max_rel=0.0, mpmath_max_abs_at_zero=0.0, given_fp32_mpmath_max_rel=0.0)
    mp.mp.prec = 150
    order = torch.argsort(k1).tolist()
    g = torch.Generator().manual_seed(0)
    picks = order[:n_mp] + [order[i] for i in torch.randperm(len(order), generator=g)[:n_rand].tolist() if order[i] not in order[:n_mp]]
    for i in picks:
        bv, qv = ([mp.mpf(x) for x in b[i].double().tolist()], [mp.mpf(x) for x in q[i].double().tolist()])
        lsb, lsq = (mp.log(mp.fsum((mp.exp(x) for x in bv))), mp.log(mp.fsum((mp.exp(x) for x in qv))))
        ex = mp.fsum((mp.exp(x - lsb) * (x - lsb - (y - lsq)) for x, y in zip(bv, qv)))
        err = abs(mp.mpf(k1[i].item()) - ex)
        if ex == 0:
            c['mpmath_max_abs_at_zero'] = max(c['mpmath_max_abs_at_zero'], float(err))
        else:
            c['mpmath_max_rel'] = max(c['mpmath_max_rel'], float(err / abs(ex)))
    bb, qq = (b.float(), q.float())
    pb, lq = (bb.softmax(-1), qq.log_softmax(-1))
    gk = torch.xlogy(pb.double(), pb.double()).sum(-1) - (pb.double() * lq.double()).sum(-1)
    for i in order[:n_given]:
        t, x = ([mp.mpf(v) for v in pb[i].double().tolist()], [mp.mpf(v) for v in lq[i].double().tolist()])
        ex = mp.fsum((ti * (mp.log(ti) - xi) for ti, xi in zip(t, x) if ti > 0))
        if ex != 0:
            c['given_fp32_mpmath_max_rel'] = max(c['given_fp32_mpmath_max_rel'], float(abs(mp.mpf(gk[i].item()) - ex) / abs(ex)))
    return c

def stage_kl(pairs, store, out, device, kldivfix, chunk, vocab):
    res = {}
    for base_name, var in pairs:
        B = torch.load(store / f'{base_name}.pt')
        Q = torch.load(store / f'{var}.pt')
        nw, T, V = B.shape
        done_npz = out / f'pertoken_{base_name}__{var}.npz'
        done_json = out / f'pair_{base_name}__{var}.json'
        if done_npz.exists() and done_json.exists():
            prev = json.load(open(done_json))
            missing = expected_methods(kldivfix) - set(prev['methods'])
            if not missing and all((m['complete'] for m in prev['methods'].values())):
                res[f'{base_name}__{var}'] = prev
                print(f'[kl] {var}: reused', flush=True)
                continue
            print(f'[kl] {var}: previous result incomplete' + (f' (missing {sorted(missing)})' if missing else '') + ', recomputing', flush=True)
        t1 = time.time()
        per = {}
        chunks = {}
        given = {}
        ex = []
        exd = {d: [] for d in DT}
        refc = {}
        ncl = []
        cmask = {}
        for w in range(nw):
            for s0 in range(0, T, chunk):
                b, q = (B[w, s0:s0 + chunk].to(device), Q[w, s0:s0 + chunk].to(device))
                ex.append(exact_kl(b, q).cpu())
                ncl.append(b.shape[0])
                if s0 == 0 and w < 2:
                    refc = {k: max(v, refc.get(k, 0.0)) for k, v in ref_check(b, q).items()}
                for d, dt in DT.items():
                    exd[d].append(exact_kl(b.to(dt), q.to(dt)).cpu())
                for k, (pt, cv, gv, comp) in methods(b, q, kldivfix).items():
                    per.setdefault(k, []).append(pt.cpu())
                    cmask.setdefault(k, []).append(np.full(pt.shape[0], comp))
                    chunks.setdefault(k, []).append(cv)
                    given.setdefault(k, []).append(gv.cpu())
                del b, q
        ex = torch.cat(ex).numpy()
        exd = {d: torch.cat(v).numpy() for d, v in exd.items()}
        per = {k: torch.cat(v).numpy() for k, v in per.items()}
        given = {k: torch.cat(v).numpy() for k, v in given.items()}
        if len({len(v) for v in per.values()}) != 1:
            raise SystemExit('methods produced different token counts (kldivfix failed part-way?)')
        np.savez_compressed(done_npz, exact=ex, **{f'exact_{d}': v for d, v in exd.items()}, **{k.replace(':', '__'): v for k, v in per.items()}, **{'given__' + k.replace(':', '__'): v for k, v in given.items()}, **{'computed__' + k.replace(':', '__'): np.concatenate(v) for k, v in cmask.items()})
        rows = {}
        exm = float(ex.mean())
        for k, v in per.items():
            d = k.split(':')[1]
            ref = given[k]
            cm = np.concatenate(cmask[k])
            ncomp = int(cm.sum())
            vals = {red: float(np.average([c[red] for c in chunks[k]], weights=ncl)) for red in chunks[k][0]}
            ex_red = {'batchmean': exm, 'mean': exm / vocab}
            row = dict(complete=bool(cm.all()), tokens_computed=ncomp, reported=vals, rel_err_of_reported={red: rel_or_inf(x, ex_red[red]) for red, x in vals.items()}, chunk_values_nonfinite={red: int(sum((not np.isfinite(c[red]) for c in chunks[k]))) for red in chunks[k][0]}, mean_exact_model_logits=exm, mean_exact_dtype_logits=float(exd[d].mean()), mean_exact_given_inputs=float(ref.mean()), tokens=int(v.size), inf=int(np.isinf(v[cm]).sum()), nan=int(np.isnan(v[cm]).sum()), negative=int((v[cm] < 0).sum()), nonfinite_frac=float(np.mean(~np.isfinite(v[cm]))) if ncomp else None)
            if ncomp:
                row.update(err_stats(v[cm], ref[cm], 'operator'))
                row.update(err_stats(v[cm], ex[cm], 'total'))
                row.update(err_stats(ref[cm], ex[cm], 'input_representation'))
                row.update(operator_cr_frac=cr_frac(v[cm], ref[cm], d), total_cr_frac=cr_frac(v[cm], ex[cm], d))
            else:
                row.update({f'{p}_err_{t}': None for p in ('operator', 'total', 'input_representation') for t in ('ge_1pct', 'ge_10pct', 'ge_100pct', 'median_finite')})
                row.update(operator_cr_frac=None, total_cr_frac=None)
            rows[k] = row
        res[f'{base_name}__{var}'] = dict(exact_mean=exm, exact_median=float(np.median(ex)), exact_min=float(ex.min()), methods=rows, reference_check=refc, seconds=round(time.time() - t1, 1))
        json.dump(res[f'{base_name}__{var}'], open(done_json, 'w'), indent=1)
        print(f"[kl] {var}: {res[f'{base_name}__{var}']['seconds']} s, reference check {refc}", flush=True)
        del B, Q
        gc.collect()
    return res

def compare_pairs(exact, reported):
    ks = list(exact)
    c = dict(pairs=0, strict_inversions=0, reported_ties=0, reference_ties=0)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = (ks[i], ks[j])
            c['pairs'] += 1
            if exact[a] == exact[b]:
                c['reference_ties'] += 1
            elif reported[a] == reported[b]:
                c['reported_ties'] += 1
            elif (exact[a] < exact[b]) != (reported[a] < reported[b]):
                c['strict_inversions'] += 1
    return c

def ranking(res, family_pairs):
    out = {}
    for fam, pairs in family_pairs.items():
        keys = [f'{b}__{v}' for b, v in pairs if f'{b}__{v}' in res]
        if len(keys) < 2:
            out[fam] = dict(note='fewer than two variants passed the sanity gate')
            continue
        exact = sorted(keys, key=lambda k: res[k]['exact_mean'])
        rk = {}
        for m in res[keys[0]]['methods']:
            for red in res[keys[0]]['methods'][m]['reported']:
                vals = {k: res[k]['methods'][m]['reported'][red] for k in keys}
                if not all((np.isfinite(x) for x in vals.values())):
                    rk[f'{m}[{red}]'] = dict(rankable=False, values=vals)
                    continue
                order = sorted(keys, key=lambda k: vals[k])
                rk[f'{m}[{red}]'] = dict(rankable=True, order=[k.split('__')[1] for k in order], values=vals, **compare_pairs({k: res[k]['exact_mean'] for k in keys}, vals))
        out[fam] = dict(exact_order=[k.split('__')[1] for k in exact], methods=rk)
    return out
SANITY = {'base_ppl_max': 100.0, 'ratio_max': {8: 1.05, 4: 1.5, 3: 10.0}}

def sanity(fm, lim_cfg=SANITY):
    base = fm['variants']['base']['ppl']
    bad = {}
    if not (math.isfinite(base) and base < lim_cfg['base_ppl_max']):
        bad['base'] = f'base perplexity {base}'
    for v, info in fm['variants'].items():
        if v == 'base':
            continue
        bits = info.get('bits') or (info.get('quantization_config') or {}).get('bits')
        ratio = info['ppl'] / base
        info['ppl_ratio_to_base'] = ratio
        lim = {int(k): x for k, x in lim_cfg['ratio_max'].items()}.get(bits)
        if lim is None or not math.isfinite(ratio) or ratio > lim or (ratio < 1 / 1.05):
            bad[v] = f'perplexity ratio {ratio:.4f} (limit {lim}, bits {bits})'
    return bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--hub', default=os.environ.get('HF_HUB_CACHE', ''))
    ap.add_argument('--text-file', default=None)
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--no-fix', action='store_true')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = Path(a.scratch)
    store.mkdir(parents=True, exist_ok=True)
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, device=a.device, python=sys.version.split()[0])
    if a.device == 'cuda':
        env.update(gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0), bf16=torch.cuda.is_bf16_supported())
    import transformers
    env['transformers'] = transformers.__version__
    kldivfix = None
    if not a.no_fix:
        from stablekl import interface_b as kldivfix
        env['kldivfix'] = kldivfix.__version__
        env['kldivfix_sha256'] = hashlib.sha256(open(kldivfix.__file__, 'rb').read()).hexdigest()
    meta = dict(env=env, config=cfg, families={})
    t0 = time.time()
    for fam in cfg['families']:
        base_path = fam['base_path'] if 'base_path' in fam else snapshot(fam['base'], a.hub)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_path)
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
        W, L = (fam['windows'], fam['window_len'])
        if len(ids) < W * L:
            raise SystemExit('text too short')
        windows = [ids[i * L:(i + 1) * L] for i in range(W)]
        fm = dict(base_path=base_path, tokens_sha256=hashlib.sha256(np.array(windows, dtype=np.int64).tobytes()).hexdigest(), n_tokens=W * L, variants={})
        for v in fam['variants']:
            name = f"{fam['name']}:{v['name']}"
            t1 = time.time()
            model, info = build(v, base_path, a.hub, a.device)
            Lg, chk = forward_logits(model, windows, a.device)
            info.update(chk)
            info['seconds'] = round(time.time() - t1, 1)
            if chk['nonfinite_logits']:
                raise SystemExit(f'{name}: non-finite logits')
            fm['vocab'] = int(Lg.shape[-1])
            torch.save(Lg, store / f'{name}.pt')
            torch.save(Lg[0].clone(), out / f"logits_window0_{name.replace(':', '_')}.pt")
            info['logits_sha256'] = hashlib.sha256(Lg.numpy().tobytes()).hexdigest()
            fm['variants'][v['name']] = info
            print(f"[logits] {name}: ppl {chk['ppl']:.4f}  ({info['seconds']} s)", flush=True)
            del model, Lg
            gc.collect()
            torch.cuda.empty_cache() if a.device == 'cuda' else None
        fm['sanity_failures'] = sanity(fm, cfg.get('sanity', SANITY))
        fm['sanity_limits'] = cfg.get('sanity', SANITY)
        meta['families'][fam['name']] = fm
        if fm['sanity_failures']:
            print('SANITY GATE FAILED:', fm['sanity_failures'], flush=True)
        json.dump(meta, open(out / 'meta.json', 'w'), indent=1, default=str)
    ok = lambda f, v: 'base' not in meta['families'][f['name']]['sanity_failures'] and v['name'] not in meta['families'][f['name']]['sanity_failures']
    fam_pairs = {f['name']: [(f"{f['name']}:base", f"{f['name']}:{v['name']}") for v in f['variants'] if v['kind'] != 'base' and ok(f, v)] for f in cfg['families']}
    vocab = {f['name']: meta['families'][f['name']]['vocab'] for f in cfg['families']}
    res = {}
    for f in cfg['families']:
        res.update(stage_kl(fam_pairs[f['name']], store, out, a.device, kldivfix, a.chunk, vocab[f['name']]))
    rk = ranking(res, fam_pairs)
    meta['kldivfix'] = dict(FIX_STATE)
    sane = not any((fm['sanity_failures'] for fm in meta['families'].values()))
    fix_ok = kldivfix is not None and FIX_STATE['error'] is None
    status = 'COMPLETE' if sane and fix_ok else 'HOLD_SANITY' if not sane else 'PARTIAL_FIX_INCOMPLETE'
    json.dump(dict(status=status, sanity={k: v['sanity_failures'] for k, v in meta['families'].items()}, kldivfix=meta['kldivfix']), open(out / 'status.json', 'w'), indent=1)
    json.dump(dict(meta=meta, results=res, ranking=rk, seconds=round(time.time() - t0, 1)), open(out / 'summary.json', 'w'), indent=1, default=str)
    write_md(out, meta, res, rk)
    print(f'done in {time.time() - t0:.0f} s, status {status}', flush=True)
    sys.exit({'COMPLETE': 0, 'PARTIAL_FIX_INCOMPLETE': 3, 'HOLD_SANITY': 4}[status])

def write_md(out, meta, res, rk):
    pc = lambda v: '—' if v is None else f'{v:.1%}'
    ee = lambda v: '—' if v is None or not np.isfinite(v) else f'{v:.4e}'
    L = ['# KL(base || quantised) as an evaluation number', '', 'Reference = float64 evaluation (not an Arb-certified truth); its self-check is listed per pair.  The KL of logits is evaluated in a cancellation-free form (about 1e-15 relative); the given-input value of the probability-input calls (kl_div and its repaired version) is a plain FP64 sum whose relative error on near-identical rows can reach ~1e-10 (listed per pair), so their fp32 CR fraction is a lower bound (an fp32 unit is ~6e-8 relative).', 'Errors are counted over ALL tokens: a non-finite output counts as an error.  operator = against the exact KL of the inputs passed to the call; total = against the KL of the FP16 model logits; input repr. = difference between the two references.  CR = output equals the reference rounded once to the dtype (correct rounding; an FP16/BF16 output cannot do better).  logitsA = logits-path KL (interface-A formula); in bf16 it is exploratory.', '']
    for fam, fm in meta['families'].items():
        L += [f'## {fam}', '', f"Sanity gate failures: {fm['sanity_failures'] or 'none'}", '', '| variant | perplexity | ratio to base | reference mean KL per token | reference self-check (naive-form difference / reference vs mpmath; given-input fp32 reference vs mpmath) |', '|---|---|---|---|---|']
        for v, info in fm['variants'].items():
            key = f'{fam}:base__{fam}:{v}'
            r = res.get(key)
            rc = r['reference_check'] if r else None
            chk = '%.1e / %.1e (at zero KL: abs %.1e / %.1e); given fp32 vs mpmath %.1e' % (rc['two_fp64_max_rel'], rc['mpmath_max_rel'], rc['two_fp64_max_abs_at_zero'], rc['mpmath_max_abs_at_zero'], rc.get('given_fp32_mpmath_max_rel', float('nan'))) if rc else '—'
            L.append(f"| {v} | {info['ppl']:.4f} | {info.get('ppl_ratio_to_base', 1.0):.4f} | {(ee(r['exact_mean']) if r else '—')} | {chk} |")
        L += ['', '| variant | method | reported batchmean (rel. err) | reported mean (rel. err) | inf | NaN | negative | operator err >= 1% / >= 100% | total err >= 1% / >= 100% | input repr. >= 1% | CR operator / total |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for key, r in res.items():
            if not key.startswith(fam + ':'):
                continue
            for m, x in r['methods'].items():
                rep_ = lambda red: '—' if red not in x['reported'] else f"{ee(x['reported'][red])} ({pc(x['rel_err_of_reported'][red])})"
                inc = '' if x['complete'] else ' (incomplete: %d/%d tokens)' % (x['tokens_computed'], x['tokens'])
                L.append(f"| {key.split('__')[1].split(':')[1]} | {m}{inc} | {rep_('batchmean')} | {rep_('mean')} | {x['inf']} | {x['nan']} | {x['negative']} | {pc(x['operator_err_ge_1pct'])} / {pc(x['operator_err_ge_100pct'])} | {pc(x['total_err_ge_1pct'])} / {pc(x['total_err_ge_100pct'])} | {pc(x['input_representation_err_ge_1pct'])} | {pc(x.get('operator_cr_frac'))} / {pc(x.get('total_cr_frac'))} |")
        r = rk.get(fam, {})
        if 'exact_order' in r:
            L += ['', f"Reference order (smallest KL first): {' < '.join((v.split(':')[1] for v in r['exact_order']))}", '', 'Pairs: strict inversion = reported order opposite to the reference; reported tie = reference values differ but the reported values are equal (cannot tell apart, not an inversion); reference tie = reference values equal (not judged).', '', '| method [reduction] | order (by reported value) | strict inversions | reported ties | reference ties | pairs |', '|---|---|---|---|---|---|']
            for m, x in r['methods'].items():
                L.append(f"| {m} | {' < '.join((v.split(':')[1] for v in x['order']))} | {x['strict_inversions']} | {x['reported_ties']} | {x['reference_ties']} | {x['pairs']} |" if x['rankable'] else f'| {m} | not rankable (non-finite value) | — | — | — | — |')
        L.append('')
    (out / 'SUMMARY.md').write_text('\n'.join(L), encoding='utf-8')
if __name__ == '__main__':
    main()
