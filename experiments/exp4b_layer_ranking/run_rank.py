from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from fingerprint import code_md5
import argparse, gc, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval as R
from dequant import rtn
_ref_on = {'on': True}
_ref_orig = R.ref_check

def _ref_wrapped(b, q, *a, **k):
    if _ref_on['on']:
        return _ref_orig(b, q, *a, **k)
    return dict(skipped=True)
R.ref_check = _ref_wrapped

def layer_linears(model, i):
    pre = f'model.layers.{i}.'
    lin = {n: m for n, m in model.named_modules() if n.startswith(pre) and isinstance(m, torch.nn.Linear)}
    if len(lin) != 7:
        raise SystemExit(f'layer {i}: expected 7 linear modules, found {sorted(lin)}')
    return lin

class Quantiser:

    def __init__(self, model):
        self.model = model
        self.saved = {}

    def apply(self, layers, bits, group):
        assert not self.saved, 'restore before the next apply'
        for i in layers:
            for n, m in layer_linears(self.model, i).items():
                self.saved[n] = (m, m.weight.data.clone())
                m.weight.data = rtn(m.weight.data, bits, group).to(m.weight.device)

    def restore(self):
        for n, (m, w) in self.saved.items():
            m.weight.data = w
        self.saved = {}

def weights_sha(model):
    h = hashlib.sha256()
    for n, m in model.named_modules():
        if n.startswith('model.layers.') and isinstance(m, torch.nn.Linear):
            h.update(n.encode())
            h.update(m.weight.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def tokens(base_path, W, L, text_file, E=0):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_path)
    if text_file:
        text = open(text_file, encoding='utf-8').read()
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
    if len(ids) < (W + E) * L:
        raise SystemExit('text too short')
    return ([ids[i * L:(i + 1) * L] for i in range(W)], [ids[i * L:(i + 1) * L] for i in range(W, W + E)])

def kendall_tau_b(x, y):
    n = len(x)
    c = d = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = (np.sign(x[i] - x[j]), np.sign(y[i] - y[j]))
            if a == 0 and b == 0:
                continue
            if a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif a == b:
                c += 1
            else:
                d += 1
    den = math.sqrt((c + d + tx) * (c + d + ty))
    return (c - d) / den if den else float('nan')

def top_k(values, k):
    if not all((np.isfinite(v) for v in values.values())):
        return (None, False)
    order = sorted(values, key=lambda key: (-values[key], key))
    det = len(order) <= k or values[order[k - 1]] != values[order[k]]
    return (sorted(order[:k]), det)

def judge(ref_vals, vals, k):
    rs, ref_det = top_k(ref_vals, k)
    s, det = top_k(vals, k)
    thr = sorted(ref_vals.values(), reverse=True)[min(k, len(ref_vals)) - 1]
    out = dict(reference_tie=not ref_det, rankable=s is not None, determined=det)
    if s is None:
        return out
    out.update(chosen=s, n_wrong=sum((ref_vals[x] < thr for x in s)))
    return out

def exact_mean_kl(B, Q, device, chunk):
    tot = 0.0
    n = 0
    for w in range(B.shape[0]):
        for s0 in range(0, B.shape[1], chunk):
            b, q = (B[w, s0:s0 + chunk].to(device), Q[w, s0:s0 + chunk].to(device))
            tot += R.exact_kl(b, q).sum().item()
            n += b.shape[0]
    return tot / n

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
    t0 = time.time()
    fam = cfg['family']
    base_path = fam['base_path'] if 'base_path' in fam else R.snapshot(fam['base'], a.hub)
    windows, ewin = tokens(base_path, fam['windows'], fam['window_len'], a.text_file, fam.get('eval_windows', 0))
    if not ewin:
        raise SystemExit('config: family.eval_windows must be >= 1 (independent evaluation set)')
    tsha = lambda w: hashlib.sha256(np.array(w, dtype=np.int64).tobytes()).hexdigest()
    meta = dict(env=env, config=cfg, base_path=base_path, tokens_sha256=tsha(windows), eval_tokens_sha256=tsha(ewin), variants={})
    model, _ = R.build({'kind': 'base'}, base_path, a.hub, a.device)
    nl = model.config.num_hidden_layers
    meta['n_layers'] = nl
    Lb, chk = R.forward_logits(model, windows, a.device)
    base_sha = hashlib.sha256(Lb.numpy().tobytes()).hexdigest()
    if chk['nonfinite_logits']:
        raise SystemExit('base: non-finite logits')
    base_ppl = chk['ppl']
    meta['base'] = dict(ppl=base_ppl, logits_sha256=base_sha)
    vocab = int(Lb.shape[-1])
    torch.save(Lb, store / 'base.pt')
    del Lb
    Le, ce = R.forward_logits(model, ewin, a.device)
    meta['base'].update(eval_ppl=ce['ppl'], eval_logits_sha256=hashlib.sha256(Le.numpy().tobytes()).hexdigest())
    torch.save(Le, store / 'base_eval.pt')
    del Le
    here = Path(__file__).resolve().parent
    fp = dict(code={f: code_md5(f, here) for f in ('run_rank.py', 'run_eval.py', 'logits_kl.py', 'dequant.py')}, config=hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest(), tokens=meta['tokens_sha256'], eval_tokens=meta['eval_tokens_sha256'], base_logits=base_sha, base_eval_logits=meta['base']['eval_logits_sha256'], torch=torch.__version__)
    if (out / 'fingerprint.json').exists():
        old = json.load(open(out / 'fingerprint.json'))
        if old != fp:
            raise SystemExit(f'RESUME REFUSED: {out} was written with different code / config / tokens / base logits\n old {old}\n now {fp}')
        print('resume: fingerprint matches', flush=True)
    else:
        json.dump(fp, open(out / 'fingerprint.json', 'w'), indent=1)
    print(f'[base] ppl {base_ppl:.4f}, {nl} layers', flush=True)
    if not (math.isfinite(base_ppl) and base_ppl < cfg['sanity']['base_ppl_max']):
        raise SystemExit(f'base perplexity {base_ppl} fails the gate')
    w_sha0 = weights_sha(model)
    Qz = Quantiser(model)
    lim = {int(k): v for k, v in cfg['sanity']['ratio_max'].items()}
    res = {}
    sweep_pairs = {}
    failures = {}
    every = int(cfg.get('ref_check_every', 7))
    for sw in cfg['sweeps']:
        sweep_pairs[sw['name']] = []
        for i in range(nl):
            name = f"{sw['name']}_L{i:02d}"
            pair = ('base', name)
            vinfo = out / f'var_{name}.json'
            pj, pz = (out / f'pair_base__{name}.json', out / f'pertoken_base__{name}.npz')
            if vinfo.exists() and pj.exists() and pz.exists():
                prev = json.load(open(pj))
                if not R.expected_methods(kldivfix) - set(prev['methods']) and all((m['complete'] for m in prev['methods'].values())):
                    meta['variants'][name] = json.load(open(vinfo))
                    res[f'base__{name}'] = prev
                    sweep_pairs[sw['name']].append(pair)
                    print(f'[layer] {name}: reused', flush=True)
                    continue
            t1 = time.time()
            Qz.apply([i], sw['bits'], sw.get('group'))
            L, c = R.forward_logits(model, windows, a.device)
            Qz.restore()
            info = dict(layer=i, bits=sw['bits'], group=sw.get('group'), ppl=c['ppl'], ppl_ratio_to_base=c['ppl'] / base_ppl, nonfinite_logits=c['nonfinite_logits'], logits_sha256=hashlib.sha256(L.numpy().tobytes()).hexdigest())
            r = info['ppl_ratio_to_base']
            if c['nonfinite_logits'] or not cfg['sanity'].get('ratio_min', 1 / 1.05) <= r <= lim[sw['bits']]:
                failures[name] = f'ratio {r:.4f}'
                json.dump(info, open(vinfo, 'w'))
                del L
                continue
            torch.save(L, store / f'{name}.pt')
            del L
            _ref_on['on'] = i % every == 0
            res.update(R.stage_kl([pair], store, out, a.device, kldivfix, a.chunk, vocab))
            os.remove(store / f'{name}.pt')
            info['seconds'] = round(time.time() - t1, 1)
            meta['variants'][name] = info
            json.dump(info, open(vinfo, 'w'))
            sweep_pairs[sw['name']].append(pair)
            print(f"[layer] {name}: ppl ratio {r:.5f}, exact KL {res[f'base__{name}']['exact_mean']:.4e}, {info['seconds']} s", flush=True)
            gc.collect()
    meta['restore_check'] = weights_sha(model) == w_sha0
    if not meta['restore_check']:
        raise SystemExit('restore check failed: weights not restored exactly')
    rk = R.ranking(res, sweep_pairs)
    B = torch.load(store / 'base.pt')
    BE = torch.load(store / 'base_eval.pt')
    dec = {}
    for sw in cfg['sweeps']:
        keys = [f'base__{v}' for _, v in sweep_pairs[sw['name']]]
        layer_of = {k: meta['variants'][k.split('__')[1]]['layer'] for k in keys}
        if len(keys) < 2:
            dec[sw['name']] = dict(note='fewer than two layers passed the gate')
            continue
        ref_vals = {k: res[k]['exact_mean'] for k in keys}
        dsw = dict(methods={}, mixed={})
        methods = {f'{m}[{red}]': {k: res[k]['methods'][m]['reported'][red] for k in keys} for m in res[keys[0]]['methods'] for red in res[keys[0]]['methods'][m]['reported']}
        tagf = lambda kk, S: f"{sw['name']}_k{kk}_" + '-'.join(map(str, S))
        for k_top in cfg['top_k']:
            ref_set, ref_det = top_k(ref_vals, k_top)
            ref_layers = sorted((layer_of[x] for x in ref_set))
            entry = dict(reference=dict(layers=ref_layers, determined=ref_det, reference_tie=not ref_det), methods={})
            for mname, vals in methods.items():
                j = judge(ref_vals, vals, k_top)
                if not j['rankable']:
                    entry['methods'][mname] = dict(rankable=False)
                    continue
                entry['methods'][mname] = dict(rankable=True, layers=sorted((layer_of[x] for x in j['chosen'])), determined=j['determined'], n_wrong=j['n_wrong'], reference_tie=j['reference_tie'], kendall_tau_b=kendall_tau_b([ref_vals[x] for x in keys], [vals[x] for x in keys]))
            sets = {tuple(ref_layers)} | {tuple(v['layers']) for v in entry['methods'].values() if v.get('rankable') and v['determined']}
            for S in sorted(sets):
                tag = tagf(k_top, S)
                mj = out / f'mixed_{tag}.json'
                if mj.exists():
                    prev = json.load(open(mj))
                    if 'eval' in prev:
                        dsw['mixed'][tag] = prev
                        continue
                q_layers = [i for i in range(nl) if i not in S]
                Qz.apply(q_layers, sw['bits'], sw.get('group'))
                L, c = R.forward_logits(model, windows, a.device)
                Le, ce = R.forward_logits(model, ewin, a.device)
                Qz.restore()
                mk = dict(sweep=sw['name'], k=k_top, fp16_layers=list(S), quantised_layers=len(q_layers), selection=dict(true_kl=exact_mean_kl(B, L, a.device, a.chunk), ppl=c['ppl']), eval=dict(true_kl=exact_mean_kl(BE, Le, a.device, a.chunk), ppl=ce['ppl']))
                json.dump(mk, open(mj, 'w'))
                dsw['mixed'][tag] = mk
                del L, Le
                gc.collect()
                print(f"[mixed] {tag}: selection KL {mk['selection']['true_kl']:.5e} ppl {mk['selection']['ppl']:.4f} | eval KL {mk['eval']['true_kl']:.5e} ppl {mk['eval']['ppl']:.4f}", flush=True)
            ref_m = dsw['mixed'][tagf(k_top, ref_layers)]
            entry['reference'].update(selection=ref_m['selection'], eval=ref_m['eval'])
            for mname, v in entry['methods'].items():
                if not (v.get('rankable') and v['determined']):
                    continue
                m = dsw['mixed'][tagf(k_top, v['layers'])]
                v.update(selection=m['selection'], eval=m['eval'])
                if not v['reference_tie']:
                    v['vs_reference'] = {ds: dict(kl_rel=m[ds]['true_kl'] / ref_m[ds]['true_kl'] - 1, ppl_diff=m[ds]['ppl'] - ref_m[ds]['ppl']) for ds in ('selection', 'eval')}
            dsw['methods'][f'k{k_top}'] = entry
        dec[sw['name']] = dsw
    if weights_sha(model) != w_sha0:
        raise SystemExit('restore check failed after the mixed-precision stage')
    status = 'COMPLETE' if not failures and (kldivfix is not None and R.FIX_STATE['error'] is None) else 'PARTIAL_FIX_INCOMPLETE' if not failures else 'HOLD_SANITY'
    meta['sanity_failures'] = failures
    meta['kldivfix'] = dict(R.FIX_STATE)
    meta['seconds'] = round(time.time() - t0, 1)
    json.dump(dict(meta=meta, ranking=rk, decisions=dec, layer_results={k: dict(exact_mean=v['exact_mean'], reported={m: x['reported'] for m, x in v['methods'].items()}) for k, v in res.items()}), open(out / 'rank_summary.json', 'w'), indent=1, default=str)
    json.dump(dict(status=status, sanity_failures=failures, kldivfix=meta['kldivfix']), open(out / 'status.json', 'w'), indent=1)
    write_md(out, meta, res, rk, dec, cfg)
    print(f'done in {time.time() - t0:.0f} s, status {status}', flush=True)
    sys.exit({'COMPLETE': 0, 'PARTIAL_FIX_INCOMPLETE': 3, 'HOLD_SANITY': 4}[status])

def write_md(out, meta, res, rk, dec, cfg):
    e = lambda v: '—' if v is None or not np.isfinite(v) else f'{v:.4e}'
    L = ['# Layer-wise KL sensitivity and the mixed-precision choice', '', f"Base perplexity {meta['base']['ppl']:.4f}; {meta['n_layers']} layers; restore check {meta['restore_check']}; sanity failures: {meta['sanity_failures'] or 'none'}.", '', f"Sensitivity of layer i = KL(base || model with only layer i quantised).  Reference = FP64 KL of the FP16 model logits (cancellation-free, mpmath-checked on every {cfg.get('ref_check_every', 7)}-th layer).  Decision: keep the k most sensitive layers in FP16, quantise the rest.  A choice is 'not determined' when the k-th and (k+1)-th reported values are equal.  'wrong layers' = chosen layers whose reference KL is strictly below the reference's k-th largest value (ties admissible).  Mixed models are evaluated on the selection tokens and on an independent set of {cfg['family'].get('eval_windows')} x {cfg['family']['window_len']} later tokens (base perplexity there {meta['base'].get('eval_ppl', float('nan')):.4f}): FP64 reference KL to the base model and perplexity.", '']
    for sw in cfg['sweeps']:
        n = sw['name']
        L += [f"## Sweep {n} (RTN {sw['bits']}-bit, group {sw.get('group') or 'per channel'})", '']
        r = rk.get(n, {})
        if 'methods' in r:
            L += ['| method [reduction] | strict inversions | reported ties | pairs |', '|---|---|---|---|']
            for m, x in r['methods'].items():
                L.append(f"| {m} | {x['strict_inversions']} | {x['reported_ties']} | {x['pairs']} |" if x['rankable'] else f'| {m} | not rankable | — | — |')
        d = dec.get(n, {})
        for kk, entry in d.get('methods', {}).items():
            ref = entry['reference']
            rs, re_ = (ref['selection'], ref['eval'])
            L += ['', f"### {kk}: reference keeps layers {ref['layers']} in FP16" + (' — REFERENCE_TIE (no unique reference choice; differences not judged)' if ref['reference_tie'] else ''), '', f"Reference choice: selection set KL {e(rs['true_kl'])}, ppl {rs['ppl']:.4f}; independent set KL {e(re_['true_kl'])}, ppl {re_['ppl']:.4f}.", '', '| method [reduction] | layers kept in FP16 | wrong layers | Kendall tau-b | selection KL (vs ref.) | selection ppl (diff) | independent KL (vs ref.) | independent ppl (diff) |', '|---|---|---|---|---|---|---|---|']
            for m, v in entry['methods'].items():
                if not v.get('rankable'):
                    L.append(f'| {m} | not rankable | — | — | — | — | — | — |')
                    continue
                if not v['determined']:
                    L.append(f"| {m} | not determined (tie at the k-th place) | — | {v['kendall_tau_b']:.3f} | — | — | — | — |")
                    continue
                vr = v.get('vs_reference')
                wrong = 'REFERENCE_TIE' if v['reference_tie'] else str(v['n_wrong'])
                f = lambda ds: (f"{e(v[ds]['true_kl'])} ({vr[ds]['kl_rel']:+.2%})" if vr else e(v[ds]['true_kl']), f"{v[ds]['ppl']:.4f} ({vr[ds]['ppl_diff']:+.4f})" if vr else f"{v[ds]['ppl']:.4f}")
                (sk, sp), (ek, ep) = (f('selection'), f('eval'))
                L.append(f"| {m} | {v['layers']} | {wrong} | {v['kendall_tau_b']:.3f} | {sk} | {sp} | {ek} | {ep} |")
        L.append('')
    (out / 'RANK_SUMMARY.md').write_text('\n'.join(L), encoding='utf-8')
if __name__ == '__main__':
    main()
