import glob, json, os, sys
import torch
DEC = [(0, 1e-10), (1e-10, 1e-09), (1e-09, 1e-08), (1e-08, 1e-07), (1e-07, 1e-06), (1e-06, 1e-05), (1e-05, 0.0001), (0.0001, 0.001), (0.001, float('inf'))]
TAOS = [-8, -7, -6, -5, -4, -3]
TOPK = [100, 1000, 5000]

def wrong(est, tr):
    return ~torch.isfinite(est) | (est < 0) | ((est - tr).abs() > 0.1 * tr)

def ranks(x):
    r = torch.empty(x.numel(), dtype=torch.float64)
    r[torch.argsort(x, stable=True)] = torch.arange(x.numel(), dtype=torch.float64)
    return r

def spearman(a, b):
    ra, rb = (ranks(a), ranks(b))
    ra, rb = (ra - ra.mean(), rb - rb.mean())
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))

def topk_overlap(a, b, k):
    k = min(k, a.numel())
    A = set(torch.topk(a, k).indices.tolist())
    B = set(torch.topk(b, k).indices.tolist())
    return len(A & B) / k

def main(root):
    D = {}
    for p in sorted(glob.glob(os.path.join(root, 'scores_*.pt'))):
        d = torch.load(p, weights_only=False)
        D[d['tag'], d['dtype']] = d
    out = dict(dtypes={}, verdicts={})
    L = ["# Per-edge importance scores (auto-circuit 1.0.1 ablation loop), further models and tasks: the library's KL vs the logits path", '']
    order = [(tg, x) for tg in sorted({k[0] for k in D}) for x in ('fp32', 'fp16', 'bf16') if (tg, x) in D]
    f = lambda x: f'{x:.3g}'
    for key in order:
        tg, dt = key
        d = D[key]
        ref = None
        if (tg, 'fp32') in D:
            r0 = D[tg, 'fp32']
            ref = torch.full((r0['n_edges'],), float('nan'), dtype=torch.float64)
            ref[r0['edge_index']] = r0['scores']['truth']
        S = d['scores']
        tr = S['truth']
        n = tr.numel()
        ests = [k for k in ('official', 'fp32cast', 'logits', 'logits32') if k in S]
        pos = tr > 0
        npos = int(pos.sum())
        q = torch.quantile(tr, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)).tolist()
        o = dict(tag=tg, dtype=dt, model=d.get('model'), n_planned=d['n_planned'], n_scored=n, n_edges=d['n_edges'], true_positive=npos, true_zero=int((tr == 0).sum()), true_nonfinite=int((~torch.isfinite(tr)).sum()), truth_quantiles=q, truth_max=float(tr.max()), null_max=max(d['info']['null']), seconds=d['info']['seconds'], est={}, decades=[], thresholds=[], rank={})
        L += [f"## {tg} / {dt} ({d.get('model')})", '', f"Edges scored: {n} of {d['n_planned']} planned ({d['n_edges']} in the model); true score > 0 for {npos}, exactly 0 for {o['true_zero']}.  True score: 5% / 25% / 50% / 75% / 95% quantiles " + ' / '.join((f(x) for x in q)) + f", maximum {f(o['truth_max'])}.  Forward pass with nothing ablated: largest KL {f(o['null_max'])}.  {o['seconds'] / 60:.1f} minutes.", '', '| estimator | negative | not finite | zero where the truth is > 0 | wrong (negative or > 10% off) among true > 0 | median relative error | 95% relative error |', '|---|---|---|---|---|---|---|']
        for k in ests:
            e = S[k]
            w = wrong(e, tr) & pos
            rel = ((e - tr).abs() / tr)[pos & torch.isfinite(e)]
            r = dict(negative=int((e < 0).sum()), nonfinite=int((~torch.isfinite(e)).sum()), zero_where_positive=int(((e == 0) & pos).sum()), wrong=int(w.sum()), wrong_frac=int(w.sum()) / max(npos, 1), rel_median=float(rel.median()) if rel.numel() else None, rel_p95=float(torch.quantile(rel, 0.95)) if rel.numel() else None)
            o['est'][k] = r
            L.append(f"| {k} | {r['negative']} | {r['nonfinite']} | {r['zero_where_positive']} | {r['wrong']} / {npos} ({r['wrong_frac']:.1%}) | {(f(r['rel_median']) if rel.numel() else '—')} | {(f(r['rel_p95']) if rel.numel() else '—')} |")
        L += ['', 'Wrong estimates by size of the true score:', '', '| true score | edges | ' + ' | '.join(ests) + ' |', '|---|---|' + '---|' * len(ests)]
        for lo, hi in DEC:
            m = (tr > lo) & (tr <= hi) if lo == 0 else (tr > lo) & (tr <= hi)
            c = int(m.sum())
            row = dict(lo=lo, hi=hi, edges=c, wrong={k: int((wrong(S[k], tr) & m).sum()) for k in ests})
            o['decades'].append(row)
            if c:
                L.append(f'| ({lo:g}, {hi:g}] | {c} | ' + ' | '.join((f"{row['wrong'][k]} ({row['wrong'][k] / c:.0%})" for k in ests)) + ' |')
        L += ['', "Edges selected by a threshold (score >= tao), independent decisions; 'differs' = edges whose decision differs from the true score's:", '', '| tao | selected by the true score | ' + ' | '.join((f'{k}: selected / differs' for k in ests)) + ' |', '|---|---|' + '---|' * len(ests)]
        for te in TAOS:
            tao = 10.0 ** te
            st = tr >= tao
            row = dict(tao_exp=te, true=int(st.sum()), est={k: dict(selected=int((S[k] >= tao).sum()), differs=int(((S[k] >= tao) != st).sum())) for k in ests})
            o['thresholds'].append(row)
            L.append(f"| 1e{te} | {row['true']} | " + ' | '.join((f"{row['est'][k]['selected']} / {row['est'][k]['differs']}" for k in ests)) + ' |')
        L += ['', 'Ranking of the edges, against the true score of the same model' + (' and against the true score of the FP32 model' if dt != 'fp32' and ref is not None else '') + ':', '', '| scores | Spearman | ' + ' | '.join((f'top-{k} overlap' for k in TOPK)) + ' |', '|---|---|' + '---|' * len(TOPK)]
        fin = lambda e: torch.nan_to_num(e, nan=-1.0, posinf=-1.0, neginf=-1.0)
        for k in ests:
            r = dict(spearman=spearman(fin(S[k]), tr), topk={kk: topk_overlap(fin(S[k]), tr, kk) for kk in TOPK})
            o['rank'][k] = r
            L.append(f"| {k} vs truth | {r['spearman']:.4f} | " + ' | '.join((f"{r['topk'][kk]:.3f}" for kk in TOPK)) + ' |')
        if dt != 'fp32' and ref is not None:
            rr = ref[d['edge_index']]
            ok = torch.isfinite(rr)
            for k in ['truth'] + ests:
                r = dict(spearman=spearman(fin(S[k])[ok], rr[ok]), topk={kk: topk_overlap(fin(S[k])[ok], rr[ok], kk) for kk in TOPK})
                o['rank'][k + '_vs_fp32'] = r
                L.append(f"| {k} vs FP32 truth | {r['spearman']:.4f} | " + ' | '.join((f"{r['topk'][kk]:.3f}" for kk in TOPK)) + ' |')
        L.append('')
        out['dtypes'][f'{tg}/{dt}'] = o
        if n < d['n_planned']:
            v = 'INCOMPLETE'
        else:
            v = 'SIGNAL' if o['est']['official']['wrong_frac'] >= 0.05 and o['est']['logits']['wrong_frac'] < 0.005 else 'no signal'
        out['verdicts'][f'{tg}/{dt}'] = dict(verdict=v, official_wrong=o['est']['official']['wrong_frac'], logits_wrong=o['est']['logits']['wrong_frac'])
    L += ['## Verdicts (fixed before the run)', ''] + [f"- {dt}: official wrong for {v['official_wrong']:.1%} of the edges with a true score > 0, logits path for {v['logits_wrong']:.2%} -> **{v['verdict']}**" for dt, v in out['verdicts'].items()]
    json.dump(out, open(os.path.join(root, 'edge2_summary.json'), 'w'), indent=1)
    open(os.path.join(root, 'EDGE2_SUMMARY.md'), 'w').write('\n'.join(L) + '\n')
    print('summary:', os.path.join(root, 'EDGE2_SUMMARY.md'))
if __name__ == '__main__':
    main(sys.argv[1])
