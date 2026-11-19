import glob, json, os, sys
import torch

def main(root):
    R = {}
    for p in sorted(glob.glob(os.path.join(root, '*_tao1e*.pt'))):
        d = torch.load(p, weights_only=False)
        R[d['dtype'], d['mode'], d['tao_exp']] = d
    E = {dt: torch.load(p, weights_only=False) for p in glob.glob(os.path.join(root, 'edges_*.pt')) for dt in [os.path.basename(p)[6:-3]]}
    taos = sorted({k[2] for k in R})
    out = dict(rows=[], verdicts={})
    L = ['# ACDC circuit discovery with three KL computations (auto-circuit 1.0.1, IOI, gpt2-small)', '']
    n = next(iter(E.values()))['edge_exists'].numel() if E else None
    L += [f"Edges: {n}.  Reference = FP32 model, library KL.  'differs' = edges whose keep / remove decision differs from the reference.", '', '| dtype | KL mode | tao | status | kept | removed | Jaccard(kept) vs reference | differs from reference | recall of published IOI edges | negative KL values | KL calls | minutes |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for (dt, mode, te), d in sorted(R.items(), key=lambda kv: (kv[0][0] != 'fp32', kv[0][0], kv[0][2], kv[0][1])):
        st = d['stats']
        ref = R.get(('fp32', 'official', te))
        row = dict(dtype=dt, mode=mode, tao_exp=te, status=st.get('status'))
        if d['removed'] is None:
            L.append(f"| {dt} | {mode} | 1e{te} | {st.get('status')} | — | — | — | — | — | — | — | — |")
            out['rows'].append(row)
            continue
        kept = ~d['removed']
        jac = diff = rec = None
        if ref is not None and ref['removed'] is not None:
            rk = ~ref['removed']
            u = int((kept | rk).sum())
            jac = float((kept & rk).sum()) / u if u else 1.0
            diff = int((kept != rk).sum())
        te_ = E.get(dt, {}).get('true_edges')
        if te_ is not None and int(te_.sum()):
            rec = float((kept & te_).sum()) / int(te_.sum())
        row.update(kept=int(kept.sum()), removed=int(d['removed'].sum()), jaccard=jac, differs=diff, recall_true=rec, negative=st.get('official_negative'), seconds=st.get('seconds'))
        out['rows'].append(row)
        f = lambda x, p='{:.3f}': '—' if x is None else p.format(x)
        L.append(f"| {dt} | {mode} | 1e{te} | {st.get('status')} | {row['kept']} | {row['removed']} | {f(jac)} | {f(diff, '{}')} | {f(rec)} | {(st.get('official_negative', 0) if mode == 'official' else '—')} | {st.get('kl_calls')} | {st.get('seconds', 0) / 60:.1f} |")
    L += ['', '## Verdicts (fixed before the run)', '']
    for dt in sorted({k[0] for k in R} - {'fp32'}):
        better = worse = have = 0
        det = []
        for te in taos:
            a, b, ref = (R.get((dt, 'official', te)), R.get((dt, 'logits', te)), R.get(('fp32', 'official', te)))
            if any((x is None or x['removed'] is None for x in (a, b, ref))):
                continue
            have += 1
            rk = ~ref['removed']
            ao = int((~a['removed'] == rk).sum())
            bo = int((~b['removed'] == rk).sum())
            g = (bo - ao) / rk.numel()
            better += g >= 0.05
            worse += g < -0.01
            det.append(f'1e{te}: official {ao} / logits {bo} of {rk.numel()} ({g:+.1%})')
        v = 'INCOMPLETE' if have < len(taos) or have == 0 else 'SIGNAL' if better >= 2 and worse == 0 else 'no signal'
        out['verdicts'][dt] = dict(verdict=v, taos_compared=have, better=int(better), worse=int(worse), detail=det)
        L.append(f'- {dt}: edges agreeing with the FP32 reference — ' + '; '.join(det) + f' -> **{v}**')
    json.dump(out, open(os.path.join(root, 'acdc_summary.json'), 'w'), indent=1)
    open(os.path.join(root, 'ACDC_SUMMARY.md'), 'w').write('\n'.join(L) + '\n')
    print('summary:', os.path.join(root, 'ACDC_SUMMARY.md'))
if __name__ == '__main__':
    main(sys.argv[1])
