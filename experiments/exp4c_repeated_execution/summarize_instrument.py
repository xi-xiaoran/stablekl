import glob, json, os, sys
import numpy as np

def rows(root, pat):
    out = []
    for p in sorted(glob.glob(os.path.join(root, pat))):
        out += [json.loads(l) for l in open(p) if l.strip()]
    return out

def main(root):
    R, M = (rows(root, 'R_*.jsonl'), rows(root, 'M*.jsonl'))
    S = dict(R=[], M=[], order=[])
    L = ['# KL as a measuring instrument (exploratory, Qwen2.5-1.5B-Instruct)', '']
    if R:
        L += ['## Part R: the same logits, settings that cannot change the true value', '', 'Reported mean per-token KL(base || quantised) over all settings (rows per call 1 / 16 / 128 / 512 / all, reductions, row order, CPU).', '', '| forward | variant | true KL | computed in | method | settings | min | max | spread / true | worst rel. error | negative | zero | non-finite | verdict |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r in R:
            for key, vals in r['reported'].items():
                c, m = key.split('|')
                v = np.array([x for x in vals.values() if x is not None], float)
                bad = sum((x is None for x in vals.values()))
                ref = r['ref']
                spread = float((v.max() - v.min()) / ref) if v.size else None
                worst = float(np.abs(v / ref - 1).max()) if v.size else None
                neg, zero = (int((v < 0).sum()), int((v == 0).sum()))
                verdict = 'not reproducible' if bad or neg or zero or (spread is not None and spread >= 0.1) else 'reproducible' if spread <= 0.01 and worst <= 0.01 else 'reproducible but biased' if spread <= 0.01 else 'intermediate'
                S['R'].append(dict(fwd=r['fwd'], variant=r['variant'], ref=ref, compute=c, method=m, n=len(vals), spread=spread, worst=worst, negative=neg, zero=zero, nonfinite=bad, verdict=verdict))
                f = lambda x, p='{:+.3e}': '—' if x is None else p.format(x)
                L.append(f"| {r['fwd']} | {r['variant']} | {ref:.3e} | {c} | {m} | {len(vals)} | {f(float(v.min()) if v.size else None)} | {f(float(v.max()) if v.size else None)} | {f(spread, '{:.1%}')} | {f(worst, '{:.1%}')} | {neg} | {zero} | {bad} | {verdict} |")
        L += ['', '### Does the ranking of the quantised variants depend on the setting?', '', 'For each setting (applied to every variant), the variants are ordered by the reported KL and compared with the true order.', '', '| forward | computed in | method | settings | settings with a wrong or undefined order |', '|---|---|---|---|---|']
        for fwd in sorted({r['fwd'] for r in R}):
            rs = [r for r in R if r['fwd'] == fwd]
            true = [r['variant'] for r in sorted(rs, key=lambda r: r['ref'])]
            for key in rs[0]['reported']:
                sets = list(rs[0]['reported'][key])
                wrong = 0
                for s in sets:
                    vv = [(r['reported'][key].get(s), r['variant']) for r in rs]
                    if any((x is None for x, _ in vv)) or len({x for x, _ in vv}) < len(vv) or [n for _, n in sorted(vv)] != true:
                        wrong += 1
                c, m = key.split('|')
                S['order'].append(dict(fwd=fwd, compute=c, method=m, settings=len(sets), wrong=wrong))
                L.append(f'| {fwd} | {c} | {m} | {len(sets)} | {wrong} |')
    if M:
        L += ['', '## Part M: the same weights executed in two ways', '', "P = first configuration (the 'rollout' side for the sampled estimators), Q = second.  Full-distribution methods: 128 rows per call, batchmean.", '', '| P | Q | true KL | max logit diff | top-1 differs | method (computed in) | reported | rel. error | verdict |', '|---|---|---|---|---|---|---|---|---|']
        for r in M:
            if 'skipped' in r:
                L.append(f"| {r['P']} | {r['Q']} | — | — | — | (pair skipped: {r['skipped']}) | — | — | — |")
                continue
            ref = r['ref']
            e = r['estimators']
            items = [(k.split('|')[1] + ' (' + k.split('|')[0] + ')', v, ref) for k, v in r['reported'].items()]
            items += [('k1 sampled (fp32)', e['k1_fp32'], e['k1_fp64']), ('k3 official sampled (fp32)', e['k3_official_fp32'], e['k3_fp64']), ('k3 fixed sampled (fp32)', e['k3_fixed_fp32'], e['k3_fp64'])]
            for name, v, target in items:
                if ref == 0 and target == 0:
                    rel = None if v is None else 0.0 if v == 0 else float('inf')
                else:
                    rel = None if v is None or target == 0 else v / target - 1
                bad = v is None or (target > 0 and v <= 0) or rel is None or (abs(rel) >= 0.1)
                verdict = 'cannot measure' if bad else 'measures' if abs(rel) <= 0.01 else 'intermediate'
                if name.startswith('k1'):
                    verdict += ' (numerical only; k1 itself is dominated by sampling noise)'
                S['M'].append(dict(P=r['P'], Q=r['Q'], ref=ref, method=name, value=v, target=target, rel=rel, verdict=verdict))
                L.append(f"| {r['P']} | {r['Q']} | {ref:.3e} | {r['max_abs_logit_diff']:.2e} | {r['top1_disagree']:.4f} | {name} | {('non-finite' if v is None else f'{v:+.3e}')} | {('—' if rel is None else f'{rel:+.1%}')} | {verdict} |")
        L += ['', 'For the sampled estimators the target is the same estimator on the same sampled tokens in FP64 (numerical error only); the true KL column is the full-distribution value.']
    json.dump(S, open(os.path.join(root, 'instrument_summary.json'), 'w'), indent=1)
    open(os.path.join(root, 'INSTRUMENT_SUMMARY.md'), 'w').write('\n'.join(L) + '\n')
    print('summary:', os.path.join(root, 'INSTRUMENT_SUMMARY.md'))
if __name__ == '__main__':
    main(sys.argv[1])
