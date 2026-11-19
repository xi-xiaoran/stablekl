import glob, json, os, sys, numpy as np

def main(root):
    R = {}
    for p in sorted(glob.glob(os.path.join(root, '*_seed*.json'))):
        d = json.load(open(p))
        R.setdefault(d['meta']['setting'], []).append(d)
    L = ['# Sequential Bayesian experiment with Beta posteriors: official FP32 KL vs the Rewrite (FP32)', '']
    out = {}
    for s, runs in sorted(R.items()):
        m = runs[0]['meta']
        L += [f"## Setting {s}: K = {m['K']}, T = {m['T']}, {('warm start' if m['warm'] else 'cold start')}, {len(runs)} seeds", '', '| seed | policy | wrong steps | first wrong step | EIG values <= 0 or not finite | max rel. error of the EIG | information gained / exact policy | sum of posterior variances / exact policy | largest posterior variance / exact policy | squared error of the estimates / exact policy | pulls min / max |', '|---|---|---|---|---|---|---|---|---|---|---|']
        ok1 = ok2 = True
        rows = []
        for d in sorted(runs, key=lambda d: d['seed']):
            e = d['results']['exact']
            for pol in ('official', 'rewrite', 'strict', 'exact'):
                r = d['results'].get(pol)
                if r is None:
                    continue
                row = dict(seed=d['seed'], policy=pol, wrong_frac=r['wrong_frac'], first_wrong=r['first_wrong'], bad=r['nonpositive_or_nonfinite'], relmax=r['max_rel_err_eig'], info=r['info'] / e['info'], var=r['post_var_sum'] / e['post_var_sum'], varmax=r['post_var_max'] / e['post_var_max'], sq=r['sq_err'] / e['sq_err'], pulls=(min(r['pulls']), max(r['pulls'])))
                rows.append(row)
                L.append(f"| {row['seed']} | {pol} | {row['wrong_frac']:.2%} | {row['first_wrong']} | {row['bad']} | {row['relmax']:.1e} | {row['info']:.4f} | {row['var']:.4f} | {row['varmax']:.4f} | {row['sq']:.3f} | {row['pulls'][0]} / {row['pulls'][1]} |")
            o, w = (d['results']['official'], d['results']['rewrite'])
            ok1 &= o['wrong_frac'] >= 0.1 and w['wrong_frac'] <= 0.001
            ok2 &= o['post_var_sum'] >= 1.5 * e['post_var_sum'] and w['post_var_sum'] <= 1.01 * e['post_var_sum']
        v = 'SIGNAL' if ok1 and ok2 else 'error only' if ok1 else 'no signal'
        out[s] = dict(verdict=v, rows=rows)
        g = lambda pol, k: [r[k] for r in rows if r['policy'] == pol]
        L += ['', f"Official over the seeds: wrong steps {min(g('official', 'wrong_frac')):.1%} to {max(g('official', 'wrong_frac')):.1%}; information gained {min(g('official', 'info')):.4f} to {max(g('official', 'info')):.4f} of the exact policy; sum of posterior variances {min(g('official', 'var')):.4f} to {max(g('official', 'var')):.4f} x; largest posterior variance {min(g('official', 'varmax')):.3f} to {max(g('official', 'varmax')):.3f} x.  Rewrite: wrong steps at most {max(g('rewrite', 'wrong_frac')):.3%}; sum of posterior variances at most {max(g('rewrite', 'var')):.6f} x.", '', f'**Verdict (fixed before the run): {v}**', '']
    json.dump(out, open(os.path.join(root, 'beta_ig_summary.json'), 'w'), indent=1)
    open(os.path.join(root, 'BETA_IG_SUMMARY.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
if __name__ == '__main__':
    main(sys.argv[1])
