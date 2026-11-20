import glob, json, os, sys, numpy as np

def main(root):
    L = ['# Batch design with Beta posteriors: official FP32 KL vs the Rewrite (FP32)', '', "T = 100,000 observations, K = 10, 5 seeds; ranges over the seeds.  'x exact' = relative to the policy with the exact EIG and the same batch size.", '', '| start | batch size | rounds | official: wrong rounds | official: mean share of the best EIG lost per round | official: sum of posterior variances x exact | official: largest posterior variance x exact | official: fewest pulls of an arm (exact policy) | Rewrite: wrong rounds | Rewrite: sum of posterior variances x exact | verdict |', '|---|---|---|---|---|---|---|---|---|---|---|']
    out = {}
    for st in ('cold', 'warm'):
        runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, f'batch_{st}_seed*.json')))]
        for m in runs[0]['meta']['ms']:
            E = [d['results'][f'{m}|exact'] for d in runs]
            O = [d['results'][f'{m}|official'] for d in runs]
            W = [d['results'][f'{m}|rewrite'] for d in runs]
            wf = [o['wrong_frac'] for o in O]
            vr = [o['post_var_sum'] / e['post_var_sum'] for o, e in zip(O, E)]
            vm = [o['post_var_max'] / e['post_var_max'] for o, e in zip(O, E)]
            rg = [o['mean_step_regret'] for o in O]
            ww = [w['wrong_frac'] for w in W]
            wv = [w['post_var_sum'] / e['post_var_sum'] for w, e in zip(W, E)]
            ok1 = all((x >= 0.1 for x in wf)) and all((x <= 0.001 for x in ww))
            ok2 = all((x >= 1.5 for x in vr)) and all((x <= 1.01 for x in wv))
            v = 'SIGNAL' if ok1 and ok2 else 'error only' if ok1 else 'no signal'
            out[f'{st}|{m}'] = dict(verdict=v, wrong=wf, var=vr, varmax=vm, regret=rg, rewrite_wrong=ww, rewrite_var=wv, seeds_var_ge_1p5=int(sum((x >= 1.5 for x in vr))), n_seeds=len(runs))
            r = lambda x, f='{:.2f}': f.format(min(x)) + ' – ' + f.format(max(x))
            L.append(f"| {st} | {m} | {O[0]['rounds']} | {r(wf, '{:.0%}')} | {r(rg, '{:.1%}')} | {r(vr)} | {r(vm)} | {min((min(o['pulls']) for o in O))} ({min((min(e['pulls']) for e in E))}) | {max(ww):.0%} | {max(wv):.4f} | {v} |")
    json.dump(out, open(os.path.join(root, 'batch_summary.json'), 'w'), indent=1)
    open(os.path.join(root, 'BATCH_SUMMARY.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
if __name__ == '__main__':
    main(sys.argv[1])
