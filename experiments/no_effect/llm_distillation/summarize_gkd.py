import glob, itertools, json, math, os, sys
import numpy as np
LOSSES = ('bf16', 'fp32', 'logits')
STARTS = ('S0', 'S1', 'S2')
NAME = {'bf16': 'TRL loss on BF16 logits (DeepSpeed-like)', 'fp32': 'TRL loss on FP32 logits (DDP-like)', 'logits': 'ours: logits path (BF16)'}

def read(root):
    R = {}
    for p in sorted(glob.glob(os.path.join(root, 'task_*', 'runs.jsonl'))):
        for l in open(p):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            R[r['start'], r['loss'], r['seed']] = r
    return R

def sign_test_p(k, n):
    d = abs(k - n / 2)
    return min(1.0, sum((math.comb(n, j) for j in range(n + 1) if abs(j - n / 2) >= d - 1e-12)) / 2 ** n)

def wilcoxon_p(x):
    x = [v for v in x if v != 0]
    n = len(x)
    if n == 0:
        return 1.0
    a = np.abs(np.array(x))
    order = np.argsort(a)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    W = sum((r for r, v in zip(ranks, x) if v > 0))
    mu = ranks.sum() / 2
    cnt = sum((1 for s in itertools.product((0, 1), repeat=n) if abs(sum((r for r, b in zip(ranks, s) if b)) - mu) >= abs(W - mu) - 1e-09))
    return cnt / 2 ** n

def boot_ci(x, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = [np.median(rng.choice(x, len(x))) for _ in range(B)]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))

def cell_stats(rs):
    ok = [r for r in rs if r['status'] == 'complete']
    if not ok:
        return None
    last = str(max((int(k) for k in ok[0]['evals'])))
    fin = lambda key: [r['evals'][last][key] for r in ok]
    st = np.array([s for r in ok for s in r['steps']])
    rep, ex = (st[:, 0], st[:, 1])
    pos = ex > 0
    rel = np.abs(rep[pos] / ex[pos] - 1) if pos.any() else np.array([])
    cos0 = [r['grad_probes']['0']['cosine'] for r in ok]
    cosL = [r['grad_probes'][last]['cosine'] for r in ok]
    clean = lambda v: [c for c in v if c is not None]
    return dict(n=len(ok), statuses={s: sum((r['status'] == s for r in rs)) for s in {r['status'] for r in rs}}, first_step_logits_identical=sum((bool(r['first_step_logits_identical']) for r in ok)), weights_unchanged=sum((bool(r['weights_unchanged']) for r in ok)), eval_kl_init=float(np.median([r['evals']['0']['kl'] for r in ok])), eval_kl=fin('kl'), top1=fin('top1_disagree'), ppl=fin('ppl_student'), ppl_teacher=ok[0]['evals']['0'].get('ppl_teacher'), drift=[r['drift'][last][0] for r in ok], reported_rel_err_median=float(np.median(rel)) if rel.size else None, reported_rel_err_p95=float(np.percentile(rel, 95)) if rel.size else None, reported_abs_err_max_where_exact_zero=float(np.abs(rep[~pos]).max()) if (~pos).any() else None, grad_cos_step0=clean(cos0), grad_cos_final=clean(cosL), grad_norm_step0=[r['grad_probes']['0']['method_norm'] for r in ok], mem_peak_gib=max(((r['mem_gib_peak'] or [0, 0])[0] for r in ok)), minutes=float(np.median([r['seconds'] for r in ok]) / 60), curve=[float(np.median([r['evals'][str(e)]['kl'] for r in ok])) for e in range(int(last) + 1)])

def paired(R, start, a, b, seeds):
    xs = [(R[start, a, s], R[start, b, s]) for s in seeds if (start, a, s) in R and (start, b, s) in R and (R[start, a, s]['status'] == R[start, b, s]['status'] == 'complete')]
    if not xs:
        return None
    last = str(max((int(k) for k in xs[0][0]['evals'])))
    ka = np.array([p['evals'][last]['kl'] for p, _ in xs])
    kb = np.array([q['evals'][last]['kl'] for _, q in xs])
    k1a = np.array([p['evals']['1']['kl'] for p, _ in xs])
    k1b = np.array([q['evals']['1']['kl'] for _, q in xs])
    wins = int((ka < kb).sum())
    n = len(xs)
    reld = (ka - kb) / kb
    return dict(n=n, wins=wins, sign_p=sign_test_p(wins, n), wilcoxon_p=wilcoxon_p(list(ka - kb)), median_rel_diff=float(np.median(reld)), ci95_median_rel_diff=boot_ci(reld), rel_gap_of_medians=float(np.median(ka) / np.median(kb) - 1), median_diff_epoch1=float(np.median(k1a - k1b)), median_diff_final=float(np.median(ka - kb)))

def main(root):
    R = read(root)
    seeds = sorted({k[2] for k in R})
    full = list(range(10))
    S = {f'{st}|{lo}': cell_stats([r for k, r in R.items() if k[0] == st and k[1] == lo]) for st in STARTS for lo in LOSSES}
    P = {f'{st}|{a}_vs_{b}': paired(R, st, a, b, full) for st in STARTS for a, b in (('logits', 'bf16'), ('fp32', 'bf16'), ('logits', 'fp32'))}
    complete = lambda st, lo: S[f'{st}|{lo}'] is not None and S[f'{st}|{lo}']['n'] == 10
    V = {}
    for st in ('S1', 'S2'):
        if complete(st, 'bf16') and complete(st, 'logits'):
            V[f'layer1_{st}'] = bool(S[f'{st}|bf16']['reported_rel_err_median'] >= 0.1 and S[f'{st}|logits']['reported_rel_err_median'] <= 0.01)
    if all((complete('S0', lo) for lo in ('bf16', 'logits'))):
        V['layer2_S0'] = bool(S['S0|logits']['first_step_logits_identical'] == 10 and S['S0|logits']['weights_unchanged'] == 10 and (S['S0|bf16']['weights_unchanged'] == 0))
    for st in ('S1', 'S2'):
        p = P[f'{st}|logits_vs_bf16']
        if p and p['n'] == 10 and complete(st, 'bf16') and complete(st, 'logits'):
            V[f'layer3_{st}'] = bool(p['wins'] >= 9 and p['sign_p'] <= 0.025 and (-p['rel_gap_of_medians'] >= 0.05) and (abs(p['median_diff_final']) >= abs(p['median_diff_epoch1'])))
    json.dump(dict(cells=S, paired=P, verdicts=V, seeds_seen=seeds), open(os.path.join(root, 'summary.json'), 'w'), indent=1)
    f = lambda x, d=3: '—' if x is None else f'{x:.{d}e}' if isinstance(x, float) else str(x)
    med = lambda v: f'{np.median(v):.3e} [{min(v):.3e}–{max(v):.3e}]' if v else '—'
    L = ['# Summary: real-LLM distillation on the TRL GKD numerical path (Qwen2.5-1.5B-Instruct, forward KL, T = 1.0)', '', 'Simulated numerical paths with a source basis (accelerate 1.15: DeepSpeed keeps BF16 logits; DDP / single GPU converts the outputs to FP32); not a DeepSpeed run.  Final = after the last epoch; values: median [min–max] over seeds; eval set = WikiText-2 validation, independent of training.', '']
    L += ['## Cells', '', '| start | loss | seeds | eval KL start | eval KL final | top-1 disagree final | student PPL final | rel. weight change | reported-loss rel. err (median / p95) | max |reported| where exact = 0 | param-grad cosine step 0 / final (median) | param-grad norm step 0 (median) | weights unchanged | step-1 logits identical | peak GiB | min / run |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for st in STARTS:
        for lo in LOSSES:
            c = S[f'{st}|{lo}']
            if c is None:
                L.append(f'| {st} | {lo} | 0 | | | | | | | | | | | | | |')
                continue
            cs = lambda v: f'{np.median(v):.6f}' if v else '—'
            L.append(f"| {st} | {lo} | {c['n']} | {c['eval_kl_init']:.3e} | {med(c['eval_kl'])} | {med(c['top1'])} | {med(c['ppl'])} | {med(c['drift'])} | {f(c['reported_rel_err_median'])} / {f(c['reported_rel_err_p95'])} | {f(c['reported_abs_err_max_where_exact_zero'])} | {cs(c['grad_cos_step0'])} / {cs(c['grad_cos_final'])} | {np.median(c['grad_norm_step0']):.2e} | {c['weights_unchanged']}/{c['n']} | {c['first_step_logits_identical']}/{c['n']} | {c['mem_peak_gib']:.1f} | {c['minutes']:.1f} |")
    L += ['', '## Paired comparisons of the final eval KL (lower is better; same seed = same data order)', '', '| start | comparison | n | wins of the first | sign test p (two-sided) | Wilcoxon p | median paired rel. diff [95% bootstrap CI] | gap of medians | median diff epoch 1 → final |', '|---|---|---|---|---|---|---|---|---|']
    for k, p in P.items():
        if p is None:
            continue
        st, cmp = k.split('|')
        L.append(f"| {st} | {cmp.replace('_vs_', ' vs ')} | {p['n']} | {p['wins']} | {p['sign_p']:.4f} | {p['wilcoxon_p']:.4f} | {p['median_rel_diff']:+.2%} [{p['ci95_median_rel_diff'][0]:+.2%}, {p['ci95_median_rel_diff'][1]:+.2%}] | {p['rel_gap_of_medians']:+.2%} | {p['median_diff_epoch1']:.2e} → {p['median_diff_final']:.2e} |")
    L += ['', '## Pre-registered verdicts (evaluated only when all 10 seeds of the cells involved are complete)', '']
    L += [f"- {k}: **{('met' if v else 'not met')}**" for k, v in V.items()] or ['- (not yet evaluable)']
    open(os.path.join(root, 'GKD_SUMMARY.md'), 'w').write('\n'.join(L) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        col = {'bf16': '#c4462d', 'fp32': '#2a6fdb', 'logits': '#2f8f3a'}
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for j, st in enumerate(STARTS):
            for lo in LOSSES:
                c = S[f'{st}|{lo}']
                if c:
                    ax[j].plot(range(len(c['curve'])), c['curve'], marker='o', ms=3, lw=2, color=col[lo], label=NAME[lo])
            ax[j].set_title({'S0': 'S0: student = teacher', 'S1': 'S1: RTN W8 student', 'S2': 'S2: RTN W4 student'}[st])
            ax[j].set_xlabel('epoch')
            ax[j].set_yscale('symlog', linthresh=1e-08) if st == 'S0' else ax[j].set_yscale('log')
            ax[j].grid(alpha=0.3)
        ax[0].set_ylabel('eval KL(teacher || student), nats / token')
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(root, 'fig_p2_eval_kl.png'), dpi=150)
    except Exception as e:
        print('figure skipped:', e)
    print('summary:', os.path.join(root, 'GKD_SUMMARY.md'))
if __name__ == '__main__':
    main(sys.argv[1])
