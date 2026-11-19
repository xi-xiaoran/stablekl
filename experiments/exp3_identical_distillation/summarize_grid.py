import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_gpu import read_jsonl
from distill_grid import DTS, OPT_LR, METHODS
NAMES = {'kl_div': 'official F.kl_div (softmax)', 'kl_div_lt': 'official F.kl_div, log_target', 'categorical': 'official Categorical(logits)', 'logitsA': 'ours: fixed Categorical(logits)'}

def cells(rows):
    C = {}
    for r in rows:
        C.setdefault((r['V'], r['dt'], r['opt'], float(r['lr']), r['method']), []).append(r)
    out = {}
    for k, rs in C.items():
        ok = [r for r in rs if r['status'] == 'complete']
        last = [r['log'][str(r['steps'])] for r in ok]
        t = [100 * x['top1_disagree'] for x in last]
        w = [x['rel_dW'] for x in last]
        out[k] = dict(n=len(rs), nonfinite=len(rs) - len(ok), top1_med=float(np.median(t)) if t else None, top1_min=min(t) if t else None, top1_max=max(t) if t else None, rel_dW_med=float(np.median(w)) if w else None, moved=sum((x > 0 for x in w)))
    return out

def fmt(c):
    if c is None:
        return '—'
    if c['top1_med'] is None:
        return f"all {c['n']} non-finite"
    s = '0 (no move)' if c['moved'] == 0 else f"{c['top1_med']:.1f}% [{c['top1_min']:.1f}–{c['top1_max']:.1f}] ({c['rel_dW_med']:.0e})"
    if 0 < c['moved'] < c['n'] - c['nonfinite']:
        s += f", moved {c['moved']}/{c['n'] - c['nonfinite']}"
    if c['nonfinite']:
        s += f", {c['nonfinite']} non-finite"
    return s

def main(out):
    rows = read_jsonl(os.path.join(out, 'runs.jsonl'))
    C = cells(rows)
    st = json.load(open(os.path.join(out, 'status.json')))
    VS = sorted({r['V'] for r in rows})
    json.dump({'|'.join(map(str, k)): v for k, v in C.items()}, open(os.path.join(out, 'grid_cells.json'), 'w'), indent=1)
    L = ['# Identical-initialisation distillation: full grid', '', f"Runs: {st['recorded']} / {st['planned']} planned; by status {st['by_status']}." + (f"  Runs shared with the frozen job 8589145: {st['compare_with_frozen']['identical']} / {st['compare_with_frozen']['common']} bitwise identical." if 'compare_with_frozen' in st else ''), '', "Cell: median top-1 disagreement with the teacher after the last step [min–max over seeds] (median relative weight change); '0 (no move)' = the weights did not change at all in any seed.  Non-finite runs are counted separately.  BF16 for the logits path is exploratory.", '']
    L += ['## Official F.kl_div vs ours (top-1 disagreement, median over 10 seeds)', '', '| V | dtype | ' + ' | '.join((f'{o} {lr:g}' for o, lr in OPT_LR)) + ' |', '|---|---|' + '---|' * len(OPT_LR)]
    for V in VS:
        for dt in DTS:
            cs = []
            for o, lr in OPT_LR:
                a, b = (C.get((V, dt, o, lr, 'kl_div')), C.get((V, dt, o, lr, 'logitsA')))
                fa = '—' if a is None else 'nf' if a['top1_med'] is None else f"{a['top1_med']:.1f}"
                fb = '—' if b is None else '0' if b['moved'] == 0 and (not b['nonfinite']) else f"{b['top1_med']:.1f}" if b['top1_med'] is not None else 'nf'
                cs.append(f'{fa} / **{fb}**')
            L.append(f'| {V} | {dt} | ' + ' | '.join(cs) + ' |')
    L += ['', 'Each cell: official / **ours**; nf = all seeds non-finite.', '']
    for o, lr in OPT_LR:
        L += [f'## {o} lr {lr:g}', '', '| V | dtype | ' + ' | '.join((NAMES[m] for m in METHODS)) + ' |', '|---|---|' + '---|' * len(METHODS)]
        for V in VS:
            for dt in DTS:
                L.append(f'| {V} | {dt} | ' + ' | '.join((fmt(C.get((V, dt, o, lr, m))) for m in METHODS)) + ' |')
        L.append('')
    ours = [c for k, c in C.items() if k[4] == 'logitsA']
    L += ['## Completion', '', f"- cells: {len(C)}; ours (logits path) did not move in {sum((c['moved'] == 0 and c['nonfinite'] == 0 for c in ours))} / {len(ours)} cells", '']
    open(os.path.join(out, 'GRID_SUMMARY.md'), 'w').write('\n'.join(L))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cols = {'fp32': '#2a6fdb', 'fp16': '#d6604d', 'bf16': '#4d9221'}
        fig, ax = plt.subplots(len(VS), 2, figsize=(9, 2.6 * len(VS)), sharey='row', squeeze=False)
        for i, V in enumerate(VS):
            for j, opt in enumerate(('adam', 'sgd')):
                lrs = [lr for o, lr in OPT_LR if o == opt]
                a = ax[i][j]
                for dt in DTS:
                    y = [C.get((V, dt, opt, lr, 'kl_div'), {}).get('top1_med') for lr in lrs]
                    a.plot(lrs, [np.nan if v is None else v for v in y], '-o', color=cols[dt], lw=2, ms=5, label=f'official F.kl_div, {dt}')
                    z = [C.get((V, dt, opt, lr, 'logitsA'), {}).get('top1_med') for lr in lrs]
                    a.plot(lrs, [np.nan if v is None else v for v in z], '--', color=cols[dt], lw=1.5, alpha=0.8, label=f'ours, {dt}')
                a.set_xscale('log')
                a.set_title(f"V = {V}, {('Adam' if opt == 'adam' else 'SGD')}", fontsize=10)
                a.grid(alpha=0.25, lw=0.5)
                if j == 0:
                    a.set_ylabel('top-1 disagreement (%)')
                if i == len(VS) - 1:
                    a.set_xlabel('learning rate')
        ax[0][0].legend(fontsize=7, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'fig_grid_lr.png'), dpi=150)
    except Exception as e:
        print('figure skipped:', e)
    print('summary:', os.path.join(out, 'GRID_SUMMARY.md'))
if __name__ == '__main__':
    main(sys.argv[1])
