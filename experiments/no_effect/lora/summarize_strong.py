import glob, json, os, sys
import numpy as np
PATHS = ('bf16', 'fp32', 'logits')
PLANNED_SEEDS = (0, 1, 2)

def read(root):
    R = {}
    for p in sorted(glob.glob(os.path.join(root, 'task_*', 'runs.jsonl'))):
        for l in open(p):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            R[r['tag'], r['path'], r['seed']] = r
    return R

def last(r):
    return r['evals'][str(max((int(k) for k in r['evals'])))]

def full(R, tag, pa):
    return all(((tag, pa, s) in R and R[tag, pa, s]['status'] == 'complete' and (R[tag, pa, s]['steps_done'] == R[tag, pa, s]['steps_planned']) for s in PLANNED_SEEDS))

def order(tag):
    kind, v = tag.split('_', 1)
    return (kind, float(v))

def main(root):
    R = read(root)
    tags = sorted({k[0] for k in R}, key=order)
    cells, verdicts = ({}, {})
    for tag in tags:
        for pa in PATHS:
            rs = [R[tag, pa, s] for s in PLANNED_SEEDS if (tag, pa, s) in R and R[tag, pa, s]['status'] == 'complete']
            if not rs:
                continue
            ne = min((len(r['evals']) for r in rs))
            cells[f'{tag}|{pa}'] = dict(n=len(rs), seeds=[r['seed'] for r in rs], kl0=[r['evals']['0']['kl'] for r in rs], kl=[last(r)['kl'] for r in rs], klmin=[min((e['kl'] for e in r['evals'].values())) for r in rs], top1=[last(r)['top1_disagree'] for r in rs], ppl=[last(r)['ppl_student'] for r in rs], ppl_teacher=rs[0]['evals']['0'].get('ppl_teacher'), ce_final=[float(np.mean([s[2] for s in r['steps'][-8:]])) for r in rs], train_kl_final=[float(np.mean([s[1] for s in r['steps'][-8:]])) for r in rs], gn1=[r['steps'][0][3] for r in rs], ident=sum((bool(r['first_step_logits_identical']) for r in rs)), normB0=[r['norm_B_init'] for r in rs], normB=[r['norm_B'] for r in rs], minutes=float(np.median([r['seconds'] for r in rs]) / 60), curve=[float(np.median([r['evals'][str(e)]['kl'] for r in rs])) for e in range(ne)])
        cl, cb = (cells.get(f'{tag}|logits'), cells.get(f'{tag}|bf16'))
        if not (cl or cb):
            continue
        if not (full(R, tag, 'logits') and full(R, tag, 'bf16')):
            verdicts[tag] = dict(status='INCOMPLETE')
            continue
        kl_l, kl_b, k0 = (np.array(cl['kl']), np.array(cb['kl']), np.array(cl['kl0']))
        if tag.startswith('P_'):
            ret = bool(np.all(kl_l <= 0.1 * k0)) if np.all(k0 > 0) else None
            flo = bool(np.all(kl_b >= 3 * kl_l))
            verdicts[tag] = dict(status='complete', logits_returns=ret, official_floors=flo, signal=bool(ret and flo), logits_final_over_start=[float(a / b) if b > 0 else None for a, b in zip(kl_l, k0)], bf16_final_over_logits_final=[float(a / b) if b > 0 else None for a, b in zip(kl_b, kl_l)])
        else:
            gap = float(np.median(kl_l) / np.median(kl_b) - 1) if np.median(kl_b) > 0 else None
            wins = int((kl_l < kl_b).sum())
            verdicts[tag] = dict(status='complete', wins=wins, gap_of_medians=gap, signal=bool(wins == len(PLANNED_SEEDS) and gap is not None and (gap <= -0.2)), ppl_diff_median=float(np.median(np.array(cl['ppl']) - np.array(cb['ppl']))), ce_diff_median=float(np.median(np.array(cl['ce_final']) - np.array(cb['ce_final']))))
    json.dump(dict(cells=cells, verdicts=verdicts), open(os.path.join(root, 'strong_summary.json'), 'w'), indent=1)
    med = lambda v: f'{np.median(v):.3e} [{min(v):.3e}–{max(v):.3e}]'
    L = ['# Strong-constraint LoRA screen (exploratory, Qwen2.5-1.5B-Instruct, lr 1e-4, 3 seeds)', '', 'Eval set = WikiText-2 validation; KL = exact FP64 KL(teacher || student); median [min–max] over seeds.  P_<s>: S0 with B ~ N(0, s^2), KL only.  A_<lambda>: CE + lambda x KL.', '', '| task | path | seeds | eval KL start | eval KL final | eval KL min over epochs | top-1 disagree | student PPL (eval) | train CE (last 8 steps) | train KL (last 8 steps) | step-1 grad norm | step-1 logits identical | |B| start -> final | min / run |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k, c in cells.items():
        tag, pa = k.split('|')
        L.append(f"| {tag} | {pa} | {c['n']} | {med(c['kl0'])} | {med(c['kl'])} | {med(c['klmin'])} | {med(c['top1'])} | {med(c['ppl'])} | {np.median(c['ce_final']):.4f} | {np.median(c['train_kl_final']):.3e} | {np.median(c['gn1']):.3e} | {c['ident']}/{c['n']} | {np.median(c['normB0']):.2e} -> {np.median(c['normB']):.2e} | {c['minutes']:.1f} |")
    L += ['', '## Median eval KL by epoch (0 = start)', '']
    for k, c in cells.items():
        L.append(f'- {k}: ' + ', '.join((f'{x:.2e}' for x in c['curve'])))
    L += ['', '## Verdicts (fixed before the run)', '']
    for tag, v in verdicts.items():
        if v['status'] != 'complete':
            L.append(f'- {tag}: **INCOMPLETE** (not all planned seeds / steps done)')
            continue
        if tag.startswith('P_'):
            fr = lambda xs: ', '.join(('—' if x is None else f'{x:.3g}' for x in xs))
            L.append(f"- {tag}: logits returns = {v['logits_returns']} (final / start per seed: {fr(v['logits_final_over_start'])}); official floors = {v['official_floors']} (bf16 final / logits final per seed: {fr(v['bf16_final_over_logits_final'])}) -> **{('SIGNAL' if v['signal'] else 'no signal')}**")
        else:
            g = '—' if v['gap_of_medians'] is None else f"{v['gap_of_medians']:+.1%}"
            L.append(f"- {tag}: logits beats bf16 on final eval KL in {v['wins']}/3 seeds, gap of medians {g}, eval PPL diff {v['ppl_diff_median']:+.4f}, train CE diff {v['ce_diff_median']:+.4f} -> **{('SIGNAL' if v['signal'] else 'no signal')}**")
    open(os.path.join(root, 'STRONG_SUMMARY.md'), 'w').write('\n'.join(L) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        col = {'bf16': '#c4462d', 'fp32': '#2a6fdb', 'logits': '#2f8f3a'}
        fig, ax = plt.subplots(1, max(1, len(tags)), figsize=(3.6 * max(1, len(tags)), 3.6), squeeze=False)
        for j, tag in enumerate(tags):
            for pa in PATHS:
                c = cells.get(f'{tag}|{pa}')
                if c:
                    ax[0][j].plot(range(len(c['curve'])), c['curve'], marker='o', ms=2.5, lw=1.8, color=col[pa], label=pa)
            ax[0][j].set_title(tag, fontsize=9)
            ax[0][j].set_xlabel('epoch')
            ax[0][j].set_yscale('symlog', linthresh=1e-07)
            ax[0][j].grid(alpha=0.3)
        ax[0][0].set_ylabel('eval KL(teacher || student)')
        ax[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(root, 'fig_strong.png'), dpi=150)
    except Exception as e:
        print('figure skipped:', e)
    print('summary:', os.path.join(root, 'STRONG_SUMMARY.md'))
if __name__ == '__main__':
    main(sys.argv[1])
