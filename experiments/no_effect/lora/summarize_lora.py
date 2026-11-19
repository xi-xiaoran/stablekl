import glob, json, os, sys
import numpy as np
SC = ('S0', 'S1', 'S2', 'ANCHOR')
PATHS = ('bf16', 'fp32', 'logits', 'none')
PLANNED_SEEDS = (0, 1, 2)
DESC = {'S0': 'student = teacher + LoRA, KL only', 'S1': 'W8-initialised base + LoRA, KL (recovery)', 'S2': 'W4 base + LoRA, KL', 'ANCHOR': 'teacher + LoRA, CE + KL anchor (anti-forgetting)'}

def read(root):
    R = {}
    for p in sorted(glob.glob(os.path.join(root, 'task_*', 'runs.jsonl'))):
        for l in open(p):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            R[r['scenario'], r['lr'], r['path'], r['seed']] = r
    return R

def last(r):
    return r['evals'][str(max((int(k) for k in r['evals'])))]

def full(R, sc, lr, pa, seeds=PLANNED_SEEDS):
    return all(((sc, lr, pa, s) in R and R[sc, lr, pa, s]['status'] == 'complete' and (R[sc, lr, pa, s]['steps_done'] == R[sc, lr, pa, s]['steps_planned']) for s in seeds))

def main(root):
    R = read(root)
    lrs = sorted({k[1] for k in R})
    seeds = sorted({k[3] for k in R})
    cells, pairs, flags = ({}, {}, {})
    for sc in SC:
        for lr in lrs:
            for pa in PATHS:
                rs = [r for k, r in R.items() if k[:3] == (sc, lr, pa) and r['status'] == 'complete']
                if not rs:
                    continue
                st = np.array([s for r in rs for s in r['steps']], dtype=float)
                rep, ex = (st[:, 0], st[:, 1])
                pos = (ex > 0) & np.isfinite(rep)
                rel = np.abs(rep[pos] / ex[pos] - 1) if pos.any() else np.array([])
                cells[f'{sc}|{lr}|{pa}'] = dict(n=len(rs), statuses=sorted({r['status'] for k, r in R.items() if k[:3] == (sc, lr, pa)}), kl0=rs[0]['evals']['0']['kl'], kl=[last(r)['kl'] for r in rs], top1=[last(r)['top1_disagree'] for r in rs], ppl=[last(r)['ppl_student'] for r in rs], ppl_teacher=rs[0]['evals']['0'].get('ppl_teacher'), rep_rel_med=float(np.median(rel)) if rel.size else None, rep_abs_max_exact0=float(np.abs(rep[(ex == 0) & np.isfinite(rep)]).max()) if ((ex == 0) & np.isfinite(rep)).any() else None, ce_final=[float(np.mean([s[2] for s in r['steps'][-8:]])) for r in rs], unchanged=sum((r['lora_unchanged'] for r in rs)), ident=sum((bool(r['first_step_logits_identical']) for r in rs)), normB=[r['norm_B'] for r in rs], minutes=float(np.median([r['seconds'] for r in rs]) / 60), mem=max((r['mem_gib_peak'] or 0 for r in rs)), curve=[float(np.median([r['evals'][str(e)]['kl'] for r in rs])) for e in range(len(rs[0]['evals']))])
            for a, b in (('logits', 'bf16'), ('fp32', 'bf16'), ('logits', 'fp32'), ('bf16', 'none'), ('logits', 'none')):
                ps = [(R[sc, lr, a, s], R[sc, lr, b, s]) for s in seeds if (sc, lr, a, s) in R and (sc, lr, b, s) in R and (R[sc, lr, a, s]['status'] == R[sc, lr, b, s]['status'] == 'complete')]
                if not ps:
                    continue
                ka = np.array([last(x)['kl'] for x, _ in ps])
                kb = np.array([last(y)['kl'] for _, y in ps])
                pa_ = np.array([last(x)['ppl_student'] for x, _ in ps])
                pb = np.array([last(y)['ppl_student'] for _, y in ps])
                with np.errstate(divide='ignore', invalid='ignore'):
                    reld = np.where(kb > 0, (ka - kb) / kb, np.nan)
                pairs[f'{sc}|{lr}|{a}_vs_{b}'] = dict(n=len(ps), wins=int((ka < kb).sum()), median_rel_diff=float(np.nanmedian(reld)) if np.isfinite(reld).any() else None, gap_of_medians=float(np.median(ka) / np.median(kb) - 1) if np.median(kb) > 0 else None, ppl_diff_median=float(np.median(pa_ - pb)))
            c_l, c_b = (cells.get(f'{sc}|{lr}|logits'), cells.get(f'{sc}|{lr}|bf16'))
            p = pairs.get(f'{sc}|{lr}|logits_vs_bf16')
            if c_l or c_b:
                if not (full(R, sc, lr, 'logits') and full(R, sc, lr, 'bf16')):
                    flags[f'{sc}|{lr}'] = 'INCOMPLETE'
                elif sc == 'S0':
                    flags[f'{sc}|{lr}'] = bool(c_l['unchanged'] == c_l['n'] and c_b['unchanged'] == 0)
                else:
                    flags[f'{sc}|{lr}'] = bool(p['wins'] == p['n'] == len(PLANNED_SEEDS) and p['gap_of_medians'] is not None and (p['gap_of_medians'] <= -0.05))
    json.dump(dict(cells=cells, pairs=pairs, flags=flags), open(os.path.join(root, 'lora_summary.json'), 'w'), indent=1)
    med = lambda v: f'{np.median(v):.3e} [{min(v):.3e}–{max(v):.3e}]'
    f = lambda x: '—' if x is None else f'{x:.2e}'
    L = ['# LoRA screen (exploratory, Qwen2.5-1.5B-Instruct, 3 seeds): which scenario separates the three KL paths?', '', 'Final = after the last epoch, eval set = WikiText-2 validation; median [min–max] over seeds.  KL = exact FP64 KL(teacher || student) (for ANCHOR: the drift from the base model, i.e. forgetting).', '', '| scenario | lr | path | seeds | eval KL start | eval KL final | top-1 disagree | student PPL | reported-KL rel. err (median) | max |reported| where exact = 0 | adapter unchanged | step-1 logits identical | |B| | min / run | GiB |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k, c in cells.items():
        sc, lr, pa = k.split('|')
        L.append(f"| {sc} | {lr} | {pa} | {c['n']} | {c['kl0']:.3e} | {med(c['kl'])} | {med(c['top1'])} | {med(c['ppl'])} | {f(c['rep_rel_med'])} | {f(c['rep_abs_max_exact0'])} | {c['unchanged']}/{c['n']} | {c['ident']}/{c['n']} | {np.median(c['normB']):.2e} | {c['minutes']:.1f} | {c['mem']:.1f} |")
    L += ['', '## Paired by seed (final eval KL; lower is better)', '', '| scenario | lr | comparison | n | wins of the first | median rel. diff | gap of medians | PPL diff (median) |', '|---|---|---|---|---|---|---|---|']
    for k, p in pairs.items():
        sc, lr, cmp = k.split('|')
        pct = lambda x: '—' if x is None else f'{x:+.1%}'
        L.append(f"| {sc} | {lr} | {cmp.replace('_vs_', ' vs ')} | {p['n']} | {p['wins']} | {pct(p['median_rel_diff'])} | {pct(p['gap_of_medians'])} | {p['ppl_diff_median']:+.4f} |")
    lab = lambda v: 'INCOMPLETE (not all planned seeds / steps done)' if v == 'INCOMPLETE' else 'worth confirming' if v else 'no clear separation'
    L += ['', '## Screening flags (logits path vs BF16 path)', ''] + [f"- {k} ({DESC[k.split('|')[0]]}): **{lab(v)}**" for k, v in flags.items()]
    open(os.path.join(root, 'LORA_SUMMARY.md'), 'w').write('\n'.join(L) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        col = {'bf16': '#c4462d', 'fp32': '#2a6fdb', 'logits': '#2f8f3a', 'none': '#808080'}
        ls = {lrs[0]: '-', lrs[-1]: '--'} if lrs else {}
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
        for j, sc in enumerate(SC):
            for lr in lrs:
                for pa in PATHS:
                    c = cells.get(f'{sc}|{lr}|{pa}')
                    if c:
                        ax[j].plot(range(len(c['curve'])), c['curve'], ls.get(lr, '-'), marker='o', ms=2.5, lw=1.8, color=col[pa], label=f'{pa}, lr {lr:g}')
            ax[j].set_title(f'{sc}: {DESC[sc]}', fontsize=8)
            ax[j].set_xlabel('epoch')
            ax[j].grid(alpha=0.3)
            ax[j].set_yscale('symlog', linthresh=1e-08) if sc in ('S0',) else ax[j].set_yscale('log')
        ax[0].set_ylabel('eval KL(teacher || student)')
        ax[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(root, 'fig_lora_eval_kl.png'), dpi=150)
    except Exception as e:
        print('figure skipped:', e)
    print('summary:', os.path.join(root, 'LORA_SUMMARY.md'))
if __name__ == '__main__':
    main(sys.argv[1])
