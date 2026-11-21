import json, numpy as np
from style import *
from _paths import OUT, RESULTS
R = [json.loads(l) for l in open(RESULTS / 'distillation' / 'runs.jsonl')]

def cell(V, dt, opt, method):
    lrs = sorted({r['lr'] for r in R if r['V'] == V and r['dt'] == dt and (r['opt'] == opt) and (r['method'] == method) and (r.get('kick', 0) == 0)})
    out = []
    for lr in lrs:
        rs = [r for r in R if r['V'] == V and r['dt'] == dt and (r['opt'] == opt) and (r['method'] == method) and (r['lr'] == lr) and (r.get('kick', 0) == 0)]
        out.append(np.median([100 * r['log']['300']['top1_disagree'] for r in rs]))
    return (np.array(lrs), np.array(out), len(rs))
MK = {'fp32': 'o', 'fp16': '^', 'bf16': 's'}
fig, axs = plt.subplots(2, 3, figsize=(6.75, 3.1), gridspec_kw={'wspace': 0.28, 'hspace': 0.62})
mx = 0
for i, opt in enumerate(('adam', 'sgd')):
    for j, V in enumerate((1024, 32768, 151936)):
        ax = axs[i, j]
        for dt in ('fp32', 'fp16', 'bf16'):
            x, y, n = cell(V, dt, opt, 'kl_div')
            ax.plot(x, y, ls=LS[dt], color=OFF, lw=1.3, marker=MK[dt], ms=3, mfc='white' if dt != 'fp32' else OFF, mec=OFF, mew=0.7)
            x2, y2, n2 = cell(V, dt, opt, 'logitsA')
            ax.plot(x2, y2, ls='-', color=OURS, lw=1.3, zorder=5)
            mx = max(mx, y2.max())
            assert n == 10 and n2 == 10
        ax.set_xscale('log')
        ax.set_ylim(-4, 100)
        ax.set_title(f"{('Adam' if opt == 'adam' else 'SGD')}, vocabulary {V:,}", fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel('learning rate', fontsize=7.5)
        if j == 0:
            ax.set_ylabel('student $\\neq$ teacher (%)', fontsize=7.5)
h = [plt.Line2D([], [], color=OFF, lw=1.3, ls=LS[d], marker=MK[d], ms=3, mfc='white' if d != 'fp32' else OFF, mec=OFF, mew=0.7) for d in ('fp32', 'fp16', 'bf16')] + [plt.Line2D([], [], color=OURS, lw=1.3)]
axs[0, 0].legend(h, ['official, FP32', 'official, FP16', 'official, BF16', 'ours (all three: 0)'], fontsize=6.4, loc='upper left', handlelength=2.6, labelspacing=0.25)
print('largest value of ours over all cells:', mx)
fig.savefig(OUT / 'fig_grid_lr.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT / 'fig_grid_lr.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
