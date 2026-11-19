import numpy as np, pandas as pd
from style import *
from _paths import OUT, RESULTS
m = pd.read_csv(RESULTS / 'interface_a' / 'unit_metrics.csv.gz', low_memory=False)
m = m[m.region == 'R0']
BINS = ['B_c'] + [f'B_{i}' for i in range(1, 15)]
X = np.arange(15)
PAN = [('KL_REL1', 'KL off by $\\geq$ 1%'), ('KL_NEG', 'KL negative'), ('G_REL_L2_1:PQ', 'gradient off by $\\geq$ 1%'), ('G_COS_99:PQ', 'gradient cosine $<$ 0.99')]
fig, axs = plt.subplots(1, 4, figsize=(6.75, 1.85), sharey=True, gridspec_kw={'wspace': 0.08})
for ax, (me, title) in zip(axs, PAN):
    for dt, hi in (('fp32', 'B64'), ('fp16', 'B32')):
        for meth, col, lw in ((hi, OFF2, 1.1), ('A', OFF, 1.5)):
            g = m[(m.metric == me) & (m.dtype == dt) & (m.method == meth)].groupby('group')[['events', 'denominator']].sum().reindex(BINS)
            y = np.where(g.denominator >= 300, 100 * g.events / g.denominator, np.nan)
            ax.plot(X, y, LS[dt], color=col, lw=lw, marker='o', ms=1.8)
            if me == 'KL_REL1' and meth == 'A':
                print(dt, np.round(y, 1))
    ax.set_title(title, fontsize=7.5)
    ax.set_xticks(X[::4])
    ax.set_xticklabels([f'$10^{{-{i}}}$' for i in range(1, 16, 4)], fontsize=6.5)
    ax.set_xlabel('true KL (bin lower edge)', fontsize=7)
    ax.set_ylim(-3, 103)
axs[0].set_ylabel('% of samples', fontsize=7)
h = [plt.Line2D([], [], color=OFF, lw=1.5), plt.Line2D([], [], color=OFF2, lw=1.1), plt.Line2D([], [], color=INK, lw=1.2, ls='-'), plt.Line2D([], [], color=INK, lw=1.2, ls='--')]
axs[1].legend(h, ['official', 'official formula,\nnext higher precision', 'FP32 inputs', 'FP16 inputs'], frameon=False, fontsize=5.6, loc='upper left', handlelength=2.2, labelspacing=0.3, bbox_to_anchor=(0.0, 1.0))
fig.savefig(OUT / 'fig_err_vs_kl.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_err_vs_kl.png', dpi=200, bbox_inches='tight')
