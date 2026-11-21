"""Figure 2: error of the official KL and of the official formula in the next higher precision, versus the true KL."""
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from style import *
from _paths import OUT, RESULTS
m = pd.read_csv(RESULTS / 'interface_a' / 'unit_metrics.csv.gz', low_memory=False)
m = m[m.region == 'R0']
BINS = ['B_c'] + [f'B_{i}' for i in range(1, 15)]
X = np.arange(15)
PAN = [('KL_REL1', 'KL off by $\\geq$1%'), ('KL_NEG', 'KL negative'), ('G_REL_L2_1:PQ', 'gradient off by $\\geq$1%'), ('G_COS_99:PQ', 'gradient cosine $<$0.99')]
Y = {}
for me, _ in PAN:
    for dt, hi in (('fp32', 'B64'), ('fp16', 'B32')):
        for k, meth in (('off', 'A'), ('hi', hi)):
            g = m[(m.metric == me) & (m.dtype == dt) & (m.method == meth)].groupby('group')[['events', 'denominator']].sum().reindex(BINS)
            Y[me, dt, k] = np.where(g.denominator >= 300, 100 * g.events / g.denominator, np.nan)
print('fp32', np.round(Y['KL_REL1', 'fp32', 'off'], 1)); print('fp16', np.round(Y['KL_REL1', 'fp16', 'off'], 1))
fs, lw = 8, 1.4
fig, axs = plt.subplots(1, 4, figsize=(6.75, 1.92), sharey=True, gridspec_kw={'wspace': 0.07})
st = {('fp32', 'hi'): dict(color=PROM, ls='-', lw=lw * 0.85, zorder=2), ('fp16', 'hi'): dict(color=PROM, ls=DASH, lw=lw * 0.85, zorder=1),
      ('fp32', 'off'): dict(color=OFF, ls='-', lw=lw, zorder=4), ('fp16', 'off'): dict(color=OFF, ls=DASH, lw=lw, zorder=3)}
for ax, (me, title) in zip(axs, PAN):
    for (dt, k), s in st.items():
        ax.plot(X, Y[me, dt, k], **s)
    ax.set_title(title, pad=4, fontsize=fs + 0.3)
    ax.set_xticks(X[::4]); ax.set_xticklabels([f'$10^{{-{i}}}$' for i in range(1, 16, 4)])
    ax.set_xlim(-0.6, 14.6); ax.set_ylim(-3, 103); ax.set_yticks([0, 25, 50, 75, 100])
    ax.xaxis.set_minor_locator(MultipleLocator(1)); ax.yaxis.set_minor_locator(MultipleLocator(12.5))
# zoom on the two promoted curves of the last panel, which stay below 2%
ax = axs[3]; me = PAN[3][0]
ins = ax.inset_axes([0.10, 0.55, 0.34, 0.36])
for (dt, k), s in st.items():
    if k == 'hi': ins.plot(X, Y[me, dt, k], **s)
ins.grid(False); ins.set_xlim(7.5, 14.5); ins.set_ylim(-0.15, 2.5)
ins.set_xticks([9, 13]); ins.set_xticklabels(['$10^{-10}$', '$10^{-14}$']); ins.set_yticks([0, 1, 2]); ins.set_yticklabels(['0', '1%', '2%'])
ins.tick_params(labelsize=fs - 2.4, length=1.8, pad=1.5, top=False, right=False)
for sp in ins.spines.values(): sp.set_linewidth(0.5); sp.set_color('#555555')
yv = Y[me, 'fp16', 'hi']; j = int(np.nanargmax(yv))
ins.annotate(f'{yv[j]:.1f}%', (j, yv[j]), xytext=(-3, -1), textcoords='offset points', ha='right', va='top', fontsize=fs - 2.2)
ax.plot([7.5, 14.5, 14.5, 7.5, 7.5], [-2, -2, 4.5, 4.5, -2], color='#555555', lw=0.5, zorder=6, clip_on=False)
axs[0].set_ylabel('% of samples')
fig.supxlabel('true KL divergence', fontsize=fs, y=-0.085)
order = [('fp32', 'off'), ('fp16', 'off'), ('fp32', 'hi'), ('fp16', 'hi')]
h = [Line2D([], [], **{k: v for k, v in st[o].items() if k != 'zorder'}) for o in order]
axs[1].legend(h, ['official, FP32', 'official, FP16', 'in FP64 (FP32 inputs)', 'in FP32 (FP16 inputs)'], fontsize=fs - 1.6, loc='upper left', handlelength=1.9, handletextpad=0.5, labelspacing=0.28, borderaxespad=0.3)
fig.savefig(OUT / 'fig_err_vs_kl.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT / 'fig_err_vs_kl.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
