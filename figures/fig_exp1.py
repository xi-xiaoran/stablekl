import sys, os, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'exp1_geometric_fit'))
import geo_fit2 as F
from style import *
from _paths import OUT
dt, pst = (torch.float16, (0.01, 0.002))
rng = np.random.default_rng(1)
p0 = np.stack([np.exp(rng.uniform(np.log(ps / 5), np.log(5 * ps), F.NSTART)) for ps in pst], axis=1)
res = {m: F.trajectories(m, pst, dt, p0) for m in ('official', 'D')}
popt = res['D'][3]
steps = np.unique(np.round(np.logspace(0, np.log10(F.STEPS), 70)).astype(int))
steps = np.r_[0, steps]
TL = {m: np.array([F.true_loss_theta(res[m][0][s], popt, dt) for s in steps]) for m in res}
np.savez(OUT / 'exp1_curves.npz', steps=steps, official=TL['official'], ours=TL['D'])
for m in TL:
    print(m, 'end median', np.median(TL[m][-1]), 'end max', TL[m][-1].max(), 'end min', TL[m][-1].min())
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for m, col, lab in (('official', OFF, 'official'), ('D', OURS, 'ours')):
    Y = np.maximum(TL[m], 1e-09)
    x = np.maximum(steps, 0.7)
    for i in range(Y.shape[1]):
        ax.plot(x, Y[:, i], color=col, lw=0.5, alpha=0.35)
    ax.plot(x, np.median(Y, 1), color=col, lw=1.6, label=lab)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Adam step')
ax.set_ylabel('true KL to the target')
ax.set_xlim(0.7, 3000)
ax.legend(fontsize=7.5, loc='lower left')
ax.minorticks_off()
fig.savefig(OUT / 'fig_exp1_loss.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT / 'fig_exp1_loss.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
