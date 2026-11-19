import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif', 'axes.unicode_minus': False, 'pdf.fonttype': 42})
from _paths import OUT
from heat_data import ROWSA as _RA, matA, matB, rowsB
LAB = [f'$10^{{-{i}}}$' for i in range(1, 16)]
ROWSA = [r.replace('_', ' ') for r in _RA]

for itf, dt in (('A', 'fp32'), ('A', 'fp16'), ('B', 'fp32'), ('B', 'fp16')):
    mat = matA if itf == 'A' else matB
    oc, og = (mat(dt, 'official', 'cpu'), mat(dt, 'official', 'gpu'))
    uc, ug = (mat(dt, 'ours', 'cpu'), mat(dt, 'ours', 'gpu'))
    assert np.array_equal(np.isnan(uc), np.isnan(ug)) and np.nanmax(np.abs(uc - ug)) == 0 and (np.nanmax(uc) == 0), (itf, dt)
    rows = ROWSA if itf == 'A' else rowsB(dt)
    rk = ~np.isnan(oc).all(axis=1)
    ck = ~np.isnan(oc).all(axis=0)
    rws = [r for r, k in zip(rows, rk) if k]
    lab = [l for l, k in zip(LAB, ck) if k]
    nc = int(ck.sum())
    w = 6.75 if nc > 8 else 4.6
    h = max(1.9, 0.125 * len(rws) + 0.75)
    fg, axs = plt.subplots(1, 3, figsize=(w, h), gridspec_kw={'wspace': 0.035})
    for ax, M0, t in zip(axs, (oc, og, ug), ('official, CPU', 'official, GPU', 'ours, CPU and GPU')):
        M = M0[rk][:, ck]
        im = ax.imshow(M, vmin=0, vmax=100, cmap='Blues', aspect='auto')
        for i in range(M.shape[0]):
            for j in range(nc):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=(3.3 if M[i, j] >= 99.5 else 3.9) if nc > 8 else 4.8, family='DejaVu Sans', stretch='condensed', color='white' if M[i, j] > 55 else 'black')
        ax.set_xticks(range(nc))
        ax.set_xticklabels(lab, rotation=90 if nc > 8 else 0, fontsize=4.6 if nc > 8 else 5.5)
        ax.set_xlabel('true KL (bin lower edge)', fontsize=6.5)
        ax.set_title(f'{dt.upper()}, {t}', fontsize=7)
        ax.tick_params(length=2, width=0.5)
        ax.set_yticks([])
    axs[0].set_yticks(range(len(rws)))
    axs[0].set_yticklabels(rws, fontsize=5.4)
    cb = fg.colorbar(im, ax=axs, shrink=0.6 if len(rws) > 15 else 0.95, pad=0.008, aspect=34 if len(rws) > 15 else 16, fraction=0.03)
    cb.set_label('% of samples with relative KL error $\\geq$ 1%' if len(rws) > 15 else '% with rel. KL error $\\geq$ 1%', fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    out = str(OUT / f'fig_heat_{itf}_{dt}')
    fg.savefig(out + '.pdf', bbox_inches='tight')
    fg.savefig(out + '.png', dpi=200, bbox_inches='tight')
    plt.close(fg)
    d = og - oc
    ok = ~np.isnan(d)
    print(itf, dt, 'cells', int(ok.sum()), '|gpu-cpu|>=1:', int((np.abs(d[ok]) >= 1).sum()), '>=5:', int((np.abs(d[ok]) >= 5).sum()), 'gpu higher by>=1:', int((d[ok] >= 1).sum()), 'lower:', int((d[ok] <= -1).sum()), 'range', round(float(np.nanmin(d)), 1), round(float(np.nanmax(d)), 1))
