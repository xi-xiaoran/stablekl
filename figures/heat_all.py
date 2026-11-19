import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif', 'axes.unicode_minus': False, 'pdf.fonttype': 42})
BINS = ['B_c'] + [f'B_{i}' for i in range(1, 15)]
LAB = [f'$10^{{-{i}}}$' for i in range(1, 16)]
TITLE = {'A': 'official', 'B32': 'official formula in FP32', 'B64': 'official formula in FP64', 'C': 'rewrite', 'D64': 'ours'}
SHORT = dict(TITLE, B32='official in FP32', B64='official in FP64')

def draw(Ms, names, rows, dt, tag, out, ref):
    rk = ~np.isnan(ref).all(axis=1)
    ck = ~np.isnan(ref).all(axis=0)
    rws = [r for r, k in zip(rows, rk) if k]
    lab = [l for l, k in zip(LAB, ck) if k]
    nc = int(ck.sum())
    fg, axs = plt.subplots(1, 2, figsize=(6.75 if nc > 8 else 4.3, max(2.3, 4.5 * (len(rws) + 6) / 39)), gridspec_kw={'wspace': 0.2})
    for ax, m, M0 in zip(axs, names, Ms):
        M = M0[rk][:, ck]
        im = ax.imshow(M, vmin=0, vmax=100, cmap='Blues', aspect='auto')
        for i in range(M.shape[0]):
            for j in range(nc):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=(3.6 if M[i, j] >= 99.5 else 4.3) if nc > 8 else 4.8, family='DejaVu Sans', stretch='condensed', color='white' if M[i, j] > 55 else 'black')
        ax.set_xticks(range(nc))
        ax.set_xticklabels(lab, rotation=90 if nc > 8 else 0, fontsize=5 if nc > 8 else 5.5)
        ax.set_xlabel('true KL (bin lower edge)', fontsize=7)
        ax.set_title(f'{dt.upper()}, {(SHORT if nc <= 8 else TITLE)[m]} ({tag})', fontsize=7.5 if nc > 8 else 6.5)
        ax.tick_params(length=2, width=0.5)
    axs[0].set_yticks(range(len(rws)))
    axs[0].set_yticklabels(rws, fontsize=5.8)
    axs[1].set_yticks([])
    cb = fg.colorbar(im, ax=axs, shrink=0.6 if len(rws) > 15 else 0.9, pad=0.015)
    cb.set_label('% of samples with relative KL error $\\geq$ 1%', fontsize=6.5 if len(rws) > 15 else 5.5)
    cb.ax.tick_params(labelsize=6)
    fg.savefig(out + '.pdf', bbox_inches='tight')
    fg.savefig(out + '.png', dpi=220, bbox_inches='tight')
    plt.close(fg)
from _paths import OUT
from heat_data import ROWSA, matA, matB, rowsB
NAME = {'A': 'official', 'B32': 'official_fp32', 'B64': 'official_fp64', 'C': 'rewrite', 'D64': 'ours'}
RES = {}
for dt, hi in (('fp32', 'B64'), ('fp16', 'B32')):
    for itf, mat, rows in (('A', matA, [r.replace('_', ' ') for r in ROWSA]), ('B', matB, rowsB(dt))):
        ref = mat(dt, 'official', 'cpu')
        for grp, ms in (('official', ('A', hi)), ('repair', ('C', 'D64'))):
            for dev in ('cpu', 'gpu'):
                Ms = [mat(dt, NAME[m], dev) for m in ms]
                draw(Ms, ms, rows, dt, dev.upper(), str(OUT / f'{itf}_{dt}_{grp}_{dev}'), ref)
                RES[itf, dt, grp, dev] = Ms
for (itf, dt, grp, dev), Ms in sorted(RES.items()):
    if dev == 'gpu':
        C = RES[itf, dt, grp, 'cpu']
        for k in (0, 1):
            d = Ms[k] - C[k]
            ok = ~np.isnan(d)
            print(itf, dt, grp, 'panel', k, 'cells', int(ok.sum()), 'max cpu', np.nanmax(C[k]).round(1), 'max gpu', np.nanmax(Ms[k]).round(1), '| |gpu-cpu|>=1:', int((np.abs(d[ok]) >= 1).sum()), '>=5:', int((np.abs(d[ok]) >= 5).sum()), 'range', np.nanmin(d).round(1), np.nanmax(d).round(1))
