import json, numpy as np, torch, mpmath as mp
from matplotlib.lines import Line2D
from style import *
mp.mp.dps = 50
rng = np.random.default_rng(0)
from _paths import OUT, RESULTS
import stablekl
NPF = {'fp16': np.float16, 'fp32': np.float32, 'fp64': np.float64}

def panel_a(dt, xs, n=3000):
    f = NPF[dt]
    res = []
    for x in xs:
        lp = np.exp(rng.uniform(np.log(0.5), np.log(2.0), n)).astype(f)
        lq = (lp.astype(np.float64) * (1 + x)).astype(f)
        keep = lq != lp
        lp, lq = (lp[keep], lq[keep])
        if len(lp) < 20:
            res.append((np.nan,) * 5)
            continue
        true = np.array([float(mp.mpf(float(b)) / mp.mpf(float(a)) - 1 - mp.log(mp.mpf(float(b)) / mp.mpf(float(a)))) for a, b in zip(lp, lq)])
        P = torch.distributions.Exponential(torch.from_numpy(lp))
        Q = torch.distributions.Exponential(torch.from_numpy(lq))
        off = torch.distributions.kl_divergence(P, Q).double().numpy()
        ours = stablekl.kl_divergence(P, Q).double().numpy()
        res.append((np.median(true), np.median(np.abs(off - true) / true), np.median(np.abs(ours - true) / true), np.mean(off <= 0), np.mean(ours <= 0)))
    return np.array(res)
xs = np.logspace(-0.3, -7.2, 56)
A32 = panel_a('fp32', xs)
A16 = panel_a('fp16', np.logspace(-0.3, -3.3, 28))
print('fp32 official rel err first>=1:', A32[np.argmax(A32[:, 1] >= 1), 0], 'max ours', np.nanmax(A32[:, 2]), '| fp16 first>=1:', A16[np.argmax(A16[:, 1] >= 1), 0], 'max ours', np.nanmax(A16[:, 2]))
print('share <=0 official fp32 (last 10 pts):', A32[-10:, 3].round(2), 'ours:', A32[-10:, 4].max())
runs = [json.loads(l) for l in open(RESULTS / 'distillation' / 'runs.jsonl')]

def curve(V, dt, method, lr=0.01):
    rs = [r for r in runs if r['V'] == V and r['dt'] == dt and (r['method'] == method) and (r['opt'] == 'adam') and (r['lr'] == lr) and (r.get('kick', 0) == 0)]
    st = sorted(map(int, rs[0]['log']))
    assert all((r['steps_done'] == 300 for r in rs))
    Y = np.array([[100 * r['log'][str(s)]['top1_disagree'] for s in st] for r in rs])
    return (np.array(st), np.median(Y, 0), Y.min(0), Y.max(0), len(rs))
fig, (a, b) = plt.subplots(1, 2, figsize=(6.75, 2.3), gridspec_kw={'wspace': 0.3})
for A, ls, lo_ in ((A32, '-', 0.0), (A16, DASH, 6.2e-05)):
    ok = ~np.isnan(A[:, 0]) & (A[:, 0] >= lo_)
    a.plot(A[ok, 0], 100 * A[ok, 1], ls=ls, color=OFF, lw=1.5)
    a.plot(A[ok, 0], 100 * np.maximum(A[ok, 2], 1e-10), ls=ls, color=OURS, lw=1.5)
a.set_xscale('log')
a.set_yscale('log')
a.invert_xaxis()
a.set_ylim(1e-07, 30000.0)
a.axhline(100, color=MUTED, lw=0.6, ls=':')
a.text(2e-15, 160, '100% error', fontsize=7, color=MUTED, va='bottom', ha='right')
a.set_xlabel('true KL  (distributions closer $\\rightarrow$)')
a.set_ylabel('relative error of returned KL (%)')
h = [Line2D([], [], color=OFF, lw=1.5), Line2D([], [], color=OFF, lw=1.5, ls=DASH), Line2D([], [], color=OURS, lw=1.5), Line2D([], [], color=OURS, lw=1.5, ls=DASH)]
a.legend(h, ['official, FP32', 'official, FP16', 'ours, FP32', 'ours, FP16'], fontsize=7, loc='center right', handlelength=2.2, labelspacing=0.3, bbox_to_anchor=(1.0, 0.42))
a.set_title('(a) The returned KL is wrong when it is small', loc='left')
a.minorticks_off()
V = 151936
for dt, ls in (('fp32', '-'), ('fp16', DASH)):
    s, m, lo, hi, n1 = curve(V, dt, 'kl_div')
    b.plot(np.r_[0, s], np.r_[0, m], ls=ls, color=OFF, lw=1.5)
    b.fill_between(np.r_[0, s], np.r_[0, lo], np.r_[0, hi], color=OFF, alpha=0.13, lw=0)
    s, m2, lo2, hi2, n2 = curve(V, dt, 'logitsA')
    b.plot(np.r_[0, s], np.r_[0, m2], ls=ls, color=OURS, lw=1.5)
    print('panel b', dt, 'official median', m.round(1), 'range last', lo[-1].round(1), hi[-1].round(1), 'seeds', n1, '| ours max', hi2.max(), 'seeds', n2)
b.set_xscale('symlog', linthresh=1)
b.set_xlim(0, 330)
b.set_ylim(-3, 100)
b.set_xticks([0, 1, 10, 100, 300])
b.set_xticklabels(['0', '1', '10', '100', '300'])
b.minorticks_off()
b.set_xlabel('training step')
b.set_ylabel('student $\\neq$ teacher, top-1 (%)')
b.legend(h, ['official, FP32', 'official, FP16', 'ours, FP32 (exactly 0)', 'ours, FP16 (exactly 0)'], fontsize=7, loc='upper left', handlelength=2.2, labelspacing=0.3)
b.set_title('(b) A student that starts as its teacher drifts away', loc='left')
fig.savefig(OUT / 'fig1_overview.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT / 'fig1_overview.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
