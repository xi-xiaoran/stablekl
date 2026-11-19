import math, json, sys
import numpy as np, torch, mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geo_fit as G
mp.mp.dps = 60
NSTART, STEPS, LR = (12, 3000, 0.01)
COL = {'official': '#eb6834', 'C': '#1baf7a', 'D': '#2a78d6', 'true': '#0b0b0b', 'ink2': '#52514e', 'grid': '#e4e3df'}

def starts(pstar, n=NSTART, seed=0):
    r = np.random.default_rng(seed)
    return np.exp(r.uniform(np.log(pstar / 5), np.log(min(5 * pstar, 0.95)), n))

def target(param, pstar, dt):
    th_star = math.log(pstar) - math.log1p(-pstar)
    star = float(torch.tensor(pstar if param == 'probs' else th_star, dtype=dt))
    return (star, star if param == 'probs' else float(G.sig(star)))

def p_seen(th, param, dt):
    return [float(torch.sigmoid(t.to(dt)).double()) if param == 'probs' else float(G.sig(float(t.to(dt)))) for t in th.detach()]

def loss_vec(method, param, direction, th, star, dt):
    from torch.distributions import Geometric, kl_divergence
    if param == 'probs':
        a = torch.sigmoid(th.to(dt))
        b = torch.full_like(a, star)
    else:
        a = th.to(dt)
        b = torch.full_like(a, star)
    P, Q = (a, b) if direction == 'rev' else (b, a)
    if method == 'official':
        mk = (lambda v: Geometric(probs=v)) if param == 'probs' else lambda v: Geometric(logits=v)
        return kl_divergence(mk(P), mk(Q))
    tg = {torch.float16: 'fp16', torch.float32: 'fp32'}[dt]
    if param == 'probs':
        return G.FP.GeometricProbsKLC.apply(P, Q) if method == 'C' else G.FP.GeometricProbsKLC.apply(P.double(), Q.double()).to(dt)
    return G.FL.GeometricLogitsKLC.apply(P, Q, None) if method == 'C' else G.FL.GeometricLogitsKLC.apply(P.double(), Q.double(), tg).to(dt)

def part_A():
    out = []
    lines = []
    for dt in (torch.float16, torch.float32):
        for param in ('probs', 'logits'):
            for direction in ('rev', 'fwd'):
                for pstar in (0.9, 0.3, 0.01, 0.001):
                    star, p_opt = target(param, pstar, dt)
                    p0 = starts(pstar)
                    row = {}
                    for method in ('official', 'C', 'D'):
                        th = torch.tensor(np.log(p0) - np.log1p(-p0), dtype=torch.float32, requires_grad=True)
                        opt = torch.optim.Adam([th], lr=LR)
                        neg = 0
                        failed_at = [None] * NSTART
                        alive = torch.ones(NSTART, dtype=torch.bool)
                        for s in range(STEPS):
                            L = loss_vec(method, param, direction, th, star, dt)
                            neg += int((L.detach()[alive] < 0).sum())
                            opt.zero_grad()
                            L.sum().backward()
                            bad = ~torch.isfinite(th.grad)
                            for k in torch.nonzero(bad & alive).flatten().tolist():
                                failed_at[k] = s
                            alive &= ~bad
                            if not alive.any():
                                break
                            th.grad[~alive] = 0.0
                            keep = th.detach().clone()
                            opt.step()
                            with torch.no_grad():
                                th[~alive] = keep[~alive]
                        ps = p_seen(th, param, dt)
                        rel = [abs(p - p_opt) / p_opt for p in ps]
                        kl = [G.true_kl(p, p_opt) if direction == 'rev' else G.true_kl(p_opt, p) for p in ps]
                        row[method] = dict(rel=rel, kl=kl, neg_frac=neg / (STEPS * NSTART), failed_at=failed_at, nonfinite=any((f is not None for f in failed_at)))
                    rec = dict(dt=str(dt)[-7:], param=param, dir=direction, pstar=pstar, starts=p0.tolist(), **row)
                    out.append(rec)
                    s = f"{rec['dt']} {param:6s} {direction} p*={pstar:<6g} |"
                    for m in ('official', 'C', 'D'):
                        r = row[m]
                        ok = [k for k in range(NSTART) if r['failed_at'][k] is None]
                        nfail = NSTART - len(ok)
                        s += f" {m}: rel err median {(np.median([r['rel'][k] for k in ok]) if ok else float('nan')):.1e} max {(max([r['rel'][k] for k in ok]) if ok else float('nan')):.1e} KL max {(max([r['kl'][k] for k in ok]) if ok else float('nan')):.1e} loss<0 {r['neg_frac']:.0%}{(f' | {nfail} start(s) non-finite gradient at step ' + str([f for f in r['failed_at'] if f is not None]) if nfail else '')} |"
                    print(s, flush=True)
                    lines.append(s)
    json.dump(out, open('geo_fit2_A.json', 'w'), indent=1)
    open('geo_fit2_A.txt', 'w').write('\n'.join(lines) + '\n')

def exact_grad_rev_probs(p, q):
    p, q = (mp.mpf(p), mp.mpf(q))
    dkl_dp = -mp.log((1 - p) / (1 - q)) / p ** 2
    return float(dkl_dp * p * (1 - p))

def trajectories(method, pst, dt, p0):
    stars = [target('probs', ps, dt) for ps in pst]
    th = torch.tensor(np.log(p0) - np.log1p(-p0), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([th], lr=LR)
    T = np.zeros((STEPS + 1, *p0.shape))
    Gr = np.zeros((STEPS, *p0.shape))
    Lrep = np.zeros((STEPS, p0.shape[0]))
    T[0] = th.detach().numpy()
    for s in range(STEPS):
        Ls = [loss_vec(method, 'probs', 'rev', th[:, j], stars[j][0], dt) for j in range(2)]
        Lrep[s] = sum((L.detach().double().numpy() for L in Ls))
        opt.zero_grad()
        sum((L.sum() for L in Ls)).backward()
        Gr[s] = th.grad.numpy()
        opt.step()
        T[s + 1] = th.detach().numpy()
    return (T, Gr, Lrep, [st[1] for st in stars])

def true_loss_theta(th, popt, dt):
    out = np.zeros(th.shape[:-1])
    for idx in np.ndindex(out.shape):
        ps = [float(torch.sigmoid(torch.tensor(th[idx + (j,)], dtype=torch.float32).to(dt)).double()) for j in range(2)]
        out[idx] = sum((G.true_kl(ps[j], popt[j]) for j in range(2)))
    return out

def part_BC(tag, dt, pst):
    rng = np.random.default_rng(1)
    p0 = np.stack([np.exp(rng.uniform(np.log(ps / 5), np.log(5 * ps), NSTART)) for ps in pst], axis=1)
    res = {m: trajectories(m, pst, dt, p0) for m in ('official', 'D')}
    popt = res['D'][3]
    th_opt = [math.log(p) - math.log1p(-p) for p in popt]
    worst = int(np.argmax(true_loss_theta(res['official'][0][-1], popt, dt)))
    tagdt = str(dt)[-7:]

    def style(a, xl='θ₁ (logit of p₁)', yl='θ₂ (logit of p₂)'):
        a.set_xlabel(xl, color=COL['ink2'])
        a.set_ylabel(yl, color=COL['ink2'])
        a.tick_params(colors=COL['ink2'], labelsize=8)
        [sp.set_color(COL['grid']) for sp in a.spines.values()]

    def contours(a, X, Y, Z, n=12):
        lv = np.logspace(np.log10(max(Z[Z > 0].min(), 1e-15)), np.log10(Z.max()), n)
        a.contour(X, Y, np.maximum(Z, lv[0]), levels=lv, colors=COL['grid'], linewidths=0.8)
    fig, ax = plt.subplots(2, 2, figsize=(11, 10.5), constrained_layout=True)
    allT = np.concatenate([res[m][0].reshape(-1, 2) for m in res])
    pad = 0.3
    x0, x1 = (allT[:, 0].min() - pad, allT[:, 0].max() + pad)
    y0, y1 = (allT[:, 1].min() - pad, allT[:, 1].max() + pad)
    gx, gy = np.meshgrid(np.linspace(x0, x1, 70), np.linspace(y0, y1, 70))
    Z = true_loss_theta(np.stack([gx, gy], -1), popt, dt)
    for c, m in enumerate(('official', 'D')):
        a = ax[0, c]
        T = res[m][0]
        contours(a, gx, gy, Z)
        for i in range(NSTART):
            a.plot(T[:, i, 0], T[:, i, 1], color=COL[m], lw=0.9, alpha=0.75)
            a.plot(T[0, i, 0], T[0, i, 1], 'o', ms=3.5, color=COL['ink2'])
            a.plot(T[-1, i, 0], T[-1, i, 1], 'o', ms=4.5, mfc='white', mec=COL[m], mew=1.3)
        a.plot(*th_opt, marker='*', ms=13, color=COL['true'], mec='white', mew=0.8, ls='none', label='target')
        a.plot([], [], 'o', ms=3.5, color=COL['ink2'], ls='none', label='start (same 12 for both)')
        a.plot([], [], 'o', ms=4.5, mfc='white', mec=COL[m], ls='none', label='end (step 3000)')
        a.set_title(('official torch' if m == 'official' else 'frozen fix D') + ': 12 starts, 3000 steps', fontsize=10.5)
        style(a)
        a.legend(fontsize=8, frameon=False, loc='upper right')
        a = ax[1, c]
        seg = T[-500:, worst]
        cx, cy = (seg[:, 0].mean(), seg[:, 1].mean())
        d = max(np.ptp(seg[:, 0]), np.ptp(seg[:, 1]), 1e-06) * 0.75
        X, Y = np.meshgrid(np.linspace(cx - d, cx + d, 50), np.linspace(cy - d, cy + d, 50))
        contours(a, X, Y, true_loss_theta(np.stack([X, Y], -1), popt, dt), 10)
        a.plot(seg[:, 0], seg[:, 1], color=COL[m], lw=0.6, alpha=0.5)
        sc = a.scatter(seg[:, 0], seg[:, 1], c=np.arange(STEPS - 500, STEPS), cmap='Blues' if m == 'D' else 'Oranges', s=9, zorder=3)
        if abs(th_opt[0] - cx) < d and abs(th_opt[1] - cy) < d:
            a.plot(*th_opt, marker='*', ms=13, color=COL['true'], mec='white', ls='none')
        a.set_xlim(cx - d, cx + d)
        a.set_ylim(cy - d, cy + d)
        fig.colorbar(sc, ax=a, shrink=0.8, label='step')
        path = np.linalg.norm(np.diff(seg, axis=0), axis=-1).sum()
        dist = np.linalg.norm(seg.mean(0) - np.array(th_opt))
        a.set_title(f'start {worst}, last 500 steps (tight zoom): path length {path:.2g}, net move {np.linalg.norm(seg[-1] - seg[0]):.2g},\ndistance of this region from the target {dist:.2g}', fontsize=9.5)
        style(a)
    fig.suptitle(f'Fit two Geometric targets (p* = {pst[0]:g}, {pst[1]:g}) by minimising KL(Geo(θ) ‖ Geo(p*)), {tagdt}, Adam lr 1e-2; grey = contours of the true loss', fontsize=10.5)
    fig.savefig(f'fig_{tag}_trajectories.png', dpi=130)
    plt.close(fig)
    i = worst
    steps = np.arange(STEPS)
    tl = {m: true_loss_theta(res[m][0][:-1, i], popt, dt) for m in res}
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for c, m in enumerate(('official', 'D')):
        for r, sl in enumerate((slice(None), slice(-100, None))):
            a = ax[r, c]
            a.scatter(steps[sl], res[m][2][sl, i], s=4 if r == 0 else 10, color=COL[m], label='loss value the method reports', zorder=3)
            a.plot(steps[sl], tl[m][sl], color=COL['true'], lw=1.3, ls='--', label='true loss at the same parameters')
            a.set_yscale('symlog', linthresh=1e-08)
            a.axhline(0, color=COL['ink2'], lw=0.6)
            a.set_title(('official torch' if m == 'official' else 'frozen fix D') + f', start {i}: ' + ('all steps' if r == 0 else 'last 100 steps'), fontsize=10)
            style(a, 'step', 'loss (symlog; below the line = negative)')
            if r == 0:
                a.legend(fontsize=8, frameon=False, loc='center right')
    fig.suptitle(f'Loss the method reports against the true loss, {tagdt}, p* = {pst[0]:g}, {pst[1]:g}', fontsize=10.5)
    fig.savefig(f'fig_{tag}_loss.png', dpi=130)
    plt.close(fig)
    T, Gr = (res['official'][0], res['official'][1])
    rows = []

    def tgrad(th2):
        ps = [float(torch.sigmoid(torch.tensor(th2[j], dtype=torch.float32).to(dt)).double()) for j in range(2)]
        return np.array([exact_grad_rev_probs(ps[j], popt[j]) for j in range(2)])
    for s in range(STEPS):
        gt = tgrad(T[s, i])
        go = Gr[s, i]
        ng, no = (np.linalg.norm(gt), np.linalg.norm(go))
        rows.append((s, T[s, i, 0], T[s, i, 1], go[0], go[1], gt[0], gt[1], no / ng if ng > 0 else float('inf'), float(go @ gt / (ng * no)) if ng > 0 and no > 0 else float('nan')))
    R = np.array(rows)
    late = R[R[:, 0] >= 500]
    order = np.argsort(np.nan_to_num(late[:, 8], nan=2))
    sel = []
    seen = set()
    for k in order:
        key = (round(late[k, 1], 4), round(late[k, 2], 4))
        if key not in seen:
            seen.add(key)
            sel.append(late[k])
        if len(sel) == 6:
            break
    sel = np.array(sel)
    with open(f'geo_fit2_C_{tag}.txt', 'w') as f:
        f.write(f'start {i}; official trajectory; gradient w.r.t. theta; true = exact derivative of the exact KL at the probabilities the loss sees\n')
        f.write(f'steps >= 500: median cosine(official, true) {np.nanmedian(late[:, 8]):.3f}; fraction with cosine < 0 {np.mean(late[:, 8] < 0):.1%}; median |official|/|true| {np.nanmedian(late[:, 7]):.2e}\n6 lowest-cosine steps at distinct positions:\n')
        f.write('step | theta1 theta2 | official grad | true grad | |off|/|true| | cosine\n')
        for r in sel:
            f.write(f'{int(r[0])} | {r[1]:.5f} {r[2]:.5f} | ({r[3]:+.3e}, {r[4]:+.3e}) | ({r[5]:+.3e}, {r[6]:+.3e}) | {r[7]:.2e} | {r[8]:+.3f}\n')
    ends = np.array([res['official'][0][-1, k] for k in range(NSTART)])
    span = max(np.abs(res['official'][0][-500:] - np.array(th_opt)).max() * 1.15, 0.0001)
    n = 15
    g1, g2 = np.meshgrid(np.linspace(th_opt[0] - span, th_opt[0] + span, n), np.linspace(th_opt[1] - span, th_opt[1] + span, n))
    th = torch.tensor(np.stack([g1.ravel(), g2.ravel()], 1), dtype=torch.float32, requires_grad=True)
    Lg = sum((loss_vec('official', 'probs', 'rev', th[:, j], target('probs', pst[j], dt)[0], dt).sum() for j in range(2)))
    Lg.backward()
    GO = th.grad.numpy()
    GT = np.array([tgrad(t) for t in th.detach().numpy()])
    cosg = np.einsum('ij,ij->i', GO, GT) / (np.linalg.norm(GO, axis=1) * np.linalg.norm(GT, axis=1) + 1e-300)
    fig, a = plt.subplots(figsize=(8.5, 8), constrained_layout=True)
    X, Y = np.meshgrid(np.linspace(th_opt[0] - span, th_opt[0] + span, 60), np.linspace(th_opt[1] - span, th_opt[1] + span, 60))
    contours(a, X, Y, true_loss_theta(np.stack([X, Y], -1), popt, dt))
    un = lambda V: -V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-300)
    a.quiver(g1.ravel(), g2.ravel(), *un(GT).T, color=COL['true'], angles='xy', scale=n * 1.1, width=0.0035, label='true −gradient (unit length)')
    a.quiver(g1.ravel(), g2.ravel(), *un(GO).T, color=COL['official'], angles='xy', scale=n * 1.1, width=0.0035, alpha=0.9, label='official −gradient (unit length)')
    zero = np.linalg.norm(GO, axis=1) == 0
    a.plot(g1.ravel()[zero], g2.ravel()[zero], 'x', ms=5, color=COL['official'], ls='none', label='official gradient exactly 0')
    a.plot(ends[:, 0], ends[:, 1], 'o', ms=6, mfc='white', mec=COL['official'], mew=1.5, ls='none', label='official end points (12 starts)')
    a.plot(*th_opt, marker='*', ms=15, color=COL['true'], mec='white', ls='none', label='target')
    a.set_xlim(X.min(), X.max())
    a.set_ylim(Y.min(), Y.max())
    style(a)
    a.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3)
    a.set_title(f'Descent direction around the optimum, {tagdt}: official (orange) vs true (black)\ngrid {n}×{n}: official points within 90° of the true direction at {np.mean(cosg > 0):.0%} of points, within 25° at {np.mean(cosg > 0.906):.0%}', fontsize=10)
    fig.savefig(f'fig_{tag}_gradients.png', dpi=130)
    plt.close(fig)
    with open(f'geo_fit2_C_{tag}.txt', 'a') as f:
        f.write(f'gradient field {n}x{n} around the optimum (half-width {span:.3g} in theta): cosine(official, true) > 0 at {np.mean(cosg > 0):.1%}, > 0.906 (25 deg) at {np.mean(cosg > 0.906):.1%}, official exactly 0 at {zero.mean():.1%}\n')
    summ = {}
    for m in res:
        Tm = res[m][0]
        ends = true_loss_theta(Tm[-1], popt, dt)
        ps_end = np.array([[float(torch.sigmoid(torch.tensor(Tm[-1, k, j], dtype=torch.float32).to(dt)).double()) for j in range(2)] for k in range(NSTART)])
        rel = np.abs(ps_end - popt) / np.array(popt)
        seg = np.linalg.norm(np.diff(Tm[-500:], axis=0), axis=-1).sum(0)
        net = np.linalg.norm(Tm[-1] - Tm[-500], axis=-1)
        rad = np.linalg.norm(Tm[-500:] - np.array(th_opt), axis=-1).max(0)
        summ[m] = dict(true_loss_end_median=float(np.median(ends)), true_loss_end_max=float(ends.max()), rel_err_p_max=float(rel.max()), last500_path_len_median=float(np.median(seg)), last500_net_disp_median=float(np.median(net)), last500_max_dist_to_opt_median=float(np.median(rad)))
    json.dump(dict(pstar=pst, dtype=str(dt), starts=p0.tolist(), worst_start=i, summary=summ), open(f'geo_fit2_B_{tag}.json', 'w'), indent=1)
    print(tag, json.dumps(summ, indent=1), flush=True)
    print(open(f'geo_fit2_C_{tag}.txt').read(), flush=True)
if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('A', 'all'):
        part_A()
    if what in ('B', 'all'):
        part_BC('fp16', torch.float16, (0.01, 0.002))
        part_BC('fp32', torch.float32, (0.01, 0.001))
