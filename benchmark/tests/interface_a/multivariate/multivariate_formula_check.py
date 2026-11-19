from __future__ import annotations
import random
import sys
from decimal import Decimal, localcontext
from fractions import Fraction as F
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gmath
import numpy as np
from klbench.ref.configs import CONFIGS
from klbench.ref.configs.contbern import bernoulli_numbers
PREC = 70
rng = random.Random(20261109)
fails = []
checks = 0

def D(x):
    if isinstance(x, F):
        with localcontext() as c:
            c.prec = 120
            return Decimal(x.numerator) / Decimal(x.denominator)
    if hasattr(x, 'm'):
        return x.m
    if hasattr(x, 'mid') and hasattr(x, 'rad'):
        from klbench.ref.arbutil import exact_arb_to_fraction
        return D(exact_arb_to_fraction(x.mid()))
    return Decimal(x)

def close(a, b, tol, label):
    global checks
    checks += 1
    a, b = (D(a), D(b))
    with localcontext() as c:
        c.prec = 120
        scale = max(abs(a), abs(b), Decimal('1e-300'))
        bad = abs(a - b) > (Decimal('1e-38') if a == 0 or b == 0 else Decimal(tol) * scale)
    if bad:
        fails.append(f'{label}: {a} vs {b}')

def f32(x):
    return F(float(np.float32(x)))

def f16(x):
    return F(float(np.float16(x)))

def ln(x):
    return x.ln()

def atanh(t):
    return ((1 + t) / (1 - t)).ln() / 2

def fd(func, params, side, name, idx, h=Decimal('1e-22')):

    def shifted(s):
        pp = {k: list(v) for k, v in params[0].items()}
        qq = {k: list(v) for k, v in params[1].items()}
        tgt = pp if side == 'P' else qq
        tgt[name][idx] = tgt[name][idx] + F(s)
        return func(pp, qq)
    with localcontext() as c:
        c.prec = PREC
        return (shifted(h) - shifted(-h)) / (2 * h)
b = bernoulli_numbers(14)
known = {2: F(1, 6), 4: F(-1, 30), 6: F(1, 42), 8: F(-1, 30), 10: F(5, 66), 12: F(-691, 2730), 14: F(7, 6)}
for k, v in known.items():
    checks += 1
    if b[k] != v:
        fails.append(f'B_{k} = {b[k]} != {v}')

def cb_kl_spec(lp, lq):
    with localcontext() as c:
        c.prec = PREC + 40

        def logc(lam):
            t = 1 - 2 * D(lam)
            return Decimal(2).ln() if t == 0 else (2 * atanh(t) / t).ln()

        def mu(lam):
            t = 1 - 2 * D(lam)
            return Decimal(1) / 2 if t == 0 else D(lam) / (2 * D(lam) - 1) + 1 / (2 * atanh(t))
        x, y = (D(lp), D(lq))
        logit = lambda v: (v / (1 - v)).ln()
        return logc(lp) - logc(lq) + (1 - x).ln() - (1 - y).ln() + mu(lp) * (logit(x) - logit(y))

def sigmoid(z):
    with localcontext() as c:
        c.prec = PREC + 60
        return 1 / (1 + (-D(z)).exp())
cbp, cbl = (CONFIGS['contbern_probs'], CONFIGS['contbern_logits'])
pairs = []
for _ in range(25):
    a = f32(rng.uniform(0.02, 0.98))
    pairs.append((a, f32(float(a) * (1 + rng.uniform(-0.05, 0.05)))))
for _ in range(25):
    a = F(1, 2) + F(rng.randrange(-200, 200), 2 ** 24)
    bb = a + F(rng.randrange(-60, 60), 2 ** 24) if rng.random() < 0.8 else F(1, 2) + F(rng.randrange(-200000, 200000), 2 ** 24)
    pairs.append((a, bb))
pairs += [(F(1, 2), F(1, 2) + F(1, 2 ** 24)), (F(1, 2) - F(1, 2 ** 11), F(1, 2)), (F(1, 2) + F(1, 130), F(1, 2) + F(1, 126))]
for a, bb in pairs:
    if a == bb or not 0 < bb < 1:
        continue
    p, q = ({'probs': [a]}, {'probs': [bb]})
    exact = cb_kl_spec(a, bb)
    close(cbp.kl_arb(p, q, 256), exact, '1e-25', f'CB probs KL {a},{bb}')
    g = cbp.grad_arb(p, q, 256)
    for side in ('P', 'Q'):
        close(g[side]['probs'][0], fd(lambda pp, qq: cb_kl_spec(pp['probs'][0], qq['probs'][0]), (p, q), side, 'probs', 0), '1e-15', f'CB probs d{side} {a},{bb}')
for _ in range(40):
    if rng.random() < 0.5:
        zp = f32(rng.uniform(-8, 8))
        zq = f32(float(zp) + rng.uniform(-0.3, 0.3))
    else:
        zp = f32(rng.uniform(-0.07, 0.07))
        zq = f32(float(zp) + rng.uniform(-0.01, 0.01))
    if rng.random() < 0.1:
        zp = F(0)
    if zp == zq:
        continue
    p, q = ({'logits': [zp]}, {'logits': [zq]})
    spec_logits = lambda pp, qq: cb_kl_spec(sigmoid(pp['logits'][0]), sigmoid(qq['logits'][0]))
    exact = spec_logits(p, q)
    close(cbl.kl_arb(p, q, 256), exact, '1e-25', f'CB logits KL {zp},{zq}')
    g = cbl.grad_arb(p, q, 256)
    for side in ('P', 'Q'):
        close(g[side]['logits'][0], fd(spec_logits, (p, q), side, 'logits', 0), '1e-15', f'CB logits d{side} {zp},{zq}')

def cat_probs_kl(pp, qq):
    with localcontext() as c:
        c.prec = PREC + 20
        x = [D(v) for v in pp['probs']]
        y = [D(v) for v in qq['probs']]
        sx, sy = (sum(x), sum(y))
        return sum((a / sx * (a / sx / (b_ / sy)).ln() for a, b_ in zip(x, y)))

def cat_logits_kl(pp, qq):
    with localcontext() as c:
        c.prec = PREC + 20
        z = [D(v) for v in pp['logits']]
        w = [D(v) for v in qq['logits']]
        lse = lambda v: max(v) + sum(((a - max(v)).exp() for a in v)).ln()
        lz, lw = (lse(z), lse(w))
        return sum(((a - lz).exp() * (a - lz - (b_ - lw)) for a, b_ in zip(z, w)))
for k in (2, 4, 16):
    cp, cl = (CONFIGS[f'categorical_probs_K{k}'], CONFIGS[f'categorical_logits_K{k}'])
    for trial in range(8):
        x = [f32(rng.uniform(0.05, 3)) for _ in range(k)]
        y = [f32(float(v) * np.exp(rng.uniform(-0.05, 0.05))) for v in x]
        p, q = ({'probs': x}, {'probs': y})
        close(cp.kl_arb(p, q, 256), cat_probs_kl(p, q), '1e-25', f'cat probs K{k} KL')
        g = cp.grad_arb(p, q, 256)
        ex = cp.exact_grad(p, q)
        for j in range(k):
            for side in ('P', 'Q'):
                close(g[side]['probs'][j], fd(cat_probs_kl, (p, q), side, 'probs', j), '1e-14', f'cat probs K{k} d{side}[{j}]')
            close(g['Q']['probs'][j], ex['Q']['probs'][j], '1e-30', f'cat probs exactQ K{k}[{j}]')
        z = [f32(rng.uniform(-5, 5)) for _ in range(k)]
        w = [f32(float(v) + rng.uniform(-0.05, 0.05)) for v in z]
        p, q = ({'logits': z}, {'logits': w})
        close(cl.kl_arb(p, q, 256), cat_logits_kl(p, q), '1e-25', f'cat logits K{k} KL')
        g = cl.grad_arb(p, q, 256)
        for j in range(k):
            for side in ('P', 'Q'):
                close(g[side]['logits'][j], fd(cat_logits_kl, (p, q), side, 'logits', j), '1e-14', f'cat logits K{k} d{side}[{j}]')
    x = [f32(rng.uniform(0.05, 3)) for _ in range(k)]
    checks += 2
    if not cp.semantic_equal({'probs': x}, {'probs': [2 * v for v in x]}):
        fails.append('cat probs semantic')
    if not cl.semantic_equal({'logits': x}, {'logits': [v + F(3, 8) for v in x]}):
        fails.append('cat logits semantic')

def dir_kl(pp, qq):
    with localcontext() as c:
        c.prec = PREC + 20
        a = [D(v) for v in pp['concentration']]
        b_ = [D(v) for v in qq['concentration']]
        a0, b0 = (sum(a), sum(b_))
        lg = lambda v: gmath.lgamma(v, PREC + 10)
        dg = lambda v: gmath.digamma(v, PREC + 10)
        return lg(a0) - lg(b0) - sum((lg(x) - lg(y) for x, y in zip(a, b_))) + sum(((x - y) * (dg(x) - dg(a0)) for x, y in zip(a, b_)))
beta = CONFIGS['beta']
for k in (2, 4, 16):
    dc = CONFIGS[f'dirichlet_K{k}']
    for trial in range(6):
        a = [f32(np.exp(rng.uniform(np.log(0.05), np.log(50)))) for _ in range(k)]
        bq = [f32(float(v) * np.exp(rng.uniform(-0.03, 0.03))) for v in a]
        p, q = ({'concentration': a}, {'concentration': bq})
        close(dc.kl_arb(p, q, 256), dir_kl(p, q), '1e-25', f'dirichlet K{k} KL')
        g = dc.grad_arb(p, q, 256)
        for j in range(k):
            for side in ('P', 'Q'):
                close(g[side]['concentration'][j], fd(dir_kl, (p, q), side, 'concentration', j, h=Decimal('1e-25')), '1e-12', f'dirichlet K{k} d{side}[{j}]')
        if k == 2:
            pb = {'concentration1': [a[0]], 'concentration0': [a[1]]}
            qb = {'concentration1': [bq[0]], 'concentration0': [bq[1]]}
            close(dc.kl_arb(p, q, 256), beta.kl_arb(pb, qb, 256), '1e-30', 'dirichlet K2 = beta KL')
            gb = beta.grad_arb(pb, qb, 256)
            for side in ('P', 'Q'):
                close(g[side]['concentration'][0], gb[side]['concentration1'][0], '1e-30', 'dir=beta g1')
                close(g[side]['concentration'][1], gb[side]['concentration0'][0], '1e-30', 'dir=beta g0')
    a = [F(rng.randrange(1, 60)) for _ in range(k)]
    bq = list(a)
    j = rng.randrange(k)
    bq[j] = a[j] + 3
    p, q = ({'concentration': a}, {'concentration': bq})
    ex = dc.exact_grad(p, q)
    g = dc.grad_arb(p, q, 256)
    for i in range(k):
        for side in ('P', 'Q'):
            if ex[side]['concentration'][i] is not None:
                close(g[side]['concentration'][i], ex[side]['concentration'][i], '1e-30', f'dirichlet K{k} exact {side}[{i}]')
    perm = list(reversed(a))
    if perm != a:
        ek = dc.exact_kl({'concentration': a}, {'concentration': perm})
        close(dc.kl_arb({'concentration': a}, {'concentration': perm}, 256), ek, '1e-30', f'dirichlet K{k} exact_kl permutation')
    if k >= 3:
        a = [f32(rng.uniform(1, 5)) for _ in range(k)]
        bq = list(a)
        bq[0] = a[0] + F(1, 4)
        bq[1] = a[1] - F(1, 4)
        m = dc.exact_zero_mask({'concentration': a}, {'concentration': bq})
        g = dc.grad_arb({'concentration': a}, {'concentration': bq}, 256)
        for i in range(2, k):
            checks += 1
            if not (m['P']['concentration'][i] and abs(D(g['P']['concentration'][i])) < Decimal('1e-40') and (abs(D(g['Q']['concentration'][i])) < Decimal('1e-40'))):
                fails.append(f'dirichlet structural zero K{k}[{i}]')

def mvn_kl_indep(sig_p, sig_q, delta):
    with localcontext() as c:
        c.prec = PREC + 30
        d = len(sig_p)
        a = [[D(v) for v in row] + [Decimal(int(i == j)) for j in range(d)] for i, row in enumerate(sig_q)]
        det = Decimal(1)
        for col in range(d):
            pv = a[col][col]
            det *= pv
            a[col] = [v / pv for v in a[col]]
            for r_ in range(d):
                if r_ != col:
                    f = a[r_][col]
                    a[r_] = [x - f * y for x, y in zip(a[r_], a[col])]
        inv = [row[d:] for row in a]
        sp = [[D(v) for v in row] for row in sig_p]
        tr = sum((inv[i][j] * sp[j][i] for i in range(d) for j in range(d)))
        dl = [D(v) for v in delta]
        quad = sum((dl[i] * inv[i][j] * dl[j] for i in range(d) for j in range(d)))
        b_ = [list(row) for row in sp]
        detp = Decimal(1)
        for col in range(d):
            pv = b_[col][col]
            detp *= pv
            for r_ in range(col + 1, d):
                f = b_[r_][col] / pv
                b_[r_] = [x - f * y for x, y in zip(b_[r_], b_[col])]
        return (tr + quad - d + det.ln() - detp.ln()) / 2

def mvn_params_kl(d):

    def kl(pp, qq):
        lp = [pp['scale_tril'][i * d:(i + 1) * d] for i in range(d)]
        lq = [qq['scale_tril'][i * d:(i + 1) * d] for i in range(d)]
        sp = [[sum((lp[i][t] * lp[j][t] for t in range(d)), F(0)) for j in range(d)] for i in range(d)]
        sq = [[sum((lq[i][t] * lq[j][t] for t in range(d)), F(0)) for j in range(d)] for i in range(d)]
        return mvn_kl_indep(sp, sq, [a - b_ for a, b_ in zip(pp['loc'], qq['loc'])])
    return kl

def lowrank_params_kl(d, r):

    def kl(pp, qq):

        def sig(side):
            w = [side['cov_factor'][i * r:(i + 1) * r] for i in range(d)]
            s = [[sum((w[i][t] * w[j][t] for t in range(r)), F(0)) for j in range(d)] for i in range(d)]
            for i in range(d):
                s[i][i] += side['cov_diag'][i]
            return s
        return mvn_kl_indep(sig(pp), sig(qq), [a - b_ for a, b_ in zip(pp['loc'], qq['loc'])])
    return kl
for d in (2, 4):
    mc = CONFIGS[f'mvn_d{d}']
    klf = mvn_params_kl(d)
    for trial in range(6):
        low = [[F(0)] * d for _ in range(d)]
        for i in range(d):
            low[i][i] = f32(np.exp(rng.uniform(np.log(0.2), np.log(5))))
            for j in range(i):
                low[i][j] = f32(rng.uniform(-0.5, 0.5))
        lq = [[f32(float(v) * (1 + rng.uniform(-0.02, 0.02))) if j <= i else F(0) for j, v in enumerate(row)] for i, row in enumerate(low)]
        loc = [f32(rng.uniform(-5, 5)) for _ in range(d)]
        locq = [f32(float(v) + rng.uniform(-0.05, 0.05)) for v in loc]
        if trial == 0:
            lq = [list(r_) for r_ in low]
        p = {'loc': loc, 'scale_tril': [v for row in low for v in row]}
        q = {'loc': locq, 'scale_tril': [v for row in lq for v in row]}
        ref = klf(p, q)
        close(mc.kl_arb(p, q, 256), ref, '1e-25', f'mvn d{d} KL')
        ek = mc.exact_kl(p, q)
        if trial == 0:
            close(ek, ref, '1e-40', f'mvn d{d} exact_kl')
        ex = mc.exact_grad(p, q)
        g = mc.grad_arb(p, q, 256)
        m = mc.nondiff_mask(p, q)
        for name, n in (('loc', d), ('scale_tril', d * d)):
            for idx in range(n):
                for side in ('P', 'Q'):
                    e = ex[side][name][idx]
                    if name == 'scale_tril' and idx % d > idx // d:
                        checks += 1
                        if not (e is None and m[side][name][idx]):
                            fails.append('mvn upper nondiff')
                        continue
                    close(g[side][name][idx], e, '1e-28', f'mvn d{d} exact vs arb {side}.{name}[{idx}]')
                    close(e, fd(klf, (p, q), side, name, idx), '1e-14', f'mvn d{d} fd {side}.{name}[{idx}]')
    for r in (1, 2):
        lc = CONFIGS[f'lowrank_mvn_d{d}_r{r}']
        klf2 = lowrank_params_kl(d, r)
        for trial in range(5):
            w = [f32(rng.uniform(-1, 1)) for _ in range(d * r)]
            dg = [f32(np.exp(rng.uniform(np.log(0.2), np.log(5)))) for _ in range(d)]
            wq = [f32(float(v) + rng.uniform(-0.02, 0.02)) for v in w]
            dq = [f32(float(v) * (1 + rng.uniform(-0.02, 0.02))) for v in dg]
            loc = [f32(rng.uniform(-5, 5)) for _ in range(d)]
            locq = [f32(float(v) + rng.uniform(-0.05, 0.05)) for v in loc]
            if trial == 0:
                wq = [-v for v in w]
                dq = list(dg)
            p = {'loc': loc, 'cov_factor': w, 'cov_diag': dg}
            q = {'loc': locq, 'cov_factor': wq, 'cov_diag': dq}
            ref = klf2(p, q)
            close(lc.kl_arb(p, q, 256), ref, '1e-25', f'lowrank d{d}r{r} KL')
            if trial == 0:
                close(lc.exact_kl(p, q), ref, '1e-40', f'lowrank d{d}r{r} exact_kl')
                checks += 1
                if lc.semantic_equal(p, {'loc': loc, 'cov_factor': wq, 'cov_diag': dq}) is not True:
                    fails.append('lowrank semantic W -> -W')
            ex = lc.exact_grad(p, q)
            g = lc.grad_arb(p, q, 256)
            for name, n in (('loc', d), ('cov_factor', d * r), ('cov_diag', d)):
                for idx in range(n):
                    for side in ('P', 'Q'):
                        close(g[side][name][idx], ex[side][name][idx], '1e-28', f'lowrank exact vs arb {side}.{name}[{idx}]')
                        close(ex[side][name][idx], fd(klf2, (p, q), side, name, idx), '1e-14', f'lowrank fd d{d}r{r} {side}.{name}[{idx}]')
print(f'checks {checks}, failures {len(fails)}')
for f in fails[:40]:
    print('  FAIL', f)
sys.exit(1 if fails else 0)
