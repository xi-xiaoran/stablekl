from __future__ import annotations
import math
import numpy as np
from . import gen as G, truth as T
FAMILIES = ('supp_interp', 'supp_ulp', 'supp_ulp_zeroed')
LAYER = {'S1': 'S1s', 'S1t': 'S1ts'}

def propose(rng, K, region, dt, glo, ghi):
    f, _ = G.NP[dt]
    zp = G.sample_logits(rng, K, region, dt)
    target = math.exp(rng.uniform(math.log(float(glo)), math.log(float(ghi))))
    fam = FAMILIES[rng.integers(3)]
    if fam == 'supp_interp':
        z = zp.astype(np.float64)
        w = G.sample_logits(rng, K, region, dt).astype(np.float64)
        if G.kl_f64(z, w) < target:
            return (None, None, 'TARGET_UNREACHABLE')
        lo, hi = (0.0, 1.0)
        for _ in range(60):
            mid = (lo + hi) / 2
            if G.kl_f64(z, z + mid * (w - z)) < target:
                lo = mid
            else:
                hi = mid
        return (zp, (z + hi * (w - z)).astype(f), fam)
    order = rng.permutation(K)[:32]
    if fam == 'supp_ulp_zeroed':
        zp = zp.copy()
        zp[order[0]] = 0
        order = order[:1]
    z = zp.astype(np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    for i in order:
        v = float(p[i] * (1 - p[i]))
        if v <= 0:
            continue
        d = math.sqrt(2 * target / v) * (1 if rng.integers(2) else -1)
        zq = zp.copy()
        with np.errstate(all='ignore'):
            zq[i] = f(z[i] + d)
        if zq[i] == zp[i] or not np.isfinite(zq[i]):
            continue
        if float(glo) <= G.kl_f64(z, zq.astype(np.float64)) < float(ghi):
            return (zp, zq, fam)
    return (None, None, 'NO_REPRESENTABLE_STEP')

def gen_cell(log_target, dt, K, base_layer, gname, region, gap, existing, max_factor=30):
    glo, ghi = next(((a, b) for n, a, b in G.GROUPS if n == gname))
    layer = LAYER[base_layer]
    torch_inputs = base_layer == 'S1t'
    u = G.Unit(layer, gname, region, True)
    rng = np.random.default_rng(G.seed_for(layer, log_target, dt, K, gname, region))
    tries = 0
    u.counters['GAP'] = gap
    while len(u.rows) < gap and tries < max_factor * gap:
        tries += 1
        u.count('CANDIDATES')
        zp, zq, fam = propose(rng, K, region, dt, glo, ghi)
        if zp is None:
            u.count(fam)
            continue
        zpb, zqb = (G.to_bits(zp, dt), G.to_bits(zq, dt))
        if (zpb == zqb).all():
            u.count('ROUNDED_TO_IDENTICAL')
            continue
        if float(zq.astype(np.float64).max() - zq.astype(np.float64).min()) > 1000.0:
            u.count('OUT_OF_REGION')
            continue
        lo, hi = T.interval(T.e2e_row(G.exact(zpb, dt), G.exact(zqb, dt)))
        g = G.classify(lo, hi)
        if g is None:
            u.count('UNCERT_BIN')
            continue
        if (g, G.worst(G.region_of(zpb, dt), G.region_of(zqb, dt))) != (gname, region):
            u.count('OTHER_CELL')
            continue
        if (tuple((int(v) for v in zpb)), tuple((int(v) for v in zqb))) in existing:
            u.count('DUPLICATE_OF_FROZEN')
            continue
        if torch_inputs:
            x = t = [0] * K
        else:
            try:
                x, t = G.make_xt(zpb, zqb, dt, log_target)
            except ArithmeticError:
                u.count('INPUT_ROUNDING_UNCERT')
                continue
        if u.add(x, t, zpb, zqb, direction=fam, e2e_lo=str(lo), e2e_hi=str(hi), needs_torch_inputs=torch_inputs):
            u.count('ACCEPTED_' + fam)
    if len(u.rows) < gap:
        u.counters['STATUS'] = 'ZERO_ACCEPTED' if not u.rows else f'INSUFFICIENT_SAMPLES({len(u.rows)})'
    return u
