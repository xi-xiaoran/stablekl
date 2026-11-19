from __future__ import annotations
import hashlib, math
from fractions import Fraction as Fr
import numpy as np
from klbench.fp import get_format, exact_value
from . import truth as T
from .store import xt_hash
NP = {'fp16': (np.float16, np.uint16), 'fp32': (np.float32, np.uint32)}
GROUPS = [('B_c', Fr(1, 10), Fr(1))] + [(f'B_{k}', Fr(1, 10 ** (k + 1)), Fr(1, 10 ** k)) for k in range(1, 15)]
REGIONS = ('R0', 'R1', 'R2')
BASE_SEED = 20261111

def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join((str(p) for p in (BASE_SEED,) + parts)).encode()).digest()[:8], 'little')

def to_bits(a, dt):
    f, u = NP[dt]
    return np.asarray(a, dtype=f).view(u)

def from_bits(b, dt):
    f, u = NP[dt]
    return np.asarray(b, dtype=u).view(f)

def exact(bits_row, dt):
    return [exact_value(int(b), dt) for b in bits_row]

def classify(lo, hi):
    for name, a, b in GROUPS:
        if a <= lo and hi < b:
            return name
    return None

def sample_logits(rng, K, region, dt):
    f, _ = NP[dt]
    z = rng.uniform(-5, 5, K)
    if region != 'R0':
        r = rng.uniform(10, 30) if region == 'R1' else math.exp(rng.uniform(math.log(30), math.log(1000.0)))
        z = (z - z.min()) / max(z.max() - z.min(), 1e-300) * r
        z = z - rng.uniform(0, r)
    return z.astype(f)

def region_of(zbits, dt):
    z = from_bits(zbits, dt).astype(np.float64)
    r = float(z.max() - z.min())
    return 'R2' if r >= 30 else 'R1' if r >= 10 else 'R0'

def worst(a, b):
    return max(a, b, key=REGIONS.index)

def kl_f64(zp, zq):
    lp = zp - zp.max()
    lp = lp - math.log(np.exp(lp).sum())
    lq = zq - zq.max()
    lq = lq - math.log(np.exp(lq).sum())
    return float((np.exp(lp) * (lp - lq)).sum())

def propose_q(rng, zp, target, dt):
    f, _ = NP[dt]
    K = zp.size
    z = zp.astype(np.float64)
    kind = ('single', 'subset', 'all')[rng.integers(3)]
    mask = np.zeros(K, bool)
    if kind == 'single':
        mask[rng.integers(K)] = True
    elif kind == 'subset':
        mask[rng.choice(K, size=rng.integers(1, K + 1), replace=False)] = True
    else:
        mask[:] = True
    v = np.where(mask, rng.choice((-1.0, 1.0), K) * rng.uniform(0.5, 1, K), 0.0)
    p = np.exp(z - z.max())
    p /= p.sum()
    var = float((p * v * v).sum() - (p * v).sum() ** 2)
    if var <= 0:
        return (None, kind)
    h = math.sqrt(2 * target / var)
    if target > 1e-06:
        lo, hi = (0.0, max(h, 0.001))
        while kl_f64(z, z + hi * v) < target and hi < 10000.0:
            hi *= 2
        for _ in range(60):
            mid = (lo + hi) / 2
            if kl_f64(z, z + mid * v) < target:
                lo = mid
            else:
                hi = mid
        h = hi
    return ((z + h * v).astype(f), kind)

def make_xt(zp_bits, zq_bits, dt, log_target):
    zp = exact(zp_bits, dt)
    zq = exact(zq_bits, dt)
    x = T.rn_log_softmax(zq, dt)
    t = T.rn_log_softmax(zp, dt) if log_target else T.rn_softmax(zp, dt)
    return (x, t)

class Unit:

    def __init__(self, layer, group, region, has_logits):
        self.layer, self.group, self.region, self.has_logits = (layer, group, region, has_logits)
        self.rows = []
        self.keys = set()
        self.counters = {}

    def count(self, k):
        self.counters[k] = self.counters.get(k, 0) + 1

    def add(self, x, t, zp=None, zq=None, **meta):
        key = (self.layer, tuple((int(v) for v in x)), tuple((int(v) for v in t))) + ((tuple((int(v) for v in zp)), tuple((int(v) for v in zq))) if self.has_logits else ())
        if key in self.keys:
            self.count('DUPLICATE')
            return False
        self.keys.add(key)
        h = None if meta.get('needs_torch_inputs') else xt_hash(x, t)
        self.rows.append(dict(x=list(map(int, x)), t=list(map(int, t)), zp=None if zp is None else list(map(int, zp)), zq=None if zq is None else list(map(int, zq)), xt_hash=h, **meta))
        return True

    @property
    def name(self):
        return '__'.join((p for p in (self.layer, self.group, self.region) if p))

def gen_S1(log_target, dt, K, quota, torch_inputs=False, max_factor=30):
    layer = 'S1t' if torch_inputs else 'S1'
    units = {(g[0], r): Unit(layer, g[0], r, True) for g in GROUPS for r in REGIONS}
    for gname, glo, ghi in GROUPS:
        for region in REGIONS:
            rng = np.random.default_rng(seed_for(layer, log_target, dt, K, gname, region))
            want = quota[region]
            tries = 0
            u_target = units[gname, region]
            while len(u_target.rows) < want and tries < max_factor * max(want, 1):
                tries += 1
                u_target.count('CANDIDATES')
                zp = sample_logits(rng, K, region, dt)
                target = math.exp(rng.uniform(math.log(float(glo)), math.log(float(ghi))))
                zq, kind = propose_q(rng, zp, target, dt)
                if zq is None or not np.isfinite(zq).all():
                    u_target.count('BAD_DIRECTION')
                    continue
                zpb, zqb = (to_bits(zp, dt), to_bits(zq, dt))
                if (zpb == zqb).all():
                    u_target.count('ROUNDED_TO_IDENTICAL')
                    continue
                if float(zq.astype(np.float64).max() - zq.astype(np.float64).min()) > 1000.0:
                    u_target.count('OUT_OF_REGION')
                    continue
                lo, hi = T.interval(T.e2e_row(exact(zpb, dt), exact(zqb, dt)))
                g = classify(lo, hi)
                if g is None:
                    u_target.count('UNCERT_BIN')
                    continue
                reg = worst(region_of(zpb, dt), region_of(zqb, dt))
                u = units[g, reg]
                if len(u.rows) >= quota[reg]:
                    u_target.count('BIN_FULL')
                    continue
                if torch_inputs:
                    x = t = [0] * K
                else:
                    try:
                        x, t = make_xt(zpb, zqb, dt, log_target)
                    except ArithmeticError:
                        u_target.count('INPUT_ROUNDING_UNCERT')
                        continue
                u.add(x, t, zpb, zqb, direction=kind, e2e_lo=str(lo), e2e_hi=str(hi), needs_torch_inputs=torch_inputs)
    for (g, r), u in units.items():
        if len(u.rows) < quota[r]:
            u.counters['STATUS'] = 'INFEASIBLE' if not u.rows else f'INSUFFICIENT_SAMPLES({len(u.rows)})'
    return list(units.values())

def gen_S2(log_target, dt, K, n, variant_b=False):
    u = Unit('S2b' if variant_b else 'S2', '', '', True)
    rng = np.random.default_rng(seed_for(u.layer, log_target, dt, K))
    tries = 0
    while len(u.rows) < n and tries < 20 * n:
        tries += 1
        zb = to_bits(sample_logits(rng, K, REGIONS[rng.integers(3)], dt), dt)
        try:
            x, t = make_xt(zb, zb, dt, log_target)
        except ArithmeticError:
            u.count('INPUT_ROUNDING_UNCERT')
            continue
        if variant_b:
            if any((v == 0 for v in t)):
                u.count('T_ROUNDED_TO_ZERO')
                continue
            x = T.rn_log(t, dt)
        u.add(x, t, zb, zb, e2e_lo='0', e2e_hi='0')
    return [u]

def nudge(bits, k, dt):
    f = get_format(dt)
    m = f.sign_mask - 1
    mag = bits & m
    o = -mag if bits & f.sign_mask else mag
    o += k
    return f.sign_mask | -o if o < 0 else o

def gen_S3(log_target, dt, K, n, pool):
    u = Unit('S3', '', '', False)
    rng = np.random.default_rng(seed_for('S3', log_target, dt, K))
    f = get_format(dt)
    tries = 0
    while len(u.rows) < n and pool and (tries < 20 * n):
        tries += 1
        r = pool[rng.integers(len(pool))]
        x = list(r['x'])
        t = list(r['t'])
        side = rng.integers(2)
        idx = rng.choice(K, size=rng.integers(1, K + 1), replace=False)
        ks = []
        for i in idx:
            k = int(rng.integers(1, 9)) * int(rng.choice((-1, 1)))
            ks.append(k)
            if side == 0:
                x[i] = nudge(x[i], k, dt)
            else:
                t[i] = nudge(t[i], k, dt)
        vals = from_bits(np.array(x + t, dtype=NP[dt][1]), dt)
        if not np.isfinite(vals).all() or (not log_target and (from_bits(np.array(t, dtype=NP[dt][1]), dt) < 0).any()):
            u.count('OUT_OF_RANGE')
            continue
        u.add(x, t, side='x' if side == 0 else 't', ulp_moves=ks, source=r.get('layer', 'S1'))
    return [u]

def gen_S4(log_target, dt, K, n):
    u = Unit('S4', '', '', False)
    rng = np.random.default_rng(seed_for('S4', log_target, dt, K))
    f = NP[dt][0]
    while len(u.rows) < n:
        t = rng.uniform(-20, 5, K).astype(f) if log_target else np.exp(rng.uniform(math.log(0.0001), math.log(100.0), K)).astype(f)
        mode = 'independent' if rng.random() < 0.5 else 'near'
        if mode == 'independent':
            x = rng.uniform(-20, 5, K).astype(f)
            xb = to_bits(x, dt)
        else:
            h = math.exp(rng.uniform(math.log(1e-07), 0))
            base = from_bits(np.array(T.rn_log(to_bits(t, dt), dt), dtype=NP[dt][1]), dt) if not log_target else t
            xb = to_bits((base.astype(np.float64) + h * rng.uniform(-1, 1, K)).astype(f), dt)
        u.add(xb, to_bits(t, dt), mode=mode)
    return [u]

def _far_row(rng, K, dt, log_target):
    zp = to_bits(sample_logits(rng, K, 'R0', dt), dt)
    zq = to_bits(sample_logits(rng, K, 'R0', dt), dt)
    return make_xt(zp, zq, dt, log_target)

def gen_S5(log_target, dt, K, n):
    u = Unit('S5', '', '', False)
    rng = np.random.default_rng(seed_for('S5', log_target, dt, K))
    f = get_format(dt)
    ninf = f.sign_mask | f.infinity_bits
    while len(u.rows) < n:
        x, t = _far_row(rng, K, dt, log_target)
        x, t = (list(x), list(t))
        kind = 'zero' if log_target or rng.random() < 0.7 else 'subnormal'
        idx = rng.choice(K, size=rng.integers(1, K), replace=False) if K > 1 else [0]
        for i in idx:
            if kind == 'zero':
                t[i] = ninf if log_target else 0
            else:
                t[i] = int(rng.integers(1, 1 << f.fraction_bits))
        u.add(x, t, kind=kind)
    return [u]

def gen_S6(log_target, dt, K, n):
    u = Unit('S6', '', '', False)
    rng = np.random.default_rng(seed_for('S6', log_target, dt, K))
    f = get_format(dt)
    fl = NP[dt][0]
    ninf = f.sign_mask | f.infinity_bits
    fmax = float(np.finfo(fl).max)
    while len(u.rows) < n:
        x, t = _far_row(rng, K, dt, log_target)
        x, t = (list(x), list(t))
        kind = ('big_x', 'big_t', 'x_neg_inf', 'x_neg_inf_t_zero')[rng.integers(4)]
        idx = rng.choice(K, size=rng.integers(1, K + 1), replace=False)
        for i in idx:
            if kind == 'big_x':
                x[i] = int(to_bits(-math.exp(rng.uniform(math.log(100.0), math.log(1000.0))), dt))
            elif kind == 'big_t':
                if log_target:
                    t[i] = int(to_bits(rng.uniform(*((10.0, 12.0) if dt == 'fp16' else (80.0, 100.0))), dt))
                else:
                    t[i] = int(to_bits(fmax * math.exp(-rng.uniform(0, math.log(16))), dt))
            elif kind == 'x_neg_inf':
                x[i] = ninf
            else:
                x[i] = ninf
                t[i] = ninf if log_target else 0
        u.add(x, t, kind=kind)
    return [u]

def gen_S7(dt, K, n):
    u = Unit('S7', '', '', False)
    rng = np.random.default_rng(seed_for('S7', dt, K))
    f = get_format(dt)
    while len(u.rows) < n:
        x, t = _far_row(rng, K, dt, False)
        x, t = (list(x), list(t))
        for i in rng.choice(K, size=rng.integers(1, K + 1), replace=False):
            t[i] |= f.sign_mask
        u.add(x, t)
    return [u]

def generate(log_target, dt, K, q):
    units = gen_S1(log_target, dt, K, q['S1']) + gen_S1(log_target, dt, K, q['S1t'], torch_inputs=True) + gen_S2(log_target, dt, K, q['S2'])
    if not log_target:
        units += gen_S2(log_target, dt, K, q['S2b'], variant_b=True)
    pool = [dict(r, layer=u.layer) for u in units if u.layer in ('S1', 'S2') for r in u.rows]
    units += gen_S3(log_target, dt, K, q['S3'], pool) + gen_S4(log_target, dt, K, q['S4']) + gen_S5(log_target, dt, K, q['S5']) + gen_S6(log_target, dt, K, q['S6'])
    if not log_target:
        units += gen_S7(dt, K, q['S7'])
    return units
