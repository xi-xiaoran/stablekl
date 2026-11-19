from __future__ import annotations
import random
from fractions import Fraction as F
import mpmath as mp
from ..fp import FloatFormat, get_format, round_fraction
from ..mv_dims import base_config
from .constants import REGION_PRIORITY, ROOT_DPS
REGIONS_MV = {'contbern_probs': ('R0', 'R1', 'R2', 'R3'), 'contbern_logits': ('R0', 'R1', 'R2', 'R3'), 'categorical_probs': ('R0', 'R1', 'R2', 'R3'), 'categorical_logits': ('R0', 'R1', 'R2'), 'dirichlet': ('R0', 'R1', 'R2'), 'mvn': ('R0', 'R1', 'R2'), 'lowrank_mvn': ('R0', 'R1', 'R2')}
CB_R3 = (F(499, 1000), F(501, 1000))
CAT_PROB_R0_MIN = F(1, 1000)
CAT_LOGIT_R0 = F(10)
CAT_LOGIT_R1 = F(30)
CAT_LOGIT_R2 = F(1000)
CAT_LOGIT_ABS_MAX = F(1005)
MVN_DIAG = (F(1, 10), F(10))
MVN_COND_R0 = F(100)
MVN_COND_R1 = (F(100), F(10000))
MVN_LOC_RATIO_R2 = (F(100), F(10000))
LOC_BOX = F(10)
SAMPLE_TRIES = 400

def regions_for(config_id: str) -> tuple[str, ...]:
    return REGIONS_MV[base_config(config_id)]

def draw_ulp_p_region(config_id: str, rng: random.Random) -> str:
    roll = rng.randrange(100)
    if 'R3' in regions_for(config_id):
        return 'R0' if roll < 70 else 'R1' if roll < 80 else 'R2' if roll < 90 else 'R3'
    return 'R0' if roll < 70 else 'R1' if roll < 85 else 'R2'

def _mp(x) -> mp.mpf:
    if isinstance(x, F):
        return mp.mpf(x.numerator) / mp.mpf(x.denominator)
    return mp.mpf(x)

def _rand(rng: random.Random) -> mp.mpf:
    return mp.mpf(rng.getrandbits(64)) / (1 << 64)

def _between(rng, a, b) -> mp.mpf:
    return _mp(a) + (_mp(b) - _mp(a)) * _rand(rng)

def _logbetween(rng, a, b) -> mp.mpf:
    return mp.exp(_between(rng, mp.log(_mp(a)), mp.log(_mp(b))))

def _frac(x) -> F:
    from .runner import _frac as runner_frac
    return runner_frac(x)

def _quantize(params, dtype: str):
    fmt = get_format(dtype)
    out = {}
    for name, values in params.items():
        bits = []
        for v in values:
            fv = v if isinstance(v, F) else _frac(mp.mpf(v))
            if abs(fv) > fmt.maximum:
                raise ValueError('multivariate candidate outside the finite range of the target format')
            bits.append(round_fraction(fv, fmt))
        out[name] = bits
    return out

def _dims(config_id: str) -> tuple[int, int]:
    parts = config_id.split('_')
    k = r = 0
    for part in parts:
        if part[:1] in ('K', 'd') and part[1:].isdigit():
            k = int(part[1:])
        if part[:1] == 'r' and part[1:].isdigit():
            r = int(part[1:])
    return (k, r)

def _dirichlet1(rng, k: int) -> list[mp.mpf]:
    e = [-mp.log(1 - _rand(rng)) for _ in range(k)]
    s = mp.fsum(e)
    return [v / s for v in e]

def _sym_eig_minmax(matrix: list[list[F]] | list[list[mp.mpf]]) -> tuple[mp.mpf, mp.mpf]:
    with mp.workdps(50):
        a = mp.matrix([[_mp(v) for v in row] for row in matrix])
        eig, _ = mp.eigsy(a)
        vals = [eig[i] for i in range(a.rows)]
        return (min(vals), max(vals))

def _cov_mvn(low):
    d = len(low)
    return [[sum((low[i][t] * low[j][t] for t in range(d)), F(0) if isinstance(low[0][0], F) else mp.mpf(0)) for j in range(d)] for i in range(d)]

def _cov_lowrank(w, diag):
    d = len(w)
    r = len(w[0])
    zero = F(0) if isinstance(diag[0], F) else mp.mpf(0)
    out = [[sum((w[i][t] * w[j][t] for t in range(r)), zero) for j in range(d)] for i in range(d)]
    for i in range(d):
        out[i][i] += diag[i]
    return out

def _unflat(values, rows: int, cols: int):
    return [list(values[i * cols:(i + 1) * cols]) for i in range(rows)]

def _positive_tag(value: F) -> str:
    if F(1, 100) <= value <= F(100):
        return 'R0'
    if F(100) < value <= F(10000):
        return 'R1'
    if F(1, 10000) <= value < F(1, 100):
        return 'R2'
    return 'OUTSIDE'

def side_tags(config_id: str, side: dict[str, list[F]], fmt: FloatFormat) -> tuple[set[str], bool]:
    base = base_config(config_id)
    tags: set[str] = set()
    outside = False
    if base == 'contbern_probs':
        v = side['probs'][0]
        if fmt.minimum_subnormal <= v < fmt.eps or F(1) - fmt.eps < v < 1:
            tags.add('R2')
        elif fmt.eps <= v < F(1, 100) or F(99, 100) < v <= 1 - fmt.eps:
            tags.add('R1')
        elif F(1, 100) <= v <= F(99, 100):
            tags.add('R0')
        if CB_R3[0] < v <= CB_R3[1]:
            tags.add('R3')
    elif base == 'contbern_logits':
        z = side['logits'][0]
        with mp.workdps(90):
            x = _mp(z)
            av = abs(x)
            if av <= mp.log(99):
                tags.add('R0')
            elif av < 15:
                tags.add('R1')
            elif av <= 1000:
                tags.add('R2')
            else:
                outside = True
            lo = mp.log(_mp(CB_R3[0]) / (1 - _mp(CB_R3[0])))
            hi = mp.log(_mp(CB_R3[1]) / (1 - _mp(CB_R3[1])))
            if lo < x <= hi:
                tags.add('R3')
    elif base == 'categorical_probs':
        x = side['probs']
        total = sum(x)
        probs = [v / total for v in x]
        if any((p < fmt.minimum_subnormal for p in probs)):
            outside = True
        if any((fmt.minimum_subnormal <= p < fmt.eps for p in probs)):
            tags.add('R2')
        if any((fmt.eps <= p < CAT_PROB_R0_MIN for p in probs)):
            tags.add('R1')
        if all((p >= CAT_PROB_R0_MIN for p in probs)):
            tags.add('R0')
        if total >= fmt.maximum / 8 or any((v < fmt.minimum_normal for v in x)):
            tags.add('R3')
    elif base == 'categorical_logits':
        z = side['logits']
        spread = max(z) - min(z)
        if spread > CAT_LOGIT_R2 or any((abs(v) > CAT_LOGIT_ABS_MAX for v in z)):
            outside = True
        elif spread < CAT_LOGIT_R0:
            tags.add('R0')
        elif spread <= CAT_LOGIT_R1:
            tags.add('R1')
        else:
            tags.add('R2')
    elif base == 'dirichlet':
        labels = [_positive_tag(v) for v in side['concentration']]
        tags.update(labels)
        if 'OUTSIDE' in labels:
            outside = True
    elif base in ('mvn', 'lowrank_mvn'):
        loc = side['loc']
        d = len(loc)
        if base == 'mvn':
            low = _unflat(side['scale_tril'], d, d)
            diag = [low[i][i] for i in range(d)]
            sigma = _cov_mvn(low)
        else:
            diag = list(side['cov_diag'])
            r = len(side['cov_factor']) // d
            sigma = _cov_lowrank(_unflat(side['cov_factor'], d, r), diag)
        lmin, lmax = _sym_eig_minmax(sigma)
        with mp.workdps(50):
            cond = lmax / lmin
            ratio = mp.sqrt(mp.fsum((_mp(v) ** 2 for v in loc)) / lmin)
            if all((MVN_DIAG[0] <= v <= MVN_DIAG[1] for v in diag)) and cond <= _mp(MVN_COND_R0) and all((-LOC_BOX <= v <= LOC_BOX for v in loc)):
                tags.add('R0')
            if _mp(MVN_COND_R1[0]) <= cond <= _mp(MVN_COND_R1[1]):
                tags.add('R1')
            if _mp(MVN_LOC_RATIO_R2[0]) <= ratio <= _mp(MVN_LOC_RATIO_R2[1]):
                tags.add('R2')
    else:
        raise NotImplementedError(f'No multivariate region adapter for {config_id}')
    tags.discard('OUTSIDE')
    if not tags:
        outside = True
    return (tags, outside)

def label_sides(config_id, p, q, fmt, *, identity_group=False):

    def priority(labels):
        return max(labels, key=REGION_PRIORITY.index) if labels else 'OUTSIDE'
    (tp, op), (tq, oq) = (side_tags(config_id, p, fmt), side_tags(config_id, q, fmt))
    pl = 'OUTSIDE' if op else priority(tp)
    ql = 'OUTSIDE' if oq else priority(tq)
    return (pl, ql, priority(tp | tq), tuple(sorted(tp)), tuple(sorted(tq)))

def _sample_lower(rng, d: int, region: str):
    for _ in range(SAMPLE_TRIES):
        diag = [_logbetween(rng, *MVN_DIAG) for _ in range(d)]
        scale = mp.mpf(1) / 2 if region != 'R1' else _logbetween(rng, F(1, 2), F(20))
        low = [[mp.mpf(0)] * d for _ in range(d)]
        for i in range(d):
            low[i][i] = diag[i]
            for j in range(i):
                low[i][j] = (2 * _rand(rng) - 1) * mp.sqrt(diag[i] * diag[j]) * scale
        lmin, lmax = _sym_eig_minmax(_cov_mvn(low))
        cond = lmax / lmin
        if region == 'R1':
            if _mp(MVN_COND_R1[0]) <= cond <= _mp(MVN_COND_R1[1]):
                return (low, lmin)
        elif cond <= _mp(MVN_COND_R0):
            return (low, lmin)
    raise ValueError('MVN scale_tril sampler exhausted its tries')

def _sample_lowrank_cov(rng, d: int, r: int, region: str):
    for _ in range(SAMPLE_TRIES):
        diag = [_logbetween(rng, *MVN_DIAG) for _ in range(d)]
        base = mp.sqrt(mp.fsum(diag) / d)
        scale = mp.mpf(1) / 2 if region != 'R1' else _logbetween(rng, F(1), F(100))
        w = [[(2 * _rand(rng) - 1) * base * scale for _ in range(r)] for _ in range(d)]
        lmin, lmax = _sym_eig_minmax(_cov_lowrank(w, diag))
        cond = lmax / lmin
        if region == 'R1':
            if _mp(MVN_COND_R1[0]) <= cond <= _mp(MVN_COND_R1[1]):
                return (w, diag, lmin)
        elif cond <= _mp(MVN_COND_R0):
            return (w, diag, lmin)
    raise ValueError('LowRankMVN covariance sampler exhausted its tries')

def _sample_loc(rng, d: int, region: str, lmin: mp.mpf) -> list[mp.mpf]:
    if region == 'R2':
        direction = [mp.mpf(2 * _rand(rng) - 1) for _ in range(d)]
        norm = mp.sqrt(mp.fsum((v * v for v in direction))) or mp.mpf(1)
        t = _logbetween(rng, *MVN_LOC_RATIO_R2)
        return [v / norm * t * mp.sqrt(lmin) for v in direction]
    return [_between(rng, -LOC_BOX, LOC_BOX) for _ in range(d)]

def sample_p(config_id: str, region: str, dtype: str, rng: random.Random):
    base = base_config(config_id)
    fmt = get_format(dtype)
    k, r = _dims(config_id)
    with mp.workdps(ROOT_DPS):
        if base == 'contbern_probs':
            if region == 'R0':
                logit = _between(rng, -mp.log(99), mp.log(99))
                x = 1 / (1 + mp.exp(-logit))
            elif region == 'R1':
                x = _logbetween(rng, fmt.eps, F(1, 100))
            elif region == 'R2':
                x = _logbetween(rng, fmt.minimum_subnormal, fmt.eps)
            else:
                x = _between(rng, *CB_R3)
            if region in ('R1', 'R2') and rng.randrange(2):
                x = 1 - x
            p = {'probs': [x]}
        elif base == 'contbern_logits':
            if region == 'R0':
                z = _between(rng, -mp.log(99), mp.log(99))
            elif region == 'R1':
                z = _between(rng, mp.log(99), 15)
            elif region == 'R2':
                z = _between(rng, 15, 1000)
            else:
                lo = mp.log(_mp(CB_R3[0]) / (1 - _mp(CB_R3[0])))
                z = _between(rng, lo, -lo)
            if region in ('R1', 'R2') and rng.randrange(2):
                z = -z
            p = {'logits': [z]}
        elif base == 'categorical_probs':
            e = _dirichlet1(rng, k)
            if region in ('R0', 'R3'):
                for _ in range(SAMPLE_TRIES):
                    if min(e) >= _mp(CAT_PROB_R0_MIN):
                        break
                    e = _dirichlet1(rng, k)
                else:
                    raise ValueError('Dirichlet(1) resampling exhausted')
            s_lo, s_hi = (mp.mpf(1) / 10, mp.mpf(10))
            if region in ('R1', 'R2'):
                j = rng.randrange(k)
                small = _logbetween(rng, fmt.eps, CAT_PROB_R0_MIN) if region == 'R1' else _logbetween(rng, fmt.minimum_subnormal, fmt.eps)
                rest = mp.fsum((v for i, v in enumerate(e) if i != j))
                e = [small if i == j else v * (1 - small) / rest for i, v in enumerate(e)]
                s_lo = max(s_lo, 2 * _mp(fmt.minimum_normal) / min(e))
                s_hi = max(s_hi, 4 * s_lo)
                if s_hi > _mp(fmt.maximum) / 16:
                    raise ValueError('categorical R1/R2 scale would approach overflow')
                s = _logbetween(rng, s_lo, s_hi)
            elif region == 'R3':
                if rng.randrange(2):
                    s = _logbetween(rng, fmt.maximum / 8, fmt.maximum * F(999, 1000))
                else:
                    target = _logbetween(rng, 16 * fmt.minimum_subnormal, fmt.minimum_normal)
                    s = target / min(e)
            else:
                s = _logbetween(rng, s_lo, s_hi)
            p = {'probs': [s * v for v in e]}
        elif base == 'categorical_logits':
            if region == 'R0':
                z = [_between(rng, -5, 5) for _ in range(k)]
            else:
                spread = _logbetween(rng, CAT_LOGIT_R0, CAT_LOGIT_R1) if region == 'R1' else _logbetween(rng, CAT_LOGIT_R1, CAT_LOGIT_R2)
                center = _between(rng, -5, 5)
                lo, hi = (center - spread / 2, center + spread / 2)
                i0, i1 = rng.sample(range(k), 2)
                z = [_between(rng, lo, hi) for _ in range(k)]
                z[i0], z[i1] = (lo, hi)
            p = {'logits': z}
        elif base == 'dirichlet':
            bounds = {'R0': (F(1, 100), F(100)), 'R1': (F(100), F(10000)), 'R2': (F(1, 10000), F(1, 100))}[region]
            p = {'concentration': [_logbetween(rng, *bounds) for _ in range(k)]}
        elif base == 'mvn':
            low, lmin = _sample_lower(rng, k, region)
            p = {'loc': _sample_loc(rng, k, region, lmin), 'scale_tril': [low[i][j] if j <= i else F(0) for i in range(k) for j in range(k)]}
        elif base == 'lowrank_mvn':
            w, diag, lmin = _sample_lowrank_cov(rng, k, r, region)
            p = {'loc': _sample_loc(rng, k, region, lmin), 'cov_factor': [w[i][j] for i in range(k) for j in range(r)], 'cov_diag': diag}
        else:
            raise NotImplementedError(config_id)
        return _quantize(p, dtype)
_LOG_COORDS = {'categorical_probs': ('probs',), 'dirichlet': ('concentration',), 'lowrank_mvn': ('cov_diag',)}

def coordinates(config_id: str, p: dict) -> list[tuple[str, int]]:
    base = base_config(config_id)
    coords = []
    for name, values in p.items():
        n = len(values)
        if base == 'mvn' and name == 'scale_tril':
            d = int(round(n ** 0.5))
            coords.extend(((name, i * d + j) for i in range(d) for j in range(i + 1)))
        else:
            coords.extend(((name, i) for i in range(n)))
    return coords

def direction(config_id: str, p: dict, rng: random.Random):
    base = base_config(config_id)
    pool = coordinates(config_id, p)
    block = None
    if base in ('mvn', 'lowrank_mvn'):
        block = rng.choice(('mean', 'cov', 'both'))
        if block == 'mean':
            pool = [c for c in pool if c[0] == 'loc']
        elif block == 'cov':
            pool = [c for c in pool if c[0] != 'loc']
    kind = rng.choice(('single', 'subset', 'all'))
    if kind == 'single':
        chosen = [rng.choice(pool)]
    elif kind == 'all':
        chosen = list(pool)
    else:
        chosen = rng.sample(pool, rng.randrange(1, len(pool) + 1))
    weights = {c: rng.choice((-1, 1)) * _between(rng, F(1, 2), 1) for c in chosen}
    label = kind if block is None else f'{block}:{kind}'
    return (label, weights)

def perturb(config_id: str, p: dict[str, list[F]], weights, h: mp.mpf):
    base = base_config(config_id)
    q = {k: [_mp(v) for v in values] for k, values in p.items()}
    d = len(p['loc']) if 'loc' in p else 0
    for (name, index), weight in weights.items():
        x = _mp(p[name][index])
        step = h * weight
        if base == 'contbern_probs':
            z = mp.log(x / (1 - x))
            q[name][index] = 1 / (1 + mp.exp(-(z + step)))
        elif name in _LOG_COORDS.get(base, ()):
            q[name][index] = x * mp.exp(step)
        elif base == 'mvn' and name == 'scale_tril' and (index // d == index % d):
            q[name][index] = x * mp.exp(step)
        else:
            q[name][index] = x + step
    return q

def ulp_candidate(config_id: str, p_bits, dtype: str, rng: random.Random):
    from .runner import _neighbor
    fmt = get_format(dtype)
    pool = coordinates(config_id, p_bits)
    count = rng.randrange(1, len(pool) + 1)
    q = {k: list(v) for k, v in p_bits.items()}
    k = rng.randrange(1, 9)
    for name, index in rng.sample(pool, count):
        updated = _neighbor(q[name][index], fmt, k * rng.choice((-1, 1)))
        if updated is None:
            return (None, k)
        q[name][index] = updated
    return (q, k)
