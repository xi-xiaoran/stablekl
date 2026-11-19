from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as multiprocessing
import random
import sqlite3
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp
import numpy as np
from ..fp import get_format, round_fraction
from ..fp.bits import exact_value
from ..pipeline_io import project_root, save_unit, verify, load_unit, text_array
from ..ref.certify import CertStatus, certify_sample
from ..ref.configs import CONFIGS, get_config
from .constants import INFEASIBLE_PROBE, ATTEMPT_MULTIPLIER, BASE_SEED, DANGER_QUOTA, FAMILIES, NORMAL_QUOTA, REGIONS, ROOT_DPS, ROOT_BISECTION_STEPS, ROOT_MAX_DOUBLINGS, ULP_QUOTA
from .regions import label_sides
from ..mv_dims import expand, is_multivariate, split_quota
from .constants import SCALAR_CONFIGS
from . import multivariate

def _mp(x: F | int | mp.mpf) -> mp.mpf:
    if isinstance(x, F):
        return mp.mpf(x.numerator) / mp.mpf(x.denominator)
    return mp.mpf(x)

def _frac(x: mp.mpf) -> F:
    x = mp.mpf(x)
    if not mp.isfinite(x):
        raise ValueError('non-finite generator candidate')
    man, exp = x.man_exp
    man, exp = (abs(int(man)), int(exp))
    if x < 0:
        man = -man
    return F(man << exp) if exp >= 0 else F(man, 1 << -exp)

def _representable(x, fmt) -> bool:
    if not mp.isfinite(x):
        return False
    a = abs(x)
    if a == 0:
        return True
    return _mp(fmt.minimum_subnormal) / 2 <= a <= _mp(fmt.maximum)

def _rand(rng: random.Random) -> mp.mpf:
    return mp.mpf(rng.getrandbits(64)) / (1 << 64)

def _between(rng: random.Random, a, b) -> mp.mpf:
    return _mp(a) + (_mp(b) - _mp(a)) * _rand(rng)

def _logbetween(rng: random.Random, a, b) -> mp.mpf:
    return mp.exp(_between(rng, mp.log(_mp(a)), mp.log(_mp(b))))

def derive_seed(config: str, dtype: str, group: str, region: str) -> int:
    payload = json.dumps([BASE_SEED, config, dtype, group, region], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], 'big')

def _quantize(params: dict[str, list[mp.mpf | F]], dtype: str):
    fmt = get_format(dtype)
    return {name: [round_fraction(v if isinstance(v, F) else _frac(v), fmt) for v in values] for name, values in params.items()}

def _decode(bits, dtype):
    fmt = get_format(dtype)
    return {name: [exact_value(int(x), fmt) for x in values] for name, values in bits.items()}

def _sample_p(config_id: str, region: str, dtype: str, rng: random.Random):
    if is_multivariate(config_id):
        return multivariate.sample_p(config_id, region, dtype, rng)
    family = FAMILIES[config_id]
    fmt = get_format(dtype)
    with mp.workdps(ROOT_DPS):
        if family == 'prob':
            if region == 'R0':
                logit = _between(rng, -mp.log(99), mp.log(99))
                x = 1 / (1 + mp.exp(-logit))
            elif region == 'R1':
                x = _logbetween(rng, fmt.eps, F(1, 100))
            else:
                x = _logbetween(rng, fmt.minimum_subnormal, fmt.eps)
            if region != 'R0' and rng.randrange(2):
                x = 1 - x
            p = {'probs': [x]}
        elif family == 'logits':
            if region == 'R0':
                z = _between(rng, -mp.log(99), mp.log(99))
            elif region == 'R1':
                z = _between(rng, mp.log(99), 15)
            else:
                z = _between(rng, 15, 1000)
            if region != 'R0' and rng.randrange(2):
                z = -z
            p = {'logits': [z]}
        elif family == 'positive':
            boundaries = {'R0': (F(1, 100), F(100)), 'R1': (F(100), F(10000)), 'R2': (F(1, 10000), F(1, 100))}
            a, b = boundaries[region]
            names = ['rate'] if config_id in ('poisson', 'exponential') else ['concentration', 'rate'] if config_id == 'gamma' else ['concentration1', 'concentration0']
            p = {name: [_logbetween(rng, a, b)] for name in names}
        elif family == 'location_scale':
            scale = _logbetween(rng, F(1, 10000) if region == 'R2' else F(1, 100), F(1, 100) if region == 'R2' else F(100))
            loc = _between(rng, -10, 10) if region != 'R1' else scale * _between(rng, 100, 10000) * (1 if rng.randrange(2) else -1)
            p = {'loc': [loc], 'scale': [scale]}
        elif family == 'uniform':
            width = _logbetween(rng, F(1, 100), F(100))
            low = _between(rng, -10, 10) if region != 'R1' else width * _between(rng, 100, 10000) * (1 if rng.randrange(2) else -1)
            p = {'low': [low], 'high': [low + width]}
        elif family == 'pareto':
            p = {'scale': [_logbetween(rng, F(1, 100), F(100))], 'alpha': [_logbetween(rng, F(100), F(10000)) if region == 'R1' else _logbetween(rng, F(1, 10), F(100))]}
        else:
            raise NotImplementedError(f'multivariate generator extension point: {family}')
        if config_id.startswith('binomial_'):
            p['total_count'] = [F(rng.choice((1, 10, 100, 1000)))]
        return _quantize(p, dtype)

def _draw_ulp_p_region(rng: random.Random) -> str:
    roll = rng.randrange(100)
    return 'R0' if roll < 70 else 'R1' if roll < 85 else 'R2'

def _direction(config_id: str, rng: random.Random):
    param_names = [name for name, spec in get_config(config_id).PARAM_SPEC.items() if spec['differentiable']]
    chosen_type = rng.choice(('single', 'subset', 'all'))
    if chosen_type == 'single':
        chosen = [rng.choice(param_names)]
    elif chosen_type == 'all':
        chosen = list(param_names)
    else:
        chosen = rng.sample(param_names, rng.randrange(1, len(param_names) + 1))
    return (chosen_type, {name: rng.choice((-1, 1)) * _between(rng, F(1, 2), 1) for name in chosen})

def _perturb(config_id: str, p: dict[str, list[F]], weights, h: mp.mpf, region: str, rng_boundary: int):
    family = FAMILIES[config_id]
    q = {k: [_mp(v) for v in values] for k, values in p.items()}
    for name, weight in weights.items():
        x = _mp(p[name][0])
        d = h * weight
        if family == 'prob':
            z = mp.log(x / (1 - x))
            q[name][0] = 1 / (1 + mp.exp(-(z + d)))
        elif name in ('rate', 'scale', 'alpha', 'concentration', 'concentration1', 'concentration0'):
            q[name][0] = x * mp.exp(d)
        else:
            q[name][0] = x + d
    if family == 'uniform':
        low = _mp(p['low'][0])
        high = _mp(p['high'][0])
        width = high - low
        step = width * mp.expm1(h * abs(weights.get('high', 1)))
        if region == 'R2':
            if rng_boundary:
                q['low'] = [low - step]
                q['high'] = [high]
            else:
                q['low'] = [low]
                q['high'] = [high + step]
        else:
            q['low'] = [low - step / 2]
            q['high'] = [high + step / 2]
    elif family == 'pareto':
        ps = _mp(p['scale'][0])
        q['scale'] = [ps if region == 'R2' else ps * mp.exp(-h * abs(weights.get('scale', 1)))]
    return q

def _target(group: str, rng: random.Random) -> mp.mpf:
    from ..ref.constants import KL_BINS
    lo, hi = next(((a, b) for key, a, b in KL_BINS if key == group))
    return _logbetween(rng, lo, hi)

def _root_candidate(config_id, p, group, dtype, region, rng):
    config = get_config(config_id)
    fmt = get_format(dtype)
    if is_multivariate(config_id):
        kind, weights = multivariate.direction(config_id, p, rng)
        perturb = lambda h: multivariate.perturb(config_id, p, weights, h)
    else:
        kind, weights = _direction(config_id, rng)
        boundary = rng.randrange(2)
        perturb = lambda h: _perturb(config_id, p, weights, h, region, boundary)
    with mp.workdps(ROOT_DPS):
        t = _target(group, rng)

        def evaluate(h):
            q = perturb(h)
            if not all((_representable(v, fmt) for vs in q.values() for v in vs)):
                return None
            values = {k: [_frac(v) for v in vs] for k, vs in q.items()}
            if config.domain_ok(p, values) is not True:
                return None
            try:
                return config.kl_mpmath_alt(p, values, ROOT_DPS)
            except (ValueError, ZeroDivisionError, OverflowError):
                return None
        left = mp.mpf(0)
        right = mp.mpf('0.01')
        for _ in range(ROOT_MAX_DOUBLINGS):
            val = evaluate(right)
            if val is not None and mp.isfinite(val) and (val >= t):
                break
            right *= 2
        else:
            return (None, kind, 'ROOT_UNBRACKETED')
        for _ in range(ROOT_BISECTION_STEPS):
            mid = (left + right) / 2
            val = evaluate(mid)
            if val is None or not mp.isfinite(val):
                right = mid
            elif val < t:
                left = mid
            else:
                right = mid
            if right - left <= mp.mpf('1e-12') * right:
                break
        q = perturb((left + right) / 2)
        if is_multivariate(config_id):
            q = multivariate._quantize(q, dtype)
        else:
            q = _quantize(q, dtype)
        return (q, kind, 'OK')

def _neighbor(bits: int, fmt, delta: int) -> int | None:
    full = (1 << fmt.bits) - 1
    ordered = ~bits & full if bits & fmt.sign_mask else bits | fmt.sign_mask
    new = ordered + delta
    if not 0 <= new <= full:
        return None
    result = new & ~fmt.sign_mask if new & fmt.sign_mask else ~new & full
    return result if result & fmt.sign_mask - 1 < fmt.infinity_bits else None

def _ulp_candidate(p_bits, dtype, rng):
    fmt = get_format(dtype)
    keys = list(p_bits)
    keys = [name for name in keys if name != 'total_count']
    count = rng.randrange(1, len(keys) + 1)
    q = {k: list(v) for k, v in p_bits.items()}
    k = rng.randrange(1, 9)
    for name in rng.sample(keys, count):
        updated = _neighbor(q[name][0], fmt, k * rng.choice((-1, 1)))
        if updated is None:
            return (None, k)
        q[name][0] = updated
    return (q, k)

def _record(cert, pbits, qbits, region_info, direction, k, sample_id, group, sampled_p_region):
    return {'certificate': cert, 'p': pbits, 'q': qbits, 'regions': region_info, 'direction': direction, 'ulp_k': k, 'sample_id': sample_id, 'group': group, 'sampled_p_region': sampled_p_region}

def _signature(pbits, qbits):
    text = json.dumps([[(k, pbits[k]) for k in sorted(pbits)], [(k, qbits[k]) for k in sorted(qbits)]], separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()

def _open_dedup(root, run_id):
    path = root / 'data' / 'frozen' / run_id / '_dedup.sqlite'
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=120, isolation_level=None)
    db.execute('PRAGMA busy_timeout=120000')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS sample (config TEXT,dtype TEXT,signature TEXT,owner TEXT, PRIMARY KEY (config,dtype,signature))')
    return db

def _rebuild_dedup(root, run_id):
    db = _open_dedup(root, run_id)
    base = root / 'data' / 'frozen' / run_id
    try:
        for unit in sorted(base.glob('*/*/*.npz')):
            arrays, meta = load_unit(unit)
            pnames = [name[3:] for name in arrays if name.startswith('p__')]
            for i in range(len(arrays['sample_id'])):
                p = {name: [int(v) for v in arrays[f'p__{name}'][i]] for name in pnames}
                q = {name: [int(v) for v in arrays[f'q__{name}'][i]] for name in pnames}
                db.execute('INSERT OR IGNORE INTO sample VALUES (?,?,?,?)', (meta['config'], meta['dtype'], _signature(p, q), str(unit)))
    finally:
        db.close()

def _arrays(rows, config, dtype):
    utype = np.uint16 if dtype == 'fp16' else np.uint32
    out = {'sample_id': text_array([r['sample_id'] for r in rows]), 'direction': text_array([r['direction'] for r in rows]), 'ulp_k': np.asarray([r['ulp_k'] for r in rows], dtype=np.int16), 'region_p': text_array([r['regions'][0] for r in rows]), 'sampled_p_region': text_array([r['sampled_p_region'] for r in rows]), 'region_q': text_array([r['regions'][1] for r in rows]), 'region': text_array([r['regions'][2] for r in rows]), 'region_p_tags': text_array([','.join(r['regions'][3]) for r in rows]), 'region_q_tags': text_array([','.join(r['regions'][4]) for r in rows]), 'group': text_array([r['group'] for r in rows]), 'kl_lower': text_array([r['certificate'].kl.lower for r in rows]), 'kl_upper': text_array([r['certificate'].kl.upper for r in rows]), 'kl_truth_bits': np.asarray([r['certificate'].kl.truth_bits or 0 for r in rows], dtype=utype), 'kl_status': text_array([r['certificate'].kl.status.value for r in rows]), 'kl_group': text_array([r['certificate'].kl.group for r in rows]), 'kl_precision': np.asarray([r['certificate'].kl.precision_bits or 0 for r in rows], dtype=np.int32), 'kl_overflow': np.asarray([r['certificate'].kl.rounding_overflow for r in rows], dtype=np.bool_), 'sample_tag': text_array([r['certificate'].sample_tag for r in rows]), 'semantic_identity': np.asarray([r['certificate'].semantic_identity for r in rows], dtype=np.bool_), 'exact_bitwise_identity': np.asarray([r['certificate'].exact_bitwise_identity for r in rows], dtype=np.bool_), 'issues_json': text_array([json.dumps([issue.__dict__ for issue in r['certificate'].issues]) for r in rows]), 'xcheck': text_array([r['certificate'].xcheck.status for r in rows])}
    for side in ('P', 'Q'):
        key = 'p' if side == 'P' else 'q'
        for name, spec in config.PARAM_SPEC.items():
            size = int(np.prod(spec['shape']))
            out[f'{key}__{name}'] = np.asarray([r[key][name] for r in rows], dtype=utype).reshape(len(rows), size)
            if not spec['differentiable']:
                continue
            gs = [r['certificate'].gradients.get(side, {}).get(name, []) for r in rows]
            size = int(np.prod(spec['shape']))
            for field, func, dt in (('bits', lambda v: v.truth_bits or 0, utype), ('status', lambda v: v.status.value, np.str_), ('lower', lambda v: str(v.lower) if v.lower is not None else '', np.str_), ('upper', lambda v: str(v.upper) if v.upper is not None else '', np.str_), ('exact_zero', lambda v: v.exact_zero, np.bool_), ('nondiff', lambda v: v.status == CertStatus.NONDIFF, np.bool_), ('overflow', lambda v: v.rounding_overflow, np.bool_), ('precision', lambda v: v.precision_bits or 0, np.int32)):
                values = [[func(v) for v in vs] if vs else ['' if dt == np.str_ else 0] * size for vs in gs]
                out[f'grad__{side}__{name}__{field}'] = np.asarray(values, dtype=dt).reshape(len(rows), size)
    return out

def generate_cell(task):
    config_id, dtype, group, region, quota, run_id, root = task
    root = Path(root)
    path = root / 'data' / 'frozen' / run_id / config_id / dtype / f'{group}__{region}.npz'
    if path.exists() or Path(str(path) + '.manifest.json').exists():
        return {'path': str(path), 'skipped': True, 'count': verify(path).get('actual_count')}
    db = _open_dedup(root, run_id)
    db.execute('DELETE FROM sample WHERE owner=?', (str(path),))
    seed = derive_seed(config_id, dtype, group, region)
    rng = random.Random(seed)
    config = get_config(config_id)
    fmt = get_format(dtype)
    rows = []
    reasons = Counter()
    seen = set()
    start = time.monotonic()
    attempts = 0
    while len(rows) < quota and attempts < max(quota * ATTEMPT_MULTIPLIER, 50):
        if not rows and attempts >= INFEASIBLE_PROBE:
            break
        attempts += 1
        try:
            if group == 'ULP':
                p_region = multivariate.draw_ulp_p_region(config_id, rng) if is_multivariate(config_id) else _draw_ulp_p_region(rng)
            else:
                p_region = region
            p_bits = _sample_p(config_id, p_region, dtype, rng)
            p = _decode(p_bits, dtype)
            if group == 'identity':
                q_bits = {k: list(v) for k, v in p_bits.items()}
                direction = 'identity'
                k = 0
            elif group == 'ULP':
                q_bits, k = multivariate.ulp_candidate(config_id, p_bits, dtype, rng) if is_multivariate(config_id) else _ulp_candidate(p_bits, dtype, rng)
                direction = 'ulp'
            else:
                q_bits, direction, reason = _root_candidate(config_id, p, group, dtype, region, rng)
                k = 0
            if q_bits is None:
                reasons['INVALID_ULP' if group == 'ULP' else reason] += 1
                continue
            if group != 'identity' and p_bits == q_bits:
                reasons['ROUNDED_TO_IDENTICAL'] += 1
                continue
            q = _decode(q_bits, dtype)
            if config.domain_ok(p, q) is not True:
                reasons['DOMAIN'] += 1
                continue
            labels = label_sides(config_id, p, q, fmt, identity_group=group == 'identity')
            if group != 'ULP' and 'OUTSIDE' in (labels[0], labels[1]):
                reasons['OUTSIDE_DOMAIN'] += 1
                continue
            if group != 'ULP' and labels[2] != region:
                reasons['REGION_MISMATCH'] += 1
                continue
            key = _signature(p_bits, q_bits)
            if key in seen:
                reasons['DUPLICATE'] += 1
                continue
            cert = certify_sample(config_id, dtype, p_bits, q_bits, sample_type='identity' if group == 'identity' else 'ulp' if group == 'ULP' else 'nonzero')
            if cert.rejection_reason:
                reasons[cert.rejection_reason] += 1
                continue
            if cert.sample_tag == 'ZERO_KL_NONIDENTICAL':
                reasons['ZERO_KL_NONIDENTICAL'] += 1
                continue
            if cert.kl.status != CertStatus.CERTIFIED:
                reasons[cert.kl.status.value] += 1
                continue
            if group not in ('identity', 'ULP') and cert.kl.group != group:
                reasons['BIN_MISMATCH'] += 1
                continue
            if any((v.status not in (CertStatus.CERTIFIED, CertStatus.NONDIFF) for s in cert.gradients.values() for coord in s.values() for v in coord)):
                reasons['UNCERT_GRAD'] += 1
                continue
            inserted = db.execute('INSERT OR IGNORE INTO sample VALUES (?,?,?,?)', (config_id, dtype, key, str(path))).rowcount
            if inserted != 1:
                reasons['GLOBAL_DUPLICATE'] += 1
                continue
            seen.add(key)
            sample_id = hashlib.sha256(f'{config_id}/{dtype}/{group}/{region}/{len(rows)}'.encode()).hexdigest()[:24]
            rows.append(_record(cert, p_bits, q_bits, labels, direction, k, sample_id, group, p_region))
        except (ValueError, OverflowError, ZeroDivisionError, ArithmeticError) as exc:
            reasons[f'EXCEPTION:{type(exc).__name__}:{str(exc)[:80]}'] += 1
    status = 'COMPLETE' if len(rows) == quota else 'INSUFFICIENT_SAMPLES' if rows else 'INFEASIBLE' if attempts >= INFEASIBLE_PROBE else 'SMOKE_NO_ACCEPT'
    arrays = _arrays(rows, config, dtype)
    details = {'stage': 'generate', 'config': config_id, 'dtype': dtype, 'group': group, 'region': region, 'seed': str(seed), 'base_seed': BASE_SEED, 'quota': quota, 'actual_count': len(rows), 'attempts': attempts, 'attempt_limit': max(quota * ATTEMPT_MULTIPLIER, 50), 'status': status, 'rejections': dict(reasons), 'elapsed_seconds': time.monotonic() - start, 'parameter_shapes': {k: dict(v) for k, v in config.PARAM_SPEC.items()}, 'input_schema': 'p__/q__ components are row-major uint16/uint32 bit arrays'}
    save_unit(path, arrays, details, project_root=root)
    db.close()
    return {'path': str(path), 'status': status, 'count': len(rows), 'attempts': attempts, 'elapsed_seconds': details['elapsed_seconds']}

def build_tasks(configs, dtypes, per_cell, run_id, root):
    for config in expand(configs):
        if config not in CONFIGS:
            raise ValueError(f'Unknown configuration: {config}')
        mv_config = is_multivariate(config)
        regions = multivariate.regions_for(config) if mv_config else REGIONS
        share = (lambda n: split_quota(config, n)) if mv_config else lambda n: n
        for dtype in dtypes:
            for group in ('identity', 'B_c', *(f'B_{i}' for i in range(1, 15))):
                for region in regions:
                    quota = per_cell if per_cell is not None else share(NORMAL_QUOTA if region == 'R0' else DANGER_QUOTA)
                    yield (config, dtype, group, region, quota, run_id, str(root))
            yield (config, dtype, 'ULP', 'ALL', per_cell if per_cell is not None else share(ULP_QUOTA), run_id, str(root))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='+', default=list(SCALAR_CONFIGS), help='config ids (multivariate ids expand to their dimension sub-configs)')
    parser.add_argument('--dtypes', nargs='+', choices=('fp16', 'fp32'), default=['fp16', 'fp32'])
    parser.add_argument('--per-cell', type=int, default=None)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--root', type=Path, default=project_root())
    args = parser.parse_args(argv)
    if args.per_cell is not None and args.per_cell < 1:
        parser.error('--per-cell must be positive')
    tasks = list(build_tasks(args.configs, args.dtypes, args.per_cell, args.run_id, args.root))
    _rebuild_dedup(args.root, args.run_id)
    t = time.monotonic()
    completed = []
    if args.workers == 1:
        for task in tasks:
            result = generate_cell(task)
            completed.append(result)
            print(json.dumps(result), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(generate_cell, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(json.dumps(result), flush=True)
    seconds = time.monotonic() - t
    completed_rows = sum((item.get('count') or 0 for item in completed))
    print(json.dumps({'stage': 'generate', 'units': len(tasks), 'rows': completed_rows, 'seconds': seconds, 'projected_seconds_10k_per_cell': seconds * 10000 / max(1, args.per_cell or 10000)}, indent=2))
if __name__ == '__main__':
    main()
