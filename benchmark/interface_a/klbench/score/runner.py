from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import numpy as np
from ..fp import get_format
from ..fp.bits import bits_to_fraction
from ..fp.ulp import ulp_distance
from ..pipeline_io import load_unit, project_root, sha256
from ..gen.constants import CONFIG_DISTRIBUTIONS
Z95 = 1.959963984540054
REL_THRESHOLDS = {'KL_REL1': Fraction(1, 100), 'KL_REL10': Fraction(1, 10), 'KL_REL100': Fraction(1)}
COS_THRESHOLDS = {'G_COS_999': Decimal('0.999'), 'G_COS_99': Decimal('0.99'), 'G_COS_9': Decimal('0.9')}

def _dec(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)

def _d(bits, fmt):
    return bits_to_fraction(int(bits), fmt)

def _eq(a, b) -> bool:
    if a.kind != b.kind:
        return False
    if a.kind == 'nan':
        return False
    if a.kind == 'infinity':
        return a.sign == b.sign
    return a.value == b.value

def _normal(x, fmt):
    return x.kind == 'finite' and x.value != 0 and (abs(x.value) >= fmt.minimum_normal)

def _finite(x):
    return x.kind == 'finite'

def _cos_l2(got, truth):
    if not got or not truth or (not all((_finite(x) for x in got + truth))):
        return (None, None)
    with localcontext() as ctx:
        ctx.prec = 100
        g = [_dec(x.value) for x in got]
        t = [_dec(x.value) for x in truth]
        g2 = sum((v * v for v in g), Decimal(0))
        t2 = sum((v * v for v in t), Decimal(0))
        if t2 == 0:
            return (None, None)
        error2 = sum(((a - b) ** 2 for a, b in zip(g, t)), Decimal(0))
        rel = (error2 / t2).sqrt()
        if g2 == 0:
            return (rel, None)
        cosine = sum((a * b for a, b in zip(g, t)), Decimal(0)) / (g2 * t2).sqrt()
        return (rel, cosine)

def _wilson(event: int, n: int):
    if n <= 0:
        return (None, None)
    p = event / n
    z2 = Z95 ** 2
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    radius = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (center - radius, center + radius)

def _metric(result, metric, event, valid=True):
    result[metric] = bool(event) if valid else None

def score_one(frozen, raw, idx, dtype):
    fmt = get_format(dtype)
    status = str(raw['status'][idx])
    failure = status not in ('OK', 'UNSUPPORTED_CPU')
    unsupported = status == 'UNSUPPORTED_CPU'
    metrics = {}
    ulps = {}
    stats = {}
    layer = {'A': False, 'B': False, 'C': False}
    kl_cert = str(frozen['kl_status'][idx]) == 'CERTIFIED'
    kl_truth = _d(frozen['kl_truth_bits'][idx], fmt) if kl_cert else None
    kl_out = _d(raw['kl_bits'][idx], fmt) if status in ('OK', 'BACKWARD_ERROR', 'BACKWARD_UNSUPPORTED_CPU') else None
    if kl_cert and (not unsupported):
        cr = kl_out is not None and _eq(kl_out, kl_truth)
        _metric(metrics, 'KL_CR', not cr)
        layer['A'] |= not cr
        if kl_out is not None and _finite(kl_out) and _finite(kl_truth):
            dist = ulp_distance(int(raw['kl_bits'][idx]), int(frozen['kl_truth_bits'][idx]), fmt)
            ulps['KL_ULP'] = dist
            for n in (0, 1, 2):
                _metric(metrics, f'KL_ULP_GT{n}', dist > n)
            _metric(metrics, 'KL_ULP8', dist >= 8)
            layer['B'] |= dist >= 8
            if _normal(kl_truth, fmt):
                err = abs(kl_out.value - kl_truth.value) / abs(kl_truth.value)
                for name, threshold in REL_THRESHOLDS.items():
                    bad = err >= threshold
                    _metric(metrics, name, bad)
                    layer['B'] |= bad
                if err > 0:
                    stats['log_relative_error'] = math.log10(float(err))
            else:
                for name in REL_THRESHOLDS:
                    metrics[name] = None
                stats['precision_limited'] = kl_truth.value != 0
            if frozen['kl_lower'][idx] and frozen['kl_upper'][idx]:
                with localcontext() as ctx:
                    ctx.prec = 100
                    l = Fraction(str(frozen['kl_lower'][idx]))
                    u = Fraction(str(frozen['kl_upper'][idx]))
                    distance = max(Fraction(0), l - kl_out.value, kl_out.value - u)
                    stats['abs_error_lower_bound'] = str(distance)
                    stats['abs_error_upper_bound'] = str(max(abs(kl_out.value - l), abs(kl_out.value - u)))
                    if l > 0:
                        stats['relative_Kstar_lower_bound'] = str(distance / u)
                        stats['relative_Kstar_upper_bound'] = str(max(abs(kl_out.value - l) / l, abs(kl_out.value - u) / u))
        else:
            metrics['KL_ULP8'] = None
            if kl_out is not None and _finite(kl_out) != _finite(kl_truth):
                stats['KL_ULP_CLASS'] = 'ULP_INF'
        if kl_out is not None:
            flags = {'KL_NEG': kl_out.kind == 'finite' and kl_out.value < 0 or (kl_out.kind == 'infinity' and kl_out.sign < 0), 'KL_NAN': kl_out.kind == 'nan', 'KL_INF': kl_out.kind == 'infinity' and _finite(kl_truth), 'KL_FALSE_ZERO': kl_out.is_zero and (not kl_truth.is_zero), 'KL_NONZERO_AT_IDENTITY': str(frozen['group'][idx]) == 'identity' and (not kl_out.is_zero)}
            for name, bad in flags.items():
                _metric(metrics, name, bad)
                layer['C'] |= bad
        else:
            for name in ('KL_NEG', 'KL_NAN', 'KL_INF', 'KL_FALSE_ZERO', 'KL_NONZERO_AT_IDENTITY'):
                metrics[name] = False
    elif unsupported:
        metrics['KL_CR'] = None
    else:
        metrics['KL_CR'] = None
    stats['ROUNDING_OVERFLOW'] = bool(frozen['kl_overflow'][idx])
    gradients = {'P': ([], []), 'Q': ([], []), 'PQ': ([], [])}
    for key in frozen:
        if not key.startswith('grad__') or not key.endswith('__bits'):
            continue
        parts = key.split('__')
        if len(parts) != 4:
            continue
        _, side, name, _ = parts
        truth_bits = frozen[key][idx]
        statuses = frozen[f'grad__{side}__{name}__status'][idx]
        exact_zero = frozen[f'grad__{side}__{name}__exact_zero'][idx]
        output_key = f'grad__{side}__{name}__bits'
        output_status_key = f'grad__{side}__{name}__status'
        output_status = str(raw[output_status_key][idx]) if output_status_key in raw else 'GRAD_MISSING'
        grad_unsupported = unsupported or status == 'BACKWARD_UNSUPPORTED_CPU' or output_status == 'BACKWARD_UNSUPPORTED_CPU'
        raw_values = raw.get(output_key)
        for j, truth_bit in enumerate(truth_bits):
            if str(statuses[j]) != 'CERTIFIED':
                continue
            if grad_unsupported:
                continue
            t = _d(truth_bit, fmt)
            y = _d(raw_values[idx, j], fmt) if raw_values is not None and output_status == 'OK' else None
            suffix = f'{side}.{name}[{j}]'
            cr = y is not None and _eq(y, t)
            _metric(metrics, f'G_CR:{suffix}', not cr, not unsupported)
            layer['A'] |= not cr and (not unsupported)
            if y is not None and _finite(y) and _finite(t):
                component_ulp = ulp_distance(int(raw_values[idx, j]), int(truth_bit), fmt)
                ulps[f'G_ULP:{suffix}'] = component_ulp
                for n in (0, 1, 2):
                    _metric(metrics, f'G_ULP_GT{n}:{suffix}', component_ulp > n)
            if y is not None:
                checks = {'G_NAN': y.kind == 'nan', 'G_INF': y.kind == 'infinity' and _finite(t), 'G_SIGN': _finite(y) and _finite(t) and (y.value != 0) and (t.value != 0) and ((y.value < 0) != (t.value < 0)), 'G_VANISH': y.is_zero and (not t.is_zero), 'G_SPURIOUS': bool(exact_zero[j]) and (not y.is_zero)}
                for label, bad in checks.items():
                    _metric(metrics, f'{label}:{suffix}', bad)
                    layer['C'] |= bad
                for key_vec in (side, 'PQ'):
                    gradients[key_vec][0].append(y)
                    gradients[key_vec][1].append(t)
            else:
                for key_vec in (side, 'PQ'):
                    gradients[key_vec][0].append(None)
                    gradients[key_vec][1].append(t)
    for key, (got, truth) in gradients.items():
        if not truth:
            continue
        if any((v is None for v in got)):
            for flag in ('G_REL_L2_1', 'G_COS_999', 'G_COS_99', 'G_COS_9', 'G_REVERSED', 'G_VANISH_VEC'):
                metrics[f'{flag}:{key}'] = None
            continue
        rel, cos = _cos_l2(got, truth)
        _metric(metrics, f'G_REL_L2_1:{key}', rel >= Decimal('0.01') if rel is not None else False, rel is not None)
        if rel is not None:
            layer['B'] |= rel >= Decimal('0.01')
        for label, threshold in COS_THRESHOLDS.items():
            _metric(metrics, f'{label}:{key}', cos < threshold if cos is not None else False, cos is not None)
            if cos is not None:
                layer['B'] |= cos < threshold
        _metric(metrics, f'G_REVERSED:{key}', cos < 0 if cos is not None else False, cos is not None)
        if cos is not None:
            layer['C'] |= cos < 0
        nonzero_truth = any((t.value != 0 for t in truth))
        vanish = all((y.is_zero for y in got)) and nonzero_truth
        _metric(metrics, f'G_VANISH_VEC:{key}', vanish)
        layer['C'] |= vanish
    for l, bad in layer.items():
        _metric(metrics, f'LAYER_{l}', bad, not unsupported)
    return (metrics, ulps, stats)

def _summary_metrics(rows):
    names = sorted({key for metrics, _, _ in rows for key in metrics})
    records = []
    for name in names:
        values = [metrics.get(name) for metrics, _, _ in rows]
        applicable = [v for v in values if v is not None]
        n = len(applicable)
        bad = sum(applicable)
        lo, hi = _wilson(bad, n)
        records.append({'metric': name, 'denominator': n, 'events': bad, 'rate': bad / n if n else None, 'wilson95_lo': lo, 'wilson95_hi': hi, 'not_applicable': len(values) - n})
    for metric in sorted({key for _, ulps, _ in rows for key in ulps}):
        values = [ulp[metric] for _, ulp, _ in rows if metric in ulp]
        if values:
            records.append({'metric': metric + '_STATS', 'denominator': len(values), 'median': statistics.median(values), 'p90': float(np.percentile(values, 90)), 'p99': float(np.percentile(values, 99)), 'max': max(values)})
    return records

def _repro(config, dtype, frozen, index, shapes=None):
    dtype_numpy = 'np.float16' if dtype == 'fp16' else 'np.float32'
    bits_dtype = 'np.uint16' if dtype == 'fp16' else 'np.uint32'
    lines = ['import numpy as np', 'import torch', 'from torch.distributions.kl import kl_divergence', f'target_dtype={dtype_numpy}']
    for side in ('p', 'q'):
        lines.append(f'{side}_kw={{}}')
        for key in frozen:
            if not key.startswith(side + '__'):
                continue
            name = key.split('__', 1)[1]
            raw = [int(x) for x in frozen[key][index]]
            shape = tuple((shapes or {}).get(name, {}).get('shape', (len(raw),)))
            reshape = f'.reshape({shape!r})' if len(shape) > 1 else ''
            lines.append(f'{side}_kw[{name!r}]=torch.from_numpy(np.array({raw},dtype={bits_dtype}).view(target_dtype){reshape}.copy())')
    constructor = CONFIG_DISTRIBUTIONS[config]
    lines.append(f'p=torch.distributions.{constructor}(**p_kw)')
    lines.append(f'q=torch.distributions.{constructor}(**q_kw)')
    lines.append('print(kl_divergence(p,q))')
    return '\n'.join(lines)

def _write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in records for k in r})
    with path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=cols)
        writer.writeheader()
        writer.writerows(records)

def _gradient_coverage(frozen, raw):
    rows = np.asarray(raw['status']).astype(str)
    eligible = 0
    excluded = 0
    for key, truth in frozen.items():
        if not key.startswith('grad__') or not key.endswith('__status'):
            continue
        if key.count('__') != 3:
            continue
        valid = np.asarray(truth).astype(str) == 'CERTIFIED'
        eligible += int(valid.sum())
        backward = np.asarray(raw[key]).astype(str) if key in raw else np.full(len(rows), 'GRAD_MISSING')
        unsupported = np.isin(rows, ['UNSUPPORTED_CPU', 'BACKWARD_UNSUPPORTED_CPU']) | (backward == 'BACKWARD_UNSUPPORTED_CPU')
        excluded += int((valid & unsupported[:, None]).sum())
    covered = eligible - excluded
    return {'eligible_components': eligible, 'covered_components': covered, 'unsupported_components': excluded, 'coverage': covered / eligible if eligible else None}

def _cert_failure_count(manifest):
    return sum((int(count) for reason, count in manifest.get('rejections', {}).items() if reason.startswith('UNCERT_') or reason == 'XCHECK_MISMATCH'))
MAX_C_CASES_PER_UNIT = 20

def score_run(root, run_id, write_sample_csv=False):
    root = Path(root)
    froot = root / 'data' / 'frozen' / run_id
    result = root / 'results' / 'score' / run_id
    result.mkdir(parents=True, exist_ok=True)
    metric_records = []
    pair_records = []
    case_records = []
    sample_records = []
    grouped = defaultdict(list)
    histograms = Counter()
    coverage_records = []
    coverage_lookup = {}
    cert_failure_lookup = {}
    all_stats = Counter()
    started = time.monotonic()
    inputs = sorted(froot.glob('*/*/*.npz'))
    for frozen_path in inputs:
        frozen, meta = load_unit(frozen_path)
        config = meta['config']
        dtype = meta['dtype']
        group = meta['group']
        region = meta['region']
        cert_failure_lookup[config, dtype, group, region] = _cert_failure_count(meta)
        raw_root = root / 'results' / 'raw' / run_id / config / dtype
        evaluated = {}
        for method in ('A', 'B32', 'B64') if dtype == 'fp16' else ('A', 'B64'):
            raw_path = raw_root / f'{frozen_path.stem}__{method}.npz'
            if not raw_path.exists():
                continue
            raw, raw_meta = load_unit(raw_path)
            if raw_meta['input_sha256'] != sha256(frozen_path):
                raise ValueError(f'Input hash mismatch {raw_path}')
            if not np.array_equal(raw['sample_id'], frozen['sample_id']):
                raise ValueError(f'sample_id mismatch: {raw_path}')
            coverage = _gradient_coverage(frozen, raw)
            coverage_lookup[config, dtype, group, region, method] = coverage
            coverage_records.append({'config': config, 'dtype': dtype, 'group': group, 'region': region, 'method': method, **coverage})
            evaluated[method] = [score_one(frozen, raw, i, dtype) for i in range(len(frozen['sample_id']))]
            for i, (sample_metrics, sample_ulps, sample_stats) in enumerate(evaluated[method]):
                if write_sample_csv:
                    sample_records.append({'config': config, 'dtype': dtype, 'group': group, 'region': region, 'method': method, 'sample_id': str(frozen['sample_id'][i]), 'metrics_json': json.dumps(sample_metrics, sort_keys=True), 'ulp_json': json.dumps(sample_ulps, sort_keys=True), 'statistics_json': json.dumps(sample_stats, sort_keys=True)})
                if 'log_relative_error' in sample_stats:
                    bucket = math.floor(sample_stats['log_relative_error'] * 2) / 2
                    histograms[config, dtype, group, region, method, bucket] += 1
            for aggregate in _summary_metrics(evaluated[method]):
                r = {'config': config, 'dtype': dtype, 'group': group, 'region': region, 'method': method, 'dimension': sum((int(np.prod(v['shape'])) for v in meta['parameter_shapes'].values())), **aggregate}
                uncert = sum((v for k, v in meta.get('rejections', {}).items() if k.startswith('UNCERT')))
                if r['metric'].startswith(('KL_CR', 'G_CR', 'LAYER_')):
                    n = r.get('denominator', 0)
                    e = r.get('events', 0)
                    r['error_lower_bound'] = e / (n + uncert) if n + uncert else None
                    r['error_upper_bound'] = (e + uncert) / (n + uncert) if n + uncert else None
                    r['uncertified_count'] = uncert
                metric_records.append(r)
                grouped[config, dtype, group, region, method].append(r)
            all_stats[method] += len(evaluated[method])
            severity = []
            for i, (metrics, ulps, stats) in enumerate(evaluated[method]):
                if 'KL_ULP' in ulps:
                    severity.append((ulps['KL_ULP'], i))
            high = {i for _, i in sorted(severity, reverse=True)[:20]}
            high.update([i for i, (metrics, _, _) in enumerate(evaluated[method]) if metrics.get('LAYER_C')][:MAX_C_CASES_PER_UNIT])
            for i in sorted(high):
                metrics, ulps, _ = evaluated[method][i]
                case_records.append({'config': config, 'dtype': dtype, 'group': group, 'region': region, 'method': method, 'sample_id': str(frozen['sample_id'][i]), 'KL_ULP': ulps.get('KL_ULP'), 'C_metrics': json.dumps([key for key, val in metrics.items() if val and (key.startswith('KL_') or key.startswith('G_'))]), 'repro': _repro(config, dtype, frozen, i, meta.get('parameter_shapes'))})
        if 'A' in evaluated:
            for method in ('B32', 'B64'):
                if method not in evaluated:
                    continue
                left = evaluated['A']
                right = evaluated[method]
                metrics = {key for d, _, _ in left for key in d}
                metrics |= {key for d, _, _ in right for key in d}
                for metric in metrics:
                    pair = Counter()
                    for (a, _, _), (b, _, _) in zip(left, right):
                        x = a.get(metric)
                        y = b.get(metric)
                        if x is None or y is None:
                            continue
                        pair['persistent' if x and y else 'fixed' if x and (not y) else 'new_bad' if y and (not x) else 'both_good'] += 1
                    pair_records.append({'config': config, 'dtype': dtype, 'group': group, 'region': region, 'method': method, 'metric': metric, 'n': sum(pair.values()), **{key: pair[key] for key in ('fixed', 'new_bad', 'persistent', 'both_good')}})
    _write_csv(result / 'unit_metrics.csv', metric_records)
    if write_sample_csv:
        _write_csv(result / 'sample_metrics.csv', sample_records)
    _write_csv(result / 'relative_error_histogram.csv', [{'config': k[0], 'dtype': k[1], 'group': k[2], 'region': k[3], 'method': k[4], 'log10_bin_lower': k[5], 'events': v} for k, v in sorted(histograms.items())])
    _write_csv(result / 'pairs.csv', pair_records)
    aggregate_pairs = []
    for level in ('config_dtype_group', 'config_dtype_overall'):
        reduced = defaultdict(Counter)
        for row in pair_records:
            key = (row['config'], row['dtype'], row['method'], row['metric'], row['group'] if level == 'config_dtype_group' else 'ALL', 'ALL')
            for k in ('fixed', 'new_bad', 'persistent', 'both_good'):
                reduced[key][k] += row.get(k, 0)
        for key, counts in reduced.items():
            aggregate_pairs.append({'scope': level, 'config': key[0], 'dtype': key[1], 'method': key[2], 'metric': key[3], 'group': key[4], 'region': key[5], 'n': sum(counts.values()), **dict(counts)})
    _write_csv(result / 'paired_aggregate.csv', aggregate_pairs)
    _write_csv(result / 'gradient_coverage.csv', coverage_records)
    _write_csv(result / 'worst_cases.csv', case_records)
    summary_lines = ['# KL benchmark summary', '', 'Counts below are per sampled cell and **not population prevalence**.', '', f'Frozen cells: {len(inputs)}; scored rows by method: {dict(all_stats)}', '', 'Cert failures count only UNCERT_* / XCHECK_MISMATCH rejections in the frozen manifest.', 'Gradient coverage excludes UNSUPPORTED_CPU and BACKWARD_UNSUPPORTED_CPU; GRAD_MISSING and ordinary BACKWARD_ERROR are covered but inaccurate.', '']
    primary = []
    configs = sorted({(r['config'], r['dtype']) for r in metric_records})
    group_order = ['identity', 'B_c', *(f'B_{i}' for i in range(1, 15)), 'ULP']
    for config, dtype in configs:
        summary_lines.extend([f'## {config} / {dtype}', '', '| Group | Region | Cert failures | A KL_CR error | B32 KL_CR error | B64 KL_CR error | A grad coverage | B32 grad coverage | B64 grad coverage |', '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
        for group in group_order:
            for region in ('ALL',) if group == 'ULP' else ('R0', 'R1', 'R2'):
                rec = {'config': config, 'dtype': dtype, 'group': group, 'region': region}
                display = []
                for method in ('A', 'B32', 'B64'):
                    values = [r for r in grouped.get((config, dtype, group, region, method), []) if r['metric'] == 'KL_CR']
                    if not values:
                        display.append('—')
                        continue
                    v = values[0]
                    rate = v.get('rate')
                    display.append(f"{rate:.2%} ({v['events']}/{v['denominator']})" if rate is not None else 'N/A')
                    primary.append({**rec, 'method': method, **v})
                grad_display = []
                for method in ('A', 'B32', 'B64'):
                    coverage = coverage_lookup.get((config, dtype, group, region, method))
                    if coverage is None:
                        grad_display.append('—')
                    elif coverage['coverage'] is None:
                        grad_display.append('N/A')
                    else:
                        grad_display.append(f"{coverage['coverage']:.2%} ({coverage['covered_components']}/{coverage['eligible_components']})")
                summary_lines.append('| ' + ' | '.join([group, region, str(cert_failure_lookup.get((config, dtype, group, region), '—')), *display, *grad_display]) + ' |')
        summary_lines.append('')
    _write_csv(result / 'main_table.csv', primary)
    (result / 'summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')
    info = {'stage': 'M4_score', 'run_id': run_id, 'frozen_units': len(inputs), 'metric_rows': len(metric_records), 'sample_rows': len(sample_records), 'pair_rows': len(pair_records), 'severe_cases': len(case_records), 'elapsed_seconds': time.monotonic() - started, 'source': 'immutable NPZ, individual manifests supply hashes'}
    (result / 'score_manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return info

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--root', type=Path, default=project_root())
    parser.add_argument('--sample-csv', action='store_true', help='also write per-sample CSV (large)')
    a = parser.parse_args(argv)
    print(json.dumps(score_run(a.root, a.run_id, a.sample_csv), indent=2))
if __name__ == '__main__':
    main()
