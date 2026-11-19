from __future__ import annotations
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
Z = 1.959963984540054
UNITS = {'n': ('component', 'bad'), 'rn': ('row', 'rbad'), 'sn': ('sample', 'sbad')}
REDS = ('sum', 'mean', 'batchmean')
GENC = ('CANDIDATES', 'BAD_DIRECTION', 'ROUNDED_TO_IDENTICAL', 'OUT_OF_REGION', 'UNCERT_BIN', 'BIN_FULL', 'INPUT_ROUNDING_UNCERT', 'DUPLICATE', 'T_ROUNDED_TO_ZERO', 'OUT_OF_RANGE', 'GAP', 'OTHER_CELL', 'TARGET_UNREACHABLE', 'NO_REPRESENTABLE_STEP', 'DUPLICATE_OF_FROZEN', 'ACCEPTED_supp_interp', 'ACCEPTED_supp_ulp', 'ACCEPTED_supp_ulp_zeroed')

def wilson(k, n):
    if n <= 0:
        return (None, None)
    p = k / n
    z2 = Z * Z
    den = 1 + z2 / n
    c = (p + z2 / (2 * n)) / den
    r = Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return (max(0.0, c - r), min(1.0, c + r))

def quant(v):
    v = sorted(v)
    n = len(v)
    q = lambda p: v[min(n - 1, int(math.ceil(p * n)) - 1 if p > 0 else 0)]
    return dict(n=n, min=v[0], p50=q(0.5), p90=q(0.9), p99=q(0.99), max=v[-1])
REPRO = 'import numpy as np, torch, torch.nn.functional as F\na = np.load("{inputs}"); rows = {rows}; f = np.{npf}\nx = torch.from_numpy(a["x"][rows].view(f).copy()).requires_grad_(True); t = torch.from_numpy(a["t"][rows].view(f).copy()).requires_grad_(True)\nfor red in ("none", "sum", "mean", "batchmean"): print(red, F.kl_div(x, t, reduction=red, log_target={lt}))'

def reconcile(u, key, cnt, exp, methods_special):
    bad = []
    g = lambda k: cnt.get(k, 0)
    nb, rows, el = (exp['batches'], exp['rows'], exp['elements'])

    def chk(name, got, want):
        if got != want:
            bad.append(dict(unit=u, key=key, check=name, got=got, expected=want))
    chk('batches', g('batches'), nb)
    chk('E', g('E_CR|n') + g('E_NOT_SCORED') + g('E_UNCERT'), el)
    chk('E rows', g('E_CR|rn') <= rows, True)
    for red in REDS:
        chk(f'S:{red}', g(f'S_CR:{red}|n') + g(f'S_NOT_SCORED:{red}') + g(f'S_UNCERT:{red}'), nb)
        for side in ('gx', 'gt'):
            bg = sum((v for k, v in cnt.items() if k.startswith(f'BG:{side}:{red}:') and (k.endswith('|n') or k.endswith('|UNCERT'))))
            chk(f'G:{side}:{red}', g(f'G_CR:{side}:{red}|n') + g(f'G_NOT_SCORED:{side}:{red}') + g(f'G_UNCERT:{side}:{red}') + bg, el)
        for side in ('gx', 'gt', 'both'):
            chk(f'GV:{side}:{red}', g(f'GV_n:{side}:{red}') + g(f'GV_NOT_APPLICABLE:{side}:{red}') + g(f'GV_NO_OUTPUT:{side}:{red}'), rows)
    return bad

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--cfg', default='*', help='one configuration directory name (default: all)')
    ap.add_argument('--strict', action='store_true', help='exit with status 5 if any denominator does not reconcile or no score file is found')
    a = ap.parse_args(argv)
    root = Path(a.root)
    out = Path(a.out or root / 'report')
    out.mkdir(parents=True, exist_ok=True)
    metrics, pairs, quants, bnd, counts, cases, rec, units = ([], [], [], [], [], [], [], [])
    nfiles = 0
    for f in sorted(root.glob(f'{a.cfg}/score/*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        nfiles += 1
        base = dict(log_target=d['log_target'], dtype=d['dtype'], K=d['K'], layer=d['layer'], group=d['group'], region=d['region'])
        gc = dict(d['gen_counters'])
        units.append(dict(base, unit=d['unit'], n_rows=d['n_rows'], xt_shared_rows=d['xt_shared_rows'], status=str(gc.pop('STATUS', 'ok')).replace('INFEASIBLE', 'ZERO_ACCEPTED'), **{k: gc.get(k, 0) for k in GENC}))
        for key, r in d['results'].items():
            Bs, m = key.split('|')
            B = int(Bs[1:])
            cnt = r['cnt']
            row = dict(base, B=B, method=m)
            if m.startswith('PAIR:'):
                agg = defaultdict(Counter)
                for k, v in cnt.items():
                    met, cat = k.rsplit('|', 1)
                    agg[met][cat] = v
                for met, c in sorted(agg.items()):
                    pairs.append(dict(row, method=m[5:], metric=met, n=sum(c.values()), **{k: c.get(k, 0) for k in ('fixed', 'new_bad', 'persistent', 'both_good')}))
                continue
            if m != 'TRUTH':
                rec += reconcile(d['unit'], key, cnt, d['expected'][Bs], None)
            for k, v in sorted(cnt.items()):
                if '|' in k:
                    met, suf = k.rsplit('|', 1)
                    if met.startswith('BG:'):
                        _, side, red, st, cls = met.split(':')
                        bnd.append(dict(row, side=side, reduction=red, status=st, convention=cls, category=suf, count=v))
                        continue
                    if suf in UNITS:
                        unit, bk = UNITS[suf]
                        bad = cnt.get(f'{met}|{bk}', 0)
                        lo, hi = wilson(bad, v)
                        metrics.append(dict(row, metric=met, counting_unit=unit, n=v, events=bad, rate=bad / v, wilson95_lo=lo, wilson95_hi=hi, not_applicable=cnt.get(met + '|na', 0) if unit == 'component' else ''))
                    elif suf == 'na' and f'{met}|n' not in cnt:
                        metrics.append(dict(row, metric=met, counting_unit='component', n=0, events=0, rate='', wilson95_lo='', wilson95_hi='', not_applicable=v))
                elif k != 'batches':
                    counts.append(dict(row, counter=k, count=v))
            for k, v in sorted(r.get('stats', {}).items()):
                if v:
                    quants.append(dict(row, stat=k, **quant(v)))
            if m != 'TRUTH' and (r.get('top') or r.get('ccases')):
                rp = lambda rows: REPRO.format(inputs=f"{f.parent.parent.name}/inputs/{d['inputs']}", rows=rows, npf='float16' if d['dtype'] == 'fp16' else 'float32', lt=d['log_target'])
                cases.append(dict(row, unit=d['unit'], worst_by_sum_ulp=[dict(t, repro=rp(t['rows'])) for t in r['top'] if t['S_sum_ULP'] > 0 or t['E_maxULP'] > 0], level_C=[dict(t, repro=rp(t['rows'])) for t in r['ccases']]))

    def w(name, recs):
        if not recs:
            (out / name).write_text('', encoding='utf-8')
            return
        cols = list(recs[0].keys())
        with open(out / name, 'w', newline='', encoding='utf-8') as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            wr.writerows(recs)
    w('metrics.csv', metrics)
    w('pairs.csv', pairs)
    w('quantiles.csv', quants)
    w('boundary.csv', bnd)
    w('counts.csv', counts)
    w('units.csv', units)
    (out / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding='utf-8')
    (out / 'reconcile.json').write_text(json.dumps(dict(score_files=nfiles, mismatches=rec), indent=1), encoding='utf-8')
    print(f'aggregate: {nfiles} score files, {len(metrics)} metric lines, {len(pairs)} pair lines, reconcile mismatches {len(rec)}')
    if a.strict and (rec or not nfiles):
        import sys
        print(f'STRICT FAILURE in aggregate: {len(rec)} reconcile mismatches, {nfiles} score files', file=sys.stderr)
        sys.exit(5)
    return rec
if __name__ == '__main__':
    main()
