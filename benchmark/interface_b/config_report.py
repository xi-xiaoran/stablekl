import csv, json, sys, collections, subprocess
from pathlib import Path
root = Path(sys.argv[1])
cfg = sys.argv[2]
d = root / cfg
rep = Path(sys.argv[3]) if len(sys.argv) > 3 else root / 'report'
P = print
P(f'# Configuration summary: {cfg}\n')
P('## 1. Stages\n\n| stage | seconds | max RSS (MB) | host | result |\n|---|---|---|---|---|')
for f in sorted((d / 'logs').glob('*.json'), key=lambda p: p.stat().st_mtime):
    j = json.loads(f.read_text())
    if 'result' not in j:
        continue
    r = j['result']
    ex = {k: v for k, v in r.items() if k not in ('seconds', 'examples', 'environment') and (not k.startswith('XCHECK'))}
    P(f"| {f.stem} | {r.get('seconds', 0):.0f} | {j.get('maxrss_kb', 0) / 1024:.0f} | {j.get('host', '')} | {json.dumps(ex)[:230]} |")
    if 'environment' in r:
        P(f"\n  environment ({f.stem}): {r['environment']}\n")
P('\n## 2. Storage\n')
P('```\n' + subprocess.run(['du', '-sh'] + [str(d / x) for x in ('inputs', 'pending_s1t', 'truth', 'out', 'score') if (d / x).exists()], capture_output=True, text=True).stdout + '```')
units = [u for u in csv.DictReader(open(rep / 'units.csv', encoding='utf-8'))]
P('\n## 3. Sampling completeness\n')
tot = collections.Counter()
st = collections.Counter()
for u in units:
    tot[u['layer']] += int(u['n_rows'])
    st[u['layer'], u['status'].split('(')[0]] += 1
P('rows per layer:', dict(tot), ' total', sum(tot.values()))
P('\nunit status:', {f'{k[0]}:{k[1]}': v for k, v in sorted(st.items())})
P('\nunits below quota (every one is listed):\n\n| layer | group | region | rows | status | candidates | ROUNDED_TO_IDENTICAL | BAD_DIRECTION | UNCERT_BIN | BIN_FULL | OUT_OF_REGION | INPUT_ROUNDING_UNCERT |\n|---|---|---|---|---|---|---|---|---|---|---|---|')
for u in units:
    if u['status'] != 'ok':
        P(f"| {u['layer']} | {u['group']} | {u['region']} | {u['n_rows']} | {u['status']} | {u['CANDIDATES']} | {u['ROUNDED_TO_IDENTICAL']} | {u['BAD_DIRECTION']} | {u['UNCERT_BIN']} | {u['BIN_FULL']} | {u['OUT_OF_REGION']} | {u['INPUT_ROUNDING_UNCERT']} |")
gc = collections.Counter()
for u in units:
    for k in ('CANDIDATES', 'BAD_DIRECTION', 'ROUNDED_TO_IDENTICAL', 'OUT_OF_REGION', 'UNCERT_BIN', 'BIN_FULL', 'INPUT_ROUNDING_UNCERT', 'DUPLICATE', 'T_ROUNDED_TO_ZERO', 'OUT_OF_RANGE'):
        gc[k] += int(u[k])
    gc['xt_shared_rows'] += int(u['xt_shared_rows'])
P('\nrejection counters (all units):', dict(gc))
P('\n## 4. Certification and cross-check\n')
c = collections.Counter()
ex = []
for f in (d / 'truth').glob('*.manifest.json'):
    j = json.loads(f.read_text())
    c.update(j['counters'])
    ex += j.get('xcheck_examples', [])
for k in ('L', 'G', 'S', 'E2E'):
    n, u = (c.get(k + '_scored', 0), c.get(k + '_uncert', 0))
    P(f'- {k}: {n} values to certify, {u} not certified ({(100 * u / n if n else 0):.4f}%)')
P('- cross-check (mpmath, second algebraic form):', {k: v for k, v in sorted(c.items()) if k.startswith('XCHECK')})
P('- cross-check mismatches:', sum((v for k, v in c.items() if k.startswith('XCHECK') and k.endswith('_bad'))), ex[:5])
j = json.loads((rep / 'reconcile.json').read_text())
P(f"\n## 5. Denominators\n\nscore files {j['score_files']}; mismatches {len(j['mismatches'])} {j['mismatches'][:5]}")
und = collections.Counter()
err = collections.Counter()
for r in csv.DictReader(open(rep / 'counts.csv', encoding='utf-8')):
    k = r['counter'].split(':')[0]
    if 'UNDECIDED' in k or 'UNCERT' in k:
        und[k] += int(r['count'])
    if k in ('CALL_ERROR', 'BACKWARD_ERROR', 'GRAD_MISSING', 'NO_CALL') or 'NO_OUTPUT' in k:
        err[r['method'], k] += int(r['count'])
P('undecided / uncertified counters in scoring:', dict(und) or 'none')
P('exceptions:', {f'{a}:{b}': v for (a, b), v in err.items()} or 'none')
M = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])
for r in csv.DictReader(open(rep / 'metrics.csv', encoding='utf-8')):
    if r['layer'] in ('S1', 'S1t', 'S1s', 'S1ts') and r['counting_unit'] in ('component', 'sample'):
        k = (r['layer'], r['region'], r['group'], int(r['B']), r['method'], r['metric'], r['counting_unit'])
        M[k][0] += int(r['n'])
        M[k][1] += int(r['events'])

def pc(*k):
    n, b = M.get(k, (0, 0, 0, 0))[:2]
    return '—' if not n else f'{100 * b / n:.1f}%'
order = ['B_c'] + [f'B_{i}' for i in range(1, 15)]
sup = [u for u in units if u['layer'] in ('S1s', 'S1ts')]
if sup:
    P('\n## 5b. Supplementary cells (additional sampling; separate layers, never merged into S1 / S1t)\n\n| layer | group | region | gap | rows | status | candidates | interp | ulp | ulp_zeroed | OTHER_CELL | ROUNDED_TO_IDENTICAL | UNCERT_BIN |\n|' + '---|' * 13)
    for u in sup:
        P(f"| {u['layer']} | {u['group']} | {u['region']} | {u['GAP']} | {u['n_rows']} | {u['status']} | {u['CANDIDATES']} | {u['ACCEPTED_supp_interp']} | {u['ACCEPTED_supp_ulp']} | {u['ACCEPTED_supp_ulp_zeroed']} | {u['OTHER_CELL']} | {u['ROUNDED_TO_IDENTICAL']} | {u['UNCERT_BIN']} |")
for layer in ('S1', 'S1t', 'S1s', 'S1ts'):
    for region in ('R0', 'R1', 'R2'):
        for B in (1, 8):
            if not any((M.get((layer, region, g, B, 'A', 'S_CR:sum', 'component'), [0])[0] for g in order)):
                continue
            P(f'\n## 6. {layer} {region} B = {B}, per KL group (component level; n = batches)\n\n| group | n | A elem ≥8ULP | A sum notCR | A sum ≥8ULP | A sum rel≥1% | A mean notCR | A gt comp notCR | A gt vec L2≥1% | A LAYER_C (sample) | B64 sum notCR | B64 sum ≥8ULP | T_op<0 | repr rel≥1% | total rel≥1% (A) | total rel≥1% (B64) |\n|' + '---|' * 16)
            for g in order:
                n = M.get((layer, region, g, B, 'A', 'S_CR:sum', 'component'), [0])[0]
                a = lambda m, meth='A', u='component': pc(layer, region, g, B, meth, m, u)
                P(f"| {g} | {n} | {a('E_ULP8')} | {a('S_CR:sum')} | {a('S_ULP8:sum')} | {a('S_REL1:sum')} | {a('S_CR:mean')} | {a('G_CR:gt:sum')} | {a('GV_REL_L2_1:gt:sum')} | {a('LAYER_C', u='sample')} | {a('S_CR:sum', 'B64')} | {a('S_ULP8:sum', 'B64')} | {a('R_NEG', 'TRUTH')} | {a('R_REL1', 'TRUTH')} | {a('TOT_REL1:sum')} | {a('TOT_REL1:sum', 'B64')} |")
