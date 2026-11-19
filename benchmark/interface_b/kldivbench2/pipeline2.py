from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter
from fractions import Fraction as Fr
from pathlib import Path
import numpy as np
from kldivbench import store
from kldivbench.pipeline import BS, UINT, CALL, GRAD, REDS4, unit_files, load_out, load_truth

def p2_sources():
    from stablekl import interface_b as kldivfix
    d = Path(__file__).resolve().parent
    return {p.name: store.sha256(p) for p in sorted(d.glob('*.py')) + [Path(kldivfix.__file__).resolve()]}
REG_CAP = 20000

def stage_run2(root, root2, lt, dt, K, shard=None, device='cpu', **_):
    from . import torchrun2 as R
    ut = UINT[dt]
    t0 = time.time()
    nbt = 0
    cerr = Counter()
    for p in unit_files(root / 'inputs', shard):
        a, meta = store.load(p)
        n = meta['n_rows']
        for B in BS:
            dst = root2 / 'out' / f'{p.stem}__B{B}.npz'
            if store.exists(dst):
                continue
            fr, fm = store.load(root / 'out' / f'{p.stem}__B{B}.npz')
            bl = fr['rows'].tolist()
            nb = len(bl)
            N = B * K
            arr = {'rows': fr['rows']}
            errs = []
            for m in R.METHODS[dt]:
                arr[m + '__cs'] = np.zeros((nb, 4), np.uint8)
                arr[m + '__gs'] = np.zeros((nb, 4), np.uint8)
                arr[m + '__dtype_ok'] = np.ones(nb, np.uint8)
                for red in REDS4:
                    arr[f'{m}__{red}'] = np.zeros((nb, N) if red == 'none' else nb, ut)
                    arr[f'{m}__{red}_gx'] = np.zeros((nb, N), ut)
                    arr[f'{m}__{red}_gt'] = np.zeros((nb, N), ut)
            for k, rows in enumerate(bl):
                res = R.run_batch(a['x'][rows].tolist(), a['t'][rows].tolist(), dt, lt, device=device)
                for m, o in res.items():
                    arr[m + '__dtype_ok'][k] = o['dtype_ok']
                    for j, red in enumerate(REDS4):
                        arr[m + '__cs'][k, j] = CALL.index(o['call_status'][red])
                        arr[m + '__gs'][k, j] = GRAD.index(o['grad_status'][red])
                        if o[red] is not None:
                            arr[f'{m}__{red}'][k] = o[red]
                        if o[red + '_gx'] is not None:
                            arr[f'{m}__{red}_gx'][k] = o[red + '_gx']
                            arr[f'{m}__{red}_gt'][k] = o[red + '_gt']
                    if o['errors']:
                        cerr['uncertified_rounding'] += sum(('UncertifiedRounding' in v for v in o['errors'].values()))
                        if len(errs) < 20:
                            errs.append(dict(batch=k, method=m, **o['errors']))
            cerr['call_or_backward_errors'] += sum((int((arr[m + '__cs'] != 0).sum()) + int((arr[m + '__gs'] != 0).sum()) for m in R.METHODS[dt]))
            store.save(dst, arr, dict(unit=meta['unit'], B=B, K=K, n_batches=nb, log_target=lt, dtype=dt, stage='out2', methods=list(R.METHODS[dt]), environment=R.environment(device), errors=errs, inputs_sha256=meta['sha256'], frozen_out_sha256=fm['sha256'], p2_source_sha256=p2_sources()))
            nbt += nb
    from stablekl.interface_b import STATS
    return {'batches': nbt, 'seconds': time.time() - t0, 'ext': dict(STATS), 'device': device, 'call_or_backward_errors': cerr['call_or_backward_errors'], 'uncertified_rounding': cerr['uncertified_rounding']}

def stage_replay2(root, root2, lt, dt, K, shard=None, device='cpu', **_):
    from . import torchrun2 as R
    c = Counter()
    t0 = time.time()
    ex = []
    for p in unit_files(root / 'inputs', shard):
        a, meta = store.load(p)
        for B in BS:
            o, om = store.load(root2 / 'out' / f'{p.stem}__B{B}.npz')
            for k, rows in enumerate(o['rows'].tolist()):
                res = R.run_batch(a['x'][rows].tolist(), a['t'][rows].tolist(), dt, lt, device=device)
                c['batches'] += 1
                bad = []
                if os.environ.get('KLDIV_INJECT_REPLAY_DIFF') and c['batches'] == 1:
                    m0 = om['methods'][0]
                    res[m0]['sum'] = (res[m0]['sum'] or 0) ^ 1
                for m, r in res.items():
                    old = load_out(o, m, k)
                    for key in old:
                        c['fields'] += 1
                        if old[key] != r.get(key):
                            bad.append(f'{m}:{key}')
                if bad:
                    c['batches_differ'] += 1
                    if len(ex) < 20:
                        ex.append(dict(unit=meta['unit'], B=B, batch=k, fields=bad[:10]))
    return dict(c, batches_differ=c['batches_differ'], examples=ex, environment=R.environment(device), seconds=time.time() - t0)

def _ctx(a, meta, tr, k, fmt, lt, dt, with_logits=True):
    from kldivbench import score as S
    from klbench.fp import exact_value
    rows = tr['rows'][k].tolist()
    xb = [a['x'][i].tolist() for i in rows]
    tb = [a['t'][i].tolist() for i in rows]
    logits = meta['has_logits'] and with_logits
    zp = [[exact_value(int(b), dt) for b in a['zp'][i]] for i in rows] if logits else None
    zq = [[exact_value(int(b), dt) for b in a['zq'][i]] for i in rows] if logits else None
    return (S.TruthCtx(xb, tb, fmt, lt, zp, zq), rows)

def stage_score2(root, root2, lt, dt, K, shard=None, **_):
    from kldivbench import score as S
    from klbench.fp import get_format
    from . import score2 as S2
    fmt = get_format(dt)
    t0 = time.time()
    nu = 0
    for p in unit_files(root / 'inputs', shard):
        dst = root2 / 'score' / (p.stem + '.json')
        if dst.exists():
            continue
        a, meta = store.load(p)
        res = {}
        expected = {}
        h1 = {}
        for B in BS:
            tr, tm = store.load(root / 'truth' / f'{p.stem}__B{B}.npz')
            oA, omA = store.load(root / 'out' / f'{p.stem}__B{B}.npz')
            o, om = store.load(root2 / 'out' / f'{p.stem}__B{B}.npz')
            if tm['inputs_sha256'] != meta['sha256'] or om['inputs_sha256'] != meta['sha256'] or omA['inputs_sha256'] != meta['sha256']:
                raise ValueError(f'truth / output do not belong to these inputs: {p}')
            if not ((tr['rows'] == o['rows']).all() and (tr['rows'] == oA['rows']).all()):
                raise ValueError(f'batch layout differs: {p}')
            new = om['methods']
            methods = ['A'] + new
            accs = {m: S.Acc() for m in methods}
            pairs = {m: Counter() for m in new}
            nb = len(tr['rows'])
            hc = {m: Counter() for m in new}
            regs = {m: [] for m in new}
            exs = {m: [] for m in new}
            ev = Counter()
            for k in range(nb):
                tc, rows = _ctx(a, meta, tr, k, fmt, lt, dt)
                truth = load_truth(tr, k)
                if [v.status for v in tc.L] != truth['L_st'] or tc.S.status != truth['S_st']:
                    raise ValueError(f'frozen status differs from the recomputed status: {p} batch {k}')
                outs = {'A': load_out(oA, 'A', k)}
                outs.update({m: load_out(o, m, k) for m in new})
                flags = {}
                for m in methods:
                    flags[m], _ = S.score_method(accs[m], tc, truth, outs[m], fmt)
                for m in new:
                    S.pair(pairs[m], flags['A'], flags[m])
                    S2.h1_batch(hc[m], regs[m], exs[m], tc, truth, outs['A'], outs[m], fmt, dict(B=B, batch=k))
                S2.eimpl_batch(ev, tc, truth, outs, fmt)
            pack = lambda acc: dict(cnt=dict(acc.cnt), stats={k: v for k, v in acc.stats.items()}, top=[], ccases=acc.ccases)
            for m in methods:
                res[f'B{B}|{m}'] = pack(accs[m])
            for m in new:
                res[f'B{B}|PAIR:{m}'] = dict(cnt=dict(pairs[m]))
            expected[f'B{B}'] = dict(batches=nb, rows=nb * B, elements=nb * B * K, methods=methods)
            h1[f'B{B}'] = dict(evec=dict(ev), methods={m: dict(cnt=dict(hc[m]), n_regressions=len(regs[m]), regressions=regs[m][:REG_CAP], new_bad_examples=exs[m]) for m in new})
        common = dict(unit=meta['unit'], layer=meta['layer'], group=meta['group'], region=meta['region'], log_target=lt, dtype=dt, K=K, n_rows=meta['n_rows'], inputs=str(p.name), inputs_sha256=meta['sha256'])
        store.save_json(root2 / 'h1' / (p.stem + '.json'), dict(common, h1=h1))
        store.save_json(dst, dict(common, xt_shared_rows=0, gen_counters=meta['counters'], expected=expected, results=res))
        nu += 1
    return {'units': nu, 'seconds': time.time() - t0}

def stage_pre2(root, root2, lt, dt, K, shard=None, device='cpu', **_):
    from . import torchrun2 as R
    t0 = time.time()
    c = Counter()
    for p in unit_files(root / 'inputs', shard):
        dst = root2 / 'pre' / (p.stem + '.json')
        if dst.exists():
            continue
        h = json.loads((root2 / 'h1' / (p.stem + '.json')).read_text(encoding='utf-8'))['h1']
        a, meta = store.load(p)
        out = {}
        for B in BS:
            o, om = store.load(root2 / 'out' / f'{p.stem}__B{B}.npz')
            rows_all = o['rows'].tolist()
            rec = {}
            for m, hm in h[f'B{B}']['methods'].items():
                if m == 'C':
                    continue
                need = {}
                for r in hm['regressions']:
                    need.setdefault(r['batch'], set()).add(r['key'][0])
                for k, reds in need.items():
                    rows = rows_all[k]
                    res = R.run_batch(a['x'][rows].tolist(), a['t'][rows].tolist(), dt, lt, methods=[m], device=device, want_pre={(m, red) for red in reds})[m]
                    old = load_out(o, m, k)
                    c['batches'] += 1
                    if any((old[key] != res.get(key) for key in old)):
                        c['recompute_differs'] += 1
                    for r in hm['regressions']:
                        if r['batch'] != k:
                            continue
                        red, part, i = r['key']
                        v = res['pre'].get((red, part))
                        c['values'] += 1
                        rec[f'{m}|{k}|{red}|{part}|{i}'] = None if v is None else float(v[i if v.size > 1 else 0]).hex()
            out[f'B{B}'] = rec
        store.save_json(dst, dict(unit=meta['unit'], pre=out, environment=R.environment(device)))
    return dict(c, recompute_differs=c['recompute_differs'], seconds=time.time() - t0)

def stage_h1b(root, root2, lt, dt, K, shard=None, **_):
    from klbench.fp import get_format, bits_to_fraction, round_fraction
    from kldivbench import truth as T
    from . import score2 as S2
    fmt = get_format(dt)
    t0 = time.time()
    tot = Counter()
    for p in unit_files(root / 'inputs', shard):
        dst = root2 / 'h1b' / (p.stem + '.json')
        if dst.exists():
            continue
        hj = json.loads((root2 / 'h1' / (p.stem + '.json')).read_text(encoding='utf-8'))
        h = hj['h1']
        pre = json.loads((root2 / 'pre' / (p.stem + '.json')).read_text(encoding='utf-8'))['pre']
        a, meta = store.load(p)
        res = {}
        for B in BS:
            tr, _ = store.load(root / 'truth' / f'{p.stem}__B{B}.npz')
            cache = {}
            for m, hm in h[f'B{B}']['methods'].items():
                c = Counter()
                worst = []
                c['unchecked_over_cap'] = hm['n_regressions'] - len(hm['regressions'])
                for r in hm['regressions']:
                    k = r['batch']
                    if k not in cache:
                        cache = {k: (_ctx(a, meta, tr, k, fmt, lt, dt, with_logits=False)[0], load_truth(tr, k))}
                    tc, truth = cache[k]
                    red, part, i = r['key']
                    if m == 'C':
                        d = None if r['y'] is None else bits_to_fraction(int(r['y']), fmt)
                        val = d.value if d is not None and d.kind == 'finite' else None
                    else:
                        hx = pre[f'B{B}'].get(f'{m}|{k}|{red}|{part}|{i}')
                        fv = None if hx is None else float.fromhex(hx)
                        val = Fr(fv) if fv is not None and fv == fv and (abs(fv) != float('inf')) else None
                    verdict = S2.check_regression(tc, truth, r, m, dt, lt, val, fmt)
                    cl = r['cls']
                    if m != 'C' and val is not None and (not T._same(int(round_fraction(val, fmt)), int(r['y']), fmt)):
                        verdict = 'violation'
                        c[f'{cl}|output_is_not_the_rounding_of_the_recorded_value'] += 1
                    c[f'{cl}|{verdict}'] += 1
                    c[f'{cl}|regressed'] += 1
                    if r['ulp'] is not None and r['ulp'] > c[f'{cl}|max_ulp']:
                        c[f'{cl}|max_ulp'] = r['ulp']
                    if r['ulp'] is None:
                        c[f'{cl}|nonfinite'] += 1
                    if verdict != 'exception' and len(worst) < 20:
                        worst.append(dict(r, verdict=verdict))
                res[f'B{B}|{m}'] = dict(cnt=dict(c), not_explained=worst)
                for k_, v in c.items():
                    if '|' in k_ and k_.split('|')[1] in ('exception', 'violation', 'undecided'):
                        tot[f"{m}|{k_.split('|')[1]}"] += v
        store.save_json(dst, dict(unit=meta['unit'], layer=meta['layer'], group=meta['group'], region=meta['region'], log_target=lt, dtype=dt, K=K, h1b=res))
    return dict(tot, seconds=time.time() - t0)
STAGES = {'run2': stage_run2, 'replay2': stage_replay2, 'score2': stage_score2, 'pre2': stage_pre2, 'h1b': stage_h1b}

def strict_problems(stage, r):
    bad = {}
    if stage == 'replay2' and r.get('batches_differ', 0):
        bad['replay batches_differ'] = r['batches_differ']
    if stage == 'pre2' and r.get('recompute_differs', 0):
        bad['pre2 recompute_differs'] = r['recompute_differs']
    if stage == 'run2' and r.get('call_or_backward_errors', 0):
        bad['run2 call_or_backward_errors'] = r['call_or_backward_errors']
    if stage == 'run2' and r.get('uncertified_rounding', 0):
        bad['run2 uncertified_rounding'] = r['uncertified_rounding']
    return bad

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=list(STAGES))
    ap.add_argument('--root', required=True, help='frozen benchmark data (read only)')
    ap.add_argument('--root2', required=True)
    ap.add_argument('--lt', type=int, required=True)
    ap.add_argument('--dt', required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--shard', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--log-prefix', default='')
    ap.add_argument('--strict', action='store_true')
    a = ap.parse_args(argv)
    lt = bool(a.lt)
    cfg = store.cfg_name(lt, a.dt, a.K)
    root = Path(a.root) / cfg
    root2 = Path(a.root2) / cfg
    root2.mkdir(parents=True, exist_ok=True)
    if Path(a.root).resolve() == Path(a.root2).resolve():
        raise SystemExit('--root2 must differ from the frozen --root')
    r = STAGES[a.stage](root=root, root2=root2, lt=lt, dt=a.dt, K=a.K, shard=a.shard, device=a.device)
    log = root2 / 'logs' / f"{a.log_prefix}{a.stage}{('_' + a.shard.replace('/', 'of') if a.shard else '')}.json"
    if log.exists():
        log = log.with_name(f"{log.stem}.{time.strftime('%Y%m%dT%H%M%S')}.{os.getpid()}.json")
    bad = strict_problems(a.stage, r)
    store.save_json(log, dict(stage=a.stage, argv=sys.argv, result=r, strict=a.strict, strict_problems=bad, maxrss_kb=__import__('resource').getrusage(0).ru_maxrss, host=__import__('platform').node()))
    print(a.stage, cfg, json.dumps(r, default=str)[:600])
    if a.strict and bad:
        print(f'STRICT FAILURE in stage {a.stage}: {bad}', file=sys.stderr)
        sys.exit(4)
if __name__ == '__main__':
    main()
