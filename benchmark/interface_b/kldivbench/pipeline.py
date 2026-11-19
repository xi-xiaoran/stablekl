from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter
from fractions import Fraction as Fr
from pathlib import Path
import numpy as np
from . import store
BS = (1, 8)
E2E_LAYERS = ('S1', 'S1t', 'S2', 'S2b')
UINT = {'fp16': np.uint16, 'fp32': np.uint32}
CALL = ('OK', 'CALL_ERROR')
GRAD = ('OK', 'BACKWARD_ERROR', 'GRAD_MISSING', 'NO_CALL')
REDS4 = ('none', 'sum', 'mean', 'batchmean')
REDS = REDS4[1:]
QUOTAS = {'smoke': dict(S1={'R0': 6, 'R1': 2, 'R2': 2}, S1t={'R0': 1, 'R1': 0, 'R2': 0}, S2=8, S2b=8, S3=8, S4=8, S5=8, S6=8, S7=4), 'smoke8': dict(S1={'R0': 8, 'R1': 8, 'R2': 8}, S1t={'R0': 8, 'R1': 0, 'R2': 0}, S2=16, S2b=16, S3=16, S4=16, S5=16, S6=16, S7=8), 'full': dict(S1={'R0': 2000, 'R1': 500, 'R2': 500}, S1t={'R0': 120, 'R1': 40, 'R2': 40}, S2=3000, S2b=3000, S3=5000, S4=5000, S5=3000, S6=3000, S7=500), 'full256': dict(S1={'R0': 800, 'R1': 200, 'R2': 200}, S1t={'R0': 60, 'R1': 20, 'R2': 20}, S2=1200, S2b=1200, S3=2000, S4=2000, S5=1200, S6=1200, S7=200), 'stress': dict(S1={'R0': 100, 'R1': 0, 'R2': 0}, S1t={'R0': 0, 'R1': 0, 'R2': 0}, S2=0, S2b=0, S3=0, S4=0, S5=0, S6=0, S7=0)}

def quota(name, K):
    if name == 'full':
        return QUOTAS['full256' if K == 256 else 'full']
    return QUOTAS[name] if name in QUOTAS else json.loads(name)

def batches(n, B):
    return [list(range(i, i + B)) for i in range(0, n - B + 1, B)]

def unit_files(d, shard=None):
    fs = sorted((p for p in Path(d).glob('*.npz') if '.partial' not in p.name))
    if shard:
        i, n = (int(v) for v in shard.split('/'))
        fs = [p for k, p in enumerate(fs) if k % n == i]
    return fs

def write_inputs(path, u, dt, extra=None):
    K = u.K
    ut = UINT[dt]
    n = len(u.rows)
    arr = {'x': np.array([r['x'] for r in u.rows], dtype=ut).reshape(n, K), 't': np.array([r['t'] for r in u.rows], dtype=ut).reshape(n, K), 'meta': np.array([json.dumps({k: v for k, v in r.items() if k not in ('x', 't', 'zp', 'zq')}, sort_keys=True) for r in u.rows] or [''], dtype=np.str_)[:n]}
    if u.has_logits:
        arr['zp'] = np.array([r['zp'] for r in u.rows], dtype=ut).reshape(n, K)
        arr['zq'] = np.array([r['zq'] for r in u.rows], dtype=ut).reshape(n, K)
    det = dict(unit=u.name, layer=u.layer, group=u.group, region=u.region, has_logits=u.has_logits, n_rows=n, K=K, counters=u.counters, stage='inputs', **extra or {})
    return store.save(path, arr, det)

def stage_gen(root, lt, dt, K, q, **_):
    from . import gen as G
    t0 = time.time()
    units = G.generate(lt, dt, K, q)
    for u in units:
        u.K = K
        write_inputs(root / ('pending_s1t' if u.layer == 'S1t' else 'inputs') / (u.name + '.npz'), u, dt, dict(log_target=lt, dtype=dt, quota=q))
    return {'units': len(units), 'rows': sum((len(u.rows) for u in units)), 'seconds': time.time() - t0}

def stage_gensupp(root, lt, dt, K, q, shard=None, max_factor=30, **_):
    from . import gensupp as GS
    t0 = time.time()
    cells = []
    done = Counter()
    for base in ('S1', 'S1t'):
        for p in sorted((root / 'inputs').glob(base + '__*.npz')):
            meta = store.verify(p)
            gap = q[base][meta['region']] - meta['n_rows']
            if gap > 0:
                cells.append((base, meta['group'], meta['region'], gap, p))
    if shard:
        i, n = (int(v) for v in shard.split('/'))
        cells = [c for k, c in enumerate(cells) if k % n == i]
    for base, g, reg, gap, p in cells:
        name = f'{GS.LAYER[base]}__{g}__{reg}.npz'
        dst = root / ('pending_s1t' if base == 'S1t' else 'inputs') / name
        if store.exists(dst) or store.exists(root / 'inputs' / name):
            continue
        a, meta = store.load(p)
        existing = {(tuple(a['zp'][i].tolist()), tuple(a['zq'][i].tolist())) for i in range(meta['n_rows'])}
        u = GS.gen_cell(lt, dt, K, base, g, reg, gap, existing, max_factor=max_factor)
        u.K = K
        write_inputs(dst, u, dt, dict(log_target=lt, dtype=dt, quota=q, supplement_of=p.name, supplement_of_sha256=meta['sha256'], base_rows=meta['n_rows']))
        done.update(cells=1, gap=gap, rows=len(u.rows))
    return dict(done, cells_below_quota=len(cells), seconds=time.time() - t0)

def stage_fill(root, lt, dt, K, **_):
    from . import torchrun as R
    from .store import xt_hash
    n = 0
    for p in unit_files(root / 'pending_s1t'):
        a, meta = store.load(p)
        rows = len(a['zp'])
        x = np.zeros_like(a['zp'])
        t = np.zeros_like(a['zp'])
        metas = []
        for i in range(rows):
            xi, ti = R.torch_inputs(a['zp'][i].tolist(), a['zq'][i].tolist(), dt, lt)
            x[i] = xi
            t[i] = ti
            m = json.loads(str(a['meta'][i]))
            m['xt_hash'] = xt_hash(xi, ti)
            m['needs_torch_inputs'] = False
            metas.append(json.dumps(m, sort_keys=True))
        det = {k: meta[k] for k in meta if k in ('unit', 'layer', 'group', 'region', 'has_logits', 'n_rows', 'K', 'counters', 'log_target', 'dtype', 'quota', 'supplement_of', 'supplement_of_sha256', 'base_rows')}
        store.save(root / 'inputs' / p.name, dict(x=x, t=t, zp=a['zp'], zq=a['zq'], meta=np.array(metas or [''], dtype=np.str_)[:rows]), dict(det, stage='inputs', filled_by=R.environment(), pending_sha256=meta['sha256']))
        n += rows
    return {'rows_filled': n}

def stage_certify(root, lt, dt, K, shard=None, **_):
    from . import truth as T, xcheck as X
    from klbench.fp import get_format, exact_value
    fmt = get_format(dt)
    code = {s: i for i, s in enumerate(T.STATUS_CODES)}
    ut = UINT[dt]
    tot = Counter()
    t0 = time.time()
    for p in unit_files(root / 'inputs', shard):
        a, meta = store.load(p)
        n = meta['n_rows']
        logits = meta['has_logits']
        if all((store.exists(root / 'truth' / f'{p.stem}__B{B}.npz') for B in BS)):
            continue
        zp = [[exact_value(int(b), dt) for b in a['zp'][i]] for i in range(n)] if logits else None
        zq = [[exact_value(int(b), dt) for b in a['zq'][i]] for i in range(n)] if logits else None
        z = lambda v: [0 if b is None else b for b in v]
        arrs = {}
        cs = {B: Counter() for B in BS}
        nbs = {}
        for B in BS:
            bl = batches(n, B)
            nb = nbs[B] = len(bl)
            N = B * K
            arr = {'rows': np.array(bl, dtype=np.int64).reshape(nb, B), 'L_bits': np.zeros((nb, N), ut), 'L_q': np.zeros((nb, N), np.uint8), 'L_st': np.zeros((nb, N), np.uint8), 'gx_st': np.zeros((nb, N), np.uint8), 'gt_st': np.zeros((nb, N), np.uint8), 'S_st': np.zeros(nb, np.uint8), 'S_bits': np.zeros((nb, 3), ut), 'S_q': np.zeros((nb, 3), np.uint8)}
            for side in ('gx', 'gt'):
                for red in REDS:
                    arr[f'{side}_bits_{red}'] = np.zeros((nb, N), ut)
                    arr[f'{side}_q_{red}'] = np.zeros((nb, N), np.uint8)
            if logits:
                arr.update(E2E_st=np.zeros(nb, np.uint8), E2E_bits=np.zeros((nb, 3), ut), E2E_q=np.zeros((nb, 3), np.uint8))
            arrs[B] = arr
        for c0 in range(0, n, 8):
            chunk = list(range(c0, min(c0 + 8, n)))
            vals = {i: T.row_vals(a['x'][i].tolist(), a['t'][i].tolist(), fmt, lt) for i in chunk}
            todo = [(1, i, [i]) for i in chunk] + ([(8, c0 // 8, chunk)] if len(chunk) == 8 else [])
            for B, k, rows in todo:
                arr = arrs[B]
                c = cs[B]
                xb = [a['x'][i].tolist() for i in rows]
                tb = [a['t'][i].tolist() for i in rows]
                vv = [vals[i] for i in rows]
                tr = T.certify_batch(xb, tb, fmt, lt, vals=vv, zp=[zp[i] for i in rows] if logits else None, zq=[zq[i] for i in rows] if logits else None)
                arr['L_bits'][k] = z(tr['L_bits'])
                arr['L_q'][k] = tr['L_q']
                arr['L_st'][k] = [code[s] for s in tr['L_st']]
                c['L_scored'] += sum((s not in (T.UNDEF, T.DOMAIN) for s in tr['L_st']))
                c['L_uncert'] += sum((s not in (T.UNDEF, T.DOMAIN) and (not q & T.Q_CERT) for s, q in zip(tr['L_st'], tr['L_q'])))
                for side in ('gx', 'gt'):
                    arr[side + '_st'][k] = [code[s] for s in tr[side + '_st']]
                    for red in REDS:
                        arr[f'{side}_bits_{red}'][k] = z(tr[side + '_bits'][red])
                        arr[f'{side}_q_{red}'][k] = tr[side + '_q'][red]
                        c['G_scored'] += sum((s not in (T.UNDEF, T.DOMAIN) for s in tr[side + '_st']))
                        c['G_uncert'] += sum((s not in (T.UNDEF, T.DOMAIN) and (not q & T.Q_CERT) for s, q in zip(tr[side + '_st'], tr[side + '_q'][red])))
                arr['S_st'][k] = code[tr['S_st']]
                arr['S_bits'][k] = z([tr['S_bits'][r] for r in REDS])
                arr['S_q'][k] = [tr['S_q'][r] for r in REDS]
                if tr['S_st'] != T.DOMAIN:
                    c['S_scored'] += 3
                    c['S_uncert'] += sum((not tr['S_q'][r] & T.Q_CERT for r in REDS))
                X.check_batch(c, xb, tb, fmt, lt, vv, tr['S_val'], seeds=[f"{store.cfg_name(lt, dt, K)}/{meta['unit']}/{i}" for i in rows])
                if logits:
                    arr['E2E_st'][k] = code[tr['E2E_st']]
                    arr['E2E_bits'][k] = z([tr['E2E_bits'][r] for r in REDS])
                    arr['E2E_q'][k] = [tr['E2E_q'][r] for r in REDS]
                    c['E2E_scored'] += 3
                    c['E2E_uncert'] += sum((not tr['E2E_q'][r] & T.Q_CERT for r in REDS))
                    X.check_e2e(c, [zp[i] for i in rows], [zq[i] for i in rows], tr['E2E_val'])
        for B in BS:
            arr = arrs[B]
            c = cs[B]
            nb = nbs[B]
            ex = c.pop('examples', [])
            store.save(root / 'truth' / f'{p.stem}__B{B}.npz', arr, dict(unit=meta['unit'], layer=meta['layer'], group=meta['group'], region=meta['region'], B=B, K=K, n_batches=nb, dropped_tail_rows=n - nb * B, log_target=lt, dtype=dt, stage='truth', counters=dict(c), xcheck_examples=ex, inputs_sha256=meta['sha256']))
            tot.update(c)
    return dict(tot, seconds=time.time() - t0)

def stage_run(root, lt, dt, K, shard=None, **_):
    from . import torchrun as R
    ut = UINT[dt]
    t0 = time.time()
    nbt = 0
    for p in unit_files(root / 'inputs', shard):
        a, meta = store.load(p)
        n = meta['n_rows']
        for B in BS:
            dst = root / 'out' / f'{p.stem}__B{B}.npz'
            if store.exists(dst):
                continue
            bl = batches(n, B)
            nb = len(bl)
            N = B * K
            arr = {'rows': np.array(bl, dtype=np.int64).reshape(nb, B)}
            errs = []
            for m in R.METHODS[dt]:
                arr[m + '__cs'] = np.zeros((nb, 4), np.uint8)
                arr[m + '__gs'] = np.zeros((nb, 4), np.uint8)
                arr[m + '__dtype_ok'] = np.ones(nb, np.uint8)
                for red in REDS4:
                    arr[f'{m}__{red}'] = np.zeros((nb, N) if red == 'none' else nb, ut)
                    arr[f'{m}__{red}_gx'] = np.zeros((nb, N), ut)
                    arr[f'{m}__{red}_gt'] = np.zeros((nb, N), ut)
            for nm in ('Y_seq', 'Y_exact', 'sum_noncontig'):
                arr['A__' + nm] = np.zeros(nb, ut)
                arr['A__' + nm + '_ok'] = np.zeros(nb, np.uint8)
            for k, rows in enumerate(bl):
                res = R.run_batch(a['x'][rows].tolist(), a['t'][rows].tolist(), dt, lt)
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
                    if o['errors'] and len(errs) < 20:
                        errs.append(dict(batch=k, method=m, **o['errors']))
                    if m == 'A':
                        for nm in ('Y_seq', 'Y_exact', 'sum_noncontig'):
                            if o.get(nm) is not None:
                                arr['A__' + nm][k] = o[nm]
                                arr['A__' + nm + '_ok'][k] = 1
            store.save(dst, arr, dict(unit=meta['unit'], B=B, K=K, n_batches=nb, log_target=lt, dtype=dt, stage='out', methods=list(R.METHODS[dt]), environment=R.environment(), errors=errs, inputs_sha256=meta['sha256']))
            nbt += nb
    return {'batches': nbt, 'seconds': time.time() - t0}

def stage_replay(root, lt, dt, K, shard=None, **_):
    from . import torchrun as R
    c = Counter()
    t0 = time.time()
    ex = []
    for p in unit_files(root / 'inputs', shard):
        a, meta = store.load(p)
        for B in BS:
            o, om = store.load(root / 'out' / f'{p.stem}__B{B}.npz')
            for k, rows in enumerate(o['rows'].tolist()):
                res = R.run_batch(a['x'][rows].tolist(), a['t'][rows].tolist(), dt, lt)
                c['batches'] += 1
                bad = []
                if os.environ.get('KLDIV_INJECT_REPLAY_DIFF') and c['batches'] == 1:
                    res['A']['sum'] = (res['A']['sum'] or 0) ^ 1
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
    return dict(c, batches_differ=c['batches_differ'], examples=ex, environment=R.environment(), seconds=time.time() - t0)

def load_out(o, m, k):
    r = {'call_status': {}, 'grad_status': {}}
    for j, red in enumerate(REDS4):
        cs = CALL[int(o[m + '__cs'][k, j])]
        gs = GRAD[int(o[m + '__gs'][k, j])]
        r['call_status'][red] = cs
        r['grad_status'][red] = gs
        v = o[f'{m}__{red}'][k]
        r[red] = None if cs != 'OK' else v.tolist() if red == 'none' else int(v)
        r[red + '_gx'] = o[f'{m}__{red}_gx'][k].tolist() if gs == 'OK' else None
        r[red + '_gt'] = o[f'{m}__{red}_gt'][k].tolist() if gs == 'OK' else None
    if m == 'A':
        for nm in ('Y_seq', 'Y_exact', 'sum_noncontig'):
            r[nm] = int(o['A__' + nm][k]) if o['A__' + nm + '_ok'][k] else None
    return r

def load_truth(tr, k):
    from . import truth as T
    st = lambda v: [T.STATUS_CODES[int(i)] for i in v]
    d = {'L_bits': tr['L_bits'][k].tolist(), 'L_q': tr['L_q'][k].tolist(), 'L_st': st(tr['L_st'][k]), 'gx_st': st(tr['gx_st'][k]), 'gt_st': st(tr['gt_st'][k]), 'S_st': T.STATUS_CODES[int(tr['S_st'][k])], 'S_bits': dict(zip(REDS, tr['S_bits'][k].tolist())), 'S_q': dict(zip(REDS, tr['S_q'][k].tolist()))}
    for side in ('gx', 'gt'):
        d[side + '_bits'] = {r: tr[f'{side}_bits_{r}'][k].tolist() for r in REDS}
        d[side + '_q'] = {r: tr[f'{side}_q_{r}'][k].tolist() for r in REDS}
    if 'E2E_st' in tr:
        d['E2E_st'] = T.STATUS_CODES[int(tr['E2E_st'][k])]
        d['E2E_bits'] = dict(zip(REDS, tr['E2E_bits'][k].tolist()))
        d['E2E_q'] = dict(zip(REDS, tr['E2E_q'][k].tolist()))
    return d

def stage_score(root, lt, dt, K, shard=None, **_):
    from . import score as S, truth as T
    from klbench.fp import get_format, exact_value, ulp_distance, round_fraction, bits_to_fraction
    fmt = get_format(dt)
    t0 = time.time()
    nu = 0
    m_ = fmt.sign_mask - 1
    for p in unit_files(root / 'inputs', shard):
        dst = root / 'score' / (p.stem + '.json')
        if dst.exists():
            continue
        a, meta = store.load(p)
        logits = meta['has_logits']
        res = {}
        expected = {}
        outs = {}
        for B in BS:
            tr, tm = store.load(root / 'truth' / f'{p.stem}__B{B}.npz')
            o, om = store.load(root / 'out' / f'{p.stem}__B{B}.npz')
            outs[B] = o
            if tm['inputs_sha256'] != meta['sha256'] or om['inputs_sha256'] != meta['sha256']:
                raise ValueError(f'truth / output do not belong to these inputs: {p}')
            if not (tr['rows'] == o['rows']).all():
                raise ValueError(f'batch layout differs: {p}')
            methods = om['methods']
            accs = {m: S.Acc() for m in methods}
            accT = S.Acc()
            pairs = {m: Counter() for m in methods if m != 'A'}
            nb = len(tr['rows'])
            for k in range(nb):
                rows = tr['rows'][k].tolist()
                xb = [a['x'][i].tolist() for i in rows]
                tb = [a['t'][i].tolist() for i in rows]
                zp = [[exact_value(int(b), dt) for b in a['zp'][i]] for i in rows] if logits else None
                zq = [[exact_value(int(b), dt) for b in a['zq'][i]] for i in rows] if logits else None
                tc = S.TruthCtx(xb, tb, fmt, lt, zp, zq)
                truth = load_truth(tr, k)
                if [v.status for v in tc.L] != truth['L_st'] or tc.S.status != truth['S_st']:
                    raise ValueError(f'frozen status differs from the recomputed status: {p} batch {k}')
                if logits:
                    S.score_truth(accT, tc, truth, fmt)
                flags = {}
                for m in methods:
                    flags[m], info = S.score_method(accs[m], tc, truth, load_out(o, m, k), fmt)
                    acc = accs[m]
                    d = info['S_ULP']['sum']
                    acc.top.append((d if d is not None else -1, info['E_maxULP'], k))
                    acc.top = sorted(acc.top, reverse=True)[:20]
                    cm = sorted((n for n, b in flags[m].items() if b and n.split(':')[0].split('_', 1)[-1] in S.LEVEL_C))
                    if cm and len(acc.ccases) < 20:
                        acc.ccases.append(dict(batch=k, rows=rows, metrics=cm))
                for m in pairs:
                    S.pair(pairs[m], flags['A'], flags[m])
                if B == 8 and 'A' in methods:
                    o1 = outs[1]
                    c = accs['A'].cnt
                    n8 = o['A__none'][k].reshape(8, K)
                    if all((o1['A__cs'][i, 0] == 0 for i in rows)) and o['A__cs'][k, 0] == 0:
                        c['REPRO_row_in_batch_n'] += 8
                        c['REPRO_row_in_batch_ne'] += sum((not (n8[j] == o1['A__none'][i]).all() for j, i in enumerate(rows)))
                    if all((o1['A__cs'][i, 1] == 0 for i in rows)) and o['A__cs'][k, 1] == 0:
                        v = [bits_to_fraction(int(o1['A__sum'][i]), fmt) for i in rows]
                        yb = int(o['A__sum'][k])
                        if all((d.kind == 'finite' for d in v)) and yb & m_ < fmt.infinity_bits:
                            e = int(round_fraction(sum((d.value for d in v), Fr(0)), fmt))
                            c['REPRO_batchsum_n'] += 1
                            c['REPRO_batchsum_ne_sum_of_rowsums'] += e != yb
                            if e & m_ < fmt.infinity_bits:
                                accs['A'].stats['REPRO_ULP_batchsum'].append(ulp_distance(yb, e, fmt))
            pack = lambda acc: dict(cnt=dict(acc.cnt), stats={k: v for k, v in acc.stats.items()}, top=[dict(S_sum_ULP=d, E_maxULP=e, batch=k, rows=tr['rows'][k].tolist()) for d, e, k in acc.top], ccases=acc.ccases)
            for m in methods:
                res[f'B{B}|{m}'] = pack(accs[m])
            if logits:
                res[f'B{B}|TRUTH'] = pack(accT)
            for m in pairs:
                res[f'B{B}|PAIR:{m}'] = dict(cnt=dict(pairs[m]))
            expected[f'B{B}'] = dict(batches=nb, rows=nb * B, elements=nb * B * K, methods=methods)
        hs = Counter((json.loads(str(v))['xt_hash'] for v in a['meta'][:meta['n_rows']]))
        store.save_json(dst, dict(xt_shared_rows=sum((v for v in hs.values() if v > 1)), unit=meta['unit'], layer=meta['layer'], group=meta['group'], region=meta['region'], log_target=lt, dtype=dt, K=K, n_rows=meta['n_rows'], gen_counters=meta['counters'], inputs=str(p.name), inputs_sha256=meta['sha256'], expected=expected, results=res))
        nu += 1
    return {'units': nu, 'seconds': time.time() - t0}
STAGES = {'gen': stage_gen, 'fill': stage_fill, 'certify': stage_certify, 'run': stage_run, 'score': stage_score, 'replay': stage_replay, 'gensupp': stage_gensupp}

def forbid(names):
    import importlib.abc

    class Block(importlib.abc.MetaPathFinder):

        def find_spec(self, name, path=None, target=None):
            if name.split('.')[0] in names:
                raise ImportError(f'{name} is forbidden in this stage (KLDIV_FORBID)')
    sys.meta_path.insert(0, Block())

def strict_problems(stage, r):
    bad = {}
    if stage == 'replay' and r.get('batches_differ', 0):
        bad['replay batches_differ'] = r['batches_differ']
    if stage == 'certify':
        for k, v in r.items():
            if (k.endswith('_uncert') or (k.startswith('XCHECK') and k.endswith('_bad'))) and v:
                bad[k] = v
    return bad

def main(argv=None):
    if os.environ.get('KLDIV_FORBID'):
        forbid(set(os.environ['KLDIV_FORBID'].split(',')))
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=list(STAGES))
    ap.add_argument('--root', required=True)
    ap.add_argument('--lt', type=int, required=True)
    ap.add_argument('--dt', required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--quota', default='full')
    ap.add_argument('--shard', default=None)
    ap.add_argument('--log-prefix', default='', help='prefix of the stage log file (later runs must not replace earlier logs)')
    ap.add_argument('--strict', action='store_true', help='exit with status 4 if replay differs, a truth is not certified or the cross-check disagrees')
    ap.add_argument('--max-factor', type=int, default=30, help='gensupp: attempts per missing row')
    a = ap.parse_args(argv)
    lt = bool(a.lt)
    root = Path(a.root) / store.cfg_name(lt, a.dt, a.K)
    root.mkdir(parents=True, exist_ok=True)
    r = STAGES[a.stage](root=root, lt=lt, dt=a.dt, K=a.K, q=quota(a.quota, a.K), shard=a.shard, max_factor=a.max_factor)
    log = root / 'logs' / f"{a.log_prefix}{a.stage}{('_' + a.shard.replace('/', 'of') if a.shard else '')}.json"
    if log.exists():
        log = log.with_name(f"{log.stem}.{time.strftime('%Y%m%dT%H%M%S')}.{os.getpid()}.json")
    bad = strict_problems(a.stage, r)
    store.save_json(log, dict(stage=a.stage, argv=sys.argv, result=r, strict=a.strict, strict_problems=bad, maxrss_kb=__import__('resource').getrusage(0).ru_maxrss, host=__import__('platform').node()))
    print(a.stage, store.cfg_name(lt, a.dt, a.K), json.dumps(r, default=str)[:600])
    if a.strict and bad:
        print(f'STRICT FAILURE in stage {a.stage}: {bad}', file=sys.stderr)
        sys.exit(4)
if __name__ == '__main__':
    main()
