from __future__ import annotations
import argparse, hashlib, importlib, json, math, os, sys, time, traceback
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
UI = {'fp16': np.uint16, 'fp32': np.uint32, 'fp64': np.uint64}
NPF = {'fp16': np.float16, 'fp32': np.float32, 'fp64': np.float64}
KL_SIG = ('KL_ULP8', 'KL_REL1', 'KL_REL10', 'KL_REL100', 'KL_NEG', 'KL_NAN', 'KL_INF', 'KL_FALSE_ZERO', 'KL_NONZERO_AT_IDENTITY')
G_SIG = ('G_NAN', 'G_INF', 'G_SIGN', 'G_VANISH', 'G_SPURIOUS')
OFFICIAL = {'fp16': ['A', 'B32', 'B64'], 'fp32': ['A', 'B64']}
REWRITE = {'fp16': {'C': 'fp16', 'D32': 'fp32', 'D64': 'fp64'}, 'fp32': {'C': 'fp32', 'D64': 'fp64'}}

def module_name(config):
    for prefix in ('categorical', 'contbern', 'dirichlet', 'mvn', 'lowrank'):
        if config.startswith(prefix):
            return 'mvn' if prefix == 'lowrank' else prefix
    return config

def load_fix(name):
    M = importlib.import_module('stablekl._closed.' + module_name(name))
    if hasattr(M, 'configure'):
        M.configure(name)
    assert M.NAME == name, (M.NAME, name)
    return M

def csplit(comp):
    side, rest = comp.split('.', 1)
    if rest.endswith(']'):
        nm, i = rest[:-1].split('[')
        return (side, nm, int(i))
    return (side, rest, 0)

def paths(root, name):
    data = Path(root) / 'interface_a'
    return (data / 'frozen' / name, data / 'official_cpu' / name)

def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()

def loader(M=None):
    from klbench.pipeline_io import load_unit
    return lambda p: load_unit(Path(p))

def compute(a):
    import torch
    from klbench.torchrun import runner as R
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '1'))))
    M = load_fix(a.name)
    FROZEN, _ = paths(a.root, a.name)
    out = Path(a.out) / 'outputs' / a.name
    load_unit = loader(M)
    TD = {'fp16': torch.float16, 'fp32': torch.float32, 'fp64': torch.float64}
    dev = torch.device(a.device)
    if dev.type == 'cuda':
        assert torch.cuda.is_available(), 'CUDA unavailable'

    def to_t_on(d):
        return lambda bits, dt: torch.from_numpy(np.ascontiguousarray(bits, dtype=UI[dt]).view(NPF[dt]).copy()).to(d)

    def official(frozen, cfg, dt, m, d):
        rows, _ = R._rows(cfg, dt, m, frozen, a.batch, d, 0)
        return R._pack(rows, frozen['_parameter_shapes'], dt, frozen['sample_id'])

    def rewrite(P, Q, dt, cdt, d):
        with torch.device(d):
            o = M.run_rewrite_torch(TD, to_t_on(d), dt, cdt, P, Q, NPF)
        return {k: v.detach().double().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v, dtype=np.float64) for k, v in o.items()}
    info = dict(name=a.name, module=M.__name__, module_md5=md5(M.__file__), tool_md5=md5(__file__), torch=torch.__version__, device=str(dev), device_name=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', units={}, errors=[], bitdiff={}, seconds={})
    bd = Counter()
    t0 = time.time()
    info['skipped_dtypes'] = dict(getattr(M, 'SKIP_DTYPES', {}) or {})
    for dt in ('fp16', 'fp32'):
        if dt in info['skipped_dtypes']:
            print('SKIP', dt, info['skipped_dtypes'][dt][:120], flush=True)
            continue
        for f in sorted((FROZEN / dt).glob('*.npz'))[:a.max_units or None]:
            arr, meta = load_unit(f)
            n = len(arr['sample_id'])
            if n == 0:
                continue
            if a.smoke:
                n = min(n, a.smoke)
            frozen = {k: v[:n] if hasattr(v, 'shape') and v.shape[:1] == (len(arr['sample_id']),) else v for k, v in arr.items()}
            frozen['_parameter_shapes'] = meta['parameter_shapes']
            cfg = meta['config']
            dst = out / dt
            dst.mkdir(parents=True, exist_ok=True)
            done = dst / f'{f.stem}.done'
            if done.exists():
                info['units'][f'{dt}/{f.stem}'] = 'skipped (done)'
                continue
            sq = lambda x: x[:, 0] if x.shape[1] == 1 else x
            P = {k: sq(frozen[f'p__{k}']) for k in M.PARAMS}
            Q = {k: sq(frozen[f'q__{k}']) for k in M.PARAMS}
            for m in OFFICIAL[dt]:
                try:
                    g = official(frozen, cfg, dt, m, dev)
                    np.savez_compressed(dst / f'{f.stem}__{m}.npz', **g)
                    if a.cpu_compare and dev.type != 'cpu':
                        c = official(frozen, cfg, dt, m, torch.device('cpu'))
                        for k in g:
                            if k.endswith('bits'):
                                bd[f'{dt}|{m}|{k}|n'] += g[k].size
                                bd[f'{dt}|{m}|{k}|differ'] += int((np.asarray(g[k]) != np.asarray(c[k])).sum())
                        bd[f'{dt}|{m}|status|n'] += n
                        bd[f'{dt}|{m}|status|differ'] += int((np.asarray(g['status']) != np.asarray(c['status'])).sum())
                except Exception as ex:
                    info['errors'].append(dict(unit=f'{dt}/{f.stem}', method=m, error=f'{type(ex).__name__}: {ex}', trace=traceback.format_exc()[-1500:]))
            for m, cdt in REWRITE[dt].items():
                try:
                    g = rewrite(P, Q, dt, cdt, dev)
                    np.savez_compressed(dst / f'{f.stem}__{m}.npz', **g)
                    if a.cpu_compare and dev.type != 'cpu':
                        c = rewrite(P, Q, dt, cdt, torch.device('cpu'))
                        for k in g:
                            same = (g[k] == c[k]) | np.isnan(g[k]) & np.isnan(c[k])
                            kk = 'KL' if k == 'KL' else 'grad'
                            bd[f'{dt}|{m}|{kk}|n'] += g[k].size
                            bd[f'{dt}|{m}|{kk}|differ'] += int((~same).sum())
                except Exception as ex:
                    info['errors'].append(dict(unit=f'{dt}/{f.stem}', method=m, error=f'{type(ex).__name__}: {ex}', trace=traceback.format_exc()[-1500:]))
            done.write_text(str(n))
            info['units'][f'{dt}/{f.stem}'] = n
        info['seconds'][dt] = time.time() - t0
    info['bitdiff'] = dict(bd)
    info['n_errors'] = len(info['errors'])
    info['error_kinds'] = dict(Counter((f"{e['method']}: {e['error'][:160]}" for e in info['errors'])))
    info['errors'] = info['errors'][:20]
    (out / f'compute_{int(time.time())}.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ('name', 'torch', 'device_name', 'n_errors', 'seconds')}), flush=True)
    for e in info['errors'][:3]:
        print('ERROR', e['unit'], e['method'], e['error'], '\n', e['trace'], flush=True)
    print('units:', len(info['units']), 'bit differences GPU vs CPU:', {k[:-7]: f"{v} / {bd[k[:-7] + '|n']}" for k, v in sorted(bd.items()) if k.endswith('|differ')}, flush=True)
    for k, v in info['error_kinds'].items():
        print('ERROR KIND', v, 'x', k, flush=True)
    return 0 if not info['errors'] else 3

def judge_kl(metrics):
    return (any((metrics.get(k) is True for k in KL_SIG)), metrics.get('KL_CR') is False, metrics.get('KL_CR') is not None)

def rawlike(n, dt, vals, comps, rounded, widths, fmt, round_fraction):
    ui = UI[dt]

    def conv(x):
        x = float(x)
        if not rounded:
            return int(np.array(x, dtype=NPF[dt]).view(ui))
        if math.isnan(x):
            return fmt.infinity_bits | 1
        if math.isinf(x):
            return fmt.infinity_bits | (fmt.sign_mask if x < 0 else 0)
        return round_fraction(F(x), fmt)
    r = {'status': np.array(['OK'] * n), 'kl_bits': np.array([conv(x) for x in vals['KL'][:n]], dtype=ui)}
    for comp in comps[1:]:
        side, nm, ci = csplit(comp)
        key = f'grad__{side}__{nm}'
        if f'{key}__bits' not in r:
            r[f'{key}__bits'] = np.zeros((n, widths[key]), dtype=ui)
            r[f'{key}__status'] = np.array(['OK'] * n)
        r[f'{key}__bits'][:, ci] = [conv(x) for x in vals[comp][:n]]
    return r

def score_unit(task):
    fpath, rawdir, gdir, dt, comps, smoke, name, cache = task
    cf = Path(cache) / f'{dt}__{Path(fpath).stem}.json' if cache else None
    if cf is not None and cf.exists():
        return json.loads(cf.read_text())
    from klbench.fp import get_format, round_fraction
    from klbench.score.runner import score_one
    load_unit = loader()
    fmt = get_format(dt)
    arr, _ = load_unit(fpath)
    n = len(arr['sample_id'])
    C = Counter()
    if n == 0:
        return {}
    n = min(n, smoke) if smoke else n
    stem = Path(fpath).stem
    widths = {k[:-6]: arr[k].shape[1] for k in arr if k.startswith('grad__') and k.endswith('__bits') and (k.count('__') == 3)}
    suffix = {c: c if c.endswith(']') else f'{c}[0]' for c in comps[1:]}
    raws = {}
    for m in OFFICIAL[dt]:
        p = Path(rawdir) / dt / f'{stem}__{m}.npz'
        if p.exists():
            ra, _ = load_unit(p)
            assert list(ra['sample_id'])[:n] == list(arr['sample_id'])[:n]
            raws[m + '_cpu'] = ra
        g = Path(gdir) / dt / f'{stem}__{m}.npz'
        if g.exists():
            raws[m] = dict(np.load(g, allow_pickle=False))
        else:
            C[f'{dt}|{m}|missing_units'] += 1
    for m in REWRITE[dt]:
        g = Path(gdir) / dt / f'{stem}__{m}.npz'
        if g.exists():
            raws[m] = rawlike(n, dt, dict(np.load(g, allow_pickle=False)), comps, m != 'C', widths, fmt, round_fraction)
        else:
            C[f'{dt}|{m}|missing_units'] += 1
    for m in OFFICIAL[dt]:
        if m in raws and m + '_cpu' in raws:
            C[f'{dt}|{m}|kl_bits_vs_frozen_cpu_n'] += n
            C[f'{dt}|{m}|kl_bits_vs_frozen_cpu_differ'] += int((np.asarray(raws[m]['kl_bits'][:n]) != np.asarray(raws[m + '_cpu']['kl_bits'][:n])).sum())
    for j in range(n):
        for m, raw in raws.items():
            metrics, ulps, _ = score_one(arr, raw, j, dt)
            sig, notcr, app = judge_kl(metrics)
            k = f'{dt}|{m}|'
            C[k + 'rows'] += 1
            if app:
                C[k + 'kl_app'] += 1
                C[k + 'kl_sig'] += sig
                C[k + 'kl_cr'] += notcr
                for nm in ('KL_NEG', 'KL_NAN', 'KL_INF', 'KL_NONZERO_AT_IDENTITY', 'KL_FALSE_ZERO'):
                    C[k + nm] += metrics.get(nm) is True
            gs = ga = gn = 0
            for comp, sfx in suffix.items():
                cr = metrics.get(f'G_CR:{sfx}')
                u = ulps.get(f'G_ULP:{sfx}')
                if cr is None:
                    continue
                ga += 1
                gn += cr is False
                gs += u is not None and u >= 8 or any((metrics.get(f'{g}:{sfx}') is True for g in G_SIG))
            C[k + 'g_app'] += ga
            C[k + 'g_sig'] += gs
            C[k + 'g_cr'] += gn
            if ga:
                C[k + 'grow_app'] += 1
                C[k + 'grow_sig'] += gs > 0
    if cf is not None:
        tmp = cf.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(C)))
        os.replace(tmp, cf)
    return dict(C)

def score(a):
    from multiprocessing import get_context
    M = load_fix(a.name)
    FROZEN, RAW = paths(a.root, a.name)
    gdir = Path(a.out) / 'outputs' / a.name
    comps = list(M.COMPS)
    cache = None
    if not a.smoke:
        cache = Path(a.out) / 'score_units' / a.name
        cache.mkdir(parents=True, exist_ok=True)
    tasks = [(str(f), str(RAW), str(gdir), dt, comps, a.smoke, a.name, str(cache) if cache else '') for dt in ('fp16', 'fp32') if dt not in (getattr(M, 'SKIP_DTYPES', {}) or {}) for f in sorted((FROZEN / dt).glob('*.npz'))[:a.max_units or None]]
    t0 = time.time()
    tot = Counter()
    if a.workers > 1:
        with get_context('spawn').Pool(a.workers) as pool:
            for c in pool.imap_unordered(score_unit, tasks, chunksize=1):
                tot.update(c)
    else:
        for t in tasks:
            tot.update(score_unit(t))
    res = dict(name=a.name, module_md5=md5(M.__file__), tool_md5=md5(__file__), units=len(tasks), seconds=time.time() - t0, counts=dict(tot), compute=[json.loads(p.read_text()) for p in sorted(gdir.glob('compute_*.json'))])
    for c in res['compute']:
        c.pop('units', None)
    miss = sum((v for k, v in tot.items() if k.endswith('missing_units')))
    dst = Path(a.out) / 'score'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f'{a.name}.json').write_text(json.dumps(res, indent=1))
    for dt in ('fp16', 'fp32'):
        ms = sorted({k.split('|')[1] for k in tot if k.startswith(dt + '|')})
        for m in ms:
            g = lambda x: tot.get(f'{dt}|{m}|{x}', 0)
            if g('kl_app'):
                print(f"{a.name} {dt} {m:8s} KL significant {g('kl_sig')} / {g('kl_app')} = {100 * g('kl_sig') / g('kl_app'):.3f}%  correctly rounded {g('kl_cr')}  negative {g('KL_NEG')}  | gradient components significant {g('g_sig')} / {g('g_app')}", flush=True)
            elif g('rows'):
                print(f"{a.name} {dt} {m:8s} not scored ({g('rows')} rows)", flush=True)
    print(f'scored {len(tasks)} units in {time.time() - t0:.0f} s; unit files missing from the compute stage: {miss}', flush=True)
    return 0
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['compute', 'score'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--root', default='data', help='root of the released data (contains interface_a/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--max-units', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--cpu-compare', type=int, default=1)
    a = ap.parse_args()
    sys.exit(compute(a) if a.stage == 'compute' else score(a))
